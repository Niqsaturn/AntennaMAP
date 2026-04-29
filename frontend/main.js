const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [-80.27, 25.97],
  zoom: 11,
  pitch: 58,
  bearing: -20,
  antialias: true,
});

const details = document.getElementById('details');
const infraToggle = document.getElementById('infraToggle');
const estToggle = document.getElementById('estToggle');
const timeRange = document.getElementById('timeRange');
const timeLabel = document.getElementById('timeLabel');
const assess = document.getElementById('assessment');
let allFeatures = [];
let sortedTimes = [];

const popupHtml = (p) => p.kind === 'infrastructure'
  ? `<strong>${p.name}</strong><br>ID: ${p.id}<br>Type: ${p.structure_type}<br>Pattern: ${p.directionality}<br>Azimuth: ${p.azimuth_deg ?? 'N/A'}°<br>RF: ${p.rf_min_mhz}-${p.rf_max_mhz} MHz<br>Timestamp: ${p.timestamp}`
  : `<strong>${p.name}</strong><br>ID: ${p.id}<br>Band: ${p.freq_band}<br>Confidence: ${(p.confidence_score * 100).toFixed(0)}%<br>Ellipse: ${p.confidence_major_m}m × ${p.confidence_minor_m}m<br>Samples: ${p.sample_count}<br>Source: ${p.source ?? 'seed'}<br>Timestamp: ${p.timestamp}`;

function cutoffFromSlider() {
  const idx = Math.floor((Number(timeRange.value) / 100) * (sortedTimes.length - 1));
  return sortedTimes[idx] ?? sortedTimes[sortedTimes.length - 1];
}

async function refreshAssessment() {
  const data = await fetch('/api/assessment').then(r => r.json());
  assess.textContent = `AI loop estimates: ${data.estimated_feature_count}`;
}

async function refreshSource() {
  const cutoff = cutoffFromSlider();
  timeLabel.textContent = `Cutoff: ${cutoff}`;
  const params = new URLSearchParams({ timestamp_lte: cutoff });
  const data = await fetch(`/api/features?${params}`).then(r => r.json());
  const filtered = data.features.filter((f) => (f.properties.kind === 'infrastructure' && infraToggle.checked) || (f.properties.kind === 'estimate' && estToggle.checked));
  map.getSource('antennas').setData({ type: 'FeatureCollection', features: filtered });
}

async function bootstrap() {
  const seed = await fetch('/api/features').then(r => r.json());
  allFeatures = seed.features;
  sortedTimes = [...new Set(allFeatures.map(f => f.properties.timestamp))].sort();

  map.addSource('antennas', { type: 'geojson', data: seed });
  map.addLayer({ id: 'infra-layer', type: 'circle', source: 'antennas', filter: ['==', ['get', 'kind'], 'infrastructure'], paint: { 'circle-radius': 8, 'circle-color': '#53b7ff', 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });
  map.addLayer({ id: 'estimate-layer', type: 'circle', source: 'antennas', filter: ['==', ['get', 'kind'], 'estimate'], paint: { 'circle-radius': 10, 'circle-color': '#ff8459', 'circle-opacity': 0.8, 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });

  ['infra-layer', 'estimate-layer'].forEach((layer) => map.on('click', layer, (e) => {
    const f = e.features?.[0];
    if (!f) return;
    details.innerHTML = popupHtml(f.properties);
  }));

  [infraToggle, estToggle, timeRange].forEach((el) => el.addEventListener('input', refreshSource));
  await refreshSource();
  await refreshAssessment();
  setInterval(async () => {
    await refreshSource();
    await refreshAssessment();
  }, 15000);
}

map.on('load', async () => {
  map.addControl(new maplibregl.NavigationControl(), 'bottom-right');
  await bootstrap();
});
