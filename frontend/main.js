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
const modelSelect = document.getElementById('propModel');
let allFeatures = [];
let sortedTimes = [];
let selectedSiteId = null;

const popupHtml = (p) => p.kind === 'infrastructure'
  ? `<strong>${p.name}</strong><br>ID: ${p.id}<br>Type: ${p.structure_type}<br>Pattern: ${p.directionality}<br>Azimuth: ${p.azimuth_deg ?? 'N/A'}°<br>RF: ${p.rf_min_mhz}-${p.rf_max_mhz} MHz<br>Timestamp: ${p.timestamp}`
  : `<strong>${p.name}</strong><br>ID: ${p.id}<br>Band: ${p.freq_band}<br>Confidence: ${(p.confidence_score * 100).toFixed(0)}%<br>Ellipse: ${p.confidence_major_m}m × ${p.confidence_minor_m}m<br>Samples: ${p.sample_count}<br>Timestamp: ${p.timestamp}`;

function cutoffFromSlider() {
  const idx = Math.floor((Number(timeRange.value) / 100) * (sortedTimes.length - 1));
  return sortedTimes[idx] ?? sortedTimes[sortedTimes.length - 1];
}

async function refreshPropagation() {
  if (!selectedSiteId || !map.getSource('prop-contours')) return;
  const params = new URLSearchParams({ site_id: selectedSiteId, model: modelSelect.value });
  const p = await fetch(`/api/propagation?${params}`).then(r => r.json());
  const contours = Object.entries(p.snapshot.contours).map(([zone, info]) => ({
    type: 'Feature',
    properties: { zone },
    geometry: info.polygon,
  }));
  map.getSource('prop-contours').setData({ type: 'FeatureCollection', features: contours });
  details.innerHTML = `${details.innerHTML}<hr><strong>Propagation:</strong> ${p.snapshot.model}<br>Uncertainty ±${p.snapshot.uncertainty.sigma_db} dB`;
}

async function refreshSource() {
  const cutoff = cutoffFromSlider();
  timeLabel.textContent = `Cutoff: ${cutoff}`;
  const params = new URLSearchParams({ timestamp_lte: cutoff });
  const data = await fetch(`/api/features?${params}`).then(r => r.json());
  const filtered = data.features.filter((f) => {
    return (f.properties.kind === 'infrastructure' && infraToggle.checked) ||
      (f.properties.kind === 'estimate' && estToggle.checked);
  });
  map.getSource('antennas').setData({ type: 'FeatureCollection', features: filtered });
  if (!selectedSiteId) {
    const first = filtered.find(f => f.properties.kind === 'infrastructure');
    selectedSiteId = first?.properties?.id;
  }
  await refreshPropagation();
}

map.on('load', async () => {
  map.addControl(new maplibregl.NavigationControl(), 'bottom-right');
  const seed = await fetch('/api/features').then(r => r.json());
  allFeatures = seed.features;
  sortedTimes = [...new Set(allFeatures.map(f => f.properties.timestamp))].sort();

  map.addSource('antennas', { type: 'geojson', data: seed });
  map.addSource('prop-contours', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addLayer({ id: 'infra-layer', type: 'circle', source: 'antennas', filter: ['==', ['get', 'kind'], 'infrastructure'], paint: { 'circle-radius': 8, 'circle-color': '#53b7ff', 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });
  map.addLayer({ id: 'estimate-layer', type: 'circle', source: 'antennas', filter: ['==', ['get', 'kind'], 'estimate'], paint: { 'circle-radius': 10, 'circle-color': '#ff8459', 'circle-opacity': 0.8, 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });
  map.addLayer({ id: 'prop-fill', type: 'fill', source: 'prop-contours', paint: { 'fill-color': ['match', ['get', 'zone'], 'strong', '#00ff66', 'moderate', '#ffd53d', '#ff4f6d'], 'fill-opacity': 0.18 } });
  map.addLayer({ id: 'prop-line', type: 'line', source: 'prop-contours', paint: { 'line-color': '#ffffff', 'line-width': 1.2 } });

  ['infra-layer', 'estimate-layer'].forEach((layer) => {
    map.on('click', layer, async (e) => {
      const f = e.features?.[0];
      if (!f) return;
      details.innerHTML = popupHtml(f.properties);
      if (f.properties.kind === 'infrastructure') {
        selectedSiteId = f.properties.id;
        await refreshPropagation();
      }
    });
  });

  [infraToggle, estToggle, timeRange, modelSelect].forEach((el) => el.addEventListener('input', refreshSource));
  refreshSource();
});
