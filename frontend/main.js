const map = new maplibregl.Map({ container: 'map', style: 'https://demotiles.maplibre.org/style.json', center: [-80.27, 25.97], zoom: 11, pitch: 58, bearing: -20, antialias: true });

const details = document.getElementById('details');
const infraToggle = document.getElementById('infraToggle');
const estToggle = document.getElementById('estToggle');
const beamLinesToggle = document.getElementById('beamLinesToggle');
const coverageToggle = document.getElementById('coverageToggle');
const confidenceToggle = document.getElementById('confidenceToggle');
const timeRange = document.getElementById('timeRange');
const timeLabel = document.getElementById('timeLabel');
const modelSelect = document.getElementById('propModel');
const provenance = document.getElementById('provenance');
const occupancyTrend = document.getElementById('occupancyTrend');
const spectralStats = document.getElementById('spectralStats');
const waterfall = document.getElementById('waterfall');
const tabs = document.querySelectorAll('.tab');
const tabPanels = document.querySelectorAll('.tab-panel');
let allFeatures = [];
let sortedTimes = [];
let selectedSiteId = null;

const popupHtml = (p) => `<strong>${p.name}</strong><br>ID: ${p.id}<br>Kind: ${p.kind}<br>Timestamp: ${p.timestamp}`;

function cutoffFromSlider() {
  const idx = Math.floor((Number(timeRange.value) / 100) * (sortedTimes.length - 1));
  return sortedTimes[idx] ?? sortedTimes[sortedTimes.length - 1];
}

function renderWaterfall(payload) {
  const ctx = waterfall.getContext('2d');
  const { rows, times } = payload;
  const cellW = waterfall.width / Math.max(times.length, 1);
  const cellH = waterfall.height / Math.max(rows.length, 1);
  ctx.clearRect(0, 0, waterfall.width, waterfall.height);
  rows.forEach((row, y) => row.values.forEach((value, x) => {
    const norm = value === null ? 0 : Math.max(0, Math.min(1, (value + 110) / 60));
    const hue = 240 - (norm * 240);
    ctx.fillStyle = value === null ? '#243144' : `hsl(${hue}, 80%, 50%)`;
    ctx.fillRect(x * cellW, y * cellH, cellW, cellH);
  }));
}

function renderOccupancy(bands) {
  const maxSamples = Math.max(...bands.map((b) => b.sample_count), 1);
  occupancyTrend.innerHTML = bands.map((band) => `<div class="chart-row">${band.band} (${band.avg_snr_db.toFixed(1)} dB)<span class="bar" style="width:${(band.sample_count / maxSamples) * 180}px"></span></div>`).join('');
}

async function refreshSpectrum() {
  const cutoff = cutoffFromSlider();
  const params = new URLSearchParams({ timestamp_lte: cutoff });
  if (selectedSiteId) params.set('site_id', selectedSiteId);

  const [waterfallData, occupancyData] = await Promise.all([
    fetch(`/api/spectrum/waterfall?${params}`).then((r) => r.json()),
    fetch(`/api/spectrum/occupancy?${params}`).then((r) => r.json()),
  ]);

  renderWaterfall(waterfallData);
  renderOccupancy(occupancyData.bands);
  spectralStats.innerHTML = `<strong>Spectral stats</strong><br>Site: ${occupancyData.site_id ?? 'none'}<br>Total samples: ${occupancyData.spectral_stats.total_samples}<br>Strongest band: ${occupancyData.spectral_stats.strongest_band ?? 'N/A'}`;
  const p = waterfallData.provenance;
  provenance.textContent = `Provenance: ${p.device_id} · ${p.adapter_type} · samples ${p.sample_count} · freshness ${p.freshness_seconds ?? 'n/a'}s`;
}

async function refreshSource() {
  const cutoff = cutoffFromSlider();
  timeLabel.textContent = `Cutoff: ${cutoff}`;
  const params = new URLSearchParams({ timestamp_lte: cutoff });

  const featureData = await fetch(`/api/features?${params}`).then((r) => r.json());
  const filtered = featureData.features.filter((f) => (f.properties.kind === 'infrastructure' && infraToggle.checked) || (f.properties.kind === 'estimate' && estToggle.checked));

  map.getSource('antennas').setData({ type: 'FeatureCollection', features: filtered });
  if (!selectedSiteId) {
    const first = filtered.find((f) => f.properties.kind === 'infrastructure');
    selectedSiteId = first?.properties?.id;
  }
  await refreshSpectrum();
}

tabs.forEach((tab) => tab.addEventListener('click', () => {
  tabs.forEach((t) => t.classList.remove('active'));
  tabPanels.forEach((p) => p.classList.remove('active'));
  tab.classList.add('active');
  document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
}));

map.on('load', async () => {
  map.addControl(new maplibregl.NavigationControl(), 'bottom-right');
  const seed = await fetch('/api/features').then((r) => r.json());
  allFeatures = seed.features;
  sortedTimes = [...new Set(allFeatures.map((f) => f.properties.timestamp))].sort();

  map.addSource('antennas', { type: 'geojson', data: seed });
  map.addLayer({ id: 'infra-layer', type: 'circle', source: 'antennas', filter: ['==', ['get', 'kind'], 'infrastructure'], paint: { 'circle-radius': 8, 'circle-color': '#53b7ff', 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });
  map.addLayer({ id: 'estimate-layer', type: 'circle', source: 'antennas', filter: ['==', ['get', 'kind'], 'estimate'], paint: { 'circle-radius': 10, 'circle-color': '#ff8459', 'circle-opacity': 0.8, 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });

  ['infra-layer', 'estimate-layer'].forEach((layer) => {
    map.on('click', layer, async (e) => {
      const f = e.features?.[0];
      if (!f) return;
      details.innerHTML = popupHtml(f.properties);
      if (f.properties.kind === 'infrastructure') {
        selectedSiteId = f.properties.id;
        await refreshSpectrum();
      }
    });
  });

  [infraToggle, estToggle, beamLinesToggle, coverageToggle, confidenceToggle, timeRange, modelSelect].forEach((el) => el.addEventListener('input', refreshSource));
  refreshSource();
});
