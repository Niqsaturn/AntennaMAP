const map = new maplibregl.Map({ container: 'map', style: 'https://demotiles.maplibre.org/style.json', center: [-80.27, 25.97], zoom: 11, pitch: 58, bearing: -20, antialias: true });

const details = document.getElementById('details');
const infraToggle = document.getElementById('infraToggle');
const estToggle = document.getElementById('estToggle');
const rayToggle = document.getElementById('rayToggle');
const sectorToggle = document.getElementById('sectorToggle');
const confidenceToggle = document.getElementById('confidenceToggle');
const timeRange = document.getElementById('timeRange');
const timeLabel = document.getElementById('timeLabel');
const modelSelect = document.getElementById('propModel');
const loopStatus = document.getElementById('loopStatus');
let allFeatures = [];
let sortedTimes = [];

const popupHtml = (p) => `<strong>${p.name}</strong><br>ID: ${p.id}<br>Kind: ${p.kind}<br>Azimuth: ${p.azimuth_deg ?? 'N/A'}°<br>Beamwidth: ${p.beamwidth_deg ?? 'N/A'}°<br>Ray length: ${p.ray_length_m ?? 'N/A'} m<br>Sector radius: ${p.wedge_radius_m ?? 'N/A'} m<br>Confidence ellipse: ${p.confidence_major_m ?? 'N/A'}m × ${p.confidence_minor_m ?? 'N/A'}m`;


function formatFreqHz(hz) {
  return `${(hz / 1_000_000).toFixed(3)} MHz`;
}

async function refreshSdrStatus() {
  const response = await fetch('/api/sdr/capabilities');
  const data = await response.json();
  const active = data.active_config;
  const modelMeta = data.capabilities.models[active.model] || {};
  sdrStatus.innerHTML = `<strong>Device:</strong> ${modelMeta.label || active.model}<br>` +
    `<strong>Model ID:</strong> ${active.model}<br>` +
    `<strong>Center:</strong> ${formatFreqHz(active.center_freq_hz)}<br>` +
    `<strong>Sample Rate:</strong> ${active.sample_rate_sps.toLocaleString()} sps<br>` +
    `<strong>Bandwidth:</strong> ${active.bandwidth_hz.toLocaleString()} Hz<br>` +
    `<strong>Gain:</strong> ${active.gain_db} dB · <strong>PPM:</strong> ${active.ppm}`;
}

function cutoffFromSlider() {
  const idx = Math.floor((Number(timeRange.value) / 100) * (sortedTimes.length - 1));
  return sortedTimes[idx] ?? sortedTimes[sortedTimes.length - 1];
}

function asFeature(siteFeature, geometry, overlayType) {
  if (!geometry) return null;
  return { type: 'Feature', geometry, properties: { ...siteFeature.properties, overlay_type: overlayType } };
}


async function refreshLoopStatus() {
  try {
    const status = await fetch('/api/loop/status').then((r) => r.json());
    const lastSuccess = status.last_run?.last_successful_run_at ?? 'never';
    loopStatus.innerHTML = `<strong>Loop:</strong> ${status.active ? 'running' : 'paused'}<br><strong>Provider:</strong> ${status.config.provider}<br><strong>Model:</strong> ${status.config.model}<br><strong>Interval:</strong> ${status.config.interval_seconds}s<br><strong>Last successful run:</strong> ${lastSuccess}`;
  } catch (_err) {
    loopStatus.textContent = 'Loop: unavailable';
  }
}

async function refreshSource() {
  if (!sourcesInitialized) return;

  setAssessmentLoopStatus(true);
  const cutoff = cutoffFromSlider();
  timeLabel.textContent = `Cutoff: ${cutoff}`;
  const data = await fetch(`/api/features?timestamp_lte=${encodeURIComponent(cutoff)}`).then((r) => r.json());
  const visibleSites = data.features.filter((f) => (f.properties.kind === 'infrastructure' && infraToggle.checked) || (f.properties.kind === 'estimate' && estToggle.checked));

  const rays = [];
  const sectors = [];
  const confidence = [];
  visibleSites.forEach((f) => {
    const overlays = f.properties.overlay_geometries || {};
    const ray = asFeature(f, overlays.direction_ray, 'direction_ray');
    const sector = asFeature(f, overlays.sector_wedge, 'sector_wedge');
    const ellipse = asFeature(f, overlays.confidence_ellipse, 'confidence_ellipse');
    if (ray) rays.push(ray);
    if (sector) sectors.push(sector);
    if (ellipse) confidence.push(ellipse);
  });

  map.getSource('sites').setData({ type: 'FeatureCollection', features: visibleSites });
  map.getSource('rays').setData({ type: 'FeatureCollection', features: rayToggle.checked ? rays : [] });
  map.getSource('sectors').setData({ type: 'FeatureCollection', features: sectorToggle.checked ? sectors : [] });
  map.getSource('confidence').setData({ type: 'FeatureCollection', features: confidenceToggle.checked ? confidence : [] });
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
  sortedTimes = [...new Set(seed.features.map((f) => f.properties.timestamp))].sort();

  map.addSource('sites', { type: 'geojson', data: seed });
  map.addSource('rays', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('sectors', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('confidence', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });

  map.addLayer({ id: 'infra-layer', type: 'circle', source: 'sites', filter: ['==', ['get', 'kind'], 'infrastructure'], paint: { 'circle-radius': 7, 'circle-color': '#4ea8ff', 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });
  map.addLayer({ id: 'estimate-layer', type: 'circle', source: 'sites', filter: ['==', ['get', 'kind'], 'estimate'], paint: { 'circle-radius': 9, 'circle-color': '#ff8459', 'circle-opacity': 0.85, 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });
  map.addLayer({ id: 'ray-layer', type: 'line', source: 'rays', paint: { 'line-color': '#88d8ff', 'line-width': 2 } });
  map.addLayer({ id: 'sector-layer', type: 'fill', source: 'sectors', paint: { 'fill-color': '#7ac37a', 'fill-opacity': 0.2 } });
  map.addLayer({ id: 'confidence-layer', type: 'fill', source: 'confidence', paint: { 'fill-color': '#ffc857', 'fill-opacity': 0.2 } });

  ['infra-layer', 'estimate-layer'].forEach((layer) => map.on('click', layer, (e) => { const f = e.features?.[0]; if (f) details.innerHTML = popupHtml(f.properties); }));

  [infraToggle, estToggle, rayToggle, sectorToggle, confidenceToggle, timeRange].forEach((el) => el.addEventListener('input', refreshSource));
  refreshSource();
  refreshLoopStatus();
  setInterval(refreshLoopStatus, 10000);
});
