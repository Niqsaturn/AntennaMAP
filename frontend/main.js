const MAP_STYLES = {
  primary: 'https://demotiles.maplibre.org/style.json',
  backup: 'https://tiles.openfreemap.org/styles/bright',
};

const map = new maplibregl.Map({
  container: 'map',
  style: MAP_STYLES.primary,
  center: [-80.27, 25.97],
  zoom: 11,
  pitch: 58,
  bearing: -20,
  antialias: true,
});

const details = document.getElementById('details');
const infraToggle = document.getElementById('infraToggle');
const estToggle = document.getElementById('estToggle');
const beamLinesToggle = document.getElementById('beamLinesToggle');
const coverageToggle = document.getElementById('coverageToggle');
const confidenceToggle = document.getElementById('confidenceToggle');
const timeRange = document.getElementById('timeRange');
const timeLabel = document.getElementById('timeLabel');
const modelSelect = document.getElementById('propModel');
const mapWarningPanel = document.getElementById('mapWarning');
const statusBaseMap = document.getElementById('statusBaseMap');
const statusFeatureCount = document.getElementById('statusFeatureCount');
const statusAssessmentLoop = document.getElementById('statusAssessmentLoop');

let allFeatures = [];
let sortedTimes = [];
let selectedSiteId = null;
let fallbackInProgress = false;
let fallbackApplied = false;
let sourcesInitialized = false;
let assessmentLoopRunning = false;

function setStatus(element, value, state = 'ok') {
  element.textContent = value;
  element.classList.remove('ok', 'warn', 'err');
  element.classList.add(state);
}

function updateFeatureCount(count) {
  setStatus(statusFeatureCount, String(count), count > 0 ? 'ok' : 'warn');
}

function setAssessmentLoopStatus(running) {
  assessmentLoopRunning = running;
  setStatus(statusAssessmentLoop, running ? 'running' : 'idle', running ? 'ok' : 'warn');
}

function showMapWarning(message) {
  mapWarningPanel.hidden = false;
  mapWarningPanel.textContent = message;
}

function hideMapWarning() {
  mapWarningPanel.hidden = true;
  mapWarningPanel.textContent = '';
}

function ensureAntennaSource() {
  const source = map.getSource('antennas');
  if (!source) {
    setStatus(statusFeatureCount, 'source unavailable', 'err');
    showMapWarning('Feature layer is not ready yet. Retrying when map sources initialize.');
    return null;
  }
  return source;
}

const inferredHtml = (p) => {
  const e = p.estimated_elements || {};
  return `Antenna Type: ${p.antenna_type ?? 'unknown'} (${((p.type_confidence ?? 0) * 100).toFixed(0)}%)<br>` +
    `Beamwidth: ${e.estimated_beamwidth_deg ?? 'N/A'}°<br>` +
    `Orientation: ${e.array_orientation_deg ?? 'N/A'}°<br>` +
    `Sectors: ${e.sector_count ?? 'N/A'}<br>` +
    `Tilt: ${e.tilt_estimate_deg ?? 'N/A'}°<br>` +
    `Polarization: ${e.polarization_class ?? 'N/A'}<br>` +
    `Gain: ${e.gain_bucket ?? 'N/A'}`;
};

const popupHtml = (p) => p.kind === 'infrastructure'
  ? `<strong>${p.name}</strong><br>ID: ${p.id}<br>Type: ${p.structure_type}<br>Pattern: ${p.directionality}<br>Azimuth: ${p.azimuth_deg ?? 'N/A'}°<br>RF: ${p.rf_min_mhz}-${p.rf_max_mhz} MHz<br>${inferredHtml(p)}<br>Timestamp: ${p.timestamp}`
  : `<strong>${p.name}</strong><br>ID: ${p.id}<br>Band: ${p.freq_band}<br>Confidence: ${(p.confidence_score * 100).toFixed(0)}%<br>Ellipse: ${p.confidence_major_m}m × ${p.confidence_minor_m}m<br>Samples: ${p.sample_count}<br>${inferredHtml(p)}<br>Timestamp: ${p.timestamp}`;

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
  if (!sourcesInitialized) return;

  setAssessmentLoopStatus(true);
  const cutoff = cutoffFromSlider();
  timeLabel.textContent = `Cutoff: ${cutoff}`;
  const params = new URLSearchParams({ timestamp_lte: cutoff });

  try {
    const [featureData, countData] = await Promise.all([
      fetch(`/api/features?${params}`).then((r) => r.json()),
      fetch(`/api/features/count?${params}`).then((r) => r.json()),
    ]);

    const filtered = featureData.features.filter((f) => {
      return (f.properties.kind === 'infrastructure' && infraToggle.checked)
        || (f.properties.kind === 'estimate' && estToggle.checked);
    });

    const source = ensureAntennaSource();
    if (!source) return;

    source.setData({ type: 'FeatureCollection', features: filtered });
    updateFeatureCount(countData.count ?? filtered.length);

    if (!selectedSiteId) {
      const first = filtered.find(f => f.properties.kind === 'infrastructure');
      selectedSiteId = first?.properties?.id;
    }
    await refreshPropagation();
  } catch (error) {
    updateFeatureCount(0);
    showMapWarning(`Feature refresh failed: ${error?.message ?? 'unknown error'}`);
  } finally {
    setAssessmentLoopStatus(false);
  }
}

map.on('error', (event) => {
  const message = event?.error?.message ?? 'Unknown map rendering error';
  setStatus(statusBaseMap, 'error', 'err');
  showMapWarning(`Base map error detected: ${message}`);

  if (!fallbackInProgress && !fallbackApplied) {
    fallbackInProgress = true;
    showMapWarning('Primary base map failed. Switching to backup style…');
    map.setStyle(MAP_STYLES.backup);
    fallbackApplied = true;
  }
});

map.on('style.load', () => {
  fallbackInProgress = false;
  hideMapWarning();
  setStatus(statusBaseMap, fallbackApplied ? 'loaded (backup)' : 'loaded (primary)', 'ok');
});

map.on('load', async () => {
  map.addControl(new maplibregl.NavigationControl(), 'bottom-right');
  const seed = await fetch('/api/features').then((r) => r.json());
  allFeatures = seed.features;
  sortedTimes = [...new Set(allFeatures.map((f) => f.properties.timestamp))].sort();

  map.addSource('antennas', { type: 'geojson', data: seed });
  map.addSource('prop-contours', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  sourcesInitialized = true;

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
  updateFeatureCount(seed.features.length);
  refreshSource();
});
