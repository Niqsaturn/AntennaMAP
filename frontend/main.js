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
const beamLinesToggle = document.getElementById('beamLinesToggle');
const coverageToggle = document.getElementById('coverageToggle');
const confidenceToggle = document.getElementById('confidenceToggle');
const timeRange = document.getElementById('timeRange');
const timeLabel = document.getElementById('timeLabel');
const modelSelect = document.getElementById('propModel');
const sdrStatus = document.getElementById('sdrStatus');
let allFeatures = [];
let sortedTimes = [];
let selectedSiteId = null;

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

const beamHtml = (p) => `<strong>${p.source_name}</strong><br>Source: ${p.source_id} (${p.source_kind})<br>Layer: ${p.beam_type}<br>Azimuth: ${p.azimuth_deg.toFixed(1)}°<br>Beamwidth: ${p.beamwidth_deg.toFixed(1)}°<br>Tilt proxy: ${p.tilt_proxy_deg.toFixed(1)}°<br>Power class: ${p.power_class}<br>Radius: ${p.radius_m.toFixed(1)} m<br>Timestamp: ${p.timestamp}<br>Assumptions: ${p.assumptions}`;


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

  const [featureData, propagationData] = await Promise.all([
    fetch(`/api/features?${params}`).then((r) => r.json()),
    fetch(`/api/propagation?${params}`).then((r) => r.json()),
  ]);

  const filtered = featureData.features.filter((f) => {
    return (f.properties.kind === 'infrastructure' && infraToggle.checked)
      || (f.properties.kind === 'estimate' && estToggle.checked);
  });

  const selectedKinds = [];
  if (infraToggle.checked) selectedKinds.push('infrastructure');
  if (estToggle.checked) selectedKinds.push('estimate');

  const propagationFiltered = propagationData.features.filter((f) => {
    const typeEnabled = (f.properties.beam_type === 'centerline' && beamLinesToggle.checked)
      || (f.properties.beam_type === 'wedge' && coverageToggle.checked)
      || (f.properties.beam_type === 'confidence' && confidenceToggle.checked);
    return typeEnabled && selectedKinds.includes(f.properties.source_kind);
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
  const seed = await fetch('/api/features').then((r) => r.json());
  allFeatures = seed.features;
  sortedTimes = [...new Set(allFeatures.map((f) => f.properties.timestamp))].sort();

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
  await refreshSdrStatus();
  refreshSource();
});
