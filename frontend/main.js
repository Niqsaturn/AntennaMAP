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
let allFeatures = [];
let sortedTimes = [];

const popupHtml = (p) => p.kind === 'infrastructure'
  ? `<strong>${p.name}</strong><br>ID: ${p.id}<br>Type: ${p.structure_type}<br>Pattern: ${p.directionality}<br>Azimuth: ${p.azimuth_deg ?? 'N/A'}°<br>RF: ${p.rf_min_mhz}-${p.rf_max_mhz} MHz<br>Timestamp: ${p.timestamp}`
  : `<strong>${p.name}</strong><br>ID: ${p.id}<br>Band: ${p.freq_band}<br>Confidence: ${(p.confidence_score * 100).toFixed(0)}%<br>Ellipse: ${p.confidence_major_m}m × ${p.confidence_minor_m}m<br>Samples: ${p.sample_count}<br>Timestamp: ${p.timestamp}`;

const beamHtml = (p) => `<strong>${p.source_name}</strong><br>Source: ${p.source_id} (${p.source_kind})<br>Layer: ${p.beam_type}<br>Azimuth: ${p.azimuth_deg.toFixed(1)}°<br>Beamwidth: ${p.beamwidth_deg.toFixed(1)}°<br>Tilt proxy: ${p.tilt_proxy_deg.toFixed(1)}°<br>Power class: ${p.power_class}<br>Radius: ${p.radius_m.toFixed(1)} m<br>Timestamp: ${p.timestamp}<br>Assumptions: ${p.assumptions}`;

function cutoffFromSlider() {
  const idx = Math.floor((Number(timeRange.value) / 100) * (sortedTimes.length - 1));
  return sortedTimes[idx] ?? sortedTimes[sortedTimes.length - 1];
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
  map.getSource('propagation').setData({ type: 'FeatureCollection', features: propagationFiltered });
}

map.on('load', async () => {
  map.addControl(new maplibregl.NavigationControl(), 'bottom-right');
  const seed = await fetch('/api/features').then((r) => r.json());
  allFeatures = seed.features;
  sortedTimes = [...new Set(allFeatures.map((f) => f.properties.timestamp))].sort();

  map.addSource('antennas', { type: 'geojson', data: seed });
  map.addSource('propagation', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });

  map.addLayer({ id: 'confidence-infra-layer', type: 'fill', source: 'propagation', filter: ['all', ['==', ['get', 'beam_type'], 'confidence'], ['==', ['get', 'source_kind'], 'infrastructure']], paint: { 'fill-color': '#53b7ff', 'fill-opacity': 0.08 } });
  map.addLayer({ id: 'confidence-est-layer', type: 'fill', source: 'propagation', filter: ['all', ['==', ['get', 'beam_type'], 'confidence'], ['==', ['get', 'source_kind'], 'estimate']], paint: { 'fill-color': '#ff8459', 'fill-opacity': 0.13 } });

  map.addLayer({ id: 'wedge-infra-layer', type: 'fill', source: 'propagation', filter: ['all', ['==', ['get', 'beam_type'], 'wedge'], ['==', ['get', 'source_kind'], 'infrastructure']], paint: { 'fill-color': '#27a2f8', 'fill-opacity': 0.16 } });
  map.addLayer({ id: 'wedge-est-layer', type: 'fill', source: 'propagation', filter: ['all', ['==', ['get', 'beam_type'], 'wedge'], ['==', ['get', 'source_kind'], 'estimate']], paint: { 'fill-color': '#ff6a3d', 'fill-opacity': 0.2 } });

  map.addLayer({ id: 'beam-infra-layer', type: 'line', source: 'propagation', filter: ['all', ['==', ['get', 'beam_type'], 'centerline'], ['==', ['get', 'source_kind'], 'infrastructure']], paint: { 'line-color': '#1b77b8', 'line-width': 3 } });
  map.addLayer({ id: 'beam-est-layer', type: 'line', source: 'propagation', filter: ['all', ['==', ['get', 'beam_type'], 'centerline'], ['==', ['get', 'source_kind'], 'estimate']], paint: { 'line-color': '#c34a26', 'line-width': 3, 'line-dasharray': [2, 1] } });

  map.addLayer({ id: 'infra-layer', type: 'circle', source: 'antennas', filter: ['==', ['get', 'kind'], 'infrastructure'], paint: { 'circle-radius': 8, 'circle-color': '#53b7ff', 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });
  map.addLayer({ id: 'estimate-layer', type: 'circle', source: 'antennas', filter: ['==', ['get', 'kind'], 'estimate'], paint: { 'circle-radius': 10, 'circle-color': '#ff8459', 'circle-opacity': 0.8, 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });

  ['infra-layer', 'estimate-layer'].forEach((layer) => {
    map.on('click', layer, (e) => {
      const f = e.features?.[0];
      if (!f) return;
      details.innerHTML = popupHtml(f.properties);
    });
  });

  ['beam-infra-layer', 'beam-est-layer', 'wedge-infra-layer', 'wedge-est-layer', 'confidence-infra-layer', 'confidence-est-layer'].forEach((layer) => {
    map.on('click', layer, (e) => {
      const f = e.features?.[0];
      if (!f) return;
      details.innerHTML = beamHtml(f.properties);
    });
  });

  [infraToggle, estToggle, beamLinesToggle, coverageToggle, confidenceToggle, timeRange].forEach((el) => el.addEventListener('input', refreshSource));
  refreshSource();
});
