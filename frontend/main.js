const _DARK_STYLE = {
  version: 8,
  name: 'dark',
  sources: {
    'osm-tiles': {
      type: 'raster',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution: '© OpenStreetMap contributors',
    },
  },
  layers: [
    { id: 'background', type: 'background', paint: { 'background-color': '#0f172a' } },
    { id: 'osm', type: 'raster', source: 'osm-tiles', paint: { 'raster-opacity': 0.55, 'raster-saturation': -0.7, 'raster-brightness-min': 0.05 } },
  ],
};

const map = new maplibregl.Map({
  container: 'map',
  style: _DARK_STYLE,
  center: [-80.27, 25.97], zoom: 11, pitch: 0, bearing: 0, antialias: true,
});

// ── DOM refs ───────────────────────────────────────────────────────────────
const details         = document.getElementById('details');
const infraToggle     = document.getElementById('infraToggle');
const estToggle       = document.getElementById('estToggle');
const spectToggle     = document.getElementById('spectToggle');
const rayToggle       = document.getElementById('rayToggle');
const sectorToggle    = document.getElementById('sectorToggle');
const confidenceToggle= document.getElementById('confidenceToggle');
const coverageToggle  = document.getElementById('coverageToggle');
const satelliteToggle = document.getElementById('satelliteToggle');
const timeRange       = document.getElementById('timeRange');
const timeLabel       = document.getElementById('timeLabel');
const loopStatus        = document.getElementById('loopStatus');
const sdrStatus         = document.getElementById('sdrStatus');
const coverageStatus    = document.getElementById('coverageStatus');
const calibrationStatus = document.getElementById('calibrationStatus');
const modelDropdown   = document.getElementById('modelSelect');
const satGroupSelect  = document.getElementById('satGroup');
const analyzeBtn      = document.getElementById('analyzeNow');
const seedBtn         = document.getElementById('seedRegion');
const loopIntervalEl  = document.getElementById('loopInterval');
const analysisLog     = document.getElementById('analysisLog');
const logHeader       = document.getElementById('analysisLogHeader');

let sortedTimes = [];
let sourcesInitialized = false;
let selectedModel = '';

// ── Helpers ────────────────────────────────────────────────────────────────
const formatFreqHz = (hz) => `${(hz / 1_000_000).toFixed(3)} MHz`;

function asFeature(siteFeature, geometry, overlayType) {
  if (!geometry) return null;
  return { type: 'Feature', geometry, properties: { ...siteFeature.properties, overlay_type: overlayType } };
}

function cutoffFromSlider() {
  if (!sortedTimes.length) return null;  // no timestamps → no filter
  const idx = Math.min(
    Math.floor((Number(timeRange.value) / 100) * (sortedTimes.length - 1)),
    sortedTimes.length - 1
  );
  return sortedTimes[idx] ?? null;
}

const popupHtml = (p) => {
  const conf = p.confidence != null ? `<br>Confidence: ${(p.confidence * 100).toFixed(0)}%` : '';
  const count = p.analysis_count != null ? ` (${p.analysis_count} obs)` : '';
  const source = p.source ? `<br>Source: ${p.source}` : '';
  const notes = p.notes ? `<br>Notes: ${p.notes}` : '';
  const confirmRow = p.kind === 'speculative'
    ? `<div class="confirm-row"><button onclick="confirmFeature('${p.id}',true)">Confirm</button> <button onclick="confirmFeature('${p.id}',false)">Dismiss</button></div>`
    : '';
  return `<strong>${p.name || p.id}</strong><br>Kind: ${p.kind}${conf}${count}` +
    `<br>Freq: ${p.freq_band || '—'}<br>Type: ${p.antenna_type || p.structure_type || '—'}` +
    `<br>Azimuth: ${p.azimuth_deg ?? '—'}°${source}${notes}${confirmRow}`;
};

// ── SDR Status ─────────────────────────────────────────────────────────────
async function refreshSdrStatus() {
  if (!sdrStatus) return;
  try {
    const data = await fetch('/api/sdr/capabilities').then((r) => r.json());
    const active = data.active_config;
    const modelMeta = (data.capabilities.models || {})[active.model] || {};
    sdrStatus.innerHTML =
      `<strong>Device:</strong> ${modelMeta.label || active.model}<br>` +
      `<strong>Center:</strong> ${formatFreqHz(active.center_freq_hz)}<br>` +
      `<strong>Rate:</strong> ${active.sample_rate_sps.toLocaleString()} sps · ` +
      `<strong>Gain:</strong> ${active.gain_db} dB`;
  } catch (_) {
    if (sdrStatus) sdrStatus.textContent = 'SDR: unavailable';
  }
}

// ── Loop Status ────────────────────────────────────────────────────────────
async function refreshLoopStatus() {
  try {
    const s = await fetch('/api/loop/status').then((r) => r.json());
    const last = s.last_run?.last_successful_run_at ?? 'never';
    loopStatus.innerHTML =
      `<strong>Loop:</strong> ${s.active ? 'running' : 'paused'} · ` +
      `<strong>Provider:</strong> ${s.config.provider} · ` +
      `<strong>Model:</strong> ${s.config.model || '—'}<br>` +
      `<strong>Interval:</strong> ${s.config.interval_seconds}s · ` +
      `<strong>Last OK:</strong> ${last.slice(0, 19).replace('T', ' ')}`;
  } catch (_) { loopStatus.textContent = 'Loop: unavailable'; }
}

// ── Model Dropdown ─────────────────────────────────────────────────────────
async function populateModelDropdown() {
  if (!modelDropdown) return;
  try {
    const data = await fetch('/api/models/discover').then((r) => r.json());
    const options = [['', '— None —']];
    (data.ollama || []).forEach((m) => options.push([`ollama/${m}`, `Ollama: ${m}`]));
    (data.python_local || []).forEach((m) => options.push([`python_local/${m}`, `Local: ${m}`]));
    modelDropdown.innerHTML = options.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
    if (selectedModel) modelDropdown.value = selectedModel;
  } catch (_) {
    if (modelDropdown) modelDropdown.innerHTML = '<option value="">— unavailable —</option>';
  }
}

async function applyModelSelection(value) {
  selectedModel = value;
  if (!value) return;
  const [provider, ...rest] = value.split('/');
  const model = rest.join('/') || 'baseline-v1';
  const interval = parseInt(loopIntervalEl?.value || '60', 10);
  await fetch(
    `/api/loop/config?provider=${encodeURIComponent(provider)}&model=${encodeURIComponent(model)}&interval_seconds=${interval}`,
    { method: 'POST' }
  ).catch(() => {});
}

// ── Main site + overlay source refresh ────────────────────────────────────
async function refreshSource() {
  if (!sourcesInitialized) return;
  try {
    const cutoff = cutoffFromSlider();
    if (timeLabel) timeLabel.textContent = `Cutoff: ${cutoff ? cutoff.slice(0, 19).replace('T', ' ') : 'all'}`;

    const url = cutoff
      ? `/api/features?timestamp_lte=${encodeURIComponent(cutoff)}`
      : '/api/features';
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const visibleSites = (data.features || []).filter((f) =>
      (f.properties.kind === 'infrastructure' && infraToggle?.checked) ||
      (f.properties.kind === 'estimate' && estToggle?.checked)
    );

  const rays = [], sectors = [], confidence = [];
  visibleSites.forEach((f) => {
    const o = f.properties.overlay_geometries || {};
    const ray = asFeature(f, o.direction_ray, 'direction_ray');
    const sector = asFeature(f, o.sector_wedge, 'sector_wedge');
    const ellipse = asFeature(f, o.confidence_ellipse, 'confidence_ellipse');
    if (ray) rays.push(ray);
    if (sector) sectors.push(sector);
    if (ellipse) confidence.push(ellipse);
  });

    map.getSource('sites').setData({ type: 'FeatureCollection', features: visibleSites });
    map.getSource('rays').setData({ type: 'FeatureCollection', features: rayToggle?.checked ? rays : [] });
    map.getSource('sectors').setData({ type: 'FeatureCollection', features: sectorToggle?.checked ? sectors : [] });
    map.getSource('confidence').setData({ type: 'FeatureCollection', features: confidenceToggle?.checked ? confidence : [] });
  } catch (err) {
    console.warn('refreshSource error:', err.message);
  }
}

// ── Speculative layer ──────────────────────────────────────────────────────
async function refreshSpeculative() {
  if (!sourcesInitialized) return;
  try {
    const bounds = map.getBounds();
    const url = `/api/map/features?kind=speculative` +
      `&lat_min=${bounds.getSouth().toFixed(4)}&lat_max=${bounds.getNorth().toFixed(4)}` +
      `&lon_min=${bounds.getWest().toFixed(4)}&lon_max=${bounds.getEast().toFixed(4)}&limit=300`;
    const data = await fetch(url).then((r) => r.json());

    const rays = [], sectors = [], ellipses = [];
    (data.features || []).forEach((f) => {
      const o = f.properties.overlay_geometries || {};
      if (o.direction_ray) rays.push(asFeature(f, o.direction_ray, 'direction_ray'));
      if (o.sector_wedge) sectors.push(asFeature(f, o.sector_wedge, 'sector_wedge'));
      if (o.confidence_ellipse) ellipses.push(asFeature(f, o.confidence_ellipse, 'confidence_ellipse'));
    });

    map.getSource('speculative').setData({
      type: 'FeatureCollection',
      features: spectToggle.checked ? (data.features || []) : [],
    });
    map.getSource('spec-rays').setData({ type: 'FeatureCollection', features: rayToggle.checked ? rays : [] });
    map.getSource('spec-sectors').setData({ type: 'FeatureCollection', features: sectorToggle.checked ? sectors : [] });
    map.getSource('spec-ellipses').setData({ type: 'FeatureCollection', features: confidenceToggle.checked ? ellipses : [] });
  } catch (_) {}
}

// ── Coverage layer ─────────────────────────────────────────────────────────
async function refreshCoverage() {
  if (!sourcesInitialized || !coverageToggle.checked) return;
  try {
    const data = await fetch('/api/map/coverage').then((r) => r.json());
    map.getSource('coverage').setData(data);
    const prog = await fetch('/api/usmap/progress').then((r) => r.json());
    if (coverageStatus) {
      coverageStatus.textContent =
        `Coverage: ${prog.percent_analyzed ?? 0}% analyzed · ${prog.percent_seeded ?? 0}% seeded`;
    }
  } catch (_) {}
}

// ── Satellite layer ────────────────────────────────────────────────────────
async function refreshSatellites() {
  if (!sourcesInitialized || !satelliteToggle.checked) return;
  try {
    const group = satGroupSelect?.value || 'geo';
    const data = await fetch(`/api/satellites/positions?group=${group}&limit=40`).then((r) => r.json());
    map.getSource('satellites').setData(data.geojson || { type: 'FeatureCollection', features: [] });
  } catch (_) {}
}

// ── Analysis Log ───────────────────────────────────────────────────────────
async function refreshAnalysisLog() {
  try {
    const data = await fetch('/api/analysis/log?limit=5').then((r) => r.json());
    if (!analysisLog) return;
    if (!data.entries?.length) { analysisLog.innerHTML = '<span class="muted">No analyses yet.</span>'; return; }
    analysisLog.innerHTML = data.entries.map((e) => {
      const ts = (e.timestamp || '').slice(0, 19).replace('T', ' ');
      return `<div class="log-entry"><span class="log-ts">${ts}</span> ` +
        `<strong>${e.model || '—'}</strong> · ${e.detections_count} detection(s)<br>` +
        `<span class="muted">${e.input_summary || ''}</span></div>`;
    }).join('');
  } catch (_) { if (analysisLog) analysisLog.innerHTML = '<span class="muted">unavailable</span>'; }
}

// ── Analyze Now button ─────────────────────────────────────────────────────
async function analyzeNow() {
  if (!analyzeBtn) return;
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing…';
  try {
    const center = map.getCenter();
    const [provider, ...rest] = (selectedModel || '/').split('/');
    const model = rest.join('/');
    const body = { provider: provider || 'local', model, lat: center.lat, lon: center.lng, limit_samples: 50 };
    const result = await fetch('/api/analysis/run', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    }).then((r) => r.json());
    await refreshSpeculative();
    await refreshAnalysisLog();
    if (details) details.innerHTML = `Analysis: ${result.detections} detection(s) · ${result.speculative_features_added} features added<br><em>${result.summary || ''}</em>`;
  } catch (err) {
    if (details) details.textContent = `Analysis error: ${err.message}`;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze Now';
  }
}

// ── Seed Region button ─────────────────────────────────────────────────────
async function seedRegion() {
  if (!seedBtn) return;
  seedBtn.disabled = true;
  seedBtn.textContent = 'Seeding…';
  try {
    const center = map.getCenter();
    const result = await fetch(
      `/api/map/seed?lat=${center.lat.toFixed(5)}&lon=${center.lng.toFixed(5)}&radius_km=50`,
      { method: 'POST' }
    ).then((r) => r.json());
    await refreshSpeculative();
    await refreshCoverage();
    if (details) details.textContent = `Seeded: ${result.features_added} features added from FCC + towers.`;
  } catch (err) {
    if (details) details.textContent = `Seed error: ${err.message}`;
  } finally {
    seedBtn.disabled = false;
    seedBtn.textContent = 'Seed Region';
  }
}

// ── Feature confirm/dismiss (called from popup HTML) ──────────────────────
window.confirmFeature = async function(featureId, confirmed) {
  const body = { feature_id: featureId, confirmed };
  if (confirmed) {
    const center = map.getCenter();
    body.true_lat = center.lat;
    body.true_lon = center.lng;
  }
  try {
    await fetch('/api/map/confirm', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    await refreshSpeculative();
    if (details) details.textContent = confirmed ? 'Detection confirmed.' : 'Detection dismissed.';
  } catch (_) {}
};

// ── Calibration status ─────────────────────────────────────────────────────
async function refreshCalibration() {
  if (!calibrationStatus) return;
  try {
    const d = await fetch('/api/analysis/calibration').then((r) => r.json());
    if (d.count === 0) {
      calibrationStatus.textContent = 'Calibration: no confirmed features yet';
    } else {
      calibrationStatus.innerHTML =
        `<strong>Calibration:</strong> ${d.count} confirmed · ` +
        `mean error ${d.mean_error_m != null ? d.mean_error_m.toFixed(0) : '—'} m · ` +
        `median ${d.median_error_m != null ? d.median_error_m.toFixed(0) : '—'} m`;
    }
  } catch (_) { if (calibrationStatus) calibrationStatus.textContent = 'Calibration: —'; }
}

// ── Uncertain features ─────────────────────────────────────────────────────
async function refreshUncertain() {
  if (!sourcesInitialized) return;
  try {
    const d = await fetch('/api/analysis/uncertain?limit=10').then((r) => r.json());
    const count = d.count || 0;
    if (analyzeBtn) {
      analyzeBtn.textContent = count > 0 ? `Analyze Now (${count} need review)` : 'Analyze Now';
    }
    // Update uncertain-ring source with features needing confirmation
    const src = map.getSource('speculative-uncertain');
    if (src) src.setData({ type: 'FeatureCollection', features: d.features || [] });
  } catch (_) {}
}

// ── Map load ───────────────────────────────────────────────────────────────
map.on('load', async () => {
  map.addControl(new maplibregl.NavigationControl(), 'bottom-right');

  // Seed GeoJSON features from API
  const seed = await fetch('/api/features').then((r) => r.json());
  sortedTimes = [...new Set(seed.features.map((f) => f.properties.timestamp))].filter(Boolean).sort();

  // ── Sources ──────────────────────────────────────────────────────────────
  map.addSource('sites',      { type: 'geojson', data: seed });
  map.addSource('rays',       { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('sectors',    { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('confidence', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('speculative',{ type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('spec-rays',  { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('spec-sectors',{ type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('spec-ellipses',{ type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('coverage',             { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('satellites',           { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('speculative-uncertain',{ type: 'geojson', data: { type: 'FeatureCollection', features: [] } });

  // ── Layers — base ────────────────────────────────────────────────────────
  map.addLayer({ id: 'infra-layer',    type: 'circle', source: 'sites',
    filter: ['==', ['get', 'kind'], 'infrastructure'],
    paint: { 'circle-radius': 7, 'circle-color': '#4ea8ff', 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });
  map.addLayer({ id: 'estimate-layer', type: 'circle', source: 'sites',
    filter: ['==', ['get', 'kind'], 'estimate'],
    paint: { 'circle-radius': 9, 'circle-color': '#ff8459', 'circle-opacity': 0.85, 'circle-stroke-width': 1, 'circle-stroke-color': '#fff' } });
  map.addLayer({ id: 'ray-layer',       type: 'line', source: 'rays',
    paint: { 'line-color': '#88d8ff', 'line-width': 2 } });
  map.addLayer({ id: 'sector-layer',    type: 'fill', source: 'sectors',
    paint: { 'fill-color': '#7ac37a', 'fill-opacity': 0.2 } });
  map.addLayer({ id: 'confidence-layer',type: 'fill', source: 'confidence',
    paint: { 'fill-color': '#ffc857', 'fill-opacity': 0.2 } });

  // ── Layers — speculative ─────────────────────────────────────────────────
  map.addLayer({ id: 'spec-sector-layer', type: 'fill', source: 'spec-sectors',
    paint: { 'fill-color': '#c084fc', 'fill-opacity': 0.15 } });
  map.addLayer({ id: 'spec-ellipse-layer', type: 'fill', source: 'spec-ellipses',
    paint: { 'fill-color': '#c084fc', 'fill-opacity': 0.12 } });
  map.addLayer({ id: 'spec-ray-layer', type: 'line', source: 'spec-rays',
    paint: { 'line-color': '#c084fc', 'line-width': 1.5, 'line-dasharray': [4, 2] } });
  map.addLayer({ id: 'speculative-layer', type: 'circle', source: 'speculative',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'confidence'], 0, 5, 1, 9],
      'circle-color': '#c084fc', 'circle-opacity': 0.8,
      'circle-stroke-width': 1, 'circle-stroke-color': '#fff',
    } });

  // ── Layers — coverage grid ───────────────────────────────────────────────
  map.addLayer({ id: 'coverage-fill', type: 'fill', source: 'coverage',
    paint: {
      'fill-color': [
        'match', ['get', 'status'],
        'analyzed', '#22c55e', 'seeded', '#eab308', '#6b7280',
      ],
      'fill-opacity': 0.15,
    } });
  map.addLayer({ id: 'coverage-outline', type: 'line', source: 'coverage',
    paint: { 'line-color': '#374151', 'line-width': 0.5, 'line-opacity': 0.4 } });

  // ── Layers — satellites ──────────────────────────────────────────────────
  // ── Layer — uncertain features (pulsing outer ring) ─────────────────────
  map.addLayer({ id: 'uncertain-ring', type: 'circle', source: 'speculative-uncertain',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'confidence'], 0, 14, 1, 18],
      'circle-color': 'transparent',
      'circle-stroke-width': 2,
      'circle-stroke-color': '#facc15',  // amber ring = needs review
      'circle-opacity': 0.85,
    } });

  // ── Layers — satellites ──────────────────────────────────────────────────
  map.addLayer({ id: 'satellite-layer', type: 'circle', source: 'satellites',
    paint: { 'circle-radius': 5, 'circle-color': '#ffffff', 'circle-stroke-width': 1.5, 'circle-stroke-color': '#60a5fa' } });
  map.addLayer({ id: 'satellite-label', type: 'symbol', source: 'satellites',
    layout: { 'text-field': ['get', 'name'], 'text-size': 9, 'text-offset': [0, 1.2] },
    paint: { 'text-color': '#ffffff', 'text-halo-color': '#000', 'text-halo-width': 1 } });

  sourcesInitialized = true;

  // ── Click handlers ───────────────────────────────────────────────────────
  ['infra-layer', 'estimate-layer', 'speculative-layer', 'satellite-layer'].forEach((layer) => {
    map.on('click', layer, (e) => {
      const f = e.features?.[0];
      if (f) details.innerHTML = popupHtml(f.properties);
    });
    map.on('mouseenter', layer, () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', layer, () => { map.getCanvas().style.cursor = ''; });
  });

  // ── Toggle listeners ─────────────────────────────────────────────────────
  [infraToggle, estToggle, rayToggle, sectorToggle, confidenceToggle, timeRange].forEach(
    (el) => el?.addEventListener('input', refreshSource)
  );
  spectToggle?.addEventListener('change', refreshSpeculative);
  coverageToggle?.addEventListener('change', () => {
    const vis = coverageToggle.checked ? 'visible' : 'none';
    ['coverage-fill', 'coverage-outline'].forEach((l) => map.setLayoutProperty(l, 'visibility', vis));
    if (coverageToggle.checked) refreshCoverage();
  });
  satelliteToggle?.addEventListener('change', () => {
    const vis = satelliteToggle.checked ? 'visible' : 'none';
    ['satellite-layer', 'satellite-label'].forEach((l) => map.setLayoutProperty(l, 'visibility', vis));
    if (satelliteToggle.checked) refreshSatellites();
  });
  satGroupSelect?.addEventListener('change', refreshSatellites);

  // ── Button listeners ──────────────────────────────────────────────────────
  analyzeBtn?.addEventListener('click', analyzeNow);
  seedBtn?.addEventListener('click', seedRegion);
  loopIntervalEl?.addEventListener('change', async () => {
    const interval = parseInt(loopIntervalEl.value, 10);
    if (selectedModel) await applyModelSelection(selectedModel);
    else await fetch(`/api/loop/config?interval_seconds=${interval}`, { method: 'POST' }).catch(() => {});
  });
  logHeader?.addEventListener('click', () => {
    if (analysisLog) analysisLog.classList.toggle('collapsed');
  });

  // ── Initial data load ─────────────────────────────────────────────────────
  refreshSource();
  refreshLoopStatus();
  refreshSdrStatus();
  populateModelDropdown();
  refreshSpeculative();
  refreshAnalysisLog();
  refreshCalibration();
  refreshUncertain();

  // ── Polling intervals ─────────────────────────────────────────────────────
  setInterval(refreshLoopStatus, 10000);
  setInterval(refreshSdrStatus, 30000);
  setInterval(refreshSpeculative, 15000);
  setInterval(refreshAnalysisLog, 30000);
  setInterval(refreshCoverage, 60000);
  setInterval(refreshCalibration, 60000);
  setInterval(refreshUncertain, 60000);
});

if (modelDropdown) {
  modelDropdown.addEventListener('change', () => applyModelSelection(modelDropdown.value));
}
