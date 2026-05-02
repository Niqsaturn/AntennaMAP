const _DARK_STYLE = {
  version: 8,
  name: 'dark',
  glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
  sources: {
    'osm-tiles': {
      type: 'raster',
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      tileSize: 256,
      attribution: '© OpenStreetMap contributors',
    },
  },
  layers: [
    // Keep schema-safe base ordering: background first, then raster/vector layers only.
    { id: 'background', type: 'background', paint: { 'background-color': '#0f172a' } },
    { id: 'osm', type: 'raster', source: 'osm-tiles', paint: { 'raster-opacity': 0.55, 'raster-saturation': -0.7, 'raster-brightness-min': 0.05 } },
  ],
};

const map = new maplibregl.Map({
  container: 'map',
  style: _DARK_STYLE,
  projection: { type: 'globe' },
  center: [0, 20], zoom: 1.8, pitch: 0, bearing: 0, antialias: true,
});

// Globe slow-spin - paused while user interacts
let _spinActive = true;
let _spinResumeTimer = null;
function _spinGlobe() {
  try {
    if (_spinActive && map.getZoom() < 5) {
      map.setCenter([map.getCenter().lng + 0.06, map.getCenter().lat]);
    }
  } catch (e) {
    // silently fail spin on globe mode
  }
  requestAnimationFrame(_spinGlobe);
}

// Pause spin on user interaction
['mousedown', 'touchstart', 'wheel'].forEach((evt) =>
  map.on(evt, () => {
    _spinActive = false;
    clearTimeout(_spinResumeTimer);
    _spinResumeTimer = setTimeout(() => { _spinActive = true; }, 3000);
  })
);

// ── DOM refs ───────────────────────────────────────────────────────────────
const details         = document.getElementById('details');
const infraToggle     = document.getElementById('infraToggle');
const estToggle       = document.getElementById('estToggle');
const spectToggle     = document.getElementById('spectToggle');
const rayToggle       = document.getElementById('rayToggle');
const sectorToggle    = document.getElementById('sectorToggle');
const confidenceToggle= document.getElementById('confidenceToggle');
const rangeBandToggle = document.getElementById('rangeBandToggle');
const coverageToggle  = document.getElementById('coverageToggle');
const satelliteToggle = document.getElementById('satelliteToggle');
const bayesToggle     = document.getElementById('bayesToggle');
const timeRange       = document.getElementById('timeRange');
const timeLabel       = document.getElementById('timeLabel');
const loopStatus        = document.getElementById('loopStatus');
const sdrStatus         = document.getElementById('sdrStatus');
const coverageStatus    = document.getElementById('coverageStatus');
const calibrationStatus = document.getElementById('calibrationStatus');
const featureHealthStatus = document.getElementById('featureHealthStatus');
const renderHealthStatus = document.getElementById('renderHealthStatus');
const modelDropdown   = document.getElementById('modelSelect');
const satGroupSelect  = document.getElementById('satGroup');
const analyzeBtn      = document.getElementById('analyzeNow');
const seedBtn         = document.getElementById('seedRegion');
const loopIntervalEl  = document.getElementById('loopInterval');
const analysisLog     = document.getElementById('analysisLog');
const logHeader       = document.getElementById('analysisLogHeader');
const goToLatLon      = document.getElementById('goToLatLon');
const goToBtn         = document.getElementById('goToBtn');

let sortedTimes = [];
let sourcesInitialized = false;
let selectedModel = '';
const health = {
  lastFeatureFetchOkAt: null,
  counts: { infrastructure: 0, estimate: 0, speculative: 0, fox_targets: 0 },
  render: { features: 'pending', speculative: 'pending', satellites: 'pending', fox_targets: 'pending' },
};


function _hasValidGlyphs() {
  try {
    const styleGlyphs = map.getStyle && map.getStyle()?.glyphs;
    const configuredGlyphs = _DARK_STYLE?.glyphs;
    const glyphs = styleGlyphs || configuredGlyphs;
    return typeof glyphs === 'string' && glyphs.includes('{fontstack}') && glyphs.includes('{range}');
  } catch (_) {
    return false;
  }
}

function _safeSetSourceData(sourceId, data, statusKey) {
  try {
    const src = map.getSource(sourceId);
    if (!src) throw new Error(`source ${sourceId} missing`);
    src.setData(data);
    if (statusKey) health.render[statusKey] = 'ok';
    return true;
  } catch (err) {
    if (statusKey) health.render[statusKey] = `error: ${err.message}`;
    return false;
  }
}

function _renderHealth() {
  if (featureHealthStatus) {
    const lastOk = health.lastFeatureFetchOkAt ? new Date(health.lastFeatureFetchOkAt).toLocaleTimeString() : 'never';
    featureHealthStatus.innerHTML =
      `<strong>Features:</strong> last OK ${lastOk}<br>` +
      `infra ${health.counts.infrastructure} · estimate ${health.counts.estimate} · speculative ${health.counts.speculative} · fox ${health.counts.fox_targets}`;
  }
  if (renderHealthStatus) {
    renderHealthStatus.innerHTML =
      `<strong>Render:</strong> features ${health.render.features}<br>` +
      `spec ${health.render.speculative} · sats ${health.render.satellites} · fox ${health.render.fox_targets}`;
  }
}

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
  const overlayAssumptions = `<br>Assumptions: range from RSSI/SNR + propagation model; single-source geometry shown as likelihood, not fixed point.`;
  const count = p.analysis_count != null ? ` (${p.analysis_count} obs)` : '';
  const source = p.source ? `<br>Source: ${p.source}` : '';
  const notes = p.notes ? `<br>Notes: ${p.notes}` : '';
  const confirmRow = p.kind === 'speculative'
    ? `<div class="confirm-row"><button onclick="confirmFeature('${p.id}',true)">Confirm</button> <button onclick="confirmFeature('${p.id}',false)">Dismiss</button></div>`
    : '';
  return `<strong>${p.name || p.id}</strong><br>Kind: ${p.kind}${conf}${count}` +
    `<br>Freq: ${p.freq_band || '-'}<br>Type: ${p.antenna_type || p.structure_type || '-'}` +
    `<br>Azimuth: ${p.azimuth_deg ?? '-'}°${source}${notes}${overlayAssumptions}${confirmRow}`;
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
      `<strong>Model:</strong> ${s.config.model || '-'}<br>` +
      `<strong>Interval:</strong> ${s.config.interval_seconds}s · ` +
      `<strong>Last OK:</strong> ${last.slice(0, 19).replace('T', ' ')}`;
  } catch (_) { loopStatus.textContent = 'Loop: unavailable'; }
}

// ── Model Dropdown ─────────────────────────────────────────────────────────
async function populateModelDropdown() {
  if (!modelDropdown) return;
  try {
    const data = await fetch('/api/models/discover').then((r) => r.json());
    const options = [['', '- None -']];
    (data.ollama || []).forEach((m) => options.push([`ollama/${m}`, `Ollama: ${m}`]));
    (data.python_local || []).forEach((m) => options.push([`python_local/${m}`, `Local: ${m}`]));
    modelDropdown.innerHTML = options.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
    if (selectedModel) modelDropdown.value = selectedModel;
  } catch (_) {
    if (modelDropdown) modelDropdown.innerHTML = '<option value="">- unavailable -</option>';
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
    health.lastFeatureFetchOkAt = new Date().toISOString();
    const allFeatures = data.features || [];
    health.counts.infrastructure = allFeatures.filter((f) => f.properties.kind === 'infrastructure').length;
    health.counts.estimate = allFeatures.filter((f) => f.properties.kind === 'estimate').length;
    const visibleSites = (data.features || []).filter((f) =>
      (f.properties.kind === 'infrastructure' && infraToggle?.checked) ||
      (f.properties.kind === 'estimate' && estToggle?.checked)
    );

  const rays = [], sectors = [], confidence = [], rangeBands = [], uncertaintyPolygons = [];
  visibleSites.forEach((f) => {
    const o = f.properties.overlay_geometries || {};
    const ray = asFeature(f, o.direction_ray, 'direction_ray');
    const sector = asFeature(f, o.sector_wedge, 'sector_wedge');
    const ellipse = asFeature(f, o.confidence_ellipse, 'confidence_ellipse');
    const rangeBand = asFeature(f, o.range_likely_band, 'range_likely_band');
    const uncertainty = asFeature(f, o.uncertainty_polygon, 'uncertainty_polygon');
    if (ray) rays.push(ray);
    if (sector) sectors.push(sector);
    if (ellipse) confidence.push(ellipse);
    if (rangeBand) rangeBands.push(rangeBand);
    if (uncertainty) uncertaintyPolygons.push(uncertainty);
  });

    _safeSetSourceData('sites', { type: 'FeatureCollection', features: visibleSites }, 'features');
    _safeSetSourceData('rays', { type: 'FeatureCollection', features: rayToggle?.checked ? rays : [] }, 'features');
    _safeSetSourceData('sectors', { type: 'FeatureCollection', features: sectorToggle?.checked ? sectors : [] }, 'features');
    _safeSetSourceData('confidence', { type: 'FeatureCollection', features: confidenceToggle?.checked ? confidence : [] }, 'features');
    _safeSetSourceData('range-bands', { type: 'FeatureCollection', features: rangeBandToggle?.checked ? rangeBands : [] }, 'features');
    _safeSetSourceData('uncertainty-polygons', { type: 'FeatureCollection', features: confidenceToggle?.checked ? uncertaintyPolygons : [] }, 'features');
    _renderHealth();
  } catch (err) {
    health.render.features = `error: ${err.message}`;
    _renderHealth();
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
    health.counts.speculative = (data.features || []).length;

    const rays = [], sectors = [], ellipses = [], rangeBands = [], uncertaintyPolygons = [];
    (data.features || []).forEach((f) => {
      const o = f.properties.overlay_geometries || {};
      if (o.direction_ray) rays.push(asFeature(f, o.direction_ray, 'direction_ray'));
      if (o.sector_wedge) sectors.push(asFeature(f, o.sector_wedge, 'sector_wedge'));
      if (o.confidence_ellipse) ellipses.push(asFeature(f, o.confidence_ellipse, 'confidence_ellipse'));
      if (o.range_likely_band) rangeBands.push(asFeature(f, o.range_likely_band, 'range_likely_band'));
      if (o.uncertainty_polygon) uncertaintyPolygons.push(asFeature(f, o.uncertainty_polygon, 'uncertainty_polygon'));
    });

    _safeSetSourceData('speculative', {
      type: 'FeatureCollection',
      features: spectToggle.checked ? (data.features || []) : [],
    }, 'speculative');
    _safeSetSourceData('spec-rays', { type: 'FeatureCollection', features: rayToggle.checked ? rays : [] }, 'speculative');
    _safeSetSourceData('spec-sectors', { type: 'FeatureCollection', features: sectorToggle.checked ? sectors : [] }, 'speculative');
    _safeSetSourceData('spec-ellipses', { type: 'FeatureCollection', features: confidenceToggle.checked ? ellipses : [] }, 'speculative');
    _safeSetSourceData('spec-range-bands', { type: 'FeatureCollection', features: rangeBandToggle.checked ? rangeBands : [] }, 'speculative');
    _safeSetSourceData('spec-uncertainty-polygons', { type: 'FeatureCollection', features: confidenceToggle.checked ? uncertaintyPolygons : [] }, 'speculative');
  } catch (err) {
    health.render.speculative = `error: ${err.message}`;
  } finally { _renderHealth(); }
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
    _safeSetSourceData('satellites', data.geojson || { type: 'FeatureCollection', features: [] }, 'satellites');
  } catch (err) {
    health.render.satellites = `error: ${err.message}`;
  } finally { _renderHealth(); }
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
        `<strong>${e.model || '-'}</strong> · ${e.detections_count} detection(s)<br>` +
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
        `mean error ${d.mean_error_m != null ? d.mean_error_m.toFixed(0) : '-'} m · ` +
        `median ${d.median_error_m != null ? d.median_error_m.toFixed(0) : '-'} m`;
    }
  } catch (_) { if (calibrationStatus) calibrationStatus.textContent = 'Calibration: -'; }
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



function parseLatLon(input) {
  if (!input) return null;
  const parts = input.split(',').map((part) => Number(part.trim()));
  if (parts.length !== 2 || parts.some((n) => Number.isNaN(n))) return null;
  const [lat, lon] = parts;
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
  return { lat, lon };
}

function setupGoToLatLon() {
  const go = () => {
    const parsed = parseLatLon(goToLatLon?.value || '');
    if (!parsed) {
      if (details) details.textContent = 'Invalid lat/lon. Use format: lat, lon (e.g. 37.7749, -122.4194).';
      return;
    }
    map.flyTo({ center: [parsed.lon, parsed.lat], zoom: Math.max(map.getZoom(), 9), speed: 0.8 });
    if (details) details.textContent = `Moved map to ${parsed.lat.toFixed(5)}, ${parsed.lon.toFixed(5)}.`;
  };
  goToBtn?.addEventListener('click', go);
  goToLatLon?.addEventListener('keydown', (evt) => {
    if (evt.key === 'Enter') go();
  });
}
// ── Map load ───────────────────────────────────────────────────────────────
map.on('load', async () => {
  // Setup globe fog (when supported) and spin
  const canSetFog = typeof map.setFog === 'function';
  const projection = typeof map.getProjection === 'function' ? map.getProjection() : null;
  const projectionName = projection && typeof projection === 'object' ? projection.name : null;
  const fogCompatibleProjection = !projectionName || projectionName === 'globe';

  if (canSetFog && fogCompatibleProjection) {
    map.setFog({
      color: 'rgba(15,23,42,0.85)',
      'high-color': '#1e3a5f',
      'horizon-blend': 0.04,
      'space-color': '#0f172a',
      'star-intensity': 0.35,
    });
  }

  requestAnimationFrame(_spinGlobe);

  map.addControl(new maplibregl.NavigationControl(), 'bottom-right');
  map.addControl(new maplibregl.GeolocateControl({
    positionOptions: { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 },
    trackUserLocation: true,
    showUserHeading: true,
  }), 'bottom-right');
  map.addControl(new maplibregl.ScaleControl({ maxWidth: 120, unit: 'metric' }), 'bottom-left');
  map.addControl(new maplibregl.FullscreenControl(), 'top-right');
  setupGoToLatLon();

  // Seed GeoJSON features from API
  const seed = await fetch('/api/features').then((r) => r.json()).catch(() => ({ features: [] }));
  sortedTimes = [...new Set(seed.features.map((f) => f.properties.timestamp))].filter(Boolean).sort();

  // ── Sources ──────────────────────────────────────────────────────────────
  map.addSource('sites',      { type: 'geojson', data: seed });
  map.addSource('rays',       { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('sectors',    { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('confidence', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('range-bands', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('uncertainty-polygons', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('speculative',{ type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('spec-rays',  { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('spec-sectors',{ type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('spec-ellipses',{ type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('spec-range-bands',{ type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('spec-uncertainty-polygons',{ type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('coverage',             { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('satellites',           { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
  map.addSource('speculative-uncertain',{ type: 'geojson', data: { type: 'FeatureCollection', features: [] } });

  const canRenderSymbolText = _hasValidGlyphs();

  // ── Layers - base ────────────────────────────────────────────────────────
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
  map.addLayer({ id: 'range-band-layer',type: 'fill', source: 'range-bands',
    paint: { 'fill-color': '#38bdf8', 'fill-opacity': 0.12 } });
  map.addLayer({ id: 'uncertainty-polygon-layer',type: 'line', source: 'uncertainty-polygons',
    paint: { 'line-color': '#f59e0b', 'line-width': 1.5, 'line-dasharray': [2, 2] } });

  // ── Layers - speculative ─────────────────────────────────────────────────
  map.addLayer({ id: 'spec-sector-layer', type: 'fill', source: 'spec-sectors',
    paint: { 'fill-color': '#c084fc', 'fill-opacity': 0.15 } });
  map.addLayer({ id: 'spec-ellipse-layer', type: 'fill', source: 'spec-ellipses',
    paint: { 'fill-color': '#c084fc', 'fill-opacity': 0.12 } });
  map.addLayer({ id: 'spec-range-band-layer', type: 'fill', source: 'spec-range-bands',
    paint: { 'fill-color': '#a78bfa', 'fill-opacity': 0.1 } });
  map.addLayer({ id: 'spec-uncertainty-polygon-layer', type: 'line', source: 'spec-uncertainty-polygons',
    paint: { 'line-color': '#e879f9', 'line-width': 1.2, 'line-dasharray': [2, 2] } });
  map.addLayer({ id: 'spec-ray-layer', type: 'line', source: 'spec-rays',
    paint: { 'line-color': '#c084fc', 'line-width': 1.5, 'line-dasharray': [4, 2] } });
  map.addLayer({ id: 'speculative-layer', type: 'circle', source: 'speculative',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'confidence'], 0, 5, 1, 9],
      'circle-color': '#c084fc', 'circle-opacity': 0.8,
      'circle-stroke-width': 1, 'circle-stroke-color': '#fff',
    } });

  // ── Layers - coverage grid ───────────────────────────────────────────────
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

  // ── Layers - satellites ──────────────────────────────────────────────────
  // ── Layer - uncertain features (pulsing outer ring) ─────────────────────
  map.addLayer({ id: 'uncertain-ring', type: 'circle', source: 'speculative-uncertain',
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['get', 'confidence'], 0, 14, 1, 18],
      'circle-color': 'transparent',
      'circle-stroke-width': 2,
      'circle-stroke-color': '#facc15',  // amber ring = needs review
      'circle-opacity': 0.85,
    } });

  // ── Layers - satellites ──────────────────────────────────────────────────
  map.addLayer({ id: 'satellite-layer', type: 'circle', source: 'satellites',
    paint: { 'circle-radius': 5, 'circle-color': '#ffffff', 'circle-stroke-width': 1.5, 'circle-stroke-color': '#60a5fa' } });
  if (canRenderSymbolText) {
    map.addLayer({ id: 'satellite-label', type: 'symbol', source: 'satellites',
      layout: { 'text-field': ['get', 'name'], 'text-font': ['Noto Sans Regular'], 'text-size': 9, 'text-offset': [0, 1.2] },
      paint: { 'text-color': '#ffffff', 'text-halo-color': '#000', 'text-halo-width': 1 } });
  } else {
    console.warn('Map style glyphs unavailable; skipping satellite symbol labels.');
  }

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
  [infraToggle, estToggle, rayToggle, sectorToggle, confidenceToggle, rangeBandToggle, timeRange].forEach(
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
    ['satellite-layer', ...(map.getLayer('satellite-label') ? ['satellite-label'] : [])].forEach((l) => map.setLayoutProperty(l, 'visibility', vis));
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

  // ── Fox hunt / waterfall / SSE initialization ─────────────────────────────
  _addFoxSources();
  _initWaterfall();
  _refreshNodeList();
  _refreshFoxStatus();
  _connectSSE();
  setInterval(_refreshFoxStatus, 15000);
  setInterval(_refreshNodeList, 60000);
});

if (modelDropdown) {
  modelDropdown.addEventListener('change', () => applyModelSelection(modelDropdown.value));
}

// ═══════════════════════════════════════════════════════════════════════════
// ── FOX HUNT + WATERFALL + SSE ────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

// ── DOM refs (fox panel) ──────────────────────────────────────────────────
const foxStateBadge  = document.getElementById('foxStateBadge');
const foxTargetBox   = document.getElementById('foxTargetBox');
const foxTargetFreq  = document.getElementById('foxTargetFreq');
const foxTargetMeta  = document.getElementById('foxTargetMeta');
const foxObsCount    = document.getElementById('foxObsCount');
const foxStart       = document.getElementById('foxStart');
const foxStop        = document.getElementById('foxStop');
const foxLatEl       = document.getElementById('foxLat');
const foxLonEl       = document.getElementById('foxLon');
const addBearingBtn  = document.getElementById('addBearing');
const bearingDegEl   = document.getElementById('bearingDeg');
const bearingSnrEl   = document.getElementById('bearingSnr');
const confirmedList  = document.getElementById('confirmedList');
const nodeList       = document.getElementById('nodeList');
const rediscoverBtn       = document.getElementById('rediscoverNodes');
const nodeDiscoveryStatus = document.getElementById('nodeDiscoveryStatus');
const waterfallCanvas= document.getElementById('waterfallCanvas');
const wfFreqAxis     = document.getElementById('waterfallFreqAxis');
const waterfallStatus = document.getElementById('waterfallStatus');
const peakList       = document.getElementById('peakList');
const foxToggle      = document.getElementById('foxToggle');
const bearingRayToggle = document.getElementById('bearingRayToggle');
const waterfallHeader = document.getElementById('waterfallHeader');
const waterfallContainer = document.getElementById('waterfallContainer');
const peakListHeader = document.getElementById('peakListHeader');
const autoHuntToggle = document.getElementById('autoHuntToggle');
const autoPolicyPhase = document.getElementById('autoPolicyPhase');
let _autoHuntEnabled = false;

// ── Waterfall state ───────────────────────────────────────────────────────
const WF_BINS = 1024;
const WF_ROWS = 120;   // pixel height
let _wfCtx = null;
let _wfImageData = null;
let _wfRowCount = 0;
let _wfBins = WF_BINS;
let _wfCenterFreqHz = null;
let _wfBandwidthHz = null;
let _wfLastFrameAt = 0;
let _wfDecodeMismatchCount = 0;
let _wfLastSeqByNode = new Map();
let _wfDroppedFrameCount = 0;
const _wfStaleTimeoutMs = 15000;

function _initWaterfall() {
  if (!waterfallCanvas) return;
  waterfallCanvas.width = _wfBins;
  waterfallCanvas.height = WF_ROWS;
  _wfCtx = waterfallCanvas.getContext('2d');
  _wfImageData = _wfCtx.createImageData(_wfBins, WF_ROWS);
  _wfImageData.data.fill(0);
  _renderWaterfallFreqAxis();
  _setWaterfallStatus('connected');
}

function _renderWaterfallFreqAxis() {
  if (!wfFreqAxis) return;
  if (!_wfCenterFreqHz || !_wfBandwidthHz) {
    wfFreqAxis.innerHTML = [0,5,10,15,20,25,30].map((f) => `<span>${f}</span>`).join('');
    return;
  }
  const steps = 7;
  const low = _wfCenterFreqHz - (_wfBandwidthHz / 2);
  const binBwHz = _wfBandwidthHz / Math.max(1, _wfBins);
  wfFreqAxis.innerHTML = Array.from({ length: steps }, (_, i) => {
    const frac = i / (steps - 1);
    const bin = frac * (_wfBins - 1);
    const hz = low + (bin * binBwHz);
    return `<span>${(hz / 1e6).toFixed(3)}</span>`;
  }).join('');
}

function _setWaterfallStatus(state, detail = '') {
  if (!waterfallStatus) return;
  waterfallStatus.className = 'waterfall-status';
  if (state === 'receiving') {
    waterfallStatus.classList.add('status-receiving');
    waterfallStatus.textContent = detail || 'Receiving data';
  } else if (state === 'mismatch') {
    waterfallStatus.classList.add('status-mismatch');
    waterfallStatus.textContent = detail || 'Decode/config mismatch';
  } else {
    waterfallStatus.classList.add('status-connected');
    waterfallStatus.textContent = detail || 'Connected · waiting for data';
  }
}

// Map dB value to RGBA colour (dark blue → cyan → yellow → red)
function _dbToRgba(db) {
  // Normalise -127..−13 dBm → 0..1
  const t = Math.max(0, Math.min(1, (db + 127) / 114));
  let r, g, b;
  if (t < 0.25) {          // dark navy → blue
    const s = t / 0.25;
    r = 0; g = Math.round(s * 50); b = Math.round(50 + s * 150);
  } else if (t < 0.5) {   // blue → cyan
    const s = (t - 0.25) / 0.25;
    r = 0; g = Math.round(50 + s * 205); b = 200;
  } else if (t < 0.75) {  // cyan → yellow
    const s = (t - 0.5) / 0.25;
    r = Math.round(s * 255); g = 255; b = Math.round(200 * (1 - s));
  } else {                 // yellow → red
    const s = (t - 0.75) / 0.25;
    r = 255; g = Math.round(255 * (1 - s)); b = 0;
  }
  return [r, g, b];
}

function _pushWaterfallRow(bins) {
  if (!_wfCtx || !_wfImageData || bins.length < _wfBins) return;
  // Scroll existing rows up by 1 (shift pixel data up by WF_BINS*4 bytes)
  const data = _wfImageData.data;
  data.copyWithin(0, _wfBins * 4);
  // Write new row at bottom
  const baseIdx = (WF_ROWS - 1) * _wfBins * 4;
  for (let i = 0; i < _wfBins; i++) {
    const [r, g, b] = _dbToRgba(bins[i]);
    const idx = baseIdx + i * 4;
    data[idx] = r; data[idx + 1] = g; data[idx + 2] = b; data[idx + 3] = 255;
  }
  _wfCtx.putImageData(_wfImageData, 0, 0);
  _wfRowCount++;
}

// ── Map sources & layers for fox hunt ────────────────────────────────────
let _foxSourcesAdded = false;

function _addFoxSources() {
  if (_foxSourcesAdded || !map.getStyle()) return;
  _foxSourcesAdded = true;
  const empty = { type: 'FeatureCollection', features: [] };
  map.addSource('fox-targets',  { type: 'geojson', data: empty });
  map.addSource('fox-ellipses', { type: 'geojson', data: empty });
  map.addSource('fox-bearings', { type: 'geojson', data: empty });
  map.addSource('bayes-field',  { type: 'geojson', data: empty });

  // Bayesian posterior heatmap (rendered below fox layers)
  map.addLayer({ id: 'bayes-fill', type: 'fill', source: 'bayes-field',
    layout: { visibility: 'none' },
    paint: {
      'fill-color': ['interpolate', ['linear'], ['get', 'probability'],
        0,      'rgba(59,130,246,0)',
        0.001,  'rgba(59,130,246,0.4)',
        0.01,   'rgba(245,158,11,0.55)',
        0.05,   'rgba(239,68,68,0.7)',
      ],
      'fill-opacity': 1,
    } });

  // Uncertainty ellipses (fill + outline)
  map.addLayer({ id: 'fox-ellipse-fill', type: 'fill', source: 'fox-ellipses',
    paint: { 'fill-color': '#4ade80', 'fill-opacity': 0.1 } });
  map.addLayer({ id: 'fox-ellipse-outline', type: 'line', source: 'fox-ellipses',
    paint: { 'line-color': '#4ade80', 'line-width': 1.5, 'line-dasharray': [3, 2] } });

  // Bearing rays from observer positions
  map.addLayer({ id: 'fox-bearing-rays', type: 'line', source: 'fox-bearings',
    paint: { 'line-color': '#fbbf24', 'line-width': 1.5, 'line-opacity': 0.7 } });

  // Confirmed target circles
  map.addLayer({ id: 'fox-target-layer', type: 'circle', source: 'fox-targets',
    paint: {
      'circle-radius': 11,
      'circle-color': '#4ade80',
      'circle-stroke-width': 2.5,
      'circle-stroke-color': '#fff',
    } });
  if (_hasValidGlyphs()) {
    map.addLayer({ id: 'fox-target-label', type: 'symbol', source: 'fox-targets',
      layout: { 'text-field': ['get', 'freq_label'], 'text-font': ['Noto Sans Regular'], 'text-size': 9, 'text-offset': [0, 1.6] },
      paint: { 'text-color': '#4ade80', 'text-halo-color': '#000', 'text-halo-width': 1 } });
  } else {
    console.warn('Map style glyphs unavailable; skipping fox target symbol labels.');
  }

  // Click handler for fox targets
  map.on('click', 'fox-target-layer', (e) => {
    const f = e.features?.[0];
    if (!f || !details) return;
    const p = f.properties;
    const methods = typeof p.methods === 'string' ? JSON.parse(p.methods) : (p.methods || []);
    details.innerHTML = `<strong>${p.name}</strong><br>` +
      `Freq: ${p.freq_mhz} MHz<br>Band: ${p.band_label}<br>` +
      `Confidence: ${(p.confidence * 100).toFixed(0)}%<br>` +
      `Uncertainty: ±${p.uncertainty_m?.toFixed(0)} m<br>` +
      `Methods: ${methods.join(', ')}<br>` +
      `Bearings: ${p.bearing_obs_count} · RSSI: ${p.rssi_obs_count}`;
  });
  map.on('mouseenter', 'fox-target-layer', () => { map.getCanvas().style.cursor = 'pointer'; });
  map.on('mouseleave', 'fox-target-layer', () => { map.getCanvas().style.cursor = ''; });
}

// Live bearing rays accumulator
const _bearingObs = [];   // {lat, lon, bearing_deg}

function _updateBearingRaySource() {
  const src = map.getSource('fox-bearings');
  if (!src) return;
  // For each observation draw a 30 km line in the bearing direction
  const RAY_M = 30_000;
  const lines = _bearingObs.map((o) => {
    const brRad = (o.bearing_deg * Math.PI) / 180;
    const dR = RAY_M / 6_371_000;
    const lat1 = (o.lat * Math.PI) / 180;
    const lon1 = (o.lon * Math.PI) / 180;
    const lat2 = Math.asin(Math.sin(lat1) * Math.cos(dR) + Math.cos(lat1) * Math.sin(dR) * Math.cos(brRad));
    const lon2 = lon1 + Math.atan2(Math.sin(brRad) * Math.sin(dR) * Math.cos(lat1),
                                    Math.cos(dR) - Math.sin(lat1) * Math.sin(lat2));
    return {
      type: 'Feature',
      geometry: { type: 'LineString', coordinates: [
        [o.lon, o.lat],
        [(lon2 * 180) / Math.PI, (lat2 * 180) / Math.PI],
      ]},
      properties: { bearing_deg: o.bearing_deg },
    };
  });
  src.setData({ type: 'FeatureCollection', features: lines });
}

// ── Fox hunt state badge ──────────────────────────────────────────────────
const STATE_CLASSES = {
  IDLE: 'state-idle', SCANNING: 'state-scanning', ACQUIRING: 'state-acquiring',
  COLLECTING: 'state-collecting', SOLVING: 'state-solving', CONFIRMED: 'state-confirmed',
};

function _setFoxState(state) {
  if (!foxStateBadge) return;
  foxStateBadge.textContent = state;
  foxStateBadge.className = `fox-state-badge ${STATE_CLASSES[state] || 'state-idle'}`;
}

// ── SSE event handler ─────────────────────────────────────────────────────
let _sseConnected = false;
let _sseSource = null;

function _connectSSE() {
  if (_sseConnected) return;
  _sseSource = new EventSource('/api/events');
  _sseSource.onopen = () => { _sseConnected = true; };
  _sseSource.onerror = () => {
    _sseConnected = false;
    setTimeout(_connectSSE, 5000);
  };
  _sseSource.onmessage = (e) => {
    let ev;
    try { ev = JSON.parse(e.data); } catch { return; }
    _handleSseEvent(ev);
  };
}

function _handleSseEvent(ev) {
  switch (ev.type) {
    case 'fox_state':
      _setFoxState(ev.state);
      if (ev.state === 'IDLE' || ev.state === 'SCANNING') {
        if (foxTargetBox) foxTargetBox.style.display = 'none';
      }
      break;

    case 'scan_results':
      _renderPeakList(ev.peaks || []);
      break;

    case 'rssi_acquired':
      // Update node RSSI display
      break;

    case 'bearing_added':
      _bearingObs.push({ lat: ev.lat, lon: ev.lon, bearing_deg: ev.bearing_deg });
      _updateBearingRaySource();
      if (foxObsCount)
        foxObsCount.textContent = `${ev.total_bearings} bearing(s) logged${ev.source ? ` · ${ev.source}` : ''}`;
      break;

    case 'estimate_updated': {
      // Update live ellipse on map
      const src = map.getSource('fox-ellipses');
      if (src && ev.lat && ev.lon) {
        const coords = _buildEllipseCoords(ev.lat, ev.lon, ev.ellipse_major_m || ev.uncertainty_m || 500, ev.ellipse_minor_m || ev.uncertainty_m * 0.6 || 300);
        src.setData({ type: 'FeatureCollection', features: [{
          type: 'Feature',
          geometry: { type: 'Polygon', coordinates: [coords] },
          properties: { confidence: ev.confidence },
        }]});
      }
      break;
    }

    case 'target_pinned': {
      // Render confirmed target on map
      const src = map.getSource('fox-targets');
      if (src && ev.feature) {
        const existing = src._data?.features || [];
        ev.feature.properties.freq_label = `${(ev.freq_hz / 1e6).toFixed(2)} MHz`;
        existing.push(ev.feature);
        health.counts.fox_targets = existing.length;
        _safeSetSourceData('fox-targets', { type: 'FeatureCollection', features: existing }, 'fox_targets');
      }
      // Add to confirmed list panel
      _addConfirmedItem(ev);
      // Clear bearing rays for next target
      _bearingObs.length = 0;
      _updateBearingRaySource();
      _renderHealth();
      break;
    }

    case 'sdr_frame': {
      const bins = ev.fft_bins || ev.bins || ev.bins_db;
      const centerHz = Number(ev.center_freq_hz);
      const bwHz = Number(ev.span_hz || ev.bw_hz || ev.sample_rate_hz);
      const nodeKey = `${ev?.source?.host || ev.node || 'unknown'}:${ev?.source?.port || 8073}`;
      const frameSeq = Number(ev.frame_seq);
      if (!Array.isArray(bins) || bins.length < 4) {
        _wfDecodeMismatchCount += 1;
        _setWaterfallStatus('mismatch', 'Decode/config mismatch: missing waterfall bins');
        break;
      }
      if (!Number.isFinite(centerHz) || !Number.isFinite(bwHz) || bwHz <= 0) {
        _wfDecodeMismatchCount += 1;
        _setWaterfallStatus('mismatch', 'Decode/config mismatch: invalid center frequency/sample rate');
        break;
      }
      if (_wfBins !== bins.length) {
        _wfBins = bins.length;
        _initWaterfall();
      }
      if (Number.isFinite(frameSeq)) {
        const prevSeq = _wfLastSeqByNode.get(nodeKey);
        if (Number.isFinite(prevSeq) && frameSeq > (prevSeq + 1)) {
          _wfDroppedFrameCount += (frameSeq - prevSeq - 1);
        }
        _wfLastSeqByNode.set(nodeKey, frameSeq);
      }
      _wfCenterFreqHz = centerHz;
      _wfBandwidthHz = bwHz;
      _renderWaterfallFreqAxis();
      _pushWaterfallRow(bins);
      _wfLastFrameAt = Date.now();
      _setWaterfallStatus(
        'receiving',
        `Receiving data · ${_wfBins} bins · ${(centerHz/1e6).toFixed(3)} MHz center · ${(bwHz/1e3).toFixed(1)} kHz span · dropped ${_wfDroppedFrameCount}`
      );
      if (_wfDecodeMismatchCount > 0) {
        _wfDecodeMismatchCount = 0;
      }
      break;
    }
  }
}

setInterval(() => {
  if (!_sseConnected) {
    _setWaterfallStatus('mismatch', 'Decode/config mismatch: SSE disconnected');
    return;
  }
  if (_wfLastFrameAt === 0 || (Date.now() - _wfLastFrameAt) > _wfStaleTimeoutMs) {
    const staleMs = _wfLastFrameAt === 0 ? null : (Date.now() - _wfLastFrameAt);
    const staleNote = staleMs == null ? 'waiting for data' : `stale stream timeout (${Math.round(staleMs / 1000)}s)`;
    _setWaterfallStatus('connected', `Connected · ${staleNote} · dropped ${_wfDroppedFrameCount}`);
  }
}, 3000);

// ── Ellipse polygon helper ────────────────────────────────────────────────
function _buildEllipseCoords(lat, lon, majorM, minorM, angleDeg = 0, steps = 48) {
  const cosA = Math.cos((angleDeg * Math.PI) / 180);
  const sinA = Math.sin((angleDeg * Math.PI) / 180);
  const cosLat = Math.cos((lat * Math.PI) / 180);
  const coords = [];
  for (let i = 0; i <= steps; i++) {
    const theta = (2 * Math.PI * i) / steps;
    const x = majorM * Math.cos(theta);
    const y = minorM * Math.sin(theta);
    const xe = x * cosA - y * sinA;
    const yn = x * sinA + y * cosA;
    coords.push([
      lon + xe / (111_320 * Math.max(cosLat, 0.001)),
      lat + yn / 111_320,
    ]);
  }
  return coords;
}

// ── Confirmed target panel item ───────────────────────────────────────────
function _addConfirmedItem(ev) {
  if (!confirmedList) return;
  const item = document.createElement('div');
  item.className = 'confirmed-item';
  const freqMhz = (ev.freq_hz / 1e6).toFixed(3);
  const conf = ev.confidence != null ? `${(ev.confidence * 100).toFixed(0)}%` : '-';
  item.innerHTML = `<span class="confirmed-freq">${freqMhz} MHz</span><span class="confirmed-conf">${conf}</span>`;
  item.onclick = () => {
    if (ev.feature?.geometry?.coordinates) {
      map.flyTo({ center: ev.feature.geometry.coordinates, zoom: 14 });
    }
  };
  confirmedList.insertBefore(item, confirmedList.firstChild);
}

// ── Peak list renderer ────────────────────────────────────────────────────
function _renderPeakList(peaks) {
  if (!peakList) return;
  peakList.innerHTML = peaks.slice(0, 20).map((p) => {
    const mhz = (p.freq_hz / 1e6).toFixed(3);
    return `<div class="peak-item" onclick="_tuneToFreq(${p.freq_hz})">
      <span class="peak-freq">${mhz} MHz</span>
      <span class="peak-snr">${p.snr_db?.toFixed(1)} dB</span>
      <span class="peak-mod">${p.modulation_hint || ''}</span>
    </div>`;
  }).join('');
}

window._tuneToFreq = function(freqHz) {
  if (bearingDegEl) bearingDegEl.dataset.freq = freqHz;
  // Pan map to nearest confirmed feature at that frequency if one exists
};

// ── Node list renderer ────────────────────────────────────────────────────
async function _refreshNodeList() {
  const data = await fetch('/api/sdr/nodes').then((r) => r.json()).catch(() => ({ nodes: [] }));
  const nodes = data.nodes || [];
  if (nodeDiscoveryStatus && nodes.length > 0) {
    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    nodeDiscoveryStatus.textContent = `${nodes.length} nodes · updated ${now}`;
  }
  if (!nodeList) return;
  nodeList.innerHTML = nodes.map((n) =>
    `<div class="node-item" data-hostport="${n.host}:${n.port}">
       <span class="node-dot unknown"></span>
       <span class="node-label">${n.host}:${n.port}${n.description ? ' · ' + n.description : ''}</span>
     </div>`
  ).join('') || '<div style="color:#475569;font-size:11px;padding:4px">Discovering…</div>';
}

// ── Fox hunt button handlers ──────────────────────────────────────────────
foxStart?.addEventListener('click', async () => {
  const lat = parseFloat(foxLatEl?.value || '0');
  const lon = parseFloat(foxLonEl?.value || '0');
  await fetch('/api/foxhunt/auto/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lat, lon }),
  }).catch(() => {});
  _setFoxState('SCANNING');
  if (foxTargetBox) foxTargetBox.style.display = 'block';
});

foxStop?.addEventListener('click', async () => {
  await fetch('/api/foxhunt/auto/stop', { method: 'POST' }).catch(() => {});
  _setFoxState('IDLE');
});

addBearingBtn?.addEventListener('click', async () => {
  const bearing = parseFloat(bearingDegEl?.value);
  const snr = parseFloat(bearingSnrEl?.value || '10');
  if (isNaN(bearing)) return;
  const lat = parseFloat(foxLatEl?.value || '0');
  const lon = parseFloat(foxLonEl?.value || '0');
  await fetch('/api/foxhunt/auto/observe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ bearing_deg: bearing, snr_db: snr, lat, lon, source: 'manual' }),
  }).catch(() => {});
});

autoHuntToggle?.addEventListener('click', async () => {
  const lat = parseFloat(foxLatEl?.value || '0');
  const lon = parseFloat(foxLonEl?.value || '0');
  if (!_autoHuntEnabled) {
    await fetch('/api/foxhunt/autonomous/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat, lon }),
    }).catch(() => {});
    _autoHuntEnabled = true;
  } else {
    await fetch('/api/foxhunt/autonomous/stop', { method: 'POST' }).catch(() => {});
    _autoHuntEnabled = false;
  }
  autoHuntToggle.textContent = `Auto Hunt: ${_autoHuntEnabled ? 'ON' : 'OFF'}`;
});

rediscoverBtn?.addEventListener('click', async () => {
  if (rediscoverBtn) rediscoverBtn.textContent = '↺ Discovering…';
  const data = await fetch('/api/sdr/nodes/discover', { method: 'POST' })
    .then((r) => r.json()).catch(() => null);
  if (nodeDiscoveryStatus && data) {
    nodeDiscoveryStatus.textContent = `+${data.added} discovered · ${data.total} total`;
  }
  if (rediscoverBtn) rediscoverBtn.textContent = '↺ Rediscover';
  await _refreshNodeList();
});

// Toggle visibility of fox layers
foxToggle?.addEventListener('change', () => {
  const vis = foxToggle.checked ? 'visible' : 'none';
  ['fox-target-layer', 'fox-target-label', 'fox-ellipse-fill',
   'fox-ellipse-outline'].forEach((l) => {
    try { map.setLayoutProperty(l, 'visibility', vis); } catch {}
  });
});

bearingRayToggle?.addEventListener('change', () => {
  const vis = bearingRayToggle.checked ? 'visible' : 'none';
  try { map.setLayoutProperty('fox-bearing-rays', 'visibility', vis); } catch {}
});

bayesToggle?.addEventListener('change', () => {
  const vis = bayesToggle.checked ? 'visible' : 'none';
  try { map.setLayoutProperty('bayes-fill', 'visibility', vis); } catch {}
  if (bayesToggle.checked) refreshBayesField();
});

async function refreshBayesField() {
  if (!bayesToggle?.checked) return;
  try {
    const c = map.getCenter();
    const url = `/api/bayes/field?center_lat=${c.lat.toFixed(4)}&center_lon=${c.lng.toFixed(4)}`;
    const data = await fetch(url).then((r) => r.json());
    const src = map.getSource('bayes-field');
    if (src) src.setData(data);
  } catch {}
}

// Collapsible sections in fox panel
waterfallHeader?.addEventListener('click', () => {
  waterfallContainer?.classList.toggle('collapsed');
});
peakListHeader?.addEventListener('click', () => {
  peakList?.classList.toggle('collapsed');
});

// Poll fox hunt status to sync UI when SSE is unavailable
async function _refreshFoxStatus() {
  try {
    const s = await fetch('/api/foxhunt/auto/status').then((r) => r.json());
    _setFoxState(s.state || 'IDLE');
    if (s.target && foxTargetBox) {
      foxTargetBox.style.display = 'block';
      if (foxTargetFreq) foxTargetFreq.textContent = `${(s.target.freq_hz / 1e6).toFixed(3)} MHz`;
      if (foxTargetMeta) foxTargetMeta.textContent = `${s.target.band_label} · ${s.target.modulation_hint}`;
      if (foxObsCount) foxObsCount.textContent = `${s.target.bearing_obs_count} bearing(s) · ${s.target.rssi_obs_count} RSSI`;
    }
    if (_autoHuntEnabled) {
      const cycle = await fetch('/api/foxhunt/autonomous/cycle', { method: 'POST' }).then((r) => r.json()).catch(() => null);
      if (cycle?.phase && autoPolicyPhase) autoPolicyPhase.textContent = `Policy phase: ${cycle.phase}`;
    }
    const auto = await fetch('/api/foxhunt/autonomous/status').then((r) => r.json()).catch(() => null);
    if (auto?.policy) {
      _autoHuntEnabled = Boolean(auto.policy.running);
      if (autoHuntToggle) autoHuntToggle.textContent = `Auto Hunt: ${_autoHuntEnabled ? 'ON' : 'OFF'}`;
      if (autoPolicyPhase) autoPolicyPhase.textContent = `Policy phase: ${auto.policy.phase || 'IDLE'}`;
    }
  } catch {}
}
