# Training Data Schema

Each record in a training JSON file should include:

- `timestamp` (ISO-8601)
- `lat`, `lon` (float, GPS)
- `heading_deg` (float)
- `bearing_estimate_deg` (float)
- `rssi_dbm` (float)
- `snr_db` (float)
- `frequency_hz` (float)
- `bandwidth_hz` (float)
- `terrain` (optional object), e.g. `{"elevation_m": 25.0}`

For supervised calibration/training, store paired arrays:

```json
{
  "samples": [ ...TelemetrySample objects... ],
  "target_error_m": [12.3, 10.1, 15.6],
  "model_version": "v20260429"
}
```

`public/data/telemetry_samples.json` is used only as an example seed file.
