# ThingTrax Quality Agent Integration Guide

## Purpose

This project provides the inspection edge component for an AOI line. It reads a
camera frame, executes the exported ONNX segmentation model, audits defect
contours, records telemetry, and returns a deterministic `PASS` or `REJECT`
decision.

## Components

1. **Camera adapter:** supplies BGR `numpy.ndarray` frames.
2. **ONNX Runtime:** executes `checkpoints/best_magnetic_tile_model.onnx`.
3. **Inspection engine:** `industrial_edge_inspector.EdgeInspector`.
4. **Actuator adapter:** optional callback for a PLC, GPIO board, or Modbus
   client. The default example is simulation-only.
5. **Telemetry:** local SQLite audit queue, exportable to MES or cloud systems.

## Integration

```python
from pathlib import Path

from industrial_edge_inspector import EdgeInspector, SQLiteTelemetry

inspector = EdgeInspector(
    model_path=Path("checkpoints/best_magnetic_tile_model.onnx"),
    telemetry=SQLiteTelemetry(Path("inspection_telemetry.db")),
)
result = inspector.inspect(frame_bgr, part_id="tile-0001")
```

`result.overlay` is a BGR image with red defect regions and green contours.
`result.mask` is a binary mask. The decision changes to `REJECT` when the mask
contains at least 15 defect pixels.

## PLC/GPIO safety

Do not connect a physical actuator until the model, camera calibration,
threshold, reject timing, and fail-safe behavior have been validated on a
staging line. Supply an explicit actuator callback only after those checks.
The callback receives `(part_id, defect_pixels)` and must return a short
audit signal such as `PLC_REJECT_PULSE`.

## Runtime command

```bash
python industrial_edge_inspector.py ^
  --model checkpoints\best_magnetic_tile_model.onnx ^
  --image sample_tile.jpg ^
  --output inspection_overlay.png ^
  --telemetry inspection_telemetry.db
```

The command uses the simulated rejector and never writes to physical hardware.

## Production checklist

- Validate camera color order and image dimensions.
- Confirm the ONNX model output is a single binary segmentation channel.
- Calibrate the probability and pixel-area thresholds on held-out line data.
- Select and monitor the intended ONNX Runtime provider.
- Add PLC acknowledgement, timeout, and emergency-stop handling in the
  hardware adapter.
- Monitor latency, rejected-part rate, and telemetry write failures.
