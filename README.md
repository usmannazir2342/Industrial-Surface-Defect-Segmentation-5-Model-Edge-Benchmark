# Industrial Surface Defect Segmentation: 5-Model Edge Benchmark

An end-to-end Automated Optical Inspection (AOI) and edge defect segmentation
system for high-speed manufacturing conveyor lines. The project includes an
empirical five-model tournament, contour-based defect auditing, ONNX edge
benchmarking, and a simulated industrial PLC pneumatic rejection workflow.

## 1. Visual Inspection Demo

The notebook generates a real-time conveyor inspection simulation with:

- Red defect masks and green contour overlays
- Pixel-level defect-area auditing (`px`)
- FPS and latency telemetry
- Deterministic PASS / REJECT decisions

Run the notebook to generate the inspection outputs and benchmark artifacts.

## 2. Quantitative Benchmark Leaderboard

Results reported by the notebook on the unseen test split of the Magnetic Tile
Surface Defect Dataset (1,344 industrial tiles across `Blowhole`, `Break`,
`Crack`, `Fray`, and `Uneven` defects):

| Model | Encoder | Test Dice | Test IoU | Precision | Recall | Params | Latency | FPS | Best industrial use case |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| U-Net | ResNet34 | 0.8413 | 0.7666 | 0.8548 | 0.8616 | 24.44M | 43.46 ms | 23.0 | Critical safety lines and micro-cracks |
| DeepLabV3+ | ResNet34 | 0.8227 | 0.7468 | 0.8242 | 0.8539 | 22.44M | 32.73 ms | 30.5 | Diffuse blowholes and texture anomalies |
| FPN | ResNet34 | 0.8099 | 0.7262 | 0.8189 | 0.8381 | 23.16M | 31.61 ms | 31.6 | Balanced pyramidal features |
| SegFormer-B0 | MiT-B0 | 0.7568 | 0.6683 | 0.7315 | 0.8225 | 3.71M | 19.64 ms | 50.9 | Lightweight edge gateways |
| YOLOv8n-seg | CSPDarknet | 0.5636 | 0.4964 | 0.5548 | 0.6844 | 3.39M | 12.70 ms | 78.8 | High-speed diverters |

### Key architectural takeaways

1. **Precision champion:** U-Net (ResNet34) leads with a Dice score of
   **0.8413** and IoU of **0.7666**.
2. **Speed champion:** YOLOv8n-seg reaches **78.8 FPS** at **12.70 ms** per
   frame.
3. **Edge transformer:** SegFormer-B0 reaches **50.9 FPS** with only **3.71M**
   parameters.

## 3. End-to-End System Architecture

```text
[High-Speed Conveyor Belt]
           |
           v
[Industrial GigE / RTSP Camera] (60 FPS, Global Shutter)
           |
           v
[Edge Gateway: NVIDIA Jetson / Industrial IPC]
  |- Frame ingestion and normalization (cv2.cvtColor, ImageNet scaling)
  |- ONNX model execution
  |- Contour localization (cv2.findContours)
  |- Defect-area filtering (defect_px >= 15 px)
  |- GPIO / PLC Modbus signal to a pneumatic kicker arm
  `- SQLite audit queue -> CSV -> MES / Cloud
```

## 4. Key Engineering Features

- **Empirical five-model tournament:** U-Net, DeepLabV3+, FPN, SegFormer-B0,
  and YOLOv8n-seg.
- **Imbalance-aware training:** Combined BCE and Dice loss for defect pixels
  representing less than 2% of surface area.
- **Deterministic rejection:** Defect area is measured before the PASS / REJECT
  decision.
- **Telemetry-ready design:** Part ID, timestamp, defect class, area ratio,
  latency, and line-yield metrics can be written to local SQLite and CSV.
- **ONNX edge deployment:** The notebook benchmarks ONNX Runtime and provides
  an export path for TensorRT, OpenVINO, or ONNX Runtime deployments.

## 5. Repository Contents

```text
Industrial_Surface_Defect_Segmentation_5_Model_Edge_Benchmark.ipynb
requirements.txt
README.md
```

The notebook creates optional runtime artifacts such as prediction figures,
inspection GIFs, model checkpoints, and telemetry files. These generated
artifacts are intentionally not committed by default because they can be
large, environment-specific, or reproducible from the notebook.

## 6. Getting Started

```bash
python -m pip install -r requirements.txt
```

Open the notebook in Google Colab or Jupyter and run the cells in order. The
dataset is downloaded and audited by the notebook, followed by training,
evaluation, edge benchmarking, and export.

## Dataset

The notebook downloads the Magnetic Tile Surface Defect Dataset at runtime.
Please follow the dataset repository's terms and attribution requirements
when using the data.
