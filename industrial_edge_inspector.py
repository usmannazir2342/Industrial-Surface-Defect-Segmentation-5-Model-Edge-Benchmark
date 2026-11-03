"""ONNX-based industrial surface-defect inspection engine.

The engine is hardware-agnostic by default. A PLC/GPIO integration can be
provided as an actuator callback after the inference and telemetry paths have
been validated in the target environment.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import onnxruntime as ort

LOGGER = logging.getLogger("industrial_edge_inspector")


@dataclass(frozen=True)
class InspectionResult:
    part_id: str
    decision: str
    defect_pixels: int
    defect_ratio: float
    latency_ms: float
    fps: float
    mask: np.ndarray
    overlay: np.ndarray
    actuator_signal: str


Actuator = Callable[[str, int], str]


class SQLiteTelemetry:
    """Persist inspection results in a small local audit database."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspected_at TEXT NOT NULL,
                    part_id TEXT NOT NULL,
                    defect_pixels INTEGER NOT NULL,
                    defect_ratio REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    decision TEXT NOT NULL,
                    actuator_signal TEXT NOT NULL
                )
                """
            )

    def record(self, result: InspectionResult) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                INSERT INTO inspections (
                    inspected_at, part_id, defect_pixels, defect_ratio,
                    latency_ms, decision, actuator_signal
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    result.part_id,
                    result.defect_pixels,
                    result.defect_ratio,
                    result.latency_ms,
                    result.decision,
                    result.actuator_signal,
                ),
            )


def simulated_rejector(part_id: str, defect_pixels: int) -> str:
    """Return a safe simulated signal instead of touching physical hardware."""
    LOGGER.warning(
        "Simulated PLC reject: part=%s defect_pixels=%d", part_id, defect_pixels
    )
    return "SIMULATED_REJECT"


class EdgeInspector:
    """Run binary defect segmentation and industrial rejection decisions."""

    def __init__(
        self,
        model_path: Path,
        image_size: int = 256,
        threshold: float = 0.5,
        min_defect_pixels: int = 15,
        telemetry: Optional[SQLiteTelemetry] = None,
        actuator: Optional[Actuator] = None,
        providers: Optional[list[str]] = None,
    ) -> None:
        if image_size <= 0 or min_defect_pixels < 0:
            raise ValueError("image_size must be positive and min_defect_pixels non-negative")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        available = ort.get_available_providers()
        requested = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session_providers = [provider for provider in requested if provider in available]
        if not session_providers:
            raise RuntimeError(
                f"No requested ONNX Runtime provider is available. Available: {available}"
            )

        self.session = ort.InferenceSession(str(model_path), providers=session_providers)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if not inputs or not outputs:
            raise RuntimeError("ONNX model must expose at least one input and output")
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.image_size = image_size
        self.threshold = threshold
        self.min_defect_pixels = min_defect_pixels
        self.telemetry = telemetry
        self.actuator = actuator

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3:
            raise ValueError("frame must be a BGR HxWx3 numpy array")
        if frame_bgr.shape[2] != 3:
            raise ValueError("frame must have exactly three color channels")
        resized = cv2.resize(frame_bgr, (self.image_size, self.image_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        return np.transpose(tensor, (2, 0, 1))[None, ...]

    def _mask_from_output(self, output: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        logits = np.asarray(output)
        if logits.ndim == 4:
            logits = logits[0]
        if logits.ndim == 3:
            logits = logits[0]
        if logits.ndim != 2:
            raise RuntimeError(f"Expected a binary segmentation output, got shape {output.shape}")
        probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))
        mask = (probabilities >= self.threshold).astype(np.uint8) * 255
        return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _overlay(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        result = frame_bgr.copy()
        red = np.zeros_like(result)
        red[:, :] = (0, 0, 255)
        result[mask > 0] = cv2.addWeighted(result[mask > 0], 0.45, red[mask > 0], 0.55, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        return result

    def inspect(self, frame_bgr: np.ndarray, part_id: str) -> InspectionResult:
        if not part_id.strip():
            raise ValueError("part_id must not be empty")
        tensor = self._preprocess(frame_bgr)
        started = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        latency_ms = (time.perf_counter() - started) * 1000.0
        mask = self._mask_from_output(output, frame_bgr.shape[:2])
        defect_pixels = int(np.count_nonzero(mask))
        defect_ratio = defect_pixels / float(mask.size)
        rejected = defect_pixels >= self.min_defect_pixels
        decision = "REJECT" if rejected else "PASS"
        actuator_signal = (
            self.actuator(part_id, defect_pixels) if rejected and self.actuator else "NOT_TRIGGERED"
        )
        result = InspectionResult(
            part_id=part_id,
            decision=decision,
            defect_pixels=defect_pixels,
            defect_ratio=defect_ratio,
            latency_ms=latency_ms,
            fps=1000.0 / latency_ms if latency_ms else 0.0,
            mask=mask,
            overlay=self._overlay(frame_bgr, mask),
            actuator_signal=actuator_signal,
        )
        if self.telemetry:
            self.telemetry.record(result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ONNX industrial defect inspection")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--part-id", default="part-0001")
    parser.add_argument("--output", type=Path, default=Path("inspection_overlay.png"))
    parser.add_argument("--telemetry", type=Path, default=Path("inspection_telemetry.db"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")
    inspector = EdgeInspector(
        args.model,
        telemetry=SQLiteTelemetry(args.telemetry),
        actuator=simulated_rejector,
    )
    result = inspector.inspect(frame, args.part_id)
    if not cv2.imwrite(str(args.output), result.overlay):
        raise SystemExit(f"Could not write output image: {args.output}")
    print(
        f"{result.decision}: {result.defect_pixels}px, "
        f"{result.latency_ms:.2f}ms ({result.fps:.1f} FPS)"
    )


if __name__ == "__main__":
    main()
