#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
chronos_anomaly_detection_real.py
=================================

Production-ready anomaly detection **on a real HF dataset**.

Dataset: ElectricityLoadDiagrams20112014 (hourly household loads)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict

from chronos import ChronosPipeline
from datasets import load_dataset
import json


# ----------------------------------------------------------------------
# 1. CORE DETECTION (unchanged – only tiny tweaks for HF data)
# ----------------------------------------------------------------------
@dataclass
class AnomalyResult:
    timestamp: datetime
    actual_value: float
    predicted_median: float
    lower_95: float
    upper_95: float
    lower_99: float
    upper_99: float
    is_anomaly: bool
    anomaly_severity: float
    anomaly_type: str
    confidence: float


class ChronosAnomalyDetector:
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-base",
        num_samples: int = 200,
        confidence_level: float = 0.95,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.num_samples = num_samples
        self.confidence_level = confidence_level
        self.alpha_95 = (1 - 0.95) / 2
        self.alpha_99 = (1 - 0.99) / 2

        print(f"Loading {model_name} on {device} …")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name, device_map=device, torch_dtype=torch.bfloat16
        )
        print("Model loaded")

    # ------------------------------------------------------------------
    # point / streaming / contextual (identical to original)
    # ------------------------------------------------------------------
    def detect_point_anomalies(
        self,
        time_series: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        context_length: int = 64,
        stride: int = 1,
    ) -> List[AnomalyResult]:
        if timestamps is None:
            timestamps = [datetime.now() for _ in range(len(time_series))]

        results = []
        n = len(time_series)

        for i in range(context_length, n, stride):
            ctx = time_series[i - context_length : i]
            actual = time_series[i]

            samples = self._generate_samples(ctx, prediction_length=1)

            res = self._analyze_point(actual, samples[:, 0], timestamps[i])
            results.append(res)

            if (i - context_length) % 200 == 0:
                prog = (i - context_length) / (n - context_length) * 100
                print(f"Progress: {prog:5.1f}% – anomalies: {sum(r.is_anomaly for r in results)}")

        return results

    def detect_streaming_anomaly(
        self,
        context: np.ndarray,
        new_value: float,
        timestamp: Optional[datetime] = None,
    ) -> AnomalyResult:
        if timestamp is None:
            timestamp = datetime.now()
        samples = self._generate_samples(context, prediction_length=1)
        return self._analyze_point(new_value, samples[:, 0], timestamp)

    def _generate_samples(self, context: np.ndarray, prediction_length: int) -> np.ndarray:
        tensor = torch.from_numpy(context).unsqueeze(0).to(self.pipeline.device)
        with torch.no_grad():
            samples = self.pipeline.predict(
                tensor,
                prediction_length=prediction_length,
                num_samples=self.num_samples,
            )
        return samples.cpu().numpy().squeeze(0)

    def _analyze_point(
        self, actual_value: float, samples: np.ndarray, timestamp: datetime
    ) -> AnomalyResult:
        pred_median = np.median(samples)
        lower_95 = np.quantile(samples, self.alpha_95)
        upper_95 = np.quantile(samples, 1 - self.alpha_95)
        lower_99 = np.quantile(samples, self.alpha_99)
        upper_99 = np.quantile(samples, 1 - self.alpha_99)

        pred_mean = samples.mean()
        pred_std = samples.std() + 1e-6
        z_score = (actual_value - pred_mean) / pred_std
        severity = abs(z_score)

        is_anomaly = actual_value < lower_95 or actual_value > upper_95
        anomaly_type = "high" if actual_value > upper_95 else ("low" if actual_value < lower_95 else "normal")

        if is_anomaly:
            dist = actual_value - upper_95 if actual_value > upper_95 else lower_95 - actual_value
            band = upper_95 - lower_95
            confidence = min(dist / (band + 1e-6), 1.0)
        else:
            confidence = 0.0

        return AnomalyResult(
            timestamp=timestamp,
            actual_value=actual_value,
            predicted_median=pred_median,
            lower_95=lower_95,
            upper_95=upper_95,
            lower_99=lower_99,
            upper_99=upper_99,
            is_anomaly=is_anomaly,
            anomaly_severity=severity,
            anomaly_type=anomaly_type,
            confidence=confidence,
        )


# ----------------------------------------------------------------------
# 2. LOAD REAL DATA FROM HUGGING FACE
# ----------------------------------------------------------------------
def load_electricity_series(
    household_id: str = "MAC000002",  # Valid ID from the dataset
    max_points: int = 2500,            # Keep it snappy
) -> pd.DataFrame:
    """
    Loads real hourly electricity loads from HF.
    Returns: timestamp (datetime), load (float, kW)
    """
    print(f"Downloading electricity_load_diagrams (household {household_id}) …")
    ds = load_dataset("electricity_load_diagrams", "uci", split="train")  # ← FIXED: config="uci"

    # Dataset structure: {'item_id': str, 'start_time': datetime, 'target': array of values}
    df_list = []
    for item in ds:
        if item["item_id"] == household_id:  # Filter to one household
            start_time = item["start_time"]
            values = item["target"][:max_points]  # Limit to recent points
            timestamps = pd.date_range(start=start_time, periods=len(values), freq="H")  # Hourly freq
            df_temp = pd.DataFrame({"timestamp": timestamps, "load": values})
            df_list.append(df_temp)
            break  # Just one series

    if not df_list:
        raise ValueError(f"Household {household_id} not found. Try 'MAC000001' or check IDs.")

    series = pd.concat(df_list)
    series = series.sort_values("timestamp").reset_index(drop=True)

    print(f"Loaded {len(series)} points – from {series['timestamp'].iloc[0]} to {series['timestamp'].iloc[-1]}")
    return series[["timestamp", "load"]]


# ----------------------------------------------------------------------
# 3. RE-USE THE ORIGINAL USE-CASE CLASSES (only tiny adaptions)
# ----------------------------------------------------------------------
class ElectricityMonitor:
    """Real-world electricity-load anomaly monitor (re-uses ManufacturingMonitor logic)"""

    def __init__(self, detector: ChronosAnomalyDetector):
        self.detector = detector
        self.alert_history = []

    def monitor_household(self, data: pd.DataFrame, household_id: str) -> Dict:
        print(f"\nMonitoring household {household_id} – {len(data)} hourly readings")
        print("=" * 70)

        results = self.detector.detect_point_anomalies(
            time_series=data["load"].values,
            timestamps=data["timestamp"].tolist(),
            context_length=96,   # 4-day context (96 h)
            stride=1,
        )

        # Critical = confidence > 0.7 + severity > 3σ
        critical = [r for r in results if r.is_anomaly and r.confidence > 0.7]

        alerts = []
        for r in critical:
            severity = "critical" if r.anomaly_severity > 3.5 else "warning"
            alert = {
                "household": household_id,
                "timestamp": r.timestamp.isoformat(),
                "severity": severity,
                "load_kW": r.actual_value,
                "expected": f"[{r.lower_95:.2f}, {r.upper_95:.2f}]",
                "deviation_σ": f"{r.anomaly_severity:.2f}",
                "recommendation": self._recommend(severity, r.anomaly_type),
            }
            alerts.append(alert)
            self.alert_history.append(alert)

        summary = {
            "household": household_id,
            "total_points": len(results),
            "anomalies": len(critical),
            "critical": len([a for a in alerts if a["severity"] == "critical"]),
            "warnings": len([a for a in alerts if a["severity"] == "warning"]),
            "alerts": alerts,
            "health_score": self._health_score(results),
        }

        print(f"Total points: {summary['total_points']}")
        print(f"Anomalies: {summary['anomalies']} (critical {summary['critical']})")
        print(f"Health score: {summary['health_score']:.1f}%")

        if alerts:
            print("\nALERTS (first 5):")
            for a in alerts[:5]:
                print(f"  [{a['severity'].upper()}] {a['timestamp']} → {a['load_kW']:.2f} kW")
                print(f"    Expected: {a['expected']}   Action: {a['recommendation']}")

        return summary

    def _recommend(self, severity: str, typ: str) -> str:
        if severity == "critical":
            return "IMMEDIATE: Dispatch technician – possible appliance fault or meter error"
        return "Schedule inspection within 48 h"

    def _health_score(self, results: List[AnomalyResult]) -> float:
        if not results:
            return 100.0
        rate = sum(r.is_anomaly for r in results) / len(results)
        avg_sev = np.mean([r.anomaly_severity for r in results if r.is_anomaly] or [0])
        return max(100 * (1 - rate) * (1 - min(avg_sev / 5.0, 1.0)), 0.0)


# ----------------------------------------------------------------------
# 4. REPORTING (unchanged)
# ----------------------------------------------------------------------
class AnomalyReportGenerator:
    @staticmethod
    def to_json(results: List[AnomalyResult], path: str = "electricity_anomalies.json"):
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total": len(results),
                "anomalies": sum(r.is_anomaly for r in results),
                "rate": sum(r.is_anomaly for r in results) / len(results) if results else 0,
            },
            "anomalies": [
                {
                    "ts": r.timestamp.isoformat(),
                    "actual": r.actual_value,
                    "median": r.predicted_median,
                    "ci95": [r.lower_95, r.upper_95],
                    "ci99": [r.lower_99, r.upper_99],
                    "severity": r.anomaly_severity,
                    "type": r.anomaly_type,
                    "conf": r.confidence,
                }
                for r in results
                if r.is_anomaly
            ],
        }
        Path(path).write_text(json.dumps(report, indent=2))
        print(f"Report → {path}")


# ----------------------------------------------------------------------
# 5. MAIN DEMO
# ----------------------------------------------------------------------
def main():
    detector = ChronosAnomalyDetector(
        model_name="amazon/chronos-t5-base",  # swap to -small for CPU
        num_samples=200,
    )

    # 1. Load a real household
    df = load_electricity_series(household_id="MAC000002", max_points=2500)

    # 2. Run monitoring
    monitor = ElectricityMonitor(detector)
    summary = monitor.monitor_household(df, household_id="MAC000002")

    # 3. Persist a JSON report
    results = detector.detect_point_anomalies(
        time_series=df["load"].values,
        timestamps=df["timestamp"].tolist(),
        context_length=96,
    )
    AnomalyReportGenerator.to_json(results)

    # 4. (optional) quick plot
    try:
        import matplotlib.pyplot as plt

        anomalies = [r for r in results if r.is_anomaly]
        ts = df["timestamp"]
        plt.figure(figsize=(12, 4))
        plt.plot(ts, df["load"], label="load (kW)", color="#1f77b4")
        if anomalies:
            a_ts = [a.timestamp for a in anomalies]
            a_val = [a.actual_value for a in anomalies]
            plt.scatter(a_ts, a_val, color="red", s=40, label="anomaly")
        plt.legend()
        plt.title("Electricity load + Chronos-detected anomalies")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass  # matplotlib not required


if __name__ == "__main__":
    print("Chronos Anomaly Detection – Real Electricity Data")
    print("=" * 70)
    main()