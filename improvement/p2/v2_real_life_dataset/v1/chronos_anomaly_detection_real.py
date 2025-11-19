#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
chronos_anomaly_detection_real.py
Production-ready anomaly detection on real electricity data (GluonTS)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List,Dict

from chronos import ChronosPipeline
from gluonts.dataset.repository import get_dataset
import json

# CORE DETECTION (unchanged)
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
    ):
        self.num_samples = num_samples
        self.alpha_95 = (1 - 0.95) / 2
        self.alpha_99 = (1 - 0.99) / 2

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_name} on {device} …")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name, device_map=device, torch_dtype=torch.bfloat16
        )
        print("Model loaded")

    def detect_point_anomalies(
        self,
        time_series: np.ndarray,
        timestamps: List[datetime],
        context_length: int = 96,
    ) -> List[AnomalyResult]:
        results = []
        n = len(time_series)

        for i in range(context_length, n):
            ctx = time_series[i - context_length : i]
            actual = time_series[i]

            tensor = torch.from_numpy(ctx).unsqueeze(0)
            with torch.no_grad():
                samples = self.pipeline.predict(
                    tensor, prediction_length=1, num_samples=self.num_samples
                ).numpy().squeeze()

            pred_median = np.median(samples)
            lower_95 = np.quantile(samples, self.alpha_95)
            upper_95 = np.quantile(samples, 1 - self.alpha_95)
            lower_99 = np.quantile(samples, self.alpha_99)
            upper_99 = np.quantile(samples, 1 - self.alpha_95)

            mean = samples.mean()
            std = samples.std() + 1e-6
            z_score = (actual - mean) / std
            severity = abs(z_score)

            is_anomaly = actual < lower_95 or actual > upper_95
            anomaly_type = "high" if actual > upper_95 else ("low" if actual < lower_95 else "normal")

            confidence = 0.0
            if is_anomaly:
                dist = actual - upper_95 if actual > upper_95 else lower_95 - actual
                band = upper_95 - lower_95
                confidence = min(dist / (band + 1e-6), 1.0)

            results.append(AnomalyResult(
                timestamps[i], actual, pred_median, lower_95, upper_95, lower_99, upper_99,
                is_anomaly, severity, anomaly_type, confidence
            ))

            if (i - context_length) % 200 == 0:
                prog = (i - context_length) / (n - context_length) * 100
                print(f"Progress: {prog:.1f}% - anomalies: {sum(r.is_anomaly for r in results)}")

        return results

# LOAD REAL DATA WITH GLUONTS
def load_electricity_series(household_index: int = 0, max_points: int = 2500) -> pd.DataFrame:
    print("Loading real electricity data via GluonTS...")
    ds = get_dataset("electricity")
    entry = list(ds.train)[household_index]  # 370 households available (0-369)
    series = entry["target"][-max_points:]
    start = entry["start"]
    timestamps = pd.date_range(start.to_timestamp(), periods=len(series), freq=start.freq)
    df = pd.DataFrame({"timestamp": timestamps, "load": series})
    print(f"Loaded {len(df)} points for household_{household_index+1} – from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
    return df

# MONITOR
class ElectricityMonitor:
    def __init__(self, detector):
        self.detector = detector

    def monitor_household(self, data: pd.DataFrame, household_id: str) -> Dict:
        print(f"\nMonitoring {household_id} – {len(data)} hourly readings")
        results = self.detector.detect_point_anomalies(
            data["load"].values,
            data["timestamp"].tolist()
        )
        critical = [r for r in results if r.is_anomaly and r.confidence > 0.7]
        print(f"Anomalies: {len(critical)} (critical {sum(r.anomaly_severity > 3.5 for r in critical)})")
        return {"anomalies": len(critical), "results": results}

# REPORT
def to_json(results: List[AnomalyResult], path: str = "electricity_anomalies.json"):
    report = {
        "generated_at": datetime.now().isoformat(),
        "anomalies": [r.__dict__ for r in results if r.is_anomaly]
    }
    Path(path).write_text(json.dumps(report, indent=2, default=str))
    print(f"Report → {path}")

# MAIN
def main():
    detector = ChronosAnomalyDetector()
    df = load_electricity_series(household_index=0, max_points=2500)  # household_1 (index 0)
    monitor = ElectricityMonitor(detector)
    summary = monitor.monitor_household(df, "household_1")
    to_json(summary["results"])

if __name__ == "__main__":
    print("Chronos Anomaly Detection – Real Electricity (GluonTS)")
    print("=" * 70)
    main()