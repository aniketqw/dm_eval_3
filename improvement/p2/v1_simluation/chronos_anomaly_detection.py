#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
chronos_anomaly_detection.py - Production-Ready Anomaly Detection System

REAL-WORLD APPLICATIONS:
1. Manufacturing: Detect equipment failures before they happen
2. Finance: Flag fraudulent transactions in real-time
3. IoT: Monitor sensor data for device malfunctions
4. E-commerce: Detect inventory anomalies and demand spikes
5. Healthcare: Monitor patient vitals for critical changes
6. Network: Detect security breaches and DDoS attacks

METHODOLOGY:
- Generate 200+ forecast samples for each point
- Build probabilistic prediction bands (95%, 99%)
- Flag observations outside expected range
- Adaptive thresholds based on uncertainty
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from chronos import ChronosPipeline
from scipy import stats
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# 1️⃣ ANOMALY DETECTION CORE
# ==============================================================================

@dataclass
class AnomalyResult:
    """Structure for anomaly detection results"""
    timestamp: datetime
    actual_value: float
    predicted_median: float
    lower_95: float
    upper_95: float
    lower_99: float
    upper_99: float
    is_anomaly: bool
    anomaly_severity: float  # How many std devs from mean
    anomaly_type: str  # 'high', 'low', 'normal'
    confidence: float  # 0-1, how confident we are it's an anomaly


class ChronosAnomalyDetector:
    """
    Production-ready anomaly detection using Chronos
    
    Key Features:
    - Real-time streaming detection
    - Adaptive thresholds
    - Multiple severity levels
    - Contextual anomalies (not just point anomalies)
    - False positive reduction
    """
    
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-base",
        num_samples: int = 200,
        confidence_level: float = 0.95,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model_name: Chronos model to use
            num_samples: Number of forecast samples (more = better uncertainty)
            confidence_level: Threshold for anomaly (0.95 = 95% confidence)
            device: 'cuda' or 'cpu'
        """
        self.num_samples = num_samples
        self.confidence_level = confidence_level
        
        print(f"Loading {model_name} on {device}...")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        print("✓ Model loaded")
        
        # Calculate quantiles for confidence intervals
        self.alpha_95 = (1 - 0.95) / 2
        self.alpha_99 = (1 - 0.99) / 2
    
    def detect_point_anomalies(
        self,
        time_series: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        context_length: int = 64,
        stride: int = 1,
    ) -> List[AnomalyResult]:
        """
        Detect point anomalies in historical time series
        
        Args:
            time_series: Historical values
            timestamps: Optional timestamps for each value
            context_length: How much history to use for prediction
            stride: How many steps between predictions (1=every point)
        
        Returns:
            List of AnomalyResult objects
        """
        if timestamps is None:
            timestamps = [datetime.now() + timedelta(hours=i) for i in range(len(time_series))]
        
        results = []
        n = len(time_series)
        
        # Start detection after we have enough context
        for i in range(context_length, n, stride):
            # Get context window
            context = time_series[i - context_length:i]
            actual_value = time_series[i]
            
            # Generate forecast samples
            samples = self._generate_samples(context, prediction_length=1)
            
            # Calculate statistics
            result = self._analyze_point(
                actual_value=actual_value,
                samples=samples[:, 0],  # First prediction step
                timestamp=timestamps[i],
            )
            
            results.append(result)
            
            if (i - context_length) % 100 == 0:
                progress = (i - context_length) / (n - context_length) * 100
                print(f"Progress: {progress:.1f}% - Anomalies found: {sum(r.is_anomaly for r in results)}")
        
        return results
    
    def detect_streaming_anomaly(
        self,
        context: np.ndarray,
        new_value: float,
        timestamp: Optional[datetime] = None,
    ) -> AnomalyResult:
        """
        Real-time anomaly detection for streaming data
        
        Use this when you receive new data points one at a time
        
        Args:
            context: Recent historical values (last 64-512 points)
            new_value: The new incoming value to check
            timestamp: When this value occurred
        
        Returns:
            AnomalyResult indicating if it's anomalous
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Generate forecast samples
        samples = self._generate_samples(context, prediction_length=1)
        
        # Analyze this point
        result = self._analyze_point(
            actual_value=new_value,
            samples=samples[:, 0],
            timestamp=timestamp,
        )
        
        return result
    
    def detect_contextual_anomalies(
        self,
        time_series: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        window_size: int = 10,
        context_length: int = 64,
    ) -> List[Dict]:
        """
        Detect contextual/collective anomalies (sequences of unusual behavior)
        
        Example: Sales are normal individually but show unusual pattern together
        
        Args:
            time_series: Historical values
            timestamps: Optional timestamps
            window_size: Size of sequence to check
            context_length: History to use for prediction
        
        Returns:
            List of anomalous sequences
        """
        if timestamps is None:
            timestamps = [datetime.now() + timedelta(hours=i) for i in range(len(time_series))]
        
        anomalous_sequences = []
        n = len(time_series)
        
        for i in range(context_length, n - window_size + 1):
            # Get context and target window
            context = time_series[i - context_length:i]
            target_window = time_series[i:i + window_size]
            
            # Generate forecast for entire window
            samples = self._generate_samples(context, prediction_length=window_size)
            
            # Calculate how unusual this window is
            anomaly_scores = []
            for j in range(window_size):
                actual = target_window[j]
                predicted = samples[:, j]
                
                # Z-score: how many std devs from mean
                z_score = (actual - predicted.mean()) / (predicted.std() + 1e-6)
                anomaly_scores.append(abs(z_score))
            
            # If the average anomaly score is high, flag this sequence
            avg_score = np.mean(anomaly_scores)
            if avg_score > 2.5:  # 2.5 standard deviations
                anomalous_sequences.append({
                    'start_time': timestamps[i],
                    'end_time': timestamps[i + window_size - 1],
                    'start_index': i,
                    'end_index': i + window_size - 1,
                    'values': target_window.tolist(),
                    'anomaly_score': float(avg_score),
                    'severity': 'high' if avg_score > 3.5 else 'medium',
                })
        
        print(f"Found {len(anomalous_sequences)} contextual anomalies")
        return anomalous_sequences
    
    def _generate_samples(
        self,
        context: np.ndarray,
        prediction_length: int,
    ) -> np.ndarray:
        """Generate forecast samples"""
        tensor = torch.from_numpy(context).unsqueeze(0)
        
        with torch.no_grad():
            samples = self.pipeline.predict(
                tensor,
                prediction_length=prediction_length,
                num_samples=self.num_samples,
            )
        
        return samples.cpu().numpy().squeeze(0)
    
    def _analyze_point(
        self,
        actual_value: float,
        samples: np.ndarray,
        timestamp: datetime,
    ) -> AnomalyResult:
        """Analyze a single point for anomalies"""
        
        # Calculate prediction statistics
        pred_mean = samples.mean()
        pred_std = samples.std()
        pred_median = np.median(samples)
        
        # Calculate confidence intervals
        lower_95 = np.quantile(samples, self.alpha_95)
        upper_95 = np.quantile(samples, 1 - self.alpha_95)
        lower_99 = np.quantile(samples, self.alpha_99)
        upper_99 = np.quantile(samples, 1 - self.alpha_99)
        
        # Determine if anomaly
        is_anomaly_95 = actual_value < lower_95 or actual_value > upper_95
        is_anomaly_99 = actual_value < lower_99 or actual_value > upper_99
        
        # Calculate severity (z-score)
        z_score = (actual_value - pred_mean) / (pred_std + 1e-6)
        severity = abs(z_score)
        
        # Determine anomaly type
        if actual_value > upper_95:
            anomaly_type = 'high'
        elif actual_value < lower_95:
            anomaly_type = 'low'
        else:
            anomaly_type = 'normal'
        
        # Calculate confidence (how sure we are it's an anomaly)
        # Based on: how far outside the band + how narrow the band is
        if is_anomaly_95:
            if actual_value > upper_95:
                distance = actual_value - upper_95
            else:
                distance = lower_95 - actual_value
            
            band_width = upper_95 - lower_95
            confidence = min(distance / (band_width + 1e-6), 1.0)
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
            is_anomaly=is_anomaly_95,
            anomaly_severity=severity,
            anomaly_type=anomaly_type,
            confidence=confidence,
        )


# ==============================================================================
# 2️⃣ REAL-WORLD APPLICATION EXAMPLES
# ==============================================================================

class ManufacturingMonitor:
    """
    REAL-WORLD USE CASE 1: Manufacturing Equipment Monitoring
    
    Detect:
    - Temperature spikes indicating equipment failure
    - Vibration anomalies suggesting bearing wear
    - Pressure drops indicating leaks
    - Unusual patterns before catastrophic failures
    """
    
    def __init__(self, detector: ChronosAnomalyDetector):
        self.detector = detector
        self.alert_history = []
    
    def monitor_equipment(
        self,
        sensor_data: pd.DataFrame,
        equipment_id: str,
        sensor_type: str,
    ) -> Dict:
        """
        Monitor equipment sensor data
        
        Args:
            sensor_data: DataFrame with 'timestamp' and 'value' columns
            equipment_id: Unique equipment identifier
            sensor_type: 'temperature', 'vibration', 'pressure', etc.
        """
        print(f"\n🏭 Monitoring {equipment_id} - {sensor_type}")
        print("=" * 70)
        
        # Detect anomalies
        results = self.detector.detect_point_anomalies(
            time_series=sensor_data['value'].values,
            timestamps=pd.to_datetime(sensor_data['timestamp']).tolist(),
            context_length=64,
        )
        
        # Find critical anomalies
        critical_anomalies = [
            r for r in results 
            if r.is_anomaly and r.confidence > 0.7
        ]
        
        # Generate alerts
        alerts = []
        for anomaly in critical_anomalies:
            severity = self._assess_severity(anomaly, sensor_type)
            
            alert = {
                'equipment_id': equipment_id,
                'sensor_type': sensor_type,
                'timestamp': anomaly.timestamp.isoformat(),
                'severity': severity,
                'actual_value': anomaly.actual_value,
                'expected_range': f"[{anomaly.lower_95:.2f}, {anomaly.upper_95:.2f}]",
                'deviation': f"{anomaly.anomaly_severity:.2f}σ",
                'recommendation': self._get_recommendation(severity, sensor_type),
            }
            alerts.append(alert)
            self.alert_history.append(alert)
        
        # Summary
        summary = {
            'equipment_id': equipment_id,
            'sensor_type': sensor_type,
            'total_readings': len(results),
            'anomalies_detected': len(critical_anomalies),
            'critical_alerts': len([a for a in alerts if a['severity'] == 'critical']),
            'warnings': len([a for a in alerts if a['severity'] == 'warning']),
            'alerts': alerts,
            'health_score': self._calculate_health_score(results),
        }
        
        # Print summary
        print(f"Total readings: {summary['total_readings']}")
        print(f"Anomalies detected: {summary['anomalies_detected']}")
        print(f"Critical alerts: {summary['critical_alerts']}")
        print(f"Health score: {summary['health_score']:.1f}%")
        
        if alerts:
            print("\n⚠️  ALERTS:")
            for alert in alerts[:5]:  # Show first 5
                print(f"  [{alert['severity'].upper()}] {alert['timestamp']}")
                print(f"    Value: {alert['actual_value']:.2f} (expected: {alert['expected_range']})")
                print(f"    Action: {alert['recommendation']}")
        
        return summary
    
    def _assess_severity(self, anomaly: AnomalyResult, sensor_type: str) -> str:
        """Assess severity based on sensor type and deviation"""
        
        # Different sensors have different thresholds
        thresholds = {
            'temperature': {'warning': 2.5, 'critical': 3.5},
            'vibration': {'warning': 2.0, 'critical': 3.0},
            'pressure': {'warning': 2.5, 'critical': 3.5},
        }
        
        thresh = thresholds.get(sensor_type, {'warning': 2.5, 'critical': 3.5})
        
        if anomaly.anomaly_severity > thresh['critical']:
            return 'critical'
        elif anomaly.anomaly_severity > thresh['warning']:
            return 'warning'
        else:
            return 'info'
    
    def _get_recommendation(self, severity: str, sensor_type: str) -> str:
        """Get action recommendation"""
        
        actions = {
            'critical': {
                'temperature': 'IMMEDIATE: Shut down equipment and inspect cooling system',
                'vibration': 'IMMEDIATE: Stop operation and check bearings/alignment',
                'pressure': 'IMMEDIATE: Check for leaks and inspect seals',
            },
            'warning': {
                'temperature': 'Schedule maintenance check within 24 hours',
                'vibration': 'Monitor closely and schedule inspection',
                'pressure': 'Inspect system during next maintenance window',
            },
            'info': {
                'temperature': 'Continue monitoring',
                'vibration': 'Continue monitoring',
                'pressure': 'Continue monitoring',
            }
        }
        
        return actions.get(severity, {}).get(sensor_type, 'Investigate further')
    
    def _calculate_health_score(self, results: List[AnomalyResult]) -> float:
        """Calculate equipment health score (0-100)"""
        if not results:
            return 100.0
        
        anomaly_rate = sum(r.is_anomaly for r in results) / len(results)
        avg_severity = np.mean([r.anomaly_severity for r in results if r.is_anomaly] or [0])
        
        # Health score decreases with anomaly rate and severity
        health = 100 * (1 - anomaly_rate) * (1 - min(avg_severity / 5.0, 1.0))
        return max(health, 0.0)


class FraudDetectionSystem:
    """
    REAL-WORLD USE CASE 2: Financial Fraud Detection
    
    Detect:
    - Unusual transaction amounts
    - Abnormal transaction patterns
    - Account takeover indicators
    - Money laundering patterns
    """
    
    def __init__(self, detector: ChronosAnomalyDetector):
        self.detector = detector
    
    def analyze_transactions(
        self,
        transactions: pd.DataFrame,
        account_id: str,
    ) -> Dict:
        """
        Analyze transaction history for fraud
        
        Args:
            transactions: DataFrame with 'timestamp', 'amount' columns
            account_id: Account to analyze
        """
        print(f"\n💳 Analyzing transactions for account: {account_id}")
        print("=" * 70)
        
        # Sort by timestamp
        transactions = transactions.sort_values('timestamp')
        
        # Detect point anomalies
        results = self.detector.detect_point_anomalies(
            time_series=transactions['amount'].values,
            timestamps=pd.to_datetime(transactions['timestamp']).tolist(),
            context_length=30,  # Last 30 transactions
            stride=1,
        )
        
        # Detect contextual anomalies (unusual patterns)
        contextual = self.detector.detect_contextual_anomalies(
            time_series=transactions['amount'].values,
            timestamps=pd.to_datetime(transactions['timestamp']).tolist(),
            window_size=5,  # Look at sequences of 5 transactions
            context_length=30,
        )
        
        # Flag high-risk transactions
        high_risk = [
            {
                'transaction_id': i,
                'timestamp': r.timestamp.isoformat(),
                'amount': r.actual_value,
                'expected_range': f"${r.lower_95:.2f} - ${r.upper_95:.2f}",
                'risk_score': self._calculate_risk_score(r),
                'reason': self._get_fraud_reason(r),
            }
            for i, r in enumerate(results)
            if r.is_anomaly and r.confidence > 0.6
        ]
        
        summary = {
            'account_id': account_id,
            'total_transactions': len(transactions),
            'suspicious_transactions': len(high_risk),
            'contextual_anomalies': len(contextual),
            'overall_risk': self._calculate_overall_risk(high_risk, contextual),
            'flagged_transactions': high_risk[:10],  # Top 10
        }
        
        print(f"Total transactions: {summary['total_transactions']}")
        print(f"Suspicious: {summary['suspicious_transactions']}")
        print(f"Overall risk: {summary['overall_risk']}")
        
        if high_risk:
            print("\n⚠️  HIGH-RISK TRANSACTIONS:")
            for txn in high_risk[:5]:
                print(f"  Transaction #{txn['transaction_id']}: ${txn['amount']:.2f}")
                print(f"    Risk Score: {txn['risk_score']:.2f}")
                print(f"    Reason: {txn['reason']}")
        
        return summary
    
    def _calculate_risk_score(self, anomaly: AnomalyResult) -> float:
        """Calculate fraud risk score (0-100)"""
        # Combine severity and confidence
        risk = (anomaly.anomaly_severity / 5.0) * anomaly.confidence * 100
        return min(risk, 100.0)
    
    def _get_fraud_reason(self, anomaly: AnomalyResult) -> str:
        """Explain why transaction is suspicious"""
        if anomaly.anomaly_type == 'high':
            if anomaly.actual_value > anomaly.upper_99:
                return "Transaction amount extremely high for this account"
            else:
                return "Transaction amount unusually high"
        else:
            return "Unusual transaction pattern detected"
    
    def _calculate_overall_risk(self, high_risk: List, contextual: List) -> str:
        """Calculate overall account risk level"""
        score = len(high_risk) * 10 + len(contextual) * 20
        
        if score > 80:
            return "CRITICAL - Immediate review required"
        elif score > 50:
            return "HIGH - Review within 24 hours"
        elif score > 20:
            return "MEDIUM - Monitor closely"
        else:
            return "LOW - Normal activity"


class IoTDeviceMonitor:
    """
    REAL-WORLD USE CASE 3: IoT Device Fleet Management
    
    Monitor:
    - Smart home devices (temperature, energy usage)
    - Industrial IoT sensors
    - Vehicle telematics
    - Environmental sensors
    """
    
    def __init__(self, detector: ChronosAnomalyDetector):
        self.detector = detector
    
    def monitor_device_fleet(
        self,
        device_data: Dict[str, pd.DataFrame],
        metric: str = 'value',
    ) -> Dict:
        """
        Monitor multiple IoT devices
        
        Args:
            device_data: Dict of {device_id: DataFrame} with sensor readings
            metric: Which metric to monitor
        """
        print(f"\n📡 Monitoring {len(device_data)} IoT devices - metric: {metric}")
        print("=" * 70)
        
        fleet_summary = {
            'total_devices': len(device_data),
            'devices_with_anomalies': 0,
            'total_anomalies': 0,
            'device_reports': [],
        }
        
        for device_id, data in device_data.items():
            # Detect anomalies for this device
            results = self.detector.detect_streaming_anomaly(
                context=data[metric].values[:-1],  # All but last
                new_value=data[metric].values[-1],  # Latest reading
            )
            
            if results.is_anomaly:
                fleet_summary['devices_with_anomalies'] += 1
                fleet_summary['total_anomalies'] += 1
                
                report = {
                    'device_id': device_id,
                    'timestamp': results.timestamp.isoformat(),
                    'status': 'ANOMALY DETECTED',
                    'value': results.actual_value,
                    'expected': f"[{results.lower_95:.2f}, {results.upper_95:.2f}]",
                    'action_required': self._get_device_action(results),
                }
                fleet_summary['device_reports'].append(report)
        
        print(f"Devices monitored: {fleet_summary['total_devices']}")
        print(f"Devices with anomalies: {fleet_summary['devices_with_anomalies']}")
        
        if fleet_summary['device_reports']:
            print("\n⚠️  DEVICE ALERTS:")
            for report in fleet_summary['device_reports'][:5]:
                print(f"  Device {report['device_id']}: {report['status']}")
                print(f"    Value: {report['value']:.2f} (expected: {report['expected']})")
                print(f"    Action: {report['action_required']}")
        
        return fleet_summary
    
    def _get_device_action(self, anomaly: AnomalyResult) -> str:
        """Determine action for device anomaly"""
        if anomaly.confidence > 0.8:
            return "Send technician for immediate inspection"
        elif anomaly.confidence > 0.5:
            return "Schedule maintenance check"
        else:
            return "Continue monitoring"


# ==============================================================================
# 3️⃣ BATCH PROCESSING & REPORTING
# ==============================================================================

class AnomalyReportGenerator:
    """Generate comprehensive anomaly reports"""
    
    @staticmethod
    def generate_report(
        results: List[AnomalyResult],
        output_path: str = "anomaly_report.json",
    ):
        """Generate JSON report"""
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_points': len(results),
                'anomalies_detected': sum(r.is_anomaly for r in results),
                'anomaly_rate': sum(r.is_anomaly for r in results) / len(results) if results else 0,
                'high_severity': sum(r.anomaly_severity > 3 for r in results),
                'medium_severity': sum(2 < r.anomaly_severity <= 3 for r in results),
            },
            'anomalies': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'actual_value': r.actual_value,
                    'predicted_median': r.predicted_median,
                    'confidence_interval_95': [r.lower_95, r.upper_95],
                    'confidence_interval_99': [r.lower_99, r.upper_99],
                    'severity': r.anomaly_severity,
                    'type': r.anomaly_type,
                    'confidence': r.confidence,
                }
                for r in results if r.is_anomaly
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved to: {output_path}")
        return report
    
    @staticmethod
    def generate_csv_export(
        results: List[AnomalyResult],
        output_path: str = "anomalies.csv",
    ):
        """Export anomalies to CSV"""
        
        df = pd.DataFrame([
            {
                'timestamp': r.timestamp,
                'actual_value': r.actual_value,
                'predicted_median': r.predicted_median,
                'lower_95': r.lower_95,
                'upper_95': r.upper_95,
                'is_anomaly': r.is_anomaly,
                'severity': r.anomaly_severity,
                'type': r.anomaly_type,
                'confidence': r.confidence,
            }
            for r in results
        ])
        
        df.to_csv(output_path, index=False)
        print(f"📁 CSV exported to: {output_path}")


# ==============================================================================
# 4️⃣ EXAMPLE USAGE
# ==============================================================================

def example_manufacturing():
    """Example: Manufacturing equipment monitoring"""
    
    # Initialize detector
    detector = ChronosAnomalyDetector(
        model_name="amazon/chronos-t5-base",
        num_samples=200,
        confidence_level=0.95,
    )
    
    # Load sensor data (simulated)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    
    # Normal operation with some anomalies
    np.random.seed(42)
    normal_temp = 75 + 5 * np.sin(np.arange(1000) / 24) + np.random.normal(0, 1, 1000)
    
    # Inject anomalies
    normal_temp[500] = 95  # Overheat
    normal_temp[750] = 55  # Cooling issue
    
    sensor_data = pd.DataFrame({
        'timestamp': dates,
        'value': normal_temp,
    })
    
    # Monitor equipment
    monitor = ManufacturingMonitor(detector)
    summary = monitor.monitor_equipment(
        sensor_data=sensor_data,
        equipment_id='MACHINE-001',
        sensor_type='temperature',
    )
    
    return summary


def example_fraud_detection():
    """Example: Credit card fraud detection"""
    
    detector = ChronosAnomalyDetector(
        model_name="amazon/chronos-t5-small",  # Faster for real-time
        num_samples=200,
    )
    
    # Simulate transactions
    dates = pd.date_range('2024-01-01', periods=200, freq='1D')
    amounts = 50 + 30 * np.random.random(200)
    
    # Inject fraudulent transactions
    amounts[100] = 5000  # Large unusual purchase
    amounts[150] = 3000  # Another suspicious transaction
    
    transactions = pd.DataFrame({
        'timestamp': dates,
        'amount': amounts,
    })
    
    # Analyze
    fraud_system = FraudDetectionSystem(detector)
    summary = fraud_system.analyze_transactions(
        transactions=transactions,
        account_id='ACCT-12345',
    )
    
    return summary


if __name__ == "__main__":
    print("🚀 Chronos Anomaly Detection - Real-World Examples")
    print("=" * 70)
    
    # Run examples
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Manufacturing Equipment Monitoring")
    print("=" * 70)
    manufacturing_results = example_manufacturing()
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Financial Fraud Detection")
    print("=" * 70)
    fraud_results = example_fraud_detection()
    
    print("\n✅ Examples completed!")