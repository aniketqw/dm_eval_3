#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
chronos_anomaly_detection_enhanced.py
Enhanced anomaly detection with comprehensive visualizations
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Model imports
from chronos import ChronosPipeline
from gluonts.dataset.repository import get_dataset

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class AnomalyResult:
    """Data class for anomaly detection results"""
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
    """Enhanced Chronos-based anomaly detector with visualization capabilities"""
    
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-base",
        num_samples: int = 200,
    ):
        self.num_samples = num_samples
        self.alpha_95 = (1 - 0.95) / 2
        self.alpha_99 = (1 - 0.99) / 2
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Loading {model_name} on {device}...")
        
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name, device_map=device, torch_dtype=torch.bfloat16
        )
        print("✅ Model loaded successfully")
    
    def detect_point_anomalies(
        self,
        time_series: np.ndarray,
        timestamps: List[datetime],
        context_length: int = 96,
    ) -> List[AnomalyResult]:
        """Detect anomalies in time series data"""
        results = []
        n = len(time_series)
        
        print(f"🔍 Analyzing {n} data points with context length {context_length}...")
        
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
            upper_99 = np.quantile(samples, 1 - self.alpha_99)
            
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
                timestamps[i], actual, pred_median, lower_95, upper_95, 
                lower_99, upper_99, is_anomaly, severity, anomaly_type, confidence
            ))
            
            if (i - context_length) % 200 == 0:
                prog = (i - context_length) / (n - context_length) * 100
                anomaly_count = sum(r.is_anomaly for r in results)
                print(f"  Progress: {prog:.1f}% - Anomalies detected: {anomaly_count}")
        
        return results


class EnhancedVisualizer:
    """Enhanced visualization class for anomaly detection results"""
    
    def __init__(self, results: List[AnomalyResult], series_name: str = "Time Series"):
        self.results = results
        self.series_name = series_name
        self.df = self._results_to_dataframe()
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for easier manipulation"""
        data = []
        for r in self.results:
            data.append({
                'timestamp': r.timestamp,
                'actual': r.actual_value,
                'predicted': r.predicted_median,
                'lower_95': r.lower_95,
                'upper_95': r.upper_95,
                'lower_99': r.lower_99,
                'upper_99': r.upper_99,
                'is_anomaly': r.is_anomaly,
                'severity': r.anomaly_severity,
                'type': r.anomaly_type,
                'confidence': r.confidence
            })
        return pd.DataFrame(data)
    
    def plot_time_series_with_anomalies(self, save_path: Optional[str] = None, 
                                       show_confidence_bands: bool = True):
        """Create an interactive time series plot with anomalies highlighted"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), height_ratios=[3, 1, 1])
        
        # Main time series plot
        ax1.plot(self.df['timestamp'], self.df['actual'], 
                 label='Actual', color='#2E86AB', linewidth=1.5, alpha=0.8)
        ax1.plot(self.df['timestamp'], self.df['predicted'], 
                 label='Predicted', color='#A23B72', linewidth=1, alpha=0.7)
        
        if show_confidence_bands:
            ax1.fill_between(self.df['timestamp'], self.df['lower_95'], self.df['upper_95'],
                           alpha=0.2, color='#A23B72', label='95% CI')
            ax1.fill_between(self.df['timestamp'], self.df['lower_99'], self.df['upper_99'],
                           alpha=0.1, color='#A23B72', label='99% CI')
        
        # Highlight anomalies
        anomalies = self.df[self.df['is_anomaly']]
        for _, row in anomalies.iterrows():
            color = '#FF6B6B' if row['type'] == 'high' else '#4ECDC4'
            ax1.scatter(row['timestamp'], row['actual'], 
                       c=color, s=50 + row['severity'] * 20, 
                       alpha=0.6 + row['confidence'] * 0.4,
                       edgecolors='black', linewidth=0.5, zorder=5)
        
        ax1.set_title(f'{self.series_name} - Anomaly Detection Results', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Value', fontsize=12)
        ax1.legend(loc='upper left', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # Severity plot
        ax2.bar(anomalies['timestamp'], anomalies['severity'], 
                color=anomalies['type'].map({'high': '#FF6B6B', 'low': '#4ECDC4'}),
                width=0.01, alpha=0.7)
        ax2.set_ylabel('Severity', fontsize=11)
        ax2.set_title('Anomaly Severity (|Z-score|)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Confidence plot
        ax3.bar(anomalies['timestamp'], anomalies['confidence'],
                color='#95E1D3', width=0.01, alpha=0.7)
        ax3.set_ylabel('Confidence', fontsize=11)
        ax3.set_xlabel('Timestamp', fontsize=12)
        ax3.set_title('Detection Confidence', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_plot(self, save_html: Optional[str] = None):
        """Create an interactive Plotly visualization"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{self.series_name} with Anomaly Detection',
                'Anomaly Severity',
                'Detection Confidence'
            ),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Main time series
        fig.add_trace(
            go.Scatter(x=self.df['timestamp'], y=self.df['actual'],
                      name='Actual', line=dict(color='#2E86AB', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=self.df['timestamp'], y=self.df['predicted'],
                      name='Predicted', line=dict(color='#A23B72', width=1.5)),
            row=1, col=1
        )
        
        # Confidence bands
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'].tolist() + self.df['timestamp'].tolist()[::-1],
                y=self.df['upper_95'].tolist() + self.df['lower_95'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(162, 59, 114, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% CI',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Anomalies
        anomalies = self.df[self.df['is_anomaly']]
        high_anomalies = anomalies[anomalies['type'] == 'high']
        low_anomalies = anomalies[anomalies['type'] == 'low']
        
        if len(high_anomalies) > 0:
            fig.add_trace(
                go.Scatter(
                    x=high_anomalies['timestamp'],
                    y=high_anomalies['actual'],
                    mode='markers',
                    name='High Anomaly',
                    marker=dict(
                        color='#FF6B6B',
                        size=8 + high_anomalies['severity'] * 2,
                        line=dict(color='black', width=1)
                    ),
                    text=[f"Value: {v:.2f}<br>Severity: {s:.2f}<br>Confidence: {c:.2%}" 
                          for v, s, c in zip(high_anomalies['actual'], 
                                            high_anomalies['severity'],
                                            high_anomalies['confidence'])],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if len(low_anomalies) > 0:
            fig.add_trace(
                go.Scatter(
                    x=low_anomalies['timestamp'],
                    y=low_anomalies['actual'],
                    mode='markers',
                    name='Low Anomaly',
                    marker=dict(
                        color='#4ECDC4',
                        size=8 + low_anomalies['severity'] * 2,
                        line=dict(color='black', width=1)
                    ),
                    text=[f"Value: {v:.2f}<br>Severity: {s:.2f}<br>Confidence: {c:.2%}"
                          for v, s, c in zip(low_anomalies['actual'],
                                            low_anomalies['severity'],
                                            low_anomalies['confidence'])],
                    hovertemplate='%{text}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Severity bars
        fig.add_trace(
            go.Bar(
                x=anomalies['timestamp'],
                y=anomalies['severity'],
                name='Severity',
                marker=dict(color=anomalies['severity'], colorscale='Reds'),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Confidence bars
        fig.add_trace(
            go.Bar(
                x=anomalies['timestamp'],
                y=anomalies['confidence'],
                name='Confidence',
                marker=dict(color='#95E1D3'),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Interactive Anomaly Detection Dashboard - {self.series_name}",
            title_font_size=16,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Load (kW)", row=1, col=1)
        fig.update_yaxes(title_text="Severity", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=3, col=1)
        
        if save_html:
            fig.write_html(save_html)
            print(f"📊 Interactive plot saved to {save_html}")
        
        fig.show()
    
    def plot_anomaly_heatmap(self, save_path: Optional[str] = None):
        """Create a heatmap showing anomaly patterns over time"""
        # Resample to hourly for better visualization
        df_hourly = self.df.set_index('timestamp').resample('H').agg({
            'is_anomaly': 'sum',
            'severity': 'mean',
            'confidence': 'mean'
        }).fillna(0)
        
        # Create day-hour matrix
        df_hourly['hour'] = df_hourly.index.hour
        df_hourly['date'] = df_hourly.index.date
        
        pivot_severity = df_hourly.pivot_table(
            values='severity', 
            index='hour', 
            columns='date', 
            aggfunc='mean'
        ).fillna(0)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(20, 8))
        
        sns.heatmap(
            pivot_severity, 
            cmap='YlOrRd',
            cbar_kws={'label': 'Average Severity'},
            ax=ax,
            vmin=0,
            vmax=pivot_severity.max().max()
        )
        
        ax.set_title('Anomaly Severity Heatmap (Hour of Day vs Date)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Hour of Day', fontsize=12)
        
        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Heatmap saved to {save_path}")
        
        plt.show()
    
    def create_summary_dashboard(self, save_path: Optional[str] = None):
        """Create a comprehensive summary dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Time series with anomalies (top row, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.df['timestamp'], self.df['actual'], 'b-', alpha=0.6, linewidth=1)
        anomalies = self.df[self.df['is_anomaly']]
        ax1.scatter(anomalies['timestamp'], anomalies['actual'], 
                   c='red', s=20, alpha=0.8, label='Anomalies')
        ax1.set_title('Time Series with Detected Anomalies', fontweight='bold')
        ax1.set_ylabel('Load (kW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Statistics summary (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        total_points = len(self.df)
        anomaly_count = self.df['is_anomaly'].sum()
        anomaly_rate = anomaly_count / total_points * 100
        avg_severity = self.df[self.df['is_anomaly']]['severity'].mean()
        avg_confidence = self.df[self.df['is_anomaly']]['confidence'].mean()
        
        stats_text = f"""
        📊 ANOMALY STATISTICS
        ─────────────────────
        Total Points: {total_points:,}
        Anomalies Found: {anomaly_count}
        Anomaly Rate: {anomaly_rate:.2f}%
        
        Average Severity: {avg_severity:.2f}
        Average Confidence: {avg_confidence:.2%}
        
        High Anomalies: {(anomalies['type'] == 'high').sum()}
        Low Anomalies: {(anomalies['type'] == 'low').sum()}
        """
        
        ax2.text(0.1, 0.5, stats_text, fontsize=11, 
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # 3. Anomaly distribution by hour (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        hourly_anomalies = pd.Series([r.timestamp.hour for r in self.results if r.is_anomaly])
        if len(hourly_anomalies) > 0:
            hourly_anomalies.value_counts().sort_index().plot(kind='bar', ax=ax3, color='coral')
        ax3.set_title('Anomalies by Hour of Day', fontweight='bold')
        ax3.set_xlabel('Hour')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3)
        
        # 4. Severity distribution (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        if len(anomalies) > 0:
            ax4.hist(anomalies['severity'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.set_title('Severity Distribution', fontweight='bold')
        ax4.set_xlabel('Severity (|Z-score|)')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # 5. Confidence distribution (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        if len(anomalies) > 0:
            ax5.hist(anomalies['confidence'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax5.set_title('Confidence Distribution', fontweight='bold')
        ax5.set_xlabel('Confidence')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # 6. Time series of severity over time (bottom row, spanning all columns)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.scatter(anomalies['timestamp'], anomalies['severity'],
                   c=anomalies['confidence'], cmap='viridis', s=30, alpha=0.7)
        ax6.set_title('Anomaly Severity Over Time (colored by confidence)', fontweight='bold')
        ax6.set_xlabel('Timestamp')
        ax6.set_ylabel('Severity')
        ax6.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax6, label='Confidence')
        
        fig.suptitle(f'Anomaly Detection Dashboard - {self.series_name}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Dashboard saved to {save_path}")
        
        plt.show()
    
    def save_detailed_report(self, output_dir: str = "anomaly_reports"):
        """Save a detailed report with all visualizations and data"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = output_path / f"report_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # Save all visualizations
        print("📝 Generating comprehensive report...")
        
        self.plot_time_series_with_anomalies(
            save_path=str(report_dir / "time_series_plot.png")
        )
        
        self.create_interactive_plot(
            save_html=str(report_dir / "interactive_dashboard.html")
        )
        
        self.plot_anomaly_heatmap(
            save_path=str(report_dir / "anomaly_heatmap.png")
        )
        
        self.create_summary_dashboard(
            save_path=str(report_dir / "summary_dashboard.png")
        )
        
        # Save data
        self.df.to_csv(report_dir / "anomaly_data.csv", index=False)
        
        # Save summary statistics
        summary = {
            "total_points": len(self.df),
            "anomalies_detected": int(self.df['is_anomaly'].sum()),
            "anomaly_rate": float(self.df['is_anomaly'].mean()),
            "avg_severity": float(self.df[self.df['is_anomaly']]['severity'].mean()),
            "avg_confidence": float(self.df[self.df['is_anomaly']]['confidence'].mean()),
            "high_anomalies": int((self.df['type'] == 'high').sum()),
            "low_anomalies": int((self.df['type'] == 'low').sum())
        }
        
        with open(report_dir / "summary_stats.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Report saved to {report_dir}")
        return str(report_dir)


class ElectricityMonitor:
    """Monitor electricity consumption with anomaly detection"""
    
    def __init__(self, detector: ChronosAnomalyDetector):
        self.detector = detector
    
    def monitor_household(
        self, 
        data: pd.DataFrame, 
        household_id: str,
        visualize: bool = True
    ) -> Dict:
        """Monitor a household and optionally visualize results"""
        print(f"\n🏠 Monitoring {household_id}")
        print(f"   Data points: {len(data)}")
        print(f"   Period: {data['timestamp'].iloc[0]} to {data['timestamp'].iloc[-1]}")
        
        results = self.detector.detect_point_anomalies(
            data["load"].values,
            data["timestamp"].tolist()
        )
        
        anomalies = [r for r in results if r.is_anomaly]
        critical = [r for r in anomalies if r.confidence > 0.7]
        
        print(f"\n📊 Results:")
        print(f"   Total anomalies: {len(anomalies)}")
        print(f"   Critical anomalies (confidence > 0.7): {len(critical)}")
        print(f"   Anomaly rate: {len(anomalies)/len(results)*100:.2f}%")
        
        if visualize and len(results) > 0:
            visualizer = EnhancedVisualizer(results, household_id)
            report_dir = visualizer.save_detailed_report()
            
        return {
            "household_id": household_id,
            "anomalies": len(anomalies),
            "critical_anomalies": len(critical),
            "results": results
        }


def load_electricity_series(household_index: int = 0, max_points: int = 2500) -> pd.DataFrame:
    """Load electricity data from GluonTS"""
    print("⚡ Loading real electricity data via GluonTS...")
    ds = get_dataset("electricity")
    entry = list(ds.train)[household_index]
    series = entry["target"][-max_points:]
    start = entry["start"]
    timestamps = pd.date_range(start.to_timestamp(), periods=len(series), freq=start.freq)
    df = pd.DataFrame({"timestamp": timestamps, "load": series})
    print(f"✅ Loaded {len(df)} points for household_{household_index}")
    return df


def main():
    """Main execution function"""
    print("=" * 70)
    print(" CHRONOS ENHANCED ANOMALY DETECTION SYSTEM")
    print("=" * 70)
    
    # Initialize detector
    detector = ChronosAnomalyDetector()
    
    # Load data
    df = load_electricity_series(household_index=0, max_points=2500)
    
    # Run monitoring with visualization
    monitor = ElectricityMonitor(detector)
    summary = monitor.monitor_household(df, "household_1", visualize=True)
    
    print("\n🎯 Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()