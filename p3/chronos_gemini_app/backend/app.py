#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
chronos_gemini_backend.py - Backend API for Chronos + Gemini Chat System

This integrates:
1. Gemini API for natural language interaction
2. Your existing Chronos evaluation pipeline
3. Data analysis and visualization
4. Real-time forecasting
"""

import json
import os
import io
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from chronos import ChronosPipeline

# Import your existing modules
from chronos_common import (
    load_dataset,
    wql,
    mase,
    seasonal_naive_forecast,
)


# ==============================================================================
# FLASK APP SETUP
# ==============================================================================

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
Path(UPLOAD_FOLDER).mkdir(exist_ok=True)
Path(RESULTS_FOLDER).mkdir(exist_ok=True)

# Global cache for loaded models and datasets
MODEL_CACHE = {}
DATASET_CACHE = {}


# ==============================================================================
# GEMINI API INTEGRATION
# ==============================================================================

class GeminiAssistant:
    """
    Gemini AI assistant specialized for time series forecasting
    """
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.chat_sessions = {}  # Store conversation history
    
    def create_system_prompt(self, dataset_info: Dict = None) -> str:
        """Create detailed system prompt with dataset context"""
        
        base_prompt = """
You are an expert time series forecasting assistant specializing in the Chronos model family.

CHRONOS MODEL FAMILY:
- Architecture: Based on T5 (encoder-decoder transformer)
- Models available: tiny (8M), mini (20M), small (46M), base (200M), large (710M)
- Training: Pretrained on 100B+ time series tokens
- Capabilities: Zero-shot forecasting, probabilistic predictions, multi-horizon
- Best for: Sales, demand, finance, energy, weather, web traffic

KEY CONCEPTS:
1. Context Length: Historical data used for prediction (64-512 timesteps optimal)
2. Prediction Horizon: Number of future timesteps to forecast
3. Quantile Forecasting: Generates prediction intervals (e.g., 10%, 50%, 90%)
4. WQL (Weighted Quantile Loss): Measures probabilistic accuracy
5. MASE (Mean Absolute Scaled Error): Compares to seasonal naive baseline

YOUR TASKS:
- Analyze uploaded datasets and provide insights
- Recommend optimal Chronos configuration
- Suggest preprocessing steps
- Explain forecasting results
- Generate Python code snippets
- Help interpret metrics (WQL, MASE, etc.)
- Answer questions about time series modeling

RESPONSE STYLE:
- Be conversational but technical
- Use emojis sparingly for clarity
- Provide code examples when helpful
- Explain complex concepts simply
- Always consider the user's specific dataset
"""
        
        if dataset_info:
            dataset_context = f"""

CURRENT DATASET INFORMATION:
- Total rows: {dataset_info.get('total_rows', 'N/A')}
- Total columns: {dataset_info.get('total_columns', 'N/A')}
- Date columns: {', '.join(dataset_info.get('date_columns', [])) or 'None'}
- Target columns: {', '.join(dataset_info.get('target_columns', [])) or 'None'}
- Numeric columns: {', '.join(dataset_info.get('numeric_columns', [])) or 'None'}

{self._format_target_stats(dataset_info.get('target_stats'))}

KEY INSIGHTS:
{self._generate_insights(dataset_info)}
"""
            return base_prompt + dataset_context
        
        return base_prompt
    
    def _format_target_stats(self, stats: Dict) -> str:
        """Format target variable statistics"""
        if not stats:
            return ""
        
        return f"""
TARGET VARIABLE STATISTICS:
- Mean: {stats.get('mean', 0):.4f}
- Median: {stats.get('median', 0):.4f}
- Std Dev: {stats.get('std', 0):.4f}
- Min: {stats.get('min', 0):.4f}
- Max: {stats.get('max', 0):.4f}
- CV (Coefficient of Variation): {stats.get('cv', 0):.4f}
"""
    
    def _generate_insights(self, dataset_info: Dict) -> str:
        """Generate automatic insights from dataset"""
        insights = []
        
        stats = dataset_info.get('target_stats', {})
        if stats:
            cv = stats.get('cv', 0)
            if cv > 0.5:
                insights.append("- High variability detected (CV > 0.5) - consider log transformation")
            elif cv < 0.1:
                insights.append("- Low variability detected - data is relatively stable")
            
            if stats.get('mean', 0) > 0:
                range_ratio = (stats.get('max', 0) - stats.get('min', 0)) / stats.get('mean', 1)
                if range_ratio > 10:
                    insights.append("- Large value range - normalization recommended")
        
        total_rows = dataset_info.get('total_rows', 0)
        if total_rows < 100:
            insights.append("- Small dataset - consider using smaller Chronos model (tiny/mini)")
        elif total_rows > 1000:
            insights.append("- Large dataset - can leverage larger models (base/large)")
        
        return '\n'.join(insights) if insights else "- No specific concerns detected"
    
    def chat(self, user_message: str, session_id: str, dataset_info: Dict = None) -> str:
        """Send message and get response"""
        
        # Initialize or get chat session
        if session_id not in self.chat_sessions:
            system_prompt = self.create_system_prompt(dataset_info)
            self.chat_sessions[session_id] = self.model.start_chat(history=[])
            # Send system prompt as first message
            self.chat_sessions[session_id].send_message(system_prompt)
        
        chat = self.chat_sessions[session_id]
        
        # Send user message
        response = chat.send_message(user_message)
        
        return response.text


# ==============================================================================
# CHRONOS INTEGRATION
# ==============================================================================

class ChronosForecaster:
    """
    Wrapper for Chronos pipeline with caching and utilities
    """
    
    def __init__(self, model_name: str = "amazon/chronos-t5-base"):
        self.model_name = model_name
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load Chronos model"""
        if self.model_name in MODEL_CACHE:
            self.pipeline = MODEL_CACHE[self.model_name]
            print(f"✓ Loaded {self.model_name} from cache")
        else:
            print(f"Loading {self.model_name}...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=device,
                torch_dtype=torch.bfloat16,
            )
            MODEL_CACHE[self.model_name] = self.pipeline
            print(f"✓ Loaded {self.model_name}")
    
    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int = 8,
        num_samples: int = 20,
        quantile_levels: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[str, Any]:
        """
        Generate forecast
        
        Returns:
            Dictionary with predictions, quantiles, and samples
        """
        tensor = torch.from_numpy(context).unsqueeze(0)
        
        with torch.no_grad():
            samples = self.pipeline.predict(
                tensor,
                prediction_length=prediction_length,
                num_samples=num_samples,
            )
        
        samples_np = samples.cpu().numpy().squeeze(0)
        
        # Calculate quantiles
        quantiles = {
            f'q{int(q*100)}': np.quantile(samples_np, q, axis=0).tolist()
            for q in quantile_levels
        }
        
        return {
            'median': quantiles['q50'],
            'quantiles': quantiles,
            'samples': samples_np.tolist(),
            'mean': np.mean(samples_np, axis=0).tolist(),
            'std': np.std(samples_np, axis=0).tolist(),
        }
    
    def evaluate(
        self,
        context: np.ndarray,
        target: np.ndarray,
        prediction_length: int = 8,
        season_length: int = 4,
    ) -> Dict[str, float]:
        """
        Evaluate forecast against ground truth
        """
        # Generate forecast
        forecast_result = self.forecast(context, prediction_length)
        median_pred = np.array(forecast_result['median'])
        
        # Calculate quantiles for WQL
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantile_preds = np.column_stack([
            np.quantile(np.array(forecast_result['samples']), q, axis=0)
            for q in quantile_levels
        ])
        
        # Calculate metrics
        model_wql = wql(target, quantile_preds, quantile_levels)
        model_mase = mase(target, median_pred, context, season_length)
        
        # Baseline metrics
        naive_forecast = seasonal_naive_forecast(context, prediction_length, season_length)
        naive_quantile_preds = np.tile(
            naive_forecast.reshape(-1, 1),
            (1, len(quantile_levels))
        )
        baseline_wql = wql(target, naive_quantile_preds, quantile_levels)
        baseline_mase = mase(target, naive_forecast, context, season_length)
        
        return {
            'wql': float(model_wql),
            'mase': float(model_mase),
            'baseline_wql': float(baseline_wql),
            'baseline_mase': float(baseline_mase),
            'relative_wql': float(model_wql / baseline_wql) if baseline_wql > 0 else np.nan,
            'relative_mase': float(model_mase / baseline_mase) if baseline_mase > 0 else np.nan,
        }


# ==============================================================================
# DATA ANALYSIS UTILITIES
# ==============================================================================

def analyze_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive dataset analysis
    """
    # Basic info
    info = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'headers': df.columns.tolist(),
    }
    
    # Detect column types
    date_cols = []
    target_cols = []
    numeric_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Date detection
        if any(kw in col_lower for kw in ['date', 'time', 'timestamp']):
            date_cols.append(col)
        
        # Target detection
        if any(kw in col_lower for kw in ['target', 'value', 'sales', 'demand', 'price']):
            target_cols.append(col)
        
        # Numeric detection
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    
    info['date_columns'] = date_cols
    info['target_columns'] = target_cols
    info['numeric_columns'] = numeric_cols
    
    # Calculate statistics for target column
    if target_cols:
        target_col = target_cols[0]
        values = df[target_col].dropna()
        
        if len(values) > 0:
            info['target_stats'] = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'cv': float(values.std() / values.mean()) if values.mean() != 0 else 0,
            }
    
    # Check for missing values
    info['missing_values'] = df.isnull().sum().to_dict()
    
    # Sample data
    info['sample_data'] = df.head(5).to_dict('records')
    
    return info


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """
    Upload and analyze dataset
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        filename = file.filename
        filepath = Path(UPLOAD_FOLDER) / filename
        file.save(filepath)
        
        # Load and analyze
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Analyze
        analysis = analyze_dataframe(df)
        
        # Cache dataset
        session_id = request.form.get('session_id', 'default')
        DATASET_CACHE[session_id] = {
            'dataframe': df,
            'filepath': str(filepath),
            'analysis': analysis,
        }
        
        return jsonify({
            'success': True,
            'filename': filename,
            'analysis': analysis,
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat with Gemini assistant
    """
    try:
        data = request.json
        user_message = data.get('message')
        session_id = data.get('session_id', 'default')
        gemini_key = data.get('gemini_key')
        
        if not user_message or not gemini_key:
            return jsonify({'error': 'Missing message or API key'}), 400
        
        # Get dataset info if available
        dataset_info = None
        if session_id in DATASET_CACHE:
            dataset_info = DATASET_CACHE[session_id]['analysis']
        
        # Initialize Gemini assistant
        assistant = GeminiAssistant(gemini_key)
        
        # Get response
        response = assistant.chat(user_message, session_id, dataset_info)
        
        return jsonify({
            'success': True,
            'response': response,
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast', methods=['POST'])
def forecast():
    """
    Generate forecast for a time series
    """
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        model_name = data.get('model_name', 'amazon/chronos-t5-base')
        prediction_length = data.get('prediction_length', 8)
        num_samples = data.get('num_samples', 20)
        series_id = data.get('series_id', None)
        
        # Get dataset
        if session_id not in DATASET_CACHE:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        df = DATASET_CACHE[session_id]['dataframe']
        analysis = DATASET_CACHE[session_id]['analysis']
        
        # Get target column
        target_cols = analysis['target_columns']
        if not target_cols:
            return jsonify({'error': 'No target column found'}), 400
        
        target_col = target_cols[0]
        
        # Get series
        if 'item_id' in df.columns and series_id:
            series_df = df[df['item_id'] == series_id]
        else:
            series_df = df
        
        time_series = series_df[target_col].values.astype(np.float32)
        
        # Split into context and target (if enough data)
        if len(time_series) > prediction_length:
            context = time_series[:-prediction_length]
            target = time_series[-prediction_length:]
        else:
            context = time_series
            target = None
        
        # Initialize forecaster
        forecaster = ChronosForecaster(model_name)
        
        # Generate forecast
        forecast_result = forecaster.forecast(
            context,
            prediction_length,
            num_samples
        )
        
        # Evaluate if target available
        evaluation = None
        if target is not None:
            evaluation = forecaster.evaluate(context, target, prediction_length)
        
        return jsonify({
            'success': True,
            'forecast': forecast_result,
            'evaluation': evaluation,
            'context_length': len(context),
            'series_length': len(time_series),
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_dataset():
    """
    Evaluate entire dataset
    """
    try:
        data = request.json
        session_id = data.get('session_id', 'default')
        model_name = data.get('model_name', 'amazon/chronos-t5-base')
        prediction_length = data.get('prediction_length', 8)
        season_length = data.get('season_length', 4)
        max_series = data.get('max_series', 10)  # Limit for demo
        
        # Get dataset
        if session_id not in DATASET_CACHE:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        df = DATASET_CACHE[session_id]['dataframe']
        analysis = DATASET_CACHE[session_id]['analysis']
        
        # Get target column
        target_cols = analysis['target_columns']
        if not target_cols:
            return jsonify({'error': 'No target column found'}), 400
        
        target_col = target_cols[0]
        
        # Initialize forecaster
        forecaster = ChronosForecaster(model_name)
        
        # Evaluate each series
        results = []
        
        if 'item_id' in df.columns:
            groups = df.groupby('item_id')
            series_list = list(groups)[:max_series]
        else:
            series_list = [(None, df)]
        
        for item_id, group in series_list:
            time_series = group[target_col].values.astype(np.float32)
            
            if len(time_series) > prediction_length:
                context = time_series[:-prediction_length]
                target = time_series[-prediction_length:]
                
                eval_result = forecaster.evaluate(
                    context, target, prediction_length, season_length
                )
                eval_result['item_id'] = str(item_id) if item_id else 'series_0'
                results.append(eval_result)
        
        # Aggregate results
        if results:
            from scipy.stats import gmean
            
            wqls = [r['wql'] for r in results if not np.isnan(r['wql'])]
            mases = [r['mase'] for r in results if not np.isnan(r['mase'])]
            
            summary = {
                'wql_mean': float(gmean(wqls)) if wqls else np.nan,
                'mase_mean': float(gmean(mases)) if mases else np.nan,
                'series_evaluated': len(results),
            }
        else:
            summary = {}
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': summary,
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    print("="*70)
    print("Chronos + Gemini Backend API")
    print("="*70)
    print("\nEndpoints:")
    print("  POST /api/upload     - Upload dataset")
    print("  POST /api/chat       - Chat with Gemini")
    print("  POST /api/forecast   - Generate forecast")
    print("  POST /api/evaluate   - Evaluate dataset")
    print("\nStarting server on http://localhost:5000")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)