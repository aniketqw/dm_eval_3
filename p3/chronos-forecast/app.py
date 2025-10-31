#!/usr/bin/env python3
"""
Minimal Chronos + Gemini Backend
Upload CSV → Ask Question → Get AI-powered forecast answer
"""

import os
import io
import json
import numpy as np
import pandas as pd
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from chronos import ChronosPipeline

app = Flask(__name__)
CORS(app)

# Load Chronos model at startup
print("Loading Chronos model...")
CHRONOS_MODEL = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
print("✓ Model loaded")

# Global storage for uploaded data
DATASET_CACHE = {}


def analyze_csv(df):
    """Quick analysis of uploaded CSV"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    # Auto-detect target column (first numeric column)
    target_col = numeric_cols[0] if numeric_cols else None
    
    stats = {}
    if target_col:
        values = df[target_col].dropna()
        stats = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
            'count': len(values)
        }
    
    return {
        'rows': len(df),
        'columns': list(df.columns),
        'numeric_columns': numeric_cols,
        'date_columns': date_cols,
        'target_column': target_col,
        'target_stats': stats,
        'sample': df.head(3).to_dict('records')
    }


def run_chronos_forecast(series, prediction_length=12):
    """Run Chronos forecasting"""
    # Convert to tensor
    context = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
    
    # Generate forecast
    with torch.no_grad():
        forecast = CHRONOS_MODEL.predict(
            context,
            prediction_length=prediction_length,
            num_samples=20
        )
    
    # Extract quantiles
    forecast_np = forecast.squeeze(0).cpu().numpy()
    
    return {
        'median': np.median(forecast_np, axis=0).tolist(),
        'mean': np.mean(forecast_np, axis=0).tolist(),
        'q10': np.quantile(forecast_np, 0.1, axis=0).tolist(),
        'q90': np.quantile(forecast_np, 0.9, axis=0).tolist(),
        'prediction_length': prediction_length
    }


def ask_gemini(question, data_context, forecast_result, api_key):
    """Query Gemini with context"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Build comprehensive prompt
    prompt = f"""You are a time series forecasting expert assistant. 

DATASET INFORMATION:
{json.dumps(data_context, indent=2)}

CHRONOS FORECAST RESULTS:
{json.dumps(forecast_result, indent=2)}

USER QUESTION:
{question}

INSTRUCTIONS:
- Analyze the forecast results in context of the user's question
- Explain predictions in simple, business-friendly language
- Highlight trends, patterns, and confidence intervals
- Give actionable insights
- Use the actual numbers from the forecast
- Be concise but informative

Your answer:"""
    
    response = model.generate_content(prompt)
    return response.text


@app.route('/api/forecast', methods=['POST'])
def forecast_endpoint():
    """
    Main endpoint: Upload CSV + Ask Question → Get AI Answer
    
    Form data:
    - file: CSV file
    - question: Natural language question
    - gemini_key: Gemini API key
    - prediction_length: Optional, default 12
    """
    try:
        # Get inputs
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        question = request.form.get('question', 'What does the forecast show?')
        gemini_key = request.form.get('gemini_key')
        prediction_length = int(request.form.get('prediction_length', 12))
        
        if not gemini_key:
            return jsonify({'error': 'Gemini API key required'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Analyze dataset
        analysis = analyze_csv(df)
        
        if not analysis['target_column']:
            return jsonify({'error': 'No numeric column found for forecasting'}), 400
        
        # Extract time series
        target_col = analysis['target_column']
        series = df[target_col].dropna().values.astype(np.float32)
        
        if len(series) < prediction_length + 10:
            return jsonify({'error': f'Need at least {prediction_length + 10} data points'}), 400
        
        # Use last N points as context (leave some for potential validation)
        context = series[-100:] if len(series) > 100 else series
        
        # Run Chronos forecast
        forecast_result = run_chronos_forecast(context, prediction_length)
        
        # Get Gemini analysis
        gemini_response = ask_gemini(
            question,
            analysis,
            forecast_result,
            gemini_key
        )
        
        # Return everything
        return jsonify({
            'success': True,
            'data_analysis': analysis,
            'forecast': forecast_result,
            'ai_answer': gemini_response,
            'context_used': len(context)
        })
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'model': 'chronos-t5-small'})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Chronos + Gemini Forecasting API")
    print("="*60)
    print("Endpoints:")
    print("  POST /api/forecast - Upload CSV + ask question")
    print("  GET  /health       - Health check")
    print("\nStarting server on http://0.0.0.0:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)