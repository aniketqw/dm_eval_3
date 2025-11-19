# download_dataset.py

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import pickle
from tqdm import tqdm
import json

class ChronosDatasetDownloader:
    """Download and prepare Chronos datasets from HuggingFace"""
    
    def __init__(self, save_dir: str = "./chronos_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def download_dataset(self):
        """Download the complete chronos_datasets from HuggingFace"""
        print("Downloading Chronos datasets from HuggingFace...")
        
        # Download specific dataset configurations
        dataset_names = [
            "monash_australian_electricity",
            "monash_electricity_hourly", 
            "monash_traffic",
            "monash_weather",
            "m4_hourly",
            "m4_daily",
            "m4_weekly",
            "m4_monthly"
        ]
        
        datasets_dict = {}
        for name in dataset_names:
            try:
                ds = load_dataset(
                    "autogluon/chronos_datasets",
                    name
                )
                datasets_dict[name] = ds
                print(f"✓ Downloaded: {name}")
            except Exception as e:
                print(f"✗ Failed to download {name}: {e}")
                
        return datasets_dict
    
    def process_and_save(self, dataset):
        """Process and save datasets in training-ready format"""
        print("\nProcessing and saving datasets...")
        
        # Create subdirectories
        (self.save_dir / "train").mkdir(exist_ok=True)
        (self.save_dir / "metadata").mkdir(exist_ok=True)
        
        processed_datasets = []
        metadata = {
            'datasets': [],
            'total_series': 0,
            'total_observations': 0
        }
        
        # Handle both dataset formats
        if isinstance(dataset, dict):
            # Multiple named datasets
            for dataset_name, ds in dataset.items():
                self._process_dataset(dataset_name, ds, processed_datasets, metadata)
        else:
            # Single dataset with splits
            for split_name in dataset.keys():
                self._process_dataset(split_name, dataset[split_name], 
                                    processed_datasets, metadata)
        
        # Save processed data
        print(f"\nSaving {len(processed_datasets)} time series...")
        
        with open(self.save_dir / "train" / "timeseries.pkl", 'wb') as f:
            pickle.dump(processed_datasets, f)
        
        # Save metadata
        with open(self.save_dir / "metadata" / "info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved {metadata['total_series']} time series")
        print(f"✓ Total observations: {metadata['total_observations']:,}")
        print(f"✓ Data saved to: {self.save_dir}")
        
        return processed_datasets, metadata
    
    def _process_dataset(self, name, dataset, processed_list, metadata):
        """Process a single dataset"""
        print(f"\nProcessing: {name}")
        
        try:
            series_count = 0
            obs_count = 0
            
            # Handle DatasetDict - get the 'train' split
            if hasattr(dataset, 'keys') and 'train' in dataset:
                dataset = dataset['train']
            
            # Iterate through the dataset
            for idx, item in enumerate(tqdm(dataset)):
                series = None
                
                # Extract time series data
                # Format varies, handle common structures
                if isinstance(item, dict):
                    if 'target' in item:
                        series = np.array(item['target'], dtype=np.float32)
                    elif 'value' in item:
                        series = np.array(item['value'], dtype=np.float32)
                    elif 'data' in item:
                        series = np.array(item['data'], dtype=np.float32)
                    else:
                        # Try to find array-like data
                        for key, value in item.items():
                            if isinstance(value, (list, np.ndarray)):
                                try:
                                    series = np.array(value, dtype=np.float32)
                                    if len(series.shape) == 1 and len(series) > 10:
                                        break
                                except:
                                    continue
                elif isinstance(item, (list, np.ndarray)):
                    # Item itself is the series
                    series = np.array(item, dtype=np.float32)
                
                # Skip if no valid series found
                if series is None:
                    continue
                
                # Validate and store
                if len(series.shape) == 1 and len(series) >= 20:  # Min length
                    # Remove NaNs and Infs
                    series = series[~np.isnan(series)]
                    series = series[~np.isinf(series)]
                    
                    if len(series) >= 20:
                        processed_list.append(series)
                        series_count += 1
                        obs_count += len(series)
            
            # Update metadata
            metadata['datasets'].append({
                'name': name,
                'num_series': series_count,
                'num_observations': obs_count
            })
            metadata['total_series'] += series_count
            metadata['total_observations'] += obs_count
            
            print(f"  → Processed {series_count} series, {obs_count:,} observations")
            
        except Exception as e:
            print(f"  ✗ Error processing {name}: {e}")
    
    def load_processed_data(self):
        """Load previously processed data"""
        data_path = self.save_dir / "train" / "timeseries.pkl"
        meta_path = self.save_dir / "metadata" / "info.json"
        
        if not data_path.exists():
            raise FileNotFoundError(f"No processed data found at {data_path}")
        
        with open(data_path, 'rb') as f:
            timeseries = pickle.load(f)
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded {len(timeseries)} time series")
        print(f"Total observations: {metadata['total_observations']:,}")
        
        return timeseries, metadata


def main():
    """Main execution"""
    print("="*60)
    print("CHRONOS DATASET DOWNLOADER")
    print("="*60)
    
    # Initialize downloader
    downloader = ChronosDatasetDownloader(save_dir="./chronos_data")
    
    # Download dataset
    dataset = downloader.download_dataset()
    
    # Process and save
    if dataset:
        processed_data, metadata = downloader.process_and_save(dataset)
        
        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        print(f"Datasets: {len(metadata['datasets'])}")
        print(f"Total time series: {metadata['total_series']}")
        print(f"Total observations: {metadata['total_observations']:,}")
        print("\nYou can now run the training script!")
    else:
        print("\n✗ Failed to download datasets")


if __name__ == "__main__":
    main()