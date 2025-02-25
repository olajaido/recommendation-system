import os
import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelDriftMonitor:
    """Monitors data drift for recommendation models."""
    
    def __init__(self, 
                reference_data_path="data/processed/train_ratings.csv",
                production_data_path="data/production",
                drift_threshold=0.1,
                check_interval_minutes=60):
        """
        Initialize drift monitor.
        
        Args:
            reference_data_path (str): Path to reference data
            production_data_path (str): Path to production data directory
            drift_threshold (float): Threshold for drift detection
            check_interval_minutes (int): Interval between drift checks
        """
        self.reference_data_path = Path(reference_data_path)
        self.production_data_path = Path(production_data_path)
        self.drift_threshold = drift_threshold
        self.check_interval_minutes = check_interval_minutes
        self.monitoring_results_path = Path("metrics/drift_monitoring")
        
        # Create directories if they don't exist
        self.monitoring_results_path.mkdir(parents=True, exist_ok=True)
        self.production_data_path.mkdir(parents=True, exist_ok=True)
        
        # Load reference data
        self.reference_data = None
        self.load_reference_data()
    
    def load_reference_data(self):
        """Load reference data."""
        logger.info(f"Loading reference data from {self.reference_data_path}")
        try:
            self.reference_data = pd.read_csv(self.reference_data_path)
            logger.info(f"Loaded reference data with {len(self.reference_data)} rows")
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            raise
    
    def collect_production_data(self):
        """
        Collect and aggregate production data.
        
        In a real system, this would collect recent user interactions
        from logs, databases, or message queues.
        
        Returns:
            pd.DataFrame: Recent production data
        """
        logger.info("Collecting production data")
        
        # In a real system, this would collect data from databases or logs
        # For this example, we'll simulate production data from files
        
        production_files = list(self.production_data_path.glob("*.csv"))
        
        if not production_files:
            logger.warning("No production data files found")
            return None
        
        # Read and concatenate all production data files
        dfs = []
        for file in production_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading production data file {file}: {e}")
        
        if not dfs:
            logger.warning("No valid production data found")
            return None
        
        # Combine all production data
        production_data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Collected {len(production_data)} production data points")
        
        return production_data
    
    def detect_drift(self, production_data):
        """
        Detect drift between reference and production data.
        
        Args:
            production_data (pd.DataFrame): Recent production data
            
        Returns:
            dict: Drift detection results
        """
        logger.info("Detecting data drift")
        
        if production_data is None or len(production_data) < 100:
            logger.warning("Insufficient production data for drift detection")
            return {
                "drift_detected": False,
                "reason": "Insufficient production data",
                "timestamp": datetime.now().isoformat()
            }
        
        # Features to monitor for drift
        features_to_monitor = ["rating"]
        
        # Additional features if available
        if "timestamp" in production_data.columns:
            features_to_monitor.append("timestamp")
        
        # Calculate drift for each feature
        drift_results = {}
        drift_detected = False
        
        for feature in features_to_monitor:
            if feature not in production_data.columns:
                continue
                
            # Handle different data types
            if feature == "timestamp" and pd.api.types.is_object_dtype(production_data[feature]):
                # Convert to datetime then to numeric
                ref_values = pd.to_datetime(self.reference_data[feature]).astype(np.int64)
                prod_values = pd.to_datetime(production_data[feature]).astype(np.int64)
            else:
                ref_values = self.reference_data[feature]
                prod_values = production_data[feature]
            
            # Perform Kolmogorov-Smirnov test
            ks_stat, p_value = ks_2samp(ref_values, prod_values)
            
            # Check if drift is detected
            is_drift = p_value < self.drift_threshold
            
            drift_results[feature] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": is_drift
            }
            
            if is_drift:
                drift_detected = True
                
                # Create visualization for the drift
                self.visualize_drift(feature, ref_values, prod_values, ks_stat, p_value)
        
        # Additional checks for rating distribution changes
        if "rating" in production_data.columns:
            # Check for distribution shift
            ref_mean = self.reference_data["rating"].mean()
            prod_mean = production_data["rating"].mean()
            
            mean_shift = abs(ref_mean - prod_mean)
            relative_shift = mean_shift / ref_mean
            
            drift_results["rating_mean_shift"] = {
                "reference_mean": float(ref_mean),
                "production_mean": float(prod_mean),
                "absolute_shift": float(mean_shift),
                "relative_shift": float(relative_shift),
                "drift_detected": relative_shift > 0.1  # 10% shift threshold
            }
            
            if relative_shift > 0.1:
                drift_detected = True
        
        # Compare user activity patterns if user_id is available
        if "user_id" in production_data.columns:
            # Check for new users percentage
            ref_users = set(self.reference_data["user_id"])
            prod_users = set(production_data["user_id"])
            
            new_users = prod_users - ref_users
            new_user_percentage = len(new_users) / len(prod_users) if prod_users else 0
            
            drift_results["new_users"] = {
                "percentage": float(new_user_percentage),
                "count": len(new_users),
                "drift_detected": new_user_percentage > 0.3  # 30% new users threshold
            }
            
            if new_user_percentage > 0.3:
                drift_detected = True
        
        # Compile final results
        results = {
            "drift_detected": drift_detected,
            "feature_drift": drift_results,
            "timestamp": datetime.now().isoformat(),
            "reference_data_size": len(self.reference_data),
            "production_data_size": len(production_data)
        }
        
        # Save results
        self.save_results(results)
        
        return results
    
    def visualize_drift(self, feature, ref_values, prod_values, ks_stat, p_value):
        """
        Create visualization for drift detection.
        
        Args:
            feature (str): Feature name
            ref_values (pd.Series): Reference data values
            prod_values (pd.Series): Production data values
            ks_stat (float): KS statistic
            p_value (float): p-value
        """
        plt.figure(figsize=(10, 6))
        
        # Create distribution plots
        sns.kdeplot(ref_values, label="Reference Data", color="blue")
        sns.kdeplot(prod_values, label="Production Data", color="red")
        
        plt.title(f"Distribution Drift Detection - {feature}\nKS Stat: {ks_stat:.4f}, p-value: {p_value:.4e}")
        plt.xlabel(feature)
        plt.ylabel("Density")
        plt.legend()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.monitoring_results_path / f"drift_{feature}_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved drift visualization to {plot_path}")
    
    def save_results(self, results):
        """
        Save drift detection results.
        
        Args:
            results (dict): Drift detection results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.monitoring_results_path / f"drift_results_{timestamp}.json"
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Saved drift detection results to {results_path}")
    
    def run_monitoring_loop(self):
        """Run continuous monitoring loop."""
        logger.info(f"Starting drift monitoring loop (interval: {self.check_interval_minutes} minutes)")
        
        while True:
            try:
                # Collect production data
                production_data = self.collect_production_data()
                
                if production_data is not None and len(production_data) > 0:
                    # Detect drift
                    results = self.detect_drift(production_data)
                    
                    # Log results
                    if results["drift_detected"]:
                        logger.warning("Data drift detected!")
                        logger.warning(f"Drift results: {json.dumps(results, indent=2)}")
                    else:
                        logger.info("No significant data drift detected")
                
                # Wait for next check
                logger.info(f"Next drift check in {self.check_interval_minutes} minutes")
                time.sleep(self.check_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in drift monitoring: {e}")
                time.sleep(300)  # Wait 5 minutes on error

def simulate_production_data():
    """
    Simulate production data for testing drift detection.
    
    In a real system, this would not be needed as actual user
    interactions would be collected.
    """
    logger.info("Simulating production data")
    
    try:
        # Load original data
        reference_data = pd.read_csv("data/processed/train_ratings.csv")
        
        # Create production data directory
        production_dir = Path("data/production")
        production_dir.mkdir(parents=True, exist_ok=True)
        
        # Create normal production data (similar to reference)
        normal_data = reference_data.sample(n=min(5000, len(reference_data)), random_state=42)
        normal_data.to_csv(production_dir / "normal_production_data.csv", index=False)
        
        # Create drifted production data (with shifted ratings)
        drifted_data = reference_data.sample(n=min(5000, len(reference_data)), random_state=43)
        # Shift ratings up by 0.5
        drifted_data["rating"] = drifted_data["rating"].apply(lambda x: min(5.0, x + 0.5))
        drifted_data.to_csv(production_dir / "drifted_production_data.csv", index=False)
        
        logger.info("Created simulated production data for testing")
        
    except Exception as e:
        logger.error(f"Error simulating production data: {e}")

if __name__ == "__main__":
    logger.info("Starting model drift monitoring")
    
    # Simulate production data for testing
    simulate_production_data()
    
    # Create and run the drift monitor
    monitor = ModelDriftMonitor(
        reference_data_path="data/processed/train_ratings.csv",
        production_data_path="data/production",
        drift_threshold=0.05,
        check_interval_minutes=10  # Every 10 minutes for testing
    )
    
    monitor.run_monitoring_loop()