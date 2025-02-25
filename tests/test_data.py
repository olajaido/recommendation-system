import sys
import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

def test_data_structure():
    """Test the data structure and basic properties."""
    # This is a basic test to ensure the processed data exists and has the expected structure
    processed_dir = Path("data/processed")
    
    # Skip if data directory doesn't exist (CI environment)
    if not processed_dir.exists():
        pytest.skip("Processed data directory not found")
    
    # Check if key files exist
    train_path = processed_dir / "train_ratings.csv"
    
    if not train_path.exists():
        pytest.skip("Training data file not found")
    
    # Load the data
    train_df = pd.read_csv(train_path)
    
    # Check structure
    assert 'user_id' in train_df.columns
    assert 'movie_id' in train_df.columns
    assert 'rating' in train_df.columns
    
    # Check data properties
    assert len(train_df) > 0
    assert train_df['rating'].min() >= 1
    assert train_df['rating'].max() <= 5

if __name__ == "__main__":
    test_data_structure()
    print("All tests passed!")