import sys
import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.base import RecommenderBase

class MockRecommender(RecommenderBase):
    """Mock recommender for testing."""
    
    def __init__(self):
        super().__init__(name="MockRecommender")
        self.is_fitted = True
    
    def fit(self, train_data):
        self.is_fitted = True
    
    def predict(self, user_id, movie_id):
        return 3.5  # Mock prediction

def test_recommender_base():
    """Test the RecommenderBase class."""
    # Create mock recommender
    recommender = MockRecommender()
    
    # Test name
    assert recommender.name == "MockRecommender"
    
    # Test predict_batch
    user_movie_pairs = pd.DataFrame({
        'user_id': [1, 1, 2],
        'movie_id': [10, 20, 30]
    })
    
    predictions = recommender.predict_batch(user_movie_pairs)
    
    # Check predictions
    assert len(predictions) == 3
    assert all(predictions == 3.5)
    
    # Test recommend_for_user
    movie_pool = pd.DataFrame({
        'movie_id': [100, 200, 300],
        'title': ['Movie A', 'Movie B', 'Movie C']
    })
    
    recommendations = recommender.recommend_for_user(user_id=1, movie_pool=movie_pool, n=2)
    
    # Check recommendations
    assert len(recommendations) == 2
    assert all(recommendations['predicted_rating'] == 3.5)
    assert list(recommendations['movie_id']) == [100, 200]

if __name__ == "__main__":
    test_recommender_base()
    print("All tests passed!")