import os
import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from tqdm import tqdm

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.logger import setup_logger
from src.serving.recommender import load_model, load_movie_data

logger = setup_logger(__name__)

class BatchRecommendationGenerator:
    """Generates batch recommendations for all users."""
    
    def __init__(self, 
                model_name="hybrid_recommender",
                output_dir="data/recommendations",
                num_recommendations=20):
        """
        Initialize batch recommendation generator.
        
        Args:
            model_name (str): Name of the model to use
            output_dir (str): Directory to save recommendations
            num_recommendations (int): Number of recommendations per user
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.num_recommendations = num_recommendations
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self.load_model()
        
        # Load movie data
        self.movies_df = load_movie_data()
        
        # Get all users
        self.users = self.get_users()
    
    def load_model(self):
        """
        Load the recommendation model.
        
        Returns:
            object: Loaded model
        """
        logger.info(f"Loading {self.model_name} model")
        
        model_path = Path("models") / f"{self.model_name}.pkl"
        
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_users(self):
        """
        Get list of all users.
        
        Returns:
            list: List of user IDs
        """
        logger.info("Getting list of users")
        
        try:
            # In a real system, this would get users from a database
            # For this example, we'll use the user IDs from the model
            
            if hasattr(self.model, 'user_item_matrix'):
                return self.model.user_item_matrix.index.tolist()
            elif hasattr(self.model, 'dataset') and hasattr(self.model.dataset, 'user_encoder'):
                return self.model.dataset.user_encoder.classes_.tolist()
            else:
                # Fallback to a range of user IDs
                logger.warning("Could not determine user list from model, using range 1-943")
                return list(range(1, 944))
        
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            # Fallback to a range of user IDs
            return list(range(1, 944))
    
    def generate_user_recommendations(self, user_id):
        """
        Generate recommendations for a user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            pd.DataFrame: User recommendations
        """
        try:
            # Get user's rated movies to exclude from recommendations
            user_rated_movies = []
            if hasattr(self.model, 'user_item_matrix') and user_id in self.model.user_item_matrix.index:
                user_ratings = self.model.user_item_matrix.loc[user_id]
                user_rated_movies = user_ratings[user_ratings > 0].index.tolist()
            
            # Generate candidate movies (excluding already rated ones)
            candidate_movies = self.movies_df[~self.movies_df['movie_id'].isin(user_rated_movies)]
            
            # Generate recommendations
            recommendations = self.model.recommend_for_user(user_id, candidate_movies, n=self.num_recommendations)
            
            # Add recommendation timestamp
            recommendations['timestamp'] = datetime.now().isoformat()
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")
            return pd.DataFrame()
    
    def generate_all_recommendations(self):
        """Generate recommendations for all users."""
        logger.info(f"Generating batch recommendations for {len(self.users)} users")
        
        # Initialize results
        all_recommendations = []
        successful_users = 0
        
        # Generate recommendations for each user
        start_time = time.time()
        
        for user_id in tqdm(self.users, desc="Generating recommendations"):
            user_recs = self.generate_user_recommendations(user_id)
            
            if not user_recs.empty:
                all_recommendations.append(user_recs)
                successful_users += 1
        
        # Combine all recommendations
        if all_recommendations:
            combined_recs = pd.concat(all_recommendations, ignore_index=True)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"batch_recommendations_{timestamp}.csv"
            
            combined_recs.to_csv(output_path, index=False)
            
            logger.info(f"Generated recommendations for {successful_users} users")
            logger.info(f"Total recommendations: {len(combined_recs)}")
            logger.info(f"Saved recommendations to {output_path}")
            
            # Generate summary
            self.generate_summary(combined_recs, successful_users, start_time)
        else:
            logger.warning("No recommendations were generated")
    
    def generate_summary(self, recommendations, successful_users, start_time):
        """
        Generate summary of batch recommendations.
        
        Args:
            recommendations (pd.DataFrame): All recommendations
            successful_users (int): Number of users with recommendations
            start_time (float): Start time of batch generation
        """
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate statistics
        total_recommendations = len(recommendations)
        avg_recommendations_per_user = total_recommendations / successful_users if successful_users > 0 else 0
        
        # Get top recommended movies
        top_movies = recommendations['movie_id'].value_counts().head(10)
        top_movie_titles = []
        
        for movie_id, count in top_movies.items():
            movie_title = self.movies_df[self.movies_df['movie_id'] == movie_id]['title'].iloc[0]
            top_movie_titles.append((movie_title, int(count)))
        
        # Create summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
            "processing_time_seconds": processing_time,
            "total_users": len(self.users),
            "successful_users": successful_users,
            "total_recommendations": total_recommendations,
            "avg_recommendations_per_user": avg_recommendations_per_user,
            "top_recommended_movies": top_movie_titles
        }
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.output_dir / f"batch_summary_{timestamp}.json"
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Saved batch summary to {summary_path}")

if __name__ == "__main__":
    logger.info("Starting batch recommendation generation")
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate batch recommendations")
    parser.add_argument("--model", type=str, default="hybrid_recommender",
                        help="Model to use for recommendations")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of recommendations per user")
    parser.add_argument("--output", type=str, default="data/recommendations",
                        help="Output directory for recommendations")
    
    args = parser.parse_args()
    
    # Create generator
    generator = BatchRecommendationGenerator(
        model_name=args.model,
        output_dir=args.output,
        num_recommendations=args.n
    )
    
    # Generate recommendations
    generator.generate_all_recommendations()
    
    logger.info("Batch recommendation generation completed")