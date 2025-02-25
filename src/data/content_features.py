import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ContentFeatureProcessor:
    """Processes and extracts content features from movie metadata."""
    
    def __init__(self):
        """Initialize the content feature processor."""
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
    
    def load_movie_data(self):
        """
        Load movie metadata.
        
        Returns:
            pd.DataFrame: Movies dataframe with metadata
        """
        # Assuming movies.csv is in the processed data directory
        processed_dir = Path("data/processed")
        movies_path = processed_dir / "movies.csv"
        
        logger.info(f"Loading movie data from {movies_path}")
        movies_df = pd.read_csv(movies_path)
        return movies_df
    
    def process_features(self):
        """Process movie features and create similarity matrix."""
        logger.info("Processing movie content features")
        
        # Load movie data if not already loaded
        if self.movies_df is None:
            self.movies_df = self.load_movie_data()
        
        # Convert genres from string to list if needed
        if isinstance(self.movies_df['genres'].iloc[0], str):
            self.movies_df['genres'] = self.movies_df['genres'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
            )
        
        # Create a string representation of genres for TF-IDF
        self.movies_df['genres_str'] = self.movies_df['genres'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else str(x)
        )
        
        # Add year as a feature if available
        if 'year' in self.movies_df.columns:
            # Convert year to decade for better generalization
            self.movies_df['decade'] = self.movies_df['year'].apply(
                lambda x: f"decade_{str(x)[:-1]}0s" if pd.notna(x) and str(x).isdigit() else "unknown_decade"
            )
            # Add decade to features
            self.movies_df['content_features'] = self.movies_df['genres_str'] + ' ' + self.movies_df['decade']
        else:
            self.movies_df['content_features'] = self.movies_df['genres_str']
        
        # Create TF-IDF vectorizer
        logger.info("Creating TF-IDF matrix")
        tfidf = TfidfVectorizer(stop_words='english')
        
        # Replace NaN with empty string
        self.movies_df['content_features'] = self.movies_df['content_features'].fillna('')
        
        # Create TF-IDF matrix
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['content_features'])
        
        # Compute cosine similarity matrix
        logger.info("Computing cosine similarity matrix")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create reverse mapping of indices and movie IDs
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['movie_id']).drop_duplicates()
        
        logger.info("Content feature processing completed")
    
    def get_similar_movies(self, movie_id, n=10):
        """
        Get similar movies based on content features.
        
        Args:
            movie_id (int): Movie ID
            n (int): Number of similar movies to return
            
        Returns:
            pd.DataFrame: Similar movies with similarity scores
        """
        # Process features if not already done
        if self.cosine_sim is None:
            self.process_features()
        
        # Check if movie_id exists
        if movie_id not in self.indices:
            logger.warning(f"Movie ID {movie_id} not found in content features")
            return pd.DataFrame()
        
        # Get index of the movie
        idx = self.indices[movie_id]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:n+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        # Return dataframe with similar movies
        similar_movies = self.movies_df.iloc[movie_indices].copy()
        similar_movies['similarity'] = similarity_scores
        
        return similar_movies[['movie_id', 'title', 'similarity']]
    
    def get_content_vector(self, movie_id):
        """
        Get content feature vector for a movie.
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            np.ndarray: Feature vector
        """
        # Process features if not already done
        if self.tfidf_matrix is None:
            self.process_features()
        
        # Check if movie_id exists
        if movie_id not in self.indices:
            logger.warning(f"Movie ID {movie_id} not found in content features")
            return None
        
        # Get index of the movie
        idx = self.indices[movie_id]
        
        # Return feature vector
        return self.tfidf_matrix[idx].toarray().flatten()