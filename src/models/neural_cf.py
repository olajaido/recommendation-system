import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.models.base import RecommenderBase
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class MovieLensDataset(Dataset):
    """PyTorch dataset for MovieLens."""
    
    def __init__(self, ratings_df):
        """
        Initialize MovieLens dataset.
        
        Args:
            ratings_df (pd.DataFrame): DataFrame with columns 'user_id', 'movie_id', 'rating'
        """
        self.ratings_df = ratings_df
        
        # Encode user and movie IDs
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
        self.user_encoder.fit(ratings_df['user_id'].unique())
        self.movie_encoder.fit(ratings_df['movie_id'].unique())
        
        self.n_users = len(self.user_encoder.classes_)
        self.n_movies = len(self.movie_encoder.classes_)
        
        self.user_ids = self.user_encoder.transform(ratings_df['user_id'].values)
        self.movie_ids = self.movie_encoder.transform(ratings_df['movie_id'].values)
        self.ratings = ratings_df['rating'].values.astype(np.float32)
    
    def __len__(self):
        return len(self.ratings_df)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'movie_id': self.movie_ids[idx],
            'rating': self.ratings[idx]
        }

class NCFModel(nn.Module):
    """Neural Collaborative Filtering model architecture."""
    
    def __init__(self, n_users, n_movies, embedding_dim=32, hidden_layers=[64, 32]):
        """
        Initialize NCF model.
        
        Args:
            n_users (int): Number of unique users
            n_movies (int): Number of unique movies
            embedding_dim (int): Size of embeddings
            hidden_layers (list): List of hidden layer sizes
        """
        super(NCFModel, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        # MLP layers
        self.fc_layers = nn.ModuleList()
        
        # First layer (concatenation of user and movie embeddings)
        self.fc_layers.append(nn.Linear(2 * embedding_dim, hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.fc_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_id, movie_id):
        """Forward pass."""
        # Get embeddings
        user_embedded = self.user_embedding(user_id)
        movie_embedded = self.movie_embedding(movie_id)
        
        # Concatenate embeddings
        vector = torch.cat([user_embedded, movie_embedded], dim=1)
        
        # Apply MLP
        for layer in self.fc_layers:
            vector = self.relu(layer(vector))
        
        # Output prediction (scaled between 1 and 5)
        prediction = 1.0 + 4.0 * self.sigmoid(self.output_layer(vector)).squeeze()
        
        return prediction

class NeuralCollaborativeFiltering(RecommenderBase):
    """
    Neural Collaborative Filtering recommender.
    
    This model uses neural networks to learn user and item embeddings and their interactions.
    """
    
    def __init__(self, embedding_dim=32, hidden_layers=[64, 32], batch_size=256, 
                 lr=0.001, n_epochs=20, device=None):
        """
        Initialize NCF model.
        
        Args:
            embedding_dim (int): Size of embeddings
            hidden_layers (list): List of hidden layer sizes
            batch_size (int): Batch size for training
            lr (float): Learning rate
            n_epochs (int): Number of training epochs
            device (str): Device to use ('cuda' or 'cpu')
        """
        super().__init__(name="Neural-CF")
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model will be initialized in fit
        self.model = None
        self.dataset = None
    
    def fit(self, train_data):
        """
        Train the NCF model.
        
        Args:
            train_data (pd.DataFrame): Training data with columns 'user_id', 'movie_id', 'rating'
        """
        logger.info(f"Training {self.name} model")
        
        # Create dataset
        self.dataset = MovieLensDataset(train_data)
        
        # Initialize model
        self.model = NCFModel(
            n_users=self.dataset.n_users,
            n_movies=self.dataset.n_movies,
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers
        ).to(self.device)
        
        # Create data loader
        train_loader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Training loop
        logger.info(f"Starting training for {self.n_epochs} epochs")
        self.model.train()
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs}")
            
            for batch in progress_bar:
                user_ids = batch['user_id'].to(self.device)
                movie_ids = batch['movie_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(user_ids, movie_ids)
                
                # Calculate loss
                loss = criterion(predictions, ratings)
                epoch_loss += loss.item()
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())
            
            # Log epoch loss
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_epoch_loss:.4f}")
        
        logger.info(f"{self.name} model training completed")
        self.is_fitted = True
    
    def predict(self, user_id, movie_id):
        """
        Predict the rating for a given user-movie pair.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            float: Predicted rating
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert IDs to encoded indices
        try:
            user_idx = self.dataset.user_encoder.transform([user_id])[0]
            movie_idx = self.dataset.movie_encoder.transform([movie_id])[0]
        except:
            # If user or movie not in training set, return default rating
            return 3.0
        
        # Convert to tensors and move to device
        user_tensor = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        movie_tensor = torch.tensor([movie_idx], dtype=torch.long).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(user_tensor, movie_tensor).item()
        
        # Clip prediction to valid range
        prediction = max(1.0, min(5.0, prediction))
        
        return prediction
    
    def predict_batch(self, user_movie_pairs):
        """
        Predict ratings for multiple user-movie pairs.
        
        Args:
            user_movie_pairs (pd.DataFrame): Dataframe with 'user_id' and 'movie_id' columns
            
        Returns:
            np.ndarray: Predicted ratings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Set model to evaluation mode
        self.model.eval()
        
        predictions = []
        
        # Process in batches for efficiency
        for i in range(0, len(user_movie_pairs), self.batch_size):
            batch = user_movie_pairs.iloc[i:i+self.batch_size]
            batch_predictions = []
            
            for _, row in batch.iterrows():
                try:
                    user_id = row['user_id']
                    movie_id = row['movie_id']
                    pred = self.predict(user_id, movie_id)
                    batch_predictions.append(pred)
                except:
                    # Default prediction if error
                    batch_predictions.append(3.0)
            
            predictions.extend(batch_predictions)
        
        return np.array(predictions)