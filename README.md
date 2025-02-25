# Movie Recommendation System

An end-to-end MLOps pipeline for movie recommendations using collaborative filtering, neural networks, and hybrid approaches.

## Overview

This project implements a complete machine learning operations (MLOps) pipeline for movie recommendations. It demonstrates the evolution from simple collaborative filtering to sophisticated neural network and hybrid approaches, with a focus on production-ready features like model monitoring, explanation capabilities, and batch processing.

## Features

- **Multiple Recommendation Models**:
  - User-Based Collaborative Filtering
  - Matrix Factorization with SVD
  - Neural Collaborative Filtering
  - Hybrid Model (combining collaborative and content-based approaches)

- **End-to-End MLOps Pipeline**:
  - Data ingestion and preprocessing
  - Model training and evaluation
  - Real-time and batch recommendation serving
  - Data drift monitoring
  - Containerization for deployment

- **Production Features**:
  - Recommendation explanations ("Why was this recommended?")
  - Diversity controls for recommendations
  - Performance metrics and model comparison
  - Response caching for improved performance
  - Web-based user interface

## System Architecture

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Pipeline  │────▶│ Model Training  │────▶│  Model Registry │
│                 │     │   & Evaluation  │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
│
▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Drift     │     │ Batch Recommender│────▶│  Recommendation │
│  Monitoring     │     │                 │     │     Storage     │
└────────┬────────┘     └─────────────────┘     └────────┬────────┘
│                                               │
│            ┌─────────────────┐                │
└───────────▶│  FastAPI Server │◀───────────────┘
│    & Frontend   │
└────────┬────────┘
▲
│
┌───────┴───────┐
│     Users     │
└───────────────┘

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/recommendation-system.git
cd recommendation-system

# Create virtual environment
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate
# Activate on macOS/Linux
source .venv/bin/activate

#Install dependencies:
pip install -r requirements.txt

#Download the dataset:
python -m src.data.download

#Preprocess the data:
python -m src.data.preprocess

# Train the baseline collaborative filtering model
python -m src.training.train

# Train the neural collaborative filtering model
python -m src.training.train_advanced

# Train the hybrid model
python -m src.training.train_hybrid

##Running the API Server
# Start the recommendation API
uvicorn src.serving.enhanced_api:app --reload

The API will be available at http://localhost:8000.
You can explore the API documentation at http://localhost:8000/docs

# Start the web frontend
python -m src.frontend.server

The web interface will be available at http://localhost:8080.

# Generate recommendations for all users
python -m src.batch.recommendation_generator --model hybrid_recommender --n 20

# Start the data drift monitoring service
python -m src.monitoring.drift_monitor

##Docker Deployment
You can also run the entire system using Docker:

# Build and start all services
docker-compose up -d

# Run just the API
docker-compose up api


# Recommendation System

## Available Models

### User-Based Collaborative Filtering
A baseline model that recommends items based on preferences of similar users.
- **Performance**: RMSE ≈ 1.01

### Matrix Factorization
Uses Singular Value Decomposition (SVD) to discover latent factors for users and items.
- **Performance**: RMSE ≈ 0.99

### Neural Collaborative Filtering
A deep learning approach combining embeddings with neural networks for better recommendations.
- **Performance**: RMSE ≈ 0.98

### Hybrid Recommender
Combines collaborative filtering with content-based features from movie metadata, offering better cold-start handling.
- **Performance**: RMSE ≈ 0.99 with improved explainability

## API Endpoints

- `GET /` - API status and information
- `GET /models` - List available recommendation models
- `POST /recommendations` - Get real-time recommendations for a user
- `GET /batch-recommendations/{user_id}` - Get pre-computed recommendations
- `POST /explain` - Get explanation for a recommendation
- `GET /metrics` - Get model performance metrics

## Development

### Adding a New Model

1. Create a new model file in `src/models/`
2. Implement the `RecommenderBase` interface
3. Add the model to the training pipeline in `src/training/`
4. Update the API to include the new model


## CI/CD Pipeline

The project utilizes GitHub Actions for continuous integration and deployment. Detailed workflow configurations can be found in the `.github/workflows/ci-cd.yml` file, which automates testing, building, and deployment processes.

## Performance Metrics

We've implemented and compared multiple recommendation system models with the following performance metrics:

| Model | RMSE | MAE |
|-------|------|-----|
| User-Based CF | 1.0148 | 0.8063 |
| Matrix Factorization | 0.9956 | 0.7822 |
| Neural Collaborative Filter | 0.9777 | 0.7672 |
| Hybrid Recommender | 0.9942 | 0.7803 |

*Note: Lower RMSE and MAE values indicate better model performance.*

## License

This project is licensed under the MIT License. For complete licensing details, please refer to the `LICENSE` file in the repository.

## Acknowledgments

The development of this recommendation system was made possible by:
- MovieLens dataset provided by GroupLens Research
- Insights from various recommendation system research papers
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [PyTorch](https://pytorch.org/) for neural network implementation

## Contributing

We welcome contributions! If you'd like to improve the project:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

- **Your Name**: your.email@example.com
- **Project Link**: [GitHub Repository](https://github.com/yourusername/recommendation-system)

## Support

If you encounter any issues or have questions, please [open an issue](https://github.com/yourusername/recommendation-system/issues) on the GitHub repository.