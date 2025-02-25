import os
import yaml
import zipfile
import requests
from pathlib import Path
import shutil
from tqdm import tqdm
import sys

# Add the project root to the Python path to import from src
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_config(config_path="configs/data_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def download_file(url, destination):
    """
    Download a file from a URL to a destination with a progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
    """
    logger.info(f"Downloading from {url} to {destination}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Stream download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            file.write(data)
            bar.update(len(data))
    
    logger.info(f"Download completed: {destination}")

def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a destination folder.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
    """
    logger.info(f"Extracting {zip_path} to {extract_to}")
    
    # Create extraction directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info(f"Extraction completed to {extract_to}")

def download_movielens_dataset():
    """Download and extract the MovieLens dataset."""
    config = load_config()
    movielens_config = config["movielens"]
    
    dataset_size = movielens_config["dataset_size"]
    url = movielens_config["urls"][dataset_size]
    raw_dir = movielens_config["paths"]["raw"]
    
    # Create directories
    os.makedirs(raw_dir, exist_ok=True)
    
    # Define file paths
    zip_filename = f"ml-{dataset_size}.zip"
    zip_path = os.path.join(raw_dir, zip_filename)
    
    # Download the dataset
    download_file(url, zip_path)
    
    # Extract the dataset
    extract_zip(zip_path, raw_dir)
    
    # Get the extracted folder name
    extracted_folder = os.path.join(raw_dir, f"ml-{dataset_size}")
    
    if os.path.exists(extracted_folder):
        logger.info(f"Successfully downloaded and extracted MovieLens {dataset_size} dataset")
    else:
        logger.error(f"Extraction failed or folder structure is unexpected: {extracted_folder}")

if __name__ == "__main__":
    logger.info("Starting MovieLens dataset download")
    download_movielens_dataset()
    logger.info("Download process completed")