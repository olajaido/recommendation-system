import logging
import sys
from datetime import datetime

def setup_logger(name, log_level=logging.INFO):
    """
    Set up a logger with a specified name and log level.
    
    Args:
        name: The name of the logger
        log_level: The logging level (default: INFO)
        
    Returns:
        A configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create console handler and set level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatter = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to console handler
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    return logger