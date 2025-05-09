import logging
import os
import sys
from dotenv import load_dotenv, find_dotenv
from tradingsystem import TradingSystem

def load_config():
    """Load configuration from environment variables."""
    env_path = find_dotenv("fin580.env", raise_error_if_not_found=False)
    if env_path:
        load_dotenv(env_path)
    else:
        logging.warning("fin580.env not found, using default environment")
        load_dotenv()
    
    # Build configuration dictionary
    config = {
        'start_date': os.getenv('START_DATE', '2023-01-01'), # default time range, if not specified
        'end_date': os.getenv('END_DATE', '2023-12-31'),
        'model_type': os.getenv('MODEL_TYPE', 'gru'),
        'sentiment_method': os.getenv('SENTIMENT_METHOD', 'transformer'),
        'min_holding': int(os.getenv('MIN_HOLDING_PERIOD', '5')),
        'max_trades': int(os.getenv('MAX_TRADES_PER_WEEK', '3')),
        'initial_capital': float(os.getenv('INITIAL_CAPITAL', '100000.0')),
        'output_dir': os.getenv('OUTPUT_DIR', 'agent/output'),
        'reddit_subreddit': os.getenv('REDDIT_SUBREDDIT', 'CryptoCurrency'),
        'sentiment_threshold': float(os.getenv('SENTIMENT_THRESHOLD', '0.3')),
        'sentiment_change_threshold': float(os.getenv('SENTIMENT_CHANGE_THRESHOLD', '0.1')),
        'num_stocks': int(os.getenv('NUM_STOCKS', '5')),
        'sentiment_weight': float(os.getenv('SENTIMENT_WEIGHT', '0.6')),
        'technical_weight': float(os.getenv('TECHNICAL_WEIGHT', '0.4'))
    }
    
    return config

def main():
    """Entry point for the trading system pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_config()

    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize and run trading system
    logger.info("Initializing and running trading system...")
    try:
        trading_system = TradingSystem(config)
        results = trading_system.run()
        logger.info("Trading pipeline completed successfully.")
        return results
    
    except Exception as e:
        logger.error(f"Error in trading pipeline: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()