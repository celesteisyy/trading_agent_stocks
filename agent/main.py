import logging

from agent_config import load_env
from tradesystem import TradingSystem


def main():
    """
    Entry point for the trading system pipeline:
    1. Load environment variables and API keys
    2. Instantiate the TradingSystem orchestrator
    3. Execute the full pipeline: data load → preprocess → feature engineering → analysis → forecasting → strategy → portfolio simulation → report generation
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading environment variables and configuration...")
    config = load_env()

    logger.info("Initializing TradingSystem...")
    # Initialize TradingSystem with config (API keys, paths, hyperparameters)
    trading_system = TradingSystem(config)

    logger.info("Running full trading pipeline...")
    # Kick off the full pipeline
    trading_system.run()

    logger.info("Trading pipeline completed successfully.")


if __name__ == "__main__":
    main()