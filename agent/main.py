import logging

from agent_config import load_env
from tradesystem import TradingSystem
from report_generate import ReportGenerator


def main():
    """
    Entry point for the trading system pipeline:
    1. Load environment variables and configuration
    2. Instantiate the TradingSystem orchestrator
    3. Execute the full pipeline: data load → preprocess → feature engineering → analysis → forecasting → strategy → portfolio simulation → report generation
    4. Generate final DOCX report
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
    trading_system = TradingSystem(config)

    logger.info("Running full trading pipeline...")
    # Kick off the full pipeline including data, features, analysis, forecasting, strategy, and portfolio
    results = trading_system.run()

    logger.info("Trading pipeline completed successfully.")

    # Generate DOCX report using existing module
    logger.info("Generating DOCX report...")
    report_generator = ReportGenerator(config, config.output_dir)
    report_path = report_generator.generate(results)
    logger.info(f"Report saved to {report_path}")


def run_main():
    # alias for __main__ to facilitate testing
    main()


if __name__ == "__main__":
    main()
