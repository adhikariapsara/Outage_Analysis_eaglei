 #!/usr/bin/env python3
"""
main.py - EAGLE-i Outage Data Processing Pipeline

This is the main entry point for the EAGLE-i outage and weather data processing pipeline.
It provides a comprehensive interface to execute various analysis tasks with configurable
parameters and logging.

Usage:
    python main.py --help
    python main.py --mode full --config config.json
    python main.py --mode outage-only --state Florida --county "Miami-Dade"

License: MIT
"""

import argparse
import json
import logging
import os
import sys

from src.pipelineManager import PipelineManager
from src.configManager import ConfigManager


# ==================== Constants ====================
# DEFAULT_CONFIG = "config.json"
LOG_DIR = "logs"
LOG_FILE = "eaglei_dataprocessor.log"


# ==================== Logging Setup ====================
def setup_logging(log_file: str = LOG_FILE, log_level: str = "INFO", console_output: bool = True) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_file: Path to the log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Whether to output logs to the console
    
    Returns:
        Configured logger instance
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, log_file)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout) if console_output else logging.NullHandler()
        ]
    )
    return logging.getLogger("eagleiLogger")

def update_logging_config(logger: logging.Logger, config: ConfigManager) -> None:
    """Update logging configuration at runtime."""
    
    # Update logging level based on config
    logger.setLevel(config.get("logging.log_level", "INFO").upper())
    for handler in logger.handlers:
        handler.setLevel(config.get("logging.log_level", "INFO").upper())
    
    # Enable or disable console logging based on config
    console_logging = config.get("logging.enable_console_logging", True)
    if console_logging:
        logger.info("Console logging enabled.")
    else:
        logger.info("Console logging disabled.")
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.NOTSET if console_logging else logging.CRITICAL)

# Initialize global logger with default settings
logger = setup_logging()



# ==================== CLI Argument Parser ====================
def create_parser() -> argparse.ArgumentParser:
    """Create and configure command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="EAGLE-i Data Processing Pipeline - Process and analyze outage and weather data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with the default config file
  python main.py --mode full

  # Run outage processing for a specific state and county
  python main.py --mode outage-only --state "Florida" --county "Miami-Dade"

  # Run weather processing with custom year range
  python main.py --mode weather-only --start-year 2020 --end-year 2023

  # Align/merge data only
  python main.py --mode alignment-only

  # Show current configuration
  python main.py --show-config
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['full', 'outage-only', 'weather-only', 'alignment-only', 'multi-county'],
        default='full',
        help='Pipeline execution mode (default: full)'
    )
    
    parser.add_argument(
        '--state',
        help='State name (overrides config file)'
    )
    
    parser.add_argument(
        '--county',
        help='County name (overrides config file)'
    )
    
    parser.add_argument(
        '--start-year',
        type=int,
        help='Start year for analysis (overrides config file)'
    )
    
    parser.add_argument(
        '--end-year',
        type=int,
        help='End year for analysis (overrides config file)'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Display current configuration and exit'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    return parser


# ==================== Main Entry Point ====================
def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging with specified level
    global logger
    
    try:
        # Load and configure settings
        config = ConfigManager()
        config.update_from_args(args)

        # Update logging configuration based on loaded config
        update_logging_config(logger, config)
        
        # Show configuration if requested
        if args.show_config:
            logger.info("\nCurrent Configuration:")
            logger.info(json.dumps(config.to_dict(), indent=2))
            return 0
        
        # Validate required parameters
        params = config.config["processing_parameters"]
        if not params.get("state") or not params.get("county"):
            logger.error("Error: State and county are required parameters")
            return 1
        
        # Execute pipeline
        pipeline = PipelineManager(config)
        
        success = False
        if args.mode == "full":
            success = pipeline.run_full_pipeline()
        elif args.mode == "outage-only":
            success = pipeline.run_outage_only()
        elif args.mode == "weather-only":
            success = pipeline.run_weather_only()
        elif args.mode == "alignment-only":
            success = pipeline.run_alignment_only()
        elif args.mode == "multi-county":
            success = pipeline.run_multi_county_analysis()
        return 0 if success else 1
    
    except KeyboardInterrupt:
        logger.warning("\nExecution cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
