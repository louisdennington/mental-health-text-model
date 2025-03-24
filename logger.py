# Custom logging setup. For example, logging errors or user inputs to a file or console in a structured way.

# logger.py

import logging

# Create and configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
