import sys
import logging

# Set up logging configuration
logging.basicConfig(
    format="%(asctime)s:%(name)s:%(levelname)-8s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
    datefmt="%s"
    )