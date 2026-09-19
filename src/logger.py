import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file name with current timestamp
LOG_FILE = datetime.now().strftime("%Y_%m_%d_%H_%M_%S.log")

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, LOG_FILE),
    level=logging.INFO,
    format="[ %(asctime)s ] %(levelname)s %(message)s",
)

# Logger object
logger = logging.getLogger(__name__)