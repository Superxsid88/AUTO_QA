import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from colorlog import ColoredFormatter

# ==============================
# Log Configuration
# ==============================

LOG_FORMAT = "%(log_color)s[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

COLOR_CONFIG = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

# ==============================
# Logger Initialization
# ==============================

logger = logging.getLogger("ASRService")
logger.setLevel(logging.DEBUG)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# ==============================
# Console Handler (Color)
# ==============================

console_handler = logging.StreamHandler(sys.stdout)
console_formatter = ColoredFormatter(
    LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    log_colors=COLOR_CONFIG
)
console_handler.setFormatter(console_formatter)

# ==============================
# Rotating File Handler
# ==============================

file_handler = RotatingFileHandler(
    filename="logs/asr_service.log",
    maxBytes=5_000_000,
    backupCount=5
)
file_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
    datefmt=LOG_DATE_FORMAT
)
file_handler.setFormatter(file_formatter)

# ==============================
# Attach Handlers Only Once
# ==============================

if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ==============================
# Suppress Noisy Loggers
# ==============================

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

