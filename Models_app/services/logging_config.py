import logging
import json
import os
import contextvars
from pathlib import Path
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/app.log")

# Context variable to store `image_id` dynamically
image_id_var = contextvars.ContextVar("image_id", default=None)

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter that automatically adds image_id from context."""
    def format(self, record):
        record.image_id = image_id_var.get()  # Fetch image_id dynamically
        log_data = {
            "image_id": record.image_id ,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        return json.dumps(log_data,indent=4)

def setup_logger(logger_name, log_file, level=logging.INFO):
    """Set up a structured logger with JSON format and rotating file handler."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    logger = logging.getLogger(logger_name)
    
    if logger.hasHandlers():
        return logger  # Prevent duplicate handlers

    formatter = JSONFormatter()
    handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
    handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

app_logger = setup_logger('app_logger', LOG_FILE_PATH)
