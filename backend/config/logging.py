import json
import logging
import sys
from typing import Any, Dict

from config.settings import get_settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
            
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging() -> None:
    """Configure application logging."""
    settings = get_settings()
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)
    
    # Application logger
    app_logger = logging.getLogger("askmanyllms")
    app_logger.setLevel(getattr(logging, settings.log_level.upper()))


def log_event(kind: str, **fields: Any) -> None:
    """Log structured events."""
    logger = logging.getLogger("askmanyllms")
    logger.info("", extra={"extra_fields": {"event": kind, **fields}})