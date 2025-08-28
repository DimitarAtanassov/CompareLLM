# app/backend/core/exceptions.py
from typing import Any, Dict, Optional


class AskManyLLMsException(Exception):
    """Base exception for the application."""
    
    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)


class ModelNotFoundError(AskManyLLMsException):
    """Raised when a requested model is not available."""
    
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found",
            code="MODEL_NOT_FOUND",
            details={"model_name": model_name}
        )


class ProviderError(AskManyLLMsException):
    """Raised when a provider call fails."""
    
    def __init__(self, provider_name: str, original_error: str):
        super().__init__(
            message=f"Provider '{provider_name}' error: {original_error}",
            code="PROVIDER_ERROR",
            details={"provider_name": provider_name, "original_error": original_error}
        )


class DatasetNotFoundError(AskManyLLMsException):
    """Raised when a dataset is not found."""
    
    def __init__(self, dataset_id: str):
        super().__init__(
            message=f"Dataset '{dataset_id}' not found",
            code="DATASET_NOT_FOUND",
            details={"dataset_id": dataset_id}
        )


class ValidationError(AskManyLLMsException):
    """Raised when validation fails."""
    
    def __init__(self, field: str, message: str):
        super().__init__(
            message=f"Validation error for '{field}': {message}",
            code="VALIDATION_ERROR",
            details={"field": field}
        )