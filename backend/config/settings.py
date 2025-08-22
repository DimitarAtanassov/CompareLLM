from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    
    # Logging
    log_level: str = "INFO"
    
    # Model configuration
    models_config_path: str = Field("/config/models.yaml", env="MODELS_CONFIG")
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    deepseek_api_key: Optional[str] = Field(None, env="DEEPSEEK_API_KEY")
    google_api_key: Optional[str] = Field(None, env="GOOGLE_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    cohere_api_key: Optional[str] = Field(None, env="COHERE_API_KEY")
    voyage_api_key: Optional[str] = Field(None, env="VOYAGE_API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()