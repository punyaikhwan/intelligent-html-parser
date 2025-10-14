"""
Settings configuration for Intelligent HTML Parser.
This module loads all environment variables and provides default values.
"""
import os
from typing import Union, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


class Settings:
    """Configuration settings for the Intelligent HTML Parser application."""
    
    # Flask Application Settings
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = int(os.getenv('PORT', '5000'))
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() in ('true', '1', 'yes', 'on')
    MAX_CONTENT_LENGTH: int = int(os.getenv('MAX_CONTENT_LENGTH', str(16 * 1024 * 1024)))  # 16MB default
    JSON_SORT_KEYS: bool = os.getenv('JSON_SORT_KEYS', 'False').lower() in ('true', '1', 'yes', 'on')
    
    # ML Model Settings
    ML_MODEL_NAME: str = os.getenv('ML_MODEL_NAME', 'models/llm/flan-t5-small-tuned-v1')
    SIMILARITY_MODEL: str = os.getenv('SIMILARITY_MODEL', 'models/sentence-transformers/all-MiniLM-L6-v2-tuned-v1')
    SIMILARITY_THRESHOLD: float = float(os.getenv('SIMILARITY_THRESHOLD', '0.6'))
    MIN_ATTRIBUTES: int = int(os.getenv('MIN_ATTRIBUTES', '2'))
    
    # Hugging Face Settings
    HF_HUB_DISABLE_SYMLINKS_WARNING: bool = os.getenv('HF_HUB_DISABLE_SYMLINKS_WARNING', 'True').lower() in ('true', '1', 'yes', 'on')
    HUGGINGFACE_HUB_CACHE: Optional[str] = os.getenv('HUGGINGFACE_HUB_CACHE')
    TRANSFORMERS_CACHE: Optional[str] = os.getenv('TRANSFORMERS_CACHE')
    
    # Performance Settings
    TORCH_NUM_THREADS: Optional[int] = int(os.getenv('TORCH_NUM_THREADS', '0')) if os.getenv('TORCH_NUM_THREADS') else None
    OMP_NUM_THREADS: Optional[int] = int(os.getenv('OMP_NUM_THREADS', '0')) if os.getenv('OMP_NUM_THREADS') else None
    
    # Logging Settings
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO').upper()
    LOG_FORMAT: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Parser Strategy Settings
    ENABLE_JSON_SCRIPT_PARSER: bool = os.getenv('ENABLE_JSON_SCRIPT_PARSER', 'True').lower() in ('true', '1', 'yes', 'on')
    ENABLE_TABLE_PARSER: bool = os.getenv('ENABLE_TABLE_PARSER', 'True').lower() in ('true', '1', 'yes', 'on')
    ENABLE_GENERAL_PARSER: bool = os.getenv('ENABLE_GENERAL_PARSER', 'True').lower() in ('true', '1', 'yes', 'on')
    ENABLE_ML_FALLBACK: bool = os.getenv('ENABLE_ML_FALLBACK', 'True').lower() in ('true', '1', 'yes', 'on')
    
    # Request/Response Settings
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '30'))  # seconds
    MAX_RESULTS_PER_QUERY: int = int(os.getenv('MAX_RESULTS_PER_QUERY', '100'))
    
    # Development/Testing Settings
    FLASK_ENV: str = os.getenv('FLASK_ENV', 'production')
    TESTING: bool = os.getenv('TESTING', 'False').lower() in ('true', '1', 'yes', 'on')
    
    # Security Settings
    SECRET_KEY: Optional[str] = os.getenv('SECRET_KEY')
    CORS_ORIGINS: str = os.getenv('CORS_ORIGINS', '*')
    
    # Database/Cache Settings (for future use)
    REDIS_URL: Optional[str] = os.getenv('REDIS_URL')
    DATABASE_URL: Optional[str] = os.getenv('DATABASE_URL')
    
    # API Rate Limiting (for future use)
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '100'))
    
    @classmethod
    def get_flask_config(cls) -> dict:
        """Get Flask-specific configuration dictionary."""
        return {
            'MAX_CONTENT_LENGTH': cls.MAX_CONTENT_LENGTH,
            'JSON_SORT_KEYS': cls.JSON_SORT_KEYS,
            'SECRET_KEY': cls.SECRET_KEY or 'dev-secret-key-change-in-production',
            'TESTING': cls.TESTING,
        }
    
    @classmethod
    def get_parser_config(cls) -> dict:
        """Get parser-specific configuration dictionary."""
        return {
            'ml_model_name': cls.ML_MODEL_NAME,
            'similarity_model': cls.SIMILARITY_MODEL,
            'similarity_threshold': cls.SIMILARITY_THRESHOLD,
            'min_attributes': cls.MIN_ATTRIBUTES,
        }
    
    @classmethod
    def get_logging_config(cls) -> dict:
        """Get logging configuration dictionary."""
        return {
            'level': cls.LOG_LEVEL,
            'format': cls.LOG_FORMAT,
        }
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode."""
        return cls.FLASK_ENV.lower() == 'development' or cls.DEBUG
    
    @classmethod
    def is_production(cls) -> bool:
        """Check if running in production mode."""
        return cls.FLASK_ENV.lower() == 'production' and not cls.DEBUG
    
    @classmethod
    def validate_settings(cls) -> bool:
        """Validate critical settings and return True if valid."""
        if cls.PORT < 1 or cls.PORT > 65535:
            raise ValueError(f"Invalid PORT: {cls.PORT}. Must be between 1 and 65535.")
        
        if cls.SIMILARITY_THRESHOLD < 0 or cls.SIMILARITY_THRESHOLD > 1:
            raise ValueError(f"Invalid SIMILARITY_THRESHOLD: {cls.SIMILARITY_THRESHOLD}. Must be between 0 and 1.")
        
        if cls.MIN_ATTRIBUTES < 1:
            raise ValueError(f"Invalid MIN_ATTRIBUTES: {cls.MIN_ATTRIBUTES}. Must be at least 1.")
        
        if cls.REQUEST_TIMEOUT < 1:
            raise ValueError(f"Invalid REQUEST_TIMEOUT: {cls.REQUEST_TIMEOUT}. Must be at least 1 second.")
        
        return True


# Create a singleton instance
settings = Settings()

# Validate settings on import
try:
    settings.validate_settings()
except ValueError as e:
    print(f"Settings validation error: {e}")
    raise

# Convenience exports
__all__ = [
    'settings',
    'Settings',
]