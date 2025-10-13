"""
Example usage of settings.py in different parts of the application.
This file demonstrates how to import and use settings across the project.
"""

# Example 1: Basic import and usage
from settings import settings

def example_flask_setup():
    """Example of using settings for Flask configuration."""
    from flask import Flask
    
    app = Flask(__name__)
    
    # Apply all Flask-related settings at once
    app.config.update(settings.get_flask_config())
    
    # Or access individual settings
    app.config['MAX_CONTENT_LENGTH'] = settings.MAX_CONTENT_LENGTH
    
    return app


def example_parser_initialization():
    """Example of using settings for parser initialization."""
    from html_parser import IntelligentHTMLParser
    
    # Get parser configuration from settings
    parser_config = settings.get_parser_config()
    
    # Initialize parser with settings
    parser = IntelligentHTMLParser(**parser_config)
    
    return parser


def example_logging_setup():
    """Example of using settings for logging configuration."""
    import logging
    
    # Get logging configuration from settings
    log_config = settings.get_logging_config()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_config['level']),
        format=log_config['format']
    )
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.info("Logging configured from settings")
    
    return logger


def example_conditional_logic():
    """Example of using settings for conditional logic."""
    
    # Check environment
    if settings.is_development():
        print("Running in development mode")
        # Enable extra debugging, verbose logging, etc.
    elif settings.is_production():
        print("Running in production mode")
        # Disable debugging, optimize performance, etc.
    
    # Check feature flags
    if settings.ENABLE_ML_FALLBACK:
        print("ML fallback is enabled")
    
    if settings.ENABLE_JSON_SCRIPT_PARSER:
        print("JSON script parser is enabled")


def example_environment_specific_config():
    """Example of accessing environment-specific configurations."""
    
    # Performance settings
    if settings.TORCH_NUM_THREADS:
        import torch
        torch.set_num_threads(settings.TORCH_NUM_THREADS)
    
    # Cache settings
    if settings.TRANSFORMERS_CACHE:
        import os
        os.environ['TRANSFORMERS_CACHE'] = settings.TRANSFORMERS_CACHE
    
    # API settings
    timeout = settings.REQUEST_TIMEOUT
    max_results = settings.MAX_RESULTS_PER_QUERY
    
    print(f"Request timeout: {timeout}s")
    print(f"Max results per query: {max_results}")


# Example 2: Import specific classes/instances
from settings import Settings

def example_custom_settings():
    """Example of creating custom settings instance."""
    
    # You can create a custom settings instance if needed
    custom_settings = Settings()
    
    # Validate the settings
    try:
        custom_settings.validate_settings()
        print("Custom settings are valid")
    except ValueError as e:
        print(f"Invalid settings: {e}")


# Example 3: Using settings in different modules
def example_in_parser_module():
    """Example usage in a parser module."""
    from settings import settings
    
    # Use similarity threshold from settings
    similarity_threshold = settings.SIMILARITY_THRESHOLD
    
    # Use model name from settings
    model_name = settings.SIMILARITY_MODEL
    
    print(f"Using model: {model_name} with threshold: {similarity_threshold}")


def example_in_api_module():
    """Example usage in an API module."""
    from settings import settings
    
    # Use rate limiting settings
    rate_limit = settings.RATE_LIMIT_PER_MINUTE
    
    # Use CORS settings
    cors_origins = settings.CORS_ORIGINS
    
    print(f"Rate limit: {rate_limit}/min, CORS origins: {cors_origins}")


if __name__ == "__main__":
    print("Settings Usage Examples")
    print("=" * 50)
    
    example_conditional_logic()
    example_environment_specific_config()
    example_custom_settings()
    example_in_parser_module()
    example_in_api_module()