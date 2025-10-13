import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config

def get_training_args(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training arguments from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with training arguments
    """
    # Helper function to convert string numbers to appropriate types
    def convert_numeric(value, default):
        if isinstance(value, str):
            try:
                # Try to convert to float first
                return float(value)
            except ValueError:
                return default
        return value if value is not None else default
    
    return {
        'train_file': config.get('train_file', 'training_data.json'),
        'model_name': config.get('model_name', 'google/flan-t5-small'),
        'output_dir': config.get('output_dir', './flan-t5-html-parser'),
        'num_epochs': int(convert_numeric(config.get('num_epochs'), 5)),
        'batch_size': int(convert_numeric(config.get('batch_size'), 4)),
        'learning_rate': convert_numeric(config.get('learning_rate'), 3e-4),
        'warmup_steps': int(convert_numeric(config.get('warmup_steps'), 100)),
        'weight_decay': convert_numeric(config.get('weight_decay'), 0.01),
        'max_input_length': int(convert_numeric(config.get('max_input_length'), 2000)),
        'max_output_length': int(convert_numeric(config.get('max_output_length'), 256)),
        'use_kfold': config.get('use_kfold', False),
        'kfold_splits': int(convert_numeric(config.get('kfold_splits'), 5)),
        'validation_split': convert_numeric(config.get('validation_split'), 0.2),
        'eval_steps': int(convert_numeric(config.get('eval_steps'), 100)),
        'save_steps': int(convert_numeric(config.get('save_steps'), 100)),
        'save_total_limit': int(convert_numeric(config.get('save_total_limit'), 3)),
        'metric_for_best_model': config.get('metric_for_best_model', 'rougeL'),
        'shuffle_data': config.get('shuffle_data', True),
        'max_samples': config.get('max_samples', None),
        'num_beams': int(convert_numeric(config.get('num_beams'), 4)),
        'temperature': convert_numeric(config.get('temperature'), 0.0),
        'top_p': convert_numeric(config.get('top_p'), 0.9),
        'early_stopping': config.get('early_stopping', True),
        'use_fp16': config.get('use_fp16', True),
        'gradient_accumulation_steps': int(convert_numeric(config.get('gradient_accumulation_steps'), 1)),
        'logging_steps': int(convert_numeric(config.get('logging_steps'), 100)),
        'report_to': config.get('report_to', ['tensorboard']),
        'lr_scheduler_type': config.get('lr_scheduler_type', 'linear'),
        'optimizer': config.get('optimizer', 'adamw'),
        'dataloader_pin_memory': config.get('dataloader_pin_memory', False)
    }