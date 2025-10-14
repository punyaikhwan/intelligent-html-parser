#!/usr/bin/env python3
"""
Fine-tuning Sentence Transformer dengan Triplet Loss
untuk meningkatkan similarity score pada kasus spesifik
"""

import json
import logging
import torch
import pandas as pd
import yaml
import random
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Any
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

def setup_logging(config: Dict[str, Any]):
    """Setup logging based on configuration"""
    log_level = getattr(logging, config['logging']['level'].upper())
    
    if config['logging']['log_to_file']:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config['logging']['log_file']),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

class TripletTrainer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize triplet trainer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        model_name = config['model']['name']
        self.model = SentenceTransformer(model_name)
        
        # Set device
        if config['advanced']['use_cuda_if_available'] and torch.cuda.is_available():
            self.model = self.model.to('cuda')
            logger.info("Using CUDA for training")
        else:
            logger.info("Using CPU for training")
            
        logger.info(f"Loaded model: {model_name}")
    
    def load_triplets_from_file(self, file_path: str) -> List[InputExample]:
        """
        Load triplet data from JSON file
        
        Args:
            file_path: Path to triplet data file
            
        Returns:
            List of InputExample objects for training
        """
        examples = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                anchor = item['anchor']
                positive = item['positive']
                negative = item['negative']
                
                # Create InputExample for triplet loss
                example = InputExample(texts=[anchor, positive, negative])
                examples.append(example)
            
            logger.info(f"Loaded {len(examples)} triplet examples from {file_path}")
            return examples
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            return []
        except KeyError as e:
            logger.error(f"Missing required key in triplet data: {e}")
            return []
    
    def create_evaluation_data(self, triplets: List[InputExample]) -> Tuple[List[InputExample], List[InputExample]]:
        """
        Split triplets into training and evaluation sets
        
        Args:
            triplets: List of triplet examples
            
        Returns:
            Tuple of (training_examples, evaluation_examples)
        """
        split_ratio = self.config['data']['train_split']
        split_idx = int(len(triplets) * split_ratio)
        train_examples = triplets[:split_idx]
        eval_examples = triplets[split_idx:]
        
        logger.info(f"Training examples: {len(train_examples)}")
        logger.info(f"Evaluation examples: {len(eval_examples)}")
        
        return train_examples, eval_examples
    
    def prepare_evaluator(self, eval_examples: List[InputExample]) -> TripletEvaluator:
        """
        Prepare triplet evaluator for monitoring training progress
        
        Args:
            eval_examples: Evaluation triplet examples
            
        Returns:
            TripletEvaluator object
        """
        anchors = []
        positives = []
        negatives = []
        
        for example in eval_examples:
            anchors.append(example.texts[0])
            positives.append(example.texts[1])
            negatives.append(example.texts[2])
        
        evaluator = TripletEvaluator(anchors, positives, negatives, name='triplet_eval')
        return evaluator
    
    def train(self):
        """
        Train the model with triplet loss using configuration parameters
        """
        # Get configuration
        data_config = self.config['data']
        train_config = self.config['training']
        output_config = self.config['output']
        eval_config = self.config['evaluation']
        
        # Load triplet data
        triplets = self.load_triplets_from_file(data_config['triplet_file'])
        if not triplets:
            logger.error("No triplet data loaded. Training aborted.")
            return
        
        # Split data
        train_examples, eval_examples = self.create_evaluation_data(triplets)
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=train_config['batch_size'],
            num_workers=self.config['advanced']['num_workers']
        )
        
        # Define loss function
        train_loss = losses.TripletLoss(model=self.model, triplet_margin=train_config['margin'])
        
        # Prepare evaluator
        evaluator = self.prepare_evaluator(eval_examples) if eval_examples and eval_config['run_evaluation'] else None
        
        # Prepare output path
        output_path = output_config['model_path']
        if output_config['add_timestamp']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{output_path}_{timestamp}"
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Training arguments
        total_steps = len(train_dataloader) * train_config['epochs']
        logger.info(f"Total training steps: {total_steps}")
        
        # Log training configuration
        logger.info("Training Configuration:")
        for key, value in train_config.items():
            logger.info(f"  {key}: {value}")
        
        # Start training
        logger.info("Starting training...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=train_config['epochs'],
            warmup_steps=train_config['warmup_steps'],
            optimizer_params={'lr': train_config['learning_rate']},
            evaluation_steps=train_config['evaluation_steps'],
            evaluator=evaluator,
            output_path=output_path,
            save_best_model=output_config['save_best_model']
        )
        
        logger.info(f"Training completed. Model saved to: {output_path}")
        return output_path
    
    def evaluate_similarity(self, sentence1: str, sentence2: str) -> float:
        """
        Evaluate similarity between two sentences
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            
        Returns:
            Similarity score (cosine similarity)
        """
        embeddings = self.model.encode([sentence1, sentence2])
        # Calculate cosine similarity using torch
        embedding1 = torch.tensor(embeddings[0])
        embedding2 = torch.tensor(embeddings[1])
        similarity = torch.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
        return float(similarity.item())
    
    def test_model_improvements(self, test_cases: List[Tuple[str, str, str]]):
        """
        Test model improvements with before/after comparisons
        
        Args:
            test_cases: List of (anchor, positive, negative) tuples
        """
        logger.info("Testing model improvements...")
        
        for i, (anchor, positive, negative) in enumerate(test_cases):
            pos_sim = self.evaluate_similarity(anchor, positive)
            neg_sim = self.evaluate_similarity(anchor, negative)
            
            logger.info(f"Test case {i+1}:")
            logger.info(f"  Anchor: {anchor}")
            logger.info(f"  Positive: {positive} (similarity: {pos_sim:.4f})")
            logger.info(f"  Negative: {negative} (similarity: {neg_sim:.4f})")
            logger.info(f"  Margin: {pos_sim - neg_sim:.4f}")
            logger.info("-" * 80)

def main():
    """Main training function"""
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config)
    
    # Set random seed
    set_seed(config['advanced']['seed'])
    
    # Initialize trainer
    trainer = TripletTrainer(config)
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Triplet file: {config['data']['triplet_file']}")
    logger.info(f"Training split: {config['data']['train_split']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Margin: {config['training']['margin']}")
    logger.info("=" * 80)
    
    # Start training
    output_path = trainer.train()
    
    # Run evaluation if configured
    if config['evaluation']['run_evaluation'] and output_path:
        logger.info("Running post-training evaluation...")
        
        # Test cases for evaluation
        test_cases = [
            ("HTML table extraction", "Extract data from HTML tables", "JSON file parsing"),
            ("Web scraping techniques", "HTML web scraping methods", "Database query optimization"),
            ("Parse structured data", "Extract structured information from documents", "Image processing algorithms"),
            ("Machine learning model", "ML algorithm implementation", "Network security protocols"),
            ("Text similarity scoring", "Calculate text semantic similarity", "Video compression techniques")
        ]
        
        # Test improvements
        trainer.test_model_improvements(test_cases)
        
        logger.info(f"Training and evaluation completed successfully!")
        logger.info(f"Fine-tuned model saved to: {output_path}")

if __name__ == "__main__":
    main()