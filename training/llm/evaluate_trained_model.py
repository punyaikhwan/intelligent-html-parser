import gc
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from transformers.trainer_utils import set_seed
import evaluate
import numpy as np
from typing import Dict, List
import os
import logging

from config_loader import load_config, get_training_args
from sklearn.model_selection import KFold

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluateTrainedModel:
    def __init__(self, model_path: str, test_file: str, config: Dict = None):
        """
        Initialize the HTML parsing trainer
        
        Args:
            model_path: Pre-trained model name to fine-tune
        """
        if config is None:
            config = get_training_args(load_config())
        
        self.config = config
        self.model_path = model_path
        self.test_file = test_file
        
        # Memory optimization: Load with low CPU memory usage
        logger.info(f"Loading model: {self.model_path}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)
        
        # Add special tokens if needed
        special_tokens = ["<html>", "</html>", "<extract>", "</extract>"]
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Evaluation metrics
        self.rouge = evaluate.load("rouge")
    
    def preprocess_data(self, examples: List[Dict]) -> Dataset:
        """
        Preprocess the training data for T5 format
        
        Args:
            examples: List of training examples
            
        Returns:
            Dataset object ready for training
        """
        inputs = []
        targets = []
        
        for example in examples:
            # Format input: instruction + HTML content
            input_text = f"{example['instruction']} {example['input']}"
            inputs.append(input_text)
            targets.append(example['output'])
        
        # Tokenize inputs and targets
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding=False
        )
        
        labels = self.tokenizer(
            targets,
            max_length=256,
            truncation=True,
            padding=False
        )
        
        # Prepare dataset
        dataset_dict = {
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': labels['input_ids']
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        
        # Handle case where predictions is a tuple (logits, past_key_values, etc.)
        if isinstance(predictions, tuple):
            predictions = predictions[0]  # Take the logits (first element)
        
        # Convert to numpy array if it's a tensor
        if hasattr(predictions, 'cpu'):
            predictions = predictions.cpu().numpy()
        
        # Convert logits to token IDs by taking argmax
        if len(predictions.shape) == 3:  # logits shape: (batch_size, seq_len, vocab_size)
            predictions = np.argmax(predictions, axis=-1)
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Convert labels to numpy array if it's a tensor
        if hasattr(labels, 'cpu'):
            labels = labels.cpu().numpy()
        
        # Replace -100 in labels (they are used to mask padded tokens)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calculate ROUGE scores
        result = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Extract ROUGE-L F1 score
        result = {key: value * 100 for key, value in result.items()}
        
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
            "rougeLsum": result["rougeLsum"]
        }
    
    def load_evaluation_data(self, file_path: str) -> List[Dict]:
        """Load evaluation data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    def evaluate(self):
        """
        Evaluate the model using simple train/validation split
        """
        # Set seed for reproducibility
        set_seed(42)
        
        # Load and preprocess data
        logger.info("Loading training data...")
        if self.test_file is None:
            raise ValueError("Test file path must be provided for evaluation.")
            
        val_data = self.load_evaluation_data(self.test_file)
        val_dataset = self.preprocess_data(val_data)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # Training arguments using config
        training_args = TrainingArguments(
            output_dir="test_trainer",
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_dir=f"{self.config['output_dir']}/logs",
            logging_steps=self.config['logging_steps'],
            evaluation_strategy="steps",
            eval_steps=self.config['eval_steps'],
            save_steps=self.config['save_steps'],
            save_total_limit=self.config['save_total_limit'],
            load_best_model_at_end=True,
            metric_for_best_model=self.config['metric_for_best_model'],
            greater_is_better=True,
            learning_rate=self.config['learning_rate'],
            fp16=self.config['use_fp16'] and torch.cuda.is_available(),
            dataloader_pin_memory=self.config['dataloader_pin_memory'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            remove_unused_columns=False,
            report_to=["tensorboard"]
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        # Start evaluating
        logger.info("Startevaluating...")
        eval_results = trainer.evaluate()

        logger.info("Evaluation completed!")
        logger.info(f"Final evaluation results: {eval_results}")
        
        return trainer

def main():
    """Main training function using config.yaml"""
    # Load configuration
    config = get_training_args(load_config())

    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate FLAN-T5 HTML Parser")
    parser.add_argument("--model-path", type=str, default="./flan-t5-html-parser", 
                       help="Path to trained model")
    parser.add_argument("--test-file", type=str, default="training_samples.json",
                       help="Path to test data JSON file")
    args = parser.parse_args()
    model_path = args.model_path
    test_file = args.test_file
    # Initialize trainer with config
    trainer = EvaluateTrainedModel(model_path, test_file, config)
    
    # Train the model
    model_trainer = trainer.evaluate()

    print("Evaluation completed successfully!")

if __name__ == "__main__":
    main()