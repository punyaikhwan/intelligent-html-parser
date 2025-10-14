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

class HTMLParsingTrainer:
    def __init__(self, config: Dict = None):
        """
        Initialize the HTML parsing trainer
        
        Args:
            model_name: Pre-trained model name to fine-tune
        """
        if config is None:
            config = get_training_args(load_config())
        
        self.config = config
        self.model_name = config['model_name']
        
        # Memory optimization: Load with low CPU memory usage
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Add special tokens if needed
        special_tokens = ["<html>", "</html>", "<extract>", "</extract>"]
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Evaluation metrics
        self.rouge = evaluate.load("rouge")
        
    def load_training_data(self, file_path: str) -> List[Dict]:
        """Load training data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
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
    
    def train_single_fold(self, train_data: List[Dict], val_data: List[Dict], fold: int = 0):
        """
        Train a single fold of the model
        
        Args:
            train_data: Training data for this fold
            val_data: Validation data for this fold
            fold: Fold number (for output directory naming)
        """
        logger.info(f"Training fold {fold + 1}")
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        
        # Create datasets
        train_dataset = self.preprocess_data(train_data)
        val_dataset = self.preprocess_data(val_data)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # Create fold-specific output directory
        fold_output_dir = f"{self.config['output_dir']}/fold_{fold + 1}"
        
        # Training arguments using config
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_dir=f"{fold_output_dir}/logs",
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
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            remove_unused_columns=False,
            report_to=["tensorboard"]
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Start training
        logger.info(f"Starting training for fold {fold + 1}...")
        trainer.train()
        
        # Memory cleanup after training
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save the fold model
        logger.info(f"Saving fold {fold + 1} model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(fold_output_dir)
        
        # Evaluate on validation set
        logger.info(f"Evaluating fold {fold + 1}...")
        eval_results = trainer.evaluate()
        
        logger.info(f"Fold {fold + 1} completed!")
        logger.info(f"Fold {fold + 1} evaluation results: {eval_results}")
        
        return trainer, eval_results
    
    def train_kfold(self):
        """
        Train the model using k-fold cross validation
        """
        # Set seed for reproducibility
        set_seed(42)
        
        # Load and preprocess data
        logger.info("Loading training data for k-fold validation...")
        raw_data = self.load_training_data(self.config['train_file'])
        
        # Shuffle data if specified in config
        if self.config['shuffle_data']:
            import random
            random.shuffle(raw_data)
        
        # Limit samples if specified in config
        if self.config['max_samples'] is not None:
            raw_data = raw_data[:self.config['max_samples']]
        
        # Initialize k-fold
        kf = KFold(n_splits=self.config['kfold_splits'], shuffle=True, random_state=42)
        
        all_results = []
        best_model = None
        best_score = -1
        
        # Convert to numpy array for k-fold splitting
        data_indices = np.arange(len(raw_data))
        
        logger.info(f"Starting {self.config['kfold_splits']}-fold cross validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(data_indices)):
            # Reset model for each fold (optional - you might want to keep this or not)
            # self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            
            # Split data for this fold
            train_data = [raw_data[i] for i in train_idx]
            val_data = [raw_data[i] for i in val_idx]

            # save train and val data to files for this fold
            fold_data_dir = f"{self.config['output_dir']}/fold_{fold + 1}/data"
            os.makedirs(fold_data_dir, exist_ok=True)
            with open(f"{fold_data_dir}/train.json", 'w', encoding='utf-8') as f:
                json.dump(train_data, f, indent=2, ensure_ascii=False)
            with open(f"{fold_data_dir}/val.json", 'w', encoding='utf-8') as f:
                json.dump(val_data, f, indent=2, ensure_ascii=False)
            
            # Train this fold
            trainer, eval_results = self.train_single_fold(train_data, val_data, fold)
            all_results.append(eval_results)
            
            # Check if this is the best model so far
            current_score = eval_results.get(f'eval_{self.config["metric_for_best_model"]}', 0)
            if current_score > best_score:
                best_score = current_score
                best_model = trainer
        
        # Calculate average results across all folds
        avg_results = {}
        for key in all_results[0].keys():
            if key.startswith('eval_'):
                avg_results[key] = np.mean([result[key] for result in all_results])
                avg_results[f"{key}_std"] = np.std([result[key] for result in all_results])
        
        logger.info("K-Fold Cross Validation completed!")
        logger.info("Average results across all folds:")
        for key, value in avg_results.items():
            logger.info(f"{key}: {value:.4f}")
        
        # Save the best model to the main output directory
        if best_model:
            logger.info("Saving the best model from k-fold validation...")
            best_model.save_model(self.config['output_dir'])
            self.tokenizer.save_pretrained(self.config['output_dir'])
        
        return best_model, all_results, avg_results
    
    def train(self):
        """
        Train the FLAN-T5 model for HTML parsing using config parameters
        """
        # Check if k-fold validation is enabled
        if self.config['use_kfold']:
            logger.info("Using k-fold cross validation...")
            return self.train_kfold()
        else:
            logger.info("Using simple train/validation split...")
            return self.train_simple_split()
    
    def train_simple_split(self):
        """
        Train the model using simple train/validation split
        
        Args:
            train_file: Path to training data JSON file
        """
        # Set seed for reproducibility
        set_seed(42)
        
        # Load and preprocess data
        logger.info("Loading training data...")
        raw_data = self.load_training_data(self.config['train_file'])
        
        # Shuffle data if specified in config
        if self.config['shuffle_data']:
            import random
            random.shuffle(raw_data)
        
        # Limit samples if specified in config
        if self.config['max_samples'] is not None:
            raw_data = raw_data[:self.config['max_samples']]
        
        # Split data into train and validation
        split_idx = int(len(raw_data) * (1 - self.config['validation_split']))
        train_data = raw_data[:split_idx]
        val_data = raw_data[split_idx:]

        # save train and val data to files
        data_dir = f"{self.config['output_dir']}/data"
        os.makedirs(data_dir, exist_ok=True)
        with open(f"{data_dir}/train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        with open(f"{data_dir}/val.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Validation samples: {len(val_data)}")
        
        # Create datasets
        train_dataset = self.preprocess_data(train_data)
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
            output_dir=self.config['output_dir'],
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
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        logger.info("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        # Evaluate on validation set
        logger.info("Final evaluation...")
        eval_results = trainer.evaluate()
        
        logger.info("Training completed!")
        logger.info(f"Final evaluation results: {eval_results}")
        
        return trainer

def main():
    """Main training function using config.yaml"""
    # Load configuration
    config = get_training_args(load_config())
    
    # Initialize trainer with config
    trainer = HTMLParsingTrainer(config)
    
    # Train the model
    model_trainer = trainer.train()
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()