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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HTMLParsingTrainer:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize the HTML parsing trainer
        
        Args:
            model_name: Pre-trained model name to fine-tune
        """
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
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
    
    def train(
        self, 
        train_file: str,
        output_dir: str = "./flan-t5-html-parser",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        save_steps: int = 500,
        eval_steps: int = 500,
        validation_split: float = 0.2
    ):
        """
        Train the model
        
        Args:
            train_file: Path to training data JSON file
            output_dir: Directory to save the trained model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            save_steps: Save model every N steps
            eval_steps: Evaluate model every N steps
            validation_split: Fraction of data to use for validation
        """
        # Set seed for reproducibility
        set_seed(42)
        
        # Load and preprocess data
        logger.info("Loading training data...")
        raw_data = self.load_training_data(train_file)
        
        # Split data into train and validation
        split_idx = int(len(raw_data) * (1 - validation_split))
        train_data = raw_data[:split_idx]
        val_data = raw_data[split_idx:]
        
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
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
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
        self.tokenizer.save_pretrained(output_dir)
        
        # Evaluate on validation set
        logger.info("Final evaluation...")
        eval_results = trainer.evaluate()
        
        logger.info("Training completed!")
        logger.info(f"Final evaluation results: {eval_results}")
        
        return trainer

def main():
    """Main training function"""
    # Initialize trainer
    trainer = HTMLParsingTrainer()
    
    # Train the model
    model_trainer = trainer.train(
        train_file="training_samples.json",
        output_dir="./flan-t5-html-parser",
        num_epochs=5,
        batch_size=4,  # Smaller batch size for small model
        learning_rate=3e-4,
        save_steps=100,
        eval_steps=100
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()