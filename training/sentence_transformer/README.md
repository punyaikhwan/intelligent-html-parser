# Sentence Transformer Fine-tuning with Triplet Loss

This folder contains implementation for fine-tuning Sentence Transformer models using triplet loss to improve similarity scores for specific use cases.

## File Structure

- `config.yaml` - Training parameters configuration file
- `triplet_training.py` - Main script for model training
- `evaluate_model.py` - Script for model evaluation and comparison
- `triplet_samples.json` - Sample triplet data file (anchor, positive, negative)
- `requirements_triplet.txt` - Required dependencies
- `setup_triplet.sh` - Environment setup script

## Usage Instructions

### 1. Setup Environment
```bash
./setup_triplet.sh
```

### 2. Activate Environment
```bash
source venv_triplet/bin/activate
```

### 3. Configure Training
Edit the `config.yaml` file to set training parameters:
```yaml
model:
  name: "sentence-transformers/all-MiniLM-L6-v2"

training:
  epochs: 4
  batch_size: 16
  learning_rate: 2e-5
  margin: 0.5

# ... other parameters
```

### 4. Edit Triplet Data
Edit the `triplet_samples.json` file with your data. Format:
```json
[
  {
    "anchor": "anchor text",
    "positive": "text similar/relevant to anchor",
    "negative": "text dissimilar to anchor"
  }
]
```

### 5. Run Training
```bash
python triplet_training.py
```

### 6. Evaluate Model
```bash
python evaluate_model.py
```

## Training Parameters

Training parameters can be configured in the `config.yaml` file:

### Model Configuration
- `model.name`: Name of the pre-trained model to fine-tune

### Training Configuration
- `training.epochs`: Number of training epochs (default: 4)
- `training.batch_size`: Batch size (default: 16)
- `training.learning_rate`: Learning rate (default: 2e-5)
- `training.margin`: Margin for triplet loss (default: 0.5)
- `training.warmup_steps`: Warmup steps (default: 100)
- `training.evaluation_steps`: Steps between evaluations (default: 500)

### Data Configuration
- `data.triplet_file`: Path to triplet data file
- `data.train_split`: Training data split ratio (default: 0.8)

### Output Configuration
- `output.model_path`: Path to save the model
- `output.add_timestamp`: Add timestamp to folder name
- `output.save_best_model`: Save best performing model during training

### Advanced Configuration
- `advanced.use_cuda_if_available`: Use CUDA if available
- `advanced.num_workers`: Number of workers for data loading
- `advanced.seed`: Random seed for reproducibility

## Triplet Data Format

Each triplet consists of:
- **Anchor**: Reference text
- **Positive**: Text that should have high similarity with anchor
- **Negative**: Text that should have low similarity with anchor

The training goal is to make the model give higher similarity scores between anchor-positive compared to anchor-negative.

## Output

- Fine-tuned model will be saved in `models/fine_tuned_model_[timestamp]` folder
- Evaluation results will be saved in CSV format and visualization plots
- Training logs will be displayed in console

## Tips

1. Ensure high-quality triplet data
2. Use representative data for your domain
3. Monitor evaluation metrics during training
4. Experiment with margin parameter for optimal results

## Available Pre-trained Models

You can choose from various pre-trained models in `config.yaml`:
- `sentence-transformers/all-MiniLM-L6-v2` (default, lightweight)
- `sentence-transformers/all-mpnet-base-v2` (better performance, larger)
- `sentence-transformers/all-distilroberta-v1` (good balance)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config.yaml
- Use a smaller model
- Set `use_cuda_if_available: false` to use CPU

### Poor Performance
- Increase training `epochs`
- Adjust `margin` parameter
- Ensure quality triplet data
- Try different pre-trained models

### Slow Training
- Increase `batch_size` if memory allows
- Use GPU if available
- Increase `num_workers` for data loading