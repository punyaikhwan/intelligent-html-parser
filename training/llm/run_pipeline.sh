#!/bin/bash

# FLAN-T5 HTML Parser Training Pipeline
# This script runs the complete training pipeline

set -e  # Exit on any error

echo "=== FLAN-T5 HTML Parser Training Pipeline ==="
echo

# Check if virtual environment exists
if [ ! -d "venv_llm" ]; then
    echo "Virtual environment not found. Running setup..."
    ./setup_llm.sh
    echo
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_llm/bin/activate

# Check if training data exists
if [ ! -f "training_data.json" ]; then
    echo "Error: training_data.json not found!"
    echo "Please ensure training data is available."
    exit 1
fi

# Step 1: Generate additional training data from HTML samples (optional)
echo "Step 1: Generate additional training data (optional)"
read -p "Generate additional training data from HTML samples? (y/n): " generate_data

if [ "$generate_data" = "y" ] || [ "$generate_data" = "Y" ]; then
    echo "Generating training data from HTML samples..."
    python generate_training_data.py
    
    if [ -f "enhanced_training_data.json" ]; then
        echo "Combining training data..."
        python data_utils.py combine training_data.json enhanced_training_data.json -o combined_training_data.json
        TRAINING_FILE="combined_training_data.json"
    else
        echo "No additional data generated, using original samples..."
        TRAINING_FILE="training_data.json"
    fi
else
    echo "Using original training samples..."
    TRAINING_FILE="training_data.json"
fi

# Step 2: Analyze training data
echo
echo "Step 2: Analyzing training data..."
python data_utils.py analyze "$TRAINING_FILE"

# Step 3: Ask for training parameters
echo
echo "Step 3: Training configuration"
read -p "Number of epochs (default: 5): " epochs
epochs=${epochs:-5}

read -p "Batch size (default: 4): " batch_size
batch_size=${batch_size:-4}

read -p "Learning rate (default: 3e-4): " learning_rate
learning_rate=${learning_rate:-3e-4}

read -p "Output directory (default: ./flan-t5-html-parser): " output_dir
output_dir=${output_dir:-"./flan-t5-html-parser"}

# Step 4: Start training
echo
echo "Step 4: Starting training..."
echo "Training parameters:"
echo "  - Training file: $TRAINING_FILE"
echo "  - Epochs: $epochs"
echo "  - Batch size: $batch_size"
echo "  - Learning rate: $learning_rate"
echo "  - Output directory: $output_dir"
echo

read -p "Proceed with training? (y/n): " proceed
if [ "$proceed" != "y" ] && [ "$proceed" != "Y" ]; then
    echo "Training cancelled."
    exit 0
fi

# Create custom training script with parameters
cat > run_training.py << EOF
from flan_t5_training import HTMLParsingTrainer

def main():
    trainer = HTMLParsingTrainer()
    model_trainer = trainer.train(
        train_file="$TRAINING_FILE",
        output_dir="$output_dir",
        num_epochs=$epochs,
        batch_size=$batch_size,
        learning_rate=$learning_rate,
        save_steps=100,
        eval_steps=100
    )
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
EOF

echo "Starting training... This may take a while."
echo "Training logs will be saved to $output_dir/logs/"
echo

python run_training.py

# Step 5: Evaluate the model
echo
echo "Step 5: Evaluating trained model..."

if [ -d "$output_dir" ]; then
    echo "Running evaluation..."
    python evaluate_model.py --model_path "$output_dir" --test_file "$TRAINING_FILE"
    
    echo
    echo "Running demo inference..."
    python evaluate_model.py --model_path "$output_dir" --demo
else
    echo "Error: Trained model not found at $output_dir"
    exit 1
fi

# Step 6: Test inference
echo
echo "Step 6: Testing inference API..."
python -c "
from inference import IntelligentHTMLParser
import json

try:
    parser = IntelligentHTMLParser('$output_dir')
    
    # Test with sample HTML
    html = '''
    <div class=\"product\">
        <h2 class=\"product-name\">Test Product</h2>
        <span class=\"price\">$99.99</span>
        <p class=\"description\">This is a test product.</p>
    </div>
    '''
    
    result = parser.extract_products(html)
    print('✓ Inference API working!')
    print(f'Sample result: {result}')
    
except Exception as e:
    print(f'✗ Inference API error: {e}')
"

# Cleanup
rm -f run_training.py

echo
echo "=== Training Pipeline Complete ==="
echo
echo "Results:"
echo "  - Trained model: $output_dir"
echo "  - Training data: $TRAINING_FILE"
echo "  - Logs: $output_dir/logs/"
echo
echo "Next steps:"
echo "  1. Review training logs with: tensorboard --logdir $output_dir/logs"
echo "  2. Test more samples with: python evaluate_model.py --model_path $output_dir --demo"
echo "  3. Integrate with your application using: inference.py"
echo
echo "To use the model in your code:"
echo "  from training.llm.inference import IntelligentHTMLParser"
echo "  parser = IntelligentHTMLParser('$output_dir')"
echo "  result = parser.extract_data('instruction', 'html_content')"
echo