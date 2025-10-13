# FLAN-T5 HTML Parser Training

This directory contains code for fine-tuning Google's FLAN-T5-small model for HTML data extraction tasks.

## Overview

The system trains a language model to extract structured data from HTML content based on natural language instructions. For example:

**Input:**
- Instruction: "Extract product name, price, and description from the following HTML:"
- HTML: `<div class="product"><h2>iPhone 15</h2><span class="price">$999</span><p>Latest iPhone</p></div>`

**Output:**
```"name": "iPhone 15", "price": "$999","description": "Latest iPhone"```

## Files Description

- **`flan_t5_training.py`** - Main training script for fine-tuning FLAN-T5
- **`evaluate_model.py`** - Evaluation and testing of trained model
- **`inference.py`** - Production inference script with convenient methods
- **`training_data.json`** - Data for training
- **`requirements.txt`** - Python dependencies
- **`setup_llm.sh`** - Environment setup script

## Setup

1. **Install dependencies:**
   ```bash
   chmod +x setup_llm.sh
   ./setup_llm.sh
   source venv_llm/bin/activate
   ```

2. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
   ```

## Training Process

### 1. Prepare Training Data

**Training data format:**
```json
[
  {
    "instruction": "Extract product name and price from the following HTML:",
    "input": "<div class=\"product\"><h2>MacBook</h2><span>$1299</span></div>",
    "output": "\"name\": \"MacBook\", \"price\": \"$1299\""
  }
]
```

### 2. Run Training

```bash
python flan_t5_training.py
```

**Training parameters:**
- Model: `google/flan-t5-small` (77M parameters)
- Epochs: 5
- Batch size: 4
- Learning rate: 3e-4
- Output directory: `./flan-t5-html-parser`

**Expected training time:**
- CPU: ~2-3 hours for 50 samples
- GPU: ~20-30 minutes for 50 samples

### 3. Monitor Training

Training logs are saved to `./flan-t5-html-parser/logs/`. You can monitor with TensorBoard:

```bash
tensorboard --logdir ./flan-t5-html-parser/logs
```

## Evaluation

### Test the Model

```bash
# Evaluate on test data
python evaluate_model.py --model_path ./flan-t5-html-parser --test_file training_samples.json

# Run demo inference
python evaluate_model.py --demo
```

### Sample Evaluation Output

```
--- Example 1 ---
Instruction: Extract product name, image url, and description
Expected: "name": "iPhone 15 Pro", "image_url": "https://example.com/iphone15.jpg", "description": "Latest iPhone"
Predicted: "name": "iPhone 15 Pro", "image_url": "https://example.com/iphone15.jpg", "description": "Latest iPhone"
✓ CORRECT

Overall Accuracy: 85.50% (17/20)
```

## Usage

### Basic Inference

```python
from inference import IntelligentHTMLParser

# Load trained model
parser = IntelligentHTMLParser("./flan-t5-html-parser")

# Extract product data
html = '''
<div class="product">
    <h2>Samsung Galaxy S24</h2>
    <span class="price">$799</span>
    <p>Latest Samsung flagship</p>
</div>
'''

result = parser.extract_products(html)
print(result)  # "name": "Samsung Galaxy S24", "price": "$799", "description": "Latest Samsung flagship"
```

### Custom Extraction

```python
# Extract specific attributes
result = parser.extract_custom(
    ["title", "author", "price"], 
    book_html
)

# Custom instruction
result = parser.extract_data(
    "Get the restaurant name, cuisine, and rating from this HTML:",
    restaurant_html
)
```

## Training Data Guidelines

### Good Training Examples

1. **Clear instructions:**
   ```
   ✓ "Extract product name, price, and description from the following HTML:"
   ✗ "Get data from HTML"
   ```

2. **Consistent output format:**
   ```json
   ✓ "name": "Product Name", "price": "$99.99"
   ✗ Product: Product Name, Price: $99.99
   ```

3. **Diverse HTML structures:**
   - Different class names (`.product-title`, `.name`, `h2`)
   - Various HTML tags (`<div>`, `<article>`, `<section>`)
   - Multiple layouts and patterns

### Creating Quality Training Data

1. **Include edge cases:**
   - Missing data fields
   - Nested HTML structures
   - Multiple items per page

2. **Use realistic HTML:**
   - Actual class names and IDs
   - Common HTML patterns
   - Real-world complexity

3. **Maintain consistency:**
   - Same instruction format
   - Consistent JSON output structure
   - Proper escaping and encoding

## Model Performance

### Expected Results

- **Small datasets (10-50 samples):** 70-85% accuracy
- **Medium datasets (100-500 samples):** 85-95% accuracy  
- **Large datasets (1000+ samples):** 90-98% accuracy

### Factors Affecting Performance

1. **Training data quality and quantity**
2. **HTML structure complexity**
3. **Instruction clarity and consistency**
4. **Output format standardization**

### Improving Performance

1. **Add more training data:**
   ```bash
   python generate_training_data.py  # Generate from samples
   # Then combine with existing data
   ```

2. **Fine-tune hyperparameters:**
   - Increase epochs for small datasets
   - Adjust learning rate (1e-4 to 5e-4)
   - Modify batch size based on GPU memory

3. **Improve data quality:**
   - Fix inconsistent output formats
   - Add more diverse HTML structures
   - Include edge cases and error scenarios

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce batch size: `batch_size=2`
   - Use CPU: `device="cpu"`

2. **Poor accuracy:**
   - Check training data consistency
   - Increase training epochs
   - Add more diverse examples

3. **Model not loading:**
   - Check model path exists
   - Ensure all files are saved properly
   - Verify tokenizer compatibility

### Debug Commands

```bash
# Check model files
ls -la ./flan-t5-html-parser/

# Test simple inference
python -c "
from inference import IntelligentHTMLParser
parser = IntelligentHTMLParser('./flan-t5-html-parser')
print('Model loaded successfully!')
"

# Validate training data
python -c "
import json
with open('training_samples.json') as f:
    data = json.load(f)
print(f'Training samples: {len(data)}')
print(f'First sample: {data[0]}')
"
```

## Integration with Existing Parser

To integrate with your current intelligent HTML parser:

1. **Train the model** with your specific HTML patterns
2. **Replace sentence transformer** calls with FLAN-T5 inference
3. **Update** `src/intelligent_parser.py` to use the new model
4. **Maintain** fallback to existing methods for reliability

Example integration:

```python
# In src/intelligent_parser.py
from training.llm.inference import IntelligentHTMLParser

class IntelligentParser:
    def __init__(self):
        self.llm_parser = IntelligentHTMLParser("training/llm/flan-t5-html-parser")
        # Keep existing parsers as fallback
        
    def parse(self, instruction, html):
        try:
            # Try LLM first
            result = self.llm_parser.extract_data(instruction, html)
            return self.parse_result_to_dict(result)
        except Exception:
            # Fallback to existing method
            return self.existing_parse_method(instruction, html)
```

## Next Steps

1. **Collect more training data** from your specific HTML patterns
2. **Experiment with larger models** (flan-t5-base, flan-t5-large) if resources allow
3. **Implement active learning** to improve model with user feedback
4. **Add evaluation metrics** specific to your use cases
5. **Deploy** as API service for production use

## Resources

- [FLAN-T5 Paper](https://arxiv.org/abs/2210.11416)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)