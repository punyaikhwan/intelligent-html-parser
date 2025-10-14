# Intelligent HTML Parser

A smart HTML parser that extracts structured data from HTML based on natural language queries.

## Features

- **Natural Language Query Processing**: Extract entities and attributes from queries like "Can you give me the book: name and price?"
- **Dual Parsing Strategy**: Handles both table and general HTML structures
- **Dual Parsing Methods**: Default will use ML (the best fine-tuned flan-t5-small), and optionally choose rule-based
- **Semantic Similarity Matching**: Employs sentence transformers for intelligent attribute matching
- **REST API**: Easy-to-use HTTP endpoint for integration

## Architecture

### 1. Query Parser
- **Rule-based extraction**: Removes stopwords, detects entities after determiners
- **ML-based extraction**: Detect entities and attributes without query preprocessing

### 2. HTML Parsers
- **Table Parser**: Specialized for HTML tables with header matching (using semantic matching)
- **General Parser**: Handles divs, spans, and other general HTML elements
- **Semantic Matching**: Uses sentence transformers for intelligent attribute matching
- **Full ML HTML Parser**: Uses ML to detect entity and attributes with their values from HTML

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd intelligent-html-parser
```

2. Configure environment variables (optional):
```bash
# Copy the example environment file
cp .env.example .env

# Edit the .env file with your preferred settings
nano .env
```

3. Make the startup script executable:
```bash
chmod +x setup.sh
chmod +x start.sh
```

4. Run these scripts:
For the first time, run this script to setup virtual environment and install the requirements.
```bash
./setup.sh
```

Next, you only need to run this script (except if there are some requirements added).
```bash
./start.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Start the Flask API server

### Manual Installation

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp .env.example .env
# Edit .env file as needed

# Start the application
python app.py
```

## Configuration

The application uses environment variables for configuration. All settings are centralized in `settings.py` and loaded from environment variables or `.env` file.

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `5000` | Server port |
| `DEBUG` | `False` | Enable debug mode |

### ML Model Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ML_MODEL_NAME` | `models/llm/flan-t5-small-tuned-v1` | HuggingFace model for ML query parsing |
| `SIMILARITY_MODEL` | `models/sentence-transformers/all-MiniLM-L6-v2-tuned-v1` | Model for semantic similarity |
| `SIMILARITY_THRESHOLD` | `0.6` | Minimum similarity score (0.0-1.0) |
| `MIN_ATTRIBUTES` | `2` | Minimum attributes before ML fallback (if choose option not using full ml)|

### Performance Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_CONTENT_LENGTH` | `16777216` | Maximum request size (16MB) |
| `REQUEST_TIMEOUT` | `30` | Request timeout in seconds |
| `MAX_RESULTS_PER_QUERY` | `100` | Maximum results per query |
| `TORCH_NUM_THREADS` | `None` | PyTorch thread count |
| `OMP_NUM_THREADS` | `None` | OpenMP thread count |

### Example .env file

```bash
# Copy .env.example to .env and modify as needed
cp .env.example .env
```

See `.env.example` for a complete list of available configuration options.

## Usage

### API Endpoints

#### Parse HTML
```
POST /parse
Content-Type: application/x-www-form-urlencoded

html=<html_content>
query=<natural_language_query>
full-ml=true/false (default: true)
```

If full-ml set to true, all HTML extraction process will use ML.
For now, I use this approaches:
1. The HTML string will be preprocessed before being given to ML model, as giving a raw HTML file is not good, causing bad performance and result.
2. The parser will detect if there any script tag contains json data. If yes, parser will target that json data first.
3. If json data is not sufficient, parser will continue to parse HTML.
4. Intelligently the parser will find repeated structures that might represent entities, grouped by structure. If found, choose the most promising group and extract attributes from each.
   If no repeated structures, search for likely containers that might hold the entity. If found, extract attributes but return only one set with highest confidence.
5. For details, you can read the code. Hopefully the code quite well-structured and easy to read.
### Parse sample HTML file
If you face error 413 Entity Too Large, you can use sample of HTML files in folder samples.
```
POST /parse-from-file
Content-Type: application/x-www-form-urlencoded

html=path/to/file.html
query=<natural_language_query>
full-ml=true/false (default: true)
```

### Example Requests

#### Example 1: Table Data
```bash
curl -X POST http://localhost:5000/parse \
  -d "html=<table><tr><th>Name</th><th>Price</th></tr><tr><td>Book1</td><td>$10</td></tr></table>" \
  -d "query=Can you give me the book name and price?"
```

#### Example 2: General HTML
```bash
curl -X POST http://localhost:5000/parse \
  -d "html=<div class='product'><h3 class='name'>iPhone</h3><span class='price'>$699</span></div>" \
  -d "query=Extract product name and price"
```

### Response Format

```json
{
  "results": {
    "books": [
      {
        "name": "A Light in the Attic",
        "price": "£51.77"
      }
    ]
  },
  "message": "Found 1 books on this page",
  "metadata": {
    "processing_time_ms": 245,
    "model_used": "flan-t5-small-tuned-v1",
    "entity": "book",
    "attributes_requested": 2
  }
}
```

## Configuration

Environment variables:
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)
- `DEBUG`: Debug mode (default: False)

## Performance Notes

- First request may be slower due to model loading
- Table parsing is generally faster than general HTML parsing
- Similarity matching adds processing time but improves accuracy

## Model Selection and Fine-Tuning

### Baseline Model: FLAN-T5-Small

I chose **FLAN-T5-Small** as baseline model for the following reasons:

1. **Instruction Following**: FLAN-T5 is specifically fine-tuned for instruction following, making it ideal for natural language query parsing tasks
2. **Balanced Performance**: The small variant provides a good balance between model capability and computational efficiency
3. **Text-to-Text Framework**: T5's text-to-text approach naturally fits our task of converting natural language queries into structured entity-attribute pairs
4. **Pre-trained Knowledge**: FLAN-T5 comes with extensive pre-training on diverse tasks, providing a strong foundation for our domain-specific fine-tuning
5. **Resource Efficiency**: Small enough to run efficiently in personal PC just for learning.

### Fine-Tuning Process

The fine-tuning methodology follows a rigorous approach:

#### Dataset and Validation Strategy
- **Training Data**: 380 carefully curated examples covering various query patterns and HTML structures. The training data can be found in [training/llm/training_data.json](/training/llm/training_data.json)
- **Cross-Validation**: K-Fold validation with k=5 to ensure robust model evaluation
- **Best Fold Selection**: After evaluating all 5 folds, we selected the fold with the highest performance metrics

#### Hyperparameter Optimization
Once the best fold was identified, we used its train-test split as the foundation for systematic hyperparameter tuning:

- **Learning Rate**: Experimented with different learning rates to find optimal convergence
- **Batch Size**: Tuned for the best balance between training stability and computational efficiency
- **Training Epochs**: Optimized to prevent overfitting while maximizing performance
- **Warmup Steps**: Adjusted for smooth learning rate scheduling
- **Weight Decay**: Fine-tuned for proper regularization

This methodical approach ensures our model generalizes well to unseen queries while maintaining high accuracy on entity and attribute extraction tasks.

#### Fine Tuning Result
##### Baseline model (google/flan-t5-small):
```
{'eval_loss': 2.30395245552063, 'eval_rouge1': 37.50902808119775, 'eval_rouge2': 24.140411157954777, 'eval_rougeL': 36.381872471163746, 'eval_rougeLsum': 36.55871255010567, 'eval_runtime': 12.2113, 'eval_samples_per_second': 6.224, 'eval_steps_per_second': 1.556}
```
##### Current best result
After fine tuning, the best score now is:
```
{'eval_loss': 0.09040385484695435, 'eval_rouge1': 93.11430311944542, 'eval_rouge2': 88.68541416365092, 'eval_rougeL': 93.07590236236969, 'eval_rougeLsum': 93.01471295412385, 'eval_runtime': 87.7054, 'eval_samples_per_second': 0.867, 'eval_steps_per_second': 0.217}
```

Details about training can be found in [models/training-note.txt](/models/training-note.txt).

## Demo
Access demo video here: [demo (in bahasa).mp4](https://drive.google.com/file/d/1agVGq-e9m5moE1NnFfextp3R0q5CQCSE/view?usp=sharing)
And screenshots here: [screenshots](/demo/screenshots/)