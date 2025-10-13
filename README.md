# Intelligent HTML Parser

A smart HTML parser that extracts structured data from HTML based on natural language queries.

## Features

- **Natural Language Query Processing**: Extract entities and attributes from queries like "Can you give me the book: name and price?"
- **Dual Parsing Strategy**: Handles both table and general HTML structures
- **Machine Learning Fallback**: Uses Flan-T5 when rule-based parsing is insufficient
- **Semantic Similarity Matching**: Employs sentence transformers for intelligent attribute matching
- **REST API**: Easy-to-use HTTP endpoint for integration

## Architecture

### 1. Query Parser
- **Rule-based extraction**: Removes stopwords, detects entities after determiners
- **ML fallback**: Uses Flan-T5-small for complex queries
- **Hybrid approach**: Combines both methods for optimal results

### 2. HTML Parsers
- **Table Parser**: Specialized for HTML tables with header matching
- **General Parser**: Handles divs, spans, and other general HTML elements
- **Semantic Matching**: Uses sentence transformers for intelligent attribute matching

### 3. Main Orchestrator
- Coordinates query parsing and HTML analysis
- Automatically selects appropriate parsing strategy
- Returns structured JSON results

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd intelligent-html-parser
```

2. Make the startup script executable:
```bash
chmod +x setup.sh
chmod +x start.sh
```

3. Run these scripts:
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

# Start the application
python app.py
```

## Usage

### API Endpoints

#### Health Check
```
GET /
```

#### Parser Status
```
GET /status
```

#### Parse HTML
```
POST /parse
Content-Type: application/x-www-form-urlencoded

html=<html_content>
query=<natural_language_query>
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
    "model_used": "custom-html-parser-v1",
    "entity": "book",
    "attributes_requested": 2,
    "parsing_strategy": "table"
  }
}
```

## Dependencies

- **Flask**: Web framework for API
- **BeautifulSoup4**: HTML parsing
- **Transformers**: Flan-T5 model for ML fallback
- **Sentence Transformers**: Semantic similarity matching
- **Scikit-learn**: Cosine similarity calculations
- **NumPy**: Numerical operations

## Project Structure

```
intelligent-html-parser/
├── src/
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── query_parser.py      # Rule-based query parsing
│   │   ├── ml_query_parser.py   # ML-based query parsing
│   │   ├── table_parser.py      # HTML table parser
│   │   └── general_parser.py    # General HTML parser
│   ├── __init__.py
│   └── intelligent_parser.py    # Main orchestrator
├── tests/                       # Test files
├── app.py                      # Flask API application
├── requirements.txt            # Python dependencies
├── start.sh                   # Startup script
├── specs.md                   # Project specifications
└── README.md                  # This file
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
- ML fallback is used only when rule-based parsing finds insufficient attributes

## Limitations

- Maximum HTML size: 10MB
- Maximum query length: 1000 characters
- Similarity matching requires additional memory for models
- ML models may not be available in all environments
