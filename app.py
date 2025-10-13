"""
Flask API for the Intelligent HTML Parser.
Provides REST endpoint for parsing HTML based on natural language queries.
"""
import logging
import os
import sys
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

# Import settings
from settings import settings

# Import the parser
from html_parser import IntelligentHTMLParser

# Initialize Flask app
app = Flask(__name__)

# Configure Flask app using settings
app.config.update(settings.get_flask_config())

# Configure logging using settings
logging_config = settings.get_logging_config()
logging.basicConfig(
    level=getattr(logging, logging_config['level']),
    format=logging_config['format']
)
logger = logging.getLogger(__name__)

# Initialize the parser (singleton)
parser = None

def get_parser():
    """Get or create the parser instance."""
    global parser
    if parser is None:
        logger.info("Initializing Intelligent HTML Parser...")
        parser_config = settings.get_parser_config()
        parser = IntelligentHTMLParser(**parser_config)
        logger.info("Parser initialized successfully.")
    return parser


@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Intelligent HTML Parser API",
        "version": "1.0.0"
    })


@app.route('/status', methods=['GET'])
def parser_status():
    """Get parser component status."""
    try:
        parser_instance = get_parser()
        status = parser_instance.get_parser_status()
        
        return jsonify({
            "status": "ok",
            "parser_components": status,
            "service": "Intelligent HTML Parser API"
        })
    except Exception as e:
        logger.error(f"Error getting parser status: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/parse', methods=['POST'])
def parse_html():
    """
    Main parsing endpoint.
    
    Expected form data:
    - html: HTML string to parse
    - query: Natural language query describing what to extract
    - full_ml: (optional) Whether to use the full ML pipeline or not (default: True)
    
    Returns JSON with structure:
    {
        "results": {
            "entity_name": [
                {
                    "attribute1": "value1",
                    "attribute2": "value2"
                }
            ]
        },
        "message": "Found N entity_name on this page",
        "metadata": {
            "processing_time_ms": 245,
            "model_used": "custom-html-parser-v1"
        }
    }
    """
    try:
        # Validate request
        if not request.form:
            raise BadRequest("No form data provided")
        
        html = request.form.get('html')
        logging.debug(f"Received HTML content of length: {len(html) if html else 'None'}")
        query = request.form.get('query')
        
        if not html:
            raise BadRequest("Missing 'html' field in form data")
        
        if not query:
            raise BadRequest("Missing 'query' field in form data")
        
        if len(query) > 1000:  # 1000 character limit for query
            raise BadRequest("Query too long (max 1000 characters)")
        
        # Check HTML size (provide helpful error message)
        html_size_mb = len(html.encode('utf-8')) / (1024 * 1024)
        max_size_mb = app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
        
        if html_size_mb > max_size_mb:
            raise BadRequest(f"HTML content too large ({html_size_mb:.2f}MB). Maximum allowed size is {max_size_mb:.0f}MB")
        
        logger.info(f"Processing parse request - Query: {query[:100]}... | HTML size: {html_size_mb:.2f}MB")
        
        # Get parser and process
        parser_instance = get_parser()
        full_ml = request.query_string.decode('utf-8').lower().find('full_ml=false') == -1
        result = parser_instance.parse(html, query, full_ml=full_ml)

        logger.info(f"Parse completed - Processing time: {result.get('metadata', {}).get('processing_time_ms', 0)}ms")
        
        return jsonify(result)
        
    except BadRequest as e:
        logger.warning(f"Bad request: {e}")
        return jsonify({
            "results": {},
            "message": f"Bad request: {e}",
            "metadata": {
                "processing_time_ms": 0,
                "model_used": "custom-html-parser-v1",
                "error": True
            }
        }), 400
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            "results": {},
            "message": f"Internal server error: {str(e)}",
            "metadata": {
                "processing_time_ms": 0,
                "model_used": "custom-html-parser-v1",
                "error": True
            }
        }), 500

@app.route('/parse-from-file', methods=['POST'])
def parse_html_from_file():
    """
    Main parsing endpoint.
    
    Expected form data:
    - html: HTML file location to parse
    - query: Natural language query describing what to extract
    
    Returns JSON with structure:
    {
        "results": {
            "entity_name": [
                {
                    "attribute1": "value1",
                    "attribute2": "value2"
                }
            ]
        },
        "message": "Found N entity_name on this page",
        "metadata": {
            "processing_time_ms": 245,
            "model_used": "custom-html-parser-v1"
        }
    }
    """
    try:
        # Validate request
        if not request.form:
            raise BadRequest("No form data provided")
        
        html_path = request.form.get('html')
        html = read_html_from_file(html_path)
        query = request.form.get('query')
        
        if not html:
            raise BadRequest("Missing 'html' field in form data")
        
        if not query:
            raise BadRequest("Missing 'query' field in form data")
        
        if len(query) > 1000:  # 1000 character limit for query
            raise BadRequest("Query too long (max 1000 characters)")
        
        # Get parser and process
        parser_instance = get_parser()
        result = parser_instance.parse(html, query)
        
        logger.info(f"Parse completed - Processing time: {result.get('metadata', {}).get('processing_time_ms', 0)}ms")
        
        return jsonify(result)
        
    except BadRequest as e:
        logger.warning(f"Bad request: {e}")
        return jsonify({
            "results": {},
            "message": f"Bad request: {e}",
            "metadata": {
                "processing_time_ms": 0,
                "model_used": "custom-html-parser-v1",
                "error": True
            }
        }), 400
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({
            "results": {},
            "message": f"Internal server error: {str(e)}",
            "metadata": {
                "processing_time_ms": 0,
                "model_used": "custom-html-parser-v1",
                "error": True
            }
        }), 500

def read_html_from_file(file_path):
    """Read HTML content from a file."""
    try:
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read().strip()
        
        if not html_content:
            print(f"✗ File is empty: {file_path}")
            return None
        
        print(f"✓ HTML content loaded from: {file_path}")
        print(f"  File size: {len(html_content)} characters")
        return html_content
    
    except Exception as e:
        print(f"✗ Error reading file {file_path}: {e}")
        return None
    
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "results": {},
        "message": "Endpoint not found. Use POST /parse for HTML parsing.",
        "metadata": {
            "processing_time_ms": 0,
            "model_used": "custom-html-parser-v1",
            "error": True
        }
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle method not allowed errors."""
    return jsonify({
        "results": {},
        "message": "Method not allowed. Use POST for /parse endpoint.",
        "metadata": {
            "processing_time_ms": 0,
            "model_used": "custom-html-parser-v1",
            "error": True
        }
    }), 405


if __name__ == '__main__':
    logger.info(f"Starting Intelligent HTML Parser API on {settings.HOST}:{settings.PORT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Environment: {settings.FLASK_ENV}")
    
    # Pre-initialize parser
    try:
        get_parser()
        logger.info("Parser pre-initialized successfully")
    except Exception as e:
        logger.error(f"Failed to pre-initialize parser: {e}")
        logger.warning("Parser will be initialized on first request")
    
    # Start the Flask app
    app.run(host=settings.HOST, port=settings.PORT, debug=settings.DEBUG)