"""
Flask API for the Intelligent HTML Parser.
Provides REST endpoint for parsing HTML based on natural language queries.
"""
import logging
import os
import sys
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

# Import the parser
from html_parser import IntelligentHTMLParser

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the parser (singleton)
parser = None

def get_parser():
    """Get or create the parser instance."""
    global parser
    if parser is None:
        logger.info("Initializing Intelligent HTML Parser...")
        parser = IntelligentHTMLParser()
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
        query = request.form.get('query')
        
        if not html:
            raise BadRequest("Missing 'html' field in form data")
        
        if not query:
            raise BadRequest("Missing 'query' field in form data")
        
        # Validate input lengths
        if len(html) > 10 * 1024 * 1024:  # 10MB limit
            raise BadRequest("HTML content too large (max 10MB)")
        
        if len(query) > 1000:  # 1000 character limit for query
            raise BadRequest("Query too long (max 1000 characters)")
        
        logger.info(f"Processing parse request - Query: {query[:100]}...")
        
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
    # Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Intelligent HTML Parser API on {HOST}:{PORT}")
    logger.info(f"Debug mode: {DEBUG}")
    
    # Pre-initialize parser
    try:
        get_parser()
        logger.info("Parser pre-initialized successfully")
    except Exception as e:
        logger.error(f"Failed to pre-initialize parser: {e}")
        logger.warning("Parser will be initialized on first request")
    
    # Start the Flask app
    app.run(host=HOST, port=PORT, debug=DEBUG)