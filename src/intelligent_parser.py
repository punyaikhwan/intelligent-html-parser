"""
Main HTML parser orchestrator that coordinates query parsing, HTML analysis, and data extraction.
"""
import logging
import time
from typing import Dict, List, Any
from bs4 import BeautifulSoup

# Import our custom parsers
try:
    # Try relative imports first (when run as module)
    from .parsers.query_parser.ml_query_parser import HybridQueryParser
    from .parsers.html_parser.table_parser import TableParser
    from .parsers.html_parser.general_parser import GeneralHTMLParser
except ImportError:
    # Fall back to absolute imports (when run directly)
    try:
        from parsers.query_parser.ml_query_parser import HybridQueryParser
        from parsers.html_parser.table_parser import TableParser
        from parsers.html_parser.general_parser import GeneralHTMLParser
    except ImportError:
        # Last resort - try importing from src
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from parsers.query_parser.ml_query_parser import HybridQueryParser
        from parsers.html_parser.table_parser import TableParser
        from parsers.html_parser.general_parser import GeneralHTMLParser


class IntelligentHTMLParser:
    """
    Main parser that orchestrates the entire HTML parsing pipeline.
    """
    
    def __init__(self, 
                 ml_model_name: str = "google/flan-t5-small",
                 similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7,
                 min_attributes: int = 2):
        """
        Initialize the intelligent HTML parser.
        
        Args:
            ml_model_name: HuggingFace model name for ML query parsing
            similarity_model: Sentence transformer model for similarity matching
            similarity_threshold: Minimum similarity score for attribute matching
            min_attributes: Minimum attributes needed before falling back to ML
        """
        # Initialize query parser (hybrid: rule-based + ML fallback)
        self.query_parser = HybridQueryParser(ml_model_name, min_attributes)
        
        # Initialize HTML parsers
        self.table_parser = TableParser(similarity_model, similarity_threshold)
        self.general_parser = GeneralHTMLParser(similarity_model, similarity_threshold)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def parse(self, html: str, query: str) -> Dict[str, Any]:
        """
        Main parsing method that orchestrates the entire pipeline.
        
        Args:
            html: HTML string to parse
            query: Natural language query describing what to extract
            
        Returns:
            Dictionary with results, message, and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Parse the query to extract entity and attributes
            self.logger.info("Parsing query...")
            entity, attributes, method, entity_extraction_approach, attribute_extraction_approach = self.query_parser.parse_query(query)
            
            if not entity or not attributes:
                return self._create_error_response(
                    "Could not extract entity and attributes from query",
                    time.time() - start_time
                )
            self.logger.info(f"Extracted entity: {entity} using {entity_extraction_approach}, attributes: {attributes} using {attribute_extraction_approach}")

            # Step 2: Parse HTML and determine structure
            self.logger.info("Analyzing HTML structure...")
            soup = BeautifulSoup(html, 'html.parser')
            
            # Step 3: Choose parsing strategy based on HTML structure
            results = []
            
            html_is_table = self.table_parser.is_table(soup)
            if html_is_table:
                self.logger.info("Table structure detected, using table parser...")
                results = self.table_parser.parse_tables(html, entity, attributes)
            else:
                self.logger.info("Non-table structure detected, using general parser...")
                results = self.general_parser.parse_html(html, entity, attributes)
            
            # Step 4: Format and return results
            processing_time = time.time() - start_time
            approaches_used = {
                "query_parsing": {
                    "method": method,
                },
                "html_parsing": "table" if html_is_table else "general"
            }

            if method != "ml":
                approaches_used["query_parsing"]["entity_extraction_approach"] = entity_extraction_approach
                approaches_used["query_parsing"]["attribute_extraction_approach"] = attribute_extraction_approach

            return self._create_success_response(entity, attributes, results, processing_time, approaches_used)

        except Exception as e:
            self.logger.error(f"Error in parsing: {e}")
            return self._create_error_response(
                f"Parsing error: {str(e)}",
                time.time() - start_time
            )

    def _create_success_response(self, entity: str, attributes: List[str], results: List[Dict[str, Any]], 
                               processing_time: float, approaches_used: Dict[str, str]) -> Dict[str, Any]:
        """Create a successful response following the specified format."""
        # Format results according to the specification
        entity_plural = entity + 's' if not entity.endswith('s') else entity
        
        response = {
            "results": {
                entity_plural: results
            },
            "message": f"Found {len(results)} {entity_plural} on this page",
            "metadata": {
                "processing_time_ms": round(processing_time * 1000),
                "model_used": "custom-html-parser-v1",
                "entity": entity,
                "attributes_requested": attributes,
                "approaches_used": approaches_used,
            }
        }
        
        return response
    
    def _create_error_response(self, error_message: str, processing_time: float) -> Dict[str, Any]:
        """Create an error response."""
        response = {
            "results": {},
            "message": error_message,
            "metadata": {
                "processing_time_ms": round(processing_time * 1000),
                "model_used": "custom-html-parser-v1",
                "error": True
            }
        }
        
        return response
    
    def get_parser_status(self) -> Dict[str, Any]:
        """Get the status of all parser components."""
        status = {
            "query_parser": {
                "rule_based": True,  # Always available
                "ml_fallback": self.query_parser.ml_parser.is_available()
            },
            "table_parser": {
                "available": True,  # BeautifulSoup always available
                "similarity_matching": self.table_parser._model_loaded
            },
            "general_parser": {
                "available": True,  # BeautifulSoup always available
                "similarity_matching": self.general_parser._model_loaded
            }
        }
        
        return status


def test_intelligent_parser():
    """Test function for the intelligent HTML parser."""
    # Test with table HTML
    table_html = """
    <html>
    <body>
        <table>
            <thead>
                <tr>
                    <th>Book Title</th>
                    <th>Author</th>
                    <th>Price</th>
                    <th>Rating</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>The Great Gatsby</td>
                    <td>F. Scott Fitzgerald</td>
                    <td>$12.99</td>
                    <td>4.5</td>
                </tr>
                <tr>
                    <td>To Kill a Mockingbird</td>
                    <td>Harper Lee</td>
                    <td>$14.99</td>
                    <td>4.8</td>
                </tr>
            </tbody>
        </table>
    </body>
    </html>
    """
    
    # Test with general HTML
    general_html = """
    <html>
    <body>
        <div class="product-item">
            <h3 class="product-name">iPhone 13</h3>
            <span class="price">$699</span>
            <p class="description">Latest iPhone model</p>
        </div>
        <div class="product-item">
            <h3 class="product-name">Samsung Galaxy</h3>
            <span class="price">$599</span>
            <p class="description">Android smartphone</p>
        </div>
    </body>
    </html>
    """
    
    parser = IntelligentHTMLParser()
    
    # Test cases
    test_cases = [
        (table_html, "Can you give me the book: name and price?"),
        (general_html, "Extract product name, price, and description"),
        (table_html, "List book author, title, and rating")
    ]
    
    print("Testing Intelligent HTML Parser...")
    print("=" * 50)
    
    for i, (html, query) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Query: {query}")
        
        result = parser.parse(html, query)
        
        print(f"Results: {result}")
        print("-" * 30)
    
    # Test parser status
    print("\nParser Status:")
    print(parser.get_parser_status())


if __name__ == "__main__":
    test_intelligent_parser()