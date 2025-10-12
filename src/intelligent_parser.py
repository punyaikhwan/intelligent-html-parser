"""
Main HTML parser orchestrator that coordinates query parsing, HTML analysis, and data extraction.
"""
import logging
import time
from typing import Dict, List, Any
from bs4 import BeautifulSoup

# Import our custom parsers
from parsers.query_parser.ml_query_parser import HybridQueryParser
from parsers.html_parser.table_parser import TableParser
from parsers.html_parser.general_parser import GeneralHTMLParser
from parsers.html_parser.json_script_parser import JSONScriptParser


class IntelligentHTMLParser:
    """
    Main parser that orchestrates the entire HTML parsing pipeline.
    """
    
    def __init__(self, 
                 ml_model_name: str = "google/flan-t5-small",
                 similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.6,
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
        self.json_script_parser = JSONScriptParser(similarity_model, similarity_threshold)
        self.table_parser = TableParser(similarity_model, similarity_threshold)
        self.general_parser = GeneralHTMLParser(similarity_model, similarity_threshold)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
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
            logging.info("Parsing query...")
            entity, attributes, method, entity_extraction_approach, attribute_extraction_approach = self.query_parser.parse_query(query)
            
            if not entity or not attributes:
                return self._create_error_response(
                    "Could not extract entity and attributes from query",
                    time.time() - start_time
                )
            logging.info(f"Extracted entity: {entity} using {entity_extraction_approach}, attributes: {attributes} using {attribute_extraction_approach}")

            # Step 2: Parse HTML and determine structure
            logging.info("Analyzing HTML structure...")
            soup = BeautifulSoup(html, 'html.parser')
            
            # Step 3: Choose parsing strategy based on HTML structure
            results = []
            parsing_approach = ""
            
            # Priority 1: Check for JSON scripts first (highest priority)
            useHTMLParser = True
            if self.json_script_parser.has_json_scripts(soup):
                logging.info("JSON scripts detected, using JSON script parser...")
                results = self.json_script_parser.parse_json_scripts(html, entity, attributes)
                parsing_approach = "json_script"
                
                # If JSON script parsing didn't yield promising results, fall back to other methods
                # We define "promising" if all requested attributes are found in at least one entity
                if results and any(all(attr in res for attr in attributes) for res in results):
                    logging.info("Sufficient data extracted from JSON scripts.")
                    useHTMLParser = False
                
                # if not, continue to check for table or general parsing
                logging.info("JSON script parsing did not yield sufficient results, checking other parsing methods...")

            if useHTMLParser:
                # Priority 2: Check for table structure
                html_is_table = self.table_parser.is_table(soup)
                if html_is_table:
                    logging.info("Table structure detected, using table parser...")
                    results = self.table_parser.parse_tables(html, entity, attributes)
                    parsing_approach = "table"
                else:
                    # Priority 3: Use general parser as fallback
                    logging.info("Non-table structure detected, using general parser...")
                    results = self.general_parser.parse_html(html, entity, attributes)
                    parsing_approach = "general"
            
            # Step 4: Format and return results
            processing_time = time.time() - start_time
            approaches_used = {
                "query_parsing": {
                    "method": method,
                },
                "html_parsing": parsing_approach
            }

            if method != "ml":
                approaches_used["query_parsing"]["entity_extraction_approach"] = entity_extraction_approach
                approaches_used["query_parsing"]["attribute_extraction_approach"] = attribute_extraction_approach

            return self._create_success_response(entity, attributes, results, processing_time, approaches_used)

        except Exception as e:
            logging.error(f"Error in parsing: {e}")
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
            "json_script_parser": {
                "available": True,  # BeautifulSoup always available
                "similarity_matching": self.json_script_parser._model_loaded
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