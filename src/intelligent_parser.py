"""
Main HTML parser orchestrator that coordinates query parsing, HTML analysis, and data extraction.
"""
import logging
import time
from typing import Dict, List, Any
from bs4 import BeautifulSoup

from utils import noun

# Import our custom parsers
from parsers.query_parser.ml_query_parser import HybridQueryParser
from parsers.query_parser.ml_query_parser import MLQueryParser
from parsers.html_parser.table_parser import TableParser
from parsers.html_parser.general_parser import GeneralHTMLParser
from parsers.html_parser.json_script_parser import JSONScriptParser
from parsers.ml_html_parser.parser import MLHTMLParser


class IntelligentHTMLParser:
    """
    Main parser that orchestrates the entire HTML parsing pipeline.
    """
    
    def __init__(self, 
                 ml_model_name: str = "google/flan-t5-small",
                 similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.6,
                 min_attributes: int = 2
                 ):
        """
        Initialize the intelligent HTML parser.
        
        Args:
            ml_model_name: HuggingFace model name for ML query parsing
            similarity_model: Sentence transformer model for similarity matching
            similarity_threshold: Minimum similarity score for attribute matching
            min_attributes: Minimum attributes needed before falling back to ML
        """

        # Initialize query parser (hybrid: rule-based + ML fallback)
        self.hybrid_query_parser = HybridQueryParser(ml_model_name, min_attributes)
        self.ml_query_parser = MLQueryParser(ml_model_name)
        
        # Initialize HTML parsers
        self.ml_model_name = ml_model_name
        self.similarity_model = similarity_model
        self.ml_html_parser = MLHTMLParser(ml_model_name)
        self.json_script_parser = JSONScriptParser(similarity_model, similarity_threshold)
        self.table_parser = TableParser(similarity_model, similarity_threshold)
        self.general_parser = GeneralHTMLParser(similarity_model, similarity_threshold)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)

    def parse(self, html: str, query: str, full_ml: bool = True) -> Dict[str, Any]:
        """
        Main parsing method that orchestrates the entire pipeline.
        
        Args:
            html: HTML string to parse
            query: Natural language query describing what to extract
            full_ml: Whether to use the full ML pipeline or not

        Returns:
            Dictionary with results, message, and metadata
        """
        start_time = time.time()
        
        try:
            # Priority 1: Check for JSON scripts first (highest priority)
            logging.info("Starting HTML parsing...")
            soup = BeautifulSoup(html, 'html.parser')
            if self.json_script_parser.has_json_scripts(soup):
                logging.info("JSON scripts detected, using JSON script parser...")
                entity, attributes, method, entity_extraction_approach, attribute_extraction_approach = self.hybrid_query_parser.parse_query(query)
                results = self.json_script_parser.parse_json_scripts(html, entity, attributes)
            
                # If JSON script parsing yielded promising results, skip other methods
                logging.info(f"Extracted entity: {entity} using {entity_extraction_approach}, attributes: {attributes} using {attribute_extraction_approach}")
                if results and any(sum(1 for attr in attributes if attr in res) >= len(attributes) * 2 / 3 for res in results):
                    processing_time = time.time() - start_time
                    approaches_used = {
                        "query_parsing": {
                            "method": method,
                        },
                        "html_parsing": "json_script"
                    }

                    if method != "ml":
                        approaches_used["query_parsing"]["entity_extraction_approach"] = entity_extraction_approach
                        approaches_used["query_parsing"]["attribute_extraction_approach"] = attribute_extraction_approach

                    
                    return self._create_success_response(
                        entity,
                        attributes,
                        results,
                        processing_time,
                        approaches_used,
                        ""                    
                    )

            if full_ml:
                entity, attributes = self.ml_query_parser.parse_query(query)
                if not entity and not attributes:
                    return self._create_error_response(
                        "Could not extract entity and attributes from query",
                        time.time() - start_time
                    )
                
                logging.info("Using ML-based HTML extraction method.")
                results = self.ml_html_parser.parse_html(html, query)
                processing_time = time.time() - start_time
                approaches_used = {
                    "query_parsing": {
                        "method": "ml",
                    },
                    "html_parsing": "ml"
                }

                model_used = self.ml_model_name

                return self._create_success_response(
                    entity,
                    attributes,
                    results,
                    processing_time,
                    approaches_used,
                    model_used                    
                )
            else:
                logging.info("Using rule-based HTML extraction method.")
                
                # Step 1: Parse the query to extract entity and attributes
                logging.info("Parsing query...")
                entity, attributes, method, entity_extraction_approach, attribute_extraction_approach = self.hybrid_query_parser.parse_query(query)
                
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

                return self._create_success_response(entity, attributes, results, processing_time, approaches_used, "")

        except Exception as e:
            logging.error(f"Error in parsing: {e}")
            return self._create_error_response(
                f"Parsing error: {str(e)}",
                time.time() - start_time
            )

    def _create_success_response(self, entity: str, attributes: List[str], results: List[Dict[str, Any]], 
                               processing_time: float, approaches_used: Dict[str, str], model_used: str) -> Dict[str, Any]:
        """Create a successful response following the specified format."""
        # Format results according to the specification
        entity_plural = noun._pluralize_noun(entity)
        message = f"Found {len(results)} {entity} on this page"
        if len(results) > 1:
            message = f"Found {len(results)} {entity_plural} on this page"
        
        response = {
            "results": {
                entity_plural: results
            },
            "message": message,
            "metadata": {
                "processing_time_ms": round(processing_time * 1000),
                "model_used": {
                    "llm_model": model_used,
                    "similarity_model": self.similarity_model
                },
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
                "ml_fallback": self.hybrid_query_parser.ml_parser.is_available()
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