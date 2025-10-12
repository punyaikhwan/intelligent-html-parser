"""
ML-based query parser using Flan-T5 model as fallback.
"""
import logging
from typing import List, Tuple, Optional
import json
import re

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available. ML fallback will be disabled.")


class MLQueryParser:
    """ML-based query parser using Flan-T5 for entity and attribute extraction."""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize the ML query parser.
        
        Args:
            model_name: HuggingFace model name for Flan-T5
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._loaded = False
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load the Flan-T5 model and tokenizer."""
        try:
            logging.info(f"Loading {self.model_name} model...")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=True)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self._loaded = True
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self._loaded = False
    
    def is_available(self) -> bool:
        """Check if the ML parser is available and loaded."""
        return TRANSFORMERS_AVAILABLE and self._loaded
    
    def parse_query(self, query: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse query using ML approach to extract entity and attributes.
        
        Args:
            query: Natural language query string
            
        Returns:
            Tuple of (entity, attributes_list)
        """
        if not self.is_available():
            logging.warning("ML parser not available, returning empty results.")
            return None, []
        
        try:
            # Create a prompt for entity extraction
            entity_prompt = self._create_entity_prompt(query)
            entity = self._generate_response(entity_prompt)
            
            # Create a prompt for attribute extraction
            attributes_prompt = self._create_attributes_prompt(query, entity)
            attributes_text = self._generate_response(attributes_prompt)
            attributes = self._parse_attributes_response(attributes_text)
            
            return entity, attributes
            
        except Exception as e:
            logging.error(f"Error in ML query parsing: {e}")
            return None, []
    
    def _create_entity_prompt(self, query: str) -> str:
        """Create a prompt for entity extraction."""
        prompt = f"""
Task: Extract the main entity (noun) that the user wants to find from the following query.
Return only the entity name in singular form, nothing else.

Examples:
Query: "Can you give me the book: name and price?"
Entity: book

Query: "Extract job title, location, salary from the listings"
Entity: job

Query: "Get product name and description"
Entity: product

Query: "{query}"
Entity:"""
        return prompt
    
    def _create_attributes_prompt(self, query: str, entity: Optional[str]) -> str:
        """Create a prompt for attribute extraction."""
        entity_context = f" about {entity}" if entity else ""
        prompt = f"""
Task: Extract the specific attributes/properties{entity_context} that the user wants to find from the following query.
Return the attributes as a comma-separated list.

Examples:
Query: "Can you give me the book: name and price?"
Attributes: name, price

Query: "Extract job title, location, salary, and company name from the listings"
Attributes: title, location, salary, company name

Query: "Get product name, description and price"
Attributes: name, description, price

Query: "Get product names"
Attributes: name

Query: "{query}"
Attributes:"""
        return prompt
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using the Flan-T5 model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate response
        with self.tokenizer.as_target_tokenizer():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=50,
                num_beams=2,
                temperature=0.7,
                do_sample=True,
                early_stopping=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
    def _parse_attributes_response(self, attributes_text: str) -> List[str]:
        """Parse the attributes response into a list."""
        if not attributes_text:
            return []
        
        # Split by commas and clean each attribute
        attributes = []
        for attr in attributes_text.split(','):
            attr = attr.strip()
            if attr and len(attr) > 1:
                # Remove any quotes or extra formatting
                attr = re.sub(r'^["\']|["\']$', '', attr)
                attributes.append(attr)
        
        return attributes


class HybridQueryParser:
    """
    Hybrid query parser that combines rule-based and ML approaches.
    Uses rule-based first, falls back to ML if insufficient attributes found.
    """
    
    def __init__(self, ml_model_name: str = "google/flan-t5-small", min_attributes: int = 2):
        """
        Initialize the hybrid parser.
        
        Args:
            ml_model_name: HuggingFace model name for ML fallback
            min_attributes: Minimum number of attributes before falling back to ML
        """
        # Import the rule-based parser
        try:
            from .rule_base_query_parser import QueryParser
        except ImportError:
            from parsers.query_parser.rule_base_query_parser import QueryParser
        
        self.rule_parser = QueryParser()
        self.ml_parser = MLQueryParser(ml_model_name)
        self.min_attributes = min_attributes
    
    def parse_query(self, query: str) -> Tuple[Optional[str], List[str], str, str, str]:
        """
        Parse query using hybrid approach.
        
        Args:
            query: Natural language query string
            
        Returns:
            Tuple of (entity, attributes_list)
        """
        # First try rule-based approach
        entity, attributes, entity_approach, attr_approach = self.rule_parser.parse_query(query)
        
        # Check if we have sufficient attributes
        if len(attributes) >= self.min_attributes:
            return entity, attributes, "rule-based", entity_approach, attr_approach
        
        # Will fallback to ML only if not enough attributes found AND at least one attribute has very long length
        if len(attributes) > 0 and all(len(attr) < 10 for attr in attributes):
            logging.info("Rule-based parser found less than minimum attributes, but all are short. Not falling back to ML.")
            return entity, attributes, "rule-based", entity_approach, attr_approach
        
        # Fallback to ML approach if rule-based didn't find enough attributes
        if self.ml_parser.is_available():
            logging.info("Rule-based parser found insufficient attributes, falling back to ML parser.")
            ml_entity, ml_attributes = self.ml_parser.parse_query(query)
            
            # Use ML results if they're better
            if len(ml_attributes) > len(attributes):
                return ml_entity or entity, ml_attributes, "ml", "", ""
            
            # Otherwise, combine results
            combined_entity = ml_entity or entity
            combined_attributes = list(set(attributes + ml_attributes))  # Remove duplicates

            return combined_entity, combined_attributes, "combined", "", ""
        else:
            logging.warning("ML parser not available, using rule-based results only.")
            return entity, attributes, "rule-based", entity_approach, attr_approach


def test_hybrid_parser():
    """Test function for the hybrid parser."""
    parser = HybridQueryParser()
    
    test_cases = [
        "Can you give me the book: name and price?",
        "Extract job title, location, salary, and company name from the listings",
        "Get the product name",  # This should trigger ML fallback
        "Show me movies",  # This should also trigger ML fallback
        "List book author, title, price and rating"
    ]
    
    for query in test_cases:
        entity, attributes = parser.parse_query(query)
        print(f"Query: {query}")
        print(f"Entity: {entity}")
        print(f"Attributes: {attributes}")
        print(f"ML Available: {parser.ml_parser.is_available()}")
        print("-" * 50)


if __name__ == "__main__":
    test_hybrid_parser()