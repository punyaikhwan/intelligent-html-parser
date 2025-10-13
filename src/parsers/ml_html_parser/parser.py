"""
ML-based query parser using Flan-T5 model as fallback.
"""
import logging
from typing import Any, Dict, List, Tuple, Optional
import json

from bs4 import BeautifulSoup

from utils.html_utils import HTMLUtils

try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.error("Transformers library not available. ML parser can't be performed.")


class MLHTMLParser:
    """ML-based html parser using Flan-T5 for entity and attribute extraction."""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize the ML html parser.
        
        Args:
            model_name: HuggingFace model name for Flan-T5
        """
        self.model_name = model_name
        self.model = None
        self._loaded = False
        self.html_utils = HTMLUtils()
        
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
    
    def parse_html(self, html: str, query: str) -> List[Dict[str, Any]]:
        """
        Parse html string based on query.
        
        Args:
            html: HTML string to parse
            query: Natural language query string
            
        Returns:
            Tuple of (entity, List of attributes and values)
        """
        if not self.is_available():
            logging.warning("ML parser not available, returning empty results.")
            return None, []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
        
            # Clean up html from text property tags, to make text extraction easier
            for tag in soup.find_all(self.html_utils.TEXT_PROPERTY_TAGS):
                if len(tag.attrs) == 0:
                    tag.unwrap()

            results = []
            # Two approaches:
            # 1. Find repeated structures that might represent entities. If found, extract attributes from each.
            # 2. If no repeated structures, search for likely containers that might hold the entity. If found, extract attributes but return only one set with highest confidence.
            
            # Find potential containers that might hold the entities
            container_groups = self.html_utils.find_repeated_structures(soup)
            if container_groups and len(container_groups) > 0:
                map_groups_to_filled_attrs = {}
                for group_idx, containers in enumerate(container_groups):
                    logging.info(f"=============Evaluating group {group_idx} with {len(containers)} containers.==============")
                    if containers and len(containers) > 0:            
                        # we need to ensure that the containers are holding the attributes we are looking for
                        # And since this is a repeated structure, we can assume that if one container has the attributes, others will have them too
                        # So we will check the first container only
                        first_container = containers[0]
                        extracted_attrs = self._extract_attributes_from_container(first_container, query)
                        logging.info(f"Extracted attributes from first container: {extracted_attrs}")
                        found_attrs = [attr for attr, value in extracted_attrs.items() if value is not None]    
                        for attr in found_attrs:
                            attr_result = extracted_attrs[attr]
                            if attr_result is not None:
                                logging.info(f"Attribute '{attr}' found with value: {attr_result}")

                        logging.info(f"Found {len(found_attrs)} in first container.")
                        map_groups_to_filled_attrs[group_idx] = len(found_attrs)

            for group_idx, count in map_groups_to_filled_attrs.items():
                logging.info(f"Group {group_idx} has {count} attributes found.")

            if not map_groups_to_filled_attrs:
                logging.info("No promising group of containers found based on attributes.")
            
            most_promising_group_idx = 0
            highest_count = 0
            for group_idx, count in map_groups_to_filled_attrs.items():
                number_of_containers = len(container_groups[group_idx])
                if count > highest_count:
                    highest_count = count
                    most_promising_group_idx = group_idx
                elif count == highest_count and number_of_containers > len(container_groups[most_promising_group_idx]):
                    most_promising_group_idx = group_idx

            logging.info(f"Most promising group: {most_promising_group_idx} with {map_groups_to_filled_attrs.get(most_promising_group_idx, 0)} attributes found.")

            if highest_count > 0:
                logging.info(f"Extracting attributes from all containers in group {most_promising_group_idx}.")
                containers = container_groups[most_promising_group_idx]
                for container in containers:
                    sub_html = str(container)
                    extracted_attrs = self._extract_attributes_from_container(sub_html, query)
                    if extracted_attrs and len(extracted_attrs) > 0:
                        results.append(extracted_attrs)

            return results
            
        except Exception as e:
            logging.error(f"Error in ML html parsing: {e}")
            return None, []

    def _extract_attributes_from_container(self, sub_html: str, query: str) -> Dict[str, Any]:
        """
        Parse html string based on query.
        
        Args:
            query: Natural language query string
            
        Returns:
           List of attributes and values
        """
        if not self.is_available():
            logging.warning("ML parser not available, returning empty results.")
            return []

        try:
            # Create a prompt for attribute extraction
            prompt = self._create_html_parser_prompt(sub_html, query)
            logging.info(f"HTML Parser Prompt: {prompt}")
            results = self._generate_response(prompt)
            logging.info(f"Extracted attributes: {results}")
            parsed_results = self._parse_response(results)
            logging.info(f"Parsed attributes: {parsed_results}")

            return parsed_results

        except Exception as e:
            logging.error(f"Error in ML query parsing: {e}")
            return {}

    def _create_html_parser_prompt(self, html: str, query: str) -> str:
        """Create a prompt for HTML parsing."""
        prompt = f"""
Task: Extract relevant information from the following HTML based on the user's query.

Examples:
HTML: <div class="book"><h2 class="title">The Great Gatsby</h2><p class="author">F. Scott Fitzgerald</p><span class="price">$10.99</span></div>
Query: "Can you give me the book: name and price?"
Result: name=The Great Gatsby; price=$10.99

HTML: <div class="book"><h2 class="title">The Great Gatsby</h2><p class="author">F. Scott Fitzgerald</p></div>
Query: "Can you give me the book: name and price?"
Result: name=The Great Gatsby

HTML: <div class="book"></div>
Query: "Can you give me the book: name and price?"
Result: 

HTML: {html}
Query: {query}
Return the extracted information in a structured format.
"""
        return prompt
    
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using the Flan-T5 model."""
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate response
        with self.tokenizer.as_target_tokenizer():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=1024,
                num_beams=2,
                temperature=0.7,
                do_sample=True,
                early_stopping=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the model response into structured data."""
        try:
            # Expecting response in key=value; key=value format
            result = {}
            pairs = response.split(';')
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if key and value:
                        result[key] = value
            return result if result else {}
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return {}