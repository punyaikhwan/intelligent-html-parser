"""
ML-based query parser using Flan-T5 model as fallback.
"""
import logging
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
import torch

from utils import save_json
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            
            results = self._parse_html_from_repeated_structures(soup, query)
            if results and len(results) > 0:
                return None, results
            results = self._parse_html_from_likely_containers(soup, query)
            if results and len(results) > 0:
                return None, results
            return None, []
            
        except Exception as e:
            logging.error(f"Error in ML html parsing: {e}")
            return None, []

    def _parse_html_from_repeated_structures(self, soup: BeautifulSoup, query: str) -> List[Dict[str, Any]]:
        """
        Parse html by identifying repeated structures and extracting attributes.
        
        Args:
            soup: BeautifulSoup object of the HTML
            query: Natural language query string
        Returns:
           List of attributes and values
        """

        results = []
        if not self.is_available():
            logging.warning("ML parser not available, returning empty results.")
            return results
        
        container_groups = self.html_utils.find_repeated_structures(soup)
        if container_groups and len(container_groups) > 0:
            map_groups_to_filled_attrs = {}
            map_groups_to_confidence = {}
            
            # Process each first group container in parallel with minimum 4 workers
            high_confidence_groups = 0
            with ThreadPoolExecutor(max_workers=min(4, len(container_groups))) as executor:
                future_to_group = {
                    executor.submit(self._process_group, group_idx, containers, query): group_idx
                    for group_idx, containers in enumerate(container_groups)
                }
                
                for future in as_completed(future_to_group):
                    group_idx = future_to_group[future]
                    try:
                        filled_attrs_count, confidence = future.result()
                        if filled_attrs_count > 0:
                            map_groups_to_filled_attrs[group_idx] = filled_attrs_count
                            map_groups_to_confidence[group_idx] = confidence
                            if confidence > 0.95: # threshold for high confidence
                                high_confidence_groups += 1
                                if high_confidence_groups >= 3:
                                    # Cancel remaining futures
                                    for remaining_future in future_to_group:
                                        if not remaining_future.done():
                                            remaining_future.cancel()
                                    break
                    except Exception as e:
                        logging.error(f"Error processing group {group_idx}: {e}")
                        continue

        for group_idx, count in map_groups_to_filled_attrs.items():
            logging.info(f"Group {group_idx} has {count} attributes found with confidence {map_groups_to_confidence.get(group_idx, 0.0)}.")

        if not map_groups_to_filled_attrs:
            logging.info("No promising group of containers found based on attributes.")
            return results
        
        # Find the top 3 most promising groups
        top_3_groups = self._find_top_promising_groups(
            map_groups_to_confidence,
            top_k=3
        )

        logging.info(f"Top 3 promising groups: {[idx for idx, _ in top_3_groups]}")

        # Process all top 3 groups and calculate average confidence for each
        group_results = {}
        group_avg_confidences = {}
        
        for group_idx, _ in top_3_groups:
            logging.info(f"Extracting attributes from all containers in group {group_idx}.")
            containers = container_groups[group_idx]
            group_confidences = []
            group_extracted_results = []
            
            # Process containers in parallel
            with ThreadPoolExecutor(max_workers=min(4, len(containers))) as executor:
                future_to_container = {
                    executor.submit(self._extract_attributes_from_container, str(container), query): container
                    for container in containers
                }
                
                for future in as_completed(future_to_container):
                    try:
                        extracted_attrs, confidence = future.result()
                        if extracted_attrs and len(extracted_attrs) > 0:
                            group_extracted_results.append(extracted_attrs)
                            group_confidences.append(confidence)
                    except Exception as e:
                        logging.error(f"Error processing container: {e}")
                        continue
            
            if group_extracted_results:
                group_results[group_idx] = group_extracted_results
                avg_confidence = sum(group_confidences) / len(group_confidences) if group_confidences else 0.0
                group_avg_confidences[group_idx] = avg_confidence
                logging.info(f"Group {group_idx} average confidence: {avg_confidence}")
        
        # Return results from group with highest average confidence
        if group_avg_confidences:
            best_group_idx = max(group_avg_confidences, key=group_avg_confidences.get)
            best_avg_confidence = group_avg_confidences[best_group_idx]
            logging.info(f"Best group {best_group_idx} with average confidence: {best_avg_confidence}")
            return group_results[best_group_idx]
            
        return results

    def _parse_html_from_likely_containers(self, soup: BeautifulSoup, query: str) -> List[Dict[str, Any]]:
        containers = self.html_utils.find_likely_entity_container(soup)
        if containers and len(containers) > 0:
            # Extract attributes from each container and return the one with most attributes found
            best_result = None
            best_confidence = 0.0
            best_attributes_found = 0
            results = []
            for container in containers:
                logging.info(f"===============Extracting attributes from container ==================\n{str(container)}")
                extracted_attrs, confidence = self._extract_attributes_from_container(container, query)
                logging.info(f"Extracted attributes from container: {extracted_attrs}")
                results.append((extracted_attrs, confidence))

            logging.info("Evaluating extracted attributes from likely containers.")
            for extracted_attrs, confidence in results:
                logging.info(f"Extracted attributes from likely container: {extracted_attrs} with confidence: {confidence}")
                found_attrs = [attr for attr, value in extracted_attrs.items() if value is not None and value != ""]  
                logging.info(f"Calculating overall confidence score for attributes.")
                if len(found_attrs) > best_attributes_found or (len(found_attrs) == best_attributes_found and confidence > best_confidence):
                    best_attributes_found = len(found_attrs)
                    best_confidence = confidence
                    best_result = extracted_attrs
            
            logging.info(f"Best result: {best_result} with {best_attributes_found} attributes found and confidence {best_confidence}.")
            
            if best_result and best_attributes_found > 0:
                # keep only attributes and values without similarity score
                cleaned_result = {attr: (value.Value if value is not None else None) for attr, value in best_result.items()}
                return [cleaned_result]
            
    def _process_group(self, group_idx: int, containers: List, query: str) -> Tuple[int, float]:
        """Process a group of containers to evaluate their attributes in parallel."""
        logging.info(f"=============Evaluating group {group_idx} with {len(containers)} containers.==============")
        if containers and len(containers) > 0:            
            # we need to ensure that the containers are holding the attributes we are looking for
            # And since this is a repeated structure, we can assume that if one container has the attributes, others will have them too
            # So we will check the first container only
            first_container = containers[0]
            extracted_attrs, confidence = self._extract_attributes_from_container(first_container, query)
            logging.info(f"Extracted attributes from first container: {extracted_attrs} with confidence: {confidence}")
            found_attrs = [attr for attr, value in extracted_attrs.items() if value is not None]    
            for attr in found_attrs:
                attr_result = extracted_attrs[attr]
                if attr_result is not None and attr_result != "":
                    logging.info(f"Attribute '{attr}' found with value: {attr_result}")
                else:
                    # if attribute is found but value is empty, take it out from found_attrs
                    found_attrs.remove(attr)

            logging.info(f"Found {len(found_attrs)} in first container.")
            return len(found_attrs), confidence
        return 0, 0.0

    def _find_top_promising_groups(self, map_groups_to_confidence: Dict[int, float],top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Find the top K groups with the highest confidence and maximum attributes found.
        
        Args:
            map_groups_to_filled_attrs: Mapping of group index to number of filled attributes
            map_groups_to_confidence: Mapping of group index to confidence score
            top_k: Number of top groups to return
            
        Returns:
            List of tuples (group index, score) sorted by score descending
        """
        group_scores = []
        for group_idx, confidence in map_groups_to_confidence.items():
            # Use only confidence score
            score = confidence
            group_scores.append((group_idx, score))
        
        # Sort by score descending and return top K
        group_scores.sort(key=lambda x: x[1], reverse=True)
        return group_scores[:top_k]

    def _extract_attributes_from_container(self, sub_html: str, query: str) -> Tuple[Dict[str, Any], float]:
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
            results, confidence = self._generate_response(prompt)
            logging.info(f"Extracted attributes: {results} with confidence: {confidence}")
            parsed_results = self._parse_response(results)
            logging.info(f"Parsed attributes: {parsed_results}")

            return parsed_results, confidence

        except Exception as e:
            logging.error(f"Error in ML query parsing: {e}")
            return {}, 0.0

    def _create_html_parser_prompt(self, html: str, query: str) -> str:
        """Create a prompt for HTML parsing."""
        prompt = f"""
From the following HTML snippet:
{html}
{query}
"""
        return prompt
    
    
    def _generate_response(self, prompt: str) -> Tuple[str, float]:
        """Generate response using the Flan-T5 model."""
        inputs = self.tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )

        # Decode output
        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Calculate confidence score
        import torch.nn.functional as F
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        # Average confidence across all generated tokens
        confidence = torch.exp(transition_scores[0]).mean().item()

        return response.strip(), confidence

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the model response into structured data."""
        try:
            # Expecting response "key":"value", "key2":"value2"
            splits = response.split(',')
            result = {}
            for item in splits:
                if ':' in item:
                    key, value = item.split(':', 1)
                    key = key.strip().strip('"').strip("'")
                    value = value.strip().strip('"').strip("'")
                    result[key] = value
            return result
            
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return {}