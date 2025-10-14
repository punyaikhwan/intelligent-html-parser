"""
JSON Script parser for extracting data from script tags with JSON content.
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional, Union
from bs4 import BeautifulSoup, Tag

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence transformers not available. Will use exact string matching only.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Will use exact string matching only.")


class JSONScriptParser:
    """Parser for extracting data from script tags containing JSON or JSON-LD."""
    
    def __init__(self, 
                 similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 similarity_threshold: float = 0.6):
        """
        Initialize the JSON script parser.
        
        Args:
            similarity_model: Sentence transformer model for semantic similarity
            similarity_threshold: Minimum similarity score for attribute matching
        """
        self.similarity_threshold = similarity_threshold
        self.similarity_model = None
        self._model_loaded = False
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and SKLEARN_AVAILABLE:
            self._load_similarity_model(similarity_model)
        
        logging.info("JSONScriptParser initialized.")
    
    def _load_similarity_model(self, model_name: str):
        """Load the sentence transformer model for similarity matching."""
        try:
            logging.info(f"Loading similarity model: {model_name}")
            self.similarity_model = SentenceTransformer(model_name)
            self._model_loaded = True
            logging.info("Similarity model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load similarity model: {e}")
            self._model_loaded = False
    
    def has_json_scripts(self, soup: BeautifulSoup) -> bool:
        """
        Check if HTML contains script tags with JSON data.
        
        Args:
            soup: BeautifulSoup object of parsed HTML
            
        Returns:
            bool: True if JSON scripts are found, False otherwise
        """
        json_scripts = self._find_json_scripts(soup)
        return len(json_scripts) > 0
    
    def _find_json_scripts(self, soup: BeautifulSoup) -> List[Tag]:
        """
        Find all script tags that contain JSON or JSON-LD data.
        
        Args:
            soup: BeautifulSoup object of parsed HTML
            
        Returns:
            List of script tags with JSON content
        """
        json_scripts = []
        
        # Find script tags with JSON-LD type
        ld_json_scripts = soup.find_all('script', {'type': 'application/ld+json'})
        json_scripts.extend(ld_json_scripts)
        
        # Find script tags with JSON type
        json_type_scripts = soup.find_all('script', {'type': 'application/json'})
        json_scripts.extend(json_type_scripts)
        
        # Find script tags with specific IDs that commonly contain JSON (like Next.js)
        next_data_scripts = soup.find_all('script', {'id': '__NEXT_DATA__'})
        if next_data_scripts:
            # Only add if not already included
            existing_ids = {script.get('id') for script in json_scripts if script.get('id')}
            for script in next_data_scripts:
                if script.get('id') not in existing_ids:
                    json_scripts.append(script)

        
        logging.info(f"Found {len(json_scripts)} JSON script tags")
        return json_scripts
    
    def parse_json_scripts(self, html: str, entity: str, attributes: List[str]) -> List[Dict[str, Any]]:
        """
        Parse JSON scripts to extract relevant data.
        
        Args:
            html: HTML string to parse
            entity: Entity to look for (e.g., 'product', 'book', 'club')
            attributes: List of attributes to extract
            
        Returns:
            List of extracted entities with their attributes
        """
        soup = BeautifulSoup(html, 'html.parser')
        json_scripts = self._find_json_scripts(soup)
        
        if not json_scripts:
            logging.info("No JSON scripts found")
            return []
        
        all_results = []
        
        for script in json_scripts:
            try:
                script_content = script.string
                if not script_content:
                    continue
                
                # Clean the JSON content
                script_content = script_content.strip()
                
                # Parse JSON
                json_data = json.loads(script_content)
                
                # Flatten the JSON structure
                flattened_data = self._flatten_json(json_data)
                flattened_data = self._clean_flattened_json(flattened_data)
                
                # Extract entities using regex matching first, then semantic matching
                entities = self._extract_entities_from_flattened(flattened_data, entity, attributes)
                all_results.extend(entities)
                
                
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON in script tag: {e}")
                continue
            except Exception as e:
                logging.error(f"Error processing script tag: {e}")
                continue
        
        # Remove duplicates based on all attribute values
        unique_results = self._remove_duplicates(all_results)
        
        logging.info(f"Extracted {len(unique_results)} unique entities from JSON scripts")
        return unique_results
    
    def _flatten_json(self, data: Union[Dict, List], parent_key: str = '', separator: str = '.') -> Dict[str, Any]:
        """
        Flatten JSON structure, handling arrays by adding indices.
        
        Args:
            data: JSON data to flatten
            parent_key: Parent key for nested structures
            separator: Separator for nested keys
            
        Returns:
            Flattened dictionary with dot-notation keys
        """
        items = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                
                if isinstance(value, (dict, list)):
                    items.extend(self._flatten_json(value, new_key, separator).items())
                else:
                    items.append((new_key, value))
                    
        elif isinstance(data, list):
            for i, value in enumerate(data):
                new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
                
                if isinstance(value, (dict, list)):
                    items.extend(self._flatten_json(value, new_key, separator).items())
                else:
                    items.append((new_key, value))
        else:
            items.append((parent_key, data))
            
        return dict(items)
    
    def _clean_flattened_json(self, flattened_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean up flattened JSON by removing common prefixes from keys.
        
        Args:
            flattened_data: Flattened JSON data with dot-notation keys
            
        Returns:
            Cleaned flattened data with common prefixes removed
        """
        if not flattened_data:
            return flattened_data
        
        keys = list(flattened_data.keys())
        if len(keys) <= 1:
            return flattened_data
        
        # Find all possible prefixes
        prefix_counts = {}
        
        for key in keys:
            parts = key.split('.')
            # Generate all possible prefixes
            for i in range(1, len(parts)):
                prefix = '.'.join(parts[:i])
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
        logging.info(f"Prefix counts: {prefix_counts}")
        if not prefix_counts:
            return flattened_data
            
        # Find the most common prefix that appears in at least 70% of keys
        total_keys = len(keys)
        logging.info(f"Total keys found: {total_keys}")
        threshold = total_keys * 0.5  # At least 50% of keys
        logging.info(f"Threshold: {threshold}")
        
        most_common_prefix = None
        max_count = 0
        
        for prefix, count in prefix_counts.items():
            if count >= threshold and count > max_count:
                # Verify this prefix actually starts the keys (not just contained within)
                matching_keys = [k for k in keys if k.startswith(prefix + '.')]
                if len(matching_keys) >= threshold:
                    most_common_prefix = prefix
                    max_count = count
        
        # Remove the most common prefix if found
        if most_common_prefix:
            cleaned_data = {}
            prefix_with_dot = most_common_prefix + '.'
            
            for key, value in flattened_data.items():
                if key.startswith(prefix_with_dot):
                    # Remove the prefix and the following dot
                    new_key = key[len(prefix_with_dot):]
                    cleaned_data[new_key] = value
                else:
                    # Keep keys that don't have the common prefix
                    cleaned_data[key] = value
            
            logging.info(f"Removed common prefix '{most_common_prefix}' from {max_count} keys")
            if max_count > 0:
                logging.info(f"Cleaning the next most common prefix...")
                cleaned_data = self._clean_flattened_json(cleaned_data)
                return cleaned_data
        
        return flattened_data
    
    def _extract_entities_from_flattened(self, flattened_data: Dict[str, Any], entity: str, attributes: List[str]) -> List[Dict[str, Any]]:
        """
        Extract entities from flattened JSON data using regex matching first, then semantic matching.
        
        Args:
            flattened_data: Flattened JSON data
            entity: Entity to look for (e.g., 'product', 'book', 'club', 'property')
            attributes: List of attributes to extract
            
        Returns:
            List of extracted entities
        """
        results = []
        
        # First, try exact regex matching
        regex_results = self._extract_using_regex(flattened_data, entity, attributes)
        if regex_results:
            logging.info(f"Found {len(regex_results)} entities using regex matching")
            results.extend(regex_results)
        
        logging.info(f"Results from regex: {results}")
        # If regex didn't find result for some or all attributes, try semantic matching only for those attributes
        empty_attributes = []
        if not results:
            empty_attributes = attributes

        else:
            # Get attributes that were not found in regex results
            found_attributes = set()
            for result in regex_results:
                found_attributes.update(result.keys())
            empty_attributes = [attr for attr in attributes if attr not in found_attributes]
            logging.info(f"Attributes not available in regex searching: {empty_attributes}. Process further using semantic matching...")
        
        if len(empty_attributes) > 0:
            semantic_results = self._extract_using_semantic_matching(flattened_data, empty_attributes)
            if semantic_results:
                logging.info(f"Found {len(semantic_results)} entities using semantic matching")
                if not results:
                    results.extend(semantic_results)
                else:
                    # Process semantic_results to fill the empty attributes on results
                    for idx, result in enumerate(semantic_results):
                        existing_result = results[idx] if idx < len(results) else None
                        if existing_result:
                            for attr, value in result.items():
                                existing_result[attr] = value
                            results[idx] = existing_result
                        else:
                            results.append(result)
        
        return results
    
    def _extract_using_regex(self, flattened_data: Dict[str, Any], entity: str, attributes: List[str]) -> List[Dict[str, Any]]:
        """
        Extract entities using regex pattern matching.
        
        Args:
            flattened_data: Flattened JSON data
            entity: Entity to look for
            attributes: List of attributes to extract
            
        Returns:
            List of extracted entities
        """
        results = []
        entity_instances = {}
        
        # Create regex patterns for each attribute
        for attr in attributes:
            # Pattern 1: attribute (e.g., name)
            pattern1 = re.compile(rf'^{re.escape(attr)}$', re.IGNORECASE)

            # Pattern 2: entity.attribute (e.g., property.name)
            pattern2 = re.compile(rf'^{re.escape(entity)}\.{re.escape(attr)}$', re.IGNORECASE)

            # Pattern 3: entity[index].attribute (e.g., property[0].name)
            pattern3 = re.compile(rf'^{re.escape(entity)}\[(\d+)\]\.{re.escape(attr)}$', re.IGNORECASE)

            # Pattern 4: pluralized entity[index].attribute (e.g., properties[0].name)
            entity_plural = entity + 's' if not entity.endswith('s') else entity
            pattern4 = re.compile(rf'^{re.escape(entity_plural)}\[(\d+)\]\.{re.escape(attr)}$', re.IGNORECASE)

            # Pattern 5: nested structures like data.entity[index].attribute
            pattern5 = re.compile(rf'\.{re.escape(entity)}\[(\d+)\]\.{re.escape(attr)}$', re.IGNORECASE)
            pattern6 = re.compile(rf'\.{re.escape(entity_plural)}\[(\d+)\]\.{re.escape(attr)}$', re.IGNORECASE)

            # # Pattern 7: Special pattern for property data like homeDetails.title (for property name)
            # if entity.lower() == 'property':
            #     # Map property attributes to their likely JSON field names
            #     property_mappings = {
            #         'name': ['title', 'name', 'displayName'],
            #         'address': ['address', 'fullAddress', 'streetAddress', 'city'],  # City as fallback for address
            #         'latitude': ['latitude', 'lat', 'y'],
            #         'longitude': ['longitude', 'lng', 'lon', 'x'],
            #         'price': ['price', 'cost', 'amount', 'rate'],
            #         'description': ['description', 'summary', 'details']
            #     }
                
            #     if attr in property_mappings:
            #         for field_name in property_mappings[attr]:
            #             # Look for homeDetails.field_name or similar structures
            #             pattern7 = re.compile(rf'\.homedetails\.{re.escape(field_name)}$', re.IGNORECASE)
            #             pattern8 = re.compile(rf'^homedetails\.{re.escape(field_name)}$', re.IGNORECASE)
                        
            #             for key, value in flattened_data.items():
            #                 if pattern7.search(key) or pattern8.match(key):
            #                     if 'homeDetails' not in entity_instances:
            #                         entity_instances['homeDetails'] = {}
            #                     entity_instances['homeDetails'][attr] = str(value)
            #                     break
            
            # Search for matches with original patterns
            for key, value in flattened_data.items():
                # Check pattern 1 and 2 (single entity)
                if pattern1.match(key) or pattern2.match(key):
                    if 'single' not in entity_instances:
                        entity_instances['single'] = {}
                    entity_instances['single'][attr] = str(value)
                
                # Check pattern 3 (entity array)
                match3 = pattern3.match(key)
                if match3:
                    index = match3.group(1)
                    if index not in entity_instances:
                        entity_instances[index] = {}
                    entity_instances[index][attr] = str(value)
                
                # Check pattern 4 (plural entity array)
                match4 = pattern4.match(key)
                if match4:
                    index = match4.group(1)
                    if index not in entity_instances:
                        entity_instances[index] = {}
                    entity_instances[index][attr] = str(value)
                
                # Check pattern 5 (nested entity array)
                match5 = pattern5.match(key)
                if match5:
                    index = match5.group(1)
                    if index not in entity_instances:
                        entity_instances[index] = {}
                    entity_instances[index][attr] = str(value)

                # Check pattern 6 (nested plural entity array)
                match6 = pattern6.match(key)
                if match6:
                    index = match6.group(1)
                    if index not in entity_instances:
                        entity_instances[index] = {}
                    entity_instances[index][attr] = str(value)
        
        # Convert entity instances to results
        for instance_id, attrs in entity_instances.items():
            if attrs:  # Only include if we found at least one attribute
                results.append(attrs)
        
        return results
    
    def _extract_using_semantic_matching(self, flattened_data: Dict[str, Any], attributes: List[str]) -> List[Dict[str, Any]]:
        """
        Extract entities using semantic matching with sentence transformers.
        
        Args:
            flattened_data: Flattened JSON data
            entity: Entity to look for
            attributes: List of attributes to extract
            
        Returns:
            List of extracted entities
        """
        if not self._model_loaded:
            return []
        
        results = []
        attr_similarities = {}
        attr_values = {}
        
        try:
            # Get embeddings for attributes
            attribute_embeddings = self.similarity_model.encode(attributes)
            
            # Get all keys from flattened data
            keys = list(flattened_data.keys())
            if not keys:
                return []
            
            # Get embeddings for keys
            key_embeddings = self.similarity_model.encode(keys)
            
            # Find best matches for each attribute
            for i, attr in enumerate(attributes):
                similarities = cosine_similarity([attribute_embeddings[i]], key_embeddings)[0]
                logging.info(f"Similarities for attribute '{attr}': {similarities}")
                
                # Find keys above threshold
                for j, similarity in enumerate(similarities):
                    best_similarity = attr_similarities.get(attr, 0)
                    if similarity >= self.similarity_threshold and similarity >= best_similarity:
                        key = keys[j]
                        value = flattened_data[key]
                        logging.info(f"Found value for attribute '{attr}: {value}")
                        attr_similarities[attr] = similarity
                        attr_values[attr] = value
            
            # Convert to results
            logging.info(f"Attributes values: {attr_values}")
            for attr in attributes:
                if attr in attr_values:
                    results.append({attr: attr_values.get(attr, '')})
            
        except Exception as e:
            logging.error(f"Error in semantic matching: {e}")
        
        return results
    
    def _extract_using_flexible_matching(self, flattened_data: Dict[str, Any], entity: str, attributes: List[str]) -> List[Dict[str, Any]]:
        """
        Extract entities using flexible string matching as a fallback.
        
        Args:
            flattened_data: Flattened JSON data
            entity: Entity to look for
            attributes: List of attributes to extract
            
        Returns:
            List of extracted entities
        """
        results = []
        entity_data = {}
        
        # Create attribute mappings for flexible matching
        attribute_mappings = {
            'name': ['title', 'label', 'heading', 'caption', 'displayName'],
            'price': ['cost', 'amount', 'value', 'fee', 'price'],
            'description': ['summary', 'content', 'text', 'details', 'desc'],
            'address': ['location', 'addr', 'street', 'place'],
            'latitude': ['lat', 'latitude', 'y'],
            'longitude': ['lng', 'lon', 'longitude', 'x'],
            'url': ['link', 'href', 'src', 'website'],
            'image': ['img', 'picture', 'photo', 'thumbnail'],
            'date': ['time', 'timestamp', 'created', 'published'],
            'author': ['creator', 'writer', 'by'],
            'phone': ['telephone', 'tel', 'phoneNumber'],
            'email': ['emailAddress', 'mail']
        }
        
        for attr in attributes:
            # Get possible attribute names
            possible_names = [attr.lower()]
            if attr.lower() in attribute_mappings:
                possible_names.extend(attribute_mappings[attr.lower()])
            
            for key, value in flattened_data.items():
                key_lower = key.lower()
                
                # Check if key contains any of the possible attribute names
                for possible_name in possible_names:
                    if possible_name in key_lower:
                        # Extract entity instance identifier
                        entity_id = self._extract_entity_id_from_key(key, entity)
                        
                        if entity_id not in entity_data:
                            entity_data[entity_id] = {}
                        
                        # Only set if not already set (first match wins)
                        if attr not in entity_data[entity_id]:
                            entity_data[entity_id][attr] = str(value)
                        
                        break
        
        # Convert to results
        for instance_data in entity_data.values():
            if instance_data:
                results.append(instance_data)
        
        return results
    
    def _extract_entity_id_from_key(self, key: str, entity: str) -> str:
        """
        Extract entity instance identifier from a key.
        
        Args:
            key: Flattened key
            entity: Entity type
            
        Returns:
            Entity instance identifier
        """
        # Look for array indices in the key
        array_pattern = re.compile(r'\[(\d+)\]')
        matches = array_pattern.findall(key)
        
        if matches:
            # Use the last index found (most specific)
            return f"instance_{matches[-1]}"
        else:
            # No array index found, treat as single instance
            return "single"
    
    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entities based on all attribute values.
        
        Args:
            results: List of extracted entities
            
        Returns:
            List with duplicates removed
        """
        seen = set()
        unique_results = []
        
        for result in results:
            # Create a signature based on all values
            signature = tuple(sorted(result.items()))
            if signature not in seen:
                seen.add(signature)
                unique_results.append(result)
        
        return unique_results
    
    