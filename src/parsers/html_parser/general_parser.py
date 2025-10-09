"""
General HTML parser for extracting data from non-table HTML elements.
"""
import logging
from typing import List, Dict, Any, Optional, Set
from bs4 import BeautifulSoup, Tag
import re

from utils import noun

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class GeneralHTMLParser:
    """Parser for extracting data from general HTML elements (non-table)."""

    # Attributes to check for matching
    TARGET_ATTRIBUTES = {'class', 'id', 'name', 'data-*'}
    
    def __init__(self, similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 similarity_threshold: float = 0.7):
        """
        Initialize the general HTML parser.
        
        Args:
            similarity_model: Sentence transformer model for semantic similarity
            similarity_threshold: Minimum similarity score for attribute matching
        """
        self.similarity_threshold = similarity_threshold
        self.similarity_model = None
        self._model_loaded = False
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and SKLEARN_AVAILABLE:
            self._load_similarity_model(similarity_model)
    
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
    
    def parse_html(self, html: str, entity: str, attributes: List[str]) -> List[Dict[str, Any]]:
        """
        Parse general HTML to extract data based on entity and attributes.
        
        Args:
            html: HTML string to parse
            entity: Entity name to look for
            attributes: List of attributes to extract
            
        Returns:
            List of dictionaries containing extracted data
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find potential containers that might hold the entities
        containers = self._find_entity_containers(soup, entity)
        
        results = []
        for container in containers:
            result = self._extract_attributes_from_container(container, attributes)
            if result and any(value.strip() for value in result.values()):
                results.append(result)
        
        return results
    
    def _find_entity_containers(self, soup: BeautifulSoup, entity: str) -> List[Tag]:
        """
        Find HTML containers that likely contain the target entity.
        
        Args:
            soup: BeautifulSoup object
            entity: Entity name to look for
            
        Returns:
            List of HTML tags that might contain entity data
        """
        containers = []
        
        # # Strategy 1: Look for containers with class/id names related to entity
        # entity_patterns = [entity, noun._pluralize_noun(entity), entity + '-item', entity + '_item', 
        #                   'item-' + entity, 'item_' + entity]
        
        # for pattern in entity_patterns:
        #     # Find by class
        #     elements = soup.find_all(attrs={'class': re.compile(pattern, re.I)})
        #     containers.extend(elements)
            
        #     # Find by id
        #     elements = soup.find_all(attrs={'id': re.compile(pattern, re.I)})
        #     containers.extend(elements)
        
        # # Strategy 2: Look for common container patterns
        # common_containers = soup.find_all(['div', 'section', 'article', 'li', 'tr'])
        
        # # Filter containers that might contain multiple attributes
        # for container in common_containers:
        #     if self._is_likely_entity_container(container, entity):
        #         containers.append(container)
        
        # # Strategy 3: If no specific containers found, look for repeated structures
        # if not containers:
        containers = self._find_repeated_structures(soup)
        
        # Remove duplicates while preserving order
        # seen = set()
        # unique_containers = []
        # for container in containers:
        #     if container not in seen:
        #         seen.add(container)
        #         unique_containers.append(container)
        
        return containers
    
    def _is_likely_entity_container(self, container: Tag, entity: str) -> bool:
        """Check if a container is likely to contain entity data."""
        # Check if container has enough child elements
        children = container.find_all()
        if len(children) < 2:  # Need at least 2 child elements for multiple attributes
            return False
        
        # Check if container text mentions the entity
        text = container.get_text().lower()
        if entity.lower() in text:
            return True
        
        # Check class and id attributes
        attrs_text = ' '.join([
            ' '.join(container.get('class', [])),
            container.get('id', ''),
        ]).lower()
        
        if entity.lower() in attrs_text:
            return True
        
        return False
    
    def _find_repeated_structures(self, soup: BeautifulSoup) -> List[Tag]:
        """Find repeated HTML structures that might represent entities."""
        # Look for patterns like multiple elements with same class
        containers = []
        
        # Find elements and group by class for div, span, and article tags
        elements_by_class = {}
        map_element_to_class = {}
        for tag_name in ['div', 'span', 'article', 'section']:
            for element in soup.find_all(tag_name, class_=True):
                class_name = ' '.join(element['class'])
                # Group elements by class name
                if class_name not in elements_by_class:
                    # Create a new list for this class
                    elements_by_class[class_name] = []
                
                elements_by_class[class_name].append(element)
                map_element_to_class[element] = class_name

        
        # Include <li> based on repeated <ul> or <ol> lists
        list_elements = soup.find_all(['ul', 'ol'])
        for list_element in list_elements:
            list_items = list_element.find_all('li')
            class_name = ' '.join(list_element.get('class', []))
            if class_name not in elements_by_class:
                elements_by_class[class_name] = []
            elements_by_class[class_name].extend(list_items)
            for li in list_items:
                map_element_to_class[li] = class_name

        # Add classes that appear multiple times as potential containers
        for class_name, elements in elements_by_class.items():
            if len(elements) > 1:  # Repeated structure
                # Check if these elements have similar child patterns
                if self._have_similar_child_structure(elements):
                    containers.extend(elements)
        
        logging.info(f"Found {len(containers)} repeated structures based on class names.")

        
        # Filter out nested repeated structures - keep only outermost ones
        containers = self._filter_outermost_containers(containers)
        logging.info(f"{len(containers)} outermost repeated structures identified as containers.")

        # get all containers that all have same class name, the most common class name
        class_count = {}
        for container in containers:
            class_name = map_element_to_class.get(container, None)
            if class_name:
                class_count[class_name] = class_count.get(class_name, 0) + 1

        # Return the most common class name
        if class_count:
            most_common_class = max(class_count, key=class_count.get)
            result = [container for container in containers if map_element_to_class.get(container) == most_common_class]
            logging.info(f"Most common class: {most_common_class}, Count: {class_count[most_common_class]}")
            return result

        return containers
    
    def _have_similar_child_structure(self, elements: List[Tag]) -> bool:
        """
        Check if elements have similar child structure patterns.
        
        Args:
            elements: List of HTML elements to compare
            
        Returns:
            True if elements have similar child structures
        """
        if len(elements) < 2:
            return False
        
        # Get child structure signatures for each element
        signatures = []
        for element in elements:
            signature = self._get_child_signature(element)
            signatures.append(signature)
        
        # Check if at least 70% of signatures are similar
        similar_count = 0
        base_signature = signatures[0]
        
        for signature in signatures:
            if self._signatures_similar(base_signature, signature):
                similar_count += 1
        
        # Return True if at least 70% of elements have similar structure
        return similar_count / len(signatures) >= 0.7

    def _get_child_signature(self, element: Tag) -> Dict[str, int]:
        """
        Get a signature representing the child structure of an element.
        
        Args:
            element: HTML element to analyze
            
        Returns:
            Dictionary with tag names as keys and counts as values
        """
        signature = {}
        
        # Count direct children by tag name
        for child in element.find_all(recursive=False):
            if hasattr(child, 'name') and child.name:
                tag_name = child.name
                signature[tag_name] = signature.get(tag_name, 0) + 1
        
        # Also consider children with specific classes/attributes
        for child in element.find_all():
            if child.get('class'):
                class_key = f"class:{' '.join(child['class'])}"
                signature[class_key] = signature.get(class_key, 0) + 1
            
            if child.get('id'):
                id_key = f"id:{child['id']}"
                signature[id_key] = signature.get(id_key, 0) + 1
        
        return signature

    def _signatures_similar(self, sig1: Dict[str, int], sig2: Dict[str, int], 
                        similarity_threshold: float = 0.6) -> bool:
        """
        Check if two child signatures are similar.
        
        Args:
            sig1: First signature
            sig2: Second signature
            similarity_threshold: Minimum similarity ratio
            
        Returns:
            True if signatures are similar enough
        """
        if not sig1 and not sig2:
            return True
        
        if not sig1 or not sig2:
            return False
        
        # Get all unique keys
        all_keys = set(sig1.keys()) | set(sig2.keys())
        
        if not all_keys:
            return True
        
        # Calculate similarity based on common structure
        common_keys = set(sig1.keys()) & set(sig2.keys())
        
        # At least some keys should be common
        if len(common_keys) == 0:
            return False
        
        # Calculate similarity ratio
        key_similarity = len(common_keys) / len(all_keys)
        
        # Also check if the counts are reasonably similar for common keys
        count_similarity = 0
        for key in common_keys:
            count1, count2 = sig1[key], sig2[key]
            max_count = max(count1, count2)
            min_count = min(count1, count2)
            if max_count > 0:
                count_similarity += min_count / max_count
        
        if common_keys:
            count_similarity /= len(common_keys)
        
        # Combine both similarities
        overall_similarity = (key_similarity + count_similarity) / 2
        
        return overall_similarity >= similarity_threshold

    def _filter_outermost_containers(self, containers: List[Tag]) -> List[Tag]:
        """Filter containers to keep only the outermost repeated structures."""
        outermost_containers = []
        
        for container in containers:
            is_nested = False
            
            # Check if this container is nested inside any other container
            for other_container in containers:
                if container != other_container:
                    # Check if container is a descendant of other_container
                    if self._is_descendant(container, other_container):
                        is_nested = True
                        break
            
            # Only add if it's not nested inside another container
            if not is_nested:
                outermost_containers.append(container)
        
        return outermost_containers

    def _is_descendant(self, element: Tag, potential_ancestor: Tag) -> bool:
        """Check if element is a descendant of potential_ancestor."""
        current = element.parent
        while current:
            if current == potential_ancestor:
                return True
            current = current.parent
        return False
    
    def _extract_attributes_from_container(self, container: Tag, attributes: List[str]) -> Dict[str, Any]:
        """
        Extract attribute values from a container element.
        
        Args:
            container: HTML container element
            attributes: List of attributes to extract
            
        Returns:
            Dictionary mapping attribute names to values
        """
        result = {}
        
        for attribute in attributes:
            logging.info(f"Extracting attribute '{attribute}' from container.")
            value = self._find_attribute_value(container, attribute)
            result[attribute] = value if value else ""
        
        return result
    
    def _find_attribute_value(self, container: Tag, attribute: str) -> Optional[str]:
        """
        Find the value for a specific attribute within a container.
        
        Args:
            container: HTML container element
            attribute: Attribute name to find
            
        Returns:
            Attribute value or None if not found
        """
        # Strategy 1: Exact string matching in class, id, name attributes
        logging.info(f"Finding value for attribute '{attribute}' using exact match.")
        exact_match = self._find_by_exact_match(container, attribute)
        if exact_match:
            logging.info(f"Found exact match for attribute '{attribute}': {exact_match}")
            return exact_match
        logging.info(f"No exact match found for attribute '{attribute}'.")  
        
        # Strategy 2: Similarity matching if model is available
        logging.info(f"Finding value for attribute '{attribute}' using similarity match.")
        if self._model_loaded:
            similarity_match = self._find_by_similarity(container, attribute)
            if similarity_match:
                logging.info(f"Found similarity match for attribute '{attribute}': {similarity_match}")
                return similarity_match
        
        # Strategy 3: Text content matching
        logging.info(f"Finding value for attribute '{attribute}' using text content match.")
        text_match = self._find_by_text_content(container, attribute)
        if text_match:
            logging.info(f"Found text content match for attribute '{attribute}': {text_match}")
            return text_match

        logging.info(f"No match found for attribute '{attribute}'.")
        return None
    
    def _find_by_exact_match(self, container: Tag, attribute: str) -> Optional[str]:
        """Find attribute value using exact string matching."""
        # Look for elements with matching class, id, or name
        for tag in container.find_all():
            # If it's a div or span with child elements, search recursively
            if tag.name in ['div', 'span'] and tag.find_all():
                result = self._find_by_exact_match(tag, attribute)
                if result:
                    return result
            
            # Check class attribute
            classes = tag.get('class', [])
            for class_name in classes:
                if attribute.lower() in class_name.lower() or class_name.lower() in attribute.lower():
                    text = self._get_element_text(tag)
                    if text:
                        return text
            
            # Check id attribute
            tag_id = tag.get('id', '')
            if tag_id and (attribute.lower() in tag_id.lower() or tag_id.lower() in attribute.lower()):
                text = self._get_element_text(tag)
                if text:
                    return text
            
            # Check name attribute
            tag_name = tag.get('name', '')
            if tag_name and (attribute.lower() in tag_name.lower() or tag_name.lower() in attribute.lower()):
                text = self._get_element_text(tag)
                if text:
                    return text
        
        return None
    
    def _find_by_similarity(self, container: Tag, attribute: str) -> Optional[str]:
        """Find attribute value using semantic similarity."""
        if not self._model_loaded:
            return None
        
        try:
            candidates = []
            elements = []
            fallback_text = str()
            for tag in container.find_all():
                # if tag name is p, h1-h6, a, span, strong, em, b, set found_text_tag to True
                if tag.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'span', 'strong', 'em', 'b'] and fallback_text == str():
                    text = self._get_element_text(tag)
                    if text:
                        fallback_text = text # use the first found text tag as fallback
                
                # If it's a div or span with child elements, search recursively
                if tag.name not in ['div', 'span'] and tag.find_all():
                    result = self._find_by_similarity(tag, attribute)
                    if result:
                        return result
                # Collect potential matching texts
                classes = ' '.join(tag.get('class', []))
                tag_id = tag.get('id', '')
                tag_name = tag.get('name', '')
                
                for text in [classes, tag_id, tag_name]:
                    if text.strip():
                        candidates.append(text)
                        elements.append(tag)
            
            if not candidates:
                return fallback_text if fallback_text else None
            
            if candidates and len(candidates) > 0:
                # Calculate similarities
                attribute_embedding = self.similarity_model.encode([attribute])
                candidate_embeddings = self.similarity_model.encode(candidates)
                
                similarities = cosine_similarity(attribute_embedding, candidate_embeddings)[0]
                
                # Find best match above threshold
                best_idx = np.argmax(similarities)
                if similarities[best_idx] >= self.similarity_threshold:
                    best_element = elements[best_idx]
                    return self._get_element_text(best_element)
            
        except Exception as e:
            logging.error(f"Error in similarity matching: {e}")
        
        return None
    
    def _find_by_text_content(self, container: Tag, attribute: str) -> Optional[str]:
        """Find attribute value by searching text content."""
        # Look for labels or text that might indicate the attribute
        for tag in container.find_all(['label', 'span', 'strong', 'em', 'b']):
            text = tag.get_text().strip().lower()
            if attribute.lower() in text:
                # Try to find the value in the next sibling or parent structure
                next_element = tag.find_next_sibling()
                if next_element:
                    value = self._get_element_text(next_element)
                    if value:
                        return value
                
                # Check parent structure
                parent = tag.parent
                if parent:
                    # Look for text after this tag within the parent
                    for sibling in tag.next_siblings:
                        if hasattr(sibling, 'get_text'):
                            value = self._get_element_text(sibling)
                            if value:
                                return value
        
        return None
    
    def _get_element_text(self, element: Tag) -> Optional[str]:
        """Extract clean text from an HTML element."""
        if not element:
            return None
        
        # For input elements, get the value attribute
        if element.name == 'input':
            return element.get('value', '')
        
        # For other elements, get text content
        text = element.get_text().strip()
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.replace('\xa0', ' ')  # Non-breaking space
        
        # Skip if text is too short or contains only whitespace/punctuation
        if len(text) < 1 or not re.search(r'[a-zA-Z0-9]', text):
            return None
        
        return text


def test_general_parser():
    """Test function for the general HTML parser."""
    # Sample HTML
    html = """
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
    
    parser = GeneralHTMLParser()
    entity = "product"
    attributes = ["name", "price", "description"]
    
    print("Testing general HTML parser...")
    results = parser.parse_html(html, entity, attributes)
    print(f"Results: {results}")


if __name__ == "__main__":
    test_general_parser()