"""
General HTML parser for extracting data from non-table HTML elements.
"""
import logging
from typing import List, Dict, Any, Optional, Set
from bs4 import BeautifulSoup, Tag
import re

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
    
    # Target tags that commonly contain data
    TARGET_TAGS = {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'input', 'li', 'a', 'strong', 'em'}

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
        
        # Strategy 1: Look for containers with class/id names related to entity
        entity_patterns = [entity, entity + 's', entity + '-item', entity + '_item', 
                          'item-' + entity, 'item_' + entity]
        
        for pattern in entity_patterns:
            # Find by class
            elements = soup.find_all(attrs={'class': re.compile(pattern, re.I)})
            containers.extend(elements)
            
            # Find by id
            elements = soup.find_all(attrs={'id': re.compile(pattern, re.I)})
            containers.extend(elements)
        
        # Strategy 2: Look for common container patterns
        common_containers = soup.find_all(['div', 'section', 'article', 'li', 'tr'])
        
        # Filter containers that might contain multiple attributes
        for container in common_containers:
            if self._is_likely_entity_container(container, entity):
                containers.append(container)
        
        # Strategy 3: If no specific containers found, look for repeated structures
        if not containers:
            containers = self._find_repeated_structures(soup)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_containers = []
        for container in containers:
            if container not in seen:
                seen.add(container)
                unique_containers.append(container)
        
        return unique_containers
    
    def _is_likely_entity_container(self, container: Tag, entity: str) -> bool:
        """Check if a container is likely to contain entity data."""
        # Check if container has enough child elements
        children = container.find_all(self.TARGET_TAGS)
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
        # Look for patterns like multiple divs with same class
        containers = []
        
        # Find all divs and group by class
        divs_by_class = {}
        for div in soup.find_all('div', class_=True):
            class_name = ' '.join(div['class'])
            if class_name not in divs_by_class:
                divs_by_class[class_name] = []
            divs_by_class[class_name].append(div)
        
        # Add classes that appear multiple times
        for class_name, divs in divs_by_class.items():
            if len(divs) > 1:  # Repeated structure
                containers.extend(divs)
        
        # Also check for list items
        li_elements = soup.find_all('li')
        if len(li_elements) > 1:
            containers.extend(li_elements)
        
        return containers
    
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
        exact_match = self._find_by_exact_match(container, attribute)
        if exact_match:
            return exact_match
        
        # Strategy 2: Similarity matching if model is available
        if self._model_loaded:
            similarity_match = self._find_by_similarity(container, attribute)
            if similarity_match:
                return similarity_match
        
        # Strategy 3: Text content matching
        text_match = self._find_by_text_content(container, attribute)
        if text_match:
            return text_match
        
        return None
    
    def _find_by_exact_match(self, container: Tag, attribute: str) -> Optional[str]:
        """Find attribute value using exact string matching."""
        # Look for elements with matching class, id, or name
        for tag in container.find_all(self.TARGET_TAGS):
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
            
            for tag in container.find_all(self.TARGET_TAGS):
                # Collect potential matching texts
                classes = ' '.join(tag.get('class', []))
                tag_id = tag.get('id', '')
                tag_name = tag.get('name', '')
                
                for text in [classes, tag_id, tag_name]:
                    if text.strip():
                        candidates.append(text)
                        elements.append(tag)
            
            if not candidates:
                return None
            
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