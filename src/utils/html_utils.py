import logging
from typing import List, Dict
from bs4 import BeautifulSoup, Tag

class HTMLUtils:
    """Parser for extracting data from general HTML elements (non-table)."""

    # Attributes to check for matching
    COMMON_ATTRIBUTES = {'name', 'names', 'title', 'description', 'info', 'information', 'detail', 'details', 'label'}
    CONTAINER_TAGS = {'body', 'div', 'span', 'article', 'section', 'ul', 'ol', 'li', 'figcaption', 'figure'}
    TEXT_TAGS = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'span'}
    TEXT_PROPERTY_TAGS = {'b', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'i', 'em', 'u', 'small', 'mark', 'abbr', 'cite'}
    IMAGE_KEYWORDS = {'image', 'img', 'photo', 'picture', 'thumbnail', 'avatar', 'logo', 'icon', 'banner'}
    LINK_KEYWORDS = {'link', 'url', 'website', 'web', 'site', 'websites', 'sites'}
    NAVIGATION_KEYWORDS = {'next', 'previous', 'prev', 'back', 'forward', 'more', 'less', 'page', 'pages'}
    ATTRIBUTES_MAY_CONTAINS_VALUES = {'src', 'alt', 'title', 'poster', 'type', 'kind', 'label', 'srclang', 'href', 'rel', 'content'}
 
    def find_repeated_structures(self, soup: BeautifulSoup) -> List[List[Tag]]:
        """Find group of repeated HTML structures that might represent entities."""
        # Look for patterns like multiple elements with same class
        containers = []
        
        # Find elements and group by class for div, span, article, section, ul, and ol tags
        elements_by_class = {}
        map_element_to_class = {}
        map_dataid_to_class = {}
        for tag_name in ['div', 'span', 'article', 'section']:
            for element in soup.find_all(tag_name, class_=True):
                class_name = ' '.join(element['class'])
                # get data-* attributes
                data_attrs = [f"{k}={v}" for k, v in element.attrs.items() if k.startswith('data-')]
                if data_attrs:
                    data_id = ' '.join(data_attrs)
                    if data_id not in map_dataid_to_class:
                        map_dataid_to_class[data_id] = class_name

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

        containers_grouped_by_class = []
        for class_name, elements in elements_by_class.items():
            if elements and len(elements) > 1:
                containers_grouped_by_class.append(elements)

        # sort container groups by length descending
        containers_grouped_by_class = sorted(containers_grouped_by_class, key=lambda x: len(x), reverse=True)
        logging.info("============= Final grouped containers =============")
        for idx, group in enumerate(containers_grouped_by_class):
            if group and len(group) > 0:
                class_name = map_element_to_class.get(group[0], None)
                logging.info(f"Group {idx} with class '{class_name}' has {len(group)} containers.")

        return containers_grouped_by_class
    
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

    def find_likely_entity_container(self, soup: BeautifulSoup, num_attributes: int = 2) -> List[Tag]:
        """Find containers that are likely to contain the specified entity."""
        likely_containers = []
        
        for child in soup.find_all(recursive=True):
            # ignore head, script, style, meta, link tags
            if child.name in ['head', 'script', 'style', 'meta', 'link']:
                continue
            if self._is_likely_entity_container(child, num_attributes):
                likely_containers.append(child)
        
        logging.info(f"Found {len(likely_containers)} likely containers before filtering.")
        return likely_containers

    def _is_likely_entity_container(self, container: Tag, num_attributes: int) -> bool:
        """Check if a container is likely to contain entity data."""
        # Check if container has enough child elements
        children = container.find_all()
        if len(children) < num_attributes:  # Need children at least same as number of attributes
            return False
        
        return True
    
    def _is_any_containers_child_of_containers(self, containers1: List[Tag], containers2: List[Tag]) -> bool:
        """Check if any container in containers1 is a child of any container in containers2."""
        for c1 in containers1:
            for c2 in containers2:
                if c1 in c2.descendants:
                    return True
        return False
