"""
General HTML parser for extracting data from non-table HTML elements.
"""
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
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

class TagIdentifier:
    """Identifier for HTML tags and attributes."""
    
    def __init__(self, TagName: str, Attributes: Dict[str, str]):
        self.TagName = TagName
        self.Attributes = Attributes

    def __repr__(self):
        return f"TagIdentifier(TagName={self.TagName}, Attributes={self.Attributes})"

    def _use_only_first_class(self):
        """Simplify class attribute to only use the first class."""
        if 'class' in self.Attributes and isinstance(self.Attributes['class'], list) and len(self.Attributes['class']) > 0:
            self.Attributes['class'] = [self.Attributes['class'][0]]

class AttributeData:
    """Data structure for attribute information."""

    def __init__(self, AttributeName: Optional[str] = None, Value: str = "", Score: Optional[float] = None, PathChains: List[TagIdentifier] = None):
        self.AttributeName = AttributeName
        self.Value = Value
        self.Score = Score
        self.PathChains = PathChains
    
    def __repr__(self):
        return f"AttributeData(AttributeName={self.AttributeName}, Value={self.Value}, Score={self.Score}, PathChains={self.PathChains})"

class GeneralHTMLParser:
    """Parser for extracting data from general HTML elements (non-table)."""

    # Attributes to check for matching
    TARGET_ATTRIBUTES = {'class', 'id', 'name', 'data-*', 'itemprop'}
    COMMON_ATTRIBUTES = {'name', 'names', 'title', 'description', 'info', 'information', 'detail', 'details', 'label'}
    CONTAINER_TAGS = {'body', 'div', 'span', 'article', 'section', 'ul', 'ol', 'li', 'figcaption', 'figure'}
    TEXT_TAGS = {'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'span'}
    TEXT_PROPERTY_TAGS = {'b', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'strong', 'i', 'em', 'u', 'small', 'mark', 'abbr', 'cite'}
    IMAGE_KEYWORDS = {'image', 'img', 'photo', 'picture', 'thumbnail', 'avatar', 'logo', 'icon', 'banner'}
    LINK_KEYWORDS = {'link', 'url', 'website', 'web', 'site', 'websites', 'sites'}
    NAVIGATION_KEYWORDS = {'next', 'previous', 'prev', 'back', 'forward', 'more', 'less', 'page', 'pages'}
    ATTRIBUTES_MAY_CONTAINS_VALUES = {'src', 'alt', 'title', 'poster', 'type', 'kind', 'label', 'srclang', 'href', 'rel', 'content'}
    
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
        
        # Clean up html from text property tags, to make text extraction easier
        for tag in soup.find_all(self.TEXT_PROPERTY_TAGS):
            if len(tag.attrs) == 0:
                tag.unwrap()
        
        # Two approaches:
        # 1. Find repeated structures that might represent entities. If found, extract attributes from each.
        # 2. If no repeated structures, search for likely containers that might hold the entity. If found, extract attributes but return only one set with highest confidence.
        # Find potential containers that might hold the entities
        container_groups = self._find_repeated_structures(soup, entity)
        if container_groups and len(container_groups) > 0:
            map_groups_to_filled_attrs = {}
            map_groups_to_first_container_extracted_attrs = {}
            map_groups_to_overall_confidence = {}
            # Evaluate the first container in each group of containers and extract attributes
            # We will get the most promising group of containers based on number of attributes found
            for group_idx, containers in enumerate(container_groups):
                logging.info(f"=============Evaluating group {group_idx} with {len(containers)} containers.==============")
                if containers and len(containers) > 0:            
                    # we need to ensure that the containers are holding the attributes we are looking for
                    # And since this is a repeated structure, we can assume that if one container has the attributes, others will have them too
                    # So we will check the first container only
                    first_container = containers[0]
                    extracted_attrs = self._extract_attributes_from_container(first_container, attributes)
                    logging.info(f"Extracted attributes from first container: {extracted_attrs}")
                    map_groups_to_first_container_extracted_attrs[group_idx] = extracted_attrs
                    found_attrs = [attr for attr, value in extracted_attrs.items() if value is not None]    
                    for attr in found_attrs:
                        attr_result = extracted_attrs[attr]
                        if attr_result is not None:
                            logging.info(f"Attribute '{attr}' found with value: {attr_result.Value} and similarity score: {attr_result.Score}")

                    logging.info(f"Found {len(found_attrs)} out of {len(attributes)} attributes in first container.")
                    if len(found_attrs) >= max(1, len(attributes) // 2):  # At least half of the attributes should be found
                        map_groups_to_filled_attrs[group_idx] = len(found_attrs)
                        overall_confidence = self._overall_attributes_confidence(extracted_attrs)
                        map_groups_to_overall_confidence[group_idx] = overall_confidence
                        if len(found_attrs) == len(attributes) and self._all_attributes_high_confidence(extracted_attrs):
                            logging.info(f"All attributes found with high confidence in group {group_idx}. Stopping further search.")
                            break  # Stop if we found a group with all attributes at high confidence
            
            for group_idx, count in map_groups_to_filled_attrs.items():
                logging.info(f"Group {group_idx} has {count} attributes found.")

            if not map_groups_to_filled_attrs:
                logging.info("No promising group of containers found based on attributes.")
            
            most_promising_group_idx = 0
            highest_count = 0
            highest_confidence = 0.0
            for group_idx, count in map_groups_to_filled_attrs.items():
                confidence = map_groups_to_overall_confidence.get(group_idx, 0.0)
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    most_promising_group_idx = group_idx
                elif confidence == highest_confidence:
                    number_of_containers = len(container_groups[group_idx])
                    if count > highest_count:
                        highest_count = count
                        most_promising_group_idx = group_idx
                    elif count == highest_count and number_of_containers > len(container_groups[most_promising_group_idx]):
                        most_promising_group_idx = group_idx

            logging.info(f"Most promising group: {most_promising_group_idx} with {map_groups_to_filled_attrs.get(most_promising_group_idx, 0)} attributes found and confidence {map_groups_to_overall_confidence.get(most_promising_group_idx, 0.0)}.")

            if highest_confidence >= 0.4:
                containers = container_groups[most_promising_group_idx] if most_promising_group_idx is not None else []
                first_container_attrs = map_groups_to_first_container_extracted_attrs.get(most_promising_group_idx, {})

                # since this is a repeated structure, we can use the path chains from the first container to extract attributes from other containers
                map_attrs_to_pathchains = {}
                if first_container_attrs:
                    for attr, value in first_container_attrs.items():
                        if value is not None:
                            map_attrs_to_pathchains[attr] = value.PathChains

                if containers and len(containers) > 0:
                    results = []
                    for i, container in enumerate(containers):
                        if i == 0: # use the already extracted attributes for the first container
                            extracted_attrs = first_container_attrs
                        else:
                            logging.info(f"===============Extracting attributes from container {i}====================\n{str(container)}")
                            extracted_attrs = self._extract_attributes_from_container(container, attributes, map_attrs_to_pathchains)
                            logging.info(f"Extracted attributes from container {i}: {extracted_attrs}")
                        if extracted_attrs and any(value for value in extracted_attrs.values()):
                            # keep only attributes and values without similarity score
                            cleaned_result = {attr: (value.Value if value is not None else None) for attr, value in extracted_attrs.items()}
                            results.append(cleaned_result)

                    return results
        
        logging.info("No group of containers met the confidence threshold.")
        # If no repeated structures found, fall back to searching for likely containers
        containers = self._find_likely_entity_container(soup, soup, entity, attributes)
        if containers and len(containers) > 0:
            # Extract attributes from each container and return the one with most attributes found
            best_result = None
            best_confidence = 0.0
            best_attributes_found = 0
            results = []
            for container in containers:
                logging.info(f"===============Extracting attributes from container ==================\n{str(container)}")
                extracted_attrs = self._extract_attributes_from_container(container, attributes)
                logging.info(f"Extracted attributes from container: {extracted_attrs}")
                results.append(extracted_attrs)

            logging.info("Evaluating extracted attributes from likely containers.")
            for extracted_attrs in results:
                logging.info(f"Extracted attributes from likely container: {extracted_attrs}")
                found_attrs = [attr for attr, value in extracted_attrs.items() if value is not None]    
                for attr in found_attrs:
                    attr_result = extracted_attrs[attr]
                    if attr_result is not None:
                        value, similarity_score = attr_result.Value, attr_result.Score
                        # sometimes, value is empty but similarity score is quite high especially if we are searching for common attribute names, like 'name', 'label', 'description'
                        # this will lead to false positive
                        # so we need to add weighted scoring based on common attributes
                        common_attr_score = self._get_common_attributes_similarity_score(attr)
                        if common_attr_score > 0.8 and (value is None or (isinstance(value, str) and len(value.strip()) == 0)):
                            logging.info(f"Attribute '{attr}' has high common attribute similarity score {common_attr_score} but no value found. Adjusting similarity score...")
                            similarity_score *= 0.6
                            extracted_attrs[attr] = AttributeData(AttributeName=attr, Value=None, Score=similarity_score, PathChains=attr_result.PathChains)
                        
                        # sometimes, common attributes value match with the navigation keywords, like 'next', 'previous', 'more', 'less', 'page'
                        # we need to ignore these results
                        if attr in self.COMMON_ATTRIBUTES:
                            if isinstance(value, str):
                                value_lower = value.strip().lower()
                                if value_lower in self.NAVIGATION_KEYWORDS:
                                    logging.info(f"Attribute '{attr}' has value '{value}' which matches navigation keywords. Ignoring this attribute.")
                                    extracted_attrs[attr] = AttributeData(AttributeName=attr, Value=None, Score=0.0, PathChains=attr_result.PathChains)

                        logging.info(f"Attribute '{attr}' found with value: {value} and similarity score: {similarity_score}")

                logging.info(f"Found {len(found_attrs)} out of {len(attributes)} attributes in likely container.")
                overall_confidence = self._overall_attributes_confidence(extracted_attrs)
                logging.info(f"Calculating overall confidence score for attributes.")
                if len(found_attrs) > best_attributes_found or (len(found_attrs) == best_attributes_found and overall_confidence > best_confidence):
                    best_attributes_found = len(found_attrs)
                    best_confidence = overall_confidence
                    best_result = extracted_attrs
            
            logging.info(f"Best result: {best_result} with {best_attributes_found} attributes found and confidence {best_confidence}.")
            if best_result and best_attributes_found > 0:
                # keep only attributes and values without similarity score
                cleaned_result = {attr: (value.Value if value is not None else None) for attr, value in best_result.items()}
                return [cleaned_result]
            
        logging.info("No likely containers found with the entity and attributes.")

    def _find_repeated_structures(self, soup: BeautifulSoup, entity: str) -> List[List[Tag]]:
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
        for group in containers_grouped_by_class:
            if group and len(group) > 0:
                class_name = map_element_to_class.get(group[0], None)
                logging.info(f"Group with class '{class_name}' has {len(group)} containers.")

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


    def _get_most_similar_tag_attr_vals(self, tag_attr_vals: Set[str], entity: str) -> Tuple[Optional[str], float]:
        """Get the attribute values most similar to the entity name."""
        if not tag_attr_vals or not entity or not self._model_loaded:
            return None, 0.0

        try:
            entity_embedding = self.similarity_model.encode([entity])
            attr_list = list(tag_attr_vals)
            class_embeddings = self.similarity_model.encode(attr_list)
            
            similarities = cosine_similarity(entity_embedding, class_embeddings)[0]
            logging.info(f"Values: {attr_list}")
            logging.info(f"Similarity scores: {similarities}")
            
            # Find the class with highest similarity above threshold
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            logging.info(f"Best similarity score: {best_score} for attribute '{attr_list[best_idx]}'")
            return attr_list[best_idx], best_score
        except Exception as e:
            logging.error(f"Error in finding most similar attribute: {e}")

        return None, 0.0
    
    def _get_similarity_score(self, text1: str, text2: str) -> float:
        """Get similarity score between two texts using the loaded model."""
        if not text1 or not text2 or not self._model_loaded:
            return 0.0
        
        try:
            embeddings = self.similarity_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return similarity
        except Exception as e:
            logging.error(f"Error in calculating similarity score: {e}")
        
        return 0.0
    
    def _get_common_attributes_similarity_score(self, target: str) -> float:
        """Get the highest similarity score between target and common attributes."""
        if not target or not self._model_loaded:
            return 0.0
        
        try:
            target_embedding = self.similarity_model.encode([target])
            common_attr_list = list(self.COMMON_ATTRIBUTES)
            common_attr_embeddings = self.similarity_model.encode(common_attr_list)
            
            similarities = cosine_similarity(target_embedding, common_attr_embeddings)[0]
            max_similarity = max(similarities) if similarities.size > 0 else 0.0
            return max_similarity
        except Exception as e:
            logging.error(f"Error in calculating common attributes similarity score: {e}")
        
        return 0.0
    
    def _find_likely_entity_container(self, soup: BeautifulSoup, container: Tag, entity: str, attributes: List[str]) -> List[Tag]:
        """Find containers that are likely to contain the specified entity."""
        likely_containers = []
        
        for child in container.find_all(recursive=True):
            # ignore head, script, style, meta, link tags
            if child.name in ['head', 'script', 'style', 'meta', 'link']:
                continue
            if self._is_likely_entity_container(child, entity, attributes):
                likely_containers.append(child)
        
        logging.info(f"Found {len(likely_containers)} likely containers for entity '{entity}' before filtering.")
        return likely_containers

    def _is_likely_entity_container(self, container: Tag, entity: str, attributes: List[str]) -> bool:
        """Check if a container is likely to contain entity data."""
        # Check if container has enough child elements
        children = container.find_all()
        if len(children) < len(attributes):  # Need children at least same as number of attributes
            return False
        
        return True

        # # Check if container text mentions the entity
        # text = container.get_text().lower()
        # if entity.lower() in text:
        #     return True
        
        # # Check class and id attributes
        # attrs_text = ' '.join([
        #     ' '.join(container.get('class', [])),
        #     container.get('id', ''),
        # ]).lower()
        
        # if entity.lower() in attrs_text:
        #     return True
        
        # return False

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

    # Extract attributes from a container, return a dict of attribute to (value, similarity score, and data identifier)

    def _extract_attributes_from_container(self, container: Tag, attributes: List[str], map_attrs_to_pathchains: Optional[Dict[str, List[TagIdentifier]]] = None) -> Dict[str, Optional[AttributeData]]:
        """
        Extract attribute values from a container element.
        
        Args:
            container: HTML container element
            attributes: List of attributes to extract
            
        Returns:
            Dictionary mapping attribute names to values
        """
        result = {}

        
        # if provided, instead of extracting attributes from scratch, we will use the path chains to extract the attributes
        if map_attrs_to_pathchains is not None:
            for attr, pathchain in map_attrs_to_pathchains.items():
                logging.info(f"Extracting attribute '{attr}' from container using path chain {pathchain}.")
                value = self._find_attribute_value_using_path_chain(attr, container, pathchain)
                logging.info(f"Extracted value for attribute '{attr}': {value}")
                if value is not None:
                    value.AttributeName = attr
                    result[attr] = value
                else: # fallback using normal extraction
                    logging.info(f"value for attribute '{attr}' can't be retrivied using path chain. Falling back to normal extraction.")
                    value = self._find_attribute_value(container, attr)
                    result[attr] = value
            return result
        
        # map_attrs_to_pathchains is not provided, extract attributes from scratch
        for attribute in attributes:
            logging.info(f"Extracting attribute '{attribute}' from container.")
            value = self._find_attribute_value(container, attribute)
            result[attribute] = value
        
        return result
    
    def _find_attribute_value_using_path_chain(self, attribute: str, container: Tag, pathchain: List[TagIdentifier]) -> Optional[AttributeData]:
        """
        Find the value for a specific attribute within a container using a path chain.
        
        Args:
            container: HTML container element
            pathchain: List of TagIdentifier representing the path to the target element
        """
        current = container
        for tag_id in pathchain:
            if current is None:
                return None
            # ignore attributes that may contains value the path chain
            for attr in self.ATTRIBUTES_MAY_CONTAINS_VALUES:
                tag_id.Attributes.pop(attr, None)

            logging.info(f"Traversing to tag '{tag_id.TagName}' with attributes {tag_id.Attributes}.")

            # handling special for image
            # first, only use tag img
            # if multiple match, get the most matched.

            if tag_id.TagName == 'img':
                img_tags = current.findAll('img')
                if img_tags and len(img_tags) > 0:
                    if len(img_tags) == 1:
                        current = img_tags[0]
                    else:
                        best_match = None
                        best_match_count = 0
                        for img in img_tags:
                            match_count = 0
                            for attr_key, attr_val in tag_id.Attributes.items():
                                if img.get(attr_key) == attr_val:
                                    match_count += 1
                            if match_count > best_match_count:
                                best_match_count = match_count
                                best_match = img
                        current = best_match
                else:
                    current = None

                if current is None:
                    return None
            else:
                if len(tag_id.Attributes) == 0:
                    # If no attributes specified, just find the first occurrence of the tag
                    current = current.find(tag_id.TagName, recursive=False)
                else:
                    currents = current.findAll(tag_id.TagName, attrs=tag_id.Attributes, recursive=False)
                    if currents and len(currents) > 0:
                        # if multiple matches, take the most match based on attributes
                        # for example, if target tag class only a and b, and we have two matches elements with class a b and a b c, then return element with class a b.
                        if len(currents) > 1:
                            best_match = None
                            best_match_count = 0
                            for curr in currents:
                                match_count = 0
                                for attr_key, attr_val in tag_id.Attributes.items():
                                    if curr.get(attr_key) == attr_val:
                                        match_count += 1
                                if match_count > best_match_count:
                                    best_match_count = match_count
                                    best_match = curr
                            current = best_match
                        else:
                            current = currents[0]
                    else:
                        current = None

        if current is None:
            return None
        
        # if image, get src or srcset
        attribute = attribute.lower()
        if any(keyword in attribute for keyword in self.IMAGE_KEYWORDS):
            img_src = current.get('src', None)
            if img_src:
                return AttributeData(Value=img_src, PathChains=pathchain)
            img_srcset = current.get('srcset', None)
            if img_srcset:
                return AttributeData(Value=img_srcset, PathChains=pathchain)
            return None
        
        # if link (a), get href
        if any(keyword in attribute for keyword in self.LINK_KEYWORDS):
            href = current.get('href', None)
            if href:
                return AttributeData(Value=href, PathChains=pathchain)
            return None
        
        # Otherwise, get the text content
        text = self._get_element_text(current)
        if text:
            return AttributeData(Value=text, PathChains=pathchain)
        return None

    def _find_attribute_value(self, container: Tag, attribute: str) -> Optional[AttributeData]:
        """
        Find the value for a specific attribute within a container.
        
        Args:
            container: HTML container element
            attribute: Attribute name to find
            
        Returns:
            Attribute value or None if not found
        """
        
        # Strategy 1: If attribute is likely an image or link, try to find <img> or <a> tags
        is_image_or_link = False
        attr_lower = attribute.lower()
        if any(keyword in attr_lower for keyword in self.IMAGE_KEYWORDS):
            is_image_or_link = True
            logging.info(f"Finding value for attribute '{attribute}' as image URL.")
            value = self._find_image_url(container, attribute)
            logging.info(f"Found value for attribute '{attribute}': {value}")
            if value:
                return value
            
        if any(keyword in attr_lower for keyword in self.LINK_KEYWORDS) and not is_image_or_link:
            is_image_or_link = True
            logging.info(f"Finding value for attribute '{attribute}' as link URL.")
            value = self._find_url(container, attribute)
            if value:
                logging.info(f"Found value for attribute '{attribute}': {value}")
                return value

        if is_image_or_link:
            logging.info(f"Attribute '{attribute}' identified as image or link. Skipping next strategy.")
            return None
        
        
        # Strategy 2: Exact string matching in class, id, name attributes
        logging.info(f"Finding value for attribute '{attribute}' using exact match.")
        exact_match = self._find_by_exact_match(container, attribute)
        if exact_match:
            logging.info(f"Found exact match for attribute '{attribute}': {exact_match}")
            return exact_match
        logging.info(f"No exact match found for attribute '{attribute}'.")  
        
        # Strategy 3: Similarity matching using the loaded model
        logging.info(f"Finding value for attribute '{attribute}' using similarity match.")
        if self._model_loaded:
            match = self._find_by_similarity(container, attribute)
            logging.info(f"Similarity match result for attribute '{attribute}': {match}")
            if match is not None:
                logging.info(f"Found similarity match for attribute '{attribute}': {match.Value} (Score: {match.Score})")
                return match

        # # Strategy 3: Text content matching
        # logging.info(f"Finding value for attribute '{attribute}' using text content match.")
        # text_match = self._find_by_text_content(container, attribute)
        # if text_match:
        #     logging.info(f"Found text content match for attribute '{attribute}': {text_match}")
        #     return text_match

        # logging.info(f"No match found for attribute '{attribute}'.")
        return None

    def _find_image_url(self, container: Tag, attribute: str) -> Optional[AttributeData]:
        img_tag = container.find('img')
        if img_tag and img_tag.get('src'):
            logging.info(f"Found image URL for attribute '{attribute}': {img_tag['src']}")

            path_chains = [
                TagIdentifier(TagName='img', Attributes=img_tag.attrs)
            ]

            parents = img_tag.parents
            for i, parent in enumerate(parents):
                if parent.attrs == container.attrs:
                    break

                # sometime divs has one common class and some specific classes, to avoid overfit we will only use the first class
                tagIdentifier = TagIdentifier(TagName=parent.name, Attributes=parent.attrs)
                tagIdentifier._use_only_first_class()
                path_chains.insert(0, tagIdentifier)

            return AttributeData(AttributeName=attribute, Value=img_tag['src'], Score=1.0, PathChains=path_chains)
        return None
    
    def _find_url(self, container: Tag, attribute: str) -> Optional[AttributeData]:
        a_tag = container.find('a')
        if a_tag and a_tag.get('href'):
            logging.info(f"Found URL for attribute '{attribute}': {a_tag['href']}")

            path_chains = [
                TagIdentifier(TagName='a', Attributes=a_tag.attrs)
            ]

            parents = a_tag.parents
            for i, parent in enumerate(parents):
                if parent.attrs == container.attrs:
                    break
                
                 # sometime divs has one common class and some specific classes, to avoid overfit we will only use the first class
                tagIdentifier = TagIdentifier(TagName=parent.name, Attributes=parent.attrs)
                tagIdentifier._use_only_first_class()
                path_chains.insert(0, tagIdentifier)

            return AttributeData(AttributeName=attribute, Value=a_tag['href'], Score=1.0, PathChains=path_chains)
        return None
        
    def _find_by_exact_match(self, container: Tag, attribute: str) -> Optional[AttributeData]:
        """Find attribute value using exact string matching."""
        # Look for elements with matching class, id, or name
        for tag in container.find_all(recursive=False):
            # If it's a div, span, article, section, ul, or ol with child elements, search recursively
            if tag.name in self.CONTAINER_TAGS and tag.find_all(recursive=False):
                result = self._find_by_exact_match(tag, attribute)
                if result:
                    result.PathChains.insert(0, TagIdentifier(TagName=tag.name, Attributes=tag.attrs))
                    return result
            
            # Check class attribute
            classes = tag.get('class', [])
            for class_name in classes:
                if attribute.lower() in class_name.lower() or class_name.lower() in attribute.lower():
                    text = self._get_element_text(tag)
                    if text:
                        logging.info(f"Exact match found in tag '{tag.name}' class '{class_name}' for attribute '{attribute}' with text '{text}'.")
                        return AttributeData(AttributeName=attribute, Value=text, Score=1.0, PathChains=[
                            TagIdentifier(TagName=tag.name, Attributes=tag.attrs)
                        ])
            
            # Check id attribute
            tag_id = tag.get('id', '')
            if tag_id and (attribute.lower() in tag_id.lower() or tag_id.lower() in attribute.lower()):
                text = self._get_element_text(tag)
                if text:
                    logging.info(f"Exact match found in id '{tag_id}' for attribute '{attribute}' with text '{text}'.")
                    return AttributeData(AttributeName=attribute, Value=text, Score=1.0, PathChains=[
                        TagIdentifier(TagName=tag.name, Attributes=tag.attrs)
                    ])
            
            # Check name attribute
            tag_name = tag.get('name', '')
            if tag_name and (attribute.lower() in tag_name.lower() or tag_name.lower() in attribute.lower()):
                text = self._get_element_text(tag)
                if text:
                    logging.info(f"Exact match found in name '{tag_name}' for attribute '{attribute}' with text '{text}'.")
                    return AttributeData(AttributeName=attribute, Value=text, Score=1.0, PathChains=[
                        TagIdentifier(TagName=tag.name, Attributes=tag.attrs)
                    ])

            # Check data-* attributes
            for attr_key, attr_value in tag.attrs.items():
                if attr_key.startswith('data-') and attr_value:
                    if attribute.lower() in attr_value.lower() or attr_value.lower() in attribute.lower():
                        text = self._get_element_text(tag)
                        if text:
                            logging.info(f"Exact match found in data attribute '{attr_key}' with value '{attr_value}' for attribute '{attribute}' with text '{text}'.")
                            return AttributeData(AttributeName=attribute, Value=text, Score=1.0, PathChains=[
                                TagIdentifier(TagName=tag.name, Attributes=tag.attrs)
                            ])
        
        return None

    def _get_element_similarity_to_common_attr(self, container: Tag) -> float:
        """Get the highest similarity score between container's attributes and common attributes."""
        if not self._model_loaded:
            return 0.0
        
        texts_to_check = []
        classes = ' '.join(container.get('class', []))
        if classes.strip():
            texts_to_check.append(classes)
        
        tag_id = container.get('id', '')
        if tag_id.strip():
            texts_to_check.append(tag_id)
        
        tag_name = container.get('name', '')
        if tag_name.strip():
            texts_to_check.append(tag_name)
        
        for attr_key, attr_value in container.attrs.items():
            if attr_key.startswith('data-') and attr_value:
                texts_to_check.append(attr_value)
        
        highest_score = 0.0
        for text in texts_to_check:
            score = self._get_common_attributes_similarity_score(text)
            if score > highest_score:
                highest_score = score
        
        return highest_score
    
    # Find attribute value using semantic similarity. Requires sentence-transformers and sklearn.
    # Output the value and the similarity score.
    def _find_by_similarity(self, container: Tag, attribute: str, depth=4) -> Optional[AttributeData]:
        """Find attribute value using semantic similarity."""
        if not self._model_loaded:
            return None
        
        if depth <= 0:
            return None
        # Get tag similarity score with common attributes
        common_attr_score = self._get_element_similarity_to_common_attr(container)
        try:
            candidates = []
            elements = []
            fallback_text = str()
            fallback_tag = None

            for tag in container.find_all(recursive=False):
                # if tag name is kind of text container set fallback text if not already set
                if tag.name in self.TEXT_TAGS and not fallback_text:
                    text = self._get_element_text(tag)
                    fallback_text = text if text else ""
                    fallback_tag = tag
                
                # Collect potential matching texts
                classes = ' '.join(tag.get('class', []))
                tag_id = tag.get('id', '')
                tag_name = tag.get('name', '')
                itemprop = tag.get('itemprop', '')

                for text in [classes, tag_id, tag_name, itemprop]:
                    if text.strip():
                        candidates.append(text)
                        elements.append(tag)

                # Check data-* attributes
                for attr_key, attr_value in tag.attrs.items():
                    if attr_key.startswith('data-') and attr_value:
                        candidates.append(attr_value)
                        elements.append(tag)
            
            logging.info(f"Found {len(candidates)} candidates for similarity matching of attribute '{attribute}'.")
            # if fallback_tag is None:
            #     # use container text for fallback text
            #     fallback_text = self._get_element_text(container)
                # fallback_tag = container

            if not candidates:
                logging.info(f"Common attributes similarity score for container is {common_attr_score:.2f}. Fallback text: {fallback_text} with score {common_attr_score:.2f}.")
                if common_attr_score > 0.0 and fallback_tag is not None:
                    logging.info(f"No candidates found, but common attributes similarity score is {common_attr_score:.2f}. Returning fallback text. {fallback_text}")
                    return AttributeData(AttributeName=attribute, Value=fallback_text, Score=common_attr_score, PathChains=[TagIdentifier(TagName=fallback_tag.name, Attributes=fallback_tag.attrs)])
                else:
                    logging.info(f"No candidates found and common attributes similarity score is {common_attr_score:.2f}. Returning None.")
                    return None
            
            if candidates and len(candidates) > 0:
                logging.info(f"Candidates for similarity matching: {candidates}")
                # Calculate similarities
                attribute_embedding = self.similarity_model.encode([attribute])
                candidate_embeddings = self.similarity_model.encode(candidates)
                
                similarities = cosine_similarity(attribute_embedding, candidate_embeddings)[0]
                logging.info(f"Similarity scores for attribute '{attribute}': {similarities}")
                
                # Find best match above threshold
                best_idx = np.argmax(similarities)
                similarity_score = similarities[best_idx]
                best_element = (candidates[best_idx], similarity_score)
                best_element_tag = elements[best_idx]

                logging.info(f"Best similarity match for attribute '{attribute}': '{best_element[0]}' with score {best_element[1]:.2f}")
                
                # If the best element is a container tag and similarity score more than threshold, search recursively for that container only
                # But if similarity score is low, search it recursively for all other tags and get the highest similarity match
                if best_element_tag.name in self.CONTAINER_TAGS and best_element_tag.find_all(recursive=False):
                    if similarity_score >= self.similarity_threshold:
                        logging.info(f"Best match is a container tag '{best_element_tag.name}', searching recursively.")
                        result = self._find_by_similarity(best_element_tag, attribute, depth=depth-1)
                        # If the found score is better than current similarity score, return it
                        if result:
                            if result.Score > similarity_score:
                                return AttributeData(AttributeName=attribute, Value=result.Value, Score=result.Score, PathChains=[
                                    TagIdentifier(TagName=best_element_tag.name, Attributes=best_element_tag.attrs)
                                ] + result.PathChains)

                        # If not, compare similarity score with common attributes similarity score
                        # If common attributes score is better than similarity score, return fallback text if available
                        if common_attr_score > similarity_score and fallback_tag is not None:
                            return AttributeData(AttributeName=attribute, Value=fallback_text, Score=common_attr_score, PathChains=[
                                TagIdentifier(TagName=fallback_tag.name, Attributes=fallback_tag.attrs)
                            ])
                        
                        # But if not, return None with similarity score 0, because it is likely not found
                        return None
                    else:
                        # Search recursively in all div/span tags and get the highest similarity match
                        logging.info(f"Best match is a container tag '{best_element_tag.name}' but similarity score {similarity_score:.2f} is below threshold {self.similarity_threshold}, searching all div/span tags recursively.")
                        highest_score = 0.0
                        result = None
                        evaluated_elements = set()
                        for element in elements:
                            if element in evaluated_elements:
                                continue
                            evaluated_elements.add(element)
                            if element.name in self.CONTAINER_TAGS and element.find_all(recursive=False):
                                logging.info(f"Searching in container {str(element)[:200]}...")
                                res = self._find_by_similarity(element, attribute, depth=depth-1)
                                if res:
                                    if res.Score > highest_score or result is None:
                                        highest_score = res.Score
                                        result = res

                        logging.info(f"Highest similarity match from other containers: '{result}' with score {highest_score:.2f}, while the similarity score of the best element was {similarity_score:.2f}.")
                        # If the found score is better than current similarity score, return it
                        if highest_score > similarity_score:
                            return AttributeData(AttributeName=attribute, Value=result.Value, Score=result.Score, PathChains=[
                                TagIdentifier(TagName=best_element_tag.name, Attributes=best_element_tag.attrs)
                            ] + result.PathChains)

                        # If not, compare similarity score with common attributes similarity score
                        # If common attributes score is better than similarity score, return fallback text if available
                        if common_attr_score > similarity_score and fallback_tag is not None:
                            return AttributeData(AttributeName=attribute, Value=fallback_text, Score=common_attr_score, PathChains=[
                                TagIdentifier(TagName=fallback_tag.name, Attributes=fallback_tag.attrs)
                            ])

                        # But if not, return None with similarity score 0, because it is likely not found
                        return None
                # If the best element is not a container tag, return its text if similarity is above threshold
                else:
                    logging.info(f"Best match is a non-container tag '{best_element_tag.name}'.")
                    # If similarity score is above threshold, return the text
                    if similarity_score >= self.similarity_threshold:
                        return AttributeData(AttributeName=attribute, Value=self._get_element_text(best_element_tag), Score=similarity_score, PathChains=[
                            TagIdentifier(TagName=best_element_tag.name, Attributes=best_element_tag.attrs)
                        ])
                    
                    # If not, compare similarity score with common attributes similarity score
                    # If common attributes score is better than similarity score, return fallback text if available
                    logging.info(f"Best match similarity score {similarity_score:.2f} is below threshold {self.similarity_threshold}. Comparing with common attributes similarity score {common_attr_score:.2f}.")
                    # if common_attr_score > similarity_score:
                    # return fallback_text, similarity_score
                    if fallback_tag is not None:
                        return AttributeData(AttributeName=attribute, Value=fallback_text, Score=similarity_score, PathChains=[
                            TagIdentifier(TagName=fallback_tag.name, Attributes=fallback_tag.attrs)
                        ])

                    # But if not, return None with similarity score 0, because it is likely not found
                    # return None, 0.0

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

    def _all_attributes_high_confidence(self, attributes: Dict[str, Optional[AttributeData]]) -> bool:
        logging.info(f"Checking if all attributes have high confidence scores.")
        if attributes is None:
            return False
        for attr, attr_data in attributes.items():
            if attr_data is None or attr_data.Score < self.similarity_threshold:
                return False
        return True

    def _overall_attributes_confidence(self, attributes: Dict[str, Optional[AttributeData]]) -> float:
        logging.info(f"Calculating overall confidence score for attributes.")
        if attributes is None:
            return 0.0
        
        """Calculate overall confidence score as average of individual attribute scores."""
        total_score = 0.0
        count = 0
        for attr, attr_data in attributes.items():
            if attr_data is not None:
                total_score += attr_data.Score
            count += 1

        return total_score / count if count > 0 else 0.0

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