"""
Query parser module for extracting entities and attributes from natural language queries.
"""
import re
from typing import List, Tuple, Optional

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
    
    # Download required NLTK data if not already present
    # Always ensure punkt is available first
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # Download punkt_tab but don't fail if it's not available
    try:
        nltk.download('punkt_tab', quiet=True)
    except:
        pass  # punkt_tab is optional, punkt is sufficient
    
    # Try the newer English-specific tagger first
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        try:
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        except:
            # Fall back to the older general tagger
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', quiet=True)
        
except ImportError:
    NLTK_AVAILABLE = False


class QueryParser:
    """Extracts entities and attributes from natural language queries using rule-based approach."""
    
    # Stopwords to remove from the beginning of queries
    FRONT_STOPWORDS = {
        'get', 'list', 'return', 'give', 'show', 'please', 'from', 'me', 'can', 'you',
        'extract', 'find', 'retrieve', 'fetch', 'obtain', 'collect', 'gather', 'pull',
        'the', 'all', 'any', 'some', 'what', 'which', 'how', 'where', 'their', 'its',
        'a', 'an', 'this', 'that', 'these', 'those'
    }
    
    # Stopwords to remove from the end of queries
    END_STOPWORDS = {
        'from', 'in', 'on', 'at', 'of', 'for', 'with', 'by', 'to', 'into',
        'the', 'page', 'website', 'document', 'html', 'content', 'listings',
        'items', 'elements', 'data', 'information'
    }
    
    def __init__(self):
        pass

    def parse_query(self, query: str) -> Tuple[Optional[str], List[str], str, str]:
        """
        Parse a natural language query to extract entity and attributes.
        
        Args:
            query: Natural language query string
            
        Returns:
            Tuple of (entity, attributes_list, entity_extraction_approach, attribute_extraction_approach)
        """
        # Clean and normalize the query
        cleaned_query = self._clean_query(query)
        
        # Extract entity using rule-based approach
        entity, entity_extraction_approach = self._extract_entity(cleaned_query)
        
        # Extract attributes using rule-based approach
        attributes, attribute_extraction_approach = self._extract_attributes(cleaned_query, entity)

        return entity, attributes, entity_extraction_approach, attribute_extraction_approach
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query string."""
        # Convert to lowercase and strip whitespace
        query = query.lower().strip()
        
        # Remove punctuation at the end
        query = re.sub(r'[.!?]+$', '', query)
        
        # Remove question marks and colons
        query = query.replace('?', '').replace(':', ' ')
        
        return query
    
    def _extract_entity(self, query: str) -> Tuple[Optional[str], str]:
        """
        Extract entity from the query using rule-based approach with POS tagging.
        Look for patterns like "the [entity]" or use POS tagging to find nouns.
        """
        words = query.split()
        
        # Remove front stopwords except 'the'
        while words and words[0] in self.FRONT_STOPWORDS - {'the'}:
            words.pop(0)

        # Remove end stopwords except 'the'
        while words and words[-1] in self.END_STOPWORDS - {'the'}:
            words.pop()
        
        if not words:
            return None
        
        # Look for "the [entity]" pattern first (high confidence)
        for i, word in enumerate(words):
            if word == 'the' and i + 1 < len(words):
                potential_entity = words[i + 1]
                potential_entity = self._singularize_noun(potential_entity)
                print(f"Detected entity using 'the' pattern: {potential_entity}")
                return potential_entity, "the-pattern"
        
        # Use POS tagging to find nouns if NLTK is available
        if NLTK_AVAILABLE:
            entity = self._extract_entity_with_pos(query)
            if entity:
                print(f"Detected entity using POS tagging: {entity}")
                return entity, "pos-tagging"
        
        # Fallback: simplified approach - look for the first meaningful noun
        # This is used when NLTK is not available or POS tagging fails
        for word in words:
            if word not in self.FRONT_STOPWORDS and word not in self.END_STOPWORDS:
                if len(word) > 2:  # Simple heuristic for meaningful words
                    word = self._singularize_noun(word)
                    print(f"Detected entity using simple approach: {word}")
                    return word, "simple-heuristic"
        
        return None
    
    def _extract_entity_with_pos(self, query: str) -> Optional[str]:
        """
        Extract entity using POS tagging to identify nouns.
        
        Args:
            query: Cleaned query string
            
        Returns:
            Detected entity or None if not found
        """
        try:
            # Tokenize the query
            tokens = word_tokenize(query)
            
            # Get POS tags
            pos_tags = pos_tag(tokens)
            
            # Look for nouns (NN, NNS, NNP, NNPS)
            # Priority: Proper nouns (NNP, NNPS) > Common nouns (NN, NNS)
            proper_nouns = []
            common_nouns = []
            
            for word, pos in pos_tags:
                word_lower = word.lower()
                
                # Skip stopwords
                if word_lower in self.FRONT_STOPWORDS or word_lower in self.END_STOPWORDS:
                    continue
                
                # Skip very short words
                if len(word) < 3:
                    continue
                
                # Collect nouns by type
                if pos in ['NNP', 'NNPS']:  # Proper nouns
                    proper_nouns.append(word_lower)
                elif pos in ['NN', 'NNS']:  # Common nouns
                    common_nouns.append(word_lower)
            
            # Prefer proper nouns, then common nouns
            candidate_nouns = proper_nouns + common_nouns
            
            if candidate_nouns:
                # Take the first meaningful noun
                entity = candidate_nouns[0]
                
                # Convert plural to singular
                entity = self._singularize_noun(entity)
                
                return entity
            
        except Exception as e:
            # If POS tagging fails, return None to fall back to simple approach
            print(f"POS tagging failed: {e}")
            return None
        
        return None
    
    def _singularize_noun(self, noun: str) -> str:
        """
        Convert plural noun to singular form using simple rules.
        
        Args:
            noun: Plural noun to convert
            
        Returns:
            Singular form of the noun
        """
        if not noun:
            return noun
        
        # Handle common irregular plurals
        irregular_plurals = {
            'children': 'child',
            'people': 'person',
            'men': 'man',
            'women': 'woman',
            'feet': 'foot',
            'teeth': 'tooth',
            'mice': 'mouse',
            'geese': 'goose'
        }
        
        if noun in irregular_plurals:
            return irregular_plurals[noun]
        
        # Handle regular plural patterns
        if noun.endswith('ies') and len(noun) > 3:
            # companies -> company, stories -> story
            return noun[:-3] + 'y'
        elif noun.endswith('ves') and len(noun) > 3:
            # knives -> knife, wolves -> wolf
            return noun[:-3] + 'f'
        elif noun.endswith('ses') and len(noun) > 3:
            # glasses -> glass, classes -> class
            return noun[:-2]
        elif noun.endswith('es') and len(noun) > 2:
            # boxes -> box, dishes -> dish
            if noun.endswith(('ches', 'shes', 'xes', 'zes')):
                return noun[:-2]
            else:
                return noun[:-1]
        elif noun.endswith('s') and len(noun) > 1:
            # books -> book, cars -> car
            return noun[:-1]
        
        return noun

    def _pluralize_noun(self, noun: str) -> str:
        """
        Convert singular noun to plural form using simple rules.
        
        Args:
            noun: Singular noun to convert
            
        Returns:
            Plural form of the noun
        """
        if not noun:
            return noun
        
        # Handle common irregular singulars
        irregular_singulars = {
            'child': 'children',
            'person': 'people',
            'man': 'men',
            'woman': 'women',
            'foot': 'feet',
            'tooth': 'teeth',
            'mouse': 'mice',
            'goose': 'geese'
        }

        if noun in irregular_singulars:
            return irregular_singulars[noun]

        # Handle regular plural patterns
        if noun.endswith('y') and len(noun) > 2:
            # baby -> babies, city -> cities
            return noun[:-1] + 'ies'
        elif noun.endswith('f') and len(noun) > 2:
            # knife -> knives, wolf -> wolves
            return noun[:-1] + 'ves'
        elif noun.endswith('s') and len(noun) > 2:
            # glass -> glasses, class -> classes
            return noun
        elif noun.endswith('o') and len(noun) > 2:
            # photo -> photos, piano -> pianos
            return noun + 's'
        elif len(noun) > 1:
            # book -> books, car -> cars
            return noun + 's'

        return noun

    def _extract_attributes(self, query: str, entity: Optional[str]) -> Tuple[List[str], str]:
        """
        Extract attributes from the query using rule-based approach.
        Split by 'and' and commas to find attribute lists.
        """
        # Remove entity from query if found
        if entity:
            # Try both singular and plural forms
            plural_form = self._pluralize_noun(entity)
            entity_patterns = [entity, plural_form, 'the ' + entity, 'the ' + plural_form]
            for pattern in entity_patterns:
                query = query.replace(pattern, '')
        
        # Remove common separators and connectors
        query = re.sub(r'\b(and|with|including|such as|like)\b', ',', query)
        
        # Split by commas first
        parts = [part.strip() for part in query.split(',')]
        
        attributes = []
        for part in parts:
            if not part:
                continue
                
            # Further split by 'and' in case there are nested conjunctions
            sub_parts = [sub_part.strip() for sub_part in part.split(' and ')]
            
            for sub_part in sub_parts:
                if not sub_part:
                    continue
                
                # Clean the attribute
                cleaned_attr = self._clean_attribute(sub_part)
                if cleaned_attr and cleaned_attr not in attributes:
                    attributes.append(cleaned_attr)
        
        return attributes, "rule-based"
    
    def _clean_attribute(self, attribute: str) -> Optional[str]:
        """Clean individual attribute strings."""
        # Remove common words and articles
        words = attribute.split()
        cleaned_words = []
        
        for word in words:
            # Skip stopwords and articles
            if word not in self.FRONT_STOPWORDS and word not in self.END_STOPWORDS:
                # Remove common prefixes
                word = re.sub(r'^(its?|their|the|a|an)\s+', '', word)
                if word and len(word) > 1:
                    cleaned_words.append(word)
        
        if not cleaned_words:
            return None
        
        # Join words with space and return
        result = ' '.join(cleaned_words)
        
        # Skip if it's too short or contains only common words
        if len(result) < 2:
            return None
            
        return result


def test_query_parser():
    """Test function for the QueryParser class."""
    parser = QueryParser()
    
    test_cases = [
        "Can you give me the book: name and price?",
        "Extract job title, location, salary, and company name from the listings",
        "Get the product name, price, and description",
        "Show me all the movie titles and ratings",
        "List book author, title, price and rating",
        "Find all companies with their revenue and employees",
        "Get customer information including names and addresses",
        "Extract vehicle details like model and year"
    ]
    
    print(f"NLTK Available: {NLTK_AVAILABLE}")
    print("=" * 60)
    
    for query in test_cases:
        entity, attributes, entity_approach, attr_approach = parser.parse_query(query)
        print(f"Query: {query}")
        print(f"Entity: {entity}")
        print(f"Attributes: {attributes}")
        print(f"Entity Extraction Approach: {entity_approach}")
        print(f"Attribute Extraction Approach: {attr_approach}")
        print("-" * 50)


if __name__ == "__main__":
    test_query_parser()