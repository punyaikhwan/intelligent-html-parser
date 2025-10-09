"""
HTML table parser for extracting structured data from HTML tables.
"""
import logging
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup, Tag
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("Sentence transformers not available. Will use exact string matching only.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Will use exact string matching only.")


class TableParser:
    """Parser specifically designed for HTML tables."""
    
    def __init__(self, similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 similarity_threshold: float = 0.7):
        """
        Initialize the table parser.
        
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
    
    def is_table(self, soup: BeautifulSoup) -> bool:
        """Check if the HTML contains tables."""
        tables = soup.find_all('table')
        return len(tables) > 0
    
    def parse_tables(self, html: str, entity: str, attributes: List[str]) -> List[Dict[str, Any]]:
        """
        Parse HTML tables to extract data based on entity and attributes.
        
        Args:
            html: HTML string to parse
            entity: Entity name to look for
            attributes: List of attributes to extract
            
        Returns:
            List of dictionaries containing extracted data
        """
        soup = BeautifulSoup(html, 'html.parser')
        tables = soup.find_all('table')
        
        if not tables:
            return []
        
        all_results = []
        
        for table in tables:
            results = self._parse_single_table(table, entity, attributes)
            all_results.extend(results)
        
        return all_results
    
    def _parse_single_table(self, table: Tag, entity: str, attributes: List[str]) -> List[Dict[str, Any]]:
        """Parse a single table element."""
        # Extract headers
        headers = self._extract_headers(table)
        if not headers:
            return []
        
        # Match attributes to headers
        header_mapping = self._match_attributes_to_headers(attributes, headers)
        
        # Extract rows data
        rows_data = self._extract_rows_data(table, header_mapping)
        
        return rows_data
    
    def _extract_headers(self, table: Tag) -> List[str]:
        """Extract header row from table."""
        headers = []
        
        # Try to find headers in various ways
        header_rows = []
        
        # Look for thead section
        thead = table.find('thead')
        if thead:
            header_rows = thead.find_all('tr')
        
        # If no thead, look for first tr with th elements
        if not header_rows:
            first_row = table.find('tr')
            if first_row and first_row.find('th'):
                header_rows = [first_row]
        
        # If still no headers, use first row
        if not header_rows:
            first_row = table.find('tr')
            if first_row:
                header_rows = [first_row]
        
        # Extract header text
        if header_rows:
            header_row = header_rows[0]
            header_cells = header_row.find_all(['th', 'td'])
            
            for cell in header_cells:
                header_text = self._clean_text(cell.get_text())
                headers.append(header_text)
        
        return headers
    
    def _match_attributes_to_headers(self, attributes: List[str], headers: List[str]) -> Dict[str, int]:
        """
        Match attributes to table headers using exact match and similarity.
        
        Args:
            attributes: List of attributes to find
            headers: List of table headers
            
        Returns:
            Dictionary mapping attribute to header index
        """
        mapping = {}
        
        for attribute in attributes:
            best_match_idx = None
            best_score = 0
            
            # First try exact string matching (case-insensitive)
            for i, header in enumerate(headers):
                if attribute.lower() == header.lower():
                    mapping[attribute] = i
                    best_match_idx = i
                    break
                
                # Try partial matching
                if attribute.lower() in header.lower() or header.lower() in attribute.lower():
                    if best_match_idx is None:
                        best_match_idx = i
                        best_score = 0.8  # High score for partial match
            
            # If no exact match and similarity model is available, try semantic similarity
            if best_match_idx is None and self._model_loaded:
                best_match_idx, best_score = self._find_most_similar_header(attribute, headers)
            
            # Add to mapping if we found a good match
            if best_match_idx is not None and best_score >= self.similarity_threshold:
                mapping[attribute] = best_match_idx
        
        return mapping
    
    def _find_most_similar_header(self, attribute: str, headers: List[str]) -> tuple:
        """Find the most similar header using semantic similarity."""
        if not self._model_loaded or not headers:
            return None, 0
        
        try:
            # Encode attribute and headers
            attribute_embedding = self.similarity_model.encode([attribute])
            header_embeddings = self.similarity_model.encode(headers)
            
            # Calculate similarities
            similarities = cosine_similarity(attribute_embedding, header_embeddings)[0]
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            return best_idx, best_score
            
        except Exception as e:
            logging.error(f"Error in similarity calculation: {e}")
            return None, 0
    
    def _extract_rows_data(self, table: Tag, header_mapping: Dict[str, int]) -> List[Dict[str, Any]]:
        """Extract data from table rows based on header mapping."""
        rows_data = []
        
        # Find all data rows (skip header row)
        all_rows = table.find_all('tr')
        
        # Skip the first row if it contains headers
        data_rows = all_rows[1:] if len(all_rows) > 1 else []
        
        for row in data_rows:
            cells = row.find_all(['td', 'th'])
            row_data = {}
            
            # Extract data for each mapped attribute
            for attribute, header_idx in header_mapping.items():
                if header_idx < len(cells):
                    cell_text = self._clean_text(cells[header_idx].get_text())
                    row_data[attribute] = cell_text
                else:
                    row_data[attribute] = ""
            
            # Only add row if it has at least some data
            if any(value.strip() for value in row_data.values()):
                rows_data.append(row_data)
        
        return rows_data
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        cleaned = ' '.join(text.split())
        
        # Remove common HTML artifacts
        cleaned = cleaned.replace('\xa0', ' ')  # Non-breaking space
        cleaned = cleaned.replace('\u2013', '-')  # En dash
        cleaned = cleaned.replace('\u2014', '-')  # Em dash
        
        return cleaned.strip()


def test_table_parser():
    """Test function for the table parser."""
    # Sample HTML table
    html = """
    <html>
    <body>
        <table>
            <thead>
                <tr>
                    <th>Book Title</th>
                    <th>Author</th>
                    <th>Price</th>
                    <th>Rating</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>The Great Gatsby</td>
                    <td>F. Scott Fitzgerald</td>
                    <td>$12.99</td>
                    <td>4.5</td>
                </tr>
                <tr>
                    <td>To Kill a Mockingbird</td>
                    <td>Harper Lee</td>
                    <td>$14.99</td>
                    <td>4.8</td>
                </tr>
            </tbody>
        </table>
    </body>
    </html>
    """
    
    parser = TableParser()
    entity = "book"
    attributes = ["title", "author", "price"]
    
    print("Testing table parser...")
    soup = BeautifulSoup(html, 'html.parser')
    print(f"Is table: {parser.is_table(soup)}")
    
    results = parser.parse_tables(html, entity, attributes)
    print(f"Results: {results}")


if __name__ == "__main__":
    test_table_parser()