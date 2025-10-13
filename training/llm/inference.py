import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Dict, List, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntelligentHTMLParser:
    """
    Intelligent HTML Parser using fine-tuned FLAN-T5 model
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the parser with trained model
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("HTML Parser model loaded successfully!")
    
    def extract_data(
        self, 
        instruction: str, 
        html_content: str, 
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Extract data from HTML based on instruction
        
        Args:
            instruction: Natural language instruction for what to extract
            html_content: HTML content to parse
            max_length: Maximum length of output
            temperature: Sampling temperature (0.0 = deterministic)
            top_p: Top-p sampling parameter
            
        Returns:
            Extracted data as JSON string
        """
        # Format input text
        input_text = f"{instruction} {html_content}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate output
        with torch.no_grad():
            if temperature > 0:
                # Use sampling for more diverse outputs
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            else:
                # Use beam search for deterministic output
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode result
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    
    def extract_products(self, html_content: str) -> str:
        """Extract product information from HTML"""
        instruction = "Extract product name, image url, price, and description from the following HTML:"
        return self.extract_data(instruction, html_content)
    
    def extract_books(self, html_content: str) -> str:
        """Extract book information from HTML"""
        instruction = "Get the book title, author, price, and description from this HTML:"
        return self.extract_data(instruction, html_content)
    
    def extract_jobs(self, html_content: str) -> str:
        """Extract job information from HTML"""
        instruction = "Extract job title, company, location, and salary from the following HTML:"
        return self.extract_data(instruction, html_content)
    
    def extract_properties(self, html_content: str) -> str:
        """Extract property information from HTML"""
        instruction = "Get the property name, price, address, and features from this real estate HTML:"
        return self.extract_data(instruction, html_content)
    
    def extract_custom(self, attributes: List[str], html_content: str) -> str:
        """
        Extract custom attributes from HTML
        
        Args:
            attributes: List of attributes to extract (e.g., ['name', 'price', 'description'])
            html_content: HTML content
            
        Returns:
            Extracted data as JSON string
        """
        attr_list = ", ".join(attributes)
        instruction = f"Extract {attr_list} from the following HTML:"
        return self.extract_data(instruction, html_content)
    
    def parse_result_to_dict(self, result: str) -> Optional[Dict]:
        """
        Parse the model output to dictionary
        
        Args:
            result: JSON string from model
            
        Returns:
            Dictionary or None if parsing fails
        """
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {result}")
            return None

# Example usage functions
def demo_usage():
    """Demonstrate how to use the parser"""
    
    # Initialize parser (make sure model path exists)
    model_path = "./flan-t5-html-parser"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please run training first: python flan_t5_training.py")
        return
    
    parser = IntelligentHTMLParser(model_path)
    
    # Example HTML content
    product_html = """
    <div class="product-card">
        <h2 class="product-name">Samsung Galaxy S24</h2>
        <img src="https://example.com/galaxy-s24.jpg" alt="Galaxy S24">
        <span class="price">$799.99</span>
        <p class="description">Latest Samsung flagship with AI features and improved camera.</p>
        <div class="rating">4.8 stars</div>
    </div>
    """
    
    # Extract product data
    print("=== Product Extraction ===")
    result = parser.extract_products(product_html)
    print(f"Raw result: {result}")
    
    # Parse to dictionary
    data = parser.parse_result_to_dict(result)
    if data:
        print("Parsed data:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    # Custom extraction
    print("\n=== Custom Extraction ===")
    custom_result = parser.extract_custom(
        ["name", "price", "rating"], 
        product_html
    )
    print(f"Custom result: {custom_result}")

def batch_process_files(model_path: str, html_files: List[str], output_file: str):
    """
    Process multiple HTML files and save results
    
    Args:
        model_path: Path to trained model
        html_files: List of HTML file paths
        output_file: Output JSON file path
    """
    parser = IntelligentHTMLParser(model_path)
    results = []
    
    for html_file in html_files:
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extract data (modify instruction as needed)
        result = parser.extract_data(
            "Extract all relevant information from the following HTML:",
            html_content
        )
        
        results.append({
            "file": html_file,
            "extracted_data": result
        })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(html_files)} files. Results saved to {output_file}")

if __name__ == "__main__":
    demo_usage()