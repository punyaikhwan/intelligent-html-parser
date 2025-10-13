import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import Dict, List
import logging
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HTMLParserModel:
    def __init__(self, model_path: str):
        """
        Initialize the trained HTML parser model
        
        Args:
            model_path: Path to the trained model directory
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully!")
    
    def parse_html(self, instruction: str, html_content: str, max_length: int = 256) -> str:
        """
        Parse HTML content based on instruction
        
        Args:
            instruction: What to extract from HTML
            html_content: HTML content to parse
            max_length: Maximum length of generated output
            
        Returns:
            Extracted data as JSON string
        """
        # Format input
        input_text = f"{instruction} {html_content}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False
            )
        
        # Decode output
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    
    def parse_html_batch(self, instructions: List[str], html_contents: List[str]) -> List[str]:
        """
        Parse multiple HTML contents in batch
        
        Args:
            instructions: List of instructions
            html_contents: List of HTML contents
            
        Returns:
            List of extracted data as JSON strings
        """
        results = []
        for instruction, html_content in zip(instructions, html_contents):
            result = self.parse_html(instruction, html_content)
            results.append(result)
        return results

def evaluate_model(model_path: str, test_file: str):
    """
    Evaluate the trained model on test data
    
    Args:
        model_path: Path to trained model
        test_file: Path to test data JSON file
    """
    # Load model
    parser = HTMLParserModel(model_path)
    
    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"Evaluating on {len(test_data)} test samples...")
    
    correct = 0
    total = len(test_data)
    
    for i, example in enumerate(test_data):
        # Get prediction
        prediction = parser.parse_html(
            example['instruction'], 
            example['input']
        )
        
        # Compare with expected output
        expected = example['output']
        
        print(f"\n--- Example {i+1} ---")
        print(f"Instruction: {example['instruction']}")
        print(f"Expected: {expected}")
        print(f"Predicted: {prediction}")
        
        # Simple accuracy check (you can make this more sophisticated)
        if prediction.strip() == expected.strip():
            correct += 1
            print("✓ CORRECT")
        else:
            print("✗ INCORRECT")
    
    accuracy = correct / total * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}% ({correct}/{total})")

def demo_inference(model_path: str):
    """
    Demo inference with the trained model
    
    Args:
        model_path: Path to trained model
    """
    parser = HTMLParserModel(model_path)
    
    # Example HTML content
    html_examples = [
        {
            "instruction": "Extract product name, price, and description from the following HTML:",
            "html": """
            <div class="product">
                <h2 class="product-title">MacBook Air M2</h2>
                <span class="price">$1,199</span>
                <p class="description">Supercharged by M2 chip. Incredibly thin and light design.</p>
                <img src="/images/macbook-air.jpg" alt="MacBook Air">
            </div>
            """
        },
        {
            "instruction": "Get the book title, author, and genre from this HTML:",
            "html": """
            <article class="book">
                <h1>1984</h1>
                <p class="author">George Orwell</p>
                <span class="genre">Dystopian Fiction</span>
                <div class="publisher">Penguin Books</div>
            </article>
            """
        },
        {
            "instruction": "Extract job title, company, and salary from the following HTML:",
            "html": """
            <div class="job-posting">
                <h3>Data Scientist</h3>
                <div class="company">Google</div>
                <span class="salary">$120,000 - $180,000</span>
                <p class="location">Mountain View, CA</p>
            </div>
            """
        }
    ]
    
    print("=== DEMO INFERENCE ===\n")
    
    for i, example in enumerate(html_examples, 1):
        print(f"--- Example {i} ---")
        print(f"Instruction: {example['instruction']}")
        print(f"HTML: {example['html'].strip()}")
        
        result = parser.parse_html(example['instruction'], example['html'])
        print(f"Result: {result}")
        print()

def test_json_comparison():
    """
    Test the JSON comparison logic with various cases
    """
    print("=== Testing JSON Comparison Logic ===\n")
    
    test_cases = [
        # Case 1: Missing brackets in prediction
        {
            "prediction": '"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "price": "$12.99"',
            "expected": '{\n  "title": "The Great Gatsby",\n  "author": "F. Scott Fitzgerald",\n  "price": "$12.99"\n}',
            "should_match": True
        },
        # Case 2: Same content, different formatting
        {
            "prediction": '{"title":"The Great Gatsby","author":"F. Scott Fitzgerald","price":"$12.99"}',
            "expected": '{\n  "title": "The Great Gatsby",\n  "author": "F. Scott Fitzgerald",\n  "price": "$12.99"\n}',
            "should_match": True
        },
        # Case 3: Different content
        {
            "prediction": '"title": "1984", "author": "George Orwell"',
            "expected": '{\n  "title": "The Great Gatsby",\n  "author": "F. Scott Fitzgerald"\n}',
            "should_match": False
        },
        # Case 4: Same keys, different order
        {
            "prediction": '"author": "F. Scott Fitzgerald", "title": "The Great Gatsby", "price": "$12.99"',
            "expected": '{\n  "title": "The Great Gatsby",\n  "price": "$12.99",\n  "author": "F. Scott Fitzgerald"\n}',
            "should_match": True
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"  Prediction: {case['prediction']}")
        print(f"  Expected: {case['expected']}")
        
        result = compare_json_outputs(case['prediction'], case['expected'])
        expected_result = case['should_match']
        
        if result == expected_result:
            print(f"  ✓ PASS (Result: {result})")
        else:
            print(f"  ✗ FAIL (Expected: {expected_result}, Got: {result})")
        
        print()

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate FLAN-T5 HTML Parser")
    parser.add_argument("--model_path", type=str, default="./flan-t5-html-parser", 
                       help="Path to trained model")
    parser.add_argument("--test_file", type=str, default="training_samples.json",
                       help="Path to test data JSON file")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo inference")
    parser.add_argument("--test_json", action="store_true",
                       help="Test JSON comparison logic")
    
    args = parser.parse_args()
    
    if args.test_json:
        test_json_comparison()
    elif args.demo:
        demo_inference(args.model_path)
    else:
        evaluate_model(args.model_path, args.test_file)

if __name__ == "__main__":
    main()