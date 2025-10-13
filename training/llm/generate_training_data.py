import json
import os
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
import random
import re
from pathlib import Path

class TrainingDataGenerator:
    """
    Generate training data from existing HTML samples
    """
    
    def __init__(self, samples_dir: str = "../../samples"):
        self.samples_dir = samples_dir
        self.instructions_templates = {
            "products": [
                "Extract product name, price, and description from the following HTML:",
                "Get the product title, cost, and details from this HTML:",
                "Parse product information including name, price, and description:",
                "Extract product data: name, price, and description from the HTML below:"
            ],
            "books": [
                "Extract book title, author, and price from the following HTML:",
                "Get the book name, writer, and cost from this HTML:",
                "Parse book information including title, author, and price:",
                "Extract book data: title, author, and price from the HTML below:"
            ],
            "jobs": [
                "Extract job title, company, and location from the following HTML:",
                "Get the position name, employer, and workplace location from this HTML:",
                "Parse job information including title, company, and location:",
                "Extract job data: title, company, and location from the HTML below:"
            ],
            "properties": [
                "Extract property name, price, and address from the following HTML:",
                "Get the property title, cost, and location from this HTML:",
                "Parse property information including name, price, and address:",
                "Extract property data: name, price, and address from the HTML below:"
            ],
            "clubs": [
                "Extract club name, location, and membership fee from the following HTML:",
                "Get the club title, address, and fee from this HTML:",
                "Parse club information including name, location, and membership cost:",
                "Extract club data: name, location, and membership fee from the HTML below:"
            ],
            "general": [
                "Extract relevant information from the following HTML:",
                "Parse the data from this HTML content:",
                "Get the important details from the HTML below:",
                "Extract structured data from the following HTML:"
            ]
        }
    
    def read_html_file(self, file_path: str) -> str:
        """Read HTML file content"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_text_from_elements(self, soup: BeautifulSoup, selectors: List[str]) -> str:
        """Extract text from HTML elements using CSS selectors"""
        for selector in selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        return ""
    
    def extract_attribute_from_elements(self, soup: BeautifulSoup, selectors: List[str], attr: str) -> str:
        """Extract attribute value from HTML elements"""
        for selector in selectors:
            element = soup.select_one(selector)
            if element and element.get(attr):
                return element.get(attr)
        return ""
    
    def generate_product_data(self, html_content: str) -> List[Dict]:
        """Generate training data for products"""
        soup = BeautifulSoup(html_content, 'html.parser')
        training_samples = []
        
        # Find product containers
        product_containers = soup.find_all(['div', 'article', 'section'], 
                                         class_=re.compile(r'product|item|card'))
        
        for container in product_containers:
            container_soup = BeautifulSoup(str(container), 'html.parser')
            
            # Extract product information
            name = self.extract_text_from_elements(container_soup, [
                '.product-name', '.product-title', '.title', '.name',
                'h1', 'h2', 'h3', '.product h1', '.product h2', '.product h3'
            ])
            
            price = self.extract_text_from_elements(container_soup, [
                '.price', '.cost', '.amount', '.product-price',
                '[class*="price"]', '[class*="cost"]'
            ])
            
            description = self.extract_text_from_elements(container_soup, [
                '.description', '.desc', '.details', '.summary',
                'p', '.product-description', '.info'
            ])
            
            image_url = self.extract_attribute_from_elements(container_soup, [
                'img', '.product-image img', '.image img'
            ], 'src')
            
            # Create training sample if we have minimum required data
            if name and (price or description):
                instruction = random.choice(self.instructions_templates["products"])
                
                output_data = {"name": name}
                if price:
                    output_data["price"] = price
                if description:
                    output_data["description"] = description
                if image_url:
                    output_data["image_url"] = image_url
                
                training_samples.append({
                    "instruction": instruction,
                    "input": str(container),
                    "output": json.dumps(output_data, ensure_ascii=False)
                })
        
        return training_samples
    
    def generate_book_data(self, html_content: str) -> List[Dict]:
        """Generate training data for books"""
        soup = BeautifulSoup(html_content, 'html.parser')
        training_samples = []
        
        # Find book containers
        book_containers = soup.find_all(['div', 'article', 'section'], 
                                       class_=re.compile(r'book|item|card'))
        
        for container in book_containers:
            container_soup = BeautifulSoup(str(container), 'html.parser')
            
            # Extract book information
            title = self.extract_text_from_elements(container_soup, [
                '.book-title', '.title', '.name', 'h1', 'h2', 'h3'
            ])
            
            author = self.extract_text_from_elements(container_soup, [
                '.author', '.writer', '.by', '[class*="author"]'
            ])
            
            price = self.extract_text_from_elements(container_soup, [
                '.price', '.cost', '.amount', '[class*="price"]'
            ])
            
            if title and author:
                instruction = random.choice(self.instructions_templates["books"])
                
                output_data = {"title": title, "author": author}
                if price:
                    output_data["price"] = price
                
                training_samples.append({
                    "instruction": instruction,
                    "input": str(container),
                    "output": json.dumps(output_data, ensure_ascii=False)
                })
        
        return training_samples
    
    def generate_job_data(self, html_content: str) -> List[Dict]:
        """Generate training data for jobs"""
        soup = BeautifulSoup(html_content, 'html.parser')
        training_samples = []
        
        # Find job containers
        job_containers = soup.find_all(['div', 'article', 'section'], 
                                      class_=re.compile(r'job|position|listing'))
        
        for container in job_containers:
            container_soup = BeautifulSoup(str(container), 'html.parser')
            
            # Extract job information
            title = self.extract_text_from_elements(container_soup, [
                '.job-title', '.position', '.title', 'h1', 'h2', 'h3'
            ])
            
            company = self.extract_text_from_elements(container_soup, [
                '.company', '.employer', '.organization', '[class*="company"]'
            ])
            
            location = self.extract_text_from_elements(container_soup, [
                '.location', '.address', '.city', '[class*="location"]'
            ])
            
            if title and company:
                instruction = random.choice(self.instructions_templates["jobs"])
                
                output_data = {"title": title, "company": company}
                if location:
                    output_data["location"] = location
                
                training_samples.append({
                    "instruction": instruction,
                    "input": str(container),
                    "output": json.dumps(output_data, ensure_ascii=False)
                })
        
        return training_samples
    
    def generate_property_data(self, html_content: str) -> List[Dict]:
        """Generate training data for properties"""
        soup = BeautifulSoup(html_content, 'html.parser')
        training_samples = []
        
        # Find property containers
        property_containers = soup.find_all(['div', 'article', 'section'], 
                                           class_=re.compile(r'property|listing|real-estate'))
        
        for container in property_containers:
            container_soup = BeautifulSoup(str(container), 'html.parser')
            
            # Extract property information
            name = self.extract_text_from_elements(container_soup, [
                '.property-name', '.title', '.name', 'h1', 'h2', 'h3'
            ])
            
            price = self.extract_text_from_elements(container_soup, [
                '.price', '.cost', '.amount', '[class*="price"]'
            ])
            
            address = self.extract_text_from_elements(container_soup, [
                '.address', '.location', 'address', '[class*="address"]'
            ])
            
            if name and price:
                instruction = random.choice(self.instructions_templates["properties"])
                
                output_data = {"name": name, "price": price}
                if address:
                    output_data["address"] = address
                
                training_samples.append({
                    "instruction": instruction,
                    "input": str(container),
                    "output": json.dumps(output_data, ensure_ascii=False)
                })
        
        return training_samples
    
    def generate_club_data(self, html_content: str) -> List[Dict]:
        """Generate training data for clubs"""
        soup = BeautifulSoup(html_content, 'html.parser')
        training_samples = []
        
        # Find club containers
        club_containers = soup.find_all(['div', 'article', 'section'], 
                                       class_=re.compile(r'club|organization|group'))
        
        for container in club_containers:
            container_soup = BeautifulSoup(str(container), 'html.parser')
            
            # Extract club information
            name = self.extract_text_from_elements(container_soup, [
                '.club-name', '.title', '.name', 'h1', 'h2', 'h3'
            ])
            
            location = self.extract_text_from_elements(container_soup, [
                '.location', '.address', '[class*="location"]'
            ])
            
            fee = self.extract_text_from_elements(container_soup, [
                '.fee', '.membership', '.cost', '.price', '[class*="fee"]'
            ])
            
            if name:
                instruction = random.choice(self.instructions_templates["clubs"])
                
                output_data = {"name": name}
                if location:
                    output_data["location"] = location
                if fee:
                    output_data["membership_fee"] = fee
                
                training_samples.append({
                    "instruction": instruction,
                    "input": str(container),
                    "output": json.dumps(output_data, ensure_ascii=False)
                })
        
        return training_samples
    
    def generate_training_data(self, output_file: str = "generated_training_data.json"):
        """Generate training data from all HTML samples"""
        all_samples = []
        samples_path = Path(self.samples_dir)
        
        if not samples_path.exists():
            print(f"Samples directory not found: {self.samples_dir}")
            return
        
        # Process each HTML file
        for html_file in samples_path.glob("*.html"):
            print(f"Processing {html_file.name}...")
            html_content = self.read_html_file(str(html_file))
            
            # Generate data based on file name
            if "product" in html_file.name.lower() or "ecommerce" in html_file.name.lower():
                samples = self.generate_product_data(html_content)
                all_samples.extend(samples)
                print(f"  Generated {len(samples)} product samples")
            
            elif "book" in html_file.name.lower():
                samples = self.generate_book_data(html_content)
                all_samples.extend(samples)
                print(f"  Generated {len(samples)} book samples")
            
            elif "job" in html_file.name.lower():
                samples = self.generate_job_data(html_content)
                all_samples.extend(samples)
                print(f"  Generated {len(samples)} job samples")
            
            elif "property" in html_file.name.lower() or "properties" in html_file.name.lower():
                samples = self.generate_property_data(html_content)
                all_samples.extend(samples)
                print(f"  Generated {len(samples)} property samples")
            
            elif "club" in html_file.name.lower():
                samples = self.generate_club_data(html_content)
                all_samples.extend(samples)
                print(f"  Generated {len(samples)} club samples")
        
        # Shuffle the data
        random.shuffle(all_samples)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        print(f"\nGenerated {len(all_samples)} training samples")
        print(f"Saved to {output_file}")
        
        return all_samples

def main():
    """Generate training data from HTML samples"""
    generator = TrainingDataGenerator()
    training_data = generator.generate_training_data("enhanced_training_data.json")
    
    if training_data:
        print(f"\nSample training data:")
        for i, sample in enumerate(training_data[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Instruction: {sample['instruction']}")
            print(f"Input: {sample['input'][:200]}...")
            print(f"Output: {sample['output']}")

if __name__ == "__main__":
    main()