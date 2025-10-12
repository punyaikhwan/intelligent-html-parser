"""
Test script for the JSON script parser functionality.
"""
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.intelligent_parser import IntelligentHTMLParser


def test_json_script_parser():
    """Test the JSON script parser with various scenarios."""
    
    # Test HTML with JSON-LD script
    html_with_jsonld = """
    <html>
    <head>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@type": "Product",
            "name": "iPhone 13",
            "price": "$699",
            "description": "Latest iPhone model with amazing features",
            "brand": "Apple",
            "offers": {
                "@type": "Offer",
                "price": "699.00",
                "priceCurrency": "USD"
            }
        }
        </script>
    </head>
    <body>
        <h1>iPhone 13</h1>
        <p>Price: $699</p>
    </body>
    </html>
    """
    
    # Test HTML with application/json script (like Next.js)
    html_with_json = """
    <html>
    <head>
        <script id="__NEXT_DATA__" type="application/json">
        {
            "props": {
                "pageProps": {
                    "products": [
                        {
                            "name": "MacBook Pro",
                            "price": "$1999",
                            "description": "Professional laptop",
                            "category": "Laptop"
                        },
                        {
                            "name": "iPad Air",
                            "price": "$599",
                            "description": "Lightweight tablet",
                            "category": "Tablet"
                        }
                    ]
                }
            }
        }
        </script>
    </head>
    <body>
        <h1>Apple Products</h1>
    </body>
    </html>
    """
    
    # Test HTML with club data (JSON-LD)
    html_with_clubs = """
    <html>
    <head>
        <script type="application/ld+json">
        {
            "@context": "https://schema.org",
            "@graph": [
                {
                    "@type": "Organization",
                    "name": "Phoenix Soccer Club",
                    "description": "Youth soccer club in Phoenix",
                    "address": "123 Main St, Phoenix, AZ",
                    "telephone": "555-123-4567"
                },
                {
                    "@type": "Organization", 
                    "name": "Tucson United FC",
                    "description": "Professional soccer club",
                    "address": "456 Oak Ave, Tucson, AZ",
                    "telephone": "555-987-6543"
                }
            ]
        }
        </script>
    </head>
    <body>
        <h1>Soccer Clubs</h1>
    </body>
    </html>
    """
    
    # Test HTML without JSON scripts (fallback test)
    html_without_json = """
    <html>
    <body>
        <table>
            <tr>
                <th>Book Title</th>
                <th>Author</th>
                <th>Price</th>
            </tr>
            <tr>
                <td>The Great Gatsby</td>
                <td>F. Scott Fitzgerald</td>
                <td>$12.99</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    parser = IntelligentHTMLParser()
    
    test_cases = [
        (html_with_jsonld, "Extract product name, price, and description", "JSON-LD Product"),
        (html_with_json, "Get product name, price, and description", "Next.js JSON Products"),
        (html_with_clubs, "Find club name, description, and address", "JSON-LD Clubs"),
        (html_without_json, "Extract book title, author, and price", "Fallback to Table Parser")
    ]
    
    print("Testing JSON Script Parser Integration")
    print("=" * 50)
    
    for i, (html, query, test_name) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_name}")
        print(f"Query: {query}")
        print("-" * 30)
        
        result = parser.parse(html, query)
        
        print(f"HTML Parsing Approach: {result['metadata']['approaches_used']['html_parsing']}")
        print(f"Results Found: {len(result['results'][list(result['results'].keys())[0]] if result['results'] else [])}")
        print(f"Message: {result['message']}")
        
        if result['results']:
            entity_key = list(result['results'].keys())[0]
            entities = result['results'][entity_key]
            for j, entity in enumerate(entities[:2], 1):  # Show first 2 results
                print(f"  Entity {j}: {entity}")
        
        print()
    
    # Test parser status
    print("Parser Status:")
    status = parser.get_parser_status()
    for parser_name, parser_status in status.items():
        print(f"  {parser_name}: {parser_status}")


def test_with_real_samples():
    """Test with real sample files that contain JSON scripts."""
    
    parser = IntelligentHTMLParser()
    
    # Test with real sample files
    sample_files = [
        ("samples/property.html", "Return the property name, address, latitude and longitude", "Real Property Sample")
        # ("samples/two-clubs.html", "Find club name and description", "Real Clubs Sample")
    ]
    
    print("\nTesting with Real Sample Files")
    print("=" * 50)
    
    for sample_file, query, test_name in sample_files:
        if os.path.exists(sample_file):
            print(f"\nTest: {test_name}")
            print(f"File: {sample_file}")
            print(f"Query: {query}")
            print("-" * 30)
            
            try:
                print(f"Reading file: {sample_file}")
                with open(sample_file, 'r', encoding='utf-8') as f:
                    html = f.read()
                print(f"File read successfully, length: {len(html)} characters")
                result = parser.parse(html, query)
                
                print(f"HTML Parsing Approach: {result['metadata']['approaches_used']['html_parsing']}")
                print(f"Results Found: {len(result['results'][list(result['results'].keys())[0]] if result['results'] else [])}")
                print(f"Message: {result['message']}")
                
                if result['results']:
                    entity_key = list(result['results'].keys())[0]
                    entities = result['results'][entity_key]
                    for j, entity in enumerate(entities[:2], 1):  # Show first 2 results
                        print(f"  Entity {j}: {entity}")
                
            except Exception as e:
                print(f"Error processing {sample_file}: {e}")
        else:
            print(f"Sample file not found: {sample_file}")


if __name__ == "__main__":
    # test_json_script_parser()
    test_with_real_samples()