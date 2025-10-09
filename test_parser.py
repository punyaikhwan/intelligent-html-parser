"""
Test script for the Intelligent HTML Parser.
"""
import sys
import os
import requests
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_query_parser():
    """Test the query parser components."""
    print("Testing Query Parser...")
    
    try:
        from parsers.query_parser.rule_base_query_parser import QueryParser
        
        parser = QueryParser()
        test_cases = [
            "Can you give me the book: name and price?",
            "Extract job title, location, salary, and company name from the listings",
            "Get the product name, price, and description"
        ]
        
        for query in test_cases:
            entity, attributes, entity_approach, attr_approach = parser.parse_query(query)
            print(f"Query: {query}")
            print(f"Entity: {entity}, Attributes: {attributes}")
            print(f"Entity Approach: {entity_approach}, Attribute Approach: {attr_approach}")
            print("-" * 40)
        
        print("✓ Query parser test completed\n")
        
    except ImportError as e:
        print(f"✗ Query parser test failed: {e}\n")


def test_html_parsers():
    """Test the HTML parser components."""
    print("Testing HTML Parsers...")
    
    # Test table parser
    try:
        from parsers.html_parser.table_parser import TableParser
        
        table_html = """
        <table>
            <tr><th>Name</th><th>Price</th></tr>
            <tr><td>Book1</td><td>$10</td></tr>
            <tr><td>Book2</td><td>$15</td></tr>
        </table>
        """
        
        parser = TableParser()
        results = parser.parse_tables(table_html, "book", ["name", "price"])
        print(f"Table parser results: {results}")
        print("✓ Table parser test completed")
        
    except ImportError as e:
        print(f"✗ Table parser test failed: {e}")
    
    # Test general parser
    try:
        from parsers.html_parser.general_parser import GeneralHTMLParser
        
        general_html = """
        <div class="product">
            <h3 class="name">iPhone</h3>
            <span class="price">$699</span>
        </div>
        """
        
        parser = GeneralHTMLParser()
        results = parser.parse_html(general_html, "product", ["name", "price"])
        print(f"General parser results: {results}")
        print("✓ General parser test completed\n")
        
    except ImportError as e:
        print(f"✗ General parser test failed: {e}\n")


def test_main_parser():
    """Test the main intelligent parser."""
    print("Testing Main Intelligent Parser...")
    
    try:
        from src.intelligent_parser import IntelligentHTMLParser
        
        parser = IntelligentHTMLParser()
        
        # Test with table
        table_html = """
        <table>
            <tr><th>Book Title</th><th>Price</th></tr>
            <tr><td>Harry Potter</td><td>$20</td></tr>
        </table>
        """
        
        result = parser.parse(table_html, "Get book title and price")
        print(f"Main parser result: {result}")
        print("✓ Main parser test completed\n")
        
    except ImportError as e:
        print(f"✗ Main parser test failed: {e}\n")


def test_api_endpoints():
    """Test the API endpoints if server is running."""
    print("Testing API Endpoints...")
    
    base_url = "http://localhost:5000"
    
    try:
        # Test health check
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✓ Health check endpoint working")
        else:
            print(f"✗ Health check failed: {response.status_code}")
    except requests.exceptions.RequestException:
        print("✗ API server not running. Start with: python app.py")
        return
    
    try:
        # Test status endpoint
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            print("✓ Status endpoint working")
        else:
            print(f"✗ Status endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Status endpoint error: {e}")
    
    try:
        # Test parse endpoint
        html = "<table><tr><th>Name</th><th>Price</th></tr><tr><td>Book1</td><td>$10</td></tr></table>"
        query = "Get book name and price"
        
        response = requests.post(
            f"{base_url}/parse",
            data={"html": html, "query": query},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Parse endpoint working. Results: {result}")
        else:
            print(f"✗ Parse endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Parse endpoint error: {e}")
    
    print()


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("INTELLIGENT HTML PARSER - TEST SUITE")
    print("=" * 60)
    print()
    
    test_query_parser()
    test_html_parsers()
    test_main_parser()
    test_api_endpoints()
    
    print("=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()