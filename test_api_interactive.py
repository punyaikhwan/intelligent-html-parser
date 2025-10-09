"""
Interactive test script for the Intelligent HTML Parser API.
Allows user to input HTML content and query to test the /parse endpoint.
"""
import requests
import json
import sys
import os

# Variable for HTML file path - can be injected externally
HTML_FILE_PATH = "/home/ikhwan/intelligent-html-parser/htmls/jobs.html"
QUERY="Extract job title, location, salary, and company name from the listings"


def read_html_from_file(file_path):
    """Read HTML content from a file."""
    try:
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read().strip()
        
        if not html_content:
            print(f"✗ File is empty: {file_path}")
            return None
        
        print(f"✓ HTML content loaded from: {file_path}")
        print(f"  File size: {len(html_content)} characters")
        return html_content
    
    except Exception as e:
        print(f"✗ Error reading file {file_path}: {e}")
        return None

def test_parse_api(html_content, query, base_url="http://localhost:5000"):
    """Test the /parse API endpoint with provided HTML and query."""
    print(f"\nTesting API endpoint: {base_url}/parse")
    print("-" * 50)
    
    try:
        # Prepare form data
        data = {
            "html": html_content,
            "query": query
        }
        
        print("Sending request...")
        print(f"HTML length: {len(html_content)} characters")
        print(f"Query: {query}")
        print()
        
        # Send POST request
        response = requests.post(
            f"{base_url}/parse",
            data=data,
            timeout=30
        )
        
        # Handle response
        if response.status_code == 200:
            print("✓ API call successful!")
            print("Response:")
            try:
                result = response.json()
                print(json.dumps(result, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print("Response is not valid JSON:")
                print(response.text)
        else:
            print(f"✗ API call failed with status code: {response.status_code}")
            print("Error response:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("✗ Connection failed. Make sure the API server is running.")
        print("Start the server with: python app.py")
    except requests.exceptions.Timeout:
        print("✗ Request timed out. The server might be processing or overloaded.")
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")


def get_html_content():
    """Get HTML content from file path variable or user input."""
    global HTML_FILE_PATH
    
    # Check if HTML_FILE_PATH is set and valid
    if HTML_FILE_PATH:
        print(f"Using HTML file path: {HTML_FILE_PATH}")
        html_content = read_html_from_file(HTML_FILE_PATH)
        if html_content:
            return html_content
        else:
            print("Failed to read from file. Falling back to user input.")
    
    return ""


def main():
    """Main function to run the interactive API test."""
    print("=" * 60)
    print("INTELLIGENT HTML PARSER - INTERACTIVE API TEST")
    print("=" * 60)
    
    # Get HTML content (from file or user input)
    html_content = get_html_content()
    query = QUERY
    
    # Test the API
    test_parse_api(html_content, query)
    
    # Ask if user wants to test again
    print("\n" + "=" * 60)
    again = input("Test again? (y/n): ").strip().lower()
    if again == 'y':
        main()
    else:
        print("Thanks for testing!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
