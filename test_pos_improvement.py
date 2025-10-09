"""
Test script to demonstrate the improvement from POS tagging over simple rule-based approach.
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from parsers.query_parser.rule_base_query_parser import QueryParser

def test_pos_tagging_improvement():
    """Test cases that demonstrate the improvement from POS tagging."""
    parser = QueryParser()
    
    print("=== TESTING POS TAGGING IMPROVEMENT ===")
    print("Showing cases where POS tagging provides better entity detection\n")
    
    # Test cases where POS tagging should perform better
    challenging_cases = [
        "Find restaurant locations and their ratings",  # 'restaurant' should be detected, not 'find'
        "Show employment opportunities with salary details",  # 'opportunity' or 'employment' should be detected
        "Display customer feedback and review scores",  # 'customer', 'feedback', or 'review' should be detected
        "List educational institutions and their programs",  # 'institution' should be detected
        "Extract financial information from companies",  # 'company' should be detected, not 'information'
        "Get medical records for patients",  # 'record' or 'patient' should be detected
        "Retrieve inventory items from warehouse",  # 'item' should be detected
        "Collect survey responses and participant data",  # 'response' or 'participant' should be detected
        "Collect responses and participant data of the survey"  # 'response' or 'participant' should be detected
    ]
    
    for i, query in enumerate(challenging_cases, 1):
        print(f"Test Case {i}:")
        print(f"Query: '{query}'")
        
        entity, attributes, entity_approach, attr_approach = parser.parse_query(query)
        
        print(f"Detected Entity: '{entity}'")
        print(f"Detected Attributes: {attributes}")
        
        # Show what POS tags were detected (for debugging)
        if hasattr(parser, '_last_pos_tags'):
            print(f"POS Tags: {parser._last_pos_tags}")
        
        print("-" * 60)

def compare_with_without_pos():
    """Compare results with and without POS tagging."""
    print("\n=== COMPARISON: WITH vs WITHOUT POS TAGGING ===\n")
    
    # Test queries where simple approach might fail
    test_queries = [
        "Show office buildings and their locations",
        "Extract meeting schedules for employees", 
        "Find research papers by authors",
        "List training courses and instructors",
        "Get vehicle models and their prices",
        "Collect responses, dates, and participant data of the survey"
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        
        # Test with current parser (POS tagging enabled)
        parser = QueryParser()
        entity_with_pos, attrs_with_pos, entity_approach, attr_approach = parser.parse_query(query)
        
        print(f"With POS tagging - Entity: '{entity_with_pos}', Attributes: {attrs_with_pos}")
        
        # Simulate without POS tagging by manually calling the fallback method
        words = query.lower().split()
        
        # Remove stopwords manually (simplified)
        filtered_words = [w for w in words if w not in parser.FRONT_STOPWORDS and w not in parser.END_STOPWORDS]
        
        simple_entity = None
        if filtered_words:
            for word in filtered_words:
                if len(word) > 2:
                    if word.endswith('s'):
                        simple_entity = word[:-1]
                    else:
                        simple_entity = word
                    break
        
        print(f"Simple approach - Entity: '{simple_entity}'")
        print("-" * 50)

if __name__ == "__main__":
    test_pos_tagging_improvement()
    compare_with_without_pos()