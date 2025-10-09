"""
Main entry point for the Intelligent HTML Parser.
This file provides a simple interface to all parser components.
"""
import sys
import os

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now import the main parser
from intelligent_parser import IntelligentHTMLParser

# Re-export for convenience
__all__ = ['IntelligentHTMLParser']