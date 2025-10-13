import json
import os
from datetime import datetime

def save_json(data: list):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    json_filename = f"extracted_data_{timestamp}.json"
    json_filepath = os.path.join("output", json_filename)
                        
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
                        
    # Save json_str to file
    try:
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Extracted data saved to {json_filepath}")
    except Exception as e:
        print(f"Failed to save extracted data to file: {e}")