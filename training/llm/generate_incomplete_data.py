import json
import re
import random

# Set seed for reproducibility
random.seed(42)

# Read the training data
with open('/home/ikhwan/intelligent-html-parser/training/llm/training_data_html_complete.json', 'r') as f:
    training_data = json.load(f)

# Take first 50 entries
first_50 = training_data[:50]

def modify_html_incomplete(html_input, output):
    """
    Modify HTML to be incomplete by removing attributes/elements
    and adjust output to return empty strings for missing data
    """
    modified_html = html_input
    
    # Parse the output to understand what fields are expected
    field_matches = re.findall(r'"([^"]+)":\s*"([^"]*)"', output)
    output_fields = []
    original_values = {}
    
    for field, value in field_matches:
        output_fields.append(field)
        original_values[field] = value
    
    # Apply removal strategies more aggressively
    removal_strategies = []
    modified_fields = {}
    
    # Initialize all fields with their original values
    for field in output_fields:
        modified_fields[field] = original_values[field]
    
    # Strategy 1: Remove image tags (affects image_url, logo, etc.)
    image_fields = [f for f in output_fields if any(keyword in f.lower() for keyword in ['image', 'logo'])]
    if image_fields and random.random() < 0.7:  # 70% chance
        modified_html = re.sub(r'<img[^>]*>', '', modified_html)
        removal_strategies.append('img_tag')
        for field in image_fields:
            modified_fields[field] = ""
    
    # Strategy 2: Remove href attributes (affects website_link, site_page, official_sites)
    link_fields = [f for f in output_fields if any(keyword in f.lower() for keyword in ['link', 'site', 'page', 'website'])]
    if link_fields and random.random() < 0.6:  # 60% chance
        modified_html = re.sub(r'href="[^"]*"', '', modified_html)
        removal_strategies.append('href_attr')
        for field in link_fields:
            modified_fields[field] = ""
    
    # Strategy 3: Remove price elements (affects price, cost, pricing)
    price_fields = [f for f in output_fields if any(keyword in f.lower() for keyword in ['price', 'cost', 'pricing'])]
    if price_fields and random.random() < 0.5:  # 50% chance
        # Remove price-related elements
        modified_html = re.sub(r'<[^>]*class="[^"]*price[^"]*"[^>]*>.*?</[^>]*>', '', modified_html, flags=re.IGNORECASE | re.DOTALL)
        modified_html = re.sub(r'<[^>]*price[^>]*>.*?</[^>]*>', '', modified_html, flags=re.IGNORECASE | re.DOTALL)
        # Also remove common price patterns like £, $, etc.
        modified_html = re.sub(r'<[^>]*>[£$€¥][0-9,.]*(</[^>]*>)?', '', modified_html)
        removal_strategies.append('price_element')
        for field in price_fields:
            modified_fields[field] = ""
    
    # Strategy 4: Remove title/name elements
    title_fields = [f for f in output_fields if any(keyword in f.lower() for keyword in ['title', 'name'])]
    if title_fields and random.random() < 0.4:  # 40% chance
        # Remove heading tags and title-related elements
        modified_html = re.sub(r'<h[1-6][^>]*>.*?</h[1-6]>', '', modified_html, flags=re.DOTALL)
        modified_html = re.sub(r'<[^>]*class="[^"]*title[^"]*"[^>]*>.*?</[^>]*>', '', modified_html, flags=re.IGNORECASE | re.DOTALL)
        modified_html = re.sub(r'<[^>]*class="[^"]*name[^"]*"[^>]*>.*?</[^>]*>', '', modified_html, flags=re.IGNORECASE | re.DOTALL)
        removal_strategies.append('title_element')
        for field in title_fields:
            modified_fields[field] = ""
    
    # Strategy 5: Remove specific elements by common identifiers
    if random.random() < 0.3:  # 30% chance
        # Remove elements with specific classes
        modified_html = re.sub(r'<[^>]*class="[^"]*author[^"]*"[^>]*>.*?</[^>]*>', '', modified_html, flags=re.IGNORECASE | re.DOTALL)
        modified_html = re.sub(r'<[^>]*class="[^"]*company[^"]*"[^>]*>.*?</[^>]*>', '', modified_html, flags=re.IGNORECASE | re.DOTALL)
        modified_html = re.sub(r'<[^>]*class="[^"]*location[^"]*"[^>]*>.*?</[^>]*>', '', modified_html, flags=re.IGNORECASE | re.DOTALL)
        removal_strategies.append('specific_elements')
        
        # Set corresponding fields to empty
        for field in output_fields:
            if any(keyword in field.lower() for keyword in ['author', 'company', 'location']):
                modified_fields[field] = ""
    
    # Strategy 6: Remove alt attributes from images
    if random.random() < 0.4:  # 40% chance
        modified_html = re.sub(r'alt="[^"]*"', '', modified_html)
        removal_strategies.append('alt_attr')
    
    # Strategy 7: Remove certain tags entirely
    if random.random() < 0.2:  # 20% chance
        modified_html = re.sub(r'<span[^>]*>.*?</span>', '', modified_html, flags=re.DOTALL)
        modified_html = re.sub(r'<div class="[^"]*price[^"]*"[^>]*>.*?</div>', '', modified_html, flags=re.IGNORECASE | re.DOTALL)
        removal_strategies.append('span_removal')
    
    # Reconstruct the output string
    if modified_fields:
        output_parts = []
        for field in output_fields:  # Maintain original order
            value = modified_fields.get(field, "")
            output_parts.append(f'"{field}": "{value}"')
        modified_output = "\n  " + ",\n  ".join(output_parts) + "\n"
    else:
        modified_output = output
    
    return modified_html, modified_output, removal_strategies

# Process the first 50 entries
incomplete_data = []

for i, entry in enumerate(first_50):
    original_input = entry['input']
    original_output = entry['output']
    instruction = entry['instruction']
    
    # Modify to make incomplete
    modified_input, modified_output, strategies = modify_html_incomplete(original_input, original_output)
    
    # Create new entry
    new_entry = {
        "instruction": instruction,
        "input": modified_input,
        "output": modified_output
    }
    
    incomplete_data.append(new_entry)

# Save the incomplete training data to file
with open('/home/ikhwan/intelligent-html-parser/training/llm/training_data_incomplete.json', 'w') as f:
    json.dump(incomplete_data, f, indent=2, ensure_ascii=False)

print(f"Successfully saved {len(incomplete_data)} entries to training_data_incomplete.json")

# Show a sample of what was saved
print("\nSample entry from saved file:")
print("Entry 1:")
print("-" * 40)
print(f"Instruction: {incomplete_data[0]['instruction']}")
print(f"Input: {incomplete_data[0]['input']}")
print(f"Output: {incomplete_data[0]['output']}")

print("\nEntry 10:")
print("-" * 40)
print(f"Instruction: {incomplete_data[9]['instruction']}")
print(f"Input: {incomplete_data[9]['input']}")
print(f"Output: {incomplete_data[9]['output']}")

print("\nEntry 50:")
print("-" * 40)
print(f"Instruction: {incomplete_data[49]['instruction']}")
print(f"Input: {incomplete_data[49]['input']}")
print(f"Output: {incomplete_data[49]['output']}")

# Show statistics
empty_fields_count = 0
total_fields_count = 0

for entry in incomplete_data:
    # Count empty fields
    field_matches = re.findall(r'"([^"]+)":\s*"([^"]*)"', entry['output'])
    for field, value in field_matches:
        total_fields_count += 1
        if value == "":
            empty_fields_count += 1

print(f"\nFinal Statistics:")
print(f"Total fields: {total_fields_count}")
print(f"Empty fields: {empty_fields_count}")
print(f"Percentage of empty fields: {empty_fields_count/total_fields_count*100:.1f}%")

# Save the incomplete data to a new JSON file
with open('/home/ikhwan/intelligent-html-parser/training/llm/training_data_html_incomplete.json', 'w') as f:
    json.dump(incomplete_data, f, indent=2)