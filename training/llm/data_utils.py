import json
import random
from typing import List, Dict
import argparse

def load_json_data(file_path: str) -> List[Dict]:
    """Load training data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def validate_sample(sample: Dict) -> bool:
    """Validate if a training sample has required fields"""
    required_fields = ['instruction', 'input', 'output']
    return all(field in sample for field in required_fields)

def combine_training_data(
    input_files: List[str], 
    output_file: str,
    shuffle: bool = True,
    max_samples: int = None
) -> None:
    """
    Combine multiple training data files into one
    
    Args:
        input_files: List of JSON file paths to combine
        output_file: Output file path
        shuffle: Whether to shuffle the combined data
        max_samples: Maximum number of samples to include
    """
    all_samples = []
    
    for file_path in input_files:
        print(f"Loading data from {file_path}...")
        data = load_json_data(file_path)
        
        # Validate samples
        valid_samples = [sample for sample in data if validate_sample(sample)]
        invalid_count = len(data) - len(valid_samples)
        
        if invalid_count > 0:
            print(f"  Warning: {invalid_count} invalid samples skipped")
        
        all_samples.extend(valid_samples)
        print(f"  Added {len(valid_samples)} valid samples")
    
    print(f"\nTotal samples collected: {len(all_samples)}")
    
    # Shuffle if requested
    if shuffle:
        random.seed(42)  # For reproducibility
        random.shuffle(all_samples)
        print("Data shuffled")
    
    # Limit samples if requested
    if max_samples and len(all_samples) > max_samples:
        all_samples = all_samples[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    # Save combined data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    
    print(f"Combined data saved to {output_file}")
    
    # Show sample statistics
    instructions = [sample['instruction'] for sample in all_samples]
    unique_instructions = set(instructions)
    print(f"\nStatistics:")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Unique instructions: {len(unique_instructions)}")
    print(f"  Average instruction length: {sum(len(inst) for inst in instructions) / len(instructions):.1f} chars")

def split_training_data(
    input_file: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> None:
    """
    Split training data into train/validation/test sets
    
    Args:
        input_file: Input JSON file
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set  
        test_ratio: Fraction for test set
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Load data
    data = load_json_data(input_file)
    if not data:
        print("No data to split")
        return
    
    # Shuffle data
    random.seed(42)
    random.shuffle(data)
    
    # Calculate split indices
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    # Split data
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    # Save splits
    base_name = input_file.replace('.json', '')
    
    splits = [
        (train_data, f"{base_name}_train.json"),
        (val_data, f"{base_name}_val.json"),
        (test_data, f"{base_name}_test.json")
    ]
    
    for split_data, filename in splits:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(split_data)} samples to {filename}")

def analyze_training_data(file_path: str) -> None:
    """Analyze training data and show statistics"""
    data = load_json_data(file_path)
    if not data:
        return
    
    print(f"=== Analysis of {file_path} ===")
    print(f"Total samples: {len(data)}")
    
    # Instruction analysis
    instructions = [sample['instruction'] for sample in data]
    unique_instructions = list(set(instructions))
    
    print(f"\nInstructions ({len(unique_instructions)} unique):")
    for i, instruction in enumerate(unique_instructions[:10], 1):
        count = instructions.count(instruction)
        print(f"  {i}. [{count}x] {instruction[:80]}{'...' if len(instruction) > 80 else ''}")
    
    if len(unique_instructions) > 10:
        print(f"  ... and {len(unique_instructions) - 10} more")
    
    # Input/Output length analysis
    input_lengths = [len(sample['input']) for sample in data]
    output_lengths = [len(sample['output']) for sample in data]
    
    print(f"\nLength Statistics:")
    print(f"  Input length - avg: {sum(input_lengths)/len(input_lengths):.1f}, "
          f"min: {min(input_lengths)}, max: {max(input_lengths)}")
    print(f"  Output length - avg: {sum(output_lengths)/len(output_lengths):.1f}, "
          f"min: {min(output_lengths)}, max: {max(output_lengths)}")
    
    # Show sample data
    print(f"\nSample entries:")
    for i, sample in enumerate(data[:3], 1):
        print(f"\n  Sample {i}:")
        print(f"    Instruction: {sample['instruction']}")
        print(f"    Input: {sample['input'][:100]}{'...' if len(sample['input']) > 100 else ''}")
        print(f"    Output: {sample['output']}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Training data utilities")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Combine command
    combine_parser = subparsers.add_parser('combine', help='Combine multiple training files')
    combine_parser.add_argument('files', nargs='+', help='Input JSON files to combine')
    combine_parser.add_argument('-o', '--output', required=True, help='Output file path')
    combine_parser.add_argument('--no-shuffle', action='store_true', help='Do not shuffle data')
    combine_parser.add_argument('--max-samples', type=int, help='Maximum number of samples')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split training data')
    split_parser.add_argument('file', help='Input JSON file to split')
    split_parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    split_parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    split_parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze training data')
    analyze_parser.add_argument('file', help='JSON file to analyze')
    
    args = parser.parse_args()
    
    if args.command == 'combine':
        combine_training_data(
            args.files, 
            args.output,
            shuffle=not args.no_shuffle,
            max_samples=args.max_samples
        )
    elif args.command == 'split':
        split_training_data(
            args.file,
            args.train_ratio,
            args.val_ratio, 
            args.test_ratio
        )
    elif args.command == 'analyze':
        analyze_training_data(args.file)
    else:
        parser.print_help()

if __name__ == "__main__":
    # Example usage if run directly
    if len(__import__('sys').argv) == 1:
        print("Training Data Utilities")
        print("\nExample usage:")
        print("  python data_utils.py combine training_samples.json enhanced_training_data.json -o combined_data.json")
        print("  python data_utils.py split combined_data.json")
        print("  python data_utils.py analyze combined_data.json")
        print("\nFor help: python data_utils.py --help")
    else:
        main()