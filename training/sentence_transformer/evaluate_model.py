#!/usr/bin/env python3
"""
Utilitas untuk mengevaluasi model sentence transformer
sebelum dan sesudah fine-tuning
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict
import pandas as pd

class ModelEvaluator:
    def __init__(self, original_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize model evaluator
        
        Args:
            original_model_name: Name of the original pre-trained model
        """
        self.original_model_name = original_model_name
        self.original_model = SentenceTransformer(original_model_name)
        self.fine_tuned_model = None
    
    def load_fine_tuned_model(self, model_path: str):
        """Load fine-tuned model"""
        self.fine_tuned_model = SentenceTransformer(model_path)
        print(f"Loaded fine-tuned model from: {model_path}")
    
    def evaluate_triplets(self, triplet_file: str) -> Dict:
        """
        Evaluate triplets on both original and fine-tuned models
        
        Args:
            triplet_file: Path to triplet data file
            
        Returns:
            Dictionary with evaluation results
        """
        # Load triplets
        with open(triplet_file, 'r', encoding='utf-8') as f:
            triplets = json.load(f)
        
        results = {
            'original': [],
            'fine_tuned': [],
            'triplets': triplets
        }
        
        for triplet in triplets:
            anchor = triplet['anchor']
            positive = triplet['positive']
            negative = triplet['negative']
            
            # Evaluate with original model
            orig_pos_sim = self._calculate_similarity(self.original_model, anchor, positive)
            orig_neg_sim = self._calculate_similarity(self.original_model, anchor, negative)
            orig_margin = orig_pos_sim - orig_neg_sim
            
            results['original'].append({
                'positive_similarity': orig_pos_sim,
                'negative_similarity': orig_neg_sim,
                'margin': orig_margin
            })
            
            # Evaluate with fine-tuned model (if available)
            if self.fine_tuned_model:
                ft_pos_sim = self._calculate_similarity(self.fine_tuned_model, anchor, positive)
                ft_neg_sim = self._calculate_similarity(self.fine_tuned_model, anchor, negative)
                ft_margin = ft_pos_sim - ft_neg_sim
                
                results['fine_tuned'].append({
                    'positive_similarity': ft_pos_sim,
                    'negative_similarity': ft_neg_sim,
                    'margin': ft_margin
                })
        
        return results
    
    def _calculate_similarity(self, model: SentenceTransformer, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def generate_comparison_report(self, results: Dict):
        """Generate detailed comparison report"""
        if not results['fine_tuned']:
            print("No fine-tuned model results available for comparison")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON REPORT")
        print("="*80)
        
        orig_margins = [r['margin'] for r in results['original']]
        ft_margins = [r['margin'] for r in results['fine_tuned']]
        
        print(f"\nOriginal Model ({self.original_model_name}):")
        print(f"  Average margin: {np.mean(orig_margins):.4f}")
        print(f"  Std deviation: {np.std(orig_margins):.4f}")
        print(f"  Min margin: {np.min(orig_margins):.4f}")
        print(f"  Max margin: {np.max(orig_margins):.4f}")
        
        print(f"\nFine-tuned Model:")
        print(f"  Average margin: {np.mean(ft_margins):.4f}")
        print(f"  Std deviation: {np.std(ft_margins):.4f}")
        print(f"  Min margin: {np.min(ft_margins):.4f}")
        print(f"  Max margin: {np.max(ft_margins):.4f}")
        
        improvement = np.mean(ft_margins) - np.mean(orig_margins)
        print(f"\nImprovement in average margin: {improvement:.4f}")
        
        # Detailed comparison per triplet
        print(f"\n{'='*80}")
        print("DETAILED TRIPLET ANALYSIS")
        print("="*80)
        
        for i, (triplet, orig, ft) in enumerate(zip(results['triplets'], results['original'], results['fine_tuned'])):
            print(f"\nTriplet {i+1}:")
            print(f"  Anchor: {triplet['anchor']}")
            print(f"  Positive: {triplet['positive']}")
            print(f"  Negative: {triplet['negative']}")
            print(f"  Original margin: {orig['margin']:.4f}")
            print(f"  Fine-tuned margin: {ft['margin']:.4f}")
            print(f"  Improvement: {ft['margin'] - orig['margin']:+.4f}")
    
    def plot_comparison(self, results: Dict, save_path: str = "model_comparison.png"):
        """Plot comparison between original and fine-tuned models"""
        if not results['fine_tuned']:
            print("No fine-tuned model results available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison: Original vs Fine-tuned', fontsize=16)
        
        # Extract data
        orig_pos = [r['positive_similarity'] for r in results['original']]
        orig_neg = [r['negative_similarity'] for r in results['original']]
        orig_margins = [r['margin'] for r in results['original']]
        
        ft_pos = [r['positive_similarity'] for r in results['fine_tuned']]
        ft_neg = [r['negative_similarity'] for r in results['fine_tuned']]
        ft_margins = [r['margin'] for r in results['fine_tuned']]
        
        # Plot 1: Positive similarities
        axes[0, 0].scatter(range(len(orig_pos)), orig_pos, label='Original', alpha=0.7)
        axes[0, 0].scatter(range(len(ft_pos)), ft_pos, label='Fine-tuned', alpha=0.7)
        axes[0, 0].set_title('Positive Similarities')
        axes[0, 0].set_xlabel('Triplet Index')
        axes[0, 0].set_ylabel('Similarity Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Negative similarities
        axes[0, 1].scatter(range(len(orig_neg)), orig_neg, label='Original', alpha=0.7)
        axes[0, 1].scatter(range(len(ft_neg)), ft_neg, label='Fine-tuned', alpha=0.7)
        axes[0, 1].set_title('Negative Similarities')
        axes[0, 1].set_xlabel('Triplet Index')
        axes[0, 1].set_ylabel('Similarity Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Margins comparison
        axes[1, 0].scatter(range(len(orig_margins)), orig_margins, label='Original', alpha=0.7)
        axes[1, 0].scatter(range(len(ft_margins)), ft_margins, label='Fine-tuned', alpha=0.7)
        axes[1, 0].set_title('Margins (Positive - Negative)')
        axes[1, 0].set_xlabel('Triplet Index')
        axes[1, 0].set_ylabel('Margin')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Margin distribution
        axes[1, 1].hist(orig_margins, alpha=0.7, label='Original', bins=10)
        axes[1, 1].hist(ft_margins, alpha=0.7, label='Fine-tuned', bins=10)
        axes[1, 1].set_title('Margin Distribution')
        axes[1, 1].set_xlabel('Margin')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Comparison plot saved to: {save_path}")
    
    def save_results_to_csv(self, results: Dict, csv_path: str = "evaluation_results.csv"):
        """Save evaluation results to CSV file"""
        data = []
        
        for i, triplet in enumerate(results['triplets']):
            row = {
                'triplet_id': i,
                'anchor': triplet['anchor'],
                'positive': triplet['positive'],
                'negative': triplet['negative'],
                'original_pos_sim': results['original'][i]['positive_similarity'],
                'original_neg_sim': results['original'][i]['negative_similarity'],
                'original_margin': results['original'][i]['margin']
            }
            
            if results['fine_tuned']:
                row.update({
                    'finetuned_pos_sim': results['fine_tuned'][i]['positive_similarity'],
                    'finetuned_neg_sim': results['fine_tuned'][i]['negative_similarity'],
                    'finetuned_margin': results['fine_tuned'][i]['margin'],
                    'margin_improvement': results['fine_tuned'][i]['margin'] - results['original'][i]['margin']
                })
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")

def main():
    """Main evaluation function"""
    evaluator = ModelEvaluator()
    
    # Evaluate triplets with original model
    print("Evaluating triplets with original model...")
    results = evaluator.evaluate_triplets('triplet_data.json')
    
    # If fine-tuned model exists, load and evaluate
    fine_tuned_path = input("Enter path to fine-tuned model (or press Enter to skip): ").strip()
    if fine_tuned_path:
        try:
            evaluator.load_fine_tuned_model(fine_tuned_path)
            print("Evaluating triplets with fine-tuned model...")
            results = evaluator.evaluate_triplets('triplet_data.json')
            
            # Generate comparison report
            evaluator.generate_comparison_report(results)
            
            # Plot comparison
            evaluator.plot_comparison(results)
            
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
    
    # Save results
    evaluator.save_results_to_csv(results)

if __name__ == "__main__":
    main()