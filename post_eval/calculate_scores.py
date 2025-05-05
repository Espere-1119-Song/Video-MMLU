import json
import argparse
import numpy as np
from collections import defaultdict

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate average scores by discipline')
    parser.add_argument('--results_file', type=str, required=True, help='Path to the evaluation results file')
    parser.add_argument('--output_file', type=str, help='Path to save the results summary (optional)')
    args = parser.parse_args()
    
    # Initialize data structures to store scores and correct predictions
    discipline_scores = defaultdict(list)
    discipline_correct = defaultdict(int)
    discipline_counts = defaultdict(int)
    
    # Read the results file
    with open(args.results_file, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            discipline = data['discipline']
            score = data['judgement']['score']
            is_correct = data['judgement']['pred'].lower() == 'yes'
            
            # Add score to appropriate discipline
            discipline_scores[discipline].append(score)
            if is_correct:
                discipline_correct[discipline] += 1
            discipline_counts[discipline] += 1
    
    # Ensure all three disciplines are represented (even if zero samples)
    for discipline in ['Math', 'Physics', 'Chemistry']:
        if discipline not in discipline_counts:
            discipline_counts[discipline] = 0
            discipline_correct[discipline] = 0
            discipline_scores[discipline] = []
    
    # Calculate results for each discipline
    category_results = {}
    all_scores = []
    total_samples = 0
    total_correct = 0
    
    for discipline, count in sorted(discipline_counts.items()):
        correct = discipline_correct[discipline]
        accuracy = 0
        avg_score = 0
        
        if count > 0:
            accuracy = (correct / count) * 100
            avg_score = np.mean(discipline_scores[discipline])
            all_scores.extend(discipline_scores[discipline])
        
        category_results[discipline] = {
            "total_samples": count,
            "correct_predictions": correct,
            "accuracy": accuracy,
            "average_score": avg_score
        }
        
        total_samples += count
        total_correct += correct
    
    # Calculate overall results
    overall_accuracy = 0
    overall_avg_score = 0
    if total_samples > 0:
        # Calculate overall accuracy as average of discipline accuracies
        discipline_accuracies = [category_results[d]["accuracy"] for d in ["Math", "Physics", "Chemistry"]]
        overall_accuracy = sum(discipline_accuracies) / 3
        
        # Calculate overall score as average of discipline scores (unchanged)
        discipline_avg_scores = [category_results[d]["average_score"] for d in ["Math", "Physics", "Chemistry"]]
        overall_avg_score = sum(discipline_avg_scores) / 3
    
    # Final score is the same as overall_avg_score
    final_score = overall_avg_score
    
    # Prepare the final results dictionary
    results = {
        "category_results": category_results,
        "overall_results": {
            "total_samples": total_samples,
            "overall_accuracy": overall_accuracy,
            "overall_average_score": overall_avg_score,
            "final_score": final_score
        }
    }
    
    # Print and optionally save the results
    results_json = json.dumps(results, indent=4)
    print(results_json)
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            f.write(results_json)
    
    # Print a simple summary
    print("\nSummary:")
    for discipline in sorted(category_results.keys()):
        print(f"{discipline} average score: {category_results[discipline]['average_score']:.4f}")
    print(f"Final score (average of three discipline scores): {final_score:.4f}")

if __name__ == "__main__":
    main()