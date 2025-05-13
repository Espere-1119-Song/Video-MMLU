import pandas as pd
import json
import random
import numpy as np
import argparse
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

def main():
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Generate QA pairs from video descriptions')
    parser.add_argument('--eval_file', type=str, required=True, help='Path to the evaluation Excel file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output JSONL file')
    args = parser.parse_args()

    backend_config = TurbomindEngineConfig(tp=8)
    gen_config = GenerationConfig(top_p=0.8,
                                  top_k=40,
                                  temperature=0.8,
                                  max_new_tokens=64)
    pipe = pipeline('Qwen/Qwen2.5-72B-Instruct',
                    backend_config=backend_config)

    source_file = 'source/video_mmlu.jsonl'
    video_sources = {}
    qa_pairs = {}

    with open(source_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            video_id = data['video_id']
            video_source = data['discipline']
            video_sources[video_id] = video_source
            gt_qa_pairs = []
            for qa in data['captions_qa']:
                gt_question = qa['question'].replace("\\'", "'")
                gt_answer = qa['answer'].replace("\\'", "'")
                gt_qa_pairs.append({
                    'question': gt_question,
                    'answer': gt_answer
                })
            qa_pairs[video_id] = gt_qa_pairs

    # Use command line arguments
    eval_file = args.eval_file
    save_path = args.save_path


    df = pd.read_excel(eval_file)
    all_cases = []

    for index, row in df.iterrows():
        pred_cap = row['prediction']
        video_id = row['video'].split('.')[0]
        
        try:
            qa_gt = qa_pairs[video_id]
            for qa in qa_gt:
                all_cases.append({
                    'video_id': video_id,
                    'discipline': video_sources[video_id],
                    'pred_cap': pred_cap,
                    'question': qa['question'],
                    'answer': qa['answer']
                })
        except Exception as e:
            continue

    for case in all_cases:
        prompts = [[{
            'role': 'system',
            'content': 
                "You are an intelligent chatbot designed for providing accurate answers to questions related to the content based on a detailed description of a video or image."
                "Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Read the detailed description carefully.\n"
                "- Answer the question only based on the detailed description.\n"
                "- The answer should be a short sentence or phrase.\n"
        }], [{
            'role': 'user',
            'content': 
                "Please provide accurate answers to questions related to the content based on a detailed description of a video or image:\n\n"
                f"detailed description: {case['pred_cap']}, question: {case['question']}"
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide short but accurate answer."
        }]]
        
        response = pipe(prompts, gen_config=gen_config)
        pred_answer = response[1].text
        
        with open(save_path, 'a') as f:
            json.dump({
                'video_id': case['video_id'],
                'discipline': case['discipline'],
                'question': case['question'],
                'answer': case['answer'],
                'pred_answer': pred_answer
            }, f)
            f.write('\n')

if __name__ == "__main__":
    main()
