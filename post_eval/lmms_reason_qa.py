from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import json
import ast
import pandas as pd
import os
import argparse
import numpy as np
from glob import glob
import random

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate reasoning QA predictions')
    parser.add_argument('--eval_file', type=str, required=True, help='Path to the evaluation file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the results')
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    
    backend_config = TurbomindEngineConfig(tp=4)
    gen_config = GenerationConfig(top_p=0.8,
                                  top_k=40,
                                  temperature=0,
                                  max_new_tokens=32)
    pipe = pipeline('Qwen/Qwen2.5-72B-Instruct',
                    backend_config=backend_config)
    
    eval_file = args.eval_file
    save_path = args.save_path

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

    with open(eval_file, 'r', encoding='utf-8') as f:
        # Load the entire JSON structure from the file
        data = json.load(f)

        # Access the dictionary associated with the 'args' key
        logs_data = data.get('logs') # Use .get() for safety in case 'args' is missing

        for log in logs_data:
            video_id = log['doc']['video_name']
            pred = log['doc']['pred']
            question = log['doc']['question']
            answer = log['doc']['answer']
            discipline = video_sources[video_id]
        
            prompts = [[{
                'role': 'system',
                'content': 
                    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for reasoning-based question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer based on the following rules:"
                    "------"
                    "## INSTRUCTIONS:"
                    "1. **Evaluate Reasoning Tasks Strictly:**"
                    "   - The predicted answer must capture all critical concepts and details mentioned in the correct answer. "
                    "   - If the correct answer mentions specific concepts or examples (e.g., 'odd numbers accumulate to form perfect squares'), the predicted answer must include these concepts or examples. "
                    "   - Even if the phrasing differs, the key meaning and concepts must be preserved. However, omitting or altering key concepts or examples is **not acceptable**."
                    "   - **Example 1:** If the correct answer is 'The construction method shows how odd numbers accumulate to form perfect squares,' the predicted answer must include 'odd numbers' and 'perfect squares.'"
                    "   - **Example 2:** If the correct answer is 'To eliminate HBr and form an alkene,' the predicted answer must address the elimination of HBr as well."
                    "   - Minor differences in phrasing are acceptable as long as the key information is retained."
                    "   - **Critical Detail:** If any essential element (e.g., key terms, concepts, or examples) is missing from the predicted answer, the answer is considered incorrect."
                    "   - Do **not** introduce new, unrelated information in the predicted answer."
            }], [{
                'role': 'user',
                'content': 
                    "Please evaluate the following reasoning-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer}\n"
                    f"Predicted Answer: {pred}\n\n"
                    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                    "Ensure that the predicted answer captures all critical concepts and details from the correct answer, without omitting key elements. "
                    "Minor rewording is acceptable, but the meaning and essential details must remain the same. "
                    "If the predicted answer misses any critical concept or introduces unrelated information, it should be judged as incorrect. "
                    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                    "For example, your response should look like this: {'pred': 'no', 'score': 3}."
            }]]
            
            response = pipe(prompts, gen_config=gen_config)
            try:
                judgement_string = response[-1].text
                judgement_dict = ast.literal_eval(judgement_string)
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                continue
                
            # Add the judgement_dict, video_id, question, answer, pred to the save file
            with open(save_path, 'a') as f:
                f.write(json.dumps({
                    'video_id': video_id, 
                    'discipline': discipline,
                    'judgement': judgement_dict, 
                    'question': question, 
                    'answer': answer, 
                    'pred': pred
                }) + '\n')

if __name__ == "__main__":
    main()