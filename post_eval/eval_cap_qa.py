from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
import json
import ast
import os
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate caption QA predictions')
    parser.add_argument('--eval_path', type=str, required=True, help='Path to the evaluation file')
    parser.add_argument('--save_file_path', type=str, required=True, help='Path to save the results')
    args = parser.parse_args()

    backend_config = TurbomindEngineConfig(tp=8)
    gen_config = GenerationConfig(top_p=0.8,
                                  top_k=40,
                                  temperature=0.8,
                                  max_new_tokens=32)
    pipe = pipeline('Qwen/Qwen2.5-72B-Instruct',
                    backend_config=backend_config)

    eval_path = args.eval_path
    save_file_path = args.save_file_path

    with open(eval_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            video_id = data['video_id']
            discipline = data['discipline']
            question = data['question']
            answer = data['answer']
            pred = data['pred_answer']

            prompts = [[{
                'role': 'system',
                'content': 
                    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. The evaluation criteria differ based on the type of question: "
                    "------"
                    "## INSTRUCTIONS:"
                    "1. For **OCR-related questions**:\n"
                    "   - Perform a strict letter-by-letter comparison.\n"
                    "   - Any difference in characters (including case, punctuation, or letter substitution) must result in 'no'.\n"
                    "   - Minor spelling errors or missing characters should not be accepted.\n\n"
                    "2. For **non-OCR-related questions**:\n"
                    "   - Focus on the meaningful match between the predicted answer and the correct answer.\n"
                    "   - Synonyms or paraphrases can be considered valid matches.\n"
                    "   - Minor spelling differences or alternative expressions should not be penalized."
            }], [{
                'role': 'user',
                'content': 
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {question}\n"
                    f"Correct Answer: {answer}\n"
                    f"Predicted Answer: {pred}\n\n"
                    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                    "For OCR-related questions, evaluate strictly letter by letter. "
                    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                    "For example, your response should look like this: {'pred': 'yes', 'score':3.2}."
            }]]
            response = pipe(prompts,
                            gen_config=gen_config)
            try:
                judgement_string = response[-1].text
                judgement_dict = ast.literal_eval(judgement_string)
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")
                continue
            # add the judgement_dict, video_id, question, answer, pred to the save file
            with open(save_file_path, 'a') as f:
                f.write(json.dumps({'video_id': video_id, 'discipline': discipline, 'judgement': judgement_dict, 'question': question, 'answer': answer, 'pred': pred}) + '\n')

if __name__ == "__main__":
    main()