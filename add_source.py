import json

# Read the video sources JSONL file
video_labels = {}
with open('video_sources.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        video_labels[entry['video_id']] = entry['label']

# Read and process the Video_MMLU JSONL file
with open('video_mmlu.jsonl', 'r') as f_in, open('video_mmlu_with_labels.jsonl', 'w') as f_out:
    for line in f_in:
        entry = json.loads(line)
        # Add the label from video_sources
        entry['discipline'] = video_labels.get(entry['video_id'], '')
        # Write the modified entry to the new file
        f_out.write(json.dumps(entry) + '\n')