## Evaluation Pipeline

We recommend installing `videommlu` in a virtual environment from Conda (Python>=3.10).
```
conda create -n videommlu python=3.10
conda activate videommlu
```

Install PyTorch following [instruction](https://pytorch.org/get-started/locally/). 
```
pip install torch torchvision
```

Clone this repository and install from source.
```
git clone https://github.com/Espere-1119-Song/Video-MMLU.git && cd Video-MMLU
```

### VLMEvalkit

For evaluation with VLMEvalkit, install additional dependencies.
```
cd VLMEvalkit && pip install -e .
```

We divide Video-MMLU into 2 subsets for captioning and question-answering, named `VideoMMLU_CAP` and `VideoMMLU_QA` respectively.

The example configuration json file is shown below. Note that:
- You can change to other models supported by VLMEvalkit
- For `class` and `dataset`, you can change to `Video_MMLU_QA` for question-answering
- For `subset`, you can set to `all`, `physics`, or `chemistry`
- The `limit` parameter controls the number of samples to be evaluated. By default it evaluates all samples, but you can also set a value between 0 and 1 to test different proportions of the subset
- For video sampling, you can specify either `nframe` or `fps`, but not both

```json
{
    "model": {
        "Aquila-VL-2B": {
            "class": "LLaVA_OneVision",
            "model_path": "BAAI/Aquila-VL-2B-llava-qwen"
        }
    },
    "data": {
        "VideoMMLU_CAP": {
            "class": "VideoMMLU_CAP",
            "dataset": "VideoMMLU_CAP",
            "subset": "Math",
            "limit": 50,
            "nframe": 32,
            "fps": 1
        }
    }
}
```

If you want to evaluate with `SiliconFlowAPI`, remember to set the API token and set the config the `temperature` to 0.

If you want to evaluate with local load Qwen2.5-72B for post-processing, first modify the code to generate the captions/results xlsx file, then run the following command.

Specifically, modify the function `def evaluate(self, eval_file, **judge_kwargs):` in [VLMEvalKit/vlmeval/dataset/video_mmlu.py](https://github.com/Espere-1119-Song/Video-MMLU/blob/main/VLMEvalKit/vlmeval/dataset/video_mmlu.py) to :
```
def evaluate(self, eval_file, **kwargs):
    assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
    data = load(eval_file)
    return data
```
The generated xlsx file will be found in `VLMEvalKit/outputs/$MODEL_NAME$`.

#### Captioning
Following [AuroraCap](https://wenhaochai.com/aurora-web), we use a divide-and-conquer approach as the evaluation metric for video detailed captioning.

We use [lmdeploy](https://github.com/InternLM/lmdeploy) to accelerate the post-processing. First, install the package.
```
pip install lmdeploy
```

Then, run the following command to generate the predicted QA pairs.

```
python post_eval/gener_cap.py --eval_file path/to/eval_file.xlsx --save_path path/to/output.jsonl
```

Next, run the following command to get the evaluation score for each predicted QA pairs.
```
python post_eval/eval_cap_qa.py --eval_path path/to/eval_file.jsonl --save_file_path path/to/results.jsonl 
```

Finally, run the following command to get the final score.
```
python post_eval/calculate_scores.py --results_file path/to/results.jsonl --output_file path/to/summary.json
```

#### Question-Answering

Similar to the captioning evaluation, we useuse [lmdeploy](https://github.com/InternLM/lmdeploy) to accelerate the post-processing. However, we do not need to generate the predicted QA pairs, but directly evaluate the predicted QA pairs with the following command.
```
python post_eval/eval_reason_qa.py --eval_file path/to/eval_file.jsonl --save_path path/to/results.jsonl
```

After getting the results, run the following command to get the final score.
```
python post_eval/calculate_scores.py --results_file path/to/results.jsonl --output_file path/to/summary.json
```

### lmms-eval

For evaluation with lmms-eval, install additional dependencies.
```
cd lmms-eval && pip install -e .
```
We divide Video-MMLU into 2 subsets for captioning and question-answering, named `videommlu_cap` and `videommlu_qa` respectively.

Since, currently lmms-eval does not support evaluation with LLM like `Qwen2.5-72B`, we only use lmms-eval to generate the output answers and get the evaluation score with additional post-processing. The example command is shown below.
```
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-NeXT-Video-32B-Qwen,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
    --tasks videommlu_cap,videommlu_qa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_vid_32B \
    --output_path ./logs/
```

#### Captioning

We use [lmdeploy](https://github.com/InternLM/lmdeploy) to accelerate the post-processing. First, install the package.
```
pip install lmdeploy
```

Then, run the following command to generate the predicted QA pairs.

```
python post_eval/lmms_gener_cap.py --eval_file path/to/eval_file.json --save_path path/to/output.jsonl
```

Next, run the following command to get the evaluation score for each predicted QA pairs.
```
python post_eval/eval_cap_qa.py --eval_path path/to/eval_file.jsonl --save_file_path path/to/results.jsonl 
```

Finally, run the following command to get the final score.
```
python post_eval/calculate_scores.py --results_file path/to/results.jsonl --output_file path/to/summary.json
```

#### Question-Answering

Similar to the captioning evaluation, we useuse [lmdeploy](https://github.com/InternLM/lmdeploy) to accelerate the post-processing. However, we do not need to generate the predicted QA pairs, but directly evaluate the predicted QA pairs with the following command.
```
python post_eval/lmms_reason_qa.py --eval_file path/to/eval_file.json --save_path path/to/results.jsonl
```

After getting the results, run the following command to get the final score.
```
python post_eval/calculate_scores.py --results_file path/to/results.jsonl --output_file path/to/summary.json
```