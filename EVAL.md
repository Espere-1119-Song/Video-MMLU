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



#### Question-Answering

### lmms-eval