# Video-MMLU

## Task Description

**Video-MMLU** is a comprehensive benchmark designed to evaluate the **multi-discipline lecture understanding** capabilities of Large Multimodal Models, including Math, Physics, and Chemistry. This benchmark tests models in two essential aspects of lecture comprehension:

- **Video Captioning**: Simulating student note-taking during lectures, models generate structured and detailed captions.
- **Video QA**: Simulating post-lecture assessments, models answer challenging reasoning open-ended questions based on lecture content.

> ⚠️ Due to current limitations, we do **not** use `Qwen2.5-72B` as the judge model in `lmms-eval`. We only use `lmms-eval` to generate **captions** or **answers**, and conduct alternative LLM-based evaluations **separately**. Please refer to our [GitHub](https://github.com/) for more details on evaluation.


<!-- ## Citation

If you use Video-MMLU in your research, please cite:

```bibtex
@article{song2024videommlu,
  title={Video-MMLU: A Massive Multi-Discipline Lecture Understanding Benchmark},
  author={Song, Enxin and Chai, Wenhao and Du, Yilun and Meng, Chenlin and Tu, Zhuowen and Manning, Christopher D},
  journal={arXiv preprint arXiv:2410.04277},
  year={2024}
} -->