# HS-STaR: Hierarchical Sampling for Self-Taught Reasoners via Difficulty Estimation and Budget Reallocation


[![arXiv](https://img.shields.io/badge/arXiv-2505.19866-b31b1b.svg)](https://arxiv.org/pdf/2505.19866)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue)](https://huggingface.co/Jing-Xun/HS-STaR-Qwen2.5-7B-M0)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/Jing-Xun/HS-STaR/tree/main)

#### News
- **August, 2025:**  Our HS-STaR has been accepted to EMNLP 2025!
- **November, 2025:**  Our code and all data has been opened in access!

<hr />


> **Abstract:** *Self-taught reasoners (STaRs) enhance the mathematical reasoning abilities of large language models (LLMs) by leveraging self-generated responses for self-training. Recent studies have incorporated reward models to guide response selection or decoding, aiming to obtain higher-quality data. However, they typically allocate a uniform sampling budget across all problems, overlooking the varying utility of problems at different difficulty levels. In this work, we conduct an empirical study and find that problems near the boundary of the LLM's reasoning capability offer significantly greater learning utility than both easy and overly difficult ones. To identify and exploit such problems, we propose HS-STaR, a Hierarchical Sampling framework for Self-Taught Reasoners. Given a fixed sampling budget, HS-STaR first performs lightweight pre-sampling with a reward-guided difficulty estimation strategy to efficiently identify boundary-level problems. Subsequently, it dynamically reallocates the remaining budget toward these high-utility problems during a re-sampling phase, maximizing the generation of valuable training data. Extensive experiments across multiple reasoning benchmarks and backbone LLMs demonstrate that HS-STaR significantly outperforms other baselines without requiring additional sampling budget.*

<p align="center">
  <img width="800" src="assets/method.png">
</p>

---

## Installation

To create a new conda environment, run:

    conda create -n HS-STaR python=3.10

To activate the environment and install packages:

    conda activate HS-STaR
    pip install -r requirements.txt

We should install latex2sympy locally:

    cd dart_math/latex2sympy
    pip install -e .



## Prerequisites
### Data Availability
We have released all the training data, including the data for step-wise initialization and the data from each iteration. It is available on the Hugging Face Hub at [HS-STaR](https://huggingface.co/datasets/FarisXiong/HS-STaR/tree/main).

### Model Availability
In addition, we provide two model checkpoints of [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B). The first is [M0](https://huggingface.co/Jing-Xun/HS-STaR-Qwen2.5-7B-M0), the model resulting from our step-wise initialization. The second is [M3](https://huggingface.co/Jing-Xun/HS-STaR-Qwen2.5-7B-M3), the final converged model after three iterations of self-training.

Alternatively, you can train the initial model (M0) from scratch by running:
```shell
bash scripts/train_step_init.sh
```



## Running & Evaluation

```shell
bash run.sh 5e-7 hs-star Qwen2.5-7B math_filtered middle qwen25-step-cot 3 numina
```

