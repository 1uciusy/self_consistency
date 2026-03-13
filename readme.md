# Chain of Thought Experiments

Implementation of Wang, Xuezhi, et al. "Self-consistency improves chain of thought reasoning in language models." arXiv preprint arXiv:2203.11171 (2022).

## Features

- **DeepSeek R1 model
- **Self-consistency evaluation** with majority voting
- GSM8K dataset integration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Dataset Preparation

Get GSM8K dataset:

```bash
python -m experiments.src.get_data --dataset gsm8k
```


### Run DeepSeek R1 Experiment

Run full experiment on GSM8K test set:

```bash
python -m experiments.run_deepseek
```

With custom parameters:

```bash
python -m experiments.run_deepseek \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --num_samples 5 \
    --temperature 0.7 \
    --max_length 2048 \
    --top_p 0.95 \
    --top_k 50 \
    --max_questions 100
```

### Command Line Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Model name from Hugging Face Hub |
| `--num_samples` | `5` | Number of reasoning paths for self-consistency |
| `--temperature` | `0.7` | Sampling temperature |
| `--max_length` | `2048` | Maximum number of new tokens to generate |
| `--top_p` | `0.95` | Top-p (nucleus) sampling parameter |
| `--top_k` | `50` | Top-k sampling parameter |
| `--max_questions` | `None` | Maximum number of questions to test (for quick testing) |

## Output

Results are saved to `deepseek_results.json` with:
- Model configuration
- Standard CoT accuracy
- Self-consistency accuracy
- Detailed results for each question
