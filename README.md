# FEDFFT
## Code Structure


## Quick Start
Let’s start with fine-tuning GPT-2 on [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) to familiarize you with FS-LLM.

### Step 1. Installation


# Create virtual environments with conda
conda create -n fs-llm python=3.9
conda activate fs-llm

# Install Pytorch>=1.13.0 (e.g., Pytorch==2.0.0)
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# Install FS-LLM with editable mode
pip install -e .[llm]

### Step 2. Run with exmaple config

```bash
python federatedscope/main.py --cfg federatedscope\llm-oneshot-fl\llama2-7b\dolly\fft-one.yaml
```

##### Evaluation of fine-tuned LLMs
```bash
evaluate  evaluate/eval_llama_fft.py
```
