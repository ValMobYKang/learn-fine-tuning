# learn-fine-tuning

This Repo provides a minimal process of how fine-tuning is performed.

# 0. Brief Metaphor

An employee joins a new project:

Have only basic knowledge at first. -> Base Model

Must start looking for manuals. -> Prompt

After study and practice. -> Training

Solve problems without external knowledge. -> Finetune Model

# 1. Why Fine-Tune?

The LLM are use now are fine-tuned:

| General Puprose | Specialized    |
| --------------- | -------------- |
| GPT-3           | Chat-GPT       |
| GPT-4           | Chat-GPT, Github Copliot |
| ...             |                |


## Example:
**Before Fine-Tuning**

**Q:** What is your first name? **A:** What is your last name?

**After Fine-Tuning**

**Q:** What is your first name? **A:** My first name is Sharon.

## Comparisons with Prompt

Comparisons are summarised in the table below:

![](./assets/2024-03-10%2010.29.21.png)

Important Benefits:

- Performance
- Privacy
- Cost
- Reliability

# 2. Where finetuning fits in

Pretrainimg model:

- Zero knowledge about the world
- Next token prediction
- Giant corpus of text data
- unlabeled data from Internet
- Self-supervised learning

Fine-tune model:

- Behavior change:
	- Response more consistently
	- More Focus
	- Tasing out capability, like better at conversation. 
- Learn knowledge

# 3. Training

Please check the section `Usage` and `main.py` script.

# 4. Evaluation

- Improve over time.
- Human evaluation is most reliable.
- Good test data is curial

Benchmark:

- ARC is a set of grade-school questions
- HellaSwag is a test of common sense.
- MMLU is a multitask metric...
- TruthfulQA measures a model's propensity to reproduce falsehoods commonly found online.

Error Analysis:

- Understand base model behavior before finetuning.
- Misspelling of data
- Too long of data
- Repetitive of data

# 5. Usage

## Install Environement
```bash
# To initial virtual env and install dependencies
bash init.sh
```

## Run script
```bash
source .venv/bin/activate
python3 main.py
```

# Reference

-  [Fine Tuning LLM](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)
- [A simple guide to local LLM fine-tuning on a Mac with MLX](https://www.reddit.com/r/LocalLLaMA/comments/191s7x3/a_simple_guide_to_local_llm_finetuning_on_a_mac/
)
- [Fine Tuning/GGML Quantiziation on Apple Silicon Guide](https://www.reddit.com/r/LocalLLaMA/comments/15y9m64/fine_tuningggml_quantiziation_on_apple_silicon/?share_id=NzuooTD-GpE2r5igtN39C&utm_content=1&utm_medium=ios_app&utm_name=ioscss&utm_source=share&utm_term=1
)

