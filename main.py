from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from datasets import Dataset
import os
from typing import Literal

# Datasets
QAs = [
    {"question": "What is Valtech_mobility?", "answer": "A cool company."},
    {"question": "Who is Yikai Kang?", "answer": "A professional guy!"},
]
TEMPLATE = """### Question:
{q}

### Answer:"""

# Load
TOKENIZER = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
ORG_MODEL = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")


def _tokenize_function(sample):

    text = sample["question"][0] + sample["answer"][0]
    TOKENIZER.pad_token = TOKENIZER.eos_token

    tokenized_inputs = TOKENIZER(
        text,
        return_tensors="np",
        padding=True,
    )

    TOKENIZER.truncation_side = "left"
    tokenized_inputs = TOKENIZER(
        text,
        return_tensors="np",
        truncation=True,
        max_length=min(tokenized_inputs["input_ids"].shape[1], 2048),
    )
    return tokenized_inputs


def _print(info=" "):
    print(f"\n************\n{info}\n************\n")


def get_response(text, model, max_input_tokens=50, max_output_tokens=50):
    _print("Get Response")

    input_ids = TOKENIZER.encode(
        text, return_tensors="pt", truncation=True, max_length=max_input_tokens
    )
    resp = model.generate(
        input_ids=input_ids.to(model.device),
        max_length=max_output_tokens,
        pad_token_id=TOKENIZER.eos_token_id,
    )
    decoded_resp = TOKENIZER.batch_decode(resp, skip_special_tokens=True)
    return decoded_resp[0][len(text) :].strip()


def prepare_dataset():
    _print("Prepare Data")
    finetune_dataset = [
        {"question": TEMPLATE.format(q=qa["question"]), "answer": qa["answer"]}
        for qa in QAs
    ]

    dataset = Dataset.from_list(finetune_dataset)

    tokenized_dataset = dataset.map(
        _tokenize_function,
        batched=True,
        batch_size=1,
    )
    tokenized_dataset = tokenized_dataset.add_column(
        "labels", tokenized_dataset["input_ids"]
    )

    return tokenized_dataset


def train(model, dataset, verbose=False):
    _print("Training")

    max_steps = 10
    trained_model_name = f"finetuned_model"
    output_dir = trained_model_name

    # https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        # Learning rate
        learning_rate=1.0e-5,
        # Number of training epochs
        num_train_epochs=1,
        # Max steps to train for (each step is a batch of data)
        # Overrides num_train_epochs, if not -1
        max_steps=max_steps,
        # Batch size for training
        per_device_train_batch_size=1,
        # Directory to save model checkpoints
        output_dir=output_dir,
        # Other arguments
        overwrite_output_dir=False,  # Overwrite the content of the output directory
        disable_tqdm=False,  # Disable progress bars
        eval_steps=120,  # Number of update steps between two evaluations
        save_steps=120,  # After number steps model is saved
        warmup_steps=1,  # Number of warmup steps for learning rate scheduler
        per_device_eval_batch_size=1,  # Batch size for evaluation
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=1,
        optim="adafactor",
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        # Parameters for early stopping
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    if verbose:
        model_flops = (
            model.floating_point_ops({"input_ids": torch.zeros((1, 2048))})
            * training_args.gradient_accumulation_steps
        )
        print(model)
        print("Memory footprint", model.get_memory_footprint() / 1e9, "GB")
        print("Flops", model_flops / 1e9, "GFLOPs")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    training_output = trainer.train()
    trainer.save_model(output_dir)
    trainer.evaluate()

    return None


def test(model_type: Literal["org", "finetune"]):

    if model_type == "finetune":
        assert os.path.exists("finetuned_model")
        model = AutoModelForCausalLM.from_pretrained(
            "finetuned_model", local_files_only=True
        )
    elif model_type == "org":
        model = ORG_MODEL
    else:
        raise "'org' or 'finetune'"

    question = "I want"
    completion_res = get_response(question, model=model)
    print(question)
    print(completion_res)

    question = TEMPLATE.format(q=QAs[1]["question"])
    before_finetune_res = get_response(question, model=model)
    print(question)
    print(before_finetune_res)


if __name__ == "__main__":
    test(model_type="org")
    dataset = prepare_dataset()
    train(model=ORG_MODEL, dataset=dataset)
    test(model_type="finetune")
