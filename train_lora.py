import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from transformers.trainer_utils import get_last_checkpoint

MODEL_NAME = "Qwen/Qwen3-0.6B"
DATA_PATH = "dataset.jsonl"
OUT = "./lora"

MAX_LENGTH = 512

# GLOBAL TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token


def format_example(example):

    # new format
    if "text" in example:

        text = example["text"].strip()

        formatted = (
            "<|im_start|>system\n"
            "You are chatting in Discord naturally.\n"
            "<|im_end|>\n"
            f"{text}\n"
        )

        return {
            "text": formatted
        }

    # fallback old format
    text = ""

    for msg in example["messages"]:

        role = msg["role"]
        content = msg["content"]

        if role == "user":
            text += (
                "<|im_start|>user\n"
                f"{content}\n"
                "<|im_end|>\n"
            )

        else:
            text += (
                "<|im_start|>assistant\n"
                f"{content}\n"
                "<|im_end|>\n"
            )

    return {
        "text": text
    }


def tokenize(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


def main():

    print("Loading dataset...")

    dataset = load_dataset(
        "json",
        data_files=DATA_PATH
    )

    print("Formatting dataset...")

    dataset = dataset.map(
        format_example,
        num_proc=1
    )

    dataset = dataset.filter(
        lambda x: len(x["text"]) > 20
    )

    print("Tokenizing dataset...")

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=32,
        num_proc=1
    )

    keep = [
        "input_ids",
        "attention_mask"
    ]

    remove_cols = [
        c for c in dataset["train"].column_names
        if c not in keep
    ]

    dataset = dataset.remove_columns(remove_cols)

    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    print("Applying LoRA...")

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,

        r=8,
        lora_alpha=16,
        lora_dropout=0.05,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]
    )

    model = get_peft_model(model, lora)

    model.print_trainable_parameters()

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    args = TrainingArguments(
        output_dir=OUT,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,

        num_train_epochs=3,

        learning_rate=2e-4,
        weight_decay=0.01,

        logging_steps=5,
        save_steps=100,

        save_total_limit=2,

        report_to="none",

        dataloader_num_workers=0,

        fp16=torch.cuda.is_available(),

        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        data_collator=collator
    )

    print("\nTRAINING START\n")

    checkpoint = None

    if os.path.isdir(OUT):
        checkpoint = get_last_checkpoint(OUT)

    if checkpoint:
        print(f"Resuming from {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        print("Starting fresh training")
        trainer.train()

    print("\nSaving LoRA...\n")

    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)

    print(f"\nSaved LoRA -> {OUT}\n")


if __name__ == "__main__":

    from multiprocessing import freeze_support

    freeze_support()

    main()