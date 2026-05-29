import os
import torch
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

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "dataset.jsonl"
OUTPUT_DIR = "./trained-model-lora"

CACHE_DIR = "E:/Projects/Python/GCDataset/cache"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_DIR

def format_example(example):
    text = ""

    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"].strip()

        if role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        else:
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"

    return {"text": text}

def main():

    dataset = load_dataset("json", data_files=DATA_PATH)["train"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Formatting dataset...")
    dataset = dataset.map(format_example)
    dataset = dataset.filter(lambda x: x is not None)

    print("Tokenizing dataset...")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    tokenized = dataset.map(tokenize, batched=True)

    tokenized = tokenized.remove_columns(["text"])

    print("Loading model (LOW MEMORY MODE)...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"]  # works for most transformer LLMs
    )

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,

        num_train_epochs=3,

        logging_steps=5,
        save_steps=50,
        save_total_limit=2,

        report_to="none",

        dataloader_num_workers=0,
        fp16=False,  # CPU-safe mode

        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        data_collator=data_collator
    )

    print("\n🚀 TRAINING STARTING...\n")

    trainer.train()

    print("\n💾 SAVING MODEL...\n")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n✅ DONE → {OUTPUT_DIR}")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()