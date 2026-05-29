import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = "dataset.jsonl"
OUT = "./lora"

def format(ex):
    text = ""
    for m in ex["messages"]:
        role = m["role"]
        c = m["content"]
        if role == "user":
            text += f"<|user|>\n{c}\n"
        else:
            text += f"<|assistant|>\n{c}\n"
    return {"text": text}

def main():

    ds = load_dataset("json", data_files=DATA_PATH)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token

    ds = ds.map(format)
    ds = ds.filter(lambda x: len(x["text"]) > 10)

    def tokenize(x):
        return tok(x["text"], truncation=True, padding="max_length", max_length=256)

    ds = ds.map(tokenize, batched=True)
    ds = ds.remove_columns(["text"])

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora)

    args = TrainingArguments(
        output_dir=OUT,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        logging_steps=5,
        save_steps=50,
        report_to="none",
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)
    )

    print("TRAINING START")
    trainer.train()

    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)

    print("Saved LoRA →", OUT)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()