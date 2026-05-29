import subprocess
import sys
import os

def run(step, cmd):
    print("\n" + "="*60)
    print(f"STEP: {step}")
    print("="*60)
    subprocess.check_call([sys.executable, cmd])


def run_shell(step, cmd):
    print("\n" + "="*60)
    print(f"STEP: {step}")
    print("="*60)
    subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":

    run("1. Dataset Build", "dataset_builder.py")
    run("2. LoRA Training", "train_lora.py")
    run("3. Merge LoRA", "merge_lora.py")

    run("4. Convert HF → GGUF", "convert_and_quantize.py")

    run("5. Build Ollama Model", "ollama_build.py")

    print("\n✅ PIPELINE COMPLETE\n")
    print("👉 Run your model with:")
    print("ollama run discord-bot")