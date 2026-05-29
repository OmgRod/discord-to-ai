from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-0.5B"
LORA = "./lora"
OUT = "./merged"

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="auto")
model = PeftModel.from_pretrained(model, LORA)

model = model.merge_and_unload()

model.save_pretrained(OUT)

tok = AutoTokenizer.from_pretrained(BASE)
tok.save_pretrained(OUT)

print("Merged →", OUT)