## Discord Chat → AI Model

This project takes exported Discord conversations (for example, from [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter)) and turns them into a dataset that can be used to train a small conversational AI model. The goal is to learn the *tone, style, and structure* of real chat messages so that the resulting model can generate responses that feel like the original chat participants.

It works by processing raw Discord export JSON files, extracting meaningful message-reply relationships, and converting them into a structured dataset suitable for fine-tuning language models such as Mistral, Phi-2, or similar open models.

---

## What this project does

### 1. Dataset building

The first step is parsing exported Discord JSON files and extracting conversation pairs.

* Filters out empty messages, links, and irrelevant content
* Removes unwanted users (optional)
* Detects reply chains using Discord’s `reference.messageId`
* Builds training pairs in the form:

```json
{
  "messages": [
    { "role": "user", "content": "original message" },
    { "role": "assistant", "content": "reply message" }
  ]
}
```

This creates a dataset that mimics real conversational structure instead of random text chunks.

---

### 2. Fine-tuning a language model

The dataset is then used to fine-tune a causal language model using Hugging Face Transformers.

The training script:

* Loads a pretrained base model (e.g. Phi-2 or Mistral)
* Formats Discord-style messages into a consistent prompt format
* Tokenizes the dataset efficiently
* Runs supervised fine-tuning using `Trainer`
* Saves the resulting model locally

The output is a trained model directory that can be reused or exported.

---

### 3. Model export and conversion

Once training is complete, the model can be prepared for use in inference tools such as Ollama or llama.cpp.

Typical steps include:

* Merging fine-tuned adapters (if using LoRA/PEFT)
* Converting the model into GGUF format using `llama.cpp`
* Quantizing the model for lower memory usage
* Importing it into Ollama as a local model

---

## Running the project

### 1. Build dataset

Export your Discord server or DM using DiscordChatExporter and place the JSON file in the project directory.

Then run:

```bash
python build_dataset.py
```

This will generate:

```
dataset.jsonl
```

---

### 2. Fine-tune the model

Install dependencies:

```bash
pip install -r requirements.txt
```

Then start training:

```bash
python finetune.py
```

This will output a trained model folder such as:

```
trained-model/
```

---

### 3. Convert to GGUF (llama.cpp)

Clone and build llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

Convert model:

```bash
python convert_hf_to_gguf.py ../trained-model --outfile model.gguf
```

---

### 4. Quantize model

Example (4-bit quantization):

```bash
./build/bin/llama-quantize model.gguf model-q4.gguf q4_0
```

---

### 5. Import into Ollama

Create a Modelfile:

```
FROM ./model-q4.gguf
```

Then create the model:

```bash
ollama create discord-bot -f Modelfile
```

Run it:

```bash
ollama run discord-bot
```

---

## Notes

* Small datasets (like a few hundred conversations) will overfit quickly but can still capture tone
* More data = better generalization
* GPU training is strongly recommended for anything above small models
* LoRA fine-tuning is recommended if you want lower memory usage and faster iteration

---

## Future improvements

* Better reply-thread reconstruction
* Multi-user role modeling (so the AI can mimic different people)
* Filtering by conversation quality
* Web UI for dataset cleaning and preview
* Real-time chat simulator for testing models
