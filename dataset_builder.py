import json

INPUT_FILE = "discord.json"
OUTPUT_FILE = "dataset.jsonl"

def clean(text):
    return (text or "").strip()

def valid(msg):
    c = msg.get("content", "")
    if not c:
        return False
    if len(c.strip()) < 2:
        return False
    if c.startswith("http"):
        return False
    return True

def main():
    data = json.load(open(INPUT_FILE, "r", encoding="utf-8"))
    messages = data["messages"]

    dataset = []
    seen = set()

    msg_by_id = {m["id"]: m for m in messages if valid(m)}

    for msg in messages:
        if not valid(msg):
            continue

        ref = msg.get("reference")
        if not ref:
            continue

        parent = msg_by_id.get(ref.get("messageId"))
        if not parent:
            continue

        prompt = clean(parent["content"])
        response = clean(msg["content"])

        if len(prompt) < 2 or len(response) < 2:
            continue

        key = (prompt, response)
        if key in seen:
            continue

        seen.add(key)

        dataset.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(dataset)} samples → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()