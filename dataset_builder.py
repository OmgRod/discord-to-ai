import json
import re
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# CONFIG FROM ENV
# ----------------------------

JSON_FOLDER = "json"
OUTPUT_FILE = "dataset.jsonl"
MAX_CONTEXT_MESSAGES = 6

BLOCKED_USERS = set(
    u.strip().lower()
    for u in os.getenv("BLOCKED_USERS", "").split(",")
    if u.strip()
)

CENSORED_WORDS = set(
    w.strip()
    for w in os.getenv("CENSORED_WORDS", "").split(",")
    if w.strip()
)

CENSOR_USERNAMES = os.getenv("CENSOR_USERNAMES", "true").lower() == "true"
USERNAME_TOKEN = os.getenv("USERNAME_TOKEN", "[USER]")


# ----------------------------
# HELPERS
# ----------------------------

def clean(text):
    return (text or "").strip()


def valid(msg):
    content = clean(msg.get("content", ""))

    if not content:
        return False

    if len(content) < 2:
        return False

    if content.startswith("http"):
        return False

    return True


def is_blocked_user(msg):
    name = msg.get("author", {}).get("name", "").lower()
    return name in BLOCKED_USERS


def censor_text(text):
    if not text:
        return ""

    result = text

    for word in CENSORED_WORDS:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        result = pattern.sub("*" * len(word), result)

    return result


def get_author(msg):
    if CENSOR_USERNAMES:
        return USERNAME_TOKEN
    return msg.get("author", {}).get("name", "unknown")


# ----------------------------
# LOAD
# ----------------------------

def load_all_messages():
    all_messages = []

    for file in Path(JSON_FOLDER).glob("*.json"):
        print(f"Loading {file.name}")

        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for msg in data.get("messages", []):

                if is_blocked_user(msg):
                    continue

                if not valid(msg):
                    continue

                msg["content"] = censor_text(msg.get("content", ""))

                all_messages.append(msg)

        except Exception as e:
            print(f"Failed to load {file}: {e}")

    print(f"\nLoaded {len(all_messages)} valid messages")
    return all_messages


# ----------------------------
# SORT TIME
# ----------------------------

def parse_time(msg):
    try:
        return datetime.fromisoformat(
            msg["timestamp"].replace("Z", "+00:00")
        )
    except:
        return datetime.min


# ----------------------------
# BUILD DATASET
# ----------------------------

def build_dataset(messages):

    messages.sort(key=parse_time)

    msg_by_id = {
        msg["id"]: msg
        for msg in messages
        if "id" in msg
    }

    dataset = []
    seen = set()

    for i, msg in enumerate(messages):

        content = clean(msg["content"])
        if not content:
            continue

        conversation = []

        # reply chain
        ref = msg.get("reference")

        if ref:
            parent = msg_by_id.get(ref.get("messageId"))

            if parent and valid(parent):
                conversation.append({
                    "role": "user",
                    "content": f"{get_author(parent)}: {clean(parent['content'])}"
                })

        # fallback context window
        if not conversation:
            start = max(0, i - MAX_CONTEXT_MESSAGES)
            history = messages[start:i]

            for old in history:
                if not valid(old):
                    continue

                conversation.append({
                    "role": "user",
                    "content": f"{get_author(old)}: {clean(old['content'])}"
                })

        conversation.append({
            "role": "assistant",
            "content": f"{get_author(msg)}: {content}"
        })

        if len(conversation) < 2:
            continue

        key = json.dumps(conversation, ensure_ascii=False)

        if key in seen:
            continue

        seen.add(key)
        dataset.append({"messages": conversation})

    return dataset


# ----------------------------
# SAVE
# ----------------------------

def save_dataset(dataset):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(dataset)} samples -> {OUTPUT_FILE}")


# ----------------------------
# MAIN
# ----------------------------

def main():
    messages = load_all_messages()
    dataset = build_dataset(messages)
    save_dataset(dataset)


if __name__ == "__main__":
    main()