import subprocess

MODEL = "discord-bot"

if __name__ == "__main__":

    with open("Modelfile", "w", encoding="utf-8") as f:
        f.write("""
FROM ./model-q4.gguf

TEMPLATE \"\"\"
{{ .Prompt }}
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
""")

    subprocess.check_call(f"ollama create {MODEL} -f Modelfile", shell=True)

    print("Ollama model created:", MODEL)