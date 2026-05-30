import subprocess
import os

MERGED = "../merged"
OUT_GGUF = "../model.gguf"
Q_OUT = "../model-q4.gguf"

LLAMA_CPP = "../llama.cpp"

def run(cmd):
    print(">", cmd)
    subprocess.check_call(cmd, shell=True)

if __name__ == "__main__":

    os.chdir(LLAMA_CPP)

    # convert
    run(f"python convert_hf_to_gguf.py {MERGED} --outfile {OUT_GGUF}")

    # quantize (must exist from build OR download binaries)
    run(f"llama-quantize.exe {OUT_GGUF} {Q_OUT} q4_k_m")

    print("GGUF READY:", Q_OUT)