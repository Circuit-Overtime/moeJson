from llama_cpp import Llama
import json

llm = Llama(
    model_path="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=0
)

system_prompt = "You are TinyLlama. Always answer in JSON."

while True:
    user_msg = input("\nYou: ")

    prompt = f"""
### System:
{system_prompt}

### User:
{user_msg}

### Assistant (JSON only):
"""

    out = llm(
        prompt,
        max_tokens=200,
        temperature=0.7,
        stop=["###"]
    )

    text = out["choices"][0]["text"].strip()

    # try parsing JSON
    try:
        obj = json.loads(text)
        print("\nAssistant:", json.dumps(obj, indent=4))
    except:
        print("\nAssistant (raw):", text)
