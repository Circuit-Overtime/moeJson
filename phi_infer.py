from llama_cpp import Llama
import json
import sys

MODEL_PATH = "models/Phi-3.5-mini-instruct-Q8_0.gguf"

# Load model with maximum GPU layers
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=6,
    n_gpu_layers=-1,       # <-- use all GPU memory automatically
    verbose=False
)

SYSTEM_PROMPT = """
You are Phi 3.5 acting as a Routing-MoE.
Your job is to decide what tasks the assistant must perform.

Rules:
1. ALWAYS RESPOND IN VALID JSON. NO text outside JSON.
2. JSON structure must be:

{
  "tasks": {
      "text": "<text_response_or_empty>",
      "image": "<image_generation_query_or_empty>",
      "audio": "<audio_generation_query_or_empty>",
      "web": "<web_search_query_or_empty>"
  },
  "reasoning": "<short explanation>"
}

3. Use empty string "" for any unused task.
4. Use "web" only if:
   - The model likely does not know the answer, OR
   - The query requires up-to-date information.
"""

def run_inference(user_query: str):
    prompt = f"""
### SYSTEM:
{SYSTEM_PROMPT}

### USER:
{user_query}

### ASSISTANT (JSON ONLY):
"""

    out = llm(
        prompt,
        max_tokens=300,
        temperature=0.2,
        stop=["###"]  # stop early so JSON stays clean
    )

    raw = out["choices"][0]["text"].strip()

    # attempt to fix incomplete JSON
    try:
        parsed = json.loads(raw)
        print(json.dumps(parsed, indent=4))
    except:
        # fallback: try auto-fixing by trimming till last '}'
        try:
            fixed = raw[:raw.rfind("}")+1]
            parsed = json.loads(fixed)
            print(json.dumps(parsed, indent=4))
        except:
            print("RAW OUTPUT:\n", raw)


if __name__ == "__main__":
    query = "What is the capital of France?"
    run_inference(query)
