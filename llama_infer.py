from multiprocessing.managers import BaseManager
import json
import sys
import re

# ---------- MODEL CLIENT ----------
class ModelClient:
    def __init__(self):
        self.manager = None
        self.model = None
        self.connect()
    
    def connect(self):
        class ModelServer(BaseManager):
            pass
        
        ModelServer.register('get_model')
        
        try:
            self.manager = ModelServer(address=('localhost', 7002), authkey=b'moe_model_key')
            self.manager.connect()
            self.model = self.manager.get_model()
            print("Connected to Phi-3.5 model server successfully!")
        except Exception as e:
            print(f"Failed to connect to model server: {e}")
            print("Make sure model_server.py is running!")
            sys.exit(1)
    
    def inference(self, prompt, max_tokens=150, temperature=0.1, stop=None):
        return self.model.inference(prompt, max_tokens, temperature, stop)

# Global client instance
client = ModelClient()

SYSTEM_MOE_PROMPT = """You are a router that decides which tools to use and provides prompts for each tool.

Available tools:
- text: For answering with internal knowledge
- image: For visual generation  
- audio: For sound generation
- web: For real-time/external information

Output ONLY valid JSON:
{
  "tasks": {
    "text": "<prompt or null>",
    "image": "<prompt or null>", 
    "audio": "<prompt or null>",
    "web": "<search query or null>"
  },
  "final_decision": "<text|image|audio|web|combination>"
}

Examples:
- "What is AI?" → text: "Explain artificial intelligence"
- "Draw a cat" → image: "A cute cat"
- "Tesla stock news" → web: "Tesla stock latest news"

Rules:
- Provide PROMPTS for tools, not answers
- Use null when tool not needed
- Output JSON only, no extra text"""

# ---------- JSON REPAIR ----------
def fix_json(raw_output):
    """Try to fix common JSON issues while keeping structure intact."""
    if not raw_output:
        return '{"tasks":{"text":null,"image":null,"audio":null,"web":null},"final_decision":"text"}'

    raw_output = raw_output.strip()
    
    # Remove any text before first {
    if '{' in raw_output:
        raw_output = raw_output[raw_output.find('{'):]
    
    # Remove any text after last }
    if '}' in raw_output:
        raw_output = raw_output[:raw_output.rfind('}') + 1]

    # Force starts/ends if missing
    if not raw_output.startswith('{'):
        raw_output = '{' + raw_output
    if not raw_output.endswith('}'):
        raw_output = raw_output + '}'

    return raw_output

# ---------- MAIN MOE ROUTER ----------
def run_moe(query: str):
    """
    query   → user text
    """
    
    # Phi-3.5 chat format for maximum speed and accuracy
    prompt = f"""<|system|>
{SYSTEM_MOE_PROMPT}<|end|>
<|user|>
{query}<|end|>
<|assistant|>
"""

    # print(f"\nDEBUG: Sending Phi-3.5 formatted prompt:\n{prompt}\n")

    raw = client.inference(
        prompt,
        max_tokens=150,  # Reduced for speed
        temperature=0.1,  # Lower for faster, more deterministic sampling
        stop=["<|end|>", "<|endoftext|>", "###", "\n\n"]
    )

    # print(f"DEBUG: Raw response: {repr(raw)}")

    if isinstance(raw, dict) and "error" in raw:
        return raw

    # Empty output → fallback
    if not raw or raw.strip() == "":
        return {
            "tasks": {
                "text": query,
                "image": None,
                "audio": None,
                "web": None
            },
            "final_decision": "text",
            "error": "Empty output from Phi-3.5 model"
        }

    fixed = fix_json(raw)
    print(f"DEBUG: Fixed JSON: {fixed}")

    try:
        return json.loads(fixed)
    except Exception as e:
        return {
            "tasks": {
                "text": query,
                "image": None,
                "audio": None,
                "web": None
            },
            "final_decision": "text",
            "error": f"JSON parse error: {str(e)}",
            "raw_output": fixed
        }

# ---------- MAIN ENTRY POINT ----------
if __name__ == "__main__":
    query = "What is the capital of France?"
    
    # print(f"Query: {query}")
    out = run_moe(query)
    print(json.dumps(out, indent=2))