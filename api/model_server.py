from multiprocessing.managers import BaseManager
from llama_cpp import Llama
import sys

SYSTEM_MOE_PROMPT = """You are a router that decides which tools to use and provides prompts for each tool.
- text: For answering with internal knowledge
- image: For visual generation
- audio: For sound generation
- web: For real-time/external information (only when user explicitly asks or when current events/real-time data is needed)

Output ONLY valid JSON:
{
        "tasks": {
                "text": "<prompt or null>",
                "image": "<prompt or null>",
                "audio": "<prompt or null>",
                "web": "<search query or null>"
        }
}

Rules:
- Use web search ONLY if: user explicitly requests it, or the question requires current events/real-time information
- Use text modality for general knowledge questions
- Provide PROMPTS for tools, not answers
- Output JSON only, no extra text
"""

class ModelManager:
        def __init__(self):
                self.llm = None
                self.load_model()
                self.build_system_prefix()

        def load_model(self):
                self.llm = Llama(
                        model_path="models/Phi-3.5-mini-instruct-Q4_K_M.gguf",
                        n_ctx=2048,
                        n_gpu_layers=-1,
                        n_threads=2,
                        n_batch=1024,  # Reduced for multiple instances
                        use_mlock=True,
                        use_mmap=True,
                        flash_attn=True,
                        verbose=False
                )

        def build_system_prefix(self):
                self.system_prefix = (
                        f"<|system|>\n{SYSTEM_MOE_PROMPT}\n"
                        f"<|user|>\n"
                )
                self.system_tokens = self.llm.tokenize(self.system_prefix.encode("utf-8"))

        def fast_inference(self, user_msg, max_tokens=60):
                full_prompt = (
                        self.system_prefix
                        + user_msg
                        + "\n<|assistant|>\n"
                )

                tokens = self.llm.tokenize((full_prompt.encode("utf-8")))

                output = self.llm(
                        prompt = tokens,
                        max_tokens=max_tokens,
                        temperature=0.1,
                        top_p=0.8,
                        top_k=40,
                        echo=False
                )

                return output["choices"][0]["text"].strip()


model_manager = ModelManager()

class ModelServer(BaseManager):
        pass

ModelServer.register("get_model", callable=lambda: model_manager)

if __name__ == "__main__":
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 7002
        print(f"Starting model server on port {port}")
        server = ModelServer(address=("localhost", port), authkey=b"moe_model_key")
        server.get_server().serve_forever()