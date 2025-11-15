from multiprocessing.managers import BaseManager
from llama_cpp import Llama
import json
import sys

class ModelManager:
    def __init__(self):
        self.llm = None
        self.load_model()
    
    def load_model(self):
        print("Loading Phi-3.5-mini-instruct optimized for Tesla T4...")
        self.llm = Llama(
            model_path="models/Phi-3.5-mini-instruct-Q8_0.gguf",
            n_ctx=2048,  # Phi-3.5 works better with larger context
            n_threads=2,  # Minimal CPU threads - focus on GPU
            n_gpu_layers=-1,  # ALL layers to GPU
            n_batch=2048,  # Max batch size for T4 throughput
            use_mmap=True,
            use_mlock=True,
            verbose=False,
            # GPU optimizations for max speed
            main_gpu=0,
            low_vram=False,
            # Flash attention for speed (if available)
            flash_attn=True,
        )
        print("Phi-3.5-mini loaded with maximum GPU acceleration!")
    
    def inference(self, prompt, max_tokens=150, temperature=0.1, stop=None):
        if self.llm is None:
            return {"error": "Model not loaded"}
        
        try:
            # Ultra-fast inference settings
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,  # Lower temp = faster sampling
                stop=stop or ["###", "\n\n", "<|end|>", "<|endoftext|>"],
                echo=False,
                # Minimal sampling for max speed
                top_p=0.8,  # Reduced for speed
                top_k=20,   # Reduced for speed
                repeat_penalty=1.05,  # Minimal penalty
                stream=False,
                # Disable expensive features
                seed=-1,  # Random seed (faster)
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            return {"error": str(e)}

# Global model instance
model_manager = ModelManager()

# Register the manager
class ModelServer(BaseManager):
    pass

ModelServer.register('get_model', callable=lambda: model_manager)

if __name__ == "__main__":
    print("Starting Phi-3.5 Model Server on port 7002...")
    
    server = ModelServer(address=('localhost', 7002), authkey=b'moe_model_key')
    server_obj = server.get_server()
    
    print("Model Server ready! Listening on localhost:7002")
    print("Phi-3.5-mini with maximum Tesla T4 acceleration")
    try:
        server_obj.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Model Server...")
        server_obj.shutdown()