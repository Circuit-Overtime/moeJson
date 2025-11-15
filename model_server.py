from multiprocessing.managers import BaseManager
from multiprocessing.managers import BaseManager
from llama_cpp import Llama
import json
import sys

class ModelManager:
    def __init__(self):
        self.llm = None
        self.load_model()
    
    def load_model(self):
        print("Loading TinyLlama model optimized for Tesla T4...")
        self.llm = Llama(
            model_path="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            n_ctx=1024,
            n_threads=4,  # Reduced threads for GPU focus
            n_gpu_layers=-1,  # Move ALL layers to GPU (T4 has 16GB VRAM)
            n_batch=512,  # Larger batch size for T4
            use_mmap=True,  # Memory mapping for faster loading
            use_mlock=True,  # Lock model in RAM
            rope_scaling_type=1,  # Enable RoPE scaling
            rope_freq_base=10000.0,
            verbose=False,
            # GPU-specific optimizations
            split_mode=1,  # Split by layer for multi-GPU (even single GPU benefits)
            main_gpu=0,  # Use first GPU
            tensor_split=None,  # Let it auto-distribute
            low_vram=False,  # T4 has plenty of VRAM
        )
        print("Model loaded successfully with GPU acceleration!")
    
    def inference(self, prompt, max_tokens=250, temperature=0.2, stop=None):
        if self.llm is None:
            return {"error": "Model not loaded"}
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or ["###", "\n\n"],
                echo=False,  
                top_p=0.95,  # Slightly higher for better quality
                top_k=40,
                repeat_penalty=1.1,  # Prevent repetition
                # Performance optimizations
                stream=False,  # No streaming for faster batch processing
                frequency_penalty=0.0,
                presence_penalty=0.0,
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
    print("Starting Model Server on port 7002...")
    
    server = ModelServer(address=('localhost', 7002), authkey=b'moe_model_key')
    server_obj = server.get_server()
    
    print("Model Server ready! Listening on localhost:7002")
    print("GPU acceleration enabled for Tesla T4")
    try:
        server_obj.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Model Server...")
        server_obj.shutdown()