from quart import Quart, request, jsonify
from multiprocessing.managers import BaseManager
import asyncio
import json


app = Quart(__name__)

class ModelClient(BaseManager):
    pass

ModelClient.register("get_model")

manager = ModelClient(address=("localhost", 7002), authkey=b"moe_model_key")
manager.connect()
model = manager.get_model()

def count_words(text):
    return len(text.split())

@app.route("/gen", methods=["GET", "POST"])
async def generate():
    try:
        if request.method == "GET":
            prompt = request.args.get("prompt", "").strip()
        else:
            data = await request.get_json()
            prompt = data.get("prompt", "").strip()
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        word_count = count_words(prompt)
        if word_count > 100:
            return jsonify({"error": f"Prompt exceeds 100 words (current: {word_count})"}), 400
        
        response = model.fast_inference(prompt)
        
        result = json.loads(response)
        
        return jsonify({
            "prompt": prompt,
            "word_count": word_count,
            "result": result
        }), 200
    
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON response from model"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, workers=10)