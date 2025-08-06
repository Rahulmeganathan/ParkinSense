
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_gguf_model(model_path, settings):
    """Load GGUF model in isolated subprocess"""
    try:
        from llama_cpp import Llama
        
        # Load with provided settings
        model = Llama(
            model_path=model_path,
            **settings
        )
        
        return {"status": "success", "model_loaded": True}
        
    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}

def predict_with_model(model_path, settings, prompt, is_companion=False):
    """Load model and make prediction in subprocess"""
    try:
        from llama_cpp import Llama
        
        # Load model
        model = Llama(model_path=model_path, **settings)
        
        # Adjust parameters based on mode
        if is_companion:
            # More creative settings for companion/therapy responses
            max_tokens = 600  # Significantly increased for detailed exercise instructions and comprehensive therapy guidance
            temperature = 0.8  # Higher creativity to avoid repetition
            stop = ["\n\nHuman:", "\n\nUser:", "Human:", "User:", "\n\nQ:", "\n\nQuestion:", "Session ID:", "\n\nThis response is"]
        else:
            # Conservative settings for medical predictions
            max_tokens = 20  # Increased for better responses
            temperature = 0.3  # Slightly higher for more variability
            stop = ["\n", "Answer:", "Classification:", "Response:"]
        
        # Make prediction
        response = model.create_completion(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            echo=False,
            repeat_penalty=1.2,  # Increased to prevent repetition
            top_p=0.9,  # Add nucleus sampling for variety
            top_k=40,   # Limit vocabulary for coherence
            frequency_penalty=0.3,  # Penalize repeated tokens
            presence_penalty=0.3    # Encourage diverse responses
        )
        
        prediction = response['choices'][0]['text'].strip()
        
        return {
            "status": "success", 
            "prediction": prediction,
            "full_response": response
        }
        
    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "error": "No command provided"}))
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "test_load":
        model_path = sys.argv[2]
        settings = json.loads(sys.argv[3])
        result = load_gguf_model(model_path, settings)
        print(json.dumps(result))
        
    elif command == "predict":
        model_path = sys.argv[2]
        settings = json.loads(sys.argv[3])
        prompt = sys.argv[4]
        is_companion = False
        
        # Check if companion flag is provided
        if len(sys.argv) > 5:
            is_companion = sys.argv[5].lower() == 'true'
            
        result = predict_with_model(model_path, settings, prompt, is_companion)
        print(json.dumps(result))
        
    else:
        print(json.dumps({"status": "error", "error": "Unknown command"}))
