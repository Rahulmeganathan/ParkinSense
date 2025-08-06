#!/usr/bin/env python3
"""
Subprocess-based GGUF model wrapper for memory-constrained environments
"""

import sys
import json
import subprocess
from pathlib import Path

def create_gguf_subprocess_wrapper():
    """Create a subprocess script that can load GGUF model independently"""
    wrapper_script = '''
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

def predict_with_model(model_path, settings, prompt):
    """Load model and make prediction in subprocess"""
    try:
        from llama_cpp import Llama
        
        # Load model
        model = Llama(model_path=model_path, **settings)
        
        # Make prediction
        response = model.create_completion(
            prompt,
            max_tokens=10,
            temperature=0.1,
            stop=["\\n", ".", ",", " ", "Explanation"],
            echo=False
        )
        
        return {
            "status": "success", 
            "prediction": response['choices'][0]['text'].strip(),
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
        result = predict_with_model(model_path, settings, prompt)
        print(json.dumps(result))
        
    else:
        print(json.dumps({"status": "error", "error": "Unknown command"}))
'''
    
    return wrapper_script

# Write the wrapper script
wrapper_path = Path(__file__).parent / "gguf_subprocess_wrapper.py"
with open(wrapper_path, 'w') as f:
    f.write(create_gguf_subprocess_wrapper())

print(f"Created GGUF subprocess wrapper at: {wrapper_path}")
