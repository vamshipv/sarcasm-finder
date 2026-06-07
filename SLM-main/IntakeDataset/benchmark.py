import torch
import ast
import time
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from pathlib import Path

# ==========================================
# 1. THE TEST SUITE
# ==========================================
TEST_CASES = [
    {
        "name": "Basic Math Rhythm",
        "prompt": "def calculate_discount(price, discount_percent):\n",
        "expected_vars": ["price", "discount_percent"]
    },
    {
        "name": "Dictionary Lookup",
        "prompt": "def get_user_age(user_dict, user_name):\n    \"\"\"Return the age from the dictionary\"\"\"\n",
        "expected_vars": ["user_dict", "user_name"]
    },
    {
        "name": "Exception Handling",
        "prompt": "def read_file_safely(filepath):\n    try:\n        return open(filepath, 'r').read()\n    except",
        "expected_vars": []
    },
    {
        "name": "List Comprehension",
        "prompt": "def get_positive_numbers(numbers_list):\n",
        "expected_vars": ["numbers_list"]
    }
]

# ==========================================
# 2. AUTOMATED GRADER
# ==========================================
def grade_output(prompt, generated_code, expected_vars):
    full_code = prompt + generated_code
    score = {"syntax": False, "grounding": False}
    
    # Check 1: Structural Integrity (AST Parse)
    try:
        ast.parse(full_code)
        score["syntax"] = True
    except SyntaxError:
        score["syntax"] = False

    # Check 2: Variable Grounding
    if expected_vars:
        used_vars = all(var in generated_code for var in expected_vars)
        score["grounding"] = used_vars
    else:
        score["grounding"] = "N/A"
        
    return score

# ==========================================
# 3. BENCHMARK ENGINE
# ==========================================
def run_benchmark():
    script_dir = Path(__file__).parent
    model_path = script_dir / "v2_master_copilot" # Point to your best checkpoint
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Benchmark Engine on {device.upper()}...")

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    tokenizer.backend_tokenizer.decoder = ByteLevelDecoder()
    
    # IMPORTANT: Put your custom double newline ID here (e.g., 405, 198, etc.)
    # double_newline_id = <YOUR_ID_HERE> 
    
    model = LlamaForCausalLM.from_pretrained(model_path).to(device).eval()

    total_score = 0
    
    print("\n" + "="*50)
    print("STARTING MICRO-COPILOT BENCHMARK")
    print("="*50)

    for i, test in enumerate(TEST_CASES):
        print(f"\nTest {i+1}: {test['name']}")
        
        inputs = tokenizer(tokenizer.bos_token + test["prompt"], return_tensors="pt").to(device)
        
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                
                # The Goldilocks Parameters
                temperature=0.45,         
                top_p=0.90,               
                top_k=25,                 
                repetition_penalty=1.25,  
                no_repeat_ngram_size=3,   
                
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=[tokenizer.eos_token_id, 342] # UNCOMMENT THIS
            )
        generation_time = time.time() - start_time

        # Extract ONLY the newly generated text
        input_length = inputs.input_ids.shape[1]
        new_tokens = output_ids[0][input_length:]
        generated_code = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Grade it
        grades = grade_output(test["prompt"], generated_code, test["expected_vars"])
        
        print(f"Time: {generation_time:.2f}s")
        print(f"--- Generated Completion ---")
        print(generated_code.strip())
        print(f"--- Grades ---")
        print(f"AST Syntax Valid: {'✅ PASS' if grades['syntax'] else '❌ FAIL'}")
        print(f"Vars Grounded:    {'✅ PASS' if grades['grounding'] == True else '❌ FAIL' if grades['grounding'] == False else '⚪ N/A'}")
        
        if grades["syntax"]: total_score += 1

    print("\n" + "="*50)
    print(f"BENCHMARK COMPLETE | Final Syntax Score: {total_score}/{len(TEST_CASES)}")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()