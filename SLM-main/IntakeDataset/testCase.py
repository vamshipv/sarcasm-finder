import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from pathlib import Path

def generate_code(prompt, max_tokens=128):
    # Set the path to your new Llama checkpoint
    script_dir = Path(__file__).parent
    model_path = script_dir / "v2_master_copilot" / "checkpoint-5991"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # 1. Load Tokenizer
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        # Fix the ByteLevel Decoder to clean up Ġ and Ċ characters
        tokenizer.backend_tokenizer.decoder = ByteLevelDecoder()
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 2. Load the Llama Model (Swapped out GPT2 for Llama)
    try:
        print(f"Loading Llama weights from {model_path}...")
        model = LlamaForCausalLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Tokenize Input
    full_prompt = tokenizer.bos_token + prompt
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    print("\nThinking...")
    with torch.no_grad():
        # output_ids = model.generate(
        #     **inputs,
        #     max_new_tokens=max_tokens,
        #     do_sample=True,
        #     temperature=0.7,          # Dropped from 0.5 to crush low-probability glitches like 'type:'
        #     top_p=1,               # Slightly restricted to cut off long tails of weird tokens
        #     top_k=20,                 # Cut down from 40 to ensure it only considers top-tier candidates
        #     repetition_penalty=1.5,  # Maintained gently to prevent infinite indentation loops
        #     pad_token_id=tokenizer.pad_token_id,
        #     eos_token_id=tokenizer.eos_token_id
        # )

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,        
            do_sample=True,
            
            # 1. The Ice Box (Kill the creativity)
            temperature=0.5,               
            # 2. The Narrow Tunnel
            top_p=0.95,               
            top_k=30,                 # Only ever consider the 10 most obvious next words
            # 3. The Gentle Nudge (Fixing the Hammer)
            repetition_penalty=1.35,  # Just enough to stop infinite loops, but allows variable reuse
            pad_token_id=tokenizer.pad_token_id,
            # NEW: The "Stop Talking" Token
            # This forces the model to stop generating the moment it tries to start a new function
            # 198 is the standard byte-level token ID for a double newline (\n\n)
            eos_token_id=[tokenizer.eos_token_id, 198] 
        )

    # 4. Decode the Clean Output
    completed_code = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("\n" + "="*80)
    print("CLEAN OUTPUT GENERATION:")
    print("="*80)
    print(completed_code)
    print("="*80 + "\n")

if __name__ == "__main__":
    # A simple, grounded math function to test the syntax rhythm
    test_prompt = "def safe_divide(a, b):\n    try:\n        return a / b\n    except"
    generate_code(test_prompt, max_tokens=50)