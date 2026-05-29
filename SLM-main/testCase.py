import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from pathlib import Path

def generate_code(prompt, max_tokens=128):
    # Set the path to your new Llama checkpoint
    script_dir = Path(__file__).parent
    model_path = script_dir / "v1_alpha_copilot" / "checkpoint-335"
    
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
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.2,          
            top_p=0.90,
            top_k =40,
            repetition_penalty=1.15,  # Prevents infinite loops
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
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
    test_prompt = "def add(x, y):\n"
    generate_code(test_prompt, max_tokens=50)