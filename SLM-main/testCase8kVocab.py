import torch
from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedTokenizerFast

# 1. Setup paths
# NOTE: Check if your model is in 'baby-llama-racecar' or a sub-checkpoint folder
model_path = "./v1_29M_coder_model/checkpoint-1828" 
tokenizer_path = "./v1_8k_code_tokenizer"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Testing Racecar on: {device}")

# 2. Load the Custom 8k Tokenizer & Brain
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
model = LlamaForCausalLM.from_pretrained(model_path).to(device)

def generate_code(prompt, tokens=50, temp=0.5):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=tokens, 
        temperature=temp, 
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id or 3, # Use 3 if <pad> ID is 3
        repetition_penalty=1.15
        #,top_p = 0.9
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. The Litmus Tests
print("\n--- Test 1: Variable Logic ---")
print(generate_code("def check_name(): \n    user_age = 25\n    if user_name =="))


# Notes
# 1. The "Weight Decay" SqueezeIn your next run, increase the weight_decay to 0.1. Why: Weight decay acts like a "simplicity filter." 
# It forces the model to use the smallest possible mathematical explanation for a pattern. This often kills the "hallucination" of weird file names 
# and forces the model to stick to simpler, more common keywords.2. The "Temperature" ChillIn your test_racecar.py, drop the temperature even lower, 
# to 0.05.Why: At 0.2, the model is still allowed to be "inspired." At 0.05, it will become a boring robot that only picks the most statistically certain word.
# It will likely stop saying import_2.py and start saying print("Adult").3. The "Unfreeze" (The Risky Move)We can try unfreezing all layers but using a microscopic 
# learning rate ($1 \times 10^{-5}$).Why: This allows the "Foundation" layers to shift just a tiny bit to accommodate the new General Python logic without losing their mind.