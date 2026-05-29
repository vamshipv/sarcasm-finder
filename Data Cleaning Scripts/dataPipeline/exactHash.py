import ast
import hashlib
from datasets import load_from_disk, Dataset

# In a dataset of over 4,000 scripts, the statistical probability of having zero duplicates is almost non-existent. 
# The reason your report says "0" is that Exact Hashing is incredibly fragile.
# --- CONFIGURATION ---
# We define what "Thinking" looks like in code
LOGIC_NODES = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.FunctionDef, ast.ClassDef)

def get_complexity_and_verify(code):
    """
    Check if the code is valid Python and count logic decision points.
    Returns: (is_valid, complexity_score)
    """
    try:
        tree = ast.parse(code)
        # Walk the tree and count every time the model has to make a 'choice'
        score = sum(1 for n in ast.walk(tree) if isinstance(n, LOGIC_NODES))
        return True, score
    except Exception:
        return False, 0

def run_audit_pipeline(dataset_path):
    print("Starting the Final Data Audit...")
    ds = load_from_disk(dataset_path)
    
    seen_hashes = set()
    final_list = []
    
    stats = {
        "original": len(ds),
        "duplicates": 0,
        "broken_syntax": 0,
        "total_complexity": 0,
        "low_logic": 0
    }

    for entry in ds:
        content = entry['content']
        
        # 1. Trial of Uniqueness (Hashing)
        # We hash the content to find exact duplicates even with different names
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        if content_hash in seen_hashes:
            stats["duplicates"] += 1
            continue
        
        # 2. Trial of Integrity & Complexity (AST)
        is_valid, score = get_complexity_and_verify(content)
        
        if not is_valid:
            stats["broken_syntax"] += 1
            continue
            
        # 3. Trial of Value
        # If a script is too simple (score < 2), it's not worth training on
        if score < 2:
            stats["low_logic"] += 1
            continue

        # If it survived, add to the elite list
        seen_hashes.add(content_hash)
        stats["total_complexity"] += score
        
        # Store the complexity so we can sort by it later
        entry['complexity'] = score
        final_list.append(entry)

    # Wrap back into a Dataset object
    final_ds = Dataset.from_list(final_list)
    
    # Calculate the Average IQ (Complexity)
    avg_iq = stats["total_complexity"] / len(final_ds) if len(final_ds) > 0 else 0
    
    return final_ds, stats, avg_iq

# --- EXECUTION ---
# Path to your v3_cleaned_python folder
input_path = "./starCoder_cleaned" 
final_dataset, audit_stats, average_complexity = run_audit_pipeline(input_path)

print("\n--- FINAL AUDIT REPORT ---")
print(f"Total Scripts Evaluated: {audit_stats['original']}")
print(f"Duplicates Removed:      {audit_stats['duplicates']}")
print(f"Broken Syntax Removed:   {audit_stats['broken_syntax']}")
print(f"Low Logic Removed:       {audit_stats['low_logic']}")
print(f"Final Elite Dataset:     {len(final_dataset)}")
print(f"Average Complexity:      {average_complexity:.2f}")

# Save the Gold Standard dataset
final_dataset.save_to_disk("./exact_hash_starCoder_python")
print("\nSuccess. Gold Standard dataset saved.")


# --- FINAL AUDIT REPORT --- for 250K scripts input:
# Total Scripts Evaluated: 65309
# Duplicates Removed:      1
# Broken Syntax Removed:   34349
# Low Logic Removed:       65
# Final Elite Dataset:     30894
# Average Complexity:      12.50 The "Average Complexity" Sweet Spot
# Your new average complexity score is 12.50.

# This is an exceptionally healthy metric. Previously, your dataset showed an average complexity of 41.27, 
# which meant the files were far too dense and long to fit inside a 512-token context window without getting cut off. 
# A score of 12.52 means you have a collection of clean, highly readable functions and classes containing real conditional 
# logic (if/else, loops, try-except blocks) that a small model can completely comprehend from start to finish.