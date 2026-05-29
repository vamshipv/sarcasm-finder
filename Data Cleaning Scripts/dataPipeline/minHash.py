import hashlib
import re
from datasets import load_from_disk, Dataset

# --- CONFIGURATION ---
K_SIZE = 9             # Character window size for shingling
NUM_HASHES = 128       # Size of the signature (128 is industry standard)
THRESHOLD = 0.85       # Similarity limit (85%)
MIN_CHARS = 150        # Minimum length to be useful
MAX_CHARS = 2500       # Maximum length to fit in 512-token window

def get_shingles(text):
    """Normalize whitespace and create a set of character n-grams."""
    text = re.sub(r'\s+', ' ', text).strip()
    shingles = set()
    for i in range(len(text) - K_SIZE + 1):
        shingles.add(text[i:i + K_SIZE])
    return shingles

def get_signature(shingles):
    """Generate a MinHash signature using salts to simulate different hash functions."""
    signature = []
    for i in range(NUM_HASHES):
        min_val = float('inf')
        for s in shingles:
            # Create a unique hash for this index i
            h = int(hashlib.md5(f"{i}{s}".encode('utf-8')).hexdigest(), 16)
            if h < min_val:
                min_val = h
        signature.append(min_val)
    return signature

def is_similar(sig1, sig2):
    """Calculate the Jaccard similarity between two signatures."""
    match = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return (match / NUM_HASHES) >= THRESHOLD

def run_minhash_dedup(input_path):
    print("Loading data from Auditor output...")
    ds = load_from_disk(input_path)
    
    # 1. First Pass: Apply Length Filter and Generate Signatures
    filtered_data = []
    print(f"Applying length filter ({MIN_CHARS}-{MAX_CHARS} chars)...")
    
    for entry in ds:
        content = entry['content']
        if MIN_CHARS <= len(content) <= MAX_CHARS:
            shingles = get_shingles(content)
            if shingles:
                entry['minhash_sig'] = get_signature(shingles)
                filtered_data.append(entry)
    
    print(f"Scripts within length range: {len(filtered_data)}")

    # 2. Second Pass: Find Near-Duplicates
    final_list = []
    seen_signatures = []
    near_duplicates_found = 0

    print("Checking for near-duplicates (Similarity > 85%)...")
    for entry in filtered_data:
        current_sig = entry['minhash_sig']
        is_dup = False
        
        # Compare current script against everything we've already accepted
        for seen_sig in seen_signatures:
            if is_similar(current_sig, seen_sig):
                is_dup = True
                near_duplicates_found += 1
                break
        
        if not is_dup:
            seen_signatures.append(current_sig)
            # Remove the signature from the entry before saving to save space
            del entry['minhash_sig']
            final_list.append(entry)

    return Dataset.from_list(final_list), near_duplicates_found

# --- EXECUTION ---
input_dir = "./exact_hash_starCoder_python"
output_dir = "./minhash_deduped_starCoder_python"

final_ds, dups_count = run_minhash_dedup(input_dir)

print("\n--- MINHASH DEDUPLICATION REPORT ---")
print(f"Near-Duplicates Removed: {dups_count}")
print(f"Final Training Samples:  {len(final_ds)}")
print(f"Data saved to:           {output_dir}")

final_ds.save_to_disk(output_dir)