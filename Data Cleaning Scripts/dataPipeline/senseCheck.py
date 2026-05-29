import random
from datasets import load_from_disk

def run_sense_check(dataset_path, num_samples=5):
    print("Loading Titanium Dataset...")
    ds = load_from_disk(dataset_path)
    total_files = len(ds)
    
    if total_files == 0:
        print("Dataset is empty. Something went wrong.")
        return

    print(f"Dataset loaded. Total files: {total_files}")
    print(f"Extracting {num_samples} random samples for manual review...\n")
    
    # Generate random indices
    random_indices = random.sample(range(total_files), min(num_samples, total_files))
    
    for i, idx in enumerate(random_indices, 1):
        sample = ds[idx]
        content = sample['content']
        complexity = sample.get('complexity', 'Unknown')
        char_length = len(content)
        
        # Formatting the output for the terminal
        print("=" * 80)
        print(f"SAMPLE {i} / {num_samples} | Index in dataset: {idx}")
        print(f"Length: {char_length} chars | Complexity Score: {complexity}")
        print("-" * 80)
        print(content)
        print("=" * 80)
        print("\n")
        
        # Pause after each sample to let you read it
        if i < num_samples:
            input("Press Enter to see the next sample...")

if __name__ == "__main__":
    run_sense_check("./minhash_deduped_starCoder_python", num_samples=5)