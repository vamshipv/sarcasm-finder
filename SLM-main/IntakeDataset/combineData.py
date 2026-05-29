from datasets import load_from_disk, concatenate_datasets
from pathlib import Path

def main():
    script_dir = Path(__file__).parent
    
    print("Loading the 3 Titanium datasets...")
    # Update these paths to match your exact directory structure if needed
    ds_original = load_from_disk(script_dir / "v6_alpha_data") 
    ds_intake1 = load_from_disk(script_dir / "intake1_instruction_titanium")
    ds_intake2 = load_from_disk(script_dir / "intake2_starcoder_titanium")
    
    print(f"-> Original dataset size:    {len(ds_original)}")
    print(f"-> Instruction dataset size: {len(ds_intake1)}")
    print(f"-> StarCoder dataset size:   {len(ds_intake2)}")
    
    # 1. Concatenate them into a single master pool
    print("\nMerging datasets together...")
    master_dataset = concatenate_datasets([ds_original, ds_intake1, ds_intake2])
    
    # 2. Shuffle them thoroughly using a fixed seed
    print("Shuffling master dataset to blend patterns...")
    final_dataset = master_dataset.shuffle(seed=42)
    
    print(f"\nFinal Master Dataset Count: {len(final_dataset)} pristine functions!")
    
    # 3. Save the final version ready for the training script
    output_path = script_dir / "v7_master_titanium_data"
    final_dataset.save_to_disk(output_path)
    print(f"Successfully saved master training set to: {output_path}")

if __name__ == "__main__":
    main()