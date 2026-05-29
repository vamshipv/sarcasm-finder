import os
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast

def train_and_inspect_tokenizer():
    dataset_path = "../Data Cleaning Scripts/dataPipeline/minhash_deduped_starCoder_python"
    output_dir = "./v1_8k_code_tokenizer"
    vocab_file_path = "vocab_inspection.txt"

    print("Loading Titanium Dataset...")
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please check your path.")
        return
        
    ds = load_from_disk(dataset_path)
    print(f"Dataset loaded successfully with {len(ds)} scripts.")

    # 1. Batch iterator for streaming text to the tokenizer trainer
    def batch_iterator():
        for i in range(0, len(ds), 1000):
            yield ds[i : i + 1000]["content"]

    # 2. Base Model Setup
    # ByteLevel ensures the tokenizer can fall back to raw bytes for unknown characters
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # 3. Configure Trainer
    trainer = BpeTrainer(
        vocab_size=8000,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
        initial_alphabet=ByteLevel.alphabet()
    )

    print("\nTraining 8k Byte-Level BPE Tokenizer from scratch...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # 4. Wrap for Hugging Face Transformers compatibility
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]"
    )

    # Save tokenizer artifacts
    hf_tokenizer.save_pretrained(output_dir)
    print(f"Success. Custom tokenizer configurations saved to: {output_dir}")

    # 5. Export Vocabulary for Human Inspection
    print(f"\nExporting entire vocabulary to '{vocab_file_path}' for review...")
    
    # Get vocabulary dictionary and sort it by Token ID
    vocab = hf_tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])

    with open(vocab_file_path, "w", encoding="utf-8") as f:
        f.write(f"=== TOTAL TOKENS REGISTERED: {len(sorted_vocab)} ===\n")
        f.write("Format: Token_ID -> 'Token_String_Representation'\n")
        f.write("=" * 50 + "\n\n")
        
        for token_str, token_id in sorted_vocab:
            # Clean up the byte-level character display to make it readable
            # The 'Ġ' character represents a space in Byte-Level BPE
            readable_str = token_str.replace("Ġ", " [SPACE]")
            readable_str = readable_str.replace("Ċ", " [NEWLINE]")
            f.write(f"{token_id:04d} -> {repr(readable_str)}\n")

    print(f"Inspection file successfully generated! Open '{vocab_file_path}' to audit your vocabulary.")

if __name__ == "__main__":
    train_and_inspect_tokenizer()