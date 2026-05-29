import os
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel, 
    PreTrainedTokenizerFast, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk

from pathlib import Path

script_dir = Path(__file__).parent.parent  # Go up to SLM directory
dataset_path = script_dir / "Data Cleaning Scripts" / "dataPipeline" / "minhash_deduped_starCoder_python"
tokenizer_path = script_dir / "SLM-main" / "v1_8k_code_tokenizer"
output_dir = script_dir / "SLM-main" / "v1_29M_coder_model"


def main():

    print("-Loading dataset and custom 8k tokenizer...")
    dataset = load_from_disk(dataset_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # 1. Tokenization Function
    def tokenize_function(examples):
        # We enforce a strict max length of 512 tokens
        return tokenizer(
            examples["content"], 
            truncation=True, 
            max_length=512, 
            padding=False # Collator handles dynamic padding per batch to save VRAM
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["content"],
        desc="Tokenizing progress"
    )

    # Split into train/validation sets (95% train, 5% val)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.05, seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    print(f"Dataset Split -> Training samples: {len(train_data)} | Validation samples: {len(val_data)}")

    # 2. Define the 29M Model Configuration
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=512,
        n_ctx=512,
        n_embd=512,
        n_layer=8,
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    print("Initializing 29M Parameter Model...")
    model = GPT2LMHeadModel(config)
    
    # Calculate exact total parameters to verify our budget
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")

    # 3. Data Collator for Causal Language Modeling
    # This automatically clones 'input_ids' to create the 'labels' for next-token prediction
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Configure Training Arguments Optimized for RTX 4060 (8GB VRAM)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=4,                 # 4 passes over the data is optimal for our small pool
        per_device_train_batch_size=8,      # Batch size 8 is a safe baseline for 8GB VRAM at 512 length
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,      # Simulates a total global batch size of 32 (8 * 4)
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,                 # Slightly higher LR is effective for small models learning fast
        weight_decay=0.01,
        logging_steps=50,
        fp16=True,                          # Uses mixed precision to slash VRAM use and double training speed
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to="none"                    # Prevents mandatory logins to third-party tracking services
    )

    # 5. Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    print("\nStarting Training Loop...")
    trainer.train()

    print(f"\nTraining complete! Saving best model checkpoint to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()