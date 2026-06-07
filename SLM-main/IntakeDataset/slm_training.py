import os
from transformers import (
    GPT2Config, 
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk

from pathlib import Path

script_dir = Path(__file__).parent.parent  # Go up to SLM directory
dataset_path = "./v7_master_titanium_data"
tokenizer_path = "./v1_8k_code_tokenizer"

def main():

    print("Loading dataset and custom 8k tokenizer...")
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

    # Split into train/validation sets (90% train, 10% val)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.10, seed=42)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    print(f"Dataset Split -> Training samples: {len(train_data)} | Validation samples: {len(val_data)}")

    # 2. Define the 29M Model Configuration
    config = LlamaConfig(
        vocab_size=8000,
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=8,
        num_attention_heads=8,
        max_position_embeddings=512
    )
    model = LlamaForCausalLM(config)
    
    # Calculate exact total parameters to verify our budget
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Model Parameters: {total_params:,}")

    # 3. Data Collator for Causal Language Modeling
    # This automatically clones 'input_ids' to create the 'labels' for next-token prediction
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 4. Configure Training Arguments Optimized for RTX 4060 (8GB VRAM)
    # 4. Configure Training Arguments Optimized for 71k Scale (RTX 4060 / 8GB VRAM)
    training_args = TrainingArguments(
        output_dir="./v2_master_copilot",
        num_train_epochs=3,                 # Lowered from 10 to prevent overfitting the massive dataset
        per_device_train_batch_size=8,      # Kept at 8 to protect your 8GB VRAM
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,      # Yields an effective batch size of 32 (highly stable)
        
        # New: Smooth convergence strategy
        learning_rate=3e-4,                 # Lowered for a more stable descent
        lr_scheduler_type="cosine",         # Glides the learning rate down smoothly
        warmup_ratio=0.03,                  # Gentle ramp-up for the first 3% of training
        weight_decay=0.01,
        
        # New: Checkpoint protection
        save_strategy="steps",              # Save during the epoch, not just at the end
        save_steps=500,                     # Save a checkpoint every ~500 steps
        save_total_limit=3,                 # Delete older checkpoints to save disk space
        eval_strategy="steps",              # Evaluate alongside the saves
        eval_steps=500,
        
        logging_steps=10,                   
        fp16=True,                          # Crucial for consumer GPUs
        load_best_model_at_end=True,        
        metric_for_best_model="loss",
        report_to="none"                    
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

    print(f"\nTraining complete! Saving best model checkpoint to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()