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
dataset_path = script_dir / "SLM-main" / "IntakeDataset" / "v7_master_titanium_data"
tokenizer_path = script_dir / "SLM-main" / "v1_8k_code_tokenizer"

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
    training_args = TrainingArguments(
        output_dir="./v2_alpha_copilot",
        num_train_epochs=10,                # Cranked up to cement the syntax rhythm
        per_device_train_batch_size=8,      
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,      
        eval_strategy="epoch",        # Evaluate every epoch (since they are fast)
        save_strategy="epoch",              # Save a checkpoint every epoch
        learning_rate=5e-4,                 
        weight_decay=0.01,
        logging_steps=10,                   # Lowered so you get more frequent terminal updates
        fp16=True,                          
        load_best_model_at_end=True,        # Automatically retrieves the smartest epoch
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