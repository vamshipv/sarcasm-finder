import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from datasets import Dataset

model_name = 'GPT2'

tokenzier = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenzier.pad_token = tokenzier.eos_token
model = GPT2LMHeadModel.from_pretrained(model_name)
model.config.pad_token_id = model.config.eos_token_id


print("Downloading the dataset")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

train_texts = [text for text in dataset["train"]["text"] if len(text.strip()) > 10][:500]
val_texts = [text for text in dataset["validation"]["text"] if len(text.strip()) > 10][:100]

train_data = Dataset.from_dict({"text" : train_texts})
val_data = Dataset.from_dict({"text" : val_texts})

# Convert text to numbers (Tokenization)
def tokenize_funtion(example):
    # Truncate at 128 tokens to keep memory usage tiny
    tokens = tokenzier(example["text"], padding = "max_length", truncation = True, max_length = 128)
    # For causal LM, the labels are just the input IDs. The model shifts them internally to 
    # predict the next word.
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_train = train_data.map(tokenize_funtion, batched=True, remove_columns=["text"])
tokenized_val = val_data.map(tokenize_funtion, batched=True, remove_columns=["text"])


print("3. Setting up Training Arguments (The Overfitting Recipe)...")
training_args = TrainingArguments(
    output_dir="./day3-overfit-test",
    eval_strategy="steps",         # Evaluate frequently so we can watch the crash
    eval_steps=10,                 # Check validation loss every 10 steps
    logging_steps=10,              # Print training loss every 10 steps
    learning_rate=5e-4,            # Aggressive learning rate
    num_train_epochs=10,           # 10 passes over the same 500 lines = guaranteed memorization
    per_device_train_batch_size=4, 
    per_device_eval_batch_size=4,
    save_steps=50,                 # Save checkpoints so you can test them later
    report_to="none"               # Keep output clean in the terminal
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

print("4. Starting the training loop! Watch the terminal output...")
trainer.train()

print("5. Saving the broken model.")
trainer.save_model("./my-overfitted-gpt2")