import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = 'GPT2'

tokenzier = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

model.eval()