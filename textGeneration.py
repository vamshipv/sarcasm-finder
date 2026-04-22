from causalLm import tokenzier, model

prompt = "Explain gradient descent in machine learning. Also make sure to explain why it is important. Continue explaining in more detail."
input = tokenzier(prompt, return_tensors = "pt")

outputs = model.generate(
    **input,
    max_new_tokens = 500,
    # max_new_tokens (int, optional) The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    # Tokens is not words, 50 Tokens ~ 35-40 English words
    num_return_sequences = 1,
    # num_return_sequences (int, optional) — The number of independently computed returned sequences for each element in the batch.
    no_repeat_ngram_size = 1,
    # Prevents the model from repeating any n-gram of size n, The model cannot repeat the same 2-word sequence.
)

print(tokenzier.decode(outputs[0], skip_special_tokens = True))


