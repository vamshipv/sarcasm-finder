# sarcasm-finder

Day 1 – Environment + first inference
Objective: Get a small LM talking.
What to do

Set up environment
Shellpip install torch transformers datasets accelerate peftShow more lines

Load a small causal LM:

distilgpt2
or gpt2 (still small enough)


Run inference on:

plain text
technical text
slightly weird prompts



Things to observe (important)

Repetition
Hallucination
Sensitivity to prompt wording
How long it can stay “on topic”

📌 Write down:

3 prompts that work well
3 prompts that completely break it


Day 2 – Tokenization (this matters more than people think)
Objective: Understand how text becomes numbers.
What to do

Load the tokenizer of your model
Tokenize:

normal English
technical terms
code snippets


Inspect:

number of tokens
weird splits
domain-specific words



Example things to try:

"backpropagation"
"self-attention"
"def forward(self, x):"

Key questions to answer

Which words explode into many tokens?
Do technical terms get split badly?
Why might this hurt a small model?

📌 This insight will be critical later when you choose a domain.

Day 3 – First fine-tuning (tiny & safe)
Objective: See training dynamics without pain.
Dataset (keep it simple)
Choose one:

small text dataset from Hugging Face
or your own tiny text file (even ~5–20MB is fine)

This is causal LM fine-tuning, not classification.
What to do

Fine-tune for 1–3 epochs
Small batch size
Log:

training loss
validation loss (if possible)



What to observe

Does loss go down smoothly?
Does it plateau fast?
Does output quality actually improve?

⚠️ Important lesson:

Loss going down ≠ generations getting better

📌 Save generations before vs after training.

Day 4 – Sampling & decoding (huge learning)
Objective: Learn why decoding matters as much as training.
Try different decoding strategies

Greedy
Top-k
Top-p
Temperature changes

Prompts to test

factual
open-ended
technical
ambiguous

What you’ll learn

Why greedy decoding sucks for creativity
Why temperature can destroy coherence
Why small models collapse into repetition

📌 This explains many “bad model” complaints you see online.

Day 5 – Failure analysis (this is the real learning)
Objective: Learn to diagnose, not just train.
Create a simple failure log:

Prompt
Output
Failure type

Failure types to watch for

Repetition loops
Hallucinated facts
Ignoring instructions
Style without substance
Copying prompt too literally

Then answer:

Is this a data problem?
A model size problem?
A tokenization problem?
A decoding problem?

This habit will make you 10× better than “just training models”.