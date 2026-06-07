from transformers import PreTrainedTokenizerFast
from pathlib import Path

script_dir = Path(__file__).parent
tokenizer_path = script_dir / "v1_8k_code_tokenizer"  # Make sure this path is correct
if not tokenizer_path.exists():
    raise FileNotFoundError(
        f"Tokenizer directory not found: {tokenizer_path}\n" \
        "Put the tokenizer folder in the same directory as stopwordNumber.py, or update tokenizer_path to the correct folder."
    )

tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# Let's ask your tokenizer how it translates a double newline
double_newline_id = tokenizer.encode("\n\n")
print(f"Token IDs for '\\n\\n': {double_newline_id}")

# Let's also check a new function definition
def_start_id = tokenizer.encode("\ndef ")
print(f"Token IDs for '\\ndef ': {def_start_id}")