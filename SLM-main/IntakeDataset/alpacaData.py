import re
import ast
import io
from datasets import load_dataset, Dataset
from pyflakes.api import check as pyflakes_check
from pyflakes.reporter import Reporter
from langdetect import detect, LangDetectException

# --- TITANIUM FILTERS (from our previous script) ---
def is_computationally_dense(function_node):
    for node in ast.walk(function_node):
        if isinstance(node, ast.Raise):
            if isinstance(node.exc, ast.Name) and node.exc.id == 'NotImplementedError':
                return False
            if isinstance(node.exc, ast.Call) and getattr(node.exc.func, 'id', '') == 'NotImplementedError':
                return False
    for node in ast.walk(function_node):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.If, ast.For, ast.While, 
                             ast.Call, ast.BinOp, ast.Compare, ast.ListComp, ast.DictComp)):
            return True
    return False

def strip_docstring(function_node):
    if function_node.body and isinstance(function_node.body[0], ast.Expr):
        if isinstance(function_node.body[0].value, (ast.Constant, ast.Str)):
            function_node.body.pop(0)

def is_english_variables(function_node):
    identifiers = {function_node.name}
    for node in ast.walk(function_node):
        if isinstance(node, ast.Name): identifiers.add(node.id)
        elif isinstance(node, ast.arg): identifiers.add(node.arg)
    words = [w for w in identifiers if len(w) > 3]
    if not words: return True
    try:
        return detect(" ".join(words).replace("_", " ")) == 'en' if len(" ".join(words)) >= 10 else True
    except LangDetectException:
        return False

def is_code_clean_pyflakes(code_str):
    out_stream, err_stream = io.StringIO(), io.StringIO()
    reporter = Reporter(out_stream, err_stream)
    return pyflakes_check(code_str, "memory_file.py", reporter) == 0

# --- INSTRUCTION EXTRACTOR ---
def extract_pure_code(text):
    """Hunts for python code blocks in the AI's conversational response."""
    # Look for standard markdown python blocks
    pattern = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return "\n".join(matches)
    # If no markdown blocks, assume the entire output is code
    return text

def process_instruction_dataset():
    print("Streaming Instruction Dataset...")
    # Load dataset. No instructions, no inputs, just the output.
    dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    
    titanium_snippets = []
    
    for row in dataset:
        raw_output = row["output"]
        pure_code = extract_pure_code(raw_output)
        
        try:
            tree = ast.parse(pure_code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    node.decorator_list = []
                    if node.args.args and node.args.args[0].arg in ('self', 'cls'):
                        node.args.args.pop(0)
                        
                    if not is_computationally_dense(node): continue
                    strip_docstring(node)
                    if not is_english_variables(node): continue
                    
                    clean_text = ast.unparse(node)
                    
                    if 80 < len(clean_text) < 1200:
                        if is_code_clean_pyflakes(clean_text):
                            titanium_snippets.append(clean_text)
        except Exception:
            continue

    print(f"\nExtracted {len(titanium_snippets)} Pristine Scripts from Instruction Data!")
    
    final_ds = Dataset.from_dict({"content": titanium_snippets})
    final_ds.save_to_disk("./intake1_instruction_titanium")
    print("Saved to ./intake1_instruction_titanium")

if __name__ == "__main__":
    process_instruction_dataset()