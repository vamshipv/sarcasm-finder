import ast
import io
import os
from datasets import load_dataset, Dataset
from pyflakes.api import check as pyflakes_check
from pyflakes.reporter import Reporter
from langdetect import detect, LangDetectException

# --- THE TITANIUM MEAT-GRINDER FILTERS ---
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

# --- THE STREAMING SCALE ENGINE ---
def stream_and_mine_code(target_needed=65000):
    print(f"Opening secure Parquet stream to BigCode/StarCoderData...")
    
    # Using modern Parquet streaming. No remote scripts allowed!
    remote_stream = load_dataset(
        "bigcode/starcoderdata", 
        data_dir="python",      # Specifically target the Python subset
        split="train", 
        streaming=True,
        trust_remote_code=False # Enforce strict security
    )
    
    titanium_snippets = []
    checked_count = 0
    
    print(f"Mining active. Hunting for {target_needed} perfect functions. Please wait...")
    
    for file_data in remote_stream:
        checked_count += 1
        
        # StarCoder uses 'content' instead of 'code'
        raw_code = file_data.get("content", "") 
        if not raw_code:
            continue
            
        try:
            tree = ast.parse(raw_code)
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
                            
                            # Terminal updates every 500 successful extractions
                            if len(titanium_snippets) % 500 == 0:
                                print(f"-> Progress: Gathered {len(titanium_snippets)} / {target_needed} functions (Scanned {checked_count} files)")
                                
                            if len(titanium_snippets) >= target_needed:
                                raise StopIteration # Break out of the entire stream loop cleanly
                                
        except StopIteration:
            break
        except Exception:
            continue

    print(f"\nTarget achieved! Successfully mined {len(titanium_snippets)} pristine functions.")
    
    final_ds = Dataset.from_dict({"content": titanium_snippets})
    final_ds.save_to_disk("./intake2_starcoder_titanium")
    print("Saved to ./intake2_starcoder_titanium")

if __name__ == "__main__":
    # Feel free to adjust the target_needed down if you want to run a smaller test first
    stream_and_mine_code(target_needed=65000)