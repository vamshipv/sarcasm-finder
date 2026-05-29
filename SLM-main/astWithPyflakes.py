import ast
import io
from pyflakes.api import check as pyflakes_check
from pyflakes.reporter import Reporter
from datasets import load_from_disk, Dataset
from langdetect import detect, LangDetectException
from pathlib import Path

def is_code_clean_pyflakes(code_str):
    """Runs the dynamic linter to catch semantic garbage and ungrounded variables."""
    out_stream = io.StringIO()
    err_stream = io.StringIO()
    reporter = Reporter(out_stream, err_stream)
    warnings_count = pyflakes_check(code_str, "memory_file.py", reporter)
    return warnings_count == 0

def is_computationally_dense(function_node):
    """Ensures the function actually computes logic (no NotImplemented traps)."""
    for node in ast.walk(function_node):
        if isinstance(node, ast.Raise):
            if isinstance(node.exc, ast.Name) and node.exc.id == 'NotImplementedError':
                return False
            if isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name) and node.exc.func.id == 'NotImplementedError':
                return False

    for node in ast.walk(function_node):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.If, ast.For, ast.While, 
                             ast.Call, ast.BinOp, ast.Compare, ast.ListComp, ast.DictComp)):
            return True
            
    return False

def strip_docstring(function_node):
    """The Docstring Assassin: Removes the docstring node from the AST if it exists."""
    if function_node.body and isinstance(function_node.body[0], ast.Expr):
        if isinstance(function_node.body[0].value, ast.Constant) and isinstance(function_node.body[0].value.value, str):
            function_node.body.pop(0)
        elif isinstance(function_node.body[0].value, ast.Str): # Fallback for older Python versions
            function_node.body.pop(0)

def is_english_variables(function_node):
    """Extracts all variable and function names and checks if they are English."""
    identifiers = set()
    identifiers.add(function_node.name)
    
    for node in ast.walk(function_node):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
            
    # Remove short/meaningless variables (i, j, x, y) before testing language
    meaningful_words = [word for word in identifiers if len(word) > 3]
    
    if not meaningful_words:
        return True # If it's all math (x, y, z), it's universal language. Pass it.
        
    text_to_test = " ".join(meaningful_words).replace("_", " ")
    
    try:
        # Langdetect needs enough text to make a guess
        if len(text_to_test) >= 10:
            return detect(text_to_test) == 'en'
    except LangDetectException:
        return False
        
    return True

def ultimate_titanium_pipeline():
    print("Loading raw minHash dataset...")
    script_dir = Path(__file__).parent.parent  # Go up to SLM directory
    dataset_path = script_dir / "Data Cleaning Scripts" / "dataPipeline" / "minhash_deduped_starCoder_python"
    raw_ds = load_from_disk(dataset_path)
    
    titanium_snippets = []
    pyflakes_killed = 0
    skeleton_killed = 0
    foreign_killed = 0
    
    print("Running Final V5 Filtration (Docstring Stripping + English Enforcing)...")
    for item in raw_ds:
        raw_code = item["content"]
        
        try:
            tree = ast.parse(raw_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    node.decorator_list = []
                    
                    if node.args.args and node.args.args[0].arg in ('self', 'cls'):
                        node.args.args.pop(0)
                        
                    # 1. Kill Skeletons
                    if not is_computationally_dense(node):
                        skeleton_killed += 1
                        continue
                        
                    # 2. Strip Docstrings (Modifies the node in place)
                    strip_docstring(node)
                    
                    # 3. Enforce English Vocabulary
                    if not is_english_variables(node):
                        foreign_killed += 1
                        continue
                    
                    try:
                        clean_text = ast.unparse(node)
                        
                        if 80 < len(clean_text) < 1200:
                            # 4. Final Pyflakes Grounding Check
                            if is_code_clean_pyflakes(clean_text):
                                titanium_snippets.append(clean_text)
                            else:
                                pyflakes_killed += 1
                    except Exception:
                        pass
                        
        except SyntaxError:
            continue

    print("\n--- ULTIMATE FILTRATION REPORT ---")
    print(f"Skeleton/Empty Functions Destroyed: {skeleton_killed}")
    print(f"Pyflakes (Ungrounded Vars) Destroyed: {pyflakes_killed}")
    print(f"Foreign Language / Bad Vocab Destroyed: {foreign_killed}")
    print(f"Total True Titanium Functions Retained: {len(titanium_snippets)}")
    
    final_ds = Dataset.from_dict({"content": titanium_snippets})
    final_ds.save_to_disk("./v6_alpha_data")
    print("Exported successfully to ./v6_alpha_data")

if __name__ == "__main__":
    ultimate_titanium_pipeline()