import ast
import os
from datasets import load_from_disk, Dataset
from pathlib import Path

class FunctionExtractor(ast.NodeVisitor):
    def __init__(self):
        self.extracted_functions = []

    def visit_FunctionDef(self, node):
        # 1. Strip decorators (we don't want @classmethod or @staticmethod noise)
        node.decorator_list = []
        
        # 2. Strip 'self' or 'cls' from the arguments if it was inside a class
        if node.args.args and node.args.args[0].arg in ('self', 'cls'):
            node.args.args.pop(0)

        # 3. Unparse the AST node back into perfect, standardized text
        try:
            # ast.unparse requires Python 3.9+
            clean_code = ast.unparse(node)
            
            # 4. Filter out tiny meaningless functions or massive ones
            if 50 < len(clean_code) < 1500:
                self.extracted_functions.append(clean_code)
        except Exception:
            pass # Ignore unparse errors from heavily corrupted ASTs

        # Continue traversing in case there are nested functions
        self.generic_visit(node)


def process_dataset_with_ast():
    print("Loading original Titanium Dataset...")
    script_dir = Path(__file__).parent.parent  # Go up to SLM directory
    dataset_path = script_dir / "Data Cleaning Scripts" / "dataPipeline" / "minhash_deduped_starCoder_python"
    ds = load_from_disk(dataset_path)
    
    clean_snippets = []
    
    print("Running AST Scope Extractor over 15,000 files...")
    for item in ds:
        raw_code = item["content"]
        try:
            # Parse the raw text into a structural syntax tree
            tree = ast.parse(raw_code)
            
            # Run our extractor over the tree
            extractor = FunctionExtractor()
            extractor.visit(tree)
            
            # Add all successfully extracted, standalone functions to our new pool
            clean_snippets.extend(extractor.extracted_functions)
        except SyntaxError:
            # If it's structurally broken, we completely ignore it
            continue

    print(f"Extraction Complete! Yielded {len(clean_snippets)} isolated functions.")
    
    # Save the new micro-copilot dataset
    new_ds = Dataset.from_dict({"content": clean_snippets})
    new_ds.save_to_disk("./v2_ast_isolated_functions")
    print("Saved to ./v2_ast_isolated_functions")

if __name__ == "__main__":
    process_dataset_with_ast()