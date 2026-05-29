import ast
import os
from datasets import load_from_disk, Dataset
from pathlib import Path

class SemanticVariableAuditor(ast.NodeVisitor):
    """Audits a function to ensure ALL variables used are strictly grounded and defined."""
    def __init__(self, function_node):
        self.is_valid = True
        self.defined_vars = set()
        
        # 1. Register all function arguments as defined variables
        for arg in function_node.args.args:
            self.defined_vars.add(arg.arg)
        for arg in function_node.args.kwonlyargs:
            self.defined_vars.add(arg.arg)
        if function_node.args.vararg:
            self.defined_vars.add(function_node.args.vararg.arg)
        if function_node.args.kwarg:
            self.defined_vars.add(function_node.args.kwarg.arg)
            
    def visit_Assign(self, node):
        # Register variables created inside the function body via assignment (e.g., x = 1)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined_vars.add(target.id)
            elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.defined_vars.add(elt.id)
        self.generic_visit(node)

    def visit_For(self, node):
        # Register loop variables (e.g., for i in range)
        if isinstance(node.target, ast.Name):
            self.defined_vars.add(node.target.id)
        self.generic_visit(node)

    def visit_Name(self, node):
        # Check if a variable name being read is actually defined anywhere
        if isinstance(node.ctx, ast.Load):
            # Ignore standard Python built-in words or common module markers
            if node.id in ('print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple', 'enumerate', 'sum', 'open', 'True', 'False', 'None', 'Exception', 'ValueError', 'TypeError'):
                return
            
            # If the variable name is not in our defined set, the code is ungrounded garbage
            if node.id not in self.defined_vars:
                self.is_valid = False
        self.generic_visit(node)


def sanitize_content_pipeline():
    print("Loading raw minHash dataset...")
    script_dir = Path(__file__).parent.parent  # Go up to SLM directory
    dataset_path = script_dir / "Data Cleaning Scripts" / "dataPipeline" / "minhash_deduped_starCoder_python"
    raw_ds = load_from_disk(dataset_path)
    
    titanium_snippets = []
    ghost_vars_killed = 0
    obscure_domain_killed = 0
    
    print("Beginning Content Sense Check and Semantic Cleansing...")
    for item in raw_ds:
        raw_code = item["content"]
        
        # Immediate content-level red flags to optimize speed
        # If the script mentions niche biology or obscure custom wrappers, skip it immediately
        if any(word in raw_code for word in ["PDBParser", "PDBIO", "FastText", "Genome", "genomes"]):
            obscure_domain_killed += 1
            continue
            
        try:
            tree = ast.parse(raw_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Strip out decorators
                    node.decorator_list = []
                    
                    # Strip self/cls out of parameters if they exist
                    if node.args.args and node.args.args[0].arg in ('self', 'cls'):
                        node.args.args.pop(0)
                        
                    # Run our semantic auditor over this isolated function node
                    auditor = SemanticVariableAuditor(node)
                    auditor.visit(node)
                    
                    if auditor.is_valid:
                        try:
                            clean_text = ast.unparse(node)
                            # Keep only reasonably sized utility logic blocks
                            if 80 < len(clean_text) < 1200:
                                titanium_snippets.append(clean_text)
                        except Exception:
                            pass
                    else:
                        ghost_vars_killed += 1
                        
        except SyntaxError:
            continue

    print("\n--- CONTENT FILTRATION REPORT ---")
    print(f"Obscure Domain Blocks Incinerated: {obscure_domain_killed}")
    print(f"Ghost Variable/Ungrounded Snippets Blocked: {ghost_vars_killed}")
    print(f"Total True Titanium Functions Retained: {len(titanium_snippets)}")
    
    final_ds = Dataset.from_dict({"content": titanium_snippets})
    final_ds.save_to_disk("./v3_true_titanium_code")
    print("Exported successfully to ./v3_true_titanium_code")

if __name__ == "__main__":
    sanitize_content_pipeline()