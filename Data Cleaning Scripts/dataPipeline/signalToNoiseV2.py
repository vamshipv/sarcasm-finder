import os
from datasets import load_from_disk, Dataset

# --- CONFIGURATION ---
MIN_LOGIC_KEYWORDS = 5
MIN_CODE_CHARS = 150
MAX_CODE_CHARS = 5000  # Relaxed slightly from 2500 for better variety
LOGIC_KEYWORDS = ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ', 'return ', 'raise ']
LICENSE_FLAGS = ['copyright', 'license', 'http', 'author', 'distributed under', 'apache']

def janitor_v5_modern(example):
    content = example.get('content', "")
    if not content or len(content) < MIN_CODE_CHARS:
        return {"keep": False}

    lines = content.splitlines()
    first_meaningful_idx = -1
    
    # 1. FIND THE START (Skip # comments but KEEP """ docstrings)
    for i, line in enumerate(lines):
        clean = line.strip()
        if clean and not clean.startswith('#'):
            first_meaningful_idx = i
            break
    
    if first_meaningful_idx == -1:
        return {"keep": False}

    # 2. SHAVE THE NOISE
    header_text = "\n".join(lines[:first_meaningful_idx]).lower()
    is_license = any(flag in header_text for flag in LICENSE_FLAGS)
    
    # If it's a license, cut it. Otherwise, keep the whole thing.
    shaved_content = "\n".join(lines[first_meaningful_idx:]) if is_license else content

    # 3. LOGIC DENSITY CHECK
    # Does this script actually 'think' or is it just a list of variables?
    found_keywords = [kw for kw in LOGIC_KEYWORDS if kw in shaved_content]
    
    # 4. RATIO CHECK (Signal vs Noise)
    # If more than 50% of the lines are comments, it might be a README disguised as code
    comment_lines = sum(1 for l in shaved_content.splitlines() if l.strip().startswith('#'))
    total_lines = len(shaved_content.splitlines())
    
    low_noise = (comment_lines / total_lines) < 0.5 if total_lines > 0 else False

    # VERDICT
    keep = (
        len(found_keywords) >= MIN_LOGIC_KEYWORDS and 
        len(shaved_content) <= MAX_CODE_CHARS and
        low_noise
    )

    return {"content": shaved_content, "keep": keep}

# --- EXECUTION ---
print("Stage 1: Running the Signal-to-Noise Janitor...")

ds = load_from_disk("./raw_scripts_collection")
processed_ds = ds.map(janitor_v5_modern)
final_ds = processed_ds.filter(lambda x: x['keep'])

print(f"Raw: {len(ds)} | Logic-Dense: {len(final_ds)}")
final_ds.save_to_disk("./starCoder_cleaned")