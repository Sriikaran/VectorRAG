import os
import pymupdf

root_dir = r"c:\Users\srika\OneDrive\Desktop\yesh work"
ref_files = [f for f in sorted(os.listdir(root_dir)) if f.endswith(".pdf")]

print(f"Found {len(ref_files)} reference files in root directory.")

for rf in ref_files:
    path = os.path.join(root_dir, rf)
    doc = pymupdf.open(path)
    print(f"=== {rf} (pages: {len(doc)}) ===")
    text = ""
    for page in doc:
        text += page.get_text()
    print(text.strip())
    print("-" * 50)
