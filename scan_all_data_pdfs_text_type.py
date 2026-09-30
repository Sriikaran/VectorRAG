import os
import pymupdf

DATA_DIR = r"c:\Users\srika\OneDrive\Desktop\yesh work\data"

pdf_files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")],
    key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
)

print(f"{'PDF':<8} | {'Pages':<6} | {'Total Chars':<12} | {'Avg Chars/Page':<16} | {'Images/Page':<12} | {'Type'}")
print("-" * 75)

scanned_count = 0
digital_count = 0

for pf in pdf_files:
    path = os.path.join(DATA_DIR, pf)
    doc = pymupdf.open(path)
    total_chars = 0
    total_images = 0
    for page in doc:
        txt = page.get_text()
        total_chars += len(txt.strip())
        total_images += len(page.get_images())
    avg_chars = total_chars / len(doc) if len(doc) > 0 else 0
    avg_imgs = total_images / len(doc) if len(doc) > 0 else 0
    
    if avg_chars < 50:
        doc_type = "SCANNED/IMAGE (Needs OCR)"
        scanned_count += 1
    else:
        doc_type = "DIGITAL TEXT"
        digital_count += 1
        
    print(f"{pf:<8} | {len(doc):<6} | {total_chars:<12} | {avg_chars:<16.1f} | {avg_imgs:<12.1f} | {doc_type}")

print("-" * 75)
print(f"Summary: {digital_count} Digital Text PDFs, {scanned_count} Scanned/Image PDFs.")
