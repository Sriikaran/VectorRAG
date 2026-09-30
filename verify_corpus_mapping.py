import os
import re
import hashlib
import json
import pymupdf

ROOT_DIR = r"c:\Users\srika\OneDrive\Desktop\yesh work"
DATA_DIR = os.path.join(ROOT_DIR, "data")

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def clean_url(url_str):
    # Reference PDFs often contain zero-width spaces (\u200b) or soft hyphens in URLs
    return url_str.replace("\u200b", "").replace("\ufeff", "").replace(" ", "").strip()

def parse_reference_pdf(filepath):
    doc = pymupdf.open(filepath)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    
    # Extract reference number: "SAP Utilities M2C Source Reference (\d+)"
    ref_match = re.search(r"SAP Utilities M2C Source Reference\s*(\d+)", full_text, re.IGNORECASE)
    ref_num = int(ref_match.group(1)) if ref_match else None
    
    # Extract Category: "Category:\s*(.+)"
    cat_match = re.search(r"Category:\s*([^\n\r]+)", full_text)
    category = cat_match.group(1).strip() if cat_match else None
    
    # Extract Title: Usually line after "SAP Utilities M2C Source Reference X"
    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
    title = None
    for i, line in enumerate(lines):
        if "SAP Utilities M2C Source Reference" in line and i + 1 < len(lines):
            title = lines[i + 1]
            break
            
    # Extract What it covers
    covers_match = re.search(r"What it covers\s*([\s\S]*?)(?=Meter-to-Cash\s*relevance|Authoritative SAP Help source|$)", full_text)
    what_it_covers = " ".join(covers_match.group(1).split()) if covers_match else ""
    
    # Extract Meter-to-Cash relevance
    m2c_match = re.search(r"Meter-to-Cash\s*relevance\s*([\s\S]*?)(?=Authoritative SAP Help source|Library use|$)", full_text)
    m2c_relevance = " ".join(m2c_match.group(1).split()) if m2c_match else ""
    
    # Extract Authoritative SAP Help source URL
    url_match = re.search(r"(https?://[^\s]+)", full_text)
    sap_url = clean_url(url_match.group(1)) if url_match else ""
    
    # Extract source / authority info
    auth_info = "SAP Help Portal (Official Documentation)"
    
    return {
        "ref_num": ref_num,
        "title": title,
        "category": category,
        "what_it_covers": what_it_covers,
        "m2c_relevance": m2c_relevance,
        "source_url": sap_url,
        "authority": auth_info,
        "raw_text": full_text.strip()
    }

def main():
    root_files = [f for f in sorted(os.listdir(ROOT_DIR)) if f.lower().endswith(".pdf")]
    data_files = [f for f in sorted(os.listdir(DATA_DIR)) if f.lower().endswith(".pdf")]
    
    print(f"Root Reference PDFs detected: {len(root_files)}")
    print(f"Data PDFs detected: {len(data_files)}")
    
    ref_metadata = {}
    for rf in root_files:
        path = os.path.join(ROOT_DIR, rf)
        parsed = parse_reference_pdf(path)
        ref_metadata[rf] = parsed
        
    print("\n--- ROOT REFERENCE FILES ---")
    for rf, meta in ref_metadata.items():
        print(f"[{meta['ref_num']:02d}] {rf} -> Title: {meta['title']} | Category: {meta['category']} | URL: {meta['source_url'][:60]}...")
        
    print("\n--- DATA PDFs INSPECTION ---")
    data_metadata = {}
    hashes = {}
    duplicates = []
    
    for df in data_files:
        path = os.path.join(DATA_DIR, df)
        h = sha256_file(path)
        size = os.path.getsize(path)
        
        if h in hashes:
            duplicates.append((df, hashes[h]))
        else:
            hashes[h] = df
            
        doc = pymupdf.open(path)
        page_count = len(doc)
        first_page_text = doc[0].get_text()[:300].replace("\n", " ").strip() if page_count > 0 else ""
        
        # Match number from filename (e.g., "1.pdf" -> 1)
        num_match = re.match(r"^(\d+)\.pdf$", df, re.IGNORECASE)
        data_num = int(num_match.group(1)) if num_match else None
        
        data_metadata[df] = {
            "data_num": data_num,
            "filename": df,
            "sha256": h,
            "size_bytes": size,
            "page_count": page_count,
            "sample_text": first_page_text
        }
        print(f"File: {df:<8} | Size: {size/1024/1024:6.2f} MB | Pages: {page_count:<4} | Sample: {first_page_text[:70]}...")

    print(f"\nDuplicates check: {len(duplicates)} duplicates found: {duplicates}")
    
    # Mapping verification
    print("\n--- MAPPING VERIFICATION ---")
    mapping = {}
    unmatched_refs = []
    unmatched_data = []
    
    for rf, r_meta in ref_metadata.items():
        r_num = r_meta["ref_num"]
        matched_df = f"{r_num}.pdf"
        if matched_df in data_metadata:
            mapping[r_num] = {
                "ref_file": rf,
                "data_file": matched_df,
                "title": r_meta["title"],
                "category": r_meta["category"],
                "url": r_meta["source_url"],
                "pages": data_metadata[matched_df]["page_count"],
                "size_bytes": data_metadata[matched_df]["size_bytes"],
                "sha256": data_metadata[matched_df]["sha256"]
            }
        else:
            unmatched_refs.append(rf)
            
    for df, d_meta in data_metadata.items():
        if d_meta["data_num"] not in mapping:
            unmatched_data.append(df)
            
    print(f"Successfully matched: {len(mapping)} / 29 pairs.")
    print(f"Unmatched reference files: {unmatched_refs}")
    print(f"Unmatched data files: {unmatched_data}")
    
    with open("corpus_inventory.json", "w", encoding="utf-8") as f:
        json.dump({
            "root_reference_count": len(root_files),
            "data_pdf_count": len(data_files),
            "duplicates": duplicates,
            "unmatched_refs": unmatched_refs,
            "unmatched_data": unmatched_data,
            "mapping": mapping
        }, f, indent=2)
    print("Wrote corpus_inventory.json")

if __name__ == "__main__":
    main()
