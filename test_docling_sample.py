import json
from docling.document_converter import DocumentConverter

def main():
    converter = DocumentConverter()
    result = converter.convert(r"c:\Users\srika\OneDrive\Desktop\yesh work\data\8.pdf")
    doc = result.document
    print("Docling conversion successful!")
    print(f"Pages: {len(doc.pages)}")
    print(f"Num items: {len(doc.texts) if hasattr(doc, 'texts') else 'N/A'}")
    
    # Export to markdown or inspect dict
    md = doc.export_to_markdown()
    print("Markdown preview (first 500 chars):")
    print(md[:500])
    
    # Test iterating through body items / elements
    print("\nSample elements:")
    for i, item in enumerate(doc.iterate_items()):
        if i < 10:
            print(f"[{i}] {type(item[0]).__name__} | label: {getattr(item[0], 'label', 'N/A')} | text: {getattr(item[0], 'text', '')[:60]}")

if __name__ == "__main__":
    main()
