import os
import time
from concurrent.futures import ProcessPoolExecutor
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions

def init_converter():
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    opts.ocr_options.scale = 1.5
    opts.do_table_structure = True
    opts.accelerator_options = AcceleratorOptions(num_threads=2, device="cpu")
    return DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opts)})

def parse_worker(pdf_path):
    conv = init_converter()
    t0 = time.time()
    res = conv.convert(pdf_path)
    return {
        "path": pdf_path,
        "elapsed": time.time() - t0,
        "pages": len(res.document.pages),
        "texts": len(res.document.texts)
    }

if __name__ == "__main__":
    test_files = [
        r"c:\Users\srika\OneDrive\Desktop\yesh work\data\8.pdf",
        r"c:\Users\srika\OneDrive\Desktop\yesh work\data\10.pdf"
    ]
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(parse_worker, test_files))
    print(f"Parallel test done in {time.time()-t_start:.2f}s:")
    for r in results:
        print(r)
