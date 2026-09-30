# Tesseract language data (vendored, offline)

`eng.traineddata` and `osd.traineddata` are required by the Part 2 OCR layer
(`ocr_pipeline.py`, via tesserocr). The environment has no system Tesseract
and no GitHub access, so these files were vendored from the PyPI wheel
`tesseract_ocr_data 1.6` (package `tesseract-ocr-data`), which ships the
tessdata_fast English + orientation models for Tesseract 4/5.

- License: Apache-2.0 (see https://github.com/tesseract-ocr/tessdata_fast)
- Used with: tesserocr 2.11.0 (bundled Tesseract 5.5.1, LSTM)
- Path referenced by: `ocr_pipeline.py` (`TESSDATA_DIR = ./tessdata`)
