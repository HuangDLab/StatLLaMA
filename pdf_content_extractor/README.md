# PDF Content Extractor

This repository contains a robust and flexible command-line tool for extracting clean, high-quality text content from PDF files. The project is designed to handle a variety of PDF types, from standard digital documents to complex, scanned, or image-based files.

It offers two distinct extraction strategies to tackle different challenges:
1.  **Text-based Extraction:** A fast method ideal for digitally-native PDFs where text is selectable. It uses `pdfplumber`.
2.  **OCR-based Extraction:** A powerful, more intensive method that treats each page as an image. It uses `PyMuPDF` to render pages and `Pix2Text` to perform Optical Character Recognition (OCR), making it suitable for scanned documents or PDFs with complex layouts.

The pipeline also includes a sophisticated text-cleaning module to filter out common artifacts like headers, footers, and reference sections, ensuring the final output is ready for downstream NLP tasks.

---

## Project Structure

This project is organized into two core files for simplicity and clarity.

```
.
├── main.py             # The main executable script with a command-line interface (CLI).
├── pdf_processor.py      # Contains all core logic for PDF extraction and text cleaning.
└── requirements.txt    # Lists all necessary Python packages.
```

-   **`main.py`**: The user-facing entry point. It handles command-line arguments (e.g., source folder, extraction method, page range) and orchestrates the entire processing workflow by calling functions from `pdf_processor.py`.
-   **`pdf_processor.py`**: The workhorse of the project. This single module contains all the heavy lifting:
    -   Functions for both direct text extraction (`process_pdf_with_text`) and OCR-based extraction (`process_pdf_with_ocr`).
    -   A helper function (`_clean_and_filter_text`) that performs all post-processing and cleaning on the raw extracted text.

---

## Core Features & Technical Details

### 1. Dual Extraction Methods
The tool's key feature is its flexibility. Users can choose the best extraction method for their specific needs via the `--method` flag:
-   `--method text`: Fast and efficient. Best for large batches of clean, digitally-created PDFs.
-   `--method ocr`: Slower but more powerful. Essential for scanned documents, PDFs with complex multi-column layouts, or files where direct text extraction fails.

### 2. Advanced Text Cleaning (`pdf_processor.py`)
-   **Reference Section Removal:** Intelligently identifies and removes the "References" or "Bibliography" section from the end of the document, which is often irrelevant for content-focused NLP tasks.
-   **Header/Footer Filtering:** Removes common, repetitive page headers and footers to reduce noise.
-   **Formatting Normalization:** Consolidates excessive newlines and spaces, and attempts to intelligently rejoin words that were hyphenated across line breaks, improving text readability and coherence.

### 3. Configurable Processing
The command-line interface allows for fine-grained control over the extraction process, including:
-   Specifying a page range (`--start_page`, `--end_page`) for each document.
-   Adjusting the image resolution (`--dpi`) for the OCR method to balance speed and accuracy.

---

## Getting Started

### 1. Prerequisites
-   Python 3.9+
-   PyTorch (required by `pix2text`). A CUDA-enabled environment is highly recommended for OCR performance.
-   All packages listed in `requirements.txt`.

### 2. Installation
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. Usage Examples

The tool is run from the command line, specifying the source folder of PDFs and the desired output file.

**Example 1: Fast text extraction from all PDFs in a folder.**
```bash
python main.py ./my_digital_pdfs/ output_text.json --method text
```

**Example 2: OCR-based extraction for scanned documents.**
```bash
python main.py ./scanned_books/ output_ocr.json --method ocr
```

**Example 3: Extracting only pages 10 through 50 from each PDF using OCR with higher resolution.**
```bash
python main.py ./specific_chapters/ chapter_content.json --method ocr --start_page 10 --end_page 50 --dpi 300
```

For a full list of options and their descriptions, run:
```bash
python main.py --help
```

### 4. Output Format

The output is a single JSON file containing a list of strings. Each string in the list corresponds to the full, cleaned text content extracted from one PDF file in the source folder.
```json
[
  "The entire cleaned text content of the first PDF file goes here...",
  "The entire cleaned text content of the second PDF file goes here...",
  ...
]
```