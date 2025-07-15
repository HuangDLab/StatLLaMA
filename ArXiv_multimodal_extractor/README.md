# ArXiv Statistics Paper Parser & Content Extractor

This repository contains a sophisticated, multi-stage pipeline designed to crawl, parse, and extract structured content from academic papers in the statistics domain from arXiv. The primary goal of this project is to build a high-quality, structured dataset suitable for training Large Language Models (LLMs) on specialized academic content.

The pipeline automates several complex tasks:
1.  **Crawling:** Fetches paper metadata from arXiv's advanced search.
2.  **PDF Parsing:** Downloads and parses PDF files to extract text and identify section structures.
3.  **Image-to-Text (Multimodal):** Extracts figures and tables as images and uses a BLIP model to generate descriptive captions.
4.  **Content Structuring:** Organizes extracted text into key academic sections (e.g., `abstract`, `method`, `result`, `conclusion`).
5.  **Deduplication:** Intelligently removes redundant content using string similarity algorithms.

---

## Project Structure

This project is modularized for clarity, maintainability, and reusability.

```
.
├── main.py             # The main executable script to run the entire pipeline.
├── arxiv_crawler.py      # Contains the ArxivCrawler class for fetching paper metadata.
├── paper_parser.py       # Contains the ArxivPaperParser class for downloading and processing PDFs.
└── requirements.txt    # Lists all necessary Python packages.
```

-   **`arxiv_crawler.py`**: Handles all interactions with the arXiv website. It is responsible for paginating through search results and extracting metadata like title, authors, abstract, and the arXiv ID.
-   **`paper_parser.py`**: The core of the project. This module takes an arXiv ID, downloads the corresponding PDF, and performs all heavy lifting: text extraction with `PyMuPDF`, section boundary detection, image extraction, batch caption generation with the BLIP model, and content deduplication with `Levenshtein` distance.
-   **`main.py`**: The orchestrator. It initializes the crawler and parser, feeds the list of papers from the crawler to the parser, collects the structured results, and saves the final, cleaned dataset to a JSON file.

---

## Core Features & Technical Details

### 1. Robust Web Crawling (`arxiv_crawler.py`)
-   **Polite Crawling:** Implements `time.sleep()` delays between requests to avoid overwhelming arXiv's servers.
-   **Resilient Error Handling:** Gracefully handles HTTP errors and network issues, logging warnings and continuing the crawl process where possible.
-   **Flexible Pagination:** Allows users to specify a range of pages to crawl, making it easy to run small tests or large-scale data collection jobs.

### 2. Advanced PDF Parsing & Structuring (`paper_parser.py`)
-   **Section Boundary Detection:** Instead of relying on simple keywords, the parser first identifies the positions of all major section headers (e.g., "1. Introduction", "2. Method") within the document. It then defines a section's content as the text between its header and the start of the next one, making the extraction robust to complex sub-sectioning.
-   **Intelligent Deduplication:** Uses `Levenshtein` distance to calculate string similarity. This is applied to:
    1.  Remove sections that are nearly identical to the abstract.
    2.  Remove sections (e.g., `conclusion`) that are just a repetition of a previous section (e.g., `result`).
-   **Efficient Image Captioning (Batch Processing):** To maximize GPU utilization, the parser first extracts all images from the PDF and then sends them to the BLIP model as a single **batch**. This is significantly faster than processing images one by one.
-   **Automated Cleanup:** Utilizes Python's `tempfile` module to handle downloaded PDFs. This ensures that temporary files are **automatically deleted** after processing, even if an error occurs, keeping the workspace clean.

---

## Getting Started

### 1. Prerequisites
-   Python 3.9+
-   PyTorch with CUDA support (for GPU acceleration of the BLIP model).
-   All packages listed in `requirements.txt`.

### 2. Installation
1.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration & Usage
1.  **Configure the Crawl:** Open `main.py` and adjust the following constants to fit your needs:
    -   `BASE_URL`: You can modify the arXiv advanced search query here.
    -   `MIN_PAGES_TO_CRAWL`: The starting page number (0-indexed).
    -   `MAX_PAGES_TO_CRAWL`: The page number to stop at (exclusive). For a quick test, a range of 1 or 2 pages is recommended.
    -   `OUTPUT_FILENAME`: The name of the final JSON output file.

2.  **Run the Pipeline:** Execute the main script from your terminal.
    ```bash
    python main.py
    ```

The script will log its progress, including the current page being crawled and the papers being processed. The final, structured JSON data will be saved to the specified output file upon completion.

### 4. Output Format

The output JSON file will be a list of dictionaries. Each dictionary represents a successfully processed paper and has the following structure:
```json
[
    {
        "title": "A Statistical Model for...",
        "authors": ["Author A", "Author B"],
        "abstract": "In this paper, we propose a novel method...",
        "arxiv_id": "2401.12345",
        "method": "The methodology consists of three steps... [Figure Captions] [Figure on page 3]: A diagram of our proposed network architecture...",
        "result": "Our experimental results show a significant improvement... [Figure Captions] [Figure on page 5]: A bar chart comparing the performance...",
        "conclusion": "In conclusion, our method provides a new state-of-the-art..."
    },
    ...
]
```
