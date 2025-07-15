# S2ORC Statistics Paper Parser

This repository contains a robust, memory-efficient pipeline for parsing the large-scale S2ORC (Semantic Scholar Open Research Corpus) dataset. The primary goal of this project is to extract and structure content from academic papers within the statistics domain, creating a high-quality, targeted dataset suitable for training Large Language Models (LLMs).

The pipeline is specifically designed to handle the challenges of the S2ORC dataset, such as its massive file sizes and non-standard JSON Lines format, and to intelligently identify and extract meaningful academic sections.

---

## Project Structure

This project is modularized into a clear, maintainable structure to separate concerns and enhance reusability.

```
.
├── main.py             # The main executable script that orchestrates the entire pipeline.
├── s2orc_parser.py       # Contains the core S2orcParser class for parsing and filtering logic.
├── file_utils.py         # Contains utility functions for file operations (e.g., merging JSON files).
└── requirements.txt    # Lists all necessary Python packages.
```

-   **`s2orc_parser.py`**: This is the heart of the project. It defines the `S2orcParser` class, which is responsible for streaming data from a file, applying regular expressions to identify section headers (e.g., "METHOD", "CONCLUSION"), extracting the corresponding content, and performing data quality audits.
-   **`file_utils.py`**: This module provides helper functions for file I/O, most notably `merge_json_lists`, which combines the parsed output from multiple source files into a single, consolidated dataset.
-   **`main.py`**: This script acts as the orchestrator. It allows you to configure which S2ORC source files to process, defines the keywords for filtering, calls the parser for each file, manages intermediate outputs, and initiates the final merge and audit steps.

---

## Core Features & Technical Details

### 1. Memory-Efficient Streaming Parser
-   **Problem:** S2ORC data files can be extremely large (many gigabytes), making it impossible to load them entirely into memory.
-   **Solution:** The `S2orcParser` reads the source files **line by line** (`JSON Lines` format). This streaming approach ensures that memory consumption remains minimal and constant, regardless of the input file size, allowing the script to run on standard hardware.

### 2. Precise Section Extraction with Regular Expressions
-   **Problem:** The raw text content in S2ORC lacks explicit structure. Sections must be inferred from formatting cues.
-   **Solution:** A sophisticated regular expression is used to identify section headers. It is designed to be robust by looking for specific patterns (e.g., a newline, a mostly-uppercase title of a certain length, followed by two newlines), which helps to distinguish true section titles from other capitalized text, reducing false positives.

### 3. Keyword-Based Filtering
-   The parser filters for papers that contain at least one section title matching a predefined list of keywords (e.g., "method", "analysis", "conclusion"). This ensures that the final dataset is highly relevant to the target domain and contains the desired structured content.

### 4. Data Quality Auditing
-   The pipeline includes an optional final step (`audit_subtitles_by_length`) to identify sections whose extracted content is suspiciously short. This is a crucial data quality check that can help flag parsing errors or papers with poorly formatted content, allowing for manual review and cleaning.

---

## Getting Started

### 1. Prerequisites
-   Python 3.9+
-   All packages listed in `requirements.txt` (`tqdm`).

### 2. Installation
1.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configuration & Usage
1.  **Place Your Data:** Create a directory (e.g., `data/`) and place your raw S2ORC files (with their original long filenames) inside it.

2.  **Configure `main.py`:** Open the `main.py` script and modify the configuration constants at the top:
    -   `RAW_DATA_DIR`: Set this to the path of the directory containing your S2ORC files.
    -   `FILENAMES_TO_PROCESS`: Add the specific, full filenames of the S2ORC files you wish to process to this list.
    -   `SUBTITLE_KEYWORDS`: Customize the list of keywords used to identify relevant papers.
    -   `MERGED_OUTPUT_PATH`: Define the filename for the final, consolidated JSON dataset.
    -   `AUDIT_...`: Configure the parameters for the optional data quality audit.

3.  **Run the Pipeline:** Execute the main script from your terminal.
    ```bash
    python main.py
    ```

The script will process each specified file, save an intermediate parsed file for each in the `s2orc_parts/` directory, merge them into a single final file, and finally generate an audit report.

### 4. Output Format

The final output file (e.g., `S2ORC_parsed_merged.json`) will be a JSON array of objects. Each object represents a paper that met the filtering criteria and has the following structure:
```json
[
    {
        "arxiv_id": "2401.12345",
        "subtitles": {
            "ABSTRACT": "The abstract content goes here...",
            "1 INTRODUCTION": "The introduction content goes here...",
            "2 METHODOLOGY": "The methodology content goes here...",
            "3 EXPERIMENTAL RESULTS": "The results content goes here...",
            "4 CONCLUSION": "The conclusion content goes here..."
        }
    },
    ...
]
```