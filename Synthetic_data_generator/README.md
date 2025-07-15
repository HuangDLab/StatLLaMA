# Gemini-Powered Statistical Data Generator

This repository contains a powerful and modular command-line tool for generating high-quality, synthetic training data for statistical language models using Google's Gemini API. The project is designed to be flexible, allowing users to generate various types of data formats, including Direct Preference Optimization (DPO) pairs, "fill-in-the-blank" (cloze) questions, and multi-turn conversations.

A key feature of this tool is its integrated **semantic deduplication** pipeline, which uses sentence transformers to ensure the diversity and quality of the generated dataset.

---

## Project Structure

This project is organized into a clean, modular structure to promote code clarity, maintainability, and ease of extension.

```
.
├── main.py             # The main executable script with a command-line interface (CLI).
├── gemini_utils.py       # Core utilities for API calls, JSON parsing, and semantic deduplication.
├── prompt_factory.py     # A "factory" for constructing sophisticated prompts for different generation tasks.
└── requirements.txt    # Lists all necessary Python packages.
```

-   **`main.py`**: This is the user-facing entry point. It handles command-line arguments, orchestrates the workflow, and calls functions from the other modules.
-   **`gemini_utils.py`**: Contains the backbone of the project. It abstracts away the complexities of interacting with the Gemini API, safely parsing potentially malformed JSON responses, and performing the computationally intensive task of semantic deduplication.
-   **`prompt_factory.py`**: This module is dedicated to **prompt engineering**. It contains functions that dynamically generate detailed, context-rich prompts based on the user's desired task, ensuring the highest quality output from the Gemini model.

---

## Core Features & Technical Details

### 1. Multi-Task Data Generation
The tool supports three distinct data generation tasks, each activated by a simple command-line argument:
-   **`dpo`**: Generates preference pairs (`prompt`, `chosen`, `rejected`) for Direct Preference Optimization.
-   **`cloze`**: Generates "fill-in-the-blank" style questions (`question_text`, `answer`).
-   **`conversation`**: Generates realistic, multi-turn dialogues between a "user" and a "statistical assistant".

### 2. Sophisticated Prompt Engineering (`prompt_factory.py`)
Each prompt is carefully engineered to provide the Gemini model with clear instructions, examples, and constraints. The factory can generate different "flavors" of prompts (e.g., `simple` vs. `advanced` DPO) via command-line flags for fine-grained control.

### 3. Robust JSON Parsing & Cleaning (`gemini_utils.py`)
-   Handles common API response issues, such as markdown code fences (e.g., `` ```json ``).
-   Validates the structure of each generated item, ensuring it contains all expected keys.

### 4. Semantic Deduplication (`gemini_utils.py`)
-   An optional `--deduplicate` flag activates a powerful post-processing step.
-   It uses the `all-MiniLM-L6-v2` sentence-transformer model to compute vector embeddings for the key text field of each generated item.
-   It removes items with a cosine similarity score above a user-defined threshold, ensuring the final dataset is semantically diverse.

---

## Getting Started

### 1. Prerequisites
-   Python 3.9+
-   An active Google Gemini API key.
-   All packages listed in `requirements.txt`.

### 2. Installation & Configuration
1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Configure API Key:**
    This script requires your Google Gemini API key to be set as an environment variable. This is the most secure way to handle credentials and avoids hardcoding them in the script.

    **On Linux/macOS:**
    ```bash
    export GOOGLE_API_KEY="AIzaSy...your...key..."
    ```
    *Note: This variable will only be set for the current terminal session. To make it permanent, add the line above to your shell's configuration file (e.g., `~/.bashrc` or `~/.zshrc`) and restart your terminal.*

    **On Windows (Command Prompt):**
    ```cmd
    set GOOGLE_API_KEY="AIzaSy...your...key..."
    ```
    **On Windows (PowerShell):**
    ```powershell
    $env:GOOGLE_API_KEY="AIzaSy...your...key..."
    ```
    The script will automatically detect and use this environment variable.

### 3. Usage Examples

The tool is run from the command line. The basic structure is `python main.py [task] --output_file [filename] [options]`.

**Example 1: Generate 20 advanced DPO pairs with deduplication.**
```bash
python main.py dpo --num_items 20 --output_file advanced_dpo_data.json --deduplicate
```

**Example 2: Generate 100 simple, foundational DPO pairs.**
```bash
python main.py dpo --num_items 100 --output_file simple_dpo_data.json --simple
```

**Example 3: Generate 50 high-difficulty "cloze" questions.**
```bash
python main.py cloze --num_items 50 --output_file discriminating_cloze_questions.json
```

**Example 4: Generate 10 multi-turn conversations and save them.**
```bash
python main.py conversation --num_items 10 --output_file statistical_conversations.json
```

For a full list of options and their descriptions, run:
```bash
python main.py --help
```
