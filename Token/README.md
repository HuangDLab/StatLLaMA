# Pre-tokenization for Continual Pre-training (CoP)

This directory contains `Data2Token.py`, a crucial pre-processing script designed specifically for the **Continual Pre-training (CoP)** stage of our research pipelines (Flow 1 and Flow 2).

## 1. Purpose & Rationale

**Why Pre-tokenize?**

The CoP stage involves training a model on a very large corpus of unstructured text (e.g., academic papers, definitions). In such scenarios, the data processing pipeline can become a significant bottleneck if tokenization is performed on-the-fly during training.

**Pre-tokenization** is a strategy to overcome this. The purpose of this script is to perform the computationally expensive task of tokenizing all raw text data **once**, and then save the resulting `token_ids` and `labels` to a file. The main training script (`CoP.py`) can then directly load these processed tokens, bypassing the need for repeated tokenization and dramatically accelerating the training startup time and overall efficiency.

In essence, this script transforms raw text data into a "training-ready" format optimized for large-scale, self-supervised learning.

---

## 2. Script Overview: `Data2Token.py`

This script is responsible for loading a JSON file containing a list of raw text strings, processing them into a format suitable for causal language modeling, and saving the result.

### 2.1. Core Functionality

1.  **Load Raw Data:** Reads a JSON file (e.g., `pretrain_data.json`) which is expected to contain a list of strings.
2.  **Load Tokenizer:** Loads a specified Hugging Face tokenizer (e.g., `unsloth/Llama-3.2-3B-Instruct`), ensuring it has a padding token.
3.  **Tokenize and Chunk:**
    -   Each raw text string is tokenized into a sequence of `input_ids`.
    -   To handle texts of varying lengths and to create uniform training examples, the tokenized sequence is split into fixed-length **chunks** (e.g., 2048 tokens long).
    -   A sliding window (`stride`) is used during chunking to ensure some overlap between consecutive chunks, helping the model learn long-range dependencies.
4.  **Format for Causal LM:**
    -   Each chunk is prepended with a `bos_token_id` (Beginning of Sentence) and appended with an `eos_token_id` (End of Sentence).
    -   The script creates two identical versions for each chunk:
        -   `input_ids`: The token IDs that the model sees as input.
        -   `labels`: The token IDs that the model is expected to predict. In causal language modeling, the labels are typically a shifted version of the inputs, but for simplicity and compatibility with standard libraries, they are often set to be the same as the `input_ids`.
5.  **Padding and Masking:**
    -   Each chunk is padded to a maximum length (`chunk_max_length`) to ensure all tensors in a batch have the same dimension. Padding is added to the **left**.
    -   Crucially, the `labels` corresponding to the padding tokens are set to `-100`. This is a standard practice in Hugging Face Transformers to instruct the loss function to **ignore** these padded tokens during the calculation of the loss, ensuring the model only learns from the actual content.
6.  **Save Processed Data:** The final processed data, a dictionary containing two large lists (`"input_ids"` and `"labels"`), is saved to a new JSON file (e.g., `pretrain_data_token.json`).

### 2.2. Key Parameters

-   `model_name_or_path`: The Hugging Face model identifier to load the correct tokenizer. It's critical that this matches the tokenizer used for the actual CoP training.
-   `input_json_path`: Path to the raw JSON file containing a list of text strings.
-   `output_json_path`: Path where the final tokenized data will be saved.
-   `tokenizer_max_length`: The absolute maximum sequence length the tokenizer can handle.
-   `chunk_max_length`: The fixed length for each training chunk after tokenization and padding. This value should be set based on the GPU memory available during training.

---

## 3. Usage

To use this script, ensure you have your raw text data prepared in a JSON file as a list of strings. Then, run the script from your terminal.

1.  **Prepare your data:** Create a file named `pretrain_data.json` with content like:
    ```json
    [
        "The central limit theorem (CLT) states that the distribution of a sample mean approximates a normal distribution...",
        "Standard Deviation: A measure of the amount of variation or dispersion of a set of values...",
        "Another long text entry from an academic paper..."
    ]
    ```

2.  **Modify the script (if necessary):** Adjust the path and length parameters at the top of `Data2Token.py` to match your setup.

3.  **Run the script:**
    ```bash
    python Data2Token.py
    ```

The script will display progress bars as it processes the texts and will notify you upon successful completion. The output file (`pretrain_data_token.json`) will be ready to be used directly by the `CoP.py` training script.