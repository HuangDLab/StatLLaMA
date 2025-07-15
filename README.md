# StatLLaMA

> This repository contains the source code, experimental configurations, and documentation for the Master's thesis:
> **"Adapting Lightweight Language Models to the Statistical Domain: A Study of Multi-Stage Fine-Tuning"**

This research introduces **StatLLaMA**, a highly capable, lightweight language model specialized for the domain of statistics. The project encompasses a complete end-to-end workflow, from sophisticated data engineering pipelines for corpus creation to the systematic training and evaluation methodologies that led to its development.

---

## 1. Research Overview

### 1.1. Motivation & Problem Statement

While general-purpose LLMs have demonstrated remarkable capabilities, they often fall short in specialized domains like statistics. This research addresses a fundamental question: **How can we efficiently and effectively adapt a lightweight LLM to the statistical domain, enhancing its specialized capabilities while preserving its valuable general reasoning abilities?**

### 1.2. Core Contributions

This work provides a comprehensive and empirical investigation into LLM domain adaptation, culminating in several key contributions:
1.  **End-to-End Data Engineering Pipelines:** Developed a suite of robust tools for collecting, parsing, cleaning, and synthesizing high-quality training data from diverse sources like arXiv, S2ORC, and the Gemini API.
2.  **Validated a High-Efficacy Training Pipeline:** Established and proved the effectiveness of a `SFT -> DPO -> DT FT` pipeline starting from a high-quality instruction-tuned model (`LLaMA-3.2-3B-Instruct`).
3.  **Systematic Comparison of Techniques:** Provided direct, in-domain comparisons of key methods, such as `DPO` vs. `GRPO`, demonstrating DPO's superior stability and effectiveness.
4.  **Key Findings on Model Training:** Quantified critical principles, such as DPO's ability to recover lost capabilities and the "less is more" rule for fine-tuning highly optimized models.
5.  **Developed StatLLaMA:** Produced a lightweight (3B) yet highly capable statistical language model with significant performance gains.

---

## 2. Repository Structure

This project is organized into two primary categories: **Part I: Data Engineering & Pre-processing** and **Part II: Model Training Experimental Flows**. Each directory is a self-contained module with its own specific `README.md` for detailed instructions.

```
.
├── Part I: Data Engineering & Pre-processing
│   ├── ArXiv_multimodal_extractor/  # Crawls and parses arXiv papers with multimodal features (BLIP).
│   ├── S2ORC_section_parser/        # Memory-efficient streaming parser for the massive S2ORC dataset.
│   ├── pdf_content_extractor/       # General-purpose tool for PDF text extraction (Text & OCR).
│   ├── Synthetic_data_generator/    # Generates synthetic data (DPO, etc.) using the Gemini API.
│   └── Token/                       # Pre-tokenizes large corpora for efficient Continual Pre-training.
│
└── Part II: Model Training Experimental Flows
    ├── Flow1/                       # Hypothesis: Knowledge-First from Base Model
    │   ├── CoP.py
    │   ├── SFT.py
    │   └── DPO.py
    ├── Flow2/                       # Hypothesis: Bridging the Capability Gap
    │   ├── CoP.py
    │   ├── Instruct.py
    │   └── ...
    └── Flow3/                       # Hypothesis: Focused Specialization (The Successful Path)
        ├── SFT_v1/
        ├── SFT_v2/
        ├── SFT_v3/
        ├── SFT_v3_DPO/
        └── SFT_v3.4_DPO_DTFT/
        └── ...
```

-   **Part I** represents the collection of standalone scripts developed to collect, parse, synthesize, and prepare the data for training.
-   **Part II** groups the training scripts according to the three core experimental strategies (Flows) that were systematically tested and evaluated. Each script corresponds to a specific stage within a flow..

---

## 3. Datasets

Due to the large size of the training and evaluation datasets, they are not hosted directly in this GitHub repository. The complete, pre-processed datasets used for all experiments are available for download from Kaggle.

-   **Download Link:** **[StatLLaMA - Thesis Datasets on Kaggle](https://kaggle.com/datasets/0c6ebbb2fe0a8532ed46784b479d2122d16e8ffc35fa7c244508d19f11500921)**

The dataset on Kaggle is organized to correspond with the different stages of the training flows described in the paper.

---

## 4. Getting Started

### 4.1. Prerequisites
-   Python 3.9+
-   PyTorch 2.0+ with CUDA support.
-   An NVIDIA GPU suitable for running 3B parameter models (e.g., A100, H100).
-   API keys for services like Google Gemini, if using the synthetic data generator.

### 4.2. Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EiLaTe123/StatLLaMA
    cd StatLLaMA
    pip install -e .
    ```
2.  **Install dependencies:**
    The `requirements.txt` file in the root directory contains all packages used for the **model training** stages (`Flow1`, `Flow2`, `Flow3`). For the individual data engineering tools, please refer to the `requirements.txt` file within each specific subdirectory.
    ```bash
    # For model training
    pip install -r requirements.txt

    # Example for a specific data engineering tool
    cd ArXiv_multimodal_extractor/
    pip install -r requirements.txt
    ```

### 4.3. How to Use This Repository
1.  **Data Preparation:** Navigate to the desired data engineering directory (e.g., `ArXiv_multimodal_extractor/`, `S2ORC_section_parser/`) and follow the instructions in its specific `README.md` to generate your training datasets.
2.  **Model Training:** Navigate to the desired model training directory (`Flow1/`, `Flow2/`, or `Flow3/`) and follow the detailed instructions in its `README.md` to run the training scripts. To replicate the final **StatLLaMA** model, follow the "golden path" outlined in `Flow3/README.md`.

---

## 5. Key Results

The primary outcome of this research is the **StatLLaMA** model, which achieved a balanced and significant performance uplift.

| Model                       | GSM8K (8-shot) | AP Statistics (0-shot) | ARC (0-shot) |
| --------------------------- | :------------: | :--------------------: | :----------: |
| LLaMA-3.2-3B-Instruct (Base) |     64.44      |         37.63          |    43.60     |
| **StatLLaMA (Final)**       |   **58.83**    |       **41.46**        |  **40.61**   |

The final model demonstrates a significant improvement in the core domain-specific benchmark while largely preserving its general reasoning capabilities, validating the effectiveness of the Flow 3 pipeline.

---

## 6. Citation

If you find this research or any of the accompanying tools useful, please consider citing the thesis:

```bibtex
@mastersthesis{zeng2025adapting,
  title  = {Adapting Lightweight Language Models to the Statistical Domain: A Study of Multi-Stage Fine-Tuning},
  author = {Zeng, Jing-Yi},
  school = {National Yang Ming Chiao Tung University},
  year   = {2025}
}
```

## 7. Acknowledgements

I would like to express my sincere gratitude to my advisor, Professor Guan-Hua Huang, for his invaluable guidance and support throughout this research.
