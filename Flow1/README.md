# Flow 1: Knowledge-First Domain Adaptation

This directory contains the scripts for **Flow 1**, a training pipeline designed to explore a "knowledge-first" domain adaptation strategy for Large Language Models (LLMs).

## Overview & Hypothesis

The core hypothesis of Flow 1 is that by first immersing a base model in a vast amount of domain-specific text, we can build a strong knowledge foundation. Subsequent fine-tuning stages for task application (SFT) and preference alignment (DPO) might then become more effective and efficient.

This flow deliberately starts with a non-instruct-tuned base model to test this hypothesis in its purest form.

- **Starting Model:** `LLaMA-3.2-3B Base Model`
- **Training Pipeline:** `CoP -> SFT -> DPO`

> **Note on Experimental Results:** As detailed in the thesis, this "knowledge-first" flow was found to be suboptimal. The base model, lacking an inherent instruction-following capability, struggled to effectively utilize the knowledge absorbed during CoP in the later, more structured SFT and DPO stages. This led to minimal performance gains and even degradation in some metrics. This flow serves as a crucial baseline that validates the necessity of a strong, instruction-tuned starting point for effective domain adaptation.

---

## Scripts

This flow consists of three sequential scripts: `CoP.py`, `SFT.py`, and `DPO.py`.

### 1. `CoP.py`: Continual Pre-training

#### Purpose
This script performs the initial and most critical stage of Flow 1: Continual Pre-training. Its goal is to solve the fundamental "knowledge gap" by exposing the base model to a large corpus of domain-specific text, allowing it to absorb the terminology, concepts, and linguistic patterns of the statistical domain.

#### Methodology
- **Learning Mechanism:** Self-supervised learning (causal language modeling). The model learns to predict the next token in a sequence, forcing it to internalize the statistical patterns of the text.
- **Data:** The script uses a combination of unstructured, large-scale text corpora:
  - **`S2ORC`:** Paragraphs extracted from academic papers in statistics, providing exposure to formal scientific writing and complex concepts.
  - **`Statistical Nouns/Defs`:** A curated list of statistical terms and their definitions, ensuring the model builds a precise understanding of core vocabulary.
- **Processing:** To handle the large data volume efficiently, this stage utilizes **pre-tokenization**. All text is converted into `token_ids` and saved to disk before training begins, significantly speeding up the training loop.

#### Usage
```bash
python CoP.py
```

### 2. `SFT.py`: Supervised Fine-Tuning

#### Purpose
After the CoP stage, the model has a passive knowledge base but does not know how to apply it to specific tasks. This SFT script aims to "activate" that knowledge by teaching the model to follow instructions and solve problems in a structured manner. It injects the ability to perform specific tasks.

#### Methodology
- **Learning Mechanism:** Supervised learning. The model is trained on structured "prompt-response" pairs to minimize the loss between its generated response and the target response.
- **Data:** The script uses structured Question-Answering (QA) and Chain-of-Thought (CoT) datasets, such as:
  - **`Statistical CoT`:** QA pairs that include step-by-step reasoning, teaching the model not just the answer, but the process to get there.
  - **`GSM8K`:** Math word problems to build foundational reasoning skills.
- **Processing:** Unlike CoP, this stage applies a **Chat Template** to format the data into a conversational structure. Tokenization is handled on-the-fly by the training framework to accommodate the more complex, structured data.

#### Usage
```bash
python SFT.py
```

### 3. `DPO.py`: Direct Preference Optimization

#### Purpose

The final stage of Flow 1 aims to refine the model's output quality by aligning it with human preferences. An SFT model might provide correct but verbose, unnatural, or unhelpful answers. DPO is employed to teach the model to distinguish between "good" and "bad" responses, with the goal of making its output more useful, reliable, and aligned with expert expectations.

#### Methodology

-   **Learning Mechanism:** Direct Preference Optimization (DPO). This method bypasses the need for an explicit reward model by directly optimizing the language model on pairs of (`chosen` vs. `rejected`) responses. The model learns to maximize the likelihood of the `chosen` response while minimizing that of the `rejected` one.

-   **Data Strategy & Rationale:** The script utilizes a strategic blend of preference data from two related quantitative domains. This is a deliberate choice designed to build a more robust preference model.
    -   **`Stat DPO`:** Contains preference pairs specifically curated for the statistical domain. The `chosen` responses are typically clearer, more intuitive, and more practically useful than their `rejected` counterparts. This directly hones the model's ability in its target domain.
    -   **`Math DPO`:** Contains preference pairs from the mathematical reasoning domain. **The inclusion of this data is crucial.** It serves to reinforce the model's foundational logical and numerical reasoning capabilities, preventing the preference model from becoming overly specialized and brittle. By learning from both, the model is encouraged to develop a more generalized understanding of what constitutes a high-quality, rigorous quantitative explanation.

-   **Processing:** Both the `chosen` and `rejected` dialogues for each prompt are formatted using the standard Chat Template before being passed to the DPO loss function, ensuring consistency with the previous SFT stage.

#### Usage
```bash
python DPO.py
```

---

## How to Run the Full Pipeline

To execute the entire Flow 1 pipeline, run the scripts in the following order, ensuring the output of one stage is used as the input for the next.

1.  **Run `CoP.py`** on the base model to create the continually pre-trained model.
2.  **Run `SFT.py`** using the model saved from the CoP stage.
3.  **Run `DPO.py`** using the model saved from the SFT stage to produce the final model for Flow 1.
