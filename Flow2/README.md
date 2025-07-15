# Flow 2: Bridging the Capability Gap with Instruction Tuning

This directory contains the scripts for **Flow 2**, a training pipeline designed as a direct response to the limitations observed in Flow 1. It explores whether adding a general instruction-tuning stage can bridge the gap between passive knowledge absorption and active task application.

## Overview & Hypothesis

Flow 1 revealed a critical flaw in the "knowledge-first" approach: a base model, even after absorbing vast domain knowledge via CoP, lacks the fundamental ability to understand and follow instructions. This prevents it from effectively leveraging that knowledge in subsequent SFT and DPO stages.

The core hypothesis of Flow 2 is that by inserting a **general instruction-tuning stage** after CoP, we can act as a **"capability bridge"**. This stage aims to "unlock" the model's ability to interact and follow prompts before it is exposed to domain-specific tasks. We hypothesize that this bridge will enable a more effective fusion of the domain knowledge from CoP with the task skills from SFT.

- **Starting Model:** `LLaMA-3.2-3B Base Model`
- **Training Pipeline:** `CoP -> Instruction Tuning -> SFT -> DPO`

> **Note on Experimental Results:** As detailed in the thesis, while Flow 2 showed some improvements over Flow 1 (particularly in the SFT stage), it was still ultimately suboptimal. The results suggest that "retrofitting" instruction-following ability onto a knowledge-infused model is less effective than starting with a model that has been instruction-tuned from the outset (as explored in Flow 3). This flow provides crucial evidence that the sequence of capability acquisition matters significantly.

---

## Scripts

This flow consists of four sequential scripts: `CoP.py`, `Instruct.py`, `SFT.py`, and `DPO.py`.

### 1. `CoP.py`: Continual Pre-training

*(This script is identical in purpose and methodology to the one in Flow 1.)*

#### Purpose
To solve the fundamental "knowledge gap" by exposing the base model to a large corpus of domain-specific text, allowing it to absorb the terminology, concepts, and linguistic patterns of the statistical domain.

#### Methodology
- **Learning Mechanism:** Self-supervised learning (causal language modeling).
- **Data:** Unstructured, large-scale text corpora (`S2ORC`, `Statistical Nouns/Defs`).
- **Processing:** Utilizes **pre-tokenization** for training efficiency.

#### Usage
```bash
python CoP.py
```

### 2. `Instruct.py`: General Instruction Tuning

#### Purpose
This is the key differentiating script in Flow 2. Its purpose is to build the **"capability bridge"** by teaching the CoP-trained model how to understand and respond to general human instructions. This is done *before* introducing any domain-specific tasks.

#### Methodology
- **Learning Mechanism:** Supervised learning.
- **Data:** Large, diverse, **non-domain-specific** instruction-following datasets.
  - **`OpenHermes-2.5`, `Dolly-15k`:** These datasets contain a wide variety of general tasks (e.g., brainstorming, summarization, creative writing) that teach the model the fundamental structure of instruction-based interaction.
- **Processing:** Applies a **Chat Template** to format the data into a conversational structure. Tokenization is handled on-the-fly.

#### Usage
```bash
python Instruct.py
```

### 3. `SFT.py`: Domain-Specific Fine-Tuning

*(This script is identical in purpose and methodology to the one in Flow 1, but starts from the instruction-tuned model.)*

#### Purpose
To "activate" the domain knowledge from CoP, guided by the instruction-following ability learned in the `Instruct.py` stage. It teaches the model to apply its knowledge to solve specific statistical problems.

#### Methodology
- **Learning Mechanism:** Supervised learning.
- **Data:** Structured, **domain-specific** QA and CoT datasets (`Statistical CoT`, `GSM8K`).
- **Processing:** Uses the Chat Template.

#### Usage
```bash
python SFT.py
```

### 4. `DPO.py`: Direct Preference Optimization

*(This script is identical in purpose and methodology to the one in Flow 1.)*

#### Purpose
To refine the model's output quality by aligning it with human preferences for statistical explanations.

#### Methodology
- **Learning Mechanism:** Direct Preference Optimization (DPO).
- **Data:** A combined set of paired preference data from two domains:
    - **`Stat DPO`:** For domain-specific alignment in statistics.
    - **`Math DPO`:** To reinforce foundational quantitative reasoning and enhance logical rigor.

#### Usage
```bash
python DPO.py
```

---

## How to Run the Full Pipeline

To execute the entire Flow 2 pipeline, run the scripts in the following order:

1.  **Run `CoP.py`** on the base model.
2.  **Run `Instruct.py`** on the model from the CoP stage.
3.  **Run `SFT.py`** on the model from the Instruction Tuning stage.
4.  **Run `DPO.py`** on the model from the SFT stage to produce the final model for Flow 2.