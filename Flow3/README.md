# Flow 3: Focused Specialization from a High-Quality Starting Point

## 1. Introduction & Core Philosophy

This directory documents **Flow 3**, the primary and most successful experimental pipeline of this thesis. It represents a paradigm shift away from the foundational "building-block" approaches of Flow 1 and 2.

**Core Philosophy:** Instead of incrementally teaching a base model various capabilities, Flow 3 adopts a more pragmatic and efficient strategy: **leverage a state-of-the-art, pre-existing instruction-tuned model and focus entirely on targeted domain specialization.**

The central hypothesis is that by starting with a model that already excels at understanding and following instructions, we can bypass the most significant bottleneck identified in previous flows. This allows us to dedicate all resources to the nuanced challenges of domain adaptation: injecting deep statistical knowledge, refining output quality to match expert expectations, and performing final, delicate adjustments for real-world application scenarios.

- **Starting Model:** `LLaMA-3.2-3B-Instruct Model`
- **General Pipeline:** `SFT (Supervised Fine-Tuning) -> PA (Preference Alignment) -> DT FT (Downstream Task Fine-Tuning)`

---

## 2. Experimental Structure & Evolution

The experiments within Flow 3 are structured as a logical evolution of thought, moving from broad, complex strategies to highly focused, systematic optimizations.

### 2.1. `SFT_v1`: An Exploratory Multi-Stage, Sequential SFT Strategy

This initial exploration in Flow 3 tested a complex, "layered" or "curriculum-based" approach to Supervised Fine-Tuning (SFT).

- **Motivation:** To investigate the hypothesis that a deliberate, phased injection of different knowledge types—first foundational reasoning, then deep domain concepts, and finally conversational style—could yield a more robust and well-rounded model. This approach simulates a structured human learning curriculum.
- **Methodology & Scripts:**
  - `SFT1.py`, `SFT2.py`: The process begins by fine-tuning the base Instruct model on `Statistical CoT` and `Math_QA`. The goal is to first strengthen the model's foundational capabilities in logical and mathematical reasoning, which are prerequisites for advanced statistical understanding.
  - `SFT3.py`: The model from the previous step is then subjected to further fine-tuning on `S2ORC` data. This stage is designed for deep immersion into the dense, formal language of academic statistical literature.
  - `SFT4-1.py`, `SFT4-2.py`: Two parallel fine-tuning paths are explored to add distinct, complementary skills. `SFT4-1` uses `FineTome-100k` to improve general conversational fluency, while `SFT4-2` uses `stat. defs/nouns` to sharpen the model's precision with core terminology.
  - `adapter_merge.ipynb`: This Jupyter Notebook implements a sophisticated adapter merging technique. The LoRA adapters from the two parallel paths (`SFT4-1` and `SFT4-2`) are merged using a 2:3 weighted average. This is an attempt to synergize the "conversational" and "terminological" capabilities learned in the previous step. The resulting model is designated `SFT5`.
  - `GRPO-v2.py`: As a final step in this sub-pipeline, the merged `SFT5` model undergoes preference alignment using GRPO. The training data is a mix of `statistical GRPO` and `GSM8K` data, designed to test the model's potential for alignment after this complex SFT process.
- **Key Learning:** While theoretically appealing, this complex, multi-stage pipeline proved difficult to manage and tune. The final model did not show a clear advantage over simpler approaches, suggesting that the overhead of sequential training and adapter merging might not be justified. This led to a strategic pivot towards more unified and manageable SFT strategies.

### 2.2. `SFT_v2`: A Unified, Single-Stage SFT and Comprehensive PA Exploration

Reacting to the complexity and ambiguous results of `SFT_v1`, this directory tests a more direct and pragmatic approach. All relevant datasets are mixed and used in a single, comprehensive SFT stage, which then serves as a stable base for a thorough exploration of different Preference Alignment (PA) techniques.

- **Motivation:** To determine if a "single shot" of diverse, high-quality data could be more effective and manageable than a complex sequential curriculum. This also provides a controlled environment to directly compare the performance and stability of GRPO and DPO.
- **Methodology & Scripts:**
  - `SFT-v2.py`: A single, unified SFT is performed on the Instruct model. The training data is a carefully curated mixture of `stat. defs/nouns`, `Statistical CoT`, `statistical GRPO`, and `FineTome-100k`, designed to provide a balanced exposure to vocabulary, reasoning, and conversational style in one go.
  - `GRPO-v1.py`, `GRPO-v2.py`, `GRPO-v3.py`: The `SFT-v2` model is used as a consistent baseline to systematically investigate the hyperparameter sensitivity of the GRPO algorithm. These scripts vary the `lora_rank` (32, 8, 16) and the data mixing ratio between `statistical GRPO` and `GSM8K` data. The volatile and often contradictory results from these experiments revealed GRPO's instability and difficulty in tuning for a balanced performance profile in this specific domain.
  - `DPO.py`: As a direct, head-to-head comparison with GRPO, DPO is applied to the same `SFT-v2` model, using the same underlying preference data. This crucial experiment demonstrated DPO's superior stability, ease of use, and effectiveness in improving model performance without the unpredictable trade-offs observed with GRPO.

### 2.3. `SFT_v3`: Systematic Ablation for Optimal SFT Configuration

This directory represents the heart of our SFT optimization process. It moves beyond broad strategies to a series of fine-grained, systematic experiments designed to isolate the impact of each data component and key hyperparameter.

- **Motivation:** To move from intuition-driven to data-driven decision-making. The goal is to scientifically identify the precise data mixture and training configuration that maximizes statistical proficiency (measured by `AP Statistics` score) while quantitatively managing the inevitable trade-off with general reasoning capabilities (measured by `GSM8K` and `ARC` scores).
- **Methodology (`SFT-v3.1.py` to `SFT-v3.6.py`):**
  - Each script in this series is a carefully controlled experiment. They systematically vary the inclusion and repetition count of each dataset (`Stat. Nouns/Defs`, `Statistical CoT`, etc.).
  - They also test the impact of critical hyperparameters, most notably `train_on_responses_only` (which dictates whether the model learns from the entire dialogue or just the assistant's response) and the total number of training `epochs`.
  - The results from this comprehensive ablation study provided a clear "performance map," charting the relationship between training choices and evaluation outcomes. This allowed us to select specific SFT models (like `v3.3` and `v3.4`) that represented the optimal balance points for the next, critical DPO stage.
- **Usage Example (for the key `SFT-v3.4.py` script):**
  ```bash
  python SFT-v3.4.py
  ```

### 2.4. `SFT_v3_DPO`: Validating DPO's Dual Capability: Refinement and Recovery

This directory acts as the crucial link between the SFT and final model stages. It takes the most promising, high-performing models identified in the `SFT_v3` ablation study and subjects them to DPO.

- **Motivation:** To test a key, high-impact hypothesis: can DPO not only refine a model's output to align with expert preferences but also **actively recover the general reasoning abilities** that were inevitably compromised during the intense, domain-specific SFT process? This tests DPO's role as both a "polisher" and a "restorer."
- **Methodology & Scripts:**
  - `SFT_v3.3_DPO.py`: Applies DPO to the `SFT-v3.3` model.
  - `SFT_v3.4_DPO.py`: Applies DPO to the `SFT-v3.4` model. The success of this particular script was a major breakthrough in the research. The results showed a simultaneous improvement in the core `AP Statistics` score and a significant recovery in the `GSM8K` general math reasoning score, proving DPO's powerful dual function.
- **Usage Example:**
  ```bash
  python SFT_v3.4_DPO.py
  ```

### 2.5. `SFT_v3.4_DPO_DTFT`: The Final Polish - Downstream Task Fine-Tuning

This final directory represents the pinnacle of the training pipeline, performing what can be described as **"surgical" adjustments** on our best-performing model from the previous stage (`SFT_v3.4_DPO`).

- **Motivation:** Any further training on a highly optimized model carries a significant risk of "over-training" and catastrophic forgetting. The goal of this stage is to prove that a carefully controlled, extremely low-intensity fine-tuning protocol can successfully adapt the model to specific, high-quality downstream tasks **without degrading its hard-won core competencies**.
- **Methodology (`SFT_v3.4_DPO_DTFT_v1.py` to `v5.py`):**
  - This is a hyperparameter search in a very delicate and sensitive space. Each script tests a different, subtle fine-tuning strategy, varying:
    - **Fine-tuning Intensity:** Controlled via `LoRA Rank` and `LoRA Alpha`. The most successful strategies used very low ranks (e.g., 8).
    - **Training Duration:** Using a precise, small number of `steps` rather than full `epochs`.
    - **Data Combination:** Exploring different mixes of the highest-quality expert datasets available: `Cross Validated` (representing real-world, complex expert Q&A), `Knowledge Graph` (for structured query handling), and `Statistical Conversation` (for multi-turn dialogue flow).
- **Key Learning:** This stage provided conclusive evidence for the **"less is more" principle** in the final stages of model optimization. The most successful strategy (`v2`) involved extremely low-intensity fine-tuning (low LoRA rank, a small number of steps). This delicate approach was critical as it preserved the model's peak performance across all benchmarks while successfully adapting it to the target data, resulting in the final **StatLLaMA** model.

---

## 3. How to Run the Full (Optimal) Pipeline to Create StatLLaMA

To replicate the creation of the final **StatLLaMA** model, you must follow the most successful path identified through the rigorous experiments. This involves running specific scripts from the `SFT_v3`, `SFT_v3_DPO`, and `SFT_v3.4_DPO_DTFT` directories in sequence.

1.  **SFT Stage - Run `SFT-v3.4.py`:**
    -   **Input:** The base `LLaMA-3.2-3B-Instruct` model.
    -   **Action:** Execute the script located at `Flow3/SFT_v3/SFT-v3.4.py` with its corresponding data and configuration.
    -   **Output:** The specialized fine-tuned model, saved to `./models/sft-v3.4` and optionally pushed to the Hugging Face Hub under your specified model ID..

2.  **DPO Stage - Run `SFT_v3.4_DPO.py`:**
    -   **Input:** The `sft-v3.4` model checkpoint from the previous step.
    -   **Action:** Execute the script located at `Flow3/SFT_v3_DPO/SFT_v3.4_DPO.py`.
    -   **Output:** The preference-aligned model, saved to `./models/sft-v3.4-dpo` and optionally pushed to the Hugging Face Hub under your specified model ID..

3.  **DT FT Stage - Run `SFT_v3.4_DPO_DTFT_v2.py`:**
    -   **Input:** The `sft-v3.4-dpo` model checkpoint from the DPO stage.
    -   **Action:** Execute the script for the optimal low-intensity strategy, found at `Flow3/SFT_v3.4_DPO_DTFT/SFT_v3.4_DPO_DTFT_v2.py`.
    -   **Output:** The final, champion model of this research: **StatLLaMA**.

**Note:** Please ensure you replace all placeholder paths in the `Usage` commands with the actual locations of your models and datasets. All necessary scripts and configurations are located within their respective subdirectories.
