# prompt_factory.py

def create_dpo_prompt(num_to_generate: int, simple: bool = False) -> str:
    """Creates a prompt to generate Direct Preference Optimization (DPO) pairs."""
    if simple:
        focus_description = "focus on **foundational statistics** suitable for smaller models."
        example_topics = """
            * Interpreting p-values or confidence intervals.
            * The difference between correlation and causation.
            * Explaining standard deviation or variance.
        """
    else:
        focus_description = "present a realistic scenario or problem requiring the **application of advanced statistical concepts**."
        example_topics = """
            * "A medical researcher is studying patient survival time... How should this 'incomplete' data be handled?" (Survival Analysis)
            * "A plant manager notices increased product variability... What statistical tool could they use to monitor this?" (Statistical Process Control)
            * "When analyzing stock returns, volatility seems to cluster... What type of model captures this?" (Time Series / GARCH)
        """
        
    return f"""
You are an expert AI assistant specializing in practical statistics, tasked with generating high-quality training data for DPO (Direct Preference Optimization).
Your goal is to generate **{num_to_generate} distinct DPO data points** in a single response. Each data point should {focus_description}

**Instructions:**
1.  **Output Format:** Your entire response MUST be a single, valid JSON list. Each element must be a JSON object with the keys: `prompt`, `chosen`, `rejected`.
2.  **Content Requirements:**
    *   **`prompt`:** Craft a prompt that describes a practical problem, data analysis challenge, or a request for interpretation.
    *   **`chosen`:** Must be a statistically sound, practical, and well-explained response. It should justify method choices and mention relevant assumptions.
    *   **`rejected`:** Must address the same prompt but contain plausible flaws, such as suggesting an inappropriate method, misinterpreting output, or ignoring critical assumptions. Avoid trivially wrong answers.
3.  **JSON Validity:** Ensure the final output is a perfectly valid JSON list of {num_to_generate} objects.

**Example `prompt` ideas:**
{example_topics}

Now, please generate {num_to_generate} distinct, high-quality DPO data points.
"""

def create_cloze_prompt(num_to_generate: int, discriminating: bool = True) -> str:
    """Creates a prompt to generate "fill-in-the-blank" (cloze) statistical questions."""
    if discriminating:
        power_description = "possess **high discriminating power** – they should be designed to differentiate between models with superficial recall and those with a deeper, nuanced understanding."
        focus_areas = """
            * **Common Misconceptions:** (e.g., confusing correlation/causation).
            * **Subtle Distinctions:** (e.g., standard deviation vs. standard error).
            * **Assumptions:** (e.g., independence, normality) and *why* they matter.
        """
    else:
        power_description = "cover a range of statistical topics (from basic to applied) and be suitable for assessing general knowledge."
        focus_areas = """
            * "The measure of central tendency representing the middle value is the ______." (Answer: ["median"])
            * "Failing to reject a false null hypothesis is a Type ______ error." (Answer: ["II", "Two"])
            * "The ______ theorem describes the distribution of sample means." (Answer: ["Central Limit"])
        """

    return f"""
You are an expert AI assistant specializing in statistics education, tasked with generating high-quality test questions.
Your goal is to generate **{num_to_generate} distinct fill-in-the-blank statistical questions**. These questions must {power_description}

**Instructions:**
1.  **Output Format:** Your entire response MUST be a single, valid JSON list. Each element must be a JSON object with the keys: `id`, `category`, `question_text`, `answer`. The `answer` must be a list of strings.
2.  **Content Focus:**
    *   Use a statement format with a single blank `______` requiring a precise statistical term or concept.
    *   Focus on these areas:
        {focus_areas}
3.  **JSON Validity:** Ensure the final output is a perfectly valid JSON list of {num_to_generate} objects.

Please generate {num_to_generate} distinct questions now.
"""

def create_conversation_prompt(num_to_generate: int) -> str:
    """Creates a prompt to generate multi-turn statistical conversations."""
    return f"""
You are an AI specializing in statistics education. Your task is to generate **{num_to_generate} distinct, multi-turn conversations** between a "User" and a "Statistical Assistant".

**Instructions:**
1.  **Application Focus:** Each conversation must explore a core statistical concept grounded in a **practical, real-world scenario** (e.g., A/B testing, interpreting medical studies, modeling financial data).
2.  **Structure:**
    *   Generate at least 2-3 turns (1 User + 1 Assistant = 1 turn).
    *   The conversation must flow logically, with relevant follow-up questions and informative answers.
3.  **Output Format:** Return the entire batch as a single JSON list. Each element of the list must be *another list* representing a single conversation, containing message objects with "role" and "content" keys.

**Example Conversation:**
```json
[
  [
    {{"role": "user", "content": "We ran an A/B test... The p-value is 0.08. Can we confidently say Green is better?"}},
    {{"role": "assistant", "content": "With a p-value of 0.08, which is above the common 0.05 threshold, we can't rule out random chance..."}},
    {{"role": "user", "content": "So we shouldn't switch? What else should we consider?"}},
    {{"role": "assistant", "content": "Consider the practical significance and the confidence interval for the difference. Running the test longer for more data could also provide a clearer picture."}}
  ]
]
```
Now, generate a new JSON list containing exactly {num_to_generate} diverse conversations.
"""