# gemini_utils.py

import json
import time
import logging
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def call_gemini_api(prompt: str, model_name: str) -> str:
    """
    Calls the configured Gemini API with a given prompt and model.

    Args:
        prompt (str): The prompt to send to the model.
        model_name (str): The name of the model to use (e.g., 'gemini-1.5-pro-latest').

    Returns:
        str: The text content of the response, or an empty string on error.
    """
    try:
        logging.info(f"Generating content with model: {model_name}...")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        logging.info("Content generated successfully.")
        return response.text
    except Exception as e:
        logging.error(f"An error occurred while calling the Gemini API: {e}")
        return ""

def clean_and_parse_json(response_text: str, expected_keys: List[str] = None) -> List[Dict[str, Any]]:
    """
    Cleans markdown fences and parses a JSON string from the API response.
    Validates that the result is a list of dictionaries.

    Args:
        response_text (str): The raw text from the API response.
        expected_keys (List[str], optional): A list of keys expected in each JSON object for validation. Defaults to None.

    Returns:
        List[Dict[str, Any]]: A list of valid dictionary objects.
    """
    # Remove markdown code block fences (e.g., ```json ... ```)
    match = re.search(r'```(json)?\s*([\s\S]+?)\s*```', response_text)
    if match:
        processed_text = match.group(2).strip()
    else:
        processed_text = response_text.strip()
        
    try:
        data_list = json.loads(processed_text)
        
        if not isinstance(data_list, list):
            logging.error("Parsed content is not a JSON list.")
            return []

        if not expected_keys:
            return data_list # Skip validation if no keys are provided

        validated_data = []
        for i, item in enumerate(data_list):
            if isinstance(item, dict) and all(key in item for key in expected_keys):
                validated_data.append(item)
            else:
                logging.warning(f"Item {i+1} has invalid structure or missing keys. Skipping. Item: {str(item)[:100]}")
        
        logging.info(f"Successfully parsed and validated {len(validated_data)} items.")
        return validated_data
        
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON: {e}")
        logging.debug(f"Content that failed parsing:\n{processed_text}")
        return []

def deduplicate_semantically(data: List[Dict], key_to_check: str, threshold: float) -> List[Dict]:
    """
    Removes duplicate items from a list of dictionaries based on semantic similarity of a specified key.

    Args:
        data (List[Dict]): A list of dictionaries.
        key_to_check (str): The dictionary key containing the text to compare (e.g., 'prompt').
        threshold (float): The cosine similarity threshold (0.0 to 1.0) for marking an item as a duplicate.

    Returns:
        List[Dict]: A deduplicated list of dictionaries.
    """
    if not data or len(data) < 2:
        return data
        
    texts_to_compare = [item.get(key_to_check, "") for item in data]
    if not any(texts_to_compare):
        logging.warning("No text found in the specified key for deduplication. Returning original data.")
        return data

    logging.info("\n--- Starting Semantic Deduplication ---")
    logging.info(f"Total items to check: {len(texts_to_compare)}")
    
    model_name = 'all-MiniLM-L6-v2'
    logging.info(f"Loading Sentence Transformer model: {model_name}...")
    model = SentenceTransformer(model_name)

    logging.info("Generating embeddings...")
    embeddings = model.encode(texts_to_compare, convert_to_tensor=True, show_progress_bar=True)

    logging.info("Calculating cosine similarity matrix...")
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    
    indices_to_remove = set()
    num_items = len(texts_to_compare)
    logging.info(f"Using similarity threshold: {threshold}")

    for i in range(num_items):
        if i in indices_to_remove:
            continue
        # Compare with subsequent items
        for j in range(i + 1, num_items):
            if j in indices_to_remove:
                continue
            if similarity_matrix[i, j] > threshold:
                logging.info(f"Similarity between item {i} and {j} is {similarity_matrix[i, j]:.4f} > {threshold}. Marking item {j} for removal.")
                indices_to_remove.add(j)

    deduplicated_list = [item for i, item in enumerate(data) if i not in indices_to_remove]
    
    logging.info(f"\nTotal items marked for removal: {len(indices_to_remove)}")
    logging.info(f"Original number of items: {len(data)}")
    logging.info(f"Number of items after deduplication: {len(deduplicated_list)}")
    logging.info("--- Deduplication Complete ---")
    
    return deduplicated_list

def save_to_json(data: List[Any], filename: str):
    """Saves data to a JSON file with pretty printing."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved {len(data)} items to {filename}")
    except Exception as e:
        logging.error(f"Error saving data to {filename}: {e}")