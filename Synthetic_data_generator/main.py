import os
import argparse
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Import modularized functions
from gemini_utils import (
    call_gemini_api, 
    clean_and_parse_json, 
    deduplicate_semantically, 
    save_to_json
)
from prompt_factory import (
    create_dpo_prompt, 
    create_cloze_prompt, 
    create_conversation_prompt
)

def main():
    """Main function to orchestrate the data generation script."""
    # --- Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    load_dotenv()  # Load environment variables from .env file
    
    API_KEY = os.getenv("GOOGLE_API_KEY")
    if not API_KEY:
        logging.error("API key not found. Please set the GOOGLE_API_KEY environment variable in a .env file.")
        return
    genai.configure(api_key=API_KEY)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Generate statistical training data using the Gemini API.")
    parser.add_argument("task", choices=["dpo", "cloze", "conversation"], help="The type of data to generate.")
    parser.add_argument("--num_items", type=int, default=50, help="The number of data points to generate.")
    parser.add_argument("--model", type=str, default="gemini-1.5-pro-latest", help="The Gemini model to use.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--deduplicate", action="store_true", help="Run semantic deduplication on the generated data.")
    parser.add_argument("--dedup_threshold", type=float, default=0.95, help="Similarity threshold for deduplication.")
    parser.add_argument("--simple", action="store_true", help="For 'dpo' task, generate simpler, foundational prompts.")
    parser.add_argument("--general", action="store_true", help="For 'cloze' task, generate general instead of discriminating questions.")
    args = parser.parse_args()

    # --- Prompt and Configuration Mapping ---
    prompt_fn_map = {
        "dpo": lambda: create_dpo_prompt(args.num_items, simple=args.simple),
        "cloze": lambda: create_cloze_prompt(args.num_items, discriminating=not args.general),
        "conversation": lambda: create_conversation_prompt(args.num_items)
    }
    
    expected_keys_map = {
        "dpo": ["prompt", "chosen", "rejected"],
        "cloze": ["id", "category", "question_text", "answer"],
        "conversation": None # Handled separately
    }

    dedup_key_map = {
        "dpo": "prompt",
        "cloze": "question_text",
        "conversation": "initial_prompt" # A temporary key for deduplication
    }
    
    prompt = prompt_fn_map[args.task]()
    expected_keys = expected_keys_map[args.task]
    dedup_key = dedup_key_map[args.task]

    # --- Execution Pipeline ---
    
    # 1. Generate content from API
    response_text = call_gemini_api(prompt, args.model)
    if not response_text:
        logging.error("Failed to get a response from the API. Exiting.")
        return
        
    # 2. Parse response text into a list of dictionaries
    if args.task == "conversation":
        # Special handling for conversation format which is a list of lists
        parsed_data = clean_and_parse_json(response_text)
        if parsed_data:
            # Standardize format for easier processing downstream (e.g., deduplication)
            generated_data = []
            for i, conv in enumerate(parsed_data):
                if isinstance(conv, list) and len(conv) > 0 and 'content' in conv[0]:
                    generated_data.append({
                        "id": i,
                        "initial_prompt": conv[0]['content'],
                        "messages": conv
                    })
            logging.info(f"Successfully parsed {len(generated_data)} conversations.")
        else:
            generated_data = []
    else:
        generated_data = clean_and_parse_json(response_text, expected_keys)

    if not generated_data:
        logging.error("No valid data was generated or parsed. Exiting.")
        return

    # 3. Deduplicate if requested
    final_data = generated_data
    if args.deduplicate:
        final_data = deduplicate_semantically(generated_data, dedup_key, args.dedup_threshold)
    
    # 4. Revert conversation format if necessary before saving
    if args.task == "conversation":
        # Strip the temporary helper keys to save the original list-of-lists format
        final_data = [item['messages'] for item in final_data]

    # 5. Save final data to file
    save_to_json(final_data, args.output_file)

if __name__ == "__main__":
    main()