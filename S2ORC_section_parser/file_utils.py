# file_utils.py

import json
import logging
from typing import List, Dict

def merge_json_lists(file_paths: List[str], output_path: str):
    logging.info(f"Starting to merge {len(file_paths)} JSON files...")
    merged_list = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_list.extend(data)
                else:
                    logging.warning(f"File {file_path} does not contain a list. Skipping.")
        except FileNotFoundError:
            logging.warning(f"File not found: {file_path}. Skipping.")
        except json.JSONDecodeError:
            logging.warning(f"Error decoding JSON from {file_path}. Skipping.")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)
        logging.info(f"Merge complete. A total of {len(merged_list)} records have been saved to '{output_path}'")
    except IOError as e:
        logging.error(f"Failed to write merged file to {output_path}: {e}")