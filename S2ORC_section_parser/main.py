# main.py

import os
import json
import logging
from s2orc_parser import S2orcParser
from file_utils import merge_json_lists

def main():
    """
    Main execution pipeline for parsing S2ORC data files.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- Configuration ---
    # Directory containing your raw S2ORC data files (with no extension)
    RAW_DATA_DIR = 'data/' # IMPORTANT: Place your S2ORC files here

    FILENAMES_TO_PROCESS = [
        '20240816_112927_00009_5ziwh_008a60e2-9ff6-4203-8965-a2c54133d6ac',
    ]

    # Directory to store intermediate parsed files
    PARTS_DIR = 's2orc_parts'

    # Keywords to identify relevant sections. Should be lowercase.
    SUBTITLE_KEYWORDS = [
        "introduction", "abstract", "method", "statistic", "model",
        "statistics", "analysis", "analyses", "conclusion", "experiment"
    ]
    
    # Final merged output file path
    MERGED_OUTPUT_PATH = 'S2ORC_parsed_merged.json'
    
    # Configuration for the optional final auditing step
    AUDIT_CONTENT_LENGTH_THRESHOLD = 100  # Sections with fewer chars than this will be flagged
    AUDIT_OUTPUT_PATH = 'S2ORC_short_content_audit_report.json'


    # --- Step 1: Parse Raw S2ORC Files ---
    if not FILENAMES_TO_PROCESS:
        logging.error("The `FILENAMES_TO_PROCESS` list is empty. Please specify which files to process.")
        return

    if not os.path.exists(PARTS_DIR):
        os.makedirs(PARTS_DIR)
    
    parser = S2orcParser(keywords=SUBTITLE_KEYWORDS)
    
    part_files_to_merge = []
    for filename in FILENAMES_TO_PROCESS:
        filepath = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(filepath):
            logging.warning(f"Source file not found: {filepath}. Skipping.")
            continue
            
        part_output_path = os.path.join(PARTS_DIR, f'parsed_{filename}.json')
        
        processed_data = parser.parse_and_filter_file(filepath)
        
        if processed_data:
            with open(part_output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=4)
            logging.info(f"Intermediate result saved to: {part_output_path}")
            part_files_to_merge.append(part_output_path)
        else:
            logging.warning(f"No relevant papers found in {filepath}. No intermediate file created.")

    if not part_files_to_merge:
        logging.error("No data was processed from any of the specified source files.")
        return

    # --- Step 2: Merge All Parsed JSON Files ---
    merge_json_lists(part_files_to_merge, MERGED_OUTPUT_PATH)

    # --- Step 3: Audit Merged File for Short Content (Optional Data Quality Check) ---
    logging.info("Starting optional audit for short content sections...")
    try:
        with open(MERGED_OUTPUT_PATH, "r", encoding='utf-8') as f:
            all_articles = json.load(f)
    except FileNotFoundError:
        logging.error(f"Merged file not found at {MERGED_OUTPUT_PATH}. Cannot perform audit.")
        return
        
    short_content_list = parser.audit_subtitles_by_length(
        articles=all_articles,
        threshold=AUDIT_CONTENT_LENGTH_THRESHOLD
    )

    if short_content_list:
        with open(AUDIT_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(short_content_list, f, ensure_ascii=False, indent=4)
        logging.info(f"Audit report for short content sections saved to: {AUDIT_OUTPUT_PATH}")
    else:
        logging.info("Audit complete. No sections with short content were found.")


if __name__ == "__main__":
    main()