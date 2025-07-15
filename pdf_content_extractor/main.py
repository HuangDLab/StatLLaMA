# main.py

import os
import json
import argparse
import logging
from tqdm import tqdm
from pix2text import Pix2Text

# Local imports
from pdf_processor import process_pdf_with_text, process_pdf_with_ocr

def main():
    """Main function to run the PDF processing pipeline."""
    # --- Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Extract and clean text from all PDF files in a specified folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_folder", type=str, help="The path to the folder containing PDF files.")
    parser.add_argument("output_file", type=str, help="The path to the output JSON file where extracted texts will be saved as a list.")
    parser.add_argument(
        "--method",
        choices=["text", "ocr"],
        default="text",
        help="Extraction method: 'text' for direct text extraction (fast, for digital PDFs), 'ocr' for image-based OCR (slow, for scanned or complex PDFs)."
    )
    parser.add_argument("--start_page", type=int, default=1, help="1-based starting page for extraction from each PDF.")
    parser.add_argument("--end_page", type=int, default=-1, help="1-based ending page for extraction. Use -1 to process until the last page.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for rendering pages when using the 'ocr' method.")
    args = parser.parse_args()

    # --- Input Validation ---
    if not os.path.isdir(args.source_folder):
        logging.error(f"Source folder not found: {args.source_folder}")
        return

    # --- OCR Model Initialization (if needed) ---
    p2t_instance = None
    if args.method == 'ocr':
        logging.info("Initializing Pix2Text model for OCR. This may take a moment...")
        try:
            p2t_instance = Pix2Text()
            logging.info("Pix2Text model initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Pix2Texit. Cannot proceed with OCR. Error: {e}")
            return

    # --- Main Processing Loop ---
    all_extracted_texts = []
    pdf_files = sorted([f for f in os.listdir(args.source_folder) if f.lower().endswith(".pdf")])
    
    if not pdf_files:
        logging.warning(f"No PDF files found in the source folder: {args.source_folder}")
        return

    logging.info(f"Found {len(pdf_files)} PDF files to process using the '{args.method}' method.")

    for filename in pdf_files:
        filepath = os.path.join(args.source_folder, filename)
        logging.info(f"--- Starting processing for: {filename} ---")
        
        extracted_text = ""
        if args.method == 'text':
            extracted_text = process_pdf_with_text(filepath, args.start_page, args.end_page)
        elif args.method == 'ocr':
            extracted_text = process_pdf_with_ocr(filepath, p2t_instance, args.start_page, args.end_page, args.dpi)
        
        if extracted_text:
            all_extracted_texts.append(extracted_text)
            logging.info(f"Successfully extracted and cleaned content from {filename}.")
        else:
            logging.warning(f"No text could be extracted from {filename}.")

    # --- Save Final Output ---
    if not all_extracted_texts:
        logging.warning("No text could be extracted from any of the PDF files. No output file will be created.")
        return

    logging.info(f"Saving extracted content from {len(all_extracted_texts)} files to {args.output_file}...")
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            # Saving as a simple JSON list of strings
            json.dump(all_extracted_texts, f, ensure_ascii=False, indent=2)
        logging.info("File saved successfully.")
    except IOError as e:
        logging.error(f"Failed to write to output file: {e}")

if __name__ == "__main__":
    main()