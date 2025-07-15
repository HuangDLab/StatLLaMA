import fitz  # PyMuPDF
import logging
import requests
import io
import tempfile
import os
import re
import torch
import Levenshtein
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Dict, List, Optional, Tuple

class ArxivPaperParser:
    """
    A class to download, parse, and extract structured content from an arXiv paper.
    This includes text sections, images, and generating captions for images using a BLIP model.
    """
    def __init__(self, device: Optional[str] = None):
        """
        Initializes the parser, setting up the device and loading the BLIP model for image captioning.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Initializing ArxivPaperParser on device: {self.device}")

        # Common section headers to look for
        self.section_patterns = {
            'introduction': r'^\d*\.?\s*introduction',
            'method': r'^\d*\.?\s*(method|methodology|approach|experimental setup)',
            'result': r'^\d*\.?\s*(result|evaluation|experiment)',
            'conclusion': r'^\d*\.?\s*(conclusion|discussion|summary|future work)',
            'references': r'^\d*\.?\s*references'
        }
        
        try:
            logging.info("Loading BLIP model (Salesforce/blip-image-captioning-base)...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
            logging.info("BLIP model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load BLIP model. Image captioning will be disabled. Error: {e}")
            self.processor = None
            self.model = None

    def _download_pdf(self, arxiv_id: str) -> Optional[str]:
        """
        Downloads a PDF from arXiv to a temporary file.

        Args:
            arxiv_id (str): The arXiv identifier of the paper.

        Returns:
            Optional[str]: The file path to the downloaded temporary PDF, or None on failure.
        """
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Use a temporary file to handle download and cleanup
            fp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            fp.write(response.content)
            fp.close()
            return fp.name
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download PDF for {arxiv_id} from {pdf_url}: {e}")
            return None

    def _find_section_positions(self, doc: fitz.Document) -> Dict[str, Tuple[int, float]]:
        """
        Scans the document to find the starting page and y-position of each major section.
        
        Returns:
            A dictionary mapping section names to a tuple of (page_number, y_coordinate).
        """
        positions = {}
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        # Combine spans to get full line text
                        line_text = "".join(s["text"] for s in line["spans"]).strip().lower()
                        if not line_text:
                            continue
                        
                        for section_name, pattern in self.section_patterns.items():
                            if re.match(pattern, line_text) and section_name not in positions:
                                positions[section_name] = (page_num, line["bbox"][1]) # Y-coordinate of the line
                                break # Move to next line
        return positions

    def _extract_text_between(self, doc: fitz.Document, start_pos: Tuple[int, float], end_pos: Optional[Tuple[int, float]]) -> str:
        """
        Extracts all text between a start position (page, y) and an optional end position.
        """
        text = ""
        start_page, start_y = start_pos
        end_page, end_y = end_pos if end_pos else (len(doc) - 1, float('inf'))

        for page_num in range(start_page, end_page + 1):
            if page_num > len(doc) -1: break
            
            y_start_filter = start_y if page_num == start_page else 0
            y_end_filter = end_y if page_num == end_page else float('inf')

            text += doc[page_num].get_text(clip=fitz.Rect(0, y_start_filter, doc[page_num].rect.width, y_end_filter))
        
        return " ".join(text.replace('\n', ' ').split())


    def _extract_images_from_pdf(self, doc: fitz.Document) -> List[Image.Image]:
        """
        Extracts all images from the document as PIL Image objects.
        """
        images = []
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    images.append(image)
                except Exception as e:
                    logging.warning(f"Could not extract or process image with xref {xref}: {e}")
        return images
    
    def _generate_captions_batch(self, images: List[Image.Image]) -> List[str]:
        """
        Generates captions for a batch of images using the loaded BLIP model.
        """
        if not self.model or not self.processor or not images:
            return []
        
        logging.info(f"Generating captions for a batch of {len(images)} images...")
        prompt = "A scientific figure showing experimental results or a diagram of a model. Describe it concisely."
        prompts = [prompt] * len(images)

        try:
            # Batch process images
            inputs = self.processor(images, text=prompts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=200)
            captions = self.processor.batch_decode(out, skip_special_tokens=True)
            return captions
        except Exception as e:
            logging.error(f"Failed to generate captions in batch: {e}")
            return [f"[Caption generation failed: {e}]"] * len(images)


    def _is_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Checks if two texts are highly similar using Levenshtein distance.
        A higher threshold means they must be more similar.
        """
        if not text1 or not text2:
            return False
        distance = Levenshtein.distance(text1, text2)
        max_len = max(len(text1), len(text2))
        similarity = 1 - (distance / max_len)
        return similarity > threshold

    def _remove_duplicate_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        """
        Cleans up sections by removing content that is highly similar to the abstract or other sections.
        """
        cleaned_sections = {}
        abstract = sections.get('abstract', '')
        
        # First, filter sections that are just copies of the abstract
        for name, content in sections.items():
            if name != 'abstract' and self._is_similar(content, abstract):
                logging.info(f"Section '{name}' is highly similar to abstract, removing content.")
                cleaned_sections[name] = ""
            else:
                cleaned_sections[name] = content
        
        # Then, check for similarity between main sections (e.g., if conclusion just repeats results)
        section_order = ['method', 'result', 'conclusion']
        for i in range(len(section_order)):
            for j in range(i + 1, len(section_order)):
                name1, name2 = section_order[i], section_order[j]
                content1 = cleaned_sections.get(name1, "")
                content2 = cleaned_sections.get(name2, "")
                if content1 and content2 and self._is_similar(content1, content2):
                    logging.info(f"Section '{name2}' is highly similar to '{name1}', removing content.")
                    cleaned_sections[name2] = "" # Keep the earlier section

        cleaned_sections['abstract'] = abstract
        return cleaned_sections


    def parse_paper(self, arxiv_id: str, abstract: str) -> Optional[Dict[str, str]]:
        """
        Orchestrates the parsing of a single paper given its arXiv ID.

        Args:
            arxiv_id (str): The paper's arXiv ID.
            abstract (str): The paper's abstract, passed from the crawler.

        Returns:
            A dictionary containing the cleaned text of the 'method', 'result', and 'conclusion' sections,
            or None if parsing fails.
        """
        pdf_path = None
        try:
            pdf_path = self._download_pdf(arxiv_id)
            if not pdf_path:
                return None

            doc = fitz.open(pdf_path)
            
            # 1. Find all section headers first
            section_positions = self._find_section_positions(doc)
            
            # Define the order of sections for text extraction
            ordered_sections = ['introduction', 'method', 'result', 'conclusion', 'references']
            
            # 2. Extract text for each section based on its start and the next section's start
            extracted_texts = {}
            for i, section_name in enumerate(ordered_sections):
                if section_name in section_positions:
                    start_pos = section_positions[section_name]
                    # Find the start of the next section to define the end boundary
                    next_section_pos = None
                    for next_section_name in ordered_sections[i+1:]:
                        if next_section_name in section_positions:
                            next_section_pos = section_positions[next_section_name]
                            break
                    
                    extracted_texts[section_name] = self._extract_text_between(doc, start_pos, next_section_pos)

            # 3. Handle images and captions
            all_images = self._extract_images_from_pdf(doc)
            all_captions = self._generate_captions_batch(all_images)
            if all_captions:
                # For simplicity, append all captions to the result section if it exists
                if 'result' in extracted_texts and extracted_texts['result']:
                    extracted_texts['result'] += "\n\n[Figure Captions]\n" + "\n".join(all_captions)
                # Or method section as a fallback
                elif 'method' in extracted_texts and extracted_texts['method']:
                    extracted_texts['method'] += "\n\n[Figure Captions]\n" + "\n".join(all_captions)
            
            doc.close()
            
            # 4. Final cleanup
            final_sections = {
                'abstract': abstract,
                'method': extracted_texts.get('method', ''),
                'result': extracted_texts.get('result', ''),
                'conclusion': extracted_texts.get('conclusion', '')
            }

            return self._remove_duplicate_sections(final_sections)

        except Exception as e:
            logging.error(f"An unexpected error occurred while parsing paper {arxiv_id}: {e}")
            return None
        finally:
            # Ensure temporary file is always deleted
            if pdf_path and os.path.exists(pdf_path):
                os.remove(pdf_path)