import os
import logging
from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter
from llama_index.core import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_markdown(raw_dir: str) -> List[Document]:
    """
    Scans a directory for supported documents, uses Docling to extract layout-aware markdown,
    and returns a list of LlamaIndex Document objects with basic metadata.
    """
    converter = DocumentConverter()
    docs = []
    
    base_path = Path(raw_dir)
    if not base_path.exists():
        logger.warning(f"Directory {raw_dir} does not exist. Please create it and add files.")
        return docs

    supported_extensions = ('.pdf', '.docx', '.pptx', '.xlsx', '.html', '.md', '.csv')

    for root, _, files in os.walk(base_path):
        for file in files:
            if not file.lower().endswith(supported_extensions):
                continue
                
            file_path = os.path.join(root, file)
            logger.info(f"Docling is parsing: {file_path}")
            
            try:
                # Use Docling to convert the document (OCR, layout parsing, table extraction)
                conversion_result = converter.convert(file_path)
                markdown_content = conversion_result.document.export_to_markdown()
                
                # Wrap in a LlamaIndex Document
                doc = Document(
                    text=markdown_content,
                    metadata={
                        "file_name": file,
                        "file_path": file_path,
                        "source_type": "docling_markdown",
                        "page": "Unknown" 
                    },
                    excluded_embed_metadata_keys=["file_path", "source_type", "page"]
                )
                docs.append(doc)
                
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                
    logger.info(f"Successfully extracted {len(docs)} documents from {raw_dir}")
    return docs
