import os
import logging
from pathlib import Path
from typing import Iterator
from docling.document_converter import DocumentConverter
from llama_index.core import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_markdown(raw_dir: str) -> Iterator[Document]:
    """
    Scans a directory for supported documents, uses Docling to extract layout-aware markdown,
    and yields LlamaIndex Document objects with basic metadata one by one.
    """
    converter = DocumentConverter()
    
    base_path = Path(raw_dir)
    if not base_path.exists():
        logger.warning(f"Directory {raw_dir} does not exist. Please create it and add files.")
        return

    supported_extensions = ('.pdf', '.docx', '.pptx', '.xlsx', '.html', '.md', '.csv')
    docs_yielded = 0

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
                
                raw_metadata = {}
                docling_meta = getattr(conversion_result.document, "metadata", None)
                
                if docling_meta:
                    if hasattr(docling_meta, "model_dump"):
                        raw_metadata = docling_meta.model_dump(exclude_none=True)
                    elif hasattr(docling_meta, "dict"):
                        raw_metadata = docling_meta.dict(exclude_none=True)
                    elif isinstance(docling_meta, dict):
                        raw_metadata = docling_meta.copy()
                    else:
                        raw_metadata = vars(docling_meta)

                cleaned_metadata = {}
                for key, value in raw_metadata.items():
                    if value is not None:
                        if isinstance(value, (str, int, float, bool)):
                            cleaned_metadata[key] = value
                        else:
                            cleaned_metadata[key] = str(value)

                if "title" not in cleaned_metadata:
                    try:
                        for item in conversion_result.document.iterate_items():
                            label = getattr(item, "label", None)
                            label_name = getattr(label, "name", str(label)).upper()
                            if "TITLE" in label_name:
                                cleaned_metadata["title"] = getattr(item, "text", "")
                                break
                    except Exception as e:
                        logger.debug(f"Title fallback failed: {e}")

                if "file_name" not in cleaned_metadata and "filename" not in cleaned_metadata:
                    cleaned_metadata["file_name"] = file
                if "file_path" not in cleaned_metadata and "filepath" not in cleaned_metadata:
                    cleaned_metadata["file_path"] = file_path
                
                cleaned_metadata["source_type"] = "docling_markdown"
                if "page" not in cleaned_metadata:
                    cleaned_metadata["page"] = "Unknown"

                keys_to_exclude_from_embed = [
                    k for k in cleaned_metadata.keys() if k.lower() not in [
                        "title", "author", "file_name", "filename"]
                ]

                keys_to_exclude_from_llm = [
                    k for k in cleaned_metadata.keys() if k.lower() not in [
                        "title", "author", "file_name", "filename", "page"
                    ]
                ]

                # Wrap in a LlamaIndex Document
                doc = Document(
                    text=markdown_content,
                    metadata=cleaned_metadata,
                    excluded_embed_metadata_keys=keys_to_exclude_from_embed,
                    excluded_llm_metadata_keys=keys_to_exclude_from_llm
                )
                yield doc
                docs_yielded += 1
                
            except Exception as e:
                logger.error(f"Failed to parse {file_path}: {e}")
                
    logger.info(f"Successfully extracted {docs_yielded} documents from {raw_dir}")
