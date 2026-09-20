import os
import re
import logging
from pathlib import Path
import pandas as pd
from typing import Iterator, Tuple

logger = logging.getLogger(__name__)

def sanitize_identifier(name: str) -> str:
    """
    Sanitizes column and table names for PostgreSQL to ensure NL2SQL compatibility.
    Converts to lowercase, replaces spaces with underscores, and removes special characters.
    """
    name = str(name).lower().strip()
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^a-z0-9_]', '', name)
    if name and name[0].isdigit():
        name = '_' + name
    return name or "unnamed_column"

def process_tabular_files(raw_dir: str) -> Iterator[Tuple[str, pd.DataFrame, bool, str]]:
    """
    Scans the directory for CSV and Excel files, sanitizes their schemas,
    and yields (table_name, dataframe_chunk, is_first_chunk, file_path) as a memory-efficient generator.
    """
    base_path = Path(raw_dir)
    if not base_path.exists():
        logger.warning(f"Directory {raw_dir} does not exist. Skipping tabular ingestion.")
        return

    supported_extensions = {'.csv', '.xlsx', '.xls'}

    for root, _, files in os.walk(base_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext not in supported_extensions:
                continue

            file_path = os.path.join(root, file)
            table_name = sanitize_identifier(Path(file).stem)
            logger.info(f"Extracting tabular file: {file_path} for table '{table_name}'")

            try:
                if ext == '.csv':
                    # Read massive CSVs in memory-safe chunks
                    chunk_iter = pd.read_csv(file_path, chunksize=50000)
                    for i, chunk in enumerate(chunk_iter):
                        chunk.columns = [sanitize_identifier(col) for col in chunk.columns]
                        yield table_name, chunk, (i == 0), file_path
                else:
                    # Excel files are read entirely (pandas doesn't support chunking for Excel natively)
                    df = pd.read_excel(file_path)
                    df.columns = [sanitize_identifier(col) for col in df.columns]
                    yield table_name, df, True, file_path

            except Exception as e:
                logger.error(f"Failed to extract tabular file {file_path}: {e}")
