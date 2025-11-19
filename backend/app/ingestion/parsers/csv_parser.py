"""CSV/TSV parser with robust handling and statistical analysis."""

import csv
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from backend.app.ingestion.parsers.base import ParsedDocument, ParserBase
from backend.app.logging import get_logger
from backend.app.utils.errors import ParsingError

logger = get_logger(__name__)

# Common delimiters to try
DELIMITERS = [",", "\t", ";", "|"]

# Common encodings to try
ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1", "ascii"]


class CSVParser(ParserBase):
    """Parser for CSV and TSV files with statistical analysis."""

    SUPPORTED_EXTENSIONS = {".csv", ".tsv"}

    @classmethod
    def can_parse(cls, file_path: Path, mime_type: str | None = None) -> bool:
        """
        Check if file is a CSV/TSV file.

        Args:
            file_path: Path to the file
            mime_type: Optional MIME type of the file

        Returns:
            True if this parser can handle the file
        """
        ext = file_path.suffix.lower()
        if ext in cls.SUPPORTED_EXTENSIONS:
            return True
        if mime_type and ("csv" in mime_type.lower() or "spreadsheet" in mime_type.lower()):
            return True
        return False

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse CSV/TSV file with statistical analysis.

        Args:
            file_path: Path to CSV/TSV file

        Returns:
            ParsedDocument with extracted data, statistics, and metadata

        Raises:
            ParsingError: If file cannot be parsed
        """
        try:
            logger.info(f"Parsing CSV/TSV file: {file_path}")

            # Try to detect delimiter and encoding
            delimiter = self._detect_delimiter(file_path)
            encoding = self._detect_encoding(file_path)

            logger.debug(f"Detected delimiter: {repr(delimiter)}, encoding: {encoding}")

            # Read the CSV file with pandas
            try:
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    dtype=str,  # Read everything as string first
                    on_bad_lines="skip",  # Skip malformed lines
                    engine="python",  # More lenient parser
                )
            except Exception as e:
                logger.warning(f"Primary read failed: {e}, attempting fallback...")
                # Fallback: try without specifying engine
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    encoding=encoding,
                    dtype=str,
                    on_bad_lines="skip",
                )

            if df.empty:
                logger.warning(f"Empty CSV/TSV file: {file_path}")
                return ParsedDocument(
                    blocks=[],
                    metadata={"empty": True, "delimiter": delimiter, "encoding": encoding},
                )

            # Generate statistics and summary
            blocks = self._create_blocks(df, file_path)
            metadata = self._generate_metadata(df, file_path, delimiter, encoding)

            logger.info(
                f"Successfully parsed CSV/TSV file: {file_path} "
                f"({len(df)} rows, {len(df.columns)} columns)"
            )

            return ParsedDocument(
                blocks=blocks,
                title=file_path.stem,
                metadata=metadata,
            )

        except ParsingError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse CSV/TSV file {file_path}: {e}")
            raise ParsingError(
                f"Failed to parse CSV/TSV file: {e}",
                {"file_path": str(file_path)},
            )

    def _detect_delimiter(self, file_path: Path) -> str:
        """
        Detect the delimiter used in the CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            Detected delimiter string
        """
        try:
            # Read a sample of the file
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(8192)  # 8KB sample

            # Try each delimiter
            for delimiter in DELIMITERS:
                try:
                    sniffer = csv.Sniffer()
                    detected = sniffer.sniff(sample, delimiters=delimiter)
                    return detected.delimiter
                except Exception:
                    continue

            # Default to comma
            return ","
        except Exception as e:
            logger.warning(f"Failed to detect delimiter: {e}, using comma")
            return ","

    def _detect_encoding(self, file_path: Path) -> str:
        """
        Detect the encoding of the CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            Detected encoding string
        """
        # Try each encoding
        for encoding in ENCODINGS:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    f.read(1024)  # Try reading 1KB
                return encoding
            except (UnicodeDecodeError, LookupError):
                continue

        # Default to utf-8 with replace
        logger.warning("Could not detect encoding, defaulting to utf-8 with replace errors")
        return "utf-8"

    def _create_blocks(self, df: pd.DataFrame, file_path: Path) -> List:
        """
        Create ParsedBlocks from the DataFrame.

        Includes: header block, sample block, and statistics block.

        Args:
            df: Pandas DataFrame
            file_path: Original file path

        Returns:
            List of ParsedBlocks
        """
        blocks = []

        # Block 1: Header and column information
        header_text = self._format_header_block(df)
        blocks.append(
            self._create_block(
                text=header_text,
                section="Header",
                metadata={"block_type": "header"},
            )
        )

        # Block 2: Sample of first 100 rows
        sample_text = self._format_sample_block(df)
        blocks.append(
            self._create_block(
                text=sample_text,
                section="Sample Data",
                metadata={"block_type": "sample", "sample_size": min(100, len(df))},
            )
        )

        # Block 3: Statistical summary
        stats_text = self._format_statistics_block(df)
        if stats_text:
            blocks.append(
                self._create_block(
                    text=stats_text,
                    section="Statistics",
                    metadata={"block_type": "statistics"},
                )
            )

        # Block 4: All rows as searchable text (concatenated)
        rows_text = self._format_all_rows_text(df)
        if rows_text:
            blocks.append(
                self._create_block(
                    text=rows_text,
                    section="Full Data",
                    metadata={"block_type": "full_rows"},
                )
            )

        return blocks

    def _format_header_block(self, df: pd.DataFrame) -> str:
        """Format header and column type information."""
        lines = ["Column Information:"]
        lines.append("=" * 60)

        for idx, col in enumerate(df.columns, 1):
            col_name = str(col).strip()
            col_type = self._infer_column_type(df[col])
            non_null_count = df[col].notna().sum()
            null_count = df[col].isna().sum()
            unique_count = df[col].nunique()

            lines.append(f"\n{idx}. {col_name}")
            lines.append(f"   Type: {col_type}")
            lines.append(f"   Non-null: {non_null_count}, Null: {null_count}")
            lines.append(f"   Unique values: {unique_count}")

        return "\n".join(lines)

    def _format_sample_block(self, df: pd.DataFrame) -> str:
        """Format first 100 rows as readable text."""
        lines = ["First 100 Rows:"]
        lines.append("=" * 60)

        # Sample up to 100 rows
        sample_df = df.head(100)

        # Format as table
        for idx, row in sample_df.iterrows():
            lines.append(f"\nRow {idx + 1}:")
            for col, value in row.items():
                col_name = str(col).strip()
                value_str = str(value).strip() if pd.notna(value) else "[NULL]"
                # Truncate very long values
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                lines.append(f"  {col_name}: {value_str}")

        return "\n".join(lines)

    def _format_statistics_block(self, df: pd.DataFrame) -> str:
        """Format statistical summary for numeric columns."""
        lines = ["Statistical Summary:"]
        lines.append("=" * 60)

        stats_generated = False

        for col in df.columns:
            col_name = str(col).strip()

            # Try to convert to numeric for statistics
            try:
                numeric_col = pd.to_numeric(df[col], errors="coerce")
                non_null_numeric = numeric_col.dropna()

                if len(non_null_numeric) > 0:
                    stats_generated = True
                    lines.append(f"\n{col_name} (Numeric):")
                    lines.append(f"  Count: {len(non_null_numeric)}")
                    lines.append(f"  Mean: {non_null_numeric.mean():.4f}")
                    lines.append(f"  Median: {non_null_numeric.median():.4f}")
                    lines.append(f"  Std Dev: {non_null_numeric.std():.4f}")
                    lines.append(f"  Min: {non_null_numeric.min():.4f}")
                    lines.append(f"  Max: {non_null_numeric.max():.4f}")
                    lines.append(f"  Q1: {non_null_numeric.quantile(0.25):.4f}")
                    lines.append(f"  Q3: {non_null_numeric.quantile(0.75):.4f}")
            except Exception:
                pass

        if not stats_generated:
            return ""

        return "\n".join(lines)

    def _format_all_rows_text(self, df: pd.DataFrame) -> str:
        """Format all rows as searchable text (pipe-separated)."""
        lines = []

        # Add header row
        header = " | ".join(str(col).strip() for col in df.columns)
        lines.append(header)
        lines.append("-" * min(len(header), 200))

        # Add data rows
        for idx, row in df.iterrows():
            row_values = []
            for value in row:
                value_str = str(value).strip() if pd.notna(value) else ""
                # Replace newlines in cell values
                value_str = value_str.replace("\n", " ").replace("\r", " ")
                row_values.append(value_str)
            line = " | ".join(row_values)
            lines.append(line)

        return "\n".join(lines)

    def _generate_metadata(
        self, df: pd.DataFrame, file_path: Path, delimiter: str, encoding: str
    ) -> Dict[str, Any]:
        """Generate metadata dictionary for the document."""
        metadata = {
            "file_size": file_path.stat().st_size,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "delimiter": delimiter,
            "encoding": encoding,
        }

        # Add column types
        column_types = {}
        for col in df.columns:
            col_name = str(col).strip()
            column_types[col_name] = self._infer_column_type(df[col])
        metadata["column_types"] = column_types

        # Add memory usage estimate
        try:
            metadata["memory_usage_bytes"] = df.memory_usage(deep=True).sum()
        except Exception:
            pass

        # Add null counts
        null_counts = {}
        for col in df.columns:
            col_name = str(col).strip()
            null_counts[col_name] = int(df[col].isna().sum())
        metadata["null_counts"] = null_counts

        return metadata

    def _infer_column_type(self, series: pd.Series) -> str:
        """
        Infer the data type of a column.

        Args:
            series: Pandas Series

        Returns:
            Type string (numeric, boolean, datetime, categorical, text)
        """
        non_null = series.dropna()

        if len(non_null) == 0:
            return "unknown"

        # Try numeric
        try:
            pd.to_numeric(non_null)
            if all(non_null.astype(str).str.match(r"^[01]$|^(true|false|yes|no)$", na=False)):
                return "boolean"
            return "numeric"
        except (ValueError, TypeError):
            pass

        # Try datetime
        try:
            pd.to_datetime(non_null)
            return "datetime"
        except (ValueError, TypeError):
            pass

        # Try boolean
        bool_values = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}
        if all(str(v).lower() in bool_values for v in non_null.astype(str)):
            return "boolean"

        # Default to text
        return "text"
