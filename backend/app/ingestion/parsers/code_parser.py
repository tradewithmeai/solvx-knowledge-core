"""Code file parser for various programming languages."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pygments import lex
from pygments.lexers import get_lexer_by_name, get_lexer_for_filename
from pygments.token import Comment, Keyword, Name, String, Text, Token

from backend.app.ingestion.parsers.base import ParsedDocument, ParserBase
from backend.app.logging import get_logger
from backend.app.utils.errors import ParsingError
from backend.app.utils.io import read_file_safe

logger = get_logger(__name__)


class CodeParser(ParserBase):
    """Parser for source code files."""

    SUPPORTED_EXTENSIONS = {
        # Python
        ".py",
        ".pyw",
        # JavaScript/TypeScript
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".mjs",
        ".cjs",
        # Java
        ".java",
        # Go
        ".go",
        # Rust
        ".rs",
        # C/C++
        ".c",
        ".cpp",
        ".cc",
        ".cxx",
        ".h",
        ".hpp",
        ".hh",
        ".hxx",
        # C#
        ".cs",
        # Ruby
        ".rb",
        # PHP
        ".php",
        # Swift
        ".swift",
        # Kotlin
        ".kt",
        ".kts",
        # Scala
        ".scala",
        # R
        ".r",
        # Shell
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        # SQL
        ".sql",
    }

    LANGUAGE_MAP = {
        ".py": "python",
        ".pyw": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".hh": "cpp",
        ".hxx": "cpp",
        ".cs": "csharp",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".fish": "fish",
        ".sql": "sql",
    }

    @classmethod
    def can_parse(cls, file_path: Path, mime_type: str | None = None) -> bool:
        """Check if file is a supported code file."""
        return file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse code file and extract code structure.

        Args:
            file_path: Path to code file

        Returns:
            ParsedDocument with code blocks and metadata

        Raises:
            ParsingError: If file cannot be parsed
        """
        try:
            logger.info(f"Parsing code file: {file_path}")

            # Read file content
            content = read_file_safe(file_path)

            if not content or not content.strip():
                logger.warning(f"Empty code file: {file_path}")
                return ParsedDocument(blocks=[], metadata={"empty": True})

            # Detect language
            language = self._detect_language(file_path)
            logger.debug(f"Detected language: {language}")

            # Extract code blocks
            blocks = self._extract_blocks(content, language, file_path)

            # Create document metadata
            lines = content.split("\n")
            metadata = {
                "file_size": file_path.stat().st_size,
                "line_count": len(lines),
                "language": language,
                "block_count": len(blocks),
            }

            logger.info(f"Successfully parsed code file: {file_path} ({len(blocks)} blocks extracted)")

            return ParsedDocument(
                blocks=blocks,
                title=file_path.stem,
                language=language,
                metadata=metadata,
            )

        except ParsingError:
            raise
        except Exception as e:
            logger.error(f"Failed to parse code file {file_path}: {e}")
            raise ParsingError(f"Failed to parse code file: {e}", {"file_path": str(file_path)})

    def _detect_language(self, file_path: Path) -> str:
        """
        Detect programming language from file extension.

        Args:
            file_path: Path to code file

        Returns:
            Language name string
        """
        ext = file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(ext, "text")

    def _extract_blocks(self, content: str, language: str, file_path: Path) -> List:
        """
        Extract code blocks from content.

        Args:
            content: File content
            language: Programming language
            file_path: Path to file

        Returns:
            List of ParsedBlock objects
        """
        blocks = []
        offset = 0

        try:
            # Try to get lexer for the language
            try:
                lexer = get_lexer_by_name(language)
            except Exception:
                try:
                    lexer = get_lexer_for_filename(str(file_path))
                except Exception:
                    # Fallback to plain text
                    logger.debug(f"Could not find lexer for {language}, using text")
                    return self._extract_text_blocks(content)

            # Extract top-level definitions
            definitions = self._extract_definitions(content, language)

            if definitions:
                for def_info in definitions:
                    block = self._create_block(
                        text=def_info["signature"],
                        section=def_info["type"],
                        offset_start=def_info["offset_start"],
                        offset_end=def_info["offset_end"],
                        name=def_info["name"],
                        type=def_info["type"],
                        docstring=def_info.get("docstring"),
                        comment=def_info.get("comment"),
                        has_body=def_info.get("has_body", False),
                    )
                    blocks.append(block)
            else:
                # Fall back to creating one block with entire content
                blocks = self._extract_text_blocks(content)

        except Exception as e:
            logger.warning(f"Error extracting blocks for {language}: {e}")
            blocks = self._extract_text_blocks(content)

        return blocks

    def _extract_definitions(self, content: str, language: str) -> List[Dict[str, Any]]:
        """
        Extract top-level function and class definitions.

        Args:
            content: File content
            language: Programming language

        Returns:
            List of definition info dictionaries
        """
        definitions = []

        try:
            if language in ("python",):
                definitions = self._extract_python_definitions(content)
            elif language in ("javascript", "typescript"):
                definitions = self._extract_js_definitions(content)
            elif language in ("java",):
                definitions = self._extract_java_definitions(content)
            elif language in ("go",):
                definitions = self._extract_go_definitions(content)
            elif language in ("rust",):
                definitions = self._extract_rust_definitions(content)
            elif language in ("cpp", "c"):
                definitions = self._extract_cpp_definitions(content)
            else:
                # Generic extraction for other languages
                definitions = self._extract_generic_definitions(content)

        except Exception as e:
            logger.warning(f"Error extracting definitions for {language}: {e}")
            definitions = []

        return definitions

    def _extract_python_definitions(self, content: str) -> List[Dict[str, Any]]:
        """Extract Python function and class definitions."""
        definitions = []
        lines = content.split("\n")

        # Pattern for class and function definitions
        class_pattern = r"^class\s+(\w+)\s*(\([^)]*\))?:"
        func_pattern = r"^def\s+(\w+)\s*\((.*?)\)\s*(?:->\s*[^:]+)?:"

        for i, line in enumerate(lines):
            # Skip if line is indented (not top-level)
            if line and line[0] in (" ", "\t"):
                continue

            # Check for class definition
            class_match = re.match(class_pattern, line)
            if class_match:
                name = class_match.group(1)
                bases = class_match.group(2) or ""
                signature = f"class {name}{bases}:"
                docstring = self._extract_docstring(lines, i + 1)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": "class",
                    "signature": signature,
                    "docstring": docstring,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })
                continue

            # Check for function definition
            func_match = re.match(func_pattern, line)
            if func_match:
                name = func_match.group(1)
                args = func_match.group(2)
                signature = f"def {name}({args}):"
                docstring = self._extract_docstring(lines, i + 1)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": "function",
                    "signature": signature,
                    "docstring": docstring,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })

        return definitions

    def _extract_js_definitions(self, content: str) -> List[Dict[str, Any]]:
        """Extract JavaScript/TypeScript function and class definitions."""
        definitions = []
        lines = content.split("\n")

        # Patterns for JS/TS definitions
        class_pattern = r"^(export\s+)?(class|interface|type)\s+(\w+)"
        func_pattern = r"^(export\s+)?(async\s+)?(function\s+)?(\w+)\s*\(([^)]*)\)"
        arrow_pattern = r"^(export\s+)?const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>"

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            # Check for class/interface definition
            class_match = re.match(class_pattern, stripped)
            if class_match:
                keyword = class_match.group(2)
                name = class_match.group(3)
                signature = f"{keyword} {name}"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": keyword,
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })
                continue

            # Check for function declaration
            func_match = re.match(func_pattern, stripped)
            if func_match:
                name = func_match.group(4)
                args = func_match.group(5)
                signature = f"{name}({args})"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": "function",
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })
                continue

            # Check for arrow function
            arrow_match = re.match(arrow_pattern, stripped)
            if arrow_match:
                name = arrow_match.group(2)
                args = arrow_match.group(3)
                signature = f"const {name} = ({args}) =>"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": "arrow_function",
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })

        return definitions

    def _extract_java_definitions(self, content: str) -> List[Dict[str, Any]]:
        """Extract Java class and method definitions."""
        definitions = []
        lines = content.split("\n")

        class_pattern = r"(public\s+)?(class|interface|enum)\s+(\w+)"
        method_pattern = r"(public|private|protected)?\s*(static\s+)?(.*?)\s+(\w+)\s*\(([^)]*)\)"

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            # Check for class definition
            class_match = re.search(class_pattern, stripped)
            if class_match:
                keyword = class_match.group(2)
                name = class_match.group(3)
                signature = f"{keyword} {name}"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": keyword,
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })

        return definitions

    def _extract_go_definitions(self, content: str) -> List[Dict[str, Any]]:
        """Extract Go function and type definitions."""
        definitions = []
        lines = content.split("\n")

        func_pattern = r"^func\s+(\([^)]*\))?\s*(\w+)\s*\(([^)]*)\)"
        type_pattern = r"^type\s+(\w+)\s+(struct|interface)"

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            # Check for function definition
            func_match = re.match(func_pattern, stripped)
            if func_match:
                receiver = func_match.group(1) or ""
                name = func_match.group(2)
                args = func_match.group(3)
                signature = f"func {receiver}{name}({args})"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": "function",
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })
                continue

            # Check for type definition
            type_match = re.match(type_pattern, stripped)
            if type_match:
                name = type_match.group(1)
                kind = type_match.group(2)
                signature = f"type {name} {kind}"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": kind,
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })

        return definitions

    def _extract_rust_definitions(self, content: str) -> List[Dict[str, Any]]:
        """Extract Rust function and type definitions."""
        definitions = []
        lines = content.split("\n")

        func_pattern = r"^(pub\s+)?(async\s+)?fn\s+(\w+)\s*\(([^)]*)\)"
        struct_pattern = r"^(pub\s+)?struct\s+(\w+)"
        trait_pattern = r"^(pub\s+)?trait\s+(\w+)"
        enum_pattern = r"^(pub\s+)?enum\s+(\w+)"

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            # Check for function definition
            func_match = re.match(func_pattern, stripped)
            if func_match:
                name = func_match.group(3)
                args = func_match.group(4)
                signature = f"fn {name}({args})"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": "function",
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })
                continue

            # Check for struct definition
            struct_match = re.match(struct_pattern, stripped)
            if struct_match:
                name = struct_match.group(2)
                signature = f"struct {name}"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": "struct",
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })
                continue

            # Check for trait definition
            trait_match = re.match(trait_pattern, stripped)
            if trait_match:
                name = trait_match.group(2)
                signature = f"trait {name}"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": "trait",
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })
                continue

            # Check for enum definition
            enum_match = re.match(enum_pattern, stripped)
            if enum_match:
                name = enum_match.group(2)
                signature = f"enum {name}"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": "enum",
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })

        return definitions

    def _extract_cpp_definitions(self, content: str) -> List[Dict[str, Any]]:
        """Extract C/C++ function and class definitions."""
        definitions = []
        lines = content.split("\n")

        class_pattern = r"^(class|struct)\s+(\w+)"
        func_pattern = r"^(void|int|bool|char|float|double|auto|\w+\*?)\s+(\w+)\s*\(([^)]*)\)"

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            # Check for class/struct definition
            class_match = re.match(class_pattern, stripped)
            if class_match:
                keyword = class_match.group(1)
                name = class_match.group(2)
                signature = f"{keyword} {name}"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": keyword,
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })
                continue

            # Check for function definition
            func_match = re.match(func_pattern, stripped)
            if func_match:
                return_type = func_match.group(1)
                name = func_match.group(2)
                args = func_match.group(3)
                signature = f"{return_type} {name}({args})"
                comment = self._extract_line_comment(lines, i)
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": "function",
                    "signature": signature,
                    "comment": comment,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })

        return definitions

    def _extract_generic_definitions(self, content: str) -> List[Dict[str, Any]]:
        """Extract generic function/class definitions for unknown languages."""
        definitions = []
        lines = content.split("\n")

        # Simple generic patterns
        def_pattern = r"^(def|function|func)\s+(\w+)\s*\("
        class_pattern = r"^(class|type|struct|interface)\s+(\w+)"

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            # Check for function-like definition
            def_match = re.match(def_pattern, stripped)
            if def_match:
                keyword = def_match.group(1)
                name = def_match.group(2)
                signature = f"{keyword} {name}"
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": keyword,
                    "signature": signature,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })
                continue

            # Check for class-like definition
            class_match = re.match(class_pattern, stripped)
            if class_match:
                keyword = class_match.group(1)
                name = class_match.group(2)
                signature = f"{keyword} {name}"
                offset_start = self._get_line_offset(content, i)
                offset_end = offset_start + len(line.encode("utf-8"))

                definitions.append({
                    "name": name,
                    "type": keyword,
                    "signature": signature,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "has_body": True,
                })

        return definitions

    def _extract_docstring(self, lines: List[str], start_idx: int) -> Optional[str]:
        """
        Extract docstring following a definition.

        Args:
            lines: List of lines in file
            start_idx: Index to start searching from

        Returns:
            Docstring content or None
        """
        if start_idx >= len(lines):
            return None

        # Python docstring patterns
        docstring = None
        line = lines[start_idx].strip()

        # Check for triple-quoted string
        if line.startswith('"""') or line.startswith("'''"):
            quote_type = line[:3]
            if line.count(quote_type) >= 2:
                # Single-line docstring
                content = line[3:line.rfind(quote_type)]
                docstring = content.strip()
            else:
                # Multi-line docstring
                doc_lines = [line[3:]]
                for i in range(start_idx + 1, min(start_idx + 20, len(lines))):
                    doc_line = lines[i]
                    if quote_type in doc_line:
                        doc_lines.append(doc_line[:doc_line.find(quote_type)])
                        docstring = "\n".join(doc_lines).strip()
                        break
                    doc_lines.append(doc_line)

        return docstring

    def _extract_line_comment(self, lines: List[str], idx: int) -> Optional[str]:
        """
        Extract single-line comment from a line.

        Args:
            lines: List of lines in file
            idx: Line index

        Returns:
            Comment content or None
        """
        if idx < 0 or idx >= len(lines):
            return None

        line = lines[idx]

        # Find comment markers
        for marker in ("//", "#", "--", "/*"):
            if marker in line:
                comment_start = line.find(marker)
                comment = line[comment_start:].strip()
                if comment.startswith("/*"):
                    comment = comment[2:]
                    if "*/" in comment:
                        comment = comment[:comment.find("*/")]
                else:
                    comment = comment[len(marker):] if len(comment) > len(marker) else ""

                return comment.strip() if comment.strip() else None

        return None

    def _extract_text_blocks(self, content: str) -> List:
        """
        Fallback: extract text blocks from content.

        Args:
            content: File content

        Returns:
            List of ParsedBlock objects
        """
        lines = content.split("\n")
        blocks = []

        # Split into chunks of ~50 lines
        chunk_size = 50
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i + chunk_size]
            chunk_text = "\n".join(chunk_lines)

            offset_start = self._get_line_offset(content, i)
            offset_end = offset_start + len(chunk_text.encode("utf-8"))

            block = self._create_block(
                text=chunk_text,
                offset_start=offset_start,
                offset_end=offset_end,
                section=f"lines_{i}-{i + len(chunk_lines) - 1}",
            )
            blocks.append(block)

        return blocks

    def _get_line_offset(self, content: str, line_num: int) -> int:
        """
        Calculate byte offset for a line number.

        Args:
            content: File content
            line_num: Line number (0-indexed)

        Returns:
            Byte offset
        """
        lines = content.split("\n")
        offset = 0

        for i in range(min(line_num, len(lines))):
            offset += len(lines[i].encode("utf-8")) + 1  # +1 for newline

        return offset
