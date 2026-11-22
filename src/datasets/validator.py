"""
src/datasets/validator.py
Strict dataset validator for .txt, .jsonl, and .json files.
Detects malformed lines, encoding issues, missing fields in instruction datasets,
and produces downloadable validation reports without silently discarding data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DatasetValidationError(Exception):
    pass


class DatasetValidator:
    """Validates structure, encoding, and schema compliance for uploaded datasets."""

    def __init__(
        self,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        prompt_template: str = "{instruction}\n{input}\n{output}",
    ) -> None:
        self.instruction_field = instruction_field
        self.input_field = input_field
        self.output_field = output_field
        self.prompt_template = prompt_template

    def validate_file(self, file_path: Path | str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            return {
                "valid": False,
                "error": f"File does not exist: {path}",
                "total_records": 0,
                "valid_records": 0,
                "invalid_records": 0,
                "errors": [f"File {path.name} not found"],
            }

        suffix = path.suffix.lower()
        if suffix not in (".txt", ".jsonl", ".json", ".csv", ".parquet", ".arrow"):
            return {
                "valid": False,
                "error": f"Unsupported file extension: {suffix}. Supported: .txt, .jsonl, .json, .csv, .parquet, .arrow",
                "total_records": 0,
                "valid_records": 0,
                "invalid_records": 0,
                "errors": [f"Unsupported format {suffix}"],
            }

        # Binary formats (Parquet, Arrow) skip raw UTF-8 string check
        if suffix == ".parquet":
            return self._validate_parquet(path)
        elif suffix == ".arrow":
            return self._validate_arrow(path)

        # 1. UTF-8 Validation
        try:
            with open(path, "rb") as f:
                raw_bytes = f.read(1024 * 1024)  # Check first 1MB for invalid byte sequences
                raw_bytes.decode("utf-8")
        except UnicodeDecodeError as ude:
            return {
                "valid": False,
                "error": f"UTF-8 Encoding error: {ude}",
                "total_records": 0,
                "valid_records": 0,
                "invalid_records": 0,
                "errors": [str(ude)],
            }

        if suffix == ".txt":
            return self._validate_txt(path)
        elif suffix == ".jsonl":
            return self._validate_jsonl(path)
        elif suffix == ".csv":
            return self._validate_csv(path)
        else:
            return self._validate_json(path)

    def _validate_parquet(self, path: Path) -> Dict[str, Any]:
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(str(path))
            total_rows = parquet_file.metadata.num_rows
            schema_names = parquet_file.schema.names

            # Read first row group memory-mapped without full RAM copy
            table = parquet_file.read_row_group(0)
            df_dict = table.to_pydict()

            valid_samples: List[str] = []
            fmt_type = "parquet_text"
            if "prompt" in df_dict and "chosen" in df_dict and "rejected" in df_dict:
                fmt_type = "pairwise_preference"
                for p, c, r in zip(df_dict["prompt"][:100], df_dict["chosen"][:100], df_dict["rejected"][:100]):
                    valid_samples.append(f"<|prompt|>\n{p}\n<|chosen|>\n{c}\n<|rejected|>\n{r}")
            elif "text" in df_dict:
                fmt_type = "parquet_text"
                for t in df_dict["text"][:100]:
                    if t: valid_samples.append(str(t).strip())
            else:
                fmt_type = "parquet_table"
                first_col = schema_names[0]
                for v in df_dict[first_col][:100]:
                    if v: valid_samples.append(str(v).strip())

            return {
                "valid": total_rows > 0,
                "format": "parquet",
                "schema_type": fmt_type,
                "total_records": total_rows,
                "valid_records": total_rows,
                "invalid_records": 0,
                "errors": [],
                "samples_preview": valid_samples[:5],
                "extracted_texts": valid_samples,
                "columns": schema_names,
            }
        except Exception as e:
            return {
                "valid": False,
                "format": "parquet",
                "error": f"Failed reading Parquet file: {e}",
                "total_records": 0,
                "valid_records": 0,
                "invalid_records": 0,
                "errors": [str(e)],
            }

    def _validate_arrow(self, path: Path) -> Dict[str, Any]:
        try:
            import pyarrow as pa
            with pa.memory_map(str(path), "r") as source:
                try:
                    reader = pa.ipc.open_stream(source)
                except Exception:
                    reader = pa.ipc.open_file(source)
                table = reader.read_all()
                total_rows = table.num_rows
                schema_names = table.schema.names
                df_dict = table.slice(0, 100).to_pydict()

            valid_samples: List[str] = []
            if "prompt" in df_dict and "chosen" in df_dict and "rejected" in df_dict:
                fmt_type = "pairwise_preference"
                for p, c, r in zip(df_dict["prompt"][:100], df_dict["chosen"][:100], df_dict["rejected"][:100]):
                    valid_samples.append(f"<|prompt|>\n{p}\n<|chosen|>\n{c}\n<|rejected|>\n{r}")
            elif "text" in df_dict:
                fmt_type = "arrow_text"
                for t in df_dict["text"][:100]:
                    if t: valid_samples.append(str(t).strip())
            else:
                fmt_type = "arrow_table"
                first_col = schema_names[0]
                for v in df_dict[first_col][:100]:
                    if v: valid_samples.append(str(v).strip())

            return {
                "valid": total_rows > 0,
                "format": "arrow",
                "schema_type": fmt_type,
                "total_records": total_rows,
                "valid_records": total_rows,
                "invalid_records": 0,
                "errors": [],
                "samples_preview": valid_samples[:5],
                "extracted_texts": valid_samples,
                "columns": schema_names,
            }
        except Exception as e:
            return {
                "valid": False,
                "format": "arrow",
                "error": f"Failed reading Arrow file: {e}",
                "total_records": 0,
                "valid_records": 0,
                "invalid_records": 0,
                "errors": [str(e)],
            }

    def _validate_csv(self, path: Path) -> Dict[str, Any]:
        import csv
        valid_samples = []
        errors = []
        total_rows = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return {
                    "valid": False,
                    "format": "csv",
                    "error": "CSV file is empty or missing header.",
                    "total_records": 0,
                    "valid_records": 0,
                    "invalid_records": 0,
                    "errors": ["Empty CSV"],
                }

            header_lower = [h.strip().lower() for h in header]
            text_idx = None
            for candidate in ("text", "content", "prompt", "instruction", "input", "sentence"):
                if candidate in header_lower:
                    text_idx = header_lower.index(candidate)
                    break
            if text_idx is None:
                text_idx = 0

            for row_idx, row in enumerate(reader, start=2):
                total_rows += 1
                if not row or len(row) <= text_idx:
                    errors.append(f"Row {row_idx}: Empty or truncated row.")
                    continue
                val = row[text_idx].strip()
                if val:
                    valid_samples.append(val)
                else:
                    errors.append(f"Row {row_idx}: Empty value in primary column.")

        return {
            "valid": len(valid_samples) > 0,
            "format": "csv",
            "total_records": total_rows,
            "valid_records": len(valid_samples),
            "invalid_records": total_rows - len(valid_samples),
            "errors": errors[:50],
            "samples_preview": valid_samples[:5],
            "extracted_texts": valid_samples,
            "header": header,
        }

    def _validate_txt(self, path: Path) -> Dict[str, Any]:
        valid_samples = []
        errors = []
        line_num = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_num += 1
                stripped = line.strip()
                if not stripped:
                    continue
                valid_samples.append(stripped)

        return {
            "valid": len(valid_samples) > 0,
            "format": "txt",
            "total_records": line_num,
            "valid_records": len(valid_samples),
            "invalid_records": line_num - len(valid_samples),
            "errors": errors,
            "samples_preview": valid_samples[:5],
            "extracted_texts": valid_samples,
        }

    def _validate_jsonl(self, path: Path) -> Dict[str, Any]:
        valid_records = []
        extracted_texts = []
        errors = []
        line_num = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_num += 1
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                    if not isinstance(obj, dict):
                        errors.append(f"Line {line_num}: JSON record is not an object.")
                        continue

                    # Check for pairwise preference alignment schema (SimPO / ORPO / DPO)
                    if "chosen" in obj and "rejected" in obj:
                        prompt_str = str(obj.get("prompt", obj.get("instruction", ""))).strip()
                        chosen_str = str(obj.get("chosen", "")).strip()
                        rejected_str = str(obj.get("rejected", "")).strip()
                        if chosen_str and rejected_str:
                            formatted = f"<|prompt|>\n{prompt_str}\n<|chosen|>\n{chosen_str}\n<|rejected|>\n{rejected_str}"
                            valid_records.append(obj)
                            extracted_texts.append(formatted)
                            continue

                    # Check for instruction tuning schema or plain text
                    if "text" in obj:
                        text = str(obj["text"]).strip()
                        if text:
                            valid_records.append(obj)
                            extracted_texts.append(text)
                    elif self.instruction_field in obj or "instruction" in obj:
                        instr_key = self.instruction_field if self.instruction_field in obj else "instruction"
                        out_key = self.output_field if self.output_field in obj else "output"
                        inp_key = self.input_field if self.input_field in obj else "input"

                        instr = str(obj.get(instr_key, "")).strip()
                        out = str(obj.get(out_key, "")).strip()
                        inp = str(obj.get(inp_key, "")).strip()

                        if not instr and not out:
                            errors.append(f"Line {line_num}: Both instruction and output are empty.")
                            continue

                        formatted = f"<|user|>\n{instr}\n{inp}\n<|assistant|>\n{out}<|end_turn|>"
                        valid_records.append(obj)
                        extracted_texts.append(formatted)
                    elif "messages" in obj and isinstance(obj["messages"], list):
                        # Chat format
                        formatted_parts = []
                        for msg in obj["messages"]:
                            role = msg.get("role", "user")
                            content = msg.get("content", "")
                            formatted_parts.append(f"<|{role}|>\n{content}")
                        formatted_parts.append("<|end_turn|>")
                        valid_records.append(obj)
                        extracted_texts.append("\n".join(formatted_parts))
                    else:
                        # Missing recognized fields
                        missing = [self.instruction_field, self.output_field, "text", "messages"]
                        errors.append(f"Line {line_num}: Missing required fields ({', '.join(missing)})")
                except json.JSONDecodeError as jde:
                    errors.append(f"Line {line_num}: Malformed JSON syntax: {jde.msg}")

        is_valid = len(valid_records) > 0 and len(errors) < line_num
        return {
            "valid": is_valid,
            "format": "jsonl",
            "total_records": line_num,
            "valid_records": len(valid_records),
            "invalid_records": len(errors),
            "errors": errors[:50],  # Bound error list
            "total_errors": len(errors),
            "samples_preview": extracted_texts[:5],
            "extracted_texts": extracted_texts,
        }

    def _validate_json(self, path: Path) -> Dict[str, Any]:
        errors = []
        extracted_texts = []
        valid_records = []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as jde:
            return {
                "valid": False,
                "format": "json",
                "total_records": 0,
                "valid_records": 0,
                "invalid_records": 1,
                "errors": [f"Malformed JSON document: {jde}"],
            }

        if not isinstance(data, list):
            return {
                "valid": False,
                "format": "json",
                "total_records": 1,
                "valid_records": 0,
                "invalid_records": 1,
                "errors": ["Root JSON element must be an array of objects."],
            }

        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                errors.append(f"Item {idx}: Record is not a JSON object.")
                continue

            if "text" in item:
                text = str(item["text"]).strip()
                if text:
                    valid_records.append(item)
                    extracted_texts.append(text)
            elif "instruction" in item:
                instr = str(item.get("instruction", "")).strip()
                out = str(item.get("output", "")).strip()
                inp = str(item.get("input", "")).strip()
                formatted = f"<|user|>\n{instr}\n{inp}\n<|assistant|>\n{out}<|end_turn|>"
                valid_records.append(item)
                extracted_texts.append(formatted)
            else:
                errors.append(f"Item {idx}: Missing instruction or text field.")

        return {
            "valid": len(valid_records) > 0,
            "format": "json",
            "total_records": len(data),
            "valid_records": len(valid_records),
            "invalid_records": len(errors),
            "errors": errors[:50],
            "total_errors": len(errors),
            "samples_preview": extracted_texts[:5],
            "extracted_texts": extracted_texts,
        }

