from __future__ import annotations

import ast
import unittest
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "scruffy"


class Python310CompatibilityTests(unittest.TestCase):
    def test_production_modules_parse_as_python_310_without_datetime_utc(self) -> None:
        failures: list[str] = []
        for source_file in sorted(SOURCE_ROOT.glob("*.py")):
            tree = ast.parse(
                source_file.read_text(encoding="utf-8"),
                filename=str(source_file),
                feature_version=(3, 10),
            )
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "datetime"
                    and any(alias.name == "UTC" for alias in node.names)
                ):
                    failures.append(f"{source_file.name}:{node.lineno}: from datetime import UTC")
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "UTC"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "datetime"
                ):
                    failures.append(f"{source_file.name}:{node.lineno}: datetime.UTC")
        self.assertEqual([], failures)
