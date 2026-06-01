"""Reward validator for ASE-MTAGE Phase 2.

The validator performs static AST checks and a small runtime smoke test. It is
intentionally conservative: candidate rewards must expose a `compute_reward`
function with the ASE-MTAGE signature and return `(float, dict)`.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from ase_mtage.utils.io import save_json


REQUIRED_SIGNATURE = ["obs", "action", "next_obs", "terminated", "truncated", "info"]
FORBIDDEN_IMPORTS = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "pathlib",
    "shutil",
    "gym",
    "gymnasium",
}
FORBIDDEN_CALL_NAMES = {"open", "exec", "eval", "compile", "__import__"}
FORBIDDEN_ATTR_NAMES = {"step", "reward", "unwrapped"}


@dataclass(slots=True)
class ValidationResult:
    candidate_id: str
    reward_path: str
    syntax_ok: bool = False
    signature_ok: bool = False
    forbidden_api_used: bool = False
    runtime_smoke_test_ok: bool = False
    component_dict_ok: bool = False
    finite_output_ok: bool = False
    valid: bool = False
    warnings: list[str] | None = None
    errors: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reward_path": self.reward_path,
            "syntax_ok": self.syntax_ok,
            "signature_ok": self.signature_ok,
            "forbidden_api_used": self.forbidden_api_used,
            "runtime_smoke_test_ok": self.runtime_smoke_test_ok,
            "component_dict_ok": self.component_dict_ok,
            "finite_output_ok": self.finite_output_ok,
            "valid": self.valid,
            "warnings": self.warnings or [],
            "errors": self.errors or [],
        }


class RewardValidator:
    """Validate generated reward source files."""

    def validate_file(
        self,
        reward_path: str | Path,
        *,
        candidate_id: str | None = None,
        report_path: str | Path | None = None,
    ) -> ValidationResult:
        path = Path(reward_path)
        result = ValidationResult(
            candidate_id=candidate_id or path.parent.name,
            reward_path=str(path),
            warnings=[],
            errors=[],
        )

        if not path.exists():
            result.errors.append(f"Reward file not found: {path}")
            self._finish(result, report_path)
            return result

        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
            result.syntax_ok = True
        except SyntaxError as exc:
            result.errors.append(f"SyntaxError: {exc}")
            self._finish(result, report_path)
            return result

        result.signature_ok = self._check_signature(tree, result)
        result.forbidden_api_used = self._has_forbidden_api(tree, result)

        if result.syntax_ok and result.signature_ok and not result.forbidden_api_used:
            self._runtime_smoke_test(path, result)

        result.valid = all(
            [
                result.syntax_ok,
                result.signature_ok,
                not result.forbidden_api_used,
                result.runtime_smoke_test_ok,
                result.component_dict_ok,
                result.finite_output_ok,
            ]
        )
        self._finish(result, report_path)
        return result

    def validate_directory(self, candidates_dir: str | Path) -> list[ValidationResult]:
        """Validate every reward_fn_source.py under a candidates directory."""
        root = Path(candidates_dir)
        results: list[ValidationResult] = []
        for reward_file in sorted(root.glob("candidate_*/reward_fn_source.py")):
            report_path = reward_file.parent / "validator_report.json"
            results.append(self.validate_file(reward_file, candidate_id=reward_file.parent.name, report_path=report_path))
        return results

    def _check_signature(self, tree: ast.AST, result: ValidationResult) -> bool:
        funcs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "compute_reward"]
        if not funcs:
            result.errors.append("Missing required function: compute_reward")
            return False
        func = funcs[0]
        args = [a.arg for a in func.args.args]
        if args[: len(REQUIRED_SIGNATURE)] != REQUIRED_SIGNATURE:
            result.errors.append(
                f"compute_reward signature must start with {REQUIRED_SIGNATURE}, got {args}"
            )
            return False
        return True

    def _has_forbidden_api(self, tree: ast.AST, result: ValidationResult) -> bool:
        found = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name in FORBIDDEN_IMPORTS:
                        result.errors.append(f"Forbidden import: {name}")
                        found = True
            elif isinstance(node, ast.Call):
                call_name = self._call_name(node.func)
                if call_name in FORBIDDEN_CALL_NAMES:
                    result.errors.append(f"Forbidden call: {call_name}")
                    found = True
            elif isinstance(node, ast.Attribute):
                if node.attr in FORBIDDEN_ATTR_NAMES:
                    result.errors.append(f"Forbidden attribute access: .{node.attr}")
                    found = True
        return found

    def _call_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _runtime_smoke_test(self, reward_path: Path, result: ValidationResult) -> None:
        try:
            module = self._load_module(reward_path)
            reward_fn = getattr(module, "compute_reward")
            tests = [
                ([0.0] * 8, 0, [0.0] * 8, False, False, {}),
                ([0.1, 1.0, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0], 2, [0.05, 0.9, 0.0, -0.2, 0.1, 0.0, 0.0, 0.0], False, False, {}),
                ([0.0] * 8, 0, [0.0] * 8, True, False, {}),
            ]
            for obs, action, next_obs, terminated, truncated, info in tests:
                total, components = reward_fn(obs, action, next_obs, terminated, truncated, info)
                total = float(total)
                if not math.isfinite(total):
                    raise ValueError("total reward is not finite")
                if not isinstance(components, dict) or not components:
                    raise ValueError("components must be a non-empty dict")
                for key, value in components.items():
                    if not isinstance(key, str):
                        raise ValueError("component keys must be strings")
                    v = float(value)
                    if not math.isfinite(v):
                        raise ValueError(f"component {key} is not finite")
            result.runtime_smoke_test_ok = True
            result.component_dict_ok = True
            result.finite_output_ok = True
        except Exception as exc:
            result.errors.append(f"Runtime smoke test failed: {exc}")

    def _load_module(self, reward_path: Path) -> ModuleType:
        module_name = f"ase_mtage_candidate_{abs(hash(str(reward_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, reward_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {reward_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _finish(self, result: ValidationResult, report_path: str | Path | None) -> None:
        if result.valid and not result.warnings:
            result.warnings = []
        if report_path is not None:
            save_json(report_path, result.to_dict())
