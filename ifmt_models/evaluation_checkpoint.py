from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class CheckpointError(RuntimeError):
    pass


class CheckpointMismatchError(CheckpointError):
    pass


class CheckpointCorruptionError(CheckpointError):
    pass


class DuplicateCheckpointError(CheckpointError):
    pass


@dataclass(frozen=True)
class RunFingerprint:
    seed: int
    case_count: int
    sampling_version: str
    masking_version: str
    prompt_version: str
    model: str
    runtime_digest: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json_bytes(self.as_dict())).hexdigest()


class CheckpointStore:
    def __init__(
        self,
        run_dir: Path,
        fingerprint: RunFingerprint,
        cases_dir: Path,
    ) -> None:
        self.run_dir = run_dir
        self.fingerprint = fingerprint
        self.cases_dir = cases_dir

    @classmethod
    def create_or_open(
        cls,
        run_dir: Path,
        fingerprint: RunFingerprint,
    ) -> "CheckpointStore":
        run_dir = Path(run_dir)
        checkpoint_dir = run_dir / "checkpoint"
        cases_dir = checkpoint_dir / "cases"
        manifest_path = checkpoint_dir / "manifest.json"
        cases_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "fingerprint": fingerprint.as_dict(),
            "fingerprint_digest": fingerprint.digest(),
        }
        if manifest_path.exists():
            current = _read_json(manifest_path)
            current_digest = str(current.get("fingerprint_digest", ""))
            if current_digest != manifest["fingerprint_digest"]:
                raise CheckpointMismatchError(
                    f"checkpoint fingerprint {current_digest} does not match "
                    f"{manifest['fingerprint_digest']}"
                )
        else:
            _atomic_write_json(manifest_path, manifest)
        return cls(run_dir, fingerprint, cases_dir)

    def save_case(self, record: dict[str, Any]) -> None:
        case = record.get("case")
        if not isinstance(case, dict):
            raise CheckpointCorruptionError("checkpoint record has no case object")
        case_id = str(case.get("case_id", ""))
        if not re.fullmatch(r"case_\d{4,}", case_id):
            raise CheckpointCorruptionError(f"invalid checkpoint case_id: {case_id!r}")

        destination = self.cases_dir / f"{case_id}.json"
        payload = _canonical_json_bytes(record)
        if destination.exists():
            try:
                existing = _canonical_json_bytes(_read_json(destination))
            except CheckpointCorruptionError as exc:
                raise DuplicateCheckpointError(
                    f"existing checkpoint for {case_id} is corrupt"
                ) from exc
            if existing == payload:
                return
            raise DuplicateCheckpointError(
                f"checkpoint for {case_id} already exists with different content"
            )
        _atomic_write_bytes(destination, payload + b"\n")

    def load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in sorted(self.cases_dir.glob("case_*.json")):
            value = _read_json(path)
            if not isinstance(value, dict):
                raise CheckpointCorruptionError(
                    f"checkpoint {path} must contain a JSON object"
                )
            records.append(value)
        return records

    def completed_case_ids(self) -> set[str]:
        return {
            str(record["case"]["case_id"])
            for record in self.load_records()
        }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptionError(f"cannot read checkpoint {path}: {exc}") from exc


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
        "utf-8"
    )
    _atomic_write_bytes(path, payload + b"\n")


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
