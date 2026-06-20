from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union


JsonPath = Union[str, Path]


def read_json(path: JsonPath) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: JsonPath, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, target)
    finally:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink()
