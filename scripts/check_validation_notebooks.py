"""Execute validation-notebook code cells without requiring Jupyter."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "validation"


def execute_notebook(path: Path) -> None:
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    data = json.loads(path.read_text(encoding="utf-8"))
    for index, cell in enumerate(data["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        code = compile(source, f"{path.name}:cell-{index}", "exec")
        exec(code, namespace)


def main() -> None:
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("no validation notebooks found")
    for notebook in notebooks:
        execute_notebook(notebook)
        print(f"executed {notebook}")


if __name__ == "__main__":
    main()
