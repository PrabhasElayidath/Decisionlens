"""
Streamlit entrypoint — run from the repo root:

    streamlit run run_streamlit.py

Using this launcher keeps the main script path stable (``run_streamlit.py``) so
the browser is less likely to show Streamlit's "Page not found" banner after
you edit ``ui/app.py`` (that banner appears when the session asks for an old
``page_script_hash``).

The dashboard code still lives in ``ui/app.py``; we execute it with the correct
``__file__`` so asset paths (e.g. ``style.css``) resolve correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "ui" / "app.py"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_src = APP.read_text(encoding="utf-8")
_code = compile(_src, str(APP), "exec")
_globals: dict = {
    "__name__": "__main__",
    "__file__": str(APP),
    "__doc__": None,
    "__builtins__": __builtins__,
}
exec(_code, _globals)
