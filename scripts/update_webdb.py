"""Re-embed the gpu database into the web demo.

Run after changing gpus.py:  python scripts/update_webdb.py
tests/test_webdemo.py fails until this has been done.
"""

import json
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(__file__).parent.parent
page_path = root / "docs" / "index.html"

out = subprocess.run(
    [sys.executable, "-m", "kernelmeter.cli", "gpus", "--json"],
    capture_output=True, text=True, check=True,
)
compact = json.dumps(json.loads(out.stdout), separators=(",", ":"))

page = page_path.read_text()
new_page, n = re.subn(
    r'(<script type="application/json" id="db">).*?(</script>)',
    lambda m: m.group(1) + compact + m.group(2),
    page,
    flags=re.S,
)
if n != 1:
    raise SystemExit("db block not found in docs/index.html")
page_path.write_text(new_page)
print(f"embedded {len(json.loads(compact))} cards into docs/index.html")
