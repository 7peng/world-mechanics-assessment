"""docs/slides/talk.html from docs/slides/template.html with figures embedded."""
import base64
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
t = (ROOT / "docs" / "slides" / "template.html").read_text()
for key in set(re.findall(r"__([a-z0-9_]+)__", t)):
    png = ROOT / "outputs" / "figures" / "report" / f"{key}.png"
    t = t.replace(f"__{key}__", "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode())
(ROOT / "docs" / "slides" / "talk.html").write_text(t)
print(len(t) // 1024, "KB")
