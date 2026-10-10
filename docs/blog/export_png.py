"""Export every chart and box in clef-docs.html as its own PNG, using headless Chrome.

    .venv/bin/python docs/blog/build.py && .venv/bin/python docs/blog/export_png.py

Each chart is rendered alone, from the same data and drawing code as the post,
so a PNG can never disagree with the page. Light theme, 2x pixel density.
Writes docs/blog/images/<name>.png.
"""
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# chart id -> (drawing function, width, height in CSS pixels)
CHARTS = {
    "chart-tasks": ("drawTasks", 1000, 440),
    "chart-confidence": ("drawConfidence", 1000, 480),
    "chart-fooled": ("drawFooled", 1000, 480),
}
# box class -> (image name, width, height in CSS pixels)
BOXES = {
    "short": ("short-version", 760, 430),
    "fit": ("why-it-fits", 760, 400),
    "readout": ("clef-request", 760, 340),
    "receipts": ("near-miss-receipt", 760, 230),
    "stats": ("cost", 760, 150),
}


def shoot(src, png, w, h):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={w},{h}", "--force-device-scale-factor=2",
                    "--virtual-time-budget=15000", f"--screenshot={png}", f"file://{src}"],
                   check=True, capture_output=True)
    print(f"wrote docs/blog/images/{png.name}")


def main():
    page = (HERE / "clef-docs.html").read_text()
    head = page[: page.index("</style>") + len("</style>")]
    style = head[head.index("<link rel=\"preconnect\""):]
    scripts = page[page.index('<script src="https://cdnjs'):]
    out = HERE / "images"
    out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for chart, (fn, w, h) in CHARTS.items():
            only = scripts.replace("drawTasks(t); drawConfidence(t); drawFooled(t);", f"{fn}(t);")
            assert only != scripts, "drawing call not found"
            html = (f'<!doctype html><html data-theme="light"><head><meta charset="utf-8">{style}'
                    f'<style>body{{padding:0;margin:0}}#{chart}{{width:{w}px;height:{h}px;margin:0}}</style>'
                    f'</head><body><div id="{chart}"></div>{only}</body></html>')
            src = Path(tmp) / f"{chart}.html"
            src.write_text(html)
            shoot(src, out / f"{chart.removeprefix('chart-')}.png", w, h)
        for cls, (name, w, h) in BOXES.items():
            box = re.search(rf'<div class="{cls}"[^>]*>.*?\n</div>\n', page, re.S).group(0)
            html = (f'<!doctype html><html data-theme="light"><head><meta charset="utf-8">{style}'
                    f'<style>body{{padding:24px;margin:0}}.{cls}{{margin:0}}</style>'
                    f'</head><body>{box}</body></html>')
            src = Path(tmp) / f"{name}.html"
            src.write_text(html)
            shoot(src, out / f"{name}.png", w, h)


if __name__ == "__main__":
    main()
