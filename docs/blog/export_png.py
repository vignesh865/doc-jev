"""Export every chart and box in clef-docs.html as its own PNG, using headless Chrome.

    .venv/bin/python docs/blog/build.py && .venv/bin/python docs/blog/export_png.py

Each chart is rendered alone, from the same data and drawing code as the post,
so a PNG can never disagree with the page. Light theme, 2x pixel density.
Writes docs/blog/images/<name>.png, and the ASD-STE100 page's images to
docs/blog/images_ste/ (one to one; an image whose source did not change is
copied, not rendered again).
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# chart id -> (drawing function, width, height in CSS pixels)
CHARTS = {
    "chart-overview": ("drawOverview", 1000, 420),
    "chart-confidence": ("drawConfidence", 1000, 480),
    "chart-skills": ("drawSkills", 1000, 560),
    "chart-fooled": ("drawFooled", 1000, 480),
    "chart-contam": ("drawContam", 1000, 520),
}
# box class -> (image name, width, height in CSS pixels)
BOXES = {
    "short": ("short-version", 760, 560),
    "map-wrap": ("test-map", 760, 400),
    "fit": ("why-it-fits", 760, 400),
    "readout": ("clef-request", 760, 340),
    "receipts": ("near-miss-receipt", 760, 230),
    "stats": ("cost", 760, 150),
}


def shoot(src, png, w, h, folder="images"):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={w},{h}", "--force-device-scale-factor=2",
                    "--virtual-time-budget=15000", f"--screenshot={png}", f"file://{src}"],
                   check=True, capture_output=True)
    print(f"wrote docs/blog/{folder}/{png.name}")


def render(page_name, out_name, same_as=None):
    """Render every chart and box of one page into one folder.

    With same_as=(page, folder), an image whose source is unchanged from that
    page is copied from its folder instead of rendered again, so the two folders
    stay one to one.
    """
    page = (HERE / page_name).read_text()
    other = (HERE / same_as[0]).read_text() if same_as else None
    head = page[: page.index("</style>") + len("</style>")]
    style = head[head.index('<link rel="preconnect"'):]
    scripts = page[page.index('<script src="https://cdnjs'):]
    out = HERE / out_name
    out.mkdir(exist_ok=True)

    def unchanged(part_of, name):
        if other is None:
            return False
        if part_of(page) == part_of(other):
            shutil.copyfile(HERE / same_as[1] / name, out / name)
            print(f"copied docs/blog/{out_name}/{name} (unchanged)")
            return True
        return False

    with tempfile.TemporaryDirectory() as tmp:
        for chart, (fn, w, h) in CHARTS.items():
            name = f"{chart.removeprefix('chart-')}.png"
            # The data and the chart code decide the image; both pages share them.
            if unchanged(lambda p: p[p.index('<script src="https://cdnjs'):], name):
                continue
            only = scripts.replace("drawOverview(t); drawConfidence(t); drawSkills(t); drawFooled(t); drawContam(t);", f"{fn}(t);")
            assert only != scripts, "drawing call not found"
            html = (f'<!doctype html><html data-theme="light"><head><meta charset="utf-8">{style}'
                    f'<style>body{{padding:0;margin:0}}#{chart}{{width:{w}px;height:{h}px;margin:0}}</style>'
                    f'</head><body><div id="{chart}"></div>{only}</body></html>')
            src = Path(tmp) / f"{chart}.html"
            src.write_text(html)
            shoot(src, out / name, w, h, out_name)
        for cls, (name, w, h) in BOXES.items():
            find = lambda p: re.search(rf'<div class="{cls}"><table.*?</table></div>|<div class="{cls}"[^>]*>.*?\n</div>\n', p, re.S).group(0)
            if unchanged(find, f"{name}.png"):
                continue
            html = (f'<!doctype html><html data-theme="light"><head><meta charset="utf-8">{style}'
                    f'<style>body{{padding:24px;margin:0}}.{cls}{{margin:0}}</style>'
                    f'</head><body>{find(page)}</body></html>')
            src = Path(tmp) / f"{name}.html"
            src.write_text(html)
            shoot(src, out / f"{name}.png", w, h, out_name)


def main():
    render("clef-docs.html", "images")
    render("clef-docs-ste.html", "images_ste", same_as=("clef-docs.html", "images"))


if __name__ == "__main__":
    main()
