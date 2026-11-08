"""
Draw the README figures (SVG, light and dark mode) from the numbers below; run from the repository root:

  python docs/figures/make_figures.py

The judge numbers come from real video only (no generated video): the topology judge on the 402 one-minute NATIX
evaluation clips with route records (first 5 s), and the prompt-following judge on 30 real clips.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).parent
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
STYLE = f"""<style>
  svg {{ font-family: {FONT}; }}
  .title {{ font-size: 16px; font-weight: 600; fill: #0b0b0b; }}
  .sub, .tick, .note {{ font-size: 12px; fill: #52514e; }}
  .label {{ font-size: 13px; fill: #0b0b0b; }}
  .value {{ font-size: 12px; font-weight: 600; fill: #0b0b0b; }}
  .grid {{ stroke: #e1e0d9; stroke-width: 1; }}
  .axis {{ stroke: #c3c2b7; stroke-width: 1; }}
  .s1 {{ fill: #2a78d6; }} .s2 {{ fill: #eb6834; }}
  .box {{ fill: #fcfcfb; stroke: #c3c2b7; stroke-width: 1.5; }}
  .accent {{ fill: #eef4fc; stroke: #2a78d6; stroke-width: 1.5; }}
  .outside {{ fill: none; stroke: #898781; stroke-width: 1.5; stroke-dasharray: 5 4; }}
  .group {{ fill: none; stroke: #c3c2b7; stroke-width: 1; }}
  .arrow {{ stroke: #898781; stroke-width: 1.5; fill: none; marker-end: url(#head); }}
  .head {{ fill: #898781; }}
  @media (prefers-color-scheme: dark) {{
    .title, .label, .value {{ fill: #f0efec; }}
    .sub, .tick, .note {{ fill: #c3c2b7; }}
    .grid {{ stroke: #2c2c2a; }} .axis {{ stroke: #383835; }}
    .s1 {{ fill: #3987e5; }} .s2 {{ fill: #d95926; }}
    .box {{ fill: #1a1a19; stroke: #383835; }}
    .accent {{ fill: #16243a; stroke: #3987e5; }}
    .group {{ stroke: #383835; }}
  }}
</style>"""


def svg(width: int, height: int, body: list[str], title: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
            f'height="{height}" role="img" aria-label="{escape(title)}">\n<title>{escape(title)}</title>\n{STYLE}\n'
            '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker></defs>\n'
            + "\n".join(body) + "\n</svg>\n")


def text(x: float, y: float, value: str, cls: str, anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{escape(value)}</text>'


def bar_right(x: float, y: float, length: float, height: float, cls: str, r: float = 4) -> str:
    """Horizontal bar anchored at x, rounded at its data end."""
    r = min(r, length, height / 2)
    return (f'<path class="{cls}" d="M{x:.1f},{y:.1f} h{length - r:.1f} a{r},{r} 0 0 1 {r},{r} v{height - 2 * r:.1f} '
            f'a{r},{r} 0 0 1 {-r},{r} h{-(length - r):.1f} z"/>')


def bar_up(x: float, base: float, width: float, length: float, cls: str, r: float = 4) -> str:
    """Vertical bar anchored at the baseline, rounded at its data end."""
    r = min(r, length, width / 2)
    top = base - length
    return (f'<path class="{cls}" d="M{x:.1f},{base:.1f} v{-(length - r):.1f} a{r},{r} 0 0 1 {r},{-r} '
            f'h{width - 2 * r:.1f} a{r},{r} 0 0 1 {r},{r} v{length - r:.1f} z"/>'
            if top < base else "")


def topology_judge() -> str:
    kappa = [("Road type", 0.44), ("Pedestrian crossing", 0.33), ("One-way", 0.28), ("Intersection", 0.28),
             ("Lanes", 0.25), ("Traffic signals*", 0.15)]
    width, left, right, top, row = 760, 170, 60, 78, 34
    scale = (width - left - right) / 0.5
    height = top + row * len(kappa) + 56
    title = "Topology judge on real video: agreement with OpenStreetMap"
    body = [text(0, 20, title, "title"),
            text(0, 40, "Cohen's κ of Qwen-Drive-4B's answers vs. the map, 402 real NATIX clips (first 5 s); "
                        "0 = chance", "sub")]
    for tick in (0, 0.1, 0.2, 0.3, 0.4, 0.5):
        x = left + tick * scale
        body.append(f'<line class="{"axis" if tick == 0 else "grid"}" x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" '
                    f'y2="{top + row * len(kappa) - 6}"/>')
        body.append(text(x, top + row * len(kappa) + 10, f"{tick:.1f}", "tick", "middle"))
    for i, (name, value) in enumerate(kappa):
        y = top + i * row
        body.append(text(left - 12, y + 14, name, "label", "end"))
        body.append(bar_right(left, y, value * scale, 18, "s1"))
        body.append(text(left + value * scale + 8, y + 14, f"{value:.2f}", "value"))
    body.append(text(0, height - 8, "* only 9 of the 402 clips pass traffic signals in their first 5 s, too few "
                                    "to test. Answers are skewed, so κ rather than accuracy.", "note"))
    return svg(width, height, body, title)


def prompt_judge() -> str:
    groups = [("True statements confirmed", "static scene", 90.7, 31.4),
              ("True statements confirmed", "dynamics", 87.0, 27.8),
              ("False statements rejected", "near-misses", 90.0, 94.4)]
    width, height, left, base, top = 760, 360, 56, 290, 92
    scale = (base - top) / 100
    title = "Prompt-following judge on real video: own caption vs. another clip's caption"
    body = [text(0, 20, title, "title"),
            text(0, 40, "Share of caption statements Qwen-Drive-4B judges correctly, 30 real clips "
                        "(statements split by Qwen3-32B)", "sub")]
    legend = [("s1", "Clip with its own caption"), ("s2", "Clip with another clip's caption (control)")]
    x: float = left
    for cls, name in legend:
        body.append(f'<rect class="{cls}" x="{x:.1f}" y="56" width="12" height="12" rx="2"/>')
        body.append(text(x + 18, 66, name, "label"))
        x += 30 + 7.4 * len(name)
    for tick in range(0, 101, 25):
        y = base - tick * scale
        body.append(f'<line class="{"axis" if tick == 0 else "grid"}" x1="{left}" y1="{y:.1f}" x2="{width - 10}" '
                    f'y2="{y:.1f}"/>')
        body.append(text(left - 8, y + 4, f"{tick}%", "tick", "end"))
    slot = (width - 10 - left) / len(groups)
    bar = 56
    for i, (kind, what, own, other) in enumerate(groups):
        centre = left + slot * (i + 0.5)
        for j, (cls, value) in enumerate((("s1", own), ("s2", other))):
            x0 = centre - bar - 1 + j * (bar + 2)
            body.append(bar_up(x0, base, bar, value * scale, cls))
            body.append(text(x0 + bar / 2, base - value * scale - 6, f"{value:.0f}%", "value", "middle"))
        body.append(text(centre, base + 20, kind, "label", "middle"))
        body.append(text(centre, base + 36, what, "tick", "middle"))
    body.append(text(0, height - 6, "A judge that reads the video confirms far fewer statements of the wrong "
                                    "caption; false statements stay false on any clip.", "note"))
    return svg(width, height, body, title)


def overview() -> str:
    width, height = 960, 366
    title = "CROWD-anything: from dashcam videos to world-model training data and evaluation"
    body = [text(0, 20, title, "title")]

    def box(x: float, y: float, w: float, h: float, head: str, lines: list[str], cls: str = "box") -> None:
        body.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>')
        heading = text(x + w / 2, y + 26, head, "label", "middle")
        body.append(heading.replace('class="label"', 'class="label" font-weight="600"'))
        for k, line in enumerate(lines):
            body.append(text(x + w / 2, y + 46 + 16 * k, line, "sub", "middle"))

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        body.append(f'<path class="arrow" d="M{x1},{y1} L{x2},{y2}"/>')

    y, w, h, gap = 48, 204, 104, 48
    top_row = [("CROWD videos", ["dashcam drives worldwide", "mapping.csv: segments"]),
               ("Frames + camera poses", ["24 fps canonical grid", "MapAnything (main.py)"]),
               ("Video latents", ["LTX-2 video VAE encoder", "tokenization/"])]
    for i, (head, lines) in enumerate(top_row):
        x = 1 + i * (w + gap)
        box(x, y, w, h, head, lines)
        arrow(x + w + 4, y + h / 2, x + w + gap - 4, y + h / 2)
    xw = 1 + 3 * (w + gap)
    box(xw, y, width - xw - 2, h, "World model", ["trained on the latents", "(outside this repository)"], "outside")

    gy, gh = 204, 160
    body.append(f'<rect class="group" x="0.5" y="{gy}" width="{width - 1}" height="{gh}" rx="12"/>')
    body.append(text(16, gy + 26, "evaluation/ — each generated video vs. its real clip", "label")
                .replace('class="label"', 'class="label" font-weight="600"'))
    arrow(xw + (width - xw) / 2, y + h + 4, xw + (width - xw) / 2, gy - 4)
    evals = [("Reconstruction", ["paired.py", "PSNR · SSIM · LPIPS", "per frame, by horizon"]),
             ("Road topology", ["topology.py", "OSM facts asked to a VLM", "κ vs. the real clip"]),
             ("Prompt following", ["prompt_following.py", "LLM splits the caption,", "VLM checks yes / no"]),
             ("Traffic elements", ["traffic_elements.py", "Grounding DINO + DINOv2,", "matched in time and place"])]
    ew, egap = 218, 14
    for i, (head, lines) in enumerate(evals):
        box(16 + i * (ew + egap), gy + 42, ew, 102, head, lines, "accent")
    return svg(width, height, body, title)


if __name__ == "__main__":
    for name, draw in (("overview", overview), ("topology_judge", topology_judge),
                       ("prompt_following_judge", prompt_judge)):
        (OUT / f"{name}.svg").write_text(draw())
