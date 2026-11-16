#!/usr/bin/env python3
"""Refresh static catalog HTML from data/cases.json. Python standard library only.

The generated pages have no runtime build or JSON-fetch dependency. Keep the
marker pairs in index.html and case-studies.html when editing the layouts.
"""
from __future__ import annotations

import json
import re
import struct
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def image_markup(src: str, alt: str, *, extra: str = "") -> str:
    path = (ROOT / src).resolve()
    if not path.is_relative_to(ROOT / "assets" / "images") or not path.is_file():
        raise ValueError(f"Missing or invalid image: {src}")
    data = path.read_bytes()[:24]
    size = ""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24])
        size = f' width="{width}" height="{height}"'
    return f'<img src="{escape(src, quote=True)}" alt="{escape(alt, quote=True)}"{size} loading="lazy"{extra}>'


def read_cases() -> list[dict]:
    cases = json.loads((ROOT / "data" / "cases.json").read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("The case catalog must be a nonempty list")
    ids = set()
    for case in cases:
        for field in ("id", "title", "region", "dateLabel", "sensors", "workflow", "summary", "interpretation", "acquisitionNote", "figures", "notebooks", "paperReference"):
            if not case.get(field):
                raise ValueError(f"Case missing {field}: {case.get('id')}")
        if not re.fullmatch(r"[a-z0-9-]+", case["id"]) or case["id"] in ids:
            raise ValueError(f"Invalid or repeated case id: {case['id']}")
        ids.add(case["id"])
        for notebook in case["notebooks"]:
            if urlsplit(notebook["url"]).scheme != "https":
                raise ValueError("Notebook links must use HTTPS")
    return cases


def render_featured(cases: list[dict]) -> str:
    result = []
    for case in cases[:3]:
        figure = case["figures"][0]
        href = f'case-studies.html?case={case["id"]}#{case["id"]}'
        result.append(f'''<article class="featured-card">
  <a class="featured-image" href="{href}" aria-label="Explore {escape(case['title'], quote=True)}">{image_markup(figure['src'], figure['title'] + ' · ' + case['region'])}</a>
  <div class="featured-body">
    <p class="featured-meta">{escape(case['dateLabel'])} · {escape(' / '.join(case['sensors']))}</p>
    <h3>{escape(case['title'])}</h3>
    <p>{escape(case['summary'])}</p>
    <a class="text-link" href="{href}">Explore this case →</a>
  </div>
</article>''')
    return "\n".join(result)


def render_catalog(cases: list[dict]) -> str:
    sensors = sorted({sensor for case in cases for sensor in case["sensors"]})
    options = '\n'.join(f'<option value="{escape(sensor, quote=True)}">{escape(sensor)}</option>' for sensor in sensors)
    result = [f'''<form class="catalog-toolbar" data-case-filters hidden aria-label="Filter case studies">
  <div class="catalog-field catalog-field--search"><label for="case-search">Search the cases</label><input id="case-search" type="search" placeholder="A place, sensor, or workflow…" autocomplete="off"></div>
  <div class="catalog-field"><label for="sensor-filter">Sensor</label><select id="sensor-filter"><option value="">All sensors</option>{options}</select></div>
  <button class="button button--secondary" type="button" data-reset-filters>Reset</button>
</form>
<p class="catalog-count" id="case-count" role="status" aria-live="polite">{len(cases)} case studies</p>
<div class="empty-state" id="catalog-empty" hidden>No cases match these filters. Try another sensor or clear the search.</div>
<div class="catalog-grid">''']
    for case in cases:
        figure = case["figures"][0]
        title = case["title"].replace(" gas compression station", "").replace(" gas processing complex", "")
        search = " ".join(str(case[key]) for key in ("title", "region", "dateLabel", "sensors", "summary", "workflow"))
        tags = ''.join(f'<span class="sensor-tag">{escape(sensor)}</span>' for sensor in case["sensors"])
        result.append(f'''<a class="case-card" href="case-studies.html?case={case['id']}#{case['id']}" data-case-card="{case['id']}" data-sensors="{escape('|'.join(case['sensors']), quote=True)}" data-search="{escape(search, quote=True)}">
  <div class="case-card-visual">{image_markup(figure['src'], figure['title'])}</div>
  <div class="case-card-body"><p class="case-date">{escape(case['dateLabel'])}</p><h2>{escape(title)}</h2><div class="tag-list">{tags}</div><p>{escape(case['workflow'])}</p><span class="view-case">View the case →</span></div>
</a>''')
    result.append('</div>')
    for case in cases:
        tabs = []
        figures = []
        for index, figure in enumerate(case["figures"]):
            figure_id = f'{case["id"]}-figure-{index + 1}'
            tab_id = figure_id + '-tab'
            tabs.append(f'<button type="button" class="figure-tab" id="{tab_id}" aria-controls="{figure_id}" data-figure-tab>{escape(figure["title"])}</button>')
            alt = f'{case["region"]}: {figure["title"]}. Original figure display: {figure["unit"]}.'
            figures.append(f'''<div id="{figure_id}" data-figure-panel tabindex="0">
  <figure class="case-figure">
    <a class="image-link" href="{escape(figure['src'], quote=True)}" data-lightbox="{escape(case['title'] + ' · ' + figure['title'], quote=True)}">{image_markup(figure['src'], alt)}</a>
    <figcaption><span class="figure-unit">Original display: {escape(figure['unit'])}</span><strong>{escape(figure['title'])}.</strong> {escape(figure['caption'])}<br>{escape(figure['credit'])} <a href="{escape(figure['src'], quote=True)}">Open full size ↗</a></figcaption>
  </figure>
</div>''')
        notebooks = ''.join(f'<a class="button button--secondary button--small" href="{escape(n["url"], quote=True)}">{escape(n["label"])} ↗</a>' for n in case["notebooks"])
        result.append(f'''<section class="case-detail" id="{case['id']}" data-case-detail aria-labelledby="{case['id']}-title">
  <div class="case-detail-heading">
    <div><p class="eyebrow">Published case study</p><h2 id="{case['id']}-title" tabindex="-1">{escape(case['title'])}</h2><p class="lead">{escape(case['summary'])}</p></div>
    <dl class="case-facts"><div><dt>Location</dt><dd>{escape(case['region'])}</dd></div><div><dt>Acquisitions</dt><dd>{escape(case['dateLabel'])}</dd></div><div><dt>Sensors</dt><dd>{escape(' · '.join(case['sensors']))}</dd></div><div><dt>Input path</dt><dd>{escape(case['workflow'])}</dd></div></dl>
  </div>
  <div class="figure-tabs" data-figure-tabs aria-label="Figures for {escape(case['title'], quote=True)}">{''.join(tabs)}</div>
  {''.join(figures)}
  <div class="case-reading"><div><h3>How to read this comparison</h3><p>{escape(case['interpretation'])}</p><div class="notice notice--note"><strong>Acquisition context</strong><p>{escape(case['acquisitionNote'])}</p></div></div><div><h3>Follow the published workflow</h3><p>Open the reviewed notebooks with their saved outputs. Running them locally also requires the source products and the recorded input configuration.</p>{notebooks}<p class="figure-caption">{escape(case['paperReference'])} <a href="science.html#paper">Read the reference paper →</a></p></div></div>
</section>''')
    return '\n'.join(result)


def replace_block(path: Path, name: str, html: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = f'<!-- {name}:START -->', f'<!-- {name}:END -->'
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected one marker pair {name} in {path.name}")
    before, rest = text.split(start)
    _, after = rest.split(end)
    updated = before + start + '\n' + html + '\n' + end + after
    if updated != text:
        path.write_text(updated, encoding="utf-8")


def main() -> None:
    cases = read_cases()
    featured, catalog = render_featured(cases), render_catalog(cases)
    replace_block(ROOT / "index.html", "FEATURED_CASES", featured)
    replace_block(ROOT / "case-studies.html", "CASE_CATALOG", catalog)
    print(f"Rendered {len(cases)} case studies and {min(3, len(cases))} homepage previews.")


if __name__ == "__main__":
    main()
