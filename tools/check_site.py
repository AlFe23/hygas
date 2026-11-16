#!/usr/bin/env python3
"""Check the standalone website using only Python's standard library.

This checks local integrity and metadata. It does not contact external sources,
run the scientific software, or replace visual/interaction testing in a browser.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("index.html", "get-started.html", "user-guide.html", "case-studies.html", "science.html")
VOID = set("area base br col embed hr img input link meta param source track wbr".split())


class Page(HTMLParser):
    def __init__(self, path: Path):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.ids: list[str] = []
        self.refs: list[tuple[str, str, int]] = []
        self.aria: list[tuple[str, str, int]] = []
        self.errors: list[str] = []
        self.stack: list[str] = []
        self.counts: Counter = Counter()
        self.inputs: list[dict] = []
        self.labels: set[str] = set()
        self.lang = ""
        self.images = 0
        self.feed(path.read_text(encoding="utf-8"))
        self.close()
        if self.stack:
            self.errors.append("Unclosed tags: " + ", ".join(self.stack))

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        line = self.getpos()[0]
        self.counts[tag] += 1
        if tag == "html": self.lang = a.get("lang", "")
        if a.get("id"): self.ids.append(a["id"])
        for key in ("href", "src"):
            if a.get(key): self.refs.append((key, a[key], line))
        for key in ("aria-controls", "aria-labelledby", "aria-describedby"):
            if a.get(key): self.aria.append((key, a[key], line))
        if tag == "img":
            self.images += 1
            if "alt" not in a:
                self.errors.append(f"Line {line}: image has no alt attribute")
        if tag == "input": self.inputs.append(a)
        if tag == "label" and a.get("for"): self.labels.add(a["for"])
        if tag not in VOID: self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID: self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag in VOID: return
        line = self.getpos()[0]
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"Line {line}: unexpected closing {tag}, stack={self.stack[-4:]}")
            return
        self.stack.pop()


def main() -> int:
    errors = []
    pages = {name: Page(ROOT / name) for name in PAGES}
    cases = json.loads((ROOT / "data" / "cases.json").read_text(encoding="utf-8"))
    case_ids = {case["id"] for case in cases}
    if len(case_ids) != len(cases): errors.append("Repeated case IDs in data/cases.json")
    reference_count = 0
    for name, page in pages.items():
        errors.extend(f"{name}: {message}" for message in page.errors)
        for identifier, count in Counter(page.ids).items():
            if count > 1: errors.append(f"{name}: repeated ID {identifier}")
        for tag in ("html", "head", "body", "title", "main", "h1"):
            if page.counts[tag] != 1: errors.append(f"{name}: expected one {tag}, found {page.counts[tag]}")
        if page.lang != "en": errors.append(f"{name}: expected lang=en")
        if "main-content" not in page.ids: errors.append(f"{name}: missing skip-link target")
        for _, target_ids, line in page.aria:
            for identifier in target_ids.split():
                if identifier not in page.ids: errors.append(f"{name}:{line}: missing ARIA target {identifier}")
        for field in page.inputs:
            if field.get("type") != "hidden" and not (field.get("aria-label") or field.get("aria-labelledby") or field.get("id") in page.labels):
                errors.append(f"{name}: input has no explicit accessible label")
        for attr, value, line in page.refs:
            target = urlsplit(value)
            if target.scheme or target.netloc:
                if target.scheme not in ("https", "mailto"):
                    errors.append(f"{name}:{line}: unsupported external URL scheme in {value}")
                continue
            reference_count += 1
            if target.path.startswith("/"):
                errors.append(f"{name}:{line}: root-relative path breaks project Pages: {value}")
                continue
            file = (ROOT / unquote(target.path)).resolve() if target.path else page.path.resolve()
            if not file.is_relative_to(ROOT):
                errors.append(f"{name}:{line}: reference escapes site folder: {value}")
                continue
            if not file.is_file():
                errors.append(f"{name}:{line}: missing local file {value}")
                continue
            if target.fragment and file.suffix == ".html":
                linked = pages.get(file.name)
                if linked and unquote(target.fragment) not in linked.ids:
                    errors.append(f"{name}:{line}: missing anchor {value}")
            case = parse_qs(target.query).get("case", [])
            if case and case[0] not in case_ids:
                errors.append(f"{name}:{line}: unknown case selection {value}")
    for case in cases:
        if case["id"] not in pages["case-studies.html"].ids:
            errors.append(f"Catalog HTML missing case {case['id']}; run render_catalog.py")
        for figure in case["figures"]:
            if not (ROOT / figure["src"]).is_file(): errors.append(f"Missing case image {figure['src']}")
            for field in ("title", "caption", "unit", "credit"):
                if not figure.get(field): errors.append(f"Case {case['id']}: figure missing {field}")
    manifest = json.loads((ROOT / "data" / "asset-sources.json").read_text(encoding="utf-8"))
    assets = manifest if isinstance(manifest, list) else manifest.get("assets", [])
    if not assets: errors.append("Asset provenance inventory is empty")
    checked_hashes = 0
    for asset in assets:
        filename = asset.get("path") or asset.get("asset") or asset.get("website_asset")
        digest = asset.get("sha256")
        if not filename or not digest:
            errors.append("Asset manifest entry missing path or sha256")
            continue
        file = ROOT / filename
        if not file.is_file(): errors.append(f"Missing asset {filename}")
        elif hashlib.sha256(file.read_bytes()).hexdigest() != digest: errors.append(f"Asset hash mismatch {filename}")
        else: checked_hashes += 1
    result = {
        "result": "passed" if not errors else "failed",
        "pages_checked": len(pages),
        "local_references_checked": reference_count,
        "case_studies": len(cases),
        "case_figures": sum(len(case["figures"]) for case in cases),
        "asset_hashes_verified": checked_hashes,
        "browser_visual_qa": "Not performed: no browser or preview tool available in the execution environment.",
        "scientific_pipelines_executed": False,
        "external_links_live_checked": False,
        "deployed": False,
        "errors": errors,
    }
    (ROOT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
