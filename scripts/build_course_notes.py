"""Convert the lecture notes and the quiz into one Markdown file for AI assistants.

Runs as a Quarto pre-render script (see `_quarto.yml`) and writes
`course_notes.md` into the project root, from where Quarto copies it to the
output directory as a resource. Generating it before rendering lets Quarto
resolve links to it. The file is a build artifact and is not committed. Page order follows the sidebar sections listed in
`SECTIONS`, followed by the quiz page. Quarto-specific markup
(shortcodes, hidden blocks, raw HTML, images, cross-references) is removed
or converted to plain Markdown.
"""

import os
import re
from pathlib import Path

import yaml

SECTIONS = ["Background & Preparation", "Lectures"]
EXTRA_PAGES = ["pages/lectures/practical.qmd"]
QUIZ_PAGE = "pages/misc/quiz.qmd"

HEADER = """# {title} — Course Notes

Plain-text version of the lecture notes, generated automatically from the
course website for use with AI assistants (e.g. as a project file in
ChatGPT or Claude). Figures, videos and interactive demos are omitted;
figure captions are kept. The last part contains the quiz and exam
preparation questions with reference answers. Online version: {site_url}

Formulas use the following LaTeX macros:

```latex
{macros}
```

"""

QUIZ_NOTE = """Quiz and exam preparation questions (in German), grouped by topic. Each
question is followed by its reference answer, marked **Antwort**. Questions
that refer to a figure link to the image."""

XREF_NAMES = {
    "fig": "figure",
    "eq": "equation",
    "tbl": "table",
    "sec": "section",
    "lst": "listing",
}


def page_order(config: dict) -> list[str]:
    pages = []
    for entry in config["website"]["sidebar"]["contents"]:
        if isinstance(entry, dict) and entry.get("section") in SECTIONS:
            pages.extend(entry["contents"])
    return pages + [p for p in EXTRA_PAGES if p not in pages]


def split_front_matter(text: str) -> tuple[dict, str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not match:
        return {}, text
    return yaml.safe_load(match.group(1)) or {}, text[match.end() :]


def strip_blocks(lines: list[str]) -> list[str]:
    """Remove hidden divs, raw HTML and figure-only code cells; unwrap other divs."""
    out = []
    headings = []  # indices of heading lines in `out`
    skip_div_depth = None  # div depth at which a hidden block started
    depth = 0
    fence = None  # (marker, keep) while inside a code block
    for line in lines:
        if fence:
            marker, keep = fence
            if line.strip() == marker:
                fence = None
                if keep:
                    out.append(marker)
            elif keep and not re.match(r"^#\s?\|", line):
                out.append(line)
            continue

        code = re.match(r"^(`{3,})\s*\{?=?([\w-]*)", line)
        if code and skip_div_depth is None:
            marker, lang = code.group(1), code.group(2)
            if lang == "html" or "echo: false" in line:
                fence = (marker, False)
            else:
                fence = (marker, True)
                out.append(marker + lang)
            continue
        if code:
            fence = (code.group(1), False)
            continue

        div = re.match(r"^(:{3,})\s*(.*)$", line)
        if div:
            attrs = div.group(2).strip()
            if attrs:
                depth += 1
                if (
                    skip_div_depth is None
                    and "content-hidden" in attrs
                    and "unless-format" not in attrs
                ):
                    skip_div_depth = depth
                elif skip_div_depth is None:
                    title = re.search(r'title="([^"]*)"', attrs)
                    if title:
                        out.append(f"**{title.group(1)}**\n")
            else:
                if skip_div_depth == depth:
                    skip_div_depth = None
                depth -= 1
            continue

        if skip_div_depth is None:
            if re.match(r"^#+ ", line):
                headings.append(len(out))
            out.append(line)
    # Page titles become level-1 headings, so pages using level-1 sections are demoted.
    if any(out[i].startswith("# ") for i in headings):
        for i in headings:
            out[i] = "#" + out[i]
    return out


def clean_inline(text: str, site_url: str, page: str, link_images: bool) -> str:
    text = re.sub(r"\{\{<\s*(video|include|embed)\b.*?>\}\}\s*", "", text)
    text = re.sub(r"\{\{<.*?>\}\}", "", text)
    # Images: link to the image on the website, or keep alt text only if present.
    if link_images:
        text = re.sub(
            r"!\[([^\]]*)\]\(/?([^)]*)\)(\{[^}]*\})?",
            lambda m: (
                f"[Figure{': ' + m.group(1) if m.group(1) else ''}]({site_url}{m.group(2)})"
            ),
            text,
        )
    text = re.sub(
        r"!\[([^\]]*)\]\([^)]*\)(\{[^}]*\})?",
        lambda m: f"[Figure: {m.group(1)}]" if m.group(1) else "",
        text,
    )
    text = re.sub(r"<summary>(.*?)</summary>", r"**\1**", text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"</?(details|div|span|iframe|center)\b[^>]*>", "", text)
    # Spans and link/heading attributes: [text]{style=...} -> text, drop {#id .class key=val}.
    text = re.sub(r"\[([^\]]+)\]\{[^}]*\}", r"\1", text)
    text = re.sub(r"\s*\{[#.][^}]*\}[ \t]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\)\{[^}]*=[^}]*\}", ")", text)
    # Relative links to other pages -> absolute website URLs.
    base = Path(page).parent

    def absolute(m: re.Match) -> str:
        target, anchor = m.group(2), m.group(3) or ""
        resolved = os.path.normpath(base / target)
        return f"]({site_url}{re.sub(r'\.qmd$', '.html', resolved)}{anchor})"

    text = re.sub(
        r"\]\((?!https?:|#|mailto:)(\.{0,2}/?)([^)#\s]+)(#[^)\s]*)?\)",
        lambda m: absolute(m) if not m.group(2).startswith("/") else m.group(0),
        text,
    )
    # Cross-references and citations.
    text = re.sub(
        r"@(fig|eq|tbl|sec|lst)-[\w-]+", lambda m: f"the {XREF_NAMES[m.group(1)]}", text
    )
    text = re.sub(
        r"\[@([\w:-]+)(;\s*@([\w:-]+))*\]",
        lambda m: "(" + m.group(0)[1:-1].replace("@", "") + ")",
        text,
    )
    text = re.sub(r"(?<![\w.])@([A-Za-z][\w:-]*_\d{4}[\w-]*)", r"(\1)", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def convert_page(path: Path, site_url: str, link_images: bool = False) -> str:
    meta, body = split_front_matter(path.read_text(encoding="utf-8"))
    images_path = meta.get("params", {}).get("images_path", "")
    body = body.replace("{{< meta params.images_path >}}", images_path)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    body = "\n".join(strip_blocks(body.splitlines()))
    body = clean_inline(body, site_url, str(path), link_images)
    body = re.sub(r"## References\s*$", "", body).strip()
    url = site_url + str(path.with_suffix(".html"))
    return f"# {meta.get('title', path.stem)}\n\nSource: {url}\n\n{body}\n"


def main() -> None:
    config = yaml.safe_load(Path("_quarto.yml").read_text(encoding="utf-8"))
    site_url = config["website"]["site-url"].rstrip("/") + "/"

    macros = Path("assets/_macros.tex").read_text(encoding="utf-8").strip()
    parts = [
        HEADER.format(
            title=config["website"]["title"], site_url=site_url, macros=macros
        )
    ]
    parts += [convert_page(Path(page), site_url) for page in page_order(config)]
    title, source, quiz = convert_page(
        Path(QUIZ_PAGE), site_url, link_images=True
    ).split("\n\n", 2)
    parts.append(f"{title}\n\n{source}\n\n{QUIZ_NOTE}\n\n{quiz}")
    Path("course_notes.md").write_text("\n\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
