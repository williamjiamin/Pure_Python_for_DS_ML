from __future__ import annotations

import argparse
import ast
import html
import json
import mimetypes
import re
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

CHAPTER_DIR_RE = re.compile(r"^(?P<number>\d+)\.(?P<name>.+)$")
ORDERED_LIST_RE = re.compile(r"^\d+\.\s+")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CURRICULUM_ROW_RE = re.compile(r"^\|\s*\[([^\]]+)\]\(([^)]+)\)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$")

README_PATH = REPO_ROOT / "README.md"
CURRICULUM_SOURCE_PATH = README_PATH.resolve().relative_to(REPO_ROOT).as_posix()
DATA_FILE_SUFFIXES = {".csv", ".zip", ".json", ".txt", ".names"}
GENERIC_TITLES = {"pure python for data science & machine learning"}


def chapter_sort_key(path: Path) -> tuple[int, str]:
    match = CHAPTER_DIR_RE.match(path.name)
    if match:
        return int(match.group("number")), path.name.lower()
    if path.name == "PlusPlus.Ensemble_Algo":
        return 19_500, path.name.lower()
    return 999_999, path.name.lower()


def title_from_folder(path: Path) -> str:
    match = CHAPTER_DIR_RE.match(path.name)
    if match:
        return match.group("name").replace("_", " ")
    return path.name.replace("_", " ").replace(".", " ")


def clean_curriculum_title(title: str) -> str:
    match = re.match(r"^\d+\.\s*(.+)$", title.strip())
    if match:
        return match.group(1).strip()
    return title.strip()


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def first_paragraph(text: str | None) -> str:
    if not text:
        return ""

    pieces: list[str] = []
    for block in str(text).strip().split("\n\n"):
        cleaned = normalize_text(block)
        if cleaned:
            pieces.append(cleaned)
    return pieces[0] if pieces else ""


def is_generic_title(title: str | None) -> bool:
    return normalize_text(title).lower() in GENERIC_TITLES


def is_low_signal_summary(summary: str | None) -> bool:
    normalized = normalize_text(summary).lower()
    return not normalized or normalized.startswith("author:")


def split_key_concepts(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def curriculum_summary(entry: dict | None) -> str:
    if not entry:
        return ""
    topic = normalize_text(entry.get("topic"))
    key_concepts = normalize_text(entry.get("key_concepts_text"))
    if topic and key_concepts:
        return f"{topic}: {key_concepts}"
    return key_concepts or topic


@lru_cache(maxsize=1)
def parse_curriculum_index() -> dict[str, dict]:
    entries: dict[str, dict] = {}
    current_part = ""

    if not README_PATH.exists():
        return entries

    for line in README_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            current_part = stripped[4:].strip()
            continue

        match = CURRICULUM_ROW_RE.match(stripped)
        if not match:
            continue

        label, target, topic, key_concepts_text = match.groups()
        folder = Path(target.strip()).parts[0] if target.strip() else ""
        if not folder or not (CHAPTER_DIR_RE.match(folder) or folder == "PlusPlus.Ensemble_Algo"):
            continue

        clean_title = clean_curriculum_title(label)
        entries[folder] = {
            "folder": folder,
            "part": current_part,
            "curriculum_title": clean_title,
            "topic": normalize_text(topic),
            "key_concepts_text": normalize_text(key_concepts_text),
            "key_concepts": split_key_concepts(key_concepts_text),
        }

    return entries


def extract_heading(markdown_text: str) -> str | None:
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def extract_first_paragraph(markdown_text: str) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    seen_title = False

    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            seen_title = True
            continue
        if not seen_title:
            continue
        if not stripped:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        if stripped.startswith("## "):
            if current:
                paragraphs.append(" ".join(current).strip())
            break
        if stripped.startswith("- ") or ORDERED_LIST_RE.match(stripped):
            if current:
                paragraphs.append(" ".join(current).strip())
            break
        current.append(stripped)

    if current:
        paragraphs.append(" ".join(current).strip())

    return paragraphs[0] if paragraphs else ""


def extract_section_list(markdown_text: str, heading: str) -> list[str]:
    lines = markdown_text.splitlines()
    found = False
    items: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower() == f"## {heading.lower()}":
            found = True
            continue
        if found and stripped.startswith("## "):
            break
        if not found:
            continue
        if stripped.startswith("- "):
            items.append(stripped[2:].strip())
            continue
        if ORDERED_LIST_RE.match(stripped):
            items.append(ORDERED_LIST_RE.sub("", stripped).strip())

    return items


def apply_inline_formatting(text: str) -> str:
    escaped = html.escape(text)
    escaped = LINK_RE.sub(r'<a href="\2" target="_blank" rel="noreferrer">\1</a>', escaped)
    escaped = INLINE_CODE_RE.sub(r"<code>\1</code>", escaped)
    escaped = BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    escaped = ITALIC_RE.sub(r"<em>\1</em>", escaped)
    return escaped


def flush_list(parts: list[str], list_type: str | None, items: list[str]) -> tuple[str | None, list[str]]:
    if not list_type or not items:
        return None, []
    tag = "ol" if list_type == "ol" else "ul"
    rendered = "".join(f"<li>{apply_inline_formatting(item)}</li>" for item in items)
    parts.append(f"<{tag}>{rendered}</{tag}>")
    return None, []


def flush_table(parts: list[str], table_rows: list[list[str]]) -> list[list[str]]:
    if not table_rows:
        return []
    header = table_rows[0]
    body = table_rows[1:]
    head_html = "".join(f"<th>{apply_inline_formatting(cell)}</th>" for cell in header)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{apply_inline_formatting(cell)}</td>" for cell in row) + "</tr>"
        for row in body
    )
    parts.append(
        '<div class="table-wrap"><table><thead><tr>'
        + head_html
        + "</tr></thead><tbody>"
        + body_html
        + "</tbody></table></div>"
    )
    return []


def render_markdown(markdown_text: str) -> str:
    parts: list[str] = []
    paragraph: list[str] = []
    list_type: str | None = None
    list_items: list[str] = []
    table_rows: list[list[str]] = []
    in_code = False
    code_lines: list[str] = []
    code_lang = ""

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            parts.append(f"<p>{apply_inline_formatting(' '.join(paragraph).strip())}</p>")
            paragraph = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            list_type, list_items = flush_list(parts, list_type, list_items)
            table_rows = flush_table(parts, table_rows)
            if not in_code:
                in_code = True
                code_lang = stripped[3:].strip()
                code_lines = []
            else:
                lang_attr = f' data-lang="{html.escape(code_lang)}"' if code_lang else ""
                code_html = html.escape("\n".join(code_lines))
                parts.append(f"<pre><code{lang_attr}>{code_html}</code></pre>")
                in_code = False
                code_lang = ""
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            list_type, list_items = flush_list(parts, list_type, list_items)
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            if not all(set(cell) <= {"-", ":"} for cell in cells):
                table_rows.append(cells)
            continue

        if not stripped:
            flush_paragraph()
            list_type, list_items = flush_list(parts, list_type, list_items)
            table_rows = flush_table(parts, table_rows)
            continue

        table_rows = flush_table(parts, table_rows)

        if stripped.startswith("### "):
            flush_paragraph()
            list_type, list_items = flush_list(parts, list_type, list_items)
            parts.append(f"<h3>{apply_inline_formatting(stripped[4:].strip())}</h3>")
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            list_type, list_items = flush_list(parts, list_type, list_items)
            parts.append(f"<h2>{apply_inline_formatting(stripped[3:].strip())}</h2>")
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            list_type, list_items = flush_list(parts, list_type, list_items)
            parts.append(f"<h1>{apply_inline_formatting(stripped[2:].strip())}</h1>")
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            if list_type not in (None, "ul"):
                list_type, list_items = flush_list(parts, list_type, list_items)
            list_type = "ul"
            list_items.append(stripped[2:].strip())
            continue
        if ORDERED_LIST_RE.match(stripped):
            flush_paragraph()
            if list_type not in (None, "ol"):
                list_type, list_items = flush_list(parts, list_type, list_items)
            list_type = "ol"
            list_items.append(ORDERED_LIST_RE.sub("", stripped).strip())
            continue

        paragraph.append(stripped)

    flush_paragraph()
    list_type, list_items = flush_list(parts, list_type, list_items)
    flush_table(parts, table_rows)
    if in_code:
        code_html = html.escape("\n".join(code_lines))
        parts.append(f"<pre><code>{code_html}</code></pre>")

    return "\n".join(parts)


def safe_relative_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


@lru_cache(maxsize=None)
def inspect_python_script(script_path: str) -> dict:
    path = REPO_ROOT / script_path
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {"summary": "", "functions": []}

    module_doc = first_paragraph(ast.get_docstring(tree))
    functions: list[dict] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(
                {
                    "name": node.name,
                    "summary": first_paragraph(ast.get_docstring(node)),
                }
            )

    return {"summary": module_doc, "functions": functions}


def notebook_to_markdown(notebook_path: Path) -> str:
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    chunks: list[str] = []

    for cell in payload.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            chunks.append("".join(source).strip())
        elif isinstance(source, str):
            chunks.append(source.strip())

    return "\n\n".join(chunk for chunk in chunks if chunk)


def chapter_file_inventory(folder: Path) -> dict:
    markdown_files = sorted(folder.glob("*.md"))
    notebook_files = sorted(folder.glob("*.ipynb"))
    script_files = sorted(folder.glob("*.py"))
    data_files = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in DATA_FILE_SUFFIXES
    )
    return {
        "markdown_files": markdown_files,
        "notebook_files": notebook_files,
        "script_files": script_files,
        "data_files": data_files,
    }


def generated_assignments(script_paths: list[str], data_paths: list[str]) -> list[str]:
    assignments: list[str] = []
    if script_paths:
        assignments.append(f"Run `{Path(script_paths[0]).name}` and trace the main execution flow.")
        assignments.append(f"Open `{Path(script_paths[0]).name}` and explain each helper function in plain language.")
    if len(script_paths) > 1:
        assignments.append(
            f"Compare `{Path(script_paths[0]).name}` and `{Path(script_paths[1]).name}` to see how the chapter concepts connect."
        )
    if data_paths:
        assignments.append(f"Adapt the example so it reads or validates `{Path(data_paths[0]).name}`.")
    return assignments[:4]


def build_generated_markdown(folder: Path, curriculum: dict | None, script_paths: list[str], data_paths: list[str]) -> str:
    title = (curriculum or {}).get("curriculum_title") or title_from_folder(folder)
    part = normalize_text((curriculum or {}).get("part"))
    topic = normalize_text((curriculum or {}).get("topic"))
    key_concepts = (curriculum or {}).get("key_concepts") or []

    lines = [f"# {title}", ""]

    lines.extend(["## Why this chapter matters", ""])
    if topic and key_concepts:
        lines.append(
            f"This chapter covers **{topic}** and grounds the local curriculum in practical implementations of "
            f"{', '.join(key_concepts)}."
        )
    elif topic:
        lines.append(f"This chapter covers **{topic}** using the local files already present in this folder.")
    else:
        lines.append("This chapter is assembled from the local curriculum and implementation files in this folder.")
    lines.append(
        "The preview page is generated automatically because this chapter does not yet have a dedicated markdown lesson or notebook."
    )

    if part or topic or key_concepts:
        lines.extend(["", "## Curriculum snapshot", ""])
        if part:
            lines.append(f"- Track: {part}")
        if topic:
            lines.append(f"- Topic: {topic}")
        for concept in key_concepts:
            lines.append(f"- Key concept: {concept}")

    if script_paths:
        lines.extend(["", "## Local implementation files", ""])
        for script_path in script_paths:
            metadata = inspect_python_script(script_path)
            lines.append(f"### `{Path(script_path).name}`")
            lines.append("")
            lines.append(metadata["summary"] or f"Implementation file: `{script_path}`.")
            if metadata["functions"]:
                lines.append("")
                lines.append("Key functions:")
                for function in metadata["functions"]:
                    if function["summary"]:
                        lines.append(f"- `{function['name']}`: {function['summary']}")
                    else:
                        lines.append(f"- `{function['name']}`")
            lines.append("")

    if data_paths:
        lines.extend(["## Local data assets", ""])
        for data_path in data_paths:
            lines.append(f"- `{Path(data_path).name}`")
        lines.append("")

    assignments = generated_assignments(script_paths, data_paths)
    if assignments:
        lines.extend(["## Suggested walkthrough", ""])
        for assignment in assignments:
            lines.append(f"1. {assignment}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def chapter_source_payload(folder: Path, curriculum: dict | None) -> dict:
    inventory = chapter_file_inventory(folder)
    markdown_files = inventory["markdown_files"]
    notebook_files = inventory["notebook_files"]
    script_files = inventory["script_files"]
    data_files = inventory["data_files"]

    primary_markdown = markdown_files[0] if markdown_files else None
    primary_notebook = notebook_files[0] if notebook_files else None

    script_paths = [safe_relative_path(path) for path in script_files]
    data_paths = [safe_relative_path(path) for path in data_files]

    if primary_markdown is not None:
        lesson_source_kind = "markdown"
        lesson_source_path = safe_relative_path(primary_markdown)
        markdown_text = primary_markdown.read_text(encoding="utf-8")
    elif primary_notebook is not None:
        lesson_source_kind = "notebook"
        lesson_source_path = safe_relative_path(primary_notebook)
        markdown_text = notebook_to_markdown(primary_notebook)
    else:
        lesson_source_kind = "generated"
        lesson_source_path = None
        markdown_text = build_generated_markdown(folder, curriculum, script_paths, data_paths)

    return {
        "lesson_source_kind": lesson_source_kind,
        "lesson_source_path": lesson_source_path,
        "markdown_text": markdown_text,
        "markdown_paths": [safe_relative_path(path) for path in markdown_files],
        "notebook_paths": [safe_relative_path(path) for path in notebook_files],
        "script_paths": script_paths,
        "data_paths": data_paths,
    }


def chapter_directories() -> list[Path]:
    return [
        path
        for path in REPO_ROOT.iterdir()
        if path.is_dir() and (CHAPTER_DIR_RE.match(path.name) or path.name == "PlusPlus.Ensemble_Algo")
    ]


@lru_cache(maxsize=1)
def build_chapter_index() -> list[dict]:
    chapters: list[dict] = []
    curriculum_index = parse_curriculum_index()

    for folder in sorted(chapter_directories(), key=chapter_sort_key):
        curriculum = curriculum_index.get(folder.name)
        source_payload = chapter_source_payload(folder, curriculum)
        markdown_text = source_payload["markdown_text"]

        title = extract_heading(markdown_text)
        if not title or is_generic_title(title):
            title = (curriculum or {}).get("curriculum_title") or title_from_folder(folder)

        goals = extract_section_list(markdown_text, "Learning goals") or (curriculum or {}).get("key_concepts", [])
        assignments = extract_section_list(markdown_text, "Assignment")
        if not assignments and source_payload["lesson_source_kind"] == "generated":
            assignments = generated_assignments(source_payload["script_paths"], source_payload["data_paths"])

        summary = extract_first_paragraph(markdown_text)
        if is_low_signal_summary(summary):
            summary = curriculum_summary(curriculum)
        if is_low_signal_summary(summary) and source_payload["script_paths"]:
            summary = inspect_python_script(source_payload["script_paths"][0])["summary"]
        if is_low_signal_summary(summary):
            summary = "Local curriculum preview available."

        match = CHAPTER_DIR_RE.match(folder.name)
        number = int(match.group("number")) if match else None

        chapters.append(
            {
                "slug": folder.name,
                "number": number,
                "folder": folder.name,
                "title": title,
                "summary": summary,
                "goals": goals,
                "assignments": assignments,
                "part": (curriculum or {}).get("part"),
                "topic": (curriculum or {}).get("topic"),
                "key_concepts": (curriculum or {}).get("key_concepts", []),
                "curriculum_path": CURRICULUM_SOURCE_PATH,
                "lesson_source_kind": source_payload["lesson_source_kind"],
                "lesson_source_path": source_payload["lesson_source_path"],
                "markdown_path": source_payload["markdown_paths"][0] if source_payload["markdown_paths"] else None,
                "notebook_path": source_payload["notebook_paths"][0] if source_payload["notebook_paths"] else None,
                "markdown_paths": source_payload["markdown_paths"],
                "notebook_paths": source_payload["notebook_paths"],
                "script_paths": source_payload["script_paths"],
                "data_paths": source_payload["data_paths"],
            }
        )

    return chapters


def load_chapter_detail(slug: str) -> dict | None:
    chapters = build_chapter_index()
    target = next((chapter for chapter in chapters if chapter["slug"] == slug), None)
    if not target:
        return None

    folder = REPO_ROOT / target["folder"]
    curriculum = parse_curriculum_index().get(target["slug"])
    source_payload = chapter_source_payload(folder, curriculum)
    markdown_text = source_payload["markdown_text"]

    detail = dict(target)
    detail["lesson_source_kind"] = source_payload["lesson_source_kind"]
    detail["lesson_source_path"] = source_payload["lesson_source_path"]
    detail["markdown_path"] = source_payload["markdown_paths"][0] if source_payload["markdown_paths"] else None
    detail["notebook_path"] = source_payload["notebook_paths"][0] if source_payload["notebook_paths"] else None
    detail["markdown_paths"] = source_payload["markdown_paths"]
    detail["notebook_paths"] = source_payload["notebook_paths"]
    detail["script_paths"] = source_payload["script_paths"]
    detail["data_paths"] = source_payload["data_paths"]
    detail["html"] = render_markdown(markdown_text)
    detail["markdown"] = markdown_text
    return detail


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class PreviewHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        clean = parsed.path.lstrip("/") or "index.html"
        return str((STATIC_DIR / clean).resolve())

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/chapters":
            self.respond_json({"chapters": build_chapter_index()})
            return
        if parsed.path.startswith("/api/chapters/"):
            slug = parsed.path.split("/api/chapters/", 1)[1]
            detail = load_chapter_detail(unquote(slug))
            if detail is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Chapter not found")
                return
            self.respond_json(detail)
            return
        if parsed.path == "/api/file":
            self.handle_file(parsed)
            return
        if parsed.path == "/health":
            self.respond_json({"ok": True})
            return

        if parsed.path in ("/", "/index.html"):
            self.path = "/index.html"
        return super().do_GET()

    def handle_file(self, parsed) -> None:
        query = parse_qs(parsed.query)
        relative_path = query.get("path", [""])[0]
        if not relative_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing path")
            return

        candidate = (REPO_ROOT / relative_path).resolve()
        try:
            candidate.relative_to(REPO_ROOT)
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN, "Invalid path")
            return

        if not candidate.exists() or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        mime_type, _ = mimetypes.guess_type(candidate.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type or 'text/plain'}; charset=utf-8")
        self.end_headers()
        self.wfile.write(candidate.read_bytes())

    def respond_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local preview server for the course website.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()

    server = ReusableThreadingHTTPServer((args.host, args.port), PreviewHandler)
    print(f"Preview available at http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
