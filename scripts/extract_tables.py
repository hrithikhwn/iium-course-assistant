"""Extract course tables from the KICT study plan PDFs into one CSV per plan.

    python scripts/extract_tables.py

Reads the five study plans under data/raw/ and writes data/courses/<folder>.csv
with columns: plan, programme, category, course_code, course_name,
credit_hours, prerequisite, offering_dept, page (the PDF page, for citations).

The output is a first draft: check every CSV by hand against the PDF before
using it. Rows the script was unsure about are printed at the end.

How it works
- pdfplumber finds each ruled table on a page, so tables that sit side by side
  (the Rev 3 layout) are read separately instead of being merged line by line.
- The header row ("Course Code", "Course Name", ...) gives the left edge of
  each column. Every word goes to the column it starts in. Working with word
  positions rather than pdfplumber's cells copes with the ragged,
  Word-generated grids in the 201 plans and with names that overflow into the
  next cell.
- Rev 3 and 261 tables: one ruled row is one course.
- 201 tables have no credit column ("Usrah 1 (0.5)") and their ruled rows do
  not line up with the courses, so each section is cut at the "(n)" credit
  markers instead (see segment_courses).
- Title rows above the header ("SOFTWARE ENGINEERING (21 CREDIT HOURS)") and
  full-width title rows below it ("INDUSTRIAL ATTACHMENT") become the
  `category` of the courses under them.
- Semester-by-semester study plan tables are skipped: they repeat courses that
  are already listed in the programme structure tables.
"""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "courses"

# folder under data/raw -> study plan file in that folder
PLANS = {
    "bcs-201": "STUDY PLAN FOR BCS BATCH 201.pdf",
    "bit-201": "STUDY PLAN FOR BIT BATCH 201.pdf",
    "bcs-241": "BCS Study Plan (Batch 241) Rev_3.pdf",
    "bit-241": "BIT-Study-Plan-Batch-241-Rev-3_2.pdf",
    "bit-261": "BIT-PROGRAMME-STRUCTURE-AND-STUDY-PLAN-261.pdf",
}

COLUMNS = ["plan", "programme", "category", "course_code", "course_name",
           "credit_hours", "prerequisite", "offering_dept", "page"]   # page: for citations

# "BICS 1301", "LMBD1132", "BICS/BIIT XXXX"
CODE_RE = re.compile(r"\b([A-Z]{2,4}(?:/[A-Z]{2,4})?)\s?(\d{4}|X{4})\b")
SEMESTER_RE = re.compile(r"\bYEAR\s+\d\s+SEMESTER\s+\d\b", re.I)
CREDIT_IN_NAME_RE = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*\)")      # "Usrah 1 (0.5)"
CORE_RE = re.compile(r"(NON[\s-]*CORE|CORE)\s*$", re.I)
ELECTIVE_RE = re.compile(r"^(free|field) elective", re.I)
SKIP_TITLE_RE = re.compile(
    r"^total|^\(?inclusive|^\*|^\d+\s*specialization core|courses from any|^specialization$"
    r"|^\(?\s*\d+(\.\d+)?\s*credit|^malaysian? & international$", re.I)
SKIP_ROW_RE = re.compile(r"^total|^\(?inclusive|^\*|specialization core|courses from any"
                         r"|^specialization$", re.I)

# words that only ever appear in header rows
HEADER_WORDS = {"course", "code", "name", "title", "description", "credit", "credits",
                "hours", "number", "of", "pre", "requisite", "requisites", "prerequisite",
                "prerequisites", "offering", "offerin", "g", "dept", "no", "core", "non",
                "noncore", "type"}

warnings: list[str] = []


# ---------------------------------------------------------------- cleaning --

def clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def codes_in(text: str) -> list[str]:
    """All course codes in a string, normalised to 'ABCD 1234'."""
    return [f"{a} {b}" for a, b in CODE_RE.findall(text.upper())]


def clean_heading(text: str) -> str:
    """'University Required Courses (20 +2 credit hours)' -> 'University Required Courses'."""
    text = clean(text)
    text = re.sub(r"\(?\s*\d+(?:\.\d+)?\s*(?:\+\s*\d+\s*)?\)?\s*credit(?:\s*hours?|s)?\s*\)?",
                  "", text, flags=re.I)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\bENGINEERI NG\b", "ENGINEERING", text)    # broken word in BCS 201
    # "DISCIPLINE COURSE COURSES" (241, 261): their own summary tables say "Discipline Core Courses"
    text = re.sub(r"\bCOURSE COURSES\b", "CORE COURSES", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" :-")
    if text.isupper():
        text = text.title()
    return text[:1].upper() + text[1:]


def norm_credit(text: str) -> str:
    text = clean(text)
    try:
        value = float(text)
    except ValueError:
        return text
    return str(int(value)) if value.is_integer() else str(value)


def norm_prereq(text: str) -> str:
    text = clean(text)
    if text.lower() in {"-", "none", "nil"}:
        return ""
    # "CSCI 1303 CSCI 2301" -> "CSCI 1303, CSCI 2301"
    text = re.sub(r"(\d{4})\s+(?=[A-Z]{2,4}\s?\d{4})", r"\1, ", text)
    text = re.sub(r"\b([A-Z]{4})(\d{4})\b", r"\1 \2", text)       # "BIIT2305"
    text = re.sub(r"(\d{4})\.\s*(?=[A-Z]{2,4}\s?\d{4})", r"\1, ", text)   # "CSCI 1304. CSCI 2301"
    return text.strip(" ,")


def norm_word(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


# ----------------------------------------------------------------- columns --

def classify_header(text: str) -> str | None:
    t = clean(text).lower()
    if not t:
        return None
    if "requisite" in t:
        return "prerequisite"
    if "dept" in t or "offerin" in t:
        return "offering_dept"
    if re.search(r"\bcore\b", t) or t == "type":
        return "type"
    if "name" in t or "description" in t or "title" in t:
        return "course_name"     # before "credit": "Course Name (credit hours)"
    if "code" in t:
        return "course_code"
    if "credit" in t or t == "hours":
        return "credit_hours"
    if re.match(r"^no\b", t):
        return "no"
    return None


@dataclass
class Column:
    name: str
    x0: float


def build_columns(cells: list[tuple[str, tuple]], extra: list[tuple[str, tuple]] = ()) -> list[Column]:
    """Left edge of each column, from the header row's cells. `extra` is the
    second line of a two-row header ('Course' / 'Code'); its cells only add
    columns that the first row did not label, so '(number of credit hours)'
    under 'Description' does not become a credit column."""
    edges: dict[str, float] = {}
    for text, bbox in cells:
        kind = classify_header(text)
        if kind and bbox:
            edges[kind] = min(edges.get(kind, bbox[0]), bbox[0])
    for text, bbox in extra:
        kind = classify_header(text)
        if kind and bbox and kind not in edges and all(abs(x - bbox[0]) > 8 for x in edges.values()):
            edges[kind] = bbox[0]
    return sorted((Column(k, x) for k, x in edges.items()), key=lambda c: c.x0)


def column_for(cols: list[Column], x0: float) -> str | None:
    """The column a word starting at x0 belongs to."""
    chosen = None
    for c in cols:
        if x0 >= c.x0 - 2:
            chosen = c.name
    return chosen


# ------------------------------------------------------------------ tables --

@dataclass
class Course:
    category: str
    page: int
    fields: dict[str, list[str]] = field(default_factory=dict)
    merged: bool = False

    def add(self, col: str, text: str) -> None:
        self.fields.setdefault(col, []).append(text)

    def get(self, col: str) -> str:
        return clean(" ".join(self.fields.get(col, [])))


def words_by_row(page, table) -> list[list[dict]]:
    """Words of each table row, in reading order. Text flow keeps characters in
    the order they were drawn, so overflowing names are not interleaved with
    the neighbouring cell's text."""
    # a small x_tolerance keeps "1301" and "ELEMENTS" in tightly packed cells apart
    words = page.crop(table.bbox).extract_words(use_text_flow=True, x_tolerance=1.5)
    out: list[list[dict]] = [[] for _ in table.rows]
    # A row's band runs from its top to the next row's top. (A row's own bbox
    # can be taller than that when one of its cells spans two rows.)
    tops = [r.bbox[1] for r in table.rows] + [table.bbox[3] + 1]
    for w in words:
        mid = (w["top"] + w["bottom"]) / 2
        for i in range(len(table.rows)):
            if tops[i] <= mid < tops[i + 1]:
                out[i].append({**w, "band": i})   # each word goes to one row only
                break
    for mine in out:
        mine.sort(key=lambda w: (round(w["top"]), w["x0"]))
    return out


def row_cells(table) -> list[list[tuple[str, tuple | None]]]:
    texts = table.extract()
    return [[(clean(t), b) for t, b in zip(cells, row.cells)] for row, cells in zip(table.rows, texts)]


def title_of(cells: list[tuple[str, tuple | None]], before_header: bool, width: float) -> str | None:
    """Text of a section-title row, or None.

    Above the header row, any row without a course code is a title
    ('SOFTWARE ENGINEERING' | '(21 CREDIT HOURS)'). Below it, only a single
    cell that names a section counts, so a wrapped name like 'Systems (3)'
    stays part of its course."""
    filled = [t for t, _ in cells if t]
    boxes = [b for t, b in cells if t and b]
    if not filled:
        return None
    text = clean(" ".join(filled))
    if codes_in(text) or SKIP_TITLE_RE.search(text):
        return None
    if before_header:
        return text
    if len(filled) == 1 and re.search(r"credit hours?\)|required courses", text, re.I):
        return text
    if len(filled) == 1 and boxes and boxes[0][2] - boxes[0][0] > 0.6 * width:
        return text                    # merged cell across the whole table
    return None


def is_header_row(cells, nxt) -> bool:
    """'Course Code | Course Name | ...', possibly split over this row and the next."""
    own = " ".join(t for t, _ in cells if t).lower()
    if not re.search(r"\bcode\b|\bname\b|\btitle\b|\bdescription\b", own):
        return False
    joined = own + " " + " ".join(t for t, _ in (nxt or []) if t).lower()
    return "code" in joined and bool(re.search(r"name|title|description", joined))


def header_only(words: list[dict]) -> bool:
    tokens = [norm_word(w["text"]) for w in words]
    tokens = [t for t in tokens if t]
    return bool(tokens) and all(t in HEADER_WORDS for t in tokens)


def parse_table(page, table, page_no: int, page_cols: list) -> list[Course]:
    cells_by_row = row_cells(table)
    words_rows = words_by_row(page, table)
    all_text = " ".join(t for row in cells_by_row for t, _ in row)
    if SEMESTER_RE.search(all_text):
        return []                                  # semester study plan grid

    width = table.bbox[2] - table.bbox[0]
    seen_header = False
    # a table without its own header row reuses the header of the table above it
    cols: list[Column] = []
    for bbox, c in reversed(page_cols):
        if min(bbox[2], table.bbox[2]) - max(bbox[0], table.bbox[0]) > 0.5 * width:
            cols = c
            break

    courses: list[Course] = []
    titles: list[str] = []
    main = ""
    last_was_title = False
    # 201 plans have no credit column ("Usrah 1 (0.5)") and ragged rows; their
    # courses are cut out of a whole section at once, see segment_courses()
    pending: list[dict] = []
    pending_category = ""

    def flush() -> None:
        nonlocal pending
        if pending:
            courses.extend(segment_courses(pending, pending_category, page_no))
        pending = []

    for i, (cells, words) in enumerate(zip(cells_by_row, words_rows)):
        nxt = cells_by_row[i + 1] if i + 1 < len(cells_by_row) else None

        if is_header_row(cells, nxt) or (header_only(words) and not codes_in(" ".join(t for t, _ in cells))):
            header_cells = [(t, b) for t, b in cells if t and b]
            extra = []
            if nxt and header_only(words_rows[i + 1]):
                extra = [(t, b) for t, b in nxt if t and b]
            new = build_columns(header_cells, extra)
            if len({c.name for c in new} & {"course_code", "course_name"}) == 2:
                cols = new
                seen_header = True
            # header and first course can share a row: drop only the header lines
            words = [w for line in group_lines(words) if not header_only(line) for w in line]
            if not codes_in(" ".join(w["text"] for w in words)):
                last_was_title = False
                continue

        row_text = clean(" ".join(w["text"] for w in words))
        if SKIP_ROW_RE.search(row_text):
            last_was_title = False
            continue

        title = title_of(cells, not seen_header, width)
        if title:
            flush()
            cleaned = clean_heading(title)
            if last_was_title and titles and "credit" not in titles[-1].lower():
                titles[-1] += " " + title          # title wrapped onto a second row
            elif cleaned:
                titles.append(title)
            if titles and re.search(r"required courses|^university|^kulliyyah|^department",
                                    clean_heading(titles[-1]), re.I):
                main = clean_heading(titles[-1])
            last_was_title = True
            continue
        last_was_title = False

        if not cols or not words:
            continue

        heading = clean_heading(titles[-1]) if titles else ""
        category = f"{main}: {heading}" if main and heading and heading != main else (heading or main)

        if not any(c.name == "credit_hours" for c in cols):
            pending.extend({**w, "col": column_for(cols, w["x0"])} for w in words)
            pending_category = category
            continue

        # spread this row's words over the columns
        row: dict[str, list[str]] = {}
        for w in words:
            col = column_for(cols, w["x0"])
            if col:
                row.setdefault(col, []).append(w["text"])
        text = {k: clean(" ".join(v)) for k, v in row.items()}
        # codes that drifted into the "No" column of a ragged grid
        if not codes_in(text.get("course_code", "")) and codes_in(text.get("no", "")):
            text["course_code"] = text.get("course_code", "") + " " + text.pop("no")
        code_text = text.get("course_code", "")
        name = text.get("course_name", "")
        if name.lower().startswith("total") or code_text.lower().startswith("total"):
            continue

        if codes_in(code_text) or ELECTIVE_RE.match(name):
            courses.extend(split_row(text, category, page_no))
        elif courses and courses[-1].page == page_no:
            for k, v in text.items():               # continuation of the course above
                if k != "no":
                    courses[-1].add(k, v)

    flush()
    page_cols.append((table.bbox, cols))
    return courses


def group_lines(words: list[dict], tol: float = 2.0) -> list[list[dict]]:
    """Group words into text lines by their vertical position."""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(w["top"] - lines[-1][0]["top"]) <= tol:
            lines[-1].append(w)
        else:
            lines.append([w])
    return [sorted(line, key=lambda w: w["x0"]) for line in lines]


def blocks(words: list[dict]) -> list[list[dict]]:
    """Lines of one column, grouped into the pieces that belong together.

    A line with a course code stands alone ('CSCI 1305,' / 'INFO 2304' can be
    two courses' prerequisites in a merged row). Consecutive lines without a
    code are one piece ('4th' / 'year' / 'student', 'UNGS' / '1301'), placed
    by its middle rather than line by line."""
    out: list[list[dict]] = []
    prev_coded = True
    for line in group_lines(words):
        text = " ".join(w["text"] for w in line)
        coded = bool(codes_in(text))
        touching = out and line[0]["top"] - max(w["bottom"] for w in out[-1]) < 4
        same_row = out and line[0]["band"] == out[-1][-1]["band"]
        # a line ending in "," or "." carries on: "CSCI 1304," / "CSCI 2301,"
        continues = bool(out) and out[-1][-1]["text"].endswith((",", "."))
        if touching and same_row and (continues or (not coded and not prev_coded)):
            out[-1].extend(line)
        else:
            out.append(list(line))
        prev_coded = coded
    return out


def segment_courses(words: list[dict], category: str, page_no: int) -> list[Course]:
    """Cut a section of a 201-style table into courses.

    In these tables a course's name always ends with its credits, "(3)", but
    the ruled rows do not line up with the courses (codes are centred against
    two-line names, two courses can share a cell). So: read the name column
    top to bottom, end a course at each "(n)", and give every other word
    (code, prerequisite, dept) to the course whose name sits at its height."""
    end_re = re.compile(r"\(\s*\d+(?:\.\d+)?\s*\)\s*$")
    # tops of the code lines, to catch a name that is missing its "(n)"
    code_tops = [line[0]["top"] for line in group_lines(
        [w for w in words if w["col"] in ("course_code", "no")])
        if codes_in(" ".join(w["text"] for w in line))]
    segments: list[list] = []          # [top, bottom, [name lines]]
    current = None
    for line in group_lines([w for w in words if w["col"] == "course_name"]):
        text = " ".join(w["text"] for w in line)
        top, bottom = min(w["top"] for w in line), max(w["bottom"] for w in line)
        if current is not None and any(abs(t - top) <= 2 for t in code_tops):
            segments.append(current)   # a new code starts on this line
            current = None
        if current is None:
            current = [top, bottom, [text]]
        else:
            current[1] = bottom
            current[2].append(text)
        if end_re.search(text):
            segments.append(current)
            current = None
    if current:
        starts_course = any(abs(t - current[0]) <= 2 for t in code_tops)
        if segments and not starts_course:   # trailing line without "(n)", e.g. "Malaysian"
            segments[-1][1] = current[1]
            segments[-1][2] += current[2]
        else:
            segments.append(current)
    if not segments:
        return []

    bounds = [(a[1] + b[0]) / 2 for a, b in zip(segments, segments[1:])]
    courses = [Course(category, page_no) for _ in segments]
    for course, seg in zip(courses, segments):
        course.add("course_name", " ".join(seg[2]))
    for col in {w["col"] for w in words} - {None, "course_name"}:
        for block in blocks([w for w in words if w["col"] == col]):
            mid = sum((w["top"] + w["bottom"]) / 2 for w in block) / len(block)
            k = sum(1 for b in bounds if b < mid)
            courses[k].add(col, " ".join(w["text"] for w in block))
    for course in courses:
        # codes that drifted into the "No" column of a ragged grid
        if not codes_in(course.get("course_code")) and codes_in(course.get("no")):
            course.fields["course_code"] = course.fields.pop("no")
    return courses


def split_row(text: dict[str, str], category: str, page_no: int) -> list[Course]:
    """One table row usually holds one course, but the 201 plans sometimes
    merge two ('INFO 4331 INFO 4333'). Split them when the names can be cut
    at their '(3)' credit markers."""
    code_text = text.get("course_code", "")
    codes = codes_in(code_text)
    name = text.get("course_name", "")
    markers = list(CREDIT_IN_NAME_RE.finditer(name))

    if len(codes) > 1 and len(markers) == len(codes) and markers[-1].end() >= len(name.rstrip()) - 1:
        names, start = [], 0
        for m in markers:
            names.append(name[start:m.end()].strip())
            start = m.end()
        prereqs = split_prereqs(text.get("prerequisite", ""), len(codes))
        out = []
        for code, nm, pre in zip(codes, names, prereqs):
            c = Course(category, page_no, merged=True)
            c.add("course_code", code)
            c.add("course_name", nm)
            if pre:
                c.add("prerequisite", pre)
            if text.get("offering_dept"):
                c.add("offering_dept", text["offering_dept"])
            out.append(c)
        return out

    c = Course(category, page_no)
    for k, v in text.items():
        if k != "no":
            c.add(k, v)
    return [c]


def split_prereqs(text: str, n: int) -> list[str]:
    """'INFO 3304, INFO 3308 INFO 3304' over 2 courses -> ['INFO 3304, INFO 3308', 'INFO 3304']."""
    parts = re.split(r"(?<=\d{4})\s+(?=[A-Z]{2,4}\s?\d{4})", text) if text else []
    groups: list[str] = []
    for p in parts:
        if groups and groups[-1].endswith(","):
            groups[-1] += " " + p
        else:
            groups.append(p)
    if len(groups) == n:
        return groups
    return [text] * n


# ------------------------------------------------------------------ output --

def to_row(c: Course, plan: str, programme: str, source: str) -> dict:
    codes = codes_in(c.get("course_code"))
    code = "/".join(dict.fromkeys(codes)) if codes else c.get("course_code")
    name = c.get("course_name")
    credit = norm_credit(c.get("credit_hours"))
    kind = c.get("type")
    where = f"{source} p{c.page}"

    # name text that overflowed into the Core/Non-core cell: "... MODELLI" + "NG CORE"
    m = CORE_RE.search(kind) if kind else None
    if m and m.start() > 0:
        spill = kind[: m.start()].strip()
        warnings.append(f"{where}: re-joined overflowing name: '{name}' + '{spill}'")
        name = f"{name} {spill}"        # words are kept whole, so the spill is a full word
        kind = m.group(1)

    # overflow that touches the next cell: "Integrated Business Process and ERP SystemsCore"
    glued = re.search(r"(?<=[a-z])(Non[\s-]*Core|Core)$", name)
    if glued and not kind:
        warnings.append(f"{where}: split 'Core' off the end of '{name}'")
        kind, name = glued.group(1), name[: glued.start()]

    if not credit:                      # 201 plans: credits are in the name, "Usrah 1 (0.5)"
        found = CREDIT_IN_NAME_RE.findall(name)
        if found:
            credit = norm_credit(found[-1]) if len(found) == 1 else "+".join(norm_credit(f) for f in found)
            name = clean(CREDIT_IN_NAME_RE.sub("", name))

    category = c.category
    if kind:
        category += " (Non-core)" if kind.lower().startswith("non") else " (Core)"

    if not codes and not ELECTIVE_RE.match(name):
        warnings.append(f"{where}: no course code: '{name}' [{category}]")
    if len(codes) > 1:
        warnings.append(f"{where}: several codes in one row: {code} - {name}")
    if c.merged:
        warnings.append(f"{where}: split from a merged row, check prerequisite: {code} - {name}")
    if not credit:
        warnings.append(f"{where}: no credit hours: {code} - {name}")
    if not category:
        warnings.append(f"{where}: no category: {code} - {name}")

    return {
        "plan": plan,
        "programme": programme,
        "category": category,
        "course_code": code,
        "course_name": name,
        "credit_hours": credit,
        "prerequisite": norm_prereq(c.get("prerequisite")),
        "offering_dept": c.get("offering_dept"),
        "page": c.page,
        "_raw_code": c.get("course_code"),     # not written; used by fix_skills
    }


# Typos in the source PDFs, fixed after extraction so a re-run keeps them.
# Each fix must be backed by the KICT documents themselves.
# (folder, course code, course name starts with, field, value in the PDF,
#  corrected value, evidence)
ARABIC_1 = "201 plans code it LQAD 1003; Arabic for Ibadah 2 lists LQAD 1003 as its prerequisite"
CORRECTIONS = [
    ("bcs-241", "LQAD 2003", "Arabic for Ibadah 1", "course_code", "LQAD 2003", "LQAD 1003", ARABIC_1),
    ("bit-241", "LQAD 2003", "Arabic for Ibadah 1", "course_code", "LQAD 2003", "LQAD 1003", ARABIC_1),
    ("bcs-241", "UNGS 2380", "", "prerequisite", "UNGS 2380", "UNGS 2290",
     "the BCS 241 semester plan and the BIT 241 plan both list UNGS 2290"),
    ("bit-201", "SCSH 2163", "", "prerequisite", "SCHS 1201", "SCSH 1201",
     "the course in the same table is coded SCSH 1201"),
    ("bit-201", "SCSH 3164", "", "prerequisite", "SCHS 2163", "SCSH 2163",
     "the course in the same table is coded SCSH 2163"),
    ("bit-201", "INFO 4335", "", "credit_hours", "", "3",
     "listed as INFO 4335 (3) in the Enterprise Technology Management table"),
    ("bit-201", "INFO 4340", "", "credit_hours", "", "3",
     "listed as INFO 4340 (3) in the Information Assurance and Security table"),
]

# Rows that list the Malaysian and international versions of one course
# together. Split into one row per code; the PDF gives the names in the same
# order as the codes. (folder, codes as extracted, [(code, name), ...])
SPLITS = [
    ("bit-261", "MPUG 3122/UNGS 3122", [
        ("MPUG 3122", "FALSAFAH DAN CABARAN SEMASA (MALAYSIAN)"),
        ("UNGS 3122", "PHILOSOPHY AND CONTEMPORARY CHALLENGES (INTERNATIONAL)")]),
    ("bit-261", "MPUG 3112/MPUG 3142", [
        ("MPUG 3112", "ASPIRASI NEGARA BANGSA (MALAYSIAN)"),
        ("MPUG 3142", "MALAYSIAN CITRA (INTERNATIONAL)")]),
    ("bit-261", "LMBD 3204/MPUG 3152/MPUG 3162", [
        ("LMBD 3204", "BAHASA MELAYU KERJAYA (SCIENCE AND TECH) (MALAYSIAN)"),
        ("MPUG 3152", "BAHASA MELAYU KOMUNIKASI (NUSANTARA) (INDONESIA, SINGAPORE, BRUNEI)"),
        ("MPUG 3162", "BAHASA MELAYU KOMUNIKASI (OTHER INTERNATIONAL)")]),
]
corrections_applied: list[str] = []


def apply_corrections(rows: list[dict], folder: str, source: str) -> None:
    for plan, code, name, col, wrong, right, why in CORRECTIONS:
        if plan != folder:
            continue
        hits = [r for r in rows if r["course_code"] == code and r[col] == wrong
                and r["course_name"].startswith(name)]
        for r in hits:
            r[col] = right
        if hits:
            corrections_applied.append(f"{folder}: {code} {col} '{wrong}' -> '{right}' ({why})")
            if col == "credit_hours":
                warnings[:] = [w for w in warnings
                               if not (w.startswith(source) and f"no credit hours: {code} " in w)]
        else:
            warnings.append(f"{folder}: correction not applied, {code} with {col}='{wrong}' not found")

    for plan, combined, parts in SPLITS:
        if plan != folder:
            continue
        idx = next((i for i, r in enumerate(rows) if r["course_code"] == combined), None)
        if idx is None:
            warnings.append(f"{folder}: split not applied, {combined} not found")
            continue
        row = rows.pop(idx)
        rows[idx:idx] = [{**row, "course_code": c, "course_name": n} for c, n in parts]
        corrections_applied.append(f"{folder}: split {combined} into {len(parts)} rows")
        warnings[:] = [w for w in warnings
                       if not (w.startswith(source) and f"several codes in one row: {combined} " in w)]


SKILLS_RE = re.compile(r"^skills? [12]$", re.I)


def fix_skills(rows: list[dict], source: str) -> None:
    """Skills 1 and Skills 2 share one cell ('CCSS 1020, CCSS 2010, CCSS 3010')
    in the 201 and 241 plans, so the extracted codes and credits land on one
    row or the other. The semester plans list each as 0.5 credit hours with
    the same three codes; give both rows that."""
    skills = [r for r in rows if SKILLS_RE.match(r["course_name"])]
    # the code text can wrap from one row into the other: "CCSS 1020, CCSS" / "2010, CCSS 3010"
    codes = list(dict.fromkeys(codes_in(" ".join(r["_raw_code"] for r in skills))))
    if len(skills) != 2 or not codes:
        return
    dept = next((r["offering_dept"] for r in skills if r["offering_dept"]), "")
    for r in skills:
        r.update(course_code="/".join(codes), credit_hours="0.5", offering_dept=dept)
    # the per-row warnings about these two rows no longer apply
    warnings[:] = [w for w in warnings
                   if not (w.startswith(source) and re.search(r"[-'] ?Skills [12]\b", w))]


def page_heading(page) -> tuple[str, float]:
    """Heading above the elective tables, and where it sits on the page."""
    for pattern, label in [(r"FIELD ELECTIVES IN CONCENTRATION AREAS|LIST OF CONCENTRATION", "Field Electives"),
                           (r"LIST OF SPECIALI[SZ]ATIONS?", "Specialization")]:
        hits = page.search(pattern, regex=True, case=False)
        if hits:
            return label, hits[0]["top"]
    return "", 0.0


def extract(folder: str, filename: str) -> list[dict]:
    programme, plan = folder.split("-")
    rows: list[dict] = []
    with pdfplumber.open(RAW / folder / filename) as pdf:
        for page_no, page in enumerate(pdf.pages, 1):
            heading, heading_top = page_heading(page)
            page_cols: list = []
            # top-to-bottom, then left-to-right, so side-by-side tables keep their order
            tables = sorted(page.find_tables(), key=lambda t: (round(t.bbox[1] / 25), t.bbox[0]))
            for table in tables:
                for course in parse_table(page, table, page_no, page_cols):
                    if heading and table.bbox[3] > heading_top:
                        course.category = f"{heading}: {course.category}" if course.category else heading
                    rows.append(to_row(course, plan, programme.upper(), filename))
    fix_skills(rows, filename)
    apply_corrections(rows, folder, filename)
    return rows


# ------------------------------------------------------------------- tidy --
# Cosmetic only: make every plan read like the Rev 3 (241) plans. Facts
# (codes, credits, prerequisites) are never changed here.

# where a course appears in several plans, its name is taken from the first
# plan in this order that has it
NAME_SOURCE_ORDER = ["bcs-241", "bit-241", "bcs-201", "bit-201", "bit-261"]
SMALL_WORDS = {"a", "an", "and", "as", "at", "for", "in", "of", "on", "or", "the", "to", "with",
               "dan"}    # Malay "and": "Falsafah dan Cabaran Semasa"
KEEP_UPPER = {"ICT", "IT", "ERP", "CGI", "AI", "SD", "BCS", "BIT", "FYP", "IOT"}
SPECIAL_CASE = {"IDEX": "IDEx", "UNICORE": "UniCORE"}
ROMAN_RE = re.compile(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X)$")


def title_word(word: str, first: bool) -> str:
    """'(SCIENCE' -> '(Science', 'AL-QURAN' -> 'Al-Quran', 'IDEX)' -> 'IDEx)'."""
    m = re.match(r"^([^A-Za-z0-9]*)(.*?)([^A-Za-z0-9]*)$", word)
    pre, core, post = m.groups()
    if not core:
        return word
    upper = core.upper()
    if upper in SPECIAL_CASE:
        core = SPECIAL_CASE[upper]
    elif upper in KEEP_UPPER or ROMAN_RE.match(upper):
        core = upper
    elif "-" in core:
        core = "-".join(title_word(p, first and i == 0) for i, p in enumerate(core.split("-")))
    elif core.lower() in SMALL_WORDS and not first and not pre:
        core = core.lower()
    else:
        core = core[:1].upper() + core[1:].lower()
    return pre + core + post


def smart_title(text: str) -> str:
    """'BUSINESS INTELLIGENCE AND DATA MODELLING' -> 'Business Intelligence and Data Modelling'."""
    return " ".join(title_word(w, i == 0) for i, w in enumerate(text.split()))


def tidy_punctuation(name: str) -> str:
    """'Issues , Principles' -> 'Issues, Principles'; '1( SD: x' -> '1 (SD: x)'."""
    name = name.replace("*", "")                    # footnote marker: "International only"
    name = re.sub(r"\s+,", ",", name)
    name = re.sub(r"(?<=\w)\(", " (", name)
    name = re.sub(r"\(\s+", "(", name)
    name = re.sub(r"\s+\)", ")", name)
    if name.count("(") > name.count(")"):
        name += ")"
    return re.sub(r"\s+", " ", name).strip()


def norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def same_course_name(name: str, other: str) -> bool:
    """Same wording apart from case and punctuation, or `other` only adds a
    bracketed note: 'Bahasa Melayu I' / 'Bahasa Melayu I (International Only)',
    'Bahasa Melayu Kerjaya Malaysian' / 'Bahasa Melayu Kerjaya (Malaysian Only)'.
    'Block Chain and Application' / 'Blockchain and Applications' is not."""
    a, b = norm_name(name), norm_name(other)
    if not a:
        return False
    if a == b:
        return True
    stem = norm_name(re.sub(r"\s*\([^()]*\)\s*$", "", other))
    return stem != b and b.startswith(a) and a.startswith(stem)


def fix_word_case(name: str) -> str:
    """'Web application Development' -> 'Web Application Development',
    'Introduction To Quantum Computing' -> 'Introduction to Quantum Computing'."""
    words = name.split()
    for i, w in enumerate(words[1:], 1):
        if w.lower() in SMALL_WORDS:
            words[i] = w.lower()
        elif w[:1].islower():
            words[i] = w[:1].upper() + w[1:]
    return " ".join(words)


def tidy_plans(plans: dict[str, list[dict]]) -> int:
    """Tidy names, categories and text prerequisites in place; return how many
    course names changed."""
    originals = [r["course_name"] for rows in plans.values() for r in rows]
    for rows in plans.values():
        for r in rows:
            r["course_name"] = tidy_punctuation(r["course_name"])
            if not r["course_name"].isupper():
                r["course_name"] = fix_word_case(r["course_name"])

    # same code, same name (or this plan's name is the start of the other
    # plan's, e.g. 'Bahasa Melayu I' / 'Bahasa Melayu I (International Only)'):
    # use the other plan's wording
    canonical: dict[str, str] = {}
    for folder in NAME_SOURCE_ORDER:
        for r in plans.get(folder, []):
            if codes_in(r["course_code"]) and "XXXX" not in r["course_code"] and not r["course_name"].isupper():
                canonical.setdefault(r["course_code"], r["course_name"])

    for rows in plans.values():
        for r in rows:
            other = canonical.get(r["course_code"])
            if other and same_course_name(r["course_name"], other):
                r["course_name"] = other
            elif r["course_name"].isupper():
                r["course_name"] = smart_title(r["course_name"])
            if r["prerequisite"].isupper() and not codes_in(r["prerequisite"]):
                r["prerequisite"] = smart_title(r["prerequisite"])      # 'COURSE SPECIFIC'
            r["category"] = smart_title(r["category"]).replace("(Non-Core)", "(Non-core)")
    tidied = [r["course_name"] for rows in plans.values() for r in rows]
    return sum(a != b for a, b in zip(originals, tidied))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for folder, filename in PLANS.items():
        if not (RAW / folder / filename).exists():
            print(f"missing: data/raw/{folder}/{filename}", file=sys.stderr)
            return 1
    plans = {folder: extract(folder, filename) for folder, filename in PLANS.items()}
    tidied = tidy_plans(plans)
    for folder, rows in plans.items():
        out = OUT / f"{folder}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{out.relative_to(ROOT).as_posix()}: {len(rows)} courses")
    print(f"\ntidied {tidied} course names to match the other plans")
    if corrections_applied:
        print(f"\n{len(corrections_applied)} corrections to the source PDFs:")
        for c in corrections_applied:
            print("  " + c)
    if warnings:
        print(f"\n{len(warnings)} rows to check by hand:")
        for w in warnings:
            print("  " + w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
