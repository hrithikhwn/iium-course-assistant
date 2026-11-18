"""Extract the semester-by-semester study plan from each KICT study plan PDF.

    python scripts/extract_semesters.py

Reads the five study plans under data/raw/ and writes
data/courses/<folder>-semesters.csv with columns: plan, programme, year,
semester, course_code, course_name, credit_hours, prerequisite, page.
A semester's total credits are rows without a course code; the plan's total
graduating credits are rows without a year and semester.

The output is a first draft, like extract_tables.py's: check every CSV by
hand against the PDF page. Each course is compared with the plan's course CSV
(data/courses/<folder>.csv, so run extract_tables.py first), and whatever
doesn't match is printed at the end.

How it works
- pdfplumber reads the cells of the ruled tables. The 241 and 261 plans have
  one table per semester, titled "YEAR n SEMESTER m". The 201 plans have one
  grid with both semesters of a year side by side, and each semester's total
  in its title ("YEAR 1 SEMESTER 1 (TOTAL CH 16.5)").
- A row's filled cells are read in order: code (if the first one is a code),
  name, type (261 only, skipped), credit hours, prerequisite. The 201 plans
  have no prerequisite column.
- Codes, credits and prerequisites stay as the page gives them, apart from the
  PDF typos extract_tables.py also corrects. A course that is in the course
  CSV once takes its name from there, so both files name it the same way.
- A cell listing alternatives ("LMBD 1132 or LM 2026", "MPUG 3112/MPUG 3142")
  becomes one row per course when the course CSV has every code, or when the
  page gives one name per code. Otherwise the codes stay in one row, joined
  with "/" as in the course CSVs ("CCSS 1020/CCSS 2010/CCSS 3010").
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_tables import (CORRECTIONS, OUT, PLANS, RAW, ROOT, clean, codes_in,  # noqa: E402
                            norm_credit, norm_prereq, smart_title)

COLUMNS = ["plan", "programme", "year", "semester", "course_code", "course_name",
           "credit_hours", "prerequisite", "page"]

TITLE_RE = re.compile(r"YEAR\s+(\d)\s+SEMESTER\s+(\d)(?:\s*\(\s*TOTAL\s+CH\s+([\d.]+(?:\s*/\s*[\d.]+)?)\s*\))?",
                      re.I)
# also "CCSS 1/2/3010" (BIT 201's Skills 2), which codes_in can't read and is kept as printed
CODE_START_RE = re.compile(r"^[A-Z]{2,4}(?:/[A-Z]{2,4})?\s?(?:\d|X{4})")
TYPE_RE = re.compile(r"^(UNICORE|CORE|DISC CORE|FE|FYP|IA)$")              # 261's "Type" column
HEADER_RE = re.compile(r"^(course|code|course code|course name|description|credit|hours|credit hours"
                       r"|pre-? ?requisite|type)$", re.I)
# the 261 plan's legend: "MY-Malaysian, I-International, M-Muslim, NM-Non-Muslim, N-Nusantara"
LEGEND = {"MY": "Malaysian", "I": "International", "M": "Muslim", "NM": "Non-Muslim", "N": "Nusantara"}

notes: list[str] = []          # printed at the end: what to check by hand


def lines(cell: str) -> list[str]:
    return [clean(line) for line in (cell or "").split("\n") if clean(line)]


def page_name(text: str) -> str:
    """'FREE ELECTIVE I' -> 'Free Elective I', 'ARABIC ... II (M)' -> 'Arabic ... II (Muslim)'."""
    text = clean(text)
    upper = text.isupper()
    text = re.sub(r"\((MY|NM|I|M|N)\)", lambda m: f"({LEGEND[m.group(1)]})", text)
    return smart_title(text) if upper else text


def page_prereq(text: str) -> str:
    text = re.sub(r"\s*/\s*", "/", norm_prereq(text))          # "LQAD 1113/ LQAD 1131"
    # 'COURSE SPECIFIC' -> 'Course Specific', but not 'CELPAD' (a centre) or codes
    return smart_title(text) if text.isupper() and " " in text and not codes_in(text) else text


def read_cells(cells: list[str]) -> dict | None:
    """One semester's part of a table row -> {code, name, credit, prereq} (raw cell texts)."""
    filled = [c for c in cells if clean(c)]
    if not filled or all(HEADER_RE.match(clean(c)) for c in filled):
        return None
    code = filled.pop(0) if CODE_START_RE.match(clean(filled[0]).upper()) else ""
    name = filled.pop(0) if filled else ""
    if filled and TYPE_RE.match(clean(filled[0])):
        filled.pop(0)
    credit = filled.pop(0) if filled else ""
    prereq = filled.pop(0) if filled else ""
    return {"code": code, "name": name, "credit": credit, "prereq": prereq}


def semester_cells(page) -> list[tuple[str, str, dict]]:
    """(year, semester, cells) for every row of the page's semester tables, in
    page order; year and semester are "" for the total graduating credits."""
    found: dict[tuple[str, str], list[dict]] = {}
    totals: dict[tuple[str, str], str] = {}
    tables = sorted(page.find_tables(), key=lambda t: (round(t.bbox[1] / 25), t.bbox[0]))
    for table in tables:
        bands: list[tuple[int, str, str]] = []          # (first column, year, semester)
        for row in table.extract():
            cells = [c or "" for c in row]
            titles = [(i, m) for i, m in ((i, TITLE_RE.search(clean(c))) for i, c in enumerate(cells)) if m]
            if titles:
                # 201: two titles side by side; the left semester starts at column 0
                bands = [(0 if k == 0 else i, m.group(1), m.group(2)) for k, (i, m) in enumerate(titles)]
                for _, m in titles:
                    if m.group(3):
                        totals[(m.group(1), m.group(2))] = re.sub(r"\s+", "", m.group(3))
                continue
            if any(re.search(r"graduating", c, re.I) for c in cells):
                bands = [(0, "", "")]
                continue
            for k, (start, year, semester) in enumerate(bands):
                end = bands[k + 1][0] if k + 1 < len(bands) else len(cells)
                read = read_cells(cells[start:end])
                if read:
                    found.setdefault((year, semester), []).append(read)
    for key, credit in totals.items():          # 201: the total from the title goes last
        found.setdefault(key, []).append({"code": "", "name": "Total Credits", "credit": credit, "prereq": ""})
    order = sorted(found, key=lambda k: (k == ("", ""), k))
    return [(year, semester, cells) for year, semester in order for cells in found[(year, semester)]]


def correct(folder: str, row: dict) -> None:
    """The fixes extract_tables.py makes to the same typos in the course tables."""
    for plan, code, name, col, wrong, right, why in CORRECTIONS:
        if (plan == folder and row["course_code"] == code and row[col] == wrong
                and row["course_name"].lower().startswith(name.lower())):
            row[col] = right
            notes.append(f"{folder}: corrected {code} {col} '{wrong}' -> '{right}' ({why})")


def to_rows(folder: str, year: str, semester: str, cells: dict, courses: dict[str, list[dict]],
            page_no: int) -> list[dict]:
    programme, plan = folder.split("-")
    printed = clean(cells["code"])
    codes = codes_in(printed) or ([printed] if printed else [])
    names, credits = lines(cells["name"]), [norm_credit(c) for c in lines(cells["credit"])]
    base = {"plan": plan, "programme": programme.upper(), "year": year, "semester": semester,
            "prerequisite": page_prereq(cells["prereq"]), "page": str(page_no)}

    def row(code: str, name: str, credit: str) -> dict:
        r = {**base, "course_code": code, "course_name": name, "credit_hours": credit}
        correct(folder, r)
        known = courses.get(r["course_code"], [])
        if len(known) == 1:                          # Skills 1 and 2 share a code: keep the page's name
            r["course_name"] = known[0]["course_name"]
        return r

    credit = "/".join(credits)
    joined = "/".join(codes)
    if not year:                                     # the graduating total, not a semester's
        return [row("", page_name(" ".join(names)).replace("Total Credits", "Total Graduating Credits", 1),
                    credit)]
    if len(codes) <= 1 or joined in courses:
        return [row(joined, page_name(" ".join(names)), credit)]
    if all(c in courses for c in codes) or len(names) == len(codes):
        each = credits if len(credits) == len(codes) else [credit] * len(codes)
        page_names = names if len(names) == len(codes) else [""] * len(codes)
        return [row(c, page_name(n), cr) for c, n, cr in zip(codes, page_names, each)]
    return [row(joined, page_name(" ".join(names)), credit)]


def semester_page(pdf) -> int:
    for no, page in enumerate(pdf.pages, 1):
        if TITLE_RE.search(page.extract_text() or ""):
            return no
    raise SystemExit("no YEAR n SEMESTER m page found")


def read_courses(folder: str) -> dict[str, list[dict]]:
    path = OUT / f"{folder}.csv"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT).as_posix()} not found: run scripts/extract_tables.py first")
    with path.open(encoding="utf-8-sig", newline="") as f:
        courses: dict[str, list[dict]] = {}
        for r in csv.DictReader(f):
            courses.setdefault(r["course_code"], []).append(r)
    return courses


def check(folder: str, rows: list[dict], courses: dict[str, list[dict]], page_codes: set[str]) -> None:
    """Note what doesn't match the course CSV, and codes on the page that no row has."""
    for r in rows:
        code, when = r["course_code"], f"Y{r['year']}S{r['semester']}"
        if not code or "XXXX" in code:
            continue
        known = courses.get(code)
        if not known:
            notes.append(f"{folder} {when}: {code} {r['course_name']} is not in {folder}.csv")
        elif r["credit_hours"] not in {k["credit_hours"] for k in known}:
            notes.append(f"{folder} {when}: {code} {r['course_name']} has {r['credit_hours']} credit hours "
                         f"here, {'/'.join(sorted({k['credit_hours'] for k in known}))} in {folder}.csv")
    placed = {r["course_code"] for r in rows}
    for code, known in courses.items():
        category = known[0]["category"]
        if code not in placed and not re.match(r"(field|free) electives|specialization", category, re.I):
            notes.append(f"{folder}: {code} {known[0]['course_name']} ({category}) is in {folder}.csv "
                         f"but in no semester")
    kept = {c for r in rows for c in codes_in(r["course_code"] + " " + r["prerequisite"])}
    for code in sorted(page_codes - kept):
        notes.append(f"{folder}: {code} is on the page but in no row: check the extraction")


def extract(folder: str, filename: str) -> list[dict]:
    courses = read_courses(folder)
    with pdfplumber.open(RAW / folder / filename) as pdf:
        page_no = semester_page(pdf)
        page = pdf.pages[page_no - 1]
        rows = [r for year, semester, cells in semester_cells(page)
                for r in to_rows(folder, year, semester, cells, courses, page_no)]
        page_codes = {c for c in codes_in(page.extract_text() or "") if "XXXX" not in c}
    check(folder, rows, courses, page_codes)
    return rows


def main() -> int:
    for folder, filename in PLANS.items():
        if not (RAW / folder / filename).exists():
            print(f"missing: data/raw/{folder}/{filename}", file=sys.stderr)
            return 1
    for folder, filename in PLANS.items():
        rows = extract(folder, filename)
        out = OUT / f"{folder}-semesters.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        semesters = len({(r["year"], r["semester"]) for r in rows if r["year"]})
        print(f"{out.relative_to(ROOT).as_posix()}: {len(rows)} rows, {semesters} semesters, page {rows[0]['page']}")
    if notes:
        print(f"\n{len(notes)} things to check by hand:")
        for n in notes:
            print("  " + n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
