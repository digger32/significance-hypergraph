#!/usr/bin/env python3
"""Verify that every "quoted" passage in a response letter appears verbatim in the
compiled manuscript. A quote that has drifted from the manuscript is the classic
way a response letter loses a reviewer's trust, so this is checked mechanically.

Run: python verify_quotes.py <letter.docx> <main.pdf>
"""
import re, subprocess, sys, unicodedata, zipfile

LETTER = sys.argv[1]
PDF = sys.argv[2] if len(sys.argv) > 2 else "main.pdf"

def norm(s):
    """Collapse to bare alphanumerics: this makes the comparison immune to
    line-breaking, hyphenation and whitespace differences between the LaTeX
    source, the compiled PDF and the letter."""
    s = unicodedata.normalize("NFKD", s)
    for a, b in [("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
                 ("\u2013", "-"), ("\u2014", "-"), ("\u2212", "-"), ("\u00a0", " "),
                 ("\u00d7", "x"), ("\u03c1", "rho"), ("\u03b1", "alpha"),
                 ("\u00b1", "+-"), ("\u2026", "..."), ("\u2192", "->"),
                 ("\u2018", "'")]:
        s = s.replace(a, b)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())

# manuscript text: build the running prose from the line-numbered body lines only.
# pdftotext reads a page in visual order, so floats (tables, captions) interleave
# with the prose and would break a quote that spans them; the marginal line number
# is exactly the marker of a body-prose line.
subprocess.run(["pdftotext", "-layout", PDF, "/tmp/_vq.txt"], check=True, stderr=subprocess.DEVNULL)
def _numbered_lines(raw):
    """Yield (page, lineno, text) for body lines carrying a margin line number.
    The gap before the number varies, so candidates are disambiguated by the fact
    that line numbers increase monotonically through the document."""
    out, last = [], 0
    for pno, page in enumerate(raw.split("\f"), 1):
        for line in page.split("\n"):
            m = re.search(r"\s{2,}(\d{1,4})\s*$", line)
            if not m:
                continue
            n = int(m.group(1))
            if n <= last or n > last + 40:      # not the next body line number
                continue
            last = n
            out.append((pno, n, line[:m.start()]))
    return out

_raw = open("/tmp/_vq.txt", "rb").read().decode("utf-8", "replace")
_body = [t for _, _, t in _numbered_lines(_raw)]
man = norm(" ".join(_body))
# captions and table cells are not line-numbered; keep them in a second stream so
# a quote of caption text can still be checked
man_all = norm(_raw)

# letter text: pull the w:t runs out of the docx
xml = zipfile.ZipFile(LETTER).read("word/document.xml").decode("utf-8")
paras = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
texts = []
for p in paras:
    texts.append("".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S)))
def unesc(s):
    return (s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
             .replace("&quot;", '"').replace("&apos;", "'"))
texts = [unesc(t) for t in texts]

quotes = [t for t in texts if t.strip().startswith("\u201c") and t.strip().endswith("\u201d")]
print(f"{LETTER}: {len(quotes)} quoted passages\n")

bad = 0
for i, qraw in enumerate(quotes, 1):
    q = norm(qraw.strip().strip("\u201c\u201d"))
    if q in man or q in man_all:
        print(f"  [OK]   quote {i:2d} ({len(q.split()):3d} words): {q[:58]}...")
    else:
        bad += 1
        print(f"  [FAIL] quote {i:2d} ({len(q.split()):3d} words): {q[:58]}...")
        # locate the longest matching prefix to show where it diverges
        lo, hi = 0, len(q)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if q[:mid] in man: lo = mid
            else: hi = mid - 1
        print(f"         diverges after: ...{q[max(0,lo-60):lo]}")
        print(f"         letter has    : {q[lo:lo+60]}...")

print()
if bad:
    print(f"VERIFY FAIL: {bad} of {len(quotes)} quotes do not match the manuscript verbatim")
    sys.exit(1)
print(f"VERIFY PASS: all {len(quotes)} quotes match the compiled manuscript verbatim")
