#!/usr/bin/env python3
"""Locate each new passage in the compiled PDF and report page + line range.

The MDPI class prints line numbers in the right margin; pdftotext -layout keeps
them at the end of the line. This maps a search phrase to [p. N, ll. A-B] so the
response letters can be filled from the compiled artefact rather than by hand.

Run: python locate_lines.py main.pdf
"""
import re, subprocess, sys, unicodedata

PDF = sys.argv[1] if len(sys.argv) > 1 else "main.pdf"
subprocess.run(["pdftotext", "-layout", PDF, "/tmp/_loc.txt"], check=True,
               stderr=subprocess.DEVNULL)
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

raw = open("/tmp/_loc.txt", "rb").read().decode("utf-8", "replace")

def norm(s):
    s = unicodedata.normalize("NFKD", s)
    for a, b in [("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'), ("\u2013", "-"),
                 ("\u2014", "-"), ("\u2212", "-"), ("\u00a0", " ")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()

idx = [(p, n, norm(t)) for p, n, t in _numbered_lines(raw)]

# one running text with a char -> (page, line) map
_run, _map = [], []
for pno, lno, txt in idx:
    if _run:
        _run.append(" "); _map.append((pno, lno))
    for ch in txt:
        _run.append(ch); _map.append((pno, lno))
RUN = "".join(_run)

def find(phrase, end_phrase=None):
    """Return 'p. P, ll. A-B' for the numbered lines spanning the passage."""
    a = RUN.find(norm(phrase)[:70])
    if a < 0:
        return None
    if end_phrase is None:
        return f"p. {_map[a][0]}, l. {_map[a][1]}"
    ep = norm(end_phrase)[:60]
    b = RUN.find(ep, a)
    if b < 0:
        return f"p. {_map[a][0]}, l. {_map[a][1]}+"
    b = min(b + len(ep), len(_map) - 1)
    p0, l0 = _map[a]; p1, l1 = _map[b]
    pg = f"p. {p0}" if p0 == p1 else f"pp. {p0}-{p1}"
    return f"{pg}, ll. {l0}-{l1}" if l0 != l1 else f"{pg}, l. {l0}"

PASSAGES = [
    ("R1.1/R2 - S4.1 upstream-stack control",
     "Because our recompute runs on a newer software stack",
     "The full control is reported in Appendix"),
    ("R1.1 - Appendix C intro",
     "The recompute of Section 4.1 runs on Python 3.12.3",
     "indicates whether the rebuilt environment trains normally"),
    ("R1.1 - Appendix C conclusions",
     "Two conclusions follow. First, HJRL fails identically",
     "with the interpreter and library versions recorded in the run log"),
    ("R1.2 / truncated cells - S3.2 disclosure",
     "Seven further cells were truncated by the one-hour budget",
     "a sensitivity check for the across-datasets layer is reported"),
    ("R1.2 - S3.3 Layer 2 restriction",
     "Cells truncated by the compute budget",
     "counts only the methods with all twenty seeds on dataset"),
    ("R1.2 - S4.3 TMPHN sensitivity",
     "Six of the 153 cell means entering this block",
     "the same methods form the statistically inseparable top band"),
    ("R1.3 - S5.7 public issue",
     "We have nevertheless contacted the DHG-Bench authors",
     "incorporated in the follow-up work named in the Conclusions"),
    ("R1.4 - S5.7 variance-compression",
     "A further question is whether the default configuration could bias",
     "remains the boundary of the claim"),
    ("R1.5 - S3.2 trivago rewrite",
     "The trivago dataset required separate handling",
     "the limitations of the study"),
    ("R1.5 - Appendix B intro",
     "This appendix records the verified failure taxonomy",
     "one exceeds the GPU budget; two complete"),
    ("R1.5 - Appendix B label probe",
     "The label tensor shipped with the dataset was probed directly",
     "beyond any single-GPU budget"),
    ("R1.5 - Appendix B HNHN diagnostic",
     "The two methods that complete fall below their reported accuracies",
     "the label probe accompany the released code"),
    ("R1.5 - S5.6 third finding",
     "Third, training on trivago fails for 18 of the 20 methods",
     "does not reproduce the configuration behind the reported numbers"),
    ("R1.6/R2.3 - S4.3 CD relocation",
     "The diagram itself is placed in Appendix",
     "a compact visual summary of the average ranks"),
    ("R1.6/R2.3 - Appendix A",
     "Figure A1 shows the Nemenyi critical-difference diagram",
     "the single band that joins the leading methods"),
    ("R1.6/R2.3 - Limitations placement",
     "the diagram is accordingly provided in Appendix",
     "as a visualization rather than the primary test"),
    ("Table 3 caption note",
     "and so are the seven cells truncated by the compute budget",
     "counts the methods with all twenty seeds on each dataset"),
    ("S4.2 separability shares",
     "98% of pairs on walmart-trips-100 and 96% on coauthor_dblp",
     "collapses to 21% (36 of 171 pairs) on twitch"),
    ("S4.5 pooled count",
     "over the cells with all twenty seeds",
     "reach p < 0.05 before multiplicity correction"),
    ("S5.6 first finding (HJRL)",
     "First, HJRL does not reproduce from the public release",
     "rather than a versioning effect or a reporting error"),
    ("R2.1 - Limitations track scope",
     "The study covers the node-classification track only",
     "are left to future work"),
    ("R2.1 - Conclusions future work",
     "Future work includes extending the audit",
     "while leaving the across-datasets instability in place"),
    ("R2.2 - S5.7 default is only reproducible config",
     "The default set is therefore the only configuration reproducible",
     "every method is compared here"),
    ("R2.2 - S5.7 configuration-invariance",
     "The significance and variance analyses operate within a dataset",
     "the rankings they produce, unchanged"),
]

print(f"{'passage':46s} location")
print("-" * 82)
missing = []
for name, start, end in PASSAGES:
    loc = find(start, end)
    print(f"{name:46s} {loc}")
    if loc is None:
        missing.append(name)
if missing:
    print("\nNOT FOUND:", missing)
    raise SystemExit(1)
