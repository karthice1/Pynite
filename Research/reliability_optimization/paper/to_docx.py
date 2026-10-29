"""Build a Word (.docx) version of the manuscript from main.tex.

Run pdflatex/bibtex first so that main.aux holds the label numbers. The script flattens the
\\input files, resolves cross-references from main.aux, numbers figure, table and equation
captions, simplifies constructs pandoc cannot read, and calls pandoc with citeproc.

    python to_docx.py            # writes main.docx next to main.tex
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name):
    p = HERE / name
    if not p.suffix:
        p = p.with_suffix('.tex')
    return p.read_text()


def flatten(s):
    while True:
        s = re.sub(r'\\fittable\{(\\input\{[^}]*\})\}', r'\1', s)
        new = re.sub(r'\\input\{([^}]*)\}', lambda m: read(m.group(1)), s)
        if new == s:
            return s
        s = new


def labels():
    out = {}
    for m in re.finditer(r'\\newlabel\{([^}]*)\}\{\{([^}]*)\}', (HERE / 'main.aux').read_text()):
        out[m.group(1)] = m.group(2)
    return out


def number_floats(s, lab):
    def fix(m):
        env, body = m.group(1), m.group(2)
        lm = re.search(r'\\label\{([^}]*)\}', body)
        if lm and lm.group(1) in lab:
            name = 'Figure' if env == 'figure' else 'Table'
            body = body.replace(r'\caption{', r'\caption{%s %s. ' % (name, lab[lm.group(1)]), 1)
        return r'\begin{%s}%s\end{%s}' % (env, body, env)
    return re.sub(r'\\begin\{(figure|table)\}(.*?)\\end\{\1\}', fix, s, flags=re.S)


def number_equations(s, lab):
    """Append the LaTeX equation number to each numbered equation or align line."""
    count = [0]

    def tag(line):
        if r'\nonumber' in line or r'\notag' in line:
            return line.replace(r'\nonumber', '').replace(r'\notag', '')
        count[0] += 1
        lm = re.search(r'\\label\{([^}]*)\}', line)
        if lm and lab.get(lm.group(1)) != str(count[0]):
            raise SystemExit('equation numbering mismatch at %s' % lm.group(1))
        return line.rstrip() + r'\qquad\text{(%d)}' % count[0]

    def fix(m):
        env, body = m.group(1), m.group(2)
        if env == 'equation':
            return r'\begin{equation*}%s\end{equation*}' % tag(body)
        lines = re.split(r'\\\\', body)
        lines = [tag(ln) if ln.strip() else ln for ln in lines]
        return r'\begin{align*}%s\end{align*}' % '\\\\'.join(lines)
    return re.sub(r'\\begin\{(equation|align)\}(.*?)\\end\{\1\}', fix, s, flags=re.S)


def simplify(s, lab):
    s = re.sub(r'\\eqref\{([^}]*)\}', lambda m: '(%s)' % lab.get(m.group(1), '?'), s)
    s = re.sub(r'\\ref\{([^}]*)\}', lambda m: lab.get(m.group(1), '?'), s)
    s = re.sub(r'\\label\{[^}]*\}', '', s)
    # stacked cells: \begin{tabular}[c]{@{}r@{}}a\\b\end{tabular} -> a b
    s = re.sub(r'\\begin\{tabular\}\[c\]\{@\{\}r@\{\}\}(.*?)\\\\(.*?)\\end\{tabular\}', r'\1 \2', s)
    s = re.sub(r'\\cmidrule(\([a-z]*\))?\{[^}]*\}', '', s)
    s = re.sub(r'\\addlinespace(\[[^]]*\])?', '', s)
    s = re.sub(r'\\(small|scriptsize|footnotesize|centering)\b', '', s)
    s = s.replace(r'\path{', r'\texttt{').replace('.pdf}', '.png}')
    s = s.replace(r'\sep', ';')
    return s


def frontmatter(s, commit):
    fm = re.search(r'\\begin\{frontmatter\}(.*?)\\end\{frontmatter\}', s, re.S).group(1)
    title = re.search(r'\\title\{(.*?)\}\n', fm, re.S).group(1)
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', fm, re.S).group(1)
    keywords = re.search(r'\\begin\{keyword\}(.*?)\\end\{keyword\}', fm, re.S).group(1)
    body = s[s.index(r'\end{frontmatter}') + len(r'\end{frontmatter}'):s.index(r'\bibliographystyle')]
    body = body.replace(r'\resultscommit', commit)
    return (r'''\documentclass{article}
\title{%s}
\author{[Author name] ([Affiliation], [Country])}
\date{}
\begin{document}
\maketitle
\begin{abstract}%s\end{abstract}

\textbf{Keywords:} %s
%s
\section*{References}
\end{document}
''' % (title, abstract, keywords.strip(), body))


def polish(path):
    """Fit tables to the text width with booktabs-like rules, and size the figures."""
    import docx
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt

    d = docx.Document(path)
    for st in ('Normal', 'Body Text', 'First Paragraph', 'Compact'):
        if st in [x.name for x in d.styles]:
            d.styles[st].font.name = 'Times New Roman'
            d.styles[st].font.size = Pt(11)
    total = 9360  # twips, 6.5 in
    for t in d.tables:
        rows = t.rows
        ncol = max(len(r.cells) for r in rows)
        need = [4] * ncol
        for r in rows:
            for j, c in enumerate(r.cells[:ncol]):
                words = c.text.split() or ['']
                need[j] = max(need[j], min(28, max(len(w) for w in words) + 2, len(c.text) + 2))
        widths = [int(total * n / sum(need)) for n in need]
        tblPr = t._tbl.tblPr
        for tag in ('w:tblW', 'w:tblLayout', 'w:tblBorders'):
            for e in tblPr.findall(qn(tag)):
                tblPr.remove(e)
        w = OxmlElement('w:tblW'); w.set(qn('w:w'), str(total)); w.set(qn('w:type'), 'dxa')
        lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed')
        bd = OxmlElement('w:tblBorders')
        for side in ('top', 'bottom'):
            e = OxmlElement('w:' + side)
            e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '8'); e.set(qn('w:color'), '000000')
            bd.append(e)
        tblPr.extend([w, lay, bd])
        grid = t._tbl.tblGrid
        for g in list(grid):
            grid.remove(g)
        for wd in widths:
            g = OxmlElement('w:gridCol'); g.set(qn('w:w'), str(wd)); grid.append(g)
        for i, r in enumerate(rows):
            col = 0
            for tc in r._tr.tc_lst:
                span = tc.tcPr.find(qn('w:gridSpan')) if tc.tcPr is not None else None
                n = int(span.get(qn('w:val'))) if span is not None else 1
                tcPr = tc.get_or_add_tcPr()
                for e in tcPr.findall(qn('w:tcW')):
                    tcPr.remove(e)
                tcw = OxmlElement('w:tcW'); tcw.set(qn('w:type'), 'dxa')
                tcw.set(qn('w:w'), str(sum(widths[col:col + n])))
                col += n
                tcPr.insert(0, tcw)
                if i == 0:
                    b = OxmlElement('w:tcBorders'); e = OxmlElement('w:bottom')
                    e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), '4'); e.set(qn('w:color'), '000000')
                    b.append(e); tcPr.append(b)
                for para in tc.iter(qn('w:p')):
                    for run in para.iter(qn('w:r')):
                        rPr = run.find(qn('w:rPr'))
                        if rPr is None:
                            rPr = OxmlElement('w:rPr'); run.insert(0, rPr)
                        sz = OxmlElement('w:sz'); sz.set(qn('w:val'), '16'); rPr.append(sz)
    for sh in d.inline_shapes:
        ratio = sh.height / sh.width
        sh.width = Inches(6.3) if sh.width > Inches(3.5) else Inches(3.6)
        sh.height = int(sh.width * ratio)
    d.save(path)


def main():
    src = read('main.tex')
    commit = re.search(r'\\newcommand\{\\resultscommit\}\{([^}]*)\}', src).group(1)
    lab = labels()
    s = flatten(src)
    s = frontmatter(s, commit)
    s = number_floats(s, lab)
    s = number_equations(s, lab)
    s = simplify(s, lab)
    flat = HERE / 'main_flat_docx.tex'
    flat.write_text(s)
    subprocess.run(['pandoc', flat.name, '-f', 'latex', '-o', 'main.docx', '--citeproc',
                    '--bibliography', 'refs.bib', '--resource-path', '.:figures'],
                   cwd=HERE, check=True)
    flat.unlink()
    polish(HERE / 'main.docx')


if __name__ == '__main__':
    main()
