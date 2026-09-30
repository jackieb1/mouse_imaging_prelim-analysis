r"""Re-run the analysis notebook headlessly and rewrite every figure in figures/.

Use this after editing a plotting cell instead of reopening the notebook and running all cells
by hand. It executes the notebook's code cells in order with a non-interactive matplotlib
backend, so plt.show() becomes a no-op and each save_fig() writes its SVG/PDF/PNG.

    python regenerate_figures.py                  # all figures
    python regenerate_figures.py --list           # show the code cells, run nothing
    python regenerate_figures.py --upto 30        # stop after cell 30 (quick iteration)

IMPORTANT: activate the conda env first --

    conda activate imaging_VRbehavior
    python regenerate_figures.py

Calling the interpreter by its full path without activating leaves the env's Library/bin off
PATH, and matplotlib dies with a delay-load DLL error (0xC06D007F) as soon as it rasterises.

Y: is only read if a cache is missing: the trial, trajectory and per-session sample caches all
sit next to the notebook.
"""
import argparse, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('notebook', nargs='?', default='prelim_learning_analysis.ipynb')
ap.add_argument('--upto', type=int, default=None, help='stop after this cell index')
ap.add_argument('--list', action='store_true', help='list code cells and exit')
args = ap.parse_args()

nb_path = args.notebook if os.path.isabs(args.notebook) else os.path.join(HERE, args.notebook)
if not os.path.exists(nb_path):
    sys.exit("notebook not found: %s" % nb_path)

os.chdir(HERE)                      # so CACHE_DIR inside the notebook resolves to the repo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def strip_magics(src):
    """Blank IPython magic / shell lines (%matplotlib, !pip, ...) so exec(compile(...)) works.

    Cells are run as plain Python, where a magic line is a SyntaxError that would abort the
    whole run. But `%` also starts a continuation line of a %-format expression, and `!` can
    open a `!=`, so blanking on sight corrupts valid code. Hence: if the cell already compiles,
    it is returned untouched, and the stripping is only ever a rescue for a cell that does not.
    """
    try:
        compile(src, '<cell>', 'exec')
        return src
    except SyntaxError:
        pass
    stripped = "\n".join('' if re.match(r'^\s*(%{1,2}|!)[A-Za-z]', ln) else ln
                          for ln in src.split('\n'))
    try:
        compile(stripped, '<cell>', 'exec')
    except SyntaxError:
        return src          # stripping did not help -- let the real error surface
    return stripped


nb = json.load(open(nb_path, encoding='utf-8'))
cells = [(i, strip_magics("".join(c['source'])))
         for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']

if args.list:
    for i, src in cells:
        first = next((l for l in src.split('\n')
                      if l.strip() and not l.strip().startswith('#')), '')
        print("%3d  %s" % (i, first.strip()[:88]))
    sys.exit(0)

plt.show = lambda *a, **k: plt.close('all')          # figures are written by save_fig()
env = {'__name__': '__main__'}
t0 = time.time()
for i, src in cells:
    if args.upto is not None and i > args.upto:
        break
    try:
        exec(compile(src, 'cell%d' % i, 'exec'), env)
    except Exception:
        import traceback
        print("\n!!! cell %d failed" % i, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

figdir = os.path.join(HERE, 'figures')
figs = sorted(os.listdir(figdir)) if os.path.isdir(figdir) else []
print("\nOK - %d code cells in %.0f s" % (len(cells), time.time() - t0))
print("figures/ now holds %d files:" % len(figs))
for f in figs:
    print("   " + f)
