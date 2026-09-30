# %%
"""
plot_trajectories_JB9.py
Load specified ViRMEn sessions for one mouse, plot single-trial trajectories
per session, then the trial-averaged trajectory across sessions.

Standalone: needs only numpy, scipy, pandas, matplotlib.
Run in VSCode with the Python extension (the Run button), or run the `# %%` cells
in the Python Interactive window to view/zoom the plots.
"""
import os
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

# ----------------------------- USER CONFIG -----------------------------
DATA_ROOT = r"Y:\virmen\JB9"

# List the exact sessions you want (date_folder, session_folder):
SESSIONS = [
    ("260701", "session_1"),
    ("260714", "session_1"),
]

EXCLUDE_ITI = True     # drop inter-trial-interval samples from trajectories
N_RESAMPLE  = 100      # points per trial when averaging
# -----------------------------------------------------------------------

# Column layout from mouse_imaging/options.py -> ops['virmen_mat_columns']
VIRMEN_COLUMNS = ['world_id', 'dx', 'dy', 'dh', 'x', 'y', 'h_int',
                  'inITI', 'reward', 'dt', 'lick', 'trial']


def wrap_h(h):
    """Wrap angle to [-pi, pi]. (Replicates functions.wrap_h; not needed for
    x/y trajectories, included to mirror the repo's loading.)"""
    return (np.asarray(h) + np.pi) % (2 * np.pi) - np.pi


def load_vr(session_mat):
    """Replicate session.py:load_vr for a local sessionData.mat (MAT v5)."""
    mat = scipy.io.loadmat(session_mat)
    arr = np.asarray(mat['sessionData']).T          # (n_channels, n_samples) -> (n_samples, n_channels)
    cols = list(VIRMEN_COLUMNS)
    ncols = arr.shape[1]
    if ncols > len(cols):                            # extra user channels sit before final 'trial' col
        extra = [f'user{i}' for i in range(ncols - len(cols))]
        cols = cols[:-1] + extra + cols[-1:]
    elif ncols < len(cols):
        raise ValueError(f"{session_mat}: {ncols} channels < {len(cols)} expected columns")
    df = pd.DataFrame(arr, columns=cols)
    df['h'] = wrap_h(df['h_int']) * -1               # clockwise = positive (repo convention)
    df['t'] = df['dt'].cumsum()
    df['trial'] = df['trial'].astype(int)
    df['inITI'] = df['inITI'] > 0
    return df


def session_path(date, session):
    return os.path.join(DATA_ROOT, date, session, 'sessionData.mat')


def trial_ids(df):
    ids = df.loc[~df['inITI'], 'trial'] if EXCLUDE_ITI else df['trial']
    return sorted(i for i in ids.unique() if i > 0)


def trial_xy(df, trial):
    sel = df['trial'] == trial
    if EXCLUDE_ITI:
        sel &= ~df['inITI']
    return df.loc[sel, 'x'].to_numpy(), df.loc[sel, 'y'].to_numpy()


def resample_trial(x, y, n=N_RESAMPLE):
    """Reparameterize by normalized cumulative path length -> n points.
    (Alternative for linear tracks: interpolate x onto a fixed y grid.)"""
    if len(x) < 2:
        return None
    d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    if d[-1] == 0:
        return None
    u = d / d[-1]
    ui = np.linspace(0, 1, n)
    return np.interp(ui, u, x), np.interp(ui, u, y)


# %%  Load the sessions
sessions = []
for date, sess in SESSIONS:
    path = session_path(date, sess)
    df = load_vr(path)
    sessions.append({'label': f'{date}/{sess}', 'df': df,
                     'trials': trial_ids(df)})
    print(f"Loaded {path}: {len(df)} samples, {len(sessions[-1]['trials'])} trials")


# %%  Figure 1 - single-trial trajectories, one subplot per session
n = len(sessions)
fig1, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
for ax, s in zip(axes[0], sessions):
    tids = s['trials']
    cmap = plt.cm.viridis(np.linspace(0, 1, max(len(tids), 1)))
    for c, tid in zip(cmap, tids):
        x, y = trial_xy(s['df'], tid)
        if len(x) > 1:
            ax.plot(x, y, color=c, lw=0.7, alpha=0.7)
    ax.set_title(f"{s['label']}  ({len(tids)} trials)")
    ax.set_xlabel('x (virmen units)'); ax.set_ylabel('y (virmen units)')
    ax.set_aspect('equal', 'box')
fig1.suptitle('Single-trial trajectories')
fig1.tight_layout()


# %%  Figure 2 - trial-averaged trajectory per session (overlaid)
fig2, ax = plt.subplots(figsize=(6, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for s, col in zip(sessions, colors):
    stack = [resample_trial(*trial_xy(s['df'], t)) for t in s['trials']]
    stack = [r for r in stack if r is not None]
    if not stack:
        continue
    xs = np.vstack([r[0] for r in stack]); ys = np.vstack([r[1] for r in stack])
    mx, my = xs.mean(0), ys.mean(0)
    sx, sy = xs.std(0),  ys.std(0)
    ax.plot(mx, my, color=col, lw=2, label=f"{s['label']} (mean of {len(stack)})")
    ax.fill_betweenx(my, mx - sx, mx + sx, color=col, alpha=0.15)  # +/-1 SD in x
ax.set_title('Trial-averaged trajectory')
ax.set_xlabel('x (virmen units)'); ax.set_ylabel('y (virmen units)')
ax.set_aspect('equal', 'box'); ax.legend()
fig2.tight_layout()

plt.show()
