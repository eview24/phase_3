"""
make_rf_video.py
================

Generate a Random Forest explainer video, matched in style to the
Bayesian Hierarchical explainer in our deck (black background,
LaTeX-style math, layered equations that build up over time).

Structure (57s at 15fps, 854x480):
    Scene 0 (0-5s):   Title card
    Scene 1 (5-15s):  Step 1 - Single decision tree
    Scene 2 (15-25s): Step 2 - Bootstrap aggregation (bagging)
    Scene 3 (25-34s): Step 3 - Feature subsampling at each split
    Scene 4 (34-42s): Step 4 - League-aware features (hierarchy analog)
    Scene 5 (42-49s): Step 5 - OOB residuals give prediction intervals
    Scene 6 (49-57s): Full Model Summary (boxed, color-coded by layer)

Each scene shows a step title at the top + stacked equations, with each
newly-introduced layer in a distinct color. Older layers are carried
forward in a faded-down version so the viewer sees the full stack grow.

Requirements:
    pip install matplotlib
    ffmpeg on PATH (Ubuntu: `sudo apt-get install ffmpeg`,
                    macOS:  `brew install ffmpeg`,
                    Windows: https://ffmpeg.org/download.html)

Run:
    python make_rf_video.py
    -> writes ./rf_video_frames/*.png (scratch) and ./rf_explainer.mp4

Notes on LaTeX:
    We use matplotlib's built-in `mathtext` (not a real TeX install) for
    portability. mathtext doesn't support `\\bigl`, `\\bigr`, `\\tfrac`,
    or negative spacing `\\!`; all math below is written to avoid those.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FPS = 15
W_IN, H_IN = 8.54, 4.80   # inches @ 100dpi -> 854x480 pixels
DPI = 100

SCRIPT_DIR = Path(__file__).resolve().parent
FRAMES_DIR = SCRIPT_DIR / "rf_video_frames"
OUT_MP4    = SCRIPT_DIR / "rf_explainer.mp4"

# Color palette (matches the Bayesian video's layer-color scheme)
C_BASE = "#FFFFFF"   # white      - base equation layer
C_L1   = "#4DD0E1"   # cyan/teal  - step 2 additions (bagging / forest)
C_L2   = "#E1BEE7"   # soft purple- step 3 additions (split sampling)
C_L3   = "#FFB74D"   # orange     - step 4 additions (league features)
C_L4   = "#A5D6A7"   # sage green - step 5 additions (OOB / intervals)
C_SUB  = "#BDBDBD"   # muted grey - subtitles / captions
C_STEP = "#E0E0E0"   # light grey - step title

# Serif look without needing a system TeX install
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "serif"],
    "mathtext.fontset": "cm",   # Computer Modern (closest to Bayesian video)
})


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def make_fig():
    fig, ax = plt.subplots(figsize=(W_IN, H_IN), dpi=DPI, facecolor="black")
    ax.set_facecolor("black")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save_frame(fig, idx):
    fig.savefig(FRAMES_DIR / f"f{idx:05d}.png",
                dpi=DPI, facecolor="black", pad_inches=0)
    plt.close(fig)


def _step_title(ax, text, alpha=1.0):
    ax.text(0.5, 0.93, text, ha="center", va="center",
            fontsize=13, color=C_STEP, alpha=alpha, family="serif")


# ---------------------------------------------------------------------------
# Scenes — each takes show_new in [0, 1] which drives the fade-in
# of the currently-introduced layer. Older layers are carried forward at
# alpha 0.5-0.55 so they read as "still there but de-emphasised".
# ---------------------------------------------------------------------------
def scene_title(show_new=1.0):
    fig, ax = make_fig()
    ax.text(0.5, 0.58, "Random Forest Regression",
            ha="center", va="center",
            fontsize=30, color=C_BASE, alpha=show_new, family="serif")
    ax.text(0.5, 0.44, "How the model is built up, layer by layer",
            ha="center", va="center",
            fontsize=14, color=C_SUB, alpha=show_new, family="serif",
            style="italic")
    return fig


def scene_step1(show_new=1.0):
    """Single decision tree: y_i = T(x_i)"""
    fig, ax = make_fig()
    _step_title(ax, "Step 1 \u2014 A single decision tree")
    ax.text(0.5, 0.55,
            r"$\hat{y}_i = T(\mathbf{x}_i)$",
            ha="center", va="center",
            fontsize=36, color=C_BASE, alpha=show_new, family="serif")
    ax.text(0.5, 0.32,
            "One tree recursively splits the feature space into regions",
            ha="center", va="center",
            fontsize=11, color=C_SUB, alpha=show_new, family="serif",
            style="italic")
    return fig


def scene_step2(show_new=1.0):
    """Bootstrap aggregation (bagging)."""
    fig, ax = make_fig()
    _step_title(ax, "Step 2 \u2014 Bootstrap aggregation (bagging)")
    # Carry forward Step 1 (faded)
    ax.text(0.5, 0.68,
            r"$\hat{y}_i = T(\mathbf{x}_i)$",
            ha="center", va="center",
            fontsize=22, color=C_BASE, alpha=0.55, family="serif")
    # New
    ax.text(0.5, 0.46,
            r"$\hat{y}_i = \frac{1}{B} \sum_{b=1}^{B} T_b\left(\mathbf{x}_i;\; \mathcal{D}^{(b)}\right)$",
            ha="center", va="center",
            fontsize=32, color=C_L1, alpha=show_new, family="serif")
    ax.text(0.5, 0.22,
            r"Each tree $T_b$ trains on a bootstrap sample "
            r"$\mathcal{D}^{(b)}$; averaging cancels noise",
            ha="center", va="center",
            fontsize=11, color=C_SUB, alpha=show_new, family="serif",
            style="italic")
    return fig


def scene_step3(show_new=1.0):
    """Feature subsampling at each split."""
    fig, ax = make_fig()
    _step_title(ax, "Step 3 \u2014 Feature subsampling at each split")
    # Forest equation carried forward
    ax.text(0.5, 0.72,
            r"$\hat{y}_i = \frac{1}{B} \sum_{b=1}^{B} T_b\left(\mathbf{x}_i;\; \mathcal{D}^{(b)}\right)$",
            ha="center", va="center",
            fontsize=22, color=C_L1, alpha=0.55, family="serif")
    # New
    ax.text(0.5, 0.48,
            r"$\mathcal{S} \subset \{1,\ldots,p\},\quad |\mathcal{S}| = m_{\mathrm{try}}$",
            ha="center", va="center",
            fontsize=28, color=C_L2, alpha=show_new, family="serif")
    ax.text(0.5, 0.30,
            r"At each node, split using only $m_{\mathrm{try}}$ random "
            r"features $\Rightarrow$ decorrelated trees",
            ha="center", va="center",
            fontsize=11, color=C_SUB, alpha=show_new, family="serif",
            style="italic")
    return fig


def scene_step4(show_new=1.0):
    """League-aware features (one-hot) = hierarchy analog."""
    fig, ax = make_fig()
    _step_title(ax, "Step 4 \u2014 League-aware features (hierarchy analog)")
    ax.text(0.5, 0.76,
            r"$\hat{y}_i = \frac{1}{B} \sum_{b=1}^{B} T_b\left(\mathbf{x}_i;\; \mathcal{D}^{(b)}\right)$",
            ha="center", va="center",
            fontsize=20, color=C_L1, alpha=0.5, family="serif")
    ax.text(0.5, 0.58,
            r"$\mathcal{S} \subset \{1,\ldots,p\},\ |\mathcal{S}| = m_{\mathrm{try}}$",
            ha="center", va="center",
            fontsize=18, color=C_L2, alpha=0.5, family="serif")
    # New augmented feature vector
    ax.text(0.5, 0.38,
            r"$\tilde{\mathbf{x}}_i \;=\; (\,\mathbf{x}_i\,,\; \mathrm{onehot}(g(i))\,)$",
            ha="center", va="center",
            fontsize=28, color=C_L3, alpha=show_new, family="serif")
    ax.text(0.5, 0.18,
            r"One-hot league indicators let trees split on league "
            r"$\Rightarrow$ per-league structure",
            ha="center", va="center",
            fontsize=11, color=C_SUB, alpha=show_new, family="serif",
            style="italic")
    return fig


def scene_step5(show_new=1.0):
    """OOB residuals and prediction intervals."""
    fig, ax = make_fig()
    _step_title(ax, "Step 5 \u2014 Out-of-bag residuals give prediction intervals")
    ax.text(0.5, 0.78,
            r"$\hat{y}_i = \frac{1}{B} \sum_{b} T_b(\tilde{\mathbf{x}}_i;\ \mathcal{D}^{(b)})$",
            ha="center", va="center",
            fontsize=18, color=C_L1, alpha=0.5, family="serif")
    ax.text(0.5, 0.58,
            r"$r_i^{\mathrm{OOB}} \;=\; y_i - \frac{1}{|\mathcal{B}_i^{\,-}|} "
            r"\sum_{b \notin \mathcal{B}_i} T_b(\tilde{\mathbf{x}}_i)$",
            ha="center", va="center",
            fontsize=22, color=C_L4, alpha=show_new, family="serif")
    ax.text(0.5, 0.38,
            r"$\hat{y}_i \pm z_{\alpha/2}\,\hat{\sigma}_{\mathrm{OOB}}$",
            ha="center", va="center",
            fontsize=24, color=C_L4, alpha=show_new, family="serif")
    ax.text(0.5, 0.18,
            r"Residuals from trees that didn't see row $i$ give a free, honest interval",
            ha="center", va="center",
            fontsize=11, color=C_SUB, alpha=show_new, family="serif",
            style="italic")
    return fig


def scene_summary(show_new=1.0):
    """Full Model Summary with boxed stacked equations."""
    fig, ax = make_fig()
    ax.text(0.5, 0.92, "Full Model Summary",
            ha="center", va="center",
            fontsize=22, color=C_BASE, alpha=show_new, family="serif")

    box_x, box_y = 0.14, 0.12
    box_w, box_h = 0.72, 0.66
    ax.add_patch(Rectangle((box_x, box_y), box_w, box_h,
                           edgecolor=C_BASE, facecolor="none",
                           linewidth=1.2, alpha=show_new))

    # Each equation drawn in its "introduced" colour
    eqs = [
        (r"$\hat{y}_i = T(\mathbf{x}_i)$", C_BASE),
        (r"$\hat{y}_i = \frac{1}{B} \sum_{b=1}^{B} T_b(\mathbf{x}_i;\, \mathcal{D}^{(b)})$", C_L1),
        (r"$\mathcal{S} \subset \{1,\ldots,p\},\ |\mathcal{S}| = m_{\mathrm{try}}$", C_L2),
        (r"$\tilde{\mathbf{x}}_i = (\mathbf{x}_i,\, \mathrm{onehot}(g(i)))$", C_L3),
        (r"$r_i^{\mathrm{OOB}} = y_i - \frac{1}{|\mathcal{B}_i^{-}|}"
         r"\sum_{b \notin \mathcal{B}_i} T_b(\tilde{\mathbf{x}}_i)$", C_L4),
        (r"$\hat{y}_i \pm z_{\alpha/2}\,\hat{\sigma}_{\mathrm{OOB}}$", C_L4),
    ]
    n = len(eqs)
    top = box_y + box_h - 0.06
    bot = box_y + 0.06
    ys = [top - i * (top - bot) / (n - 1) for i in range(n)]
    for (eq, col), y in zip(eqs, ys):
        ax.text(box_x + 0.04, y, eq, ha="left", va="center",
                fontsize=14, color=col, alpha=show_new, family="serif")
    return fig


# ---------------------------------------------------------------------------
# Timeline: (scene_fn, duration_s, fade_in_s)
# ---------------------------------------------------------------------------
TIMELINE = [
    (scene_title,   5.0, 0.6),
    (scene_step1,  10.0, 0.8),
    (scene_step2,  10.0, 0.9),
    (scene_step3,   9.0, 0.9),
    (scene_step4,   8.0, 0.9),
    (scene_step5,   7.0, 0.9),
    (scene_summary, 8.0, 1.2),
]


# ---------------------------------------------------------------------------
# Main — render frames, stitch with ffmpeg
# ---------------------------------------------------------------------------
def main():
    if shutil.which("ffmpeg") is None:
        sys.exit("ERROR: ffmpeg not found on PATH. Install it and retry.")

    # Fresh frame directory each run
    if FRAMES_DIR.exists():
        for f in FRAMES_DIR.glob("*.png"):
            f.unlink()
    else:
        FRAMES_DIR.mkdir(parents=True)

    total_s = sum(d for _, d, _ in TIMELINE)
    total_frames = int(total_s * FPS)
    print(f"Rendering {total_frames} frames ({total_s:.1f}s at {FPS}fps)...")

    frame_idx = 0
    for scene_fn, dur_s, fade_s in TIMELINE:
        n_frames = int(dur_s * FPS)
        n_fade = max(1, int(fade_s * FPS))
        for k in range(n_frames):
            alpha = (k / n_fade) if k < n_fade else 1.0
            fig = scene_fn(show_new=alpha)
            save_frame(fig, frame_idx)
            frame_idx += 1
    print(f"Wrote {frame_idx} frames to {FRAMES_DIR}")

    # Stitch into H.264 mp4 (854x480, baseline profile for max compatibility,
    # matches the Bayesian video's codec settings for in-PowerPoint playback)
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(FRAMES_DIR / "f%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=854:480",
        "-profile:v", "baseline", "-level", "3.0",
        "-movflags", "+faststart",
        str(OUT_MP4),
    ]
    print("Encoding video...")
    subprocess.run(cmd, check=True, capture_output=True)
    size_kb = OUT_MP4.stat().st_size / 1024
    print(f"Done -> {OUT_MP4}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
