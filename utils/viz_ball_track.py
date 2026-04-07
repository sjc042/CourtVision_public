import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# Camera output resolution — adjust to match CameraX ImageAnalysis target resolution
IMG_W = 1280
IMG_H = 720

# Segmentation parameters
SPATIAL_JUMP_THRESH = 0.15  # normalized units; splits within a tracking run (re-acquisition on different ball)
MIN_SEGMENT_FRAMES = 5      # discard segments shorter than this

CSV_PATH = r"C:\Users\Ciel Sun\Desktop\Dev\CourtVision_Android\benchmarks\phase0\phase0-day4-track-20260407-133601.csv"

df = pd.read_csv(CSV_PATH)

# --- Segmentation ---

# 1. Drop frames where tracker has no position
active = df[df['tracking_active'] == 1].copy().reset_index(drop=True)

# 2. Assign a run ID: increment whenever tracking_active had a gap (original index discontinuity)
#    i.e. a 0→1 transition in the original dataframe
orig_idx = df[df['tracking_active'] == 1].index.to_numpy()
run_breaks = np.concatenate(([0], np.where(np.diff(orig_idx) > 1)[0] + 1))
run_ids = np.zeros(len(active), dtype=int)
for i, start in enumerate(run_breaks):
    end = run_breaks[i + 1] if i + 1 < len(run_breaks) else len(active)
    run_ids[start:end] = i
active['run_id'] = run_ids

# 3. Within each run, split on large spatial jumps (tracker re-acquired a different detection)
active['dcx'] = active.groupby('run_id')['track_cx'].diff().abs().fillna(0)
active['dcy'] = active.groupby('run_id')['track_cy'].diff().abs().fillna(0)
jump = (active['dcx'] > SPATIAL_JUMP_THRESH) | (active['dcy'] > SPATIAL_JUMP_THRESH)

sub_id = 0
segment_ids = []
prev_run = None
for i, row in active.iterrows():
    if row['run_id'] != prev_run:
        sub_id += 1  # new run always starts a new segment
        prev_run = row['run_id']
    elif jump.iloc[i]:
        sub_id += 1  # spatial teleport within a run = new segment
    segment_ids.append(sub_id)

active['segment_id'] = segment_ids

# 4. Discard short segments
seg_counts = active['segment_id'].value_counts()
valid_segs = seg_counts[seg_counts >= MIN_SEGMENT_FRAMES].index
active = active[active['segment_id'].isin(valid_segs)]

segments = sorted(active['segment_id'].unique())
print(f"Total segments after filtering: {len(segments)}")
for sid in segments:
    seg = active[active['segment_id'] == sid]
    dur = (seg['timestamp_ns'].max() - seg['timestamp_ns'].min()) / 1e9
    print(f"  Segment {sid:3d}: {len(seg):3d} frames, {dur:.2f}s, "
          f"cx=[{seg['track_cx'].min():.2f},{seg['track_cx'].max():.2f}] "
          f"cy=[{seg['track_cy'].min():.2f},{seg['track_cy'].max():.2f}]")

# --- Visualization ---

colors = cm.tab10(np.linspace(0, 1, len(segments)))

# Plot 1: All segments overlaid on a single canvas
fig, ax = plt.subplots(figsize=(IMG_W / 96, IMG_H / 96))
for color, sid in zip(colors, segments):
    seg = active[active['segment_id'] == sid]
    cx = seg['track_cx'] * IMG_W
    cy = seg['track_cy'] * IMG_H
    ax.plot(cx, cy, '-o', color=color, markersize=3, linewidth=1.2, label=f"seg {sid}")
    # Mark start with a larger dot
    ax.scatter(cx.iloc[0], cy.iloc[0], color=color, s=60, zorder=5)

ax.set_xlim(0, IMG_W)
ax.set_ylim(IMG_H, 0)
ax.set_title("Ball tracker — all segments")
ax.set_xlabel("cx (pixels)")
ax.set_ylabel("cy (pixels)")
ax.legend(fontsize=7, ncol=3, loc='lower right')
plt.tight_layout()
plt.savefig("trajectory_segments.png", dpi=96)
plt.show()

# Plot 2: Each segment in its own subplot
n = len(segments)
ncols = 4
nrows = (n + ncols - 1) // ncols
fig2, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
axes = axes.flatten()

for i, (sid, color) in enumerate(zip(segments, colors)):
    seg = active[active['segment_id'] == sid]
    cx = seg['track_cx'] * IMG_W
    cy = seg['track_cy'] * IMG_H
    ax = axes[i]
    ax.plot(cx, cy, '-o', color=color, markersize=3, linewidth=1.2)
    ax.scatter(cx.iloc[0], cy.iloc[0], color='green', s=40, zorder=5, label='start')
    ax.scatter(cx.iloc[-1], cy.iloc[-1], color='red', s=40, zorder=5, label='end')
    ax.set_xlim(0, IMG_W)
    ax.set_ylim(IMG_H, 0)
    ax.set_title(f"Seg {sid} ({len(seg)}fr)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Individual shot arcs", fontsize=12)
plt.tight_layout()
plt.savefig("trajectory_per_segment.png", dpi=96)
plt.show()

# Plot 3: miss_streak over time with segment boundaries shaded
fig3, ax3 = plt.subplots(figsize=(14, 4))
ax3.plot(df.index, df['miss_streak'], color='steelblue', linewidth=0.8)
for sid in segments:
    seg_orig = df.index[df['tracking_active'] == 1][active[active['segment_id'] == sid].index]
    if len(seg_orig):
        ax3.axvspan(seg_orig[0], seg_orig[-1], alpha=0.15, color='orange')
ax3.set_title("miss_streak over time (orange = valid segments)")
ax3.set_xlabel("frame index"); ax3.set_ylabel("miss_streak")
plt.tight_layout()
plt.savefig("miss_streak.png", dpi=96)
plt.show()
