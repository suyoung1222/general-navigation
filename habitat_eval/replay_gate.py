"""Step 4 gate: open-loop replay of the expert trajectory against its own topomap.

No control, no habitat. Feeds the expert's own frames to NoMaD as observations and asks
the distance head where it thinks it is in the topomap. If the predicted closest node
does not track the expert's actual progress here, then preprocessing, goal conditioning
or the checkpoint is wrong, and every closed-loop result would be uninterpretable.

Reports two localizations per step:
  * global  -- argmin over ALL topomap nodes (strict; tests global localization)
  * windowed -- upstream's sliding window (radius=4) seeded from the previous estimate,
                i.e. exactly what navigate.py would compute

Usage:
  python replay_gate.py --run-dir feasibility/frl0_run4
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image as PILImage

import nomad_infer as NI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="feasibility/frl0_run4")
    ap.add_argument("--radius", type=int, default=NI.RADIUS)
    ap.add_argument("--out", default=None, help="defaults to <run-dir>/replay_gate")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.run_dir, "replay_gate")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, robot, action_stats = NI.load_configs()
    print(f"[gate] device={device}  context_size={cfg['context_size']}  "
          f"image_size={cfg['image_size']}")
    model = NI.load_nomad(cfg, device)

    topomap, nodes = NI.load_topomap(args.run_dir)
    n_nodes = len(topomap)
    with open(os.path.join(args.run_dir, "traj.json")) as f:
        traj = json.load(f)
    steps = traj["steps"]
    frames_dir = os.path.join(args.run_dir, "frames")
    print(f"[gate] {len(steps)} expert frames, {n_nodes} topomap nodes")

    ctx = cfg["context_size"]
    image_size = cfg["image_size"]

    # Encode every topomap node once per observation stack (the goal encoder consumes
    # [current_obs ; goal] jointly, so goal embeddings cannot be cached across steps).
    goal_batch = torch.cat(
        [NI.transform_images(g, image_size, center_crop=False) for g in topomap],
        dim=0).to(device)                                    # (n_nodes, 3, 96, 96)

    records = []
    dist_matrix = np.full((len(steps), n_nodes), np.nan, dtype=np.float32)
    prev_closest = 0

    with torch.no_grad():
        for t in range(len(steps)):
            if t < ctx:
                continue                                     # queue not full yet
            window = [PILImage.open(os.path.join(frames_dir, f"{i:05d}.jpg")).convert("RGB")
                      for i in range(t - ctx, t + 1)]         # ctx+1 = 4 frames
            obs = NI.transform_images(window, image_size, center_crop=False).to(device)
            obs = obs.repeat(n_nodes, 1, 1, 1)                # (n_nodes, 12, 96, 96)
            mask = torch.zeros(n_nodes).long().to(device)     # 0 = USE the goal

            cond = model("vision_encoder", obs_img=obs, goal_img=goal_batch,
                         input_goal_mask=mask)                # (n_nodes, 256)
            dists = NI.to_numpy(model("dist_pred_net", obsgoal_cond=cond).flatten())
            dist_matrix[t] = dists

            global_closest = int(np.argmin(dists))

            # upstream's windowed localization, seeded from the previous estimate
            start = max(prev_closest - args.radius, 0)
            end = min(prev_closest + args.radius + 1, n_nodes - 1)
            w_dists = dists[start:end + 1]
            win_closest = int(np.argmin(w_dists)) + start
            prev_closest = win_closest

            # where the expert actually was: nearest node by ground-truth position
            p = np.array(steps[t]["position"])
            gt_node = int(np.argmin([np.linalg.norm(p - np.array(n["position"]))
                                     for n in nodes]))

            records.append(dict(t=t, gt_node=gt_node, global_closest=global_closest,
                                windowed_closest=win_closest,
                                dist_at_global=float(dists[global_closest]),
                                dist_at_gt=float(dists[gt_node])))

    # ---- verdict -----------------------------------------------------------
    ts = np.array([r["t"] for r in records])
    gt = np.array([r["gt_node"] for r in records])
    gl = np.array([r["global_closest"] for r in records])
    wn = np.array([r["windowed_closest"] for r in records])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    def regressions(seq):
        return int(np.sum(np.diff(seq) < 0))

    summary = dict(
        n_steps=len(records), n_nodes=n_nodes,
        spearman_global_vs_t=spearman(ts, gl),
        spearman_windowed_vs_t=spearman(ts, wn),
        spearman_gt_vs_t=spearman(ts, gt),
        regressions_global=regressions(gl),
        regressions_windowed=regressions(wn),
        mean_abs_node_error_global=float(np.mean(np.abs(gl - gt))),
        mean_abs_node_error_windowed=float(np.mean(np.abs(wn - gt))),
        final_global=int(gl[-1]), final_windowed=int(wn[-1]), final_gt=int(gt[-1]),
        reaches_last_node_global=bool(gl[-1] == n_nodes - 1),
        reaches_last_node_windowed=bool(wn[-1] == n_nodes - 1),
    )

    print("\n  t | gt | global | windowed |  d(global) |  d(gt)")
    for r in records:
        print(f" {r['t']:2d} | {r['gt_node']:2d} | {r['global_closest']:6d} | "
              f"{r['windowed_closest']:8d} | {r['dist_at_global']:10.3f} | "
              f"{r['dist_at_gt']:6.3f}")

    print("\n[gate] summary")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    with open(os.path.join(out_dir, "replay_gate.json"), "w") as f:
        json.dump(dict(summary=summary, records=records), f, indent=2)

    try:
        plot(dist_matrix, ts, gt, gl, wn, n_nodes,
             os.path.join(out_dir, "replay_gate.png"))
        print(f"[gate] wrote {out_dir}/replay_gate.png")
    except Exception as exc:
        print(f"[gate] plot skipped: {exc}")

    passed = (summary["spearman_windowed_vs_t"] > 0.9
              and summary["reaches_last_node_windowed"]
              and summary["mean_abs_node_error_windowed"] <= 2.0)
    print(f"\n[gate] {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def plot(dist_matrix, ts, gt, gl, wn, n_nodes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    m = np.ma.masked_invalid(dist_matrix)
    im = ax0.imshow(m.T, aspect="auto", origin="lower", cmap="viridis")
    ax0.plot(ts, gt, "w--", lw=1.5, label="expert GT node")
    ax0.plot(ts, wn, "r-", lw=1.5, label="windowed argmin")
    ax0.set_xlabel("expert frame t"); ax0.set_ylabel("topomap node")
    ax0.set_title("predicted temporal distance"); ax0.legend(loc="upper left")
    fig.colorbar(im, ax=ax0)

    ax1.plot(ts, gt, "k--", lw=2, label="expert GT node")
    ax1.plot(ts, gl, "o-", ms=3, label="global argmin")
    ax1.plot(ts, wn, "s-", ms=3, label="windowed argmin")
    ax1.set_xlabel("expert frame t"); ax1.set_ylabel("localized node")
    ax1.set_ylim(-0.5, n_nodes - 0.5); ax1.grid(alpha=0.3)
    ax1.set_title("localization vs expert progress"); ax1.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
