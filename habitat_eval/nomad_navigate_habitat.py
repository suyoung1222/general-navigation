"""Closed-loop NoMaD navigation in Habitat against a recorded topomap.

Habitat replacement for deployment/carla/carla_nomad_nav.py, which is itself a
ROS-free port of deployment/src/navigate.py. The inference path (encode -> distance
head -> DDPM sampling -> get_action -> pd_controller) is kept identical to upstream;
only the sensor source and the actuator change.

Two goal modes share one diffusion path:

  --goal-mode topomap  (default)  upstream-faithful sliding-window localization via
                                  dist_pred_net, with close_threshold subgoal
                                  promotion. This is the "NoMaD works in Habitat"
                                  baseline.
  --goal-mode fixed               one fixed goal image for the whole episode; no
                                  topomap window, no localization, no subgoal
                                  promotion. Success is judged purely against a
                                  privileged --goal-position. This is the mode for
                                  NWM-imagined-goal experiments, where model-based
                                  localization would be an uncontrolled confounder.

Usage:
  python nomad_navigate_habitat.py --run-dir feasibility/frl0_run4
  python nomad_navigate_habitat.py --run-dir feasibility/frl0_run4 \
      --goal-mode fixed --fixed-goal feasibility/frl0_run4/nodes/node_015.jpg \
      --goal-position 3.0325,0.1194,0.4069
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image as PILImage, ImageDraw
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import habitat_sim
from habitat_sim.utils.common import (quat_from_angle_axis, quat_from_magnum,
                                      quat_to_magnum)

import nomad_infer as NI
from collect_expert_topomap import make_sim, sim_args, yaw_of


# ---------------------------------------------------------------------------
# Actuation
# ---------------------------------------------------------------------------

def make_velocity_control():
    vc = habitat_sim.physics.VelocityControl()
    vc.controlling_lin_vel = True
    vc.controlling_ang_vel = True
    vc.lin_vel_is_local = True
    vc.ang_vel_is_local = True
    return vc


def apply_velocity(sim, agent, vc, v, w, dt, substeps=5):
    """Integrate a unicycle command, sliding along the navmesh on contact.

    Habitat's agent forward is -Z and yaw is CCW-positive about +Y, so a positive
    `w` from pd_controller (waypoint to the left) turns left. Verified empirically
    by --selftest; carla_nomad_nav.py carries the same warning because a sign flip
    here looks like a policy failure rather than a bug.
    """
    vc.linear_velocity = np.array([0.0, 0.0, -float(v)], dtype=np.float32)
    vc.angular_velocity = np.array([0.0, float(w), 0.0], dtype=np.float32)
    collided = False
    for _ in range(substeps):
        st = agent.get_state()
        rs = habitat_sim.RigidState(quat_to_magnum(st.rotation), st.position)
        target = vc.integrate_transform(dt / substeps, rs)
        end_pos = sim.step_filter(rs.translation, target.translation)
        if np.linalg.norm(np.array(end_pos) - np.array(target.translation)) > 1e-3:
            collided = True
        st.position = np.array(end_pos, dtype=np.float32)
        st.rotation = quat_from_magnum(target.rotation)
        agent.set_state(st)
    return collided


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class NomadPolicy:
    def __init__(self, model, cfg, action_stats, device, args):
        self.model, self.cfg, self.stats = model, cfg, action_stats
        self.device, self.args = device, args
        self.scheduler = DDPMScheduler(
            num_train_timesteps=cfg["num_diffusion_iters"],
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.ctx = cfg["context_size"]
        self.image_size = cfg["image_size"]
        self.context_queue = []

    def push(self, pil_img):
        if len(self.context_queue) >= self.ctx + 1:
            self.context_queue.pop(0)
        self.context_queue.append(pil_img)

    def ready(self):
        return len(self.context_queue) > self.ctx

    def _obs_tensor(self, n_repeat):
        obs = NI.transform_images(self.context_queue, self.image_size,
                                  center_crop=False).to(self.device)
        return obs.repeat(n_repeat, 1, 1, 1)          # (n, 3*(ctx+1), 96, 96)

    def _diffuse(self, obs_cond):
        """Shared with both goal modes -- identical to upstream navigate.py."""
        a = self.args
        cond = obs_cond.repeat(a.num_samples, 1)      # (num_samples, 256)
        naction = torch.randn((a.num_samples, self.cfg["len_traj_pred"], 2),
                              device=self.device)
        self.scheduler.set_timesteps(self.cfg["num_diffusion_iters"])
        for k in self.scheduler.timesteps:
            noise_pred = self.model("noise_pred_net", sample=naction,
                                    timestep=k, global_cond=cond)
            naction = self.scheduler.step(model_output=noise_pred, timestep=k,
                                          sample=naction).prev_sample
        actions = NI.get_action(naction, self.stats)  # (num_samples, 8, 2)
        wp = actions[0][a.waypoint_idx].copy()        # upstream uses sample 0
        if self.cfg["normalize"]:
            wp[:2] *= (a.max_v / a.rate)              # -> meters per control tick
        return wp, actions

    def step_topomap(self, topomap, closest_node, goal_node):
        a = self.args
        start = max(closest_node - a.radius, 0)
        end = min(closest_node + a.radius + 1, goal_node)
        cand = topomap[start:end + 1]
        goal_batch = torch.cat(
            [NI.transform_images(g, self.image_size, center_crop=False)
             for g in cand], dim=0).to(self.device)
        n = len(cand)
        mask = torch.zeros(n).long().to(self.device)   # 0 = USE the goal
        cond = self.model("vision_encoder", obs_img=self._obs_tensor(n),
                          goal_img=goal_batch, input_goal_mask=mask)
        dists = NI.to_numpy(self.model("dist_pred_net", obsgoal_cond=cond).flatten())
        min_idx = int(np.argmin(dists))
        new_closest = min_idx + start
        sg_idx = min(min_idx + int(dists[min_idx] < a.close_threshold), n - 1)
        wp, actions = self._diffuse(cond[sg_idx].unsqueeze(0))
        return wp, actions, dict(closest_node=new_closest, sg_idx=int(sg_idx),
                                 dist_pred=float(dists[min_idx]),
                                 dists=[float(x) for x in dists],
                                 window_start=int(start))

    def step_fixed(self, goal_img):
        """Single fixed goal. No window, no localization, no subgoal promotion."""
        goal = NI.transform_images(goal_img, self.image_size,
                                   center_crop=False).to(self.device)
        mask = torch.zeros(1).long().to(self.device)
        cond = self.model("vision_encoder", obs_img=self._obs_tensor(1),
                          goal_img=goal, input_goal_mask=mask)
        # Logged as a diagnostic only -- it must never influence control here.
        dist = float(NI.to_numpy(
            self.model("dist_pred_net", obsgoal_cond=cond).flatten())[0])
        wp, actions = self._diffuse(cond)
        return wp, actions, dict(dist_pred=dist)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def draw_hud(rgb, wp, v, w, goal_thumb, line1, line2):
    img = PILImage.fromarray(rgb).convert("RGB")
    d = ImageDraw.Draw(img)
    W, H = img.size
    cx, cy = W // 2, int(H * 0.75)
    ppm = 900.0                       # waypoint is ~0.05 m; scale so it is visible
    ex = int(np.clip(cx - wp[1] * ppm, 0, W - 1))
    ey = int(np.clip(cy - wp[0] * ppm, 0, H - 1))
    d.line([(cx, cy), (ex, ey)], fill=(0, 255, 0), width=4)
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(0, 255, 0))

    d.rectangle([0, 0, W, 44], fill=(0, 0, 0))
    d.text((6, 6), line1, fill=(0, 255, 255))
    d.text((6, 24), line2, fill=(200, 200, 255))

    if goal_thumb is not None:
        tw, th = 160, 120
        img.paste(goal_thumb.resize((tw, th)), (W - tw - 6, 50))
        d.rectangle([W - tw - 7, 49, W - 5, 50 + th], outline=(255, 170, 0), width=2)
        d.text((W - tw - 2, 52 + th), "goal", fill=(255, 170, 0))
    return img


def save_comparison_topdown(pf, out_path, expert_xyz, node_xyz, policy_xyz,
                            start, goal_gt, mpp=0.02):
    """Navmesh slice with the expert demo and the policy rollout overlaid."""
    height = float(start[1])
    grid = pf.get_topdown_view(mpp, height)
    bounds = pf.get_bounds()
    x0, z0 = bounds[0][0], bounds[0][2]

    img = np.zeros(grid.shape + (3,), dtype=np.uint8)
    img[grid] = (235, 235, 235)
    img[~grid] = (40, 40, 45)

    def dot(p, color, r):
        row = int((p[2] - z0) / mpp)
        col = int((p[0] - x0) / mpp)
        rr = slice(max(row - r, 0), min(row + r + 1, img.shape[0]))
        cc = slice(max(col - r, 0), min(col + r + 1, img.shape[1]))
        img[rr, cc] = color

    for p in expert_xyz:
        dot(p, (70, 130, 255), 2)        # blue  = expert demo
    for p in node_xyz:
        dot(p, (255, 170, 0), 4)         # orange = topomap nodes
    for p in policy_xyz:
        dot(p, (230, 60, 200), 2)        # magenta = NoMaD policy
    dot(start, (0, 200, 0), 6)
    dot(goal_gt, (230, 30, 30), 6)

    out = PILImage.fromarray(img).resize(
        (img.shape[1] * 2, img.shape[0] * 2), PILImage.NEAREST)
    d = ImageDraw.Draw(out)
    legend = [("expert demo", (70, 130, 255)), ("topomap nodes", (255, 170, 0)),
              ("NoMaD policy", (230, 60, 200)), ("start", (0, 200, 0)),
              ("goal", (230, 30, 30))]
    d.rectangle([4, 4, 150, 8 + 14 * len(legend)], fill=(0, 0, 0))
    for i, (label, color) in enumerate(legend):
        d.rectangle([8, 8 + 14 * i, 18, 16 + 14 * i], fill=color)
        d.text((24, 7 + 14 * i), label, fill=(255, 255, 255))
    out.save(out_path)
    print(f"[topdown] {out_path}")


# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="feasibility/frl0_run4",
                    help="expert run providing the topomap, navmesh and start pose")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to <run-dir>/policy_<goal-mode>")
    ap.add_argument("--goal-mode", choices=("topomap", "fixed"), default="topomap")
    ap.add_argument("--fixed-goal", default=None, help="goal image (fixed mode)")
    ap.add_argument("--goal-position", default=None,
                    help="X,Y,Z GT pose the fixed goal image was rendered from")

    ap.add_argument("--close-threshold", type=int, default=NI.CLOSE_THRESHOLD)
    ap.add_argument("--radius", type=int, default=NI.RADIUS)
    ap.add_argument("--waypoint-idx", type=int, default=NI.WAYPOINT_IDX)
    ap.add_argument("--num-samples", type=int, default=NI.NUM_SAMPLES)
    ap.add_argument("--max-v", type=float, default=None, help="default: robot.yaml")
    ap.add_argument("--max-w", type=float, default=None, help="default: robot.yaml")
    ap.add_argument("--rate", type=float, default=None, help="default: robot.yaml")
    ap.add_argument("--substeps", type=int, default=5)

    ap.add_argument("--success-radius", type=float, default=0.3)
    ap.add_argument("--no-stop-on-goal-node", dest="stop_on_goal_node",
                    action="store_false",
                    help="keep driving past the final topomap node (upstream stops)")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--stuck-window", type=int, default=40)
    ap.add_argument("--stuck-eps", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--selftest", action="store_true",
                    help="verify the angular-velocity sign, then exit")
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    out_dir = os.path.abspath(
        args.out_dir or os.path.join(run_dir, f"policy_{args.goal_mode}"))
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for fn in os.listdir(frames_dir):
        if fn.endswith(".jpg"):
            os.remove(os.path.join(frames_dir, fn))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, robot, action_stats = NI.load_configs()
    args.max_v = args.max_v if args.max_v is not None else robot["max_v"]
    args.max_w = args.max_w if args.max_w is not None else robot["max_w"]
    args.rate = args.rate if args.rate is not None else robot["frame_rate"]
    dt = 1.0 / args.rate

    with open(os.path.join(run_dir, "traj.json")) as f:
        traj = json.load(f)
    topomap, nodes = NI.load_topomap(run_dir)

    if args.goal_mode == "fixed":
        if not (args.fixed_goal and args.goal_position):
            raise SystemExit("--goal-mode fixed requires --fixed-goal and --goal-position")
        goal_image_path = os.path.abspath(args.fixed_goal)
        goal_img = PILImage.open(goal_image_path).convert("RGB")
        goal_gt = np.array([float(x) for x in args.goal_position.split(",")],
                           dtype=np.float32)
    else:
        goal_node = len(topomap) - 1
        goal_image_path = os.path.join(run_dir, "nodes", f"node_{goal_node:03d}.jpg")
        goal_img = topomap[goal_node]
        # The pose the goal *image* was rendered from -- not traj.json["goal"],
        # which sits ~0.15 m further along.
        goal_gt = np.array(nodes[goal_node]["position"], dtype=np.float32)

    print(f"[setup] mode={args.goal_mode}  goal_image={os.path.basename(goal_image_path)}"
          f"  goal_gt={np.round(goal_gt, 3)}  success_radius={args.success_radius}")
    print(f"[setup] max_v={args.max_v} max_w={args.max_w} rate={args.rate} dt={dt}")

    model = NI.load_nomad(cfg, device)
    policy = NomadPolicy(model, cfg, action_stats, device, args)

    sim = make_sim(sim_args(
        navmesh=os.path.join(run_dir, "apt_0_furnished.navmesh"),
        navmesh_mode="shipped", out_dir=out_dir,
        scene=traj["scene"], scene_dataset=traj["scene_dataset"], seed=args.seed,
        width=traj["camera"]["width"], height=traj["camera"]["height"],
        hfov=traj["camera"]["hfov_deg"], camera_height=traj["camera"]["height_m"],
    ))
    try:
        pf = sim.pathfinder
        agent = sim.get_agent(0)
        start = np.array(pf.snap_point(np.array(traj["start"], dtype=np.float32)),
                         dtype=np.float32)
        start_yaw = float(traj["steps"][0]["yaw"])
        st = agent.get_state()
        st.position = start
        st.rotation = quat_from_angle_axis(start_yaw, np.array([0.0, 1.0, 0.0]))
        agent.set_state(st)
        vc = make_velocity_control()

        if args.selftest:
            y0 = yaw_of(agent.get_state())
            apply_velocity(sim, agent, vc, 0.0, 0.4, dt, args.substeps)
            y1 = yaw_of(agent.get_state())
            d = float(np.arctan2(np.sin(y1 - y0), np.cos(y1 - y0)))
            print(f"[selftest] commanded w=+0.4 rad/s for {dt}s -> yaw {y0:.4f} "
                  f"-> {y1:.4f} (delta {d:+.4f} rad)")
            print(f"[selftest] expected +{0.4 * dt:.4f}; sign is "
                  f"{'CORRECT (positive w turns left/CCW)' if d > 0 else 'INVERTED'}")
            return 0

        geodesic = float(traj["geodesic_distance"])
        steps, prev_pos = [], start.copy()
        path_len, collisions = 0.0, 0
        closest_node, reason = 0, "max_steps"
        wp = np.zeros(2)
        v = w = 0.0
        extra = {}

        for t in range(args.max_steps):
            obs = sim.get_sensor_observations()
            rgb = obs["rgb"][..., :3]
            policy.push(PILImage.fromarray(rgb).convert("RGB"))

            s = agent.get_state()
            pos = np.array(s.position, dtype=np.float32)
            yaw = yaw_of(s)
            path_len += float(np.linalg.norm(pos - prev_pos))
            prev_pos = pos
            dist_gt = float(np.linalg.norm(pos - goal_gt))

            if policy.ready():
                with torch.no_grad():
                    if args.goal_mode == "topomap":
                        wp, _, extra = policy.step_topomap(topomap, closest_node,
                                                           len(topomap) - 1)
                        closest_node = extra["closest_node"]
                    else:
                        wp, _, extra = policy.step_fixed(goal_img)
                v, w = NI.pd_controller(wp, dt, args.max_v, args.max_w)
            else:
                v = w = 0.0

            sub = (f"node {closest_node}/{len(topomap)-1}"
                   if args.goal_mode == "topomap" else "fixed goal")
            hud = draw_hud(
                rgb, wp, v, w, goal_img,
                f"t={t}  {sub}  d_pred={extra.get('dist_pred', float('nan')):.2f}",
                f"v={v:.3f} w={w:+.3f}  dx={wp[0]:+.4f} dy={wp[1]:+.4f}  "
                f"d_goal={dist_gt:.2f}m")
            hud.save(os.path.join(frames_dir, f"{t:05d}.jpg"), quality=90)

            rec = dict(t=t, position=pos.tolist(), yaw=yaw, v=v, w=w,
                       chosen_waypoint=[float(x) for x in wp],
                       dist_to_goal_gt=dist_gt,
                       dist_pred=extra.get("dist_pred"))
            if args.goal_mode == "topomap":
                rec.update(closest_node=closest_node, sg_idx=extra.get("sg_idx"),
                           dists=extra.get("dists"))

            if dist_gt < args.success_radius:
                rec["collided"] = False
                steps.append(rec)
                reason = "reached_goal_position"
                break
            # Upstream's stop signal. navigate.py publishes reached_goal =
            # (closest_node == goal_node) and pd_controller.py is what actually halts
            # on it -- so the policy itself never decelerates. pd_controller clips v
            # to [0, max_v] and NoMaD's waypoint always has positive dx, so without
            # this the agent drives at max_v forever, straight past the goal.
            if (args.stop_on_goal_node and args.goal_mode == "topomap"
                    and policy.ready() and closest_node == len(topomap) - 1):
                rec["collided"] = False
                steps.append(rec)
                reason = "reached_goal_node"
                break

            collided = apply_velocity(sim, agent, vc, v, w, dt, args.substeps)
            collisions += int(collided)
            rec["collided"] = bool(collided)
            steps.append(rec)

            if len(steps) > args.stuck_window:
                a0 = np.array(steps[-args.stuck_window]["position"])
                if np.linalg.norm(pos - a0) < args.stuck_eps and t > args.stuck_window:
                    reason = "stuck"
                    break

        final_pos = np.array(agent.get_state().position, dtype=np.float32)
        final_dist = float(np.linalg.norm(final_pos - goal_gt))
        dists_gt = [s["dist_to_goal_gt"] for s in steps]
        success = bool(final_dist < args.success_radius)

        result = dict(
            goal_mode=args.goal_mode,
            goal_image_path=goal_image_path,
            goal_gt_position=goal_gt.tolist(),
            success_radius=args.success_radius,
            success=success,
            close_threshold=args.close_threshold, radius=args.radius,
            waypoint_idx=args.waypoint_idx, num_samples=args.num_samples,
            max_v=args.max_v, max_w=args.max_w, rate=args.rate,
            stop_on_goal_node=args.stop_on_goal_node,
            num_diffusion_iters=cfg["num_diffusion_iters"],
            context_size=cfg["context_size"], seed=args.seed,
            scene=traj["scene"], navmesh=os.path.join(run_dir, "apt_0_furnished.navmesh"),
            start_position=start.tolist(), start_yaw=start_yaw,
            termination_reason=reason,
            final_position=final_pos.tolist(),
            final_dist_to_goal_gt=final_dist,
            min_dist_to_goal_gt=float(min(dists_gt)) if dists_gt else None,
            path_length=path_len, num_steps=len(steps),
            collision_count=collisions,
            expert_path_length=float(traj["path_length"]),
            geodesic_distance=geodesic,
            spl=float(success) * geodesic / max(path_len, geodesic) if path_len else 0.0,
            steps=steps,
        )
        with open(os.path.join(out_dir, "rollout.json"), "w") as f:
            json.dump(result, f, indent=2)

        save_comparison_topdown(
            pf, os.path.join(out_dir, "topdown_compare.png"),
            [s["position"] for s in traj["steps"]],
            [n["position"] for n in nodes],
            [s["position"] for s in steps],
            start, goal_gt)
        write_video(frames_dir, os.path.join(out_dir, "overlay.mp4"), args.rate)

        print(f"\n[result] {reason}  success={success}")
        print(f"[result] final dist to goal {final_dist:.3f} m "
              f"(min over run {min(dists_gt):.3f} m, radius {args.success_radius})")
        print(f"[result] steps {len(steps)}  path {path_len:.2f} m "
              f"(expert {traj['path_length']:.2f} m)  collisions {collisions}")
        print(f"[result] SPL {result['spl']:.3f}  -> {out_dir}")
        return 0 if success else 1
    finally:
        sim.close()


def write_video(frames_dir, out_path, fps):
    try:
        import imageio.v2 as imageio
        files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
        with imageio.get_writer(out_path, fps=max(fps, 4)) as wtr:
            for fn in files:
                wtr.append_data(imageio.imread(os.path.join(frames_dir, fn)))
        print(f"[video] {out_path} ({len(files)} frames)")
    except Exception as exc:
        print(f"[video] skipped: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
