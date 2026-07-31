"""Expert demo -> topomap collector for feasibility test (ReplicaCAD frl_apartment).

Produces everything the A/B/C conditions need:
  out_dir/
    frames/%05d.jpg        # every step RGB (context source for NWM)
    nodes/node_%03d.jpg    # subsampled node images (goal images, cond A)
    traj.json              # per-step: position, rotation(yaw), action + run metadata
    nodes.json             # per-node: frame_idx, position, yaw, cum_dist
    topdown.png            # sanity-check map of the executed path + node markers

Usage:
  python collect_expert_topomap.py --list                # sample candidate endpoints
  python collect_expert_topomap.py --pair 3              # collect using candidate pair #3
  python collect_expert_topomap.py --start X,Y,Z --goal X,Y,Z

Notes on ReplicaCAD (verified against habitat-sim 0.3.3 / ReplicaCAD v1.6):
  * The scene id is ``apt_0`` .. ``apt_5``. ``frl_apartment_0`` is the *stage*
    name, not a loadable scene, and ``frl_apartment_stage`` is the bare shell
    with no furniture.
  * ``apt_*.scene_instance.json`` declares ``navmesh_instance:
    empty_stage_navmesh``, which does not resolve in v1.6. The simulator comes
    up with an *unloaded* pathfinder, and touching any PathFinder method then
    segfaults the process. We therefore always load
    ``navmeshes/<scene>.navmesh`` explicitly before anything else.
  * That shipped navmesh does **not** account for the furniture -- 53% of the
    dining table's footprint and 76% of the sofa's are marked navigable, and
    the expert drives straight through them. See ``rebake_navmesh()``; the
    default ``--navmesh-mode rebake`` fixes it.
  * ``enable_physics`` must be True or the articulated assets (fridge,
    kitchen counter, cupboards, door) silently fail to instance and are
    missing from every rendered frame.
  * ``navmeshes/apt_0.navmesh`` also contains stray polygons sitting on top of
    the apartment roof (y up to ~2.9 m), and Recast reports them as part of
    the same island as the floor. Roughly 11% of uniformly sampled "navigable"
    points land up there, looking out at the skybox. Everything is therefore
    restricted to a floor band around the modal navmesh height.
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector

# ---------------- CONFIG: defaults, all overridable on the CLI ----------------
DATA_ROOT = "/home/suyoung/Documents/habitat-sim/data/replica_cad"
SCENE_DATASET = f"{DATA_ROOT}/replicaCAD.scene_dataset_config.json"
SCENE = "apt_0"                  # frl_apartment layout 0
OUT_DIR = "feasibility/frl0_run1"
IMG_W, IMG_H = 640, 480          # 4:3 to match NoMaD training aspect
HFOV_DEG = 90.0
CAMERA_HEIGHT = 0.88             # match your data collection rig
FORWARD_M = 0.25
TURN_DEG = 15.0
GOAL_RADIUS = 0.3
NODE_SPACING_M = 1.0             # node every ~1m (keep <= NWM reliable horizon)
MAX_STEPS = 2000
SEED = 7
# ------------------------------------------------------------------------------


def make_sim(args):
    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_dataset_config_file = args.scene_dataset
    backend.scene_id = args.scene
    backend.random_seed = args.seed
    # Required: without physics the articulated objects (fridge, counter,
    # cupboards, door) fail to instance and vanish from the renders.
    backend.enable_physics = True

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb.resolution = [args.height, args.width]
    rgb.position = [0.0, args.camera_height, 0.0]
    rgb.hfov = args.hfov

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=args.forward)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=args.turn)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=args.turn)),
    }

    sim = habitat_sim.Simulator(habitat_sim.Configuration(backend, [agent_cfg]))
    sim.seed(args.seed)
    load_navmesh(sim, args)
    return sim


def load_navmesh(sim, args):
    """Get a usable navmesh onto the pathfinder.

    Every PathFinder call on an unloaded pathfinder segfaults, so the shipped
    file is loaded first no matter what, and only then optionally replaced by
    a rebake.
    """
    pf = sim.pathfinder
    navmesh = args.navmesh or os.path.join(
        os.path.dirname(args.scene_dataset), "navmeshes", f"{args.scene}.navmesh")
    if not os.path.isfile(navmesh):
        raise FileNotFoundError(f"navmesh not found: {navmesh}")
    if not pf.load_nav_mesh(navmesh):
        raise RuntimeError(f"failed to load navmesh: {navmesh}")
    args.navmesh = navmesh  # so the resolved path lands in traj.json
    report_navmesh(pf, f"shipped: {navmesh}")

    if args.navmesh_mode == "rebake":
        rebake_navmesh(sim, args)
    pf.seed(args.seed)
    return pf


def report_navmesh(pf, tag):
    areas = [round(pf.island_area(i), 2) for i in range(pf.num_islands)]
    print(f"[navmesh] {tag}")
    print(f"[navmesh] {pf.num_islands} island(s), total {pf.navigable_area:.2f} m^2, "
          f"areas {areas}")


def rebake_navmesh(sim, args):
    """Rebake the navmesh so the furniture actually blocks the expert.

    ReplicaCAD declares 102 of apt_0's 113 rigid instances as DYNAMIC -- every
    table, sofa, chair, bed and bike -- because the dataset is built for
    Rearrange tasks where those objects get picked up and moved. The shipped
    ``navmeshes/apt_0.navmesh`` therefore only bakes in the stage and the 11
    STATIC instances, which are rugs, cloths and wall decorations. Measured on
    the stock mesh, 53% of the dining table's footprint and 76% of the sofa's
    are reported navigable, and the greedy follower will happily route the
    expert straight through them.

    ``include_static_objects=True`` alone does not help (it only picks up those
    same 11 decorations, worth ~1.6% of the area). The objects have to be
    flipped to STATIC first.
    """
    n = {"rigid": 0, "articulated": 0}
    for kind, mgr in (("rigid", sim.get_rigid_object_manager()),
                      ("articulated", sim.get_articulated_object_manager())):
        for h in mgr.get_object_handles():
            obj = mgr.get_object_by_handle(h)
            try:
                if obj.motion_type != habitat_sim.physics.MotionType.STATIC:
                    obj.motion_type = habitat_sim.physics.MotionType.STATIC
                    n[kind] += 1
            except Exception as exc:  # keep going; a missed object only costs realism
                print(f"[navmesh] warning: could not set {h} STATIC ({exc})")
    print(f"[navmesh] forced {n['rigid']} rigid + {n['articulated']} articulated "
          f"objects to STATIC")

    ns = habitat_sim.nav.NavMeshSettings()
    ns.set_defaults()
    ns.agent_radius = args.agent_radius
    ns.agent_height = args.agent_height
    ns.agent_max_climb = args.agent_max_climb
    ns.cell_size = args.cell_size
    ns.include_static_objects = True
    if not sim.recompute_navmesh(sim.pathfinder, ns):
        raise RuntimeError("navmesh rebake failed")
    report_navmesh(sim.pathfinder, f"rebaked with furniture, agent_radius="
                                   f"{args.agent_radius}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.scene}_furnished.navmesh")
    sim.pathfinder.save_nav_mesh(out)
    args.navmesh = out
    print(f"[navmesh] saved -> {out}")


def largest_island(pf):
    if pf.num_islands <= 1:
        return -1
    return int(np.argmax([pf.island_area(i) for i in range(pf.num_islands)]))


def detect_floor_y(pf, island, n=2000):
    """Modal navmesh height, used to reject the roof-top polygons."""
    ys = np.array([pf.get_random_navigable_point(island_index=island)[1]
                   for _ in range(n)])
    return float(np.median(ys))


def on_floor(p, floor_y, tol):
    return abs(float(p[1]) - floor_y) <= tol


def path_on_floor(path, floor_y, tol):
    return all(on_floor(p, floor_y, tol) for p in path.points)


def yaw_of(agent_state):
    """Yaw in radians, 0 = -Z (habitat's default forward), CCW positive.

    Derived from the rotated forward vector rather than raw quaternion
    components so it stays correct even if the rotation is not pure-yaw.
    """
    fwd = quat_rotate_vector(agent_state.rotation, np.array([0.0, 0.0, -1.0]))
    return float(np.arctan2(-fwd[0], -fwd[2]))


def object_aabbs(sim):
    """World-space AABB per object, used by the clearance self-check."""
    import magnum as mn
    boxes = []
    for mgr in (sim.get_rigid_object_manager(), sim.get_articulated_object_manager()):
        for h in mgr.get_object_handles():
            obj = mgr.get_object_by_handle(h)
            try:
                bb, xf = obj.root_scene_node.cumulative_bb, obj.transformation
            except Exception:
                continue
            corners = [mn.Vector3(x, y, z)
                       for x in (bb.min.x, bb.max.x)
                       for y in (bb.min.y, bb.max.y)
                       for z in (bb.min.z, bb.max.z)]
            pts = np.array([list(xf.transform_point(c)) for c in corners])
            boxes.append((h, pts.min(0), pts.max(0)))
    return boxes


def clearance_check(sim, positions, radius, body_height=1.2):
    """Report objects the trajectory drove through.

    AABB-based, so it over-reports for thin/sparse geometry (a bike frame
    fills only a fraction of its box) -- treat a hit as "inspect this", not as
    proof. A clean run reporting nothing is the meaningful signal.
    """
    hits = {}
    for h, lo, hi in object_aabbs(sim):
        for i, p in enumerate(positions):
            in_xz = (lo[0] - radius <= p[0] <= hi[0] + radius
                     and lo[2] - radius <= p[2] <= hi[2] + radius)
            in_body_column = hi[1] > p[1] + 0.05 and lo[1] < p[1] + body_height
            if in_xz and in_body_column:
                hits.setdefault(h, []).append(i)
    return hits


def geodesic(pf, s, g):
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(s, dtype=np.float32)
    path.requested_end = np.array(g, dtype=np.float32)
    ok = pf.find_path(path)
    return ok, path


def list_candidates(sim, args):
    pf = sim.pathfinder
    island = largest_island(pf)
    floor_y = args.floor_y if args.floor_y is not None else detect_floor_y(pf, island)
    print(f"[list] sampling on island {island} "
          f"(-1 means the navmesh has a single island), "
          f"floor y={floor_y:.2f} +/- {args.floor_tol}")
    print("idx | start -> goal | geodesic (m) | tortuosity")
    pairs, tries = [], 0
    while len(pairs) < args.n_candidates and tries < 20000:
        tries += 1
        s = pf.get_random_navigable_point(island_index=island)
        g = pf.get_random_navigable_point(island_index=island)
        # Reject the roof-top polygons that Recast merged into the floor island.
        if not (on_floor(s, floor_y, args.floor_tol)
                and on_floor(g, floor_y, args.floor_tol)):
            continue
        ok, path = geodesic(pf, s, g)
        if not ok or not np.isfinite(path.geodesic_distance):
            continue
        if not (args.dmin <= path.geodesic_distance <= args.dmax):
            continue
        if not path_on_floor(path, floor_y, args.floor_tol):
            continue
        euclid = float(np.linalg.norm(np.array(s) - np.array(g)))
        # tortuosity > ~1.15 means the route actually turns a corner, which is
        # what makes the topomap interesting rather than a straight corridor.
        tort = path.geodesic_distance / max(euclid, 1e-6)
        if tort < args.min_tortuosity:
            continue
        print(f"{len(pairs):3d} | {np.round(s, 2)} -> {np.round(g, 2)} "
              f"| {path.geodesic_distance:6.2f} | {tort:.2f}")
        pairs.append(dict(start=list(map(float, s)), goal=list(map(float, g)),
                          geodesic=float(path.geodesic_distance),
                          tortuosity=float(tort)))

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "candidates.json")
    with open(out, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[list] {len(pairs)} candidates ({tries} draws) -> {out}")
    print(f"[list] rerun with --pair <idx>")


def save_topdown(pf, out_path, traj_xyz, node_xyz, start, goal, mpp=0.02):
    """Render the navmesh slice at the trajectory height with the path drawn on."""
    height = float(np.mean([p[1] for p in traj_xyz]))
    grid = pf.get_topdown_view(mpp, height)          # bool [z_bins, x_bins]
    bounds = pf.get_bounds()
    x0, z0 = bounds[0][0], bounds[0][2]

    img = np.zeros(grid.shape + (3,), dtype=np.uint8)
    img[grid] = (235, 235, 235)
    img[~grid] = (40, 40, 45)

    def to_px(p):
        return (int((p[2] - z0) / mpp), int((p[0] - x0) / mpp))  # (row, col)

    def dot(p, color, r):
        row, col = to_px(p)
        rr = slice(max(row - r, 0), min(row + r + 1, img.shape[0]))
        cc = slice(max(col - r, 0), min(col + r + 1, img.shape[1]))
        img[rr, cc] = color

    for p in traj_xyz:
        dot(p, (60, 120, 255), 1)      # blue path
    for p in node_xyz:
        dot(p, (255, 170, 0), 3)       # orange nodes
    dot(start, (0, 200, 0), 5)         # green start
    dot(goal, (230, 30, 30), 5)        # red goal

    Image.fromarray(img).resize(
        (img.shape[1] * 2, img.shape[0] * 2), Image.NEAREST).save(out_path)
    print(f"[topdown] {out_path}  (green=start, red=goal, orange=nodes)")


def collect(sim, args, start, goal):
    pf = sim.pathfinder
    frames_dir = os.path.join(args.out_dir, "frames")
    nodes_dir = os.path.join(args.out_dir, "nodes")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(nodes_dir, exist_ok=True)
    # A shorter re-run would otherwise leave stale images behind and the
    # topomap would silently contain frames from two different trajectories.
    for d in (frames_dir, nodes_dir):
        for fn in os.listdir(d):
            if fn.endswith(".jpg"):
                os.remove(os.path.join(d, fn))

    # Snap both endpoints onto the navmesh -- hand-typed coordinates and points
    # sampled under a different navmesh are usually a few cm off the surface,
    # and the follower silently fails on off-mesh goals.
    start = np.array(pf.snap_point(np.array(start, dtype=np.float32)), dtype=np.float32)
    goal = np.array(pf.snap_point(np.array(goal, dtype=np.float32)), dtype=np.float32)
    if not (pf.is_navigable(start) and pf.is_navigable(goal)):
        raise RuntimeError("start/goal not navigable after snapping")
    ok, path = geodesic(pf, start, goal)
    if not ok or not np.isfinite(path.geodesic_distance):
        raise RuntimeError("no navigable path between start and goal")

    floor_y = (args.floor_y if args.floor_y is not None
               else detect_floor_y(pf, largest_island(pf)))
    if not path_on_floor(path, floor_y, args.floor_tol):
        heights = sorted({round(float(p[1]), 2) for p in path.points})
        raise RuntimeError(
            f"path leaves the floor band (y={floor_y:.2f} +/- {args.floor_tol}); "
            f"waypoint heights {heights}. These are the roof-top navmesh "
            f"polygons -- pick a different start/goal.")
    print(f"[collect] start {np.round(start, 3)} -> goal {np.round(goal, 3)}, "
          f"geodesic {path.geodesic_distance:.2f} m, {len(path.points)} waypoints")

    agent = sim.get_agent(0)
    st = agent.get_state()
    st.position = start
    # Face the first path waypoint so the demo does not open with a dozen
    # in-place turn frames, which would waste topomap nodes.
    heading = None
    for wp in path.points[1:]:
        d = np.array(wp, dtype=np.float32) - start
        if np.linalg.norm([d[0], d[2]]) > 1e-3:
            heading = np.arctan2(-d[0], -d[2])
            break
    st.rotation = quat_from_angle_axis(
        float(heading) if heading is not None else 0.0, np.array([0.0, 1.0, 0.0]))
    agent.set_state(st)

    follower = habitat_sim.GreedyGeodesicFollower(
        pf, agent, goal_radius=args.goal_radius,
        forward_key="move_forward", left_key="turn_left", right_key="turn_right")
    follower.reset()

    traj, nodes = [], []
    cum_dist, last_node_dist = 0.0, -1e9
    prev_pos = start.copy()
    obs = sim.get_sensor_observations()
    t = 0
    status = "max_steps"

    while t < args.max_steps:
        rgb = Image.fromarray(obs["rgb"][..., :3])
        rgb.save(os.path.join(frames_dir, f"{t:05d}.jpg"), quality=95)

        s = agent.get_state()
        pos = np.array(s.position, dtype=np.float32)
        cum_dist += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos
        yaw = yaw_of(s)

        if cum_dist - last_node_dist >= args.node_spacing or t == 0:
            nid = len(nodes)
            rgb.save(os.path.join(nodes_dir, f"node_{nid:03d}.jpg"), quality=95)
            nodes.append(dict(node_id=nid, frame_idx=t, position=pos.tolist(),
                              yaw=yaw, cum_dist=cum_dist))
            last_node_dist = cum_dist

        try:
            action = follower.next_action_along(goal)
        except habitat_sim.errors.GreedyFollowerError:
            print(f"[collect] follower error at t={t} -- stopping.")
            traj.append(dict(t=t, position=pos.tolist(), yaw=yaw, action="stop"))
            status = "follower_error"
            break

        traj.append(dict(t=t, position=pos.tolist(), yaw=yaw,
                         action=action if action else "stop"))
        if action is None:
            print(f"[collect] goal reached in {t} steps, {cum_dist:.2f} m travelled.")
            status = "goal_reached"
            break
        obs = sim.step(action)
        t += 1
    else:
        print(f"[collect] hit --max-steps ({args.max_steps}) without reaching the goal.")

    # Always end the topomap on the final pose, but do not duplicate a node
    # that the spacing rule already emitted on this very frame.
    s = agent.get_state()
    if not nodes or nodes[-1]["frame_idx"] != t:
        nid = len(nodes)
        Image.fromarray(obs["rgb"][..., :3]).save(
            os.path.join(nodes_dir, f"node_{nid:03d}.jpg"), quality=95)
        nodes.append(dict(node_id=nid, frame_idx=t,
                          position=np.array(s.position, dtype=np.float32).tolist(),
                          yaw=yaw_of(s), cum_dist=cum_dist))

    final_pos = np.array(s.position, dtype=np.float32)
    dist_to_goal = float(np.linalg.norm(final_pos - goal))

    hits = clearance_check(sim, [d["position"] for d in traj], args.agent_radius)
    if hits:
        print("[clearance] trajectory intersects object bounding boxes:")
        for h, steps in sorted(hits.items(), key=lambda kv: -len(kv[1])):
            print(f"[clearance]   {h}  steps {steps[0]}..{steps[-1]} (n={len(steps)})")
        print("[clearance] AABBs over-report for sparse geometry -- eyeball the "
              "frames before trusting a hit.")
    else:
        print("[clearance] OK: no object bounding box intersects the path.")

    meta = dict(
        scene=args.scene, scene_dataset=args.scene_dataset,
        navmesh=args.navmesh, seed=args.seed, status=status,
        start=start.tolist(), goal=goal.tolist(),
        geodesic_distance=float(path.geodesic_distance),
        final_position=final_pos.tolist(),
        final_distance_to_goal=dist_to_goal,
        path_length=cum_dist, num_frames=len(traj), num_nodes=len(nodes),
        camera=dict(width=args.width, height=args.height, hfov_deg=args.hfov,
                    height_m=args.camera_height),
        actions=dict(forward_m=args.forward, turn_deg=args.turn,
                     goal_radius=args.goal_radius),
        node_spacing_m=args.node_spacing,
        floor_y=floor_y, floor_tol=args.floor_tol,
        navmesh_mode=args.navmesh_mode,
        navmesh_agent=dict(radius=args.agent_radius, height=args.agent_height,
                           max_climb=args.agent_max_climb, cell_size=args.cell_size),
        clearance_hits={h: [int(i) for i in v] for h, v in hits.items()},
        yaw_convention="radians, 0 = -Z, CCW positive",
    )
    with open(os.path.join(args.out_dir, "traj.json"), "w") as f:
        json.dump(dict(**meta, steps=traj), f, indent=2)
    with open(os.path.join(args.out_dir, "nodes.json"), "w") as f:
        json.dump(nodes, f, indent=2)

    save_topdown(pf, os.path.join(args.out_dir, "topdown.png"),
                 [d["position"] for d in traj] or [start.tolist()],
                 [n["position"] for n in nodes], start, goal)

    print(f"[collect] status={status}  final dist to goal {dist_to_goal:.2f} m")
    print(f"[collect] saved {len(traj)} frames, {len(nodes)} nodes -> {args.out_dir}")


def sim_args(**overrides):
    """The sim-relevant defaults as a Namespace, for reuse by other scripts.

    ``make_sim`` reads a dozen attributes off ``args``; this keeps callers that are
    not this file's argparse (e.g. the policy rollout) from having to know which.
    Note ``load_navmesh``/``rebake_navmesh`` mutate ``navmesh`` and write into
    ``out_dir``.
    """
    ns = argparse.Namespace(
        scene=SCENE, scene_dataset=SCENE_DATASET, navmesh=None,
        navmesh_mode="rebake", agent_radius=0.18, agent_height=1.5,
        agent_max_climb=0.15, cell_size=0.05, out_dir=OUT_DIR,
        width=IMG_W, height=IMG_H, hfov=HFOV_DEG, camera_height=CAMERA_HEIGHT,
        forward=FORWARD_M, turn=TURN_DEG, seed=SEED,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="sample candidate endpoints")
    ap.add_argument("--pair", type=int, default=None, help="index into candidates.json")
    ap.add_argument("--start", type=str, default=None, help="X,Y,Z")
    ap.add_argument("--goal", type=str, default=None, help="X,Y,Z")

    ap.add_argument("--scene", default=SCENE, help="apt_0 .. apt_5")
    ap.add_argument("--scene-dataset", default=SCENE_DATASET)
    ap.add_argument("--navmesh", default=None,
                    help="defaults to <dataset dir>/navmeshes/<scene>.navmesh")
    ap.add_argument("--navmesh-mode", choices=("rebake", "shipped"), default="rebake",
                    help="'shipped' lets the expert walk through the furniture; "
                         "see rebake_navmesh() for why")
    ap.add_argument("--agent-radius", type=float, default=0.18,
                    help="robot half-width used for the navmesh rebake")
    ap.add_argument("--agent-height", type=float, default=1.5)
    ap.add_argument("--agent-max-climb", type=float, default=0.15)
    ap.add_argument("--cell-size", type=float, default=0.05)
    ap.add_argument("--out-dir", default=OUT_DIR)

    ap.add_argument("--width", type=int, default=IMG_W)
    ap.add_argument("--height", type=int, default=IMG_H)
    ap.add_argument("--hfov", type=float, default=HFOV_DEG)
    ap.add_argument("--camera-height", type=float, default=CAMERA_HEIGHT)

    ap.add_argument("--forward", type=float, default=FORWARD_M)
    ap.add_argument("--turn", type=float, default=TURN_DEG)
    ap.add_argument("--goal-radius", type=float, default=GOAL_RADIUS)
    ap.add_argument("--node-spacing", type=float, default=NODE_SPACING_M)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--seed", type=int, default=SEED)

    ap.add_argument("--n-candidates", type=int, default=10)
    ap.add_argument("--dmin", type=float, default=4.0)
    ap.add_argument("--dmax", type=float, default=8.0)
    ap.add_argument("--min-tortuosity", type=float, default=1.0,
                    help="geodesic/euclidean ratio floor; >1.15 forces a corner")
    ap.add_argument("--floor-y", type=float, default=None,
                    help="floor height; auto-detected from the navmesh if unset")
    ap.add_argument("--floor-tol", type=float, default=0.3,
                    help="reject navmesh polys this far off the floor (roof junk)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sim = make_sim(args)
    try:
        if args.list:
            list_candidates(sim, args)
        else:
            if args.pair is not None:
                with open(os.path.join(args.out_dir, "candidates.json")) as f:
                    c = json.load(f)[args.pair]
                s, g = c["start"], c["goal"]
            elif args.start and args.goal:
                s = [float(x) for x in args.start.split(",")]
                g = [float(x) for x in args.goal.split(",")]
            else:
                raise SystemExit("need --list, or --pair N, or --start/--goal")
            collect(sim, args, s, g)
    finally:
        sim.close()
