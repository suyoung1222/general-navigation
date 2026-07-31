"""ROS-free NoMaD inference adapter.

Model definitions come from the visualnav-transformer fork (installed editable, so
``import vint_train`` / ``import diffusion_policy`` just work). Only the handful of
helpers that upstream keeps in ROS- or training-coupled modules are reimplemented here:

  * ``deployment/src/utils.py`` does ``from sensor_msgs.msg import Image`` at module
    top, so importing ``load_model``/``transform_images`` from it requires a ROS
    install. Ported verbatim below.
  * ``vint_train.training.train_utils`` pulls in wandb and the full training stack just
    to expose ``get_action``. That is eight lines; ported below.

Nothing in this module touches habitat.
"""

import os
import sys

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from torchvision import transforms

# ---------------------------------------------------------------------------
VNT_ROOT = "/home/suyoung/Documents/visualnav-transformer"

# vint_train is a normal pip-editable install. diffusion_policy cannot be: the
# submodule ships no __init__.py at any level, so its setup.py's bare
# find_packages() matches nothing and `pip install -e` produces an importable
# distribution with an empty module MAPPING -- it "succeeds" and then ImportErrors.
# It does import cleanly as a PEP 420 implicit namespace package, so put its repo
# root on the path explicitly rather than editing the fork.
if VNT_ROOT + "/diffusion_policy" not in sys.path:
    sys.path.insert(0, VNT_ROOT + "/diffusion_policy")

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork          # noqa: E402
from vint_train.models.nomad.nomad_vint import (                       # noqa: E402
    NoMaD_ViNT, replace_bn_with_gn)
from diffusion_policy.model.diffusion.conditional_unet1d import (      # noqa: E402
    ConditionalUnet1D)

# ---------------------------------------------------------------------------
NOMAD_CONFIG = f"{VNT_ROOT}/train/config/nomad.yaml"
NOMAD_CKPT = f"{VNT_ROOT}/deployment/model_weights/nomad.pth"
ROBOT_CONFIG = f"{VNT_ROOT}/deployment/config/robot.yaml"
DATA_CONFIG = f"{VNT_ROOT}/train/vint_train/data/data_config.yaml"

# Pinned upstream constants. close_threshold/radius/waypoint/num_samples are argparse
# defaults in deployment/src/navigate.py, NOT yaml entries -- do not go looking for
# them in a config file. (navigate.py's --radius help text claims "default: 2" and is
# simply wrong; the actual default is 4.)
CLOSE_THRESHOLD = 3
RADIUS = 4
WAYPOINT_IDX = 2
NUM_SAMPLES = 8
# ---------------------------------------------------------------------------


def load_configs():
    """Return (model_params, robot_params, action_stats), all read from the fork."""
    with open(NOMAD_CONFIG) as f:
        model_params = yaml.safe_load(f)
    with open(ROBOT_CONFIG) as f:
        robot_params = yaml.safe_load(f)
    with open(DATA_CONFIG) as f:
        action_stats = {k: np.array(v)
                        for k, v in yaml.safe_load(f)["action_stats"].items()}
    return model_params, robot_params, action_stats


def load_nomad(cfg, device, ckpt=NOMAD_CKPT):
    """Build NoMaD and load the checkpoint, asserting the state dict actually matches.

    Upstream's ``load_model`` uses ``strict=False``, which means a renamed key loads as
    *random weights* with no error at all. Since ``replace_bn_with_gn`` rewrites module
    names before loading, that failure mode is a real one -- so assert instead.
    """
    if cfg["vision_encoder"] != "nomad_vint":
        raise ValueError(f"unsupported vision_encoder: {cfg['vision_encoder']}")

    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=cfg["encoding_size"],
        context_size=cfg["context_size"],
        mha_num_attention_heads=cfg["mha_num_attention_heads"],
        mha_num_attention_layers=cfg["mha_num_attention_layers"],
        mha_ff_dim_factor=cfg["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=cfg["encoding_size"],
        down_dims=cfg["down_dims"],
        cond_predict_scale=cfg["cond_predict_scale"],
    )
    dist_pred_net = DenseNetwork(embedding_dim=cfg["encoding_size"])
    model = NoMaD(vision_encoder=vision_encoder,
                  noise_pred_net=noise_pred_net,
                  dist_pred_net=dist_pred_net)

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"NoMaD weights not found: {ckpt}")
    state_dict = torch.load(ckpt, map_location=device, weights_only=True)
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint does not match the model\n"
            f"  missing ({len(result.missing_keys)}): {result.missing_keys[:8]}\n"
            f"  unexpected ({len(result.unexpected_keys)}): {result.unexpected_keys[:8]}")
    print(f"[nomad] loaded {len(state_dict)} tensors from {ckpt}, 0 missing/unexpected")

    model.to(device)
    model.eval()
    return model


def transform_images(pil_imgs, image_size, center_crop=False):
    """Verbatim port of deployment/src/utils.py::transform_images.

    Returns ``(1, 3*N, H, W)`` -- images are concatenated along *channels*, not batch.
    Our habitat frames are 640x480, i.e. exactly the 4:3 ratio used in training, so
    center_crop True/False are equivalent here and neither distorts the aspect.
    """
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    out = []
    for pil_img in pil_imgs:
        w, h = pil_img.size
        if center_crop:
            import torchvision.transforms.functional as TF
            ar = 4 / 3
            pil_img = (TF.center_crop(pil_img, (h, int(h * ar))) if w > h
                       else TF.center_crop(pil_img, (int(w / ar), w)))
        pil_img = pil_img.resize(image_size)
        out.append(torch.unsqueeze(tf(pil_img), 0))
    return torch.cat(out, dim=1)


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def get_action(diffusion_output, action_stats):
    """Normalized deltas -> cumulative waypoints in the robot body frame.

    Port of vint_train.training.train_utils.get_action. Units are the training
    datasets' *metric waypoint spacing*, NOT meters -- the conversion to meters is the
    ``MAX_V / RATE`` scaling applied by the caller.
    """
    ndeltas = to_numpy(diffusion_output).reshape(diffusion_output.shape[0], -1, 2)
    ndeltas = (ndeltas + 1) / 2
    ndeltas = ndeltas * (action_stats["max"] - action_stats["min"]) + action_stats["min"]
    return np.cumsum(ndeltas, axis=1)


def pd_controller(waypoint, dt, max_v, max_w):
    """Port of deployment/src/pd_controller.py::pd_controller (ROS I/O removed).

    Despite the name it is a pure geometric P controller: forward speed is along-track
    displacement over dt, yaw rate is the required heading change over dt, both
    saturated. v never goes negative, so the robot cannot reverse.
    """
    eps = 1e-8
    if len(waypoint) == 2:
        dx, dy = waypoint
        hx = hy = None
    else:
        dx, dy, hx, hy = waypoint

    if hx is not None and abs(dx) < eps and abs(dy) < eps:
        v = 0.0
        w = float(np.mod(np.arctan2(hy, hx) + np.pi, 2 * np.pi) - np.pi) / dt
    elif abs(dx) < eps:
        v = 0.0
        w = np.sign(dy) * np.pi / (2 * dt)
    else:
        v = dx / dt
        w = np.arctan(dy / dx) / dt

    return float(np.clip(v, 0.0, max_v)), float(np.clip(w, -max_w, max_w))


def load_topomap(run_dir):
    """Load a topomap from a collect_expert_topomap.py run directory.

    Returns (images, nodes) where nodes carries each node's ground-truth pose -- we
    need those poses for privileged success evaluation, which is why this reads
    nodes.json rather than globbing the directory the way upstream does.
    """
    import json
    with open(os.path.join(run_dir, "nodes.json")) as f:
        nodes = json.load(f)
    imgs = [PILImage.open(os.path.join(run_dir, "nodes", f"node_{n['node_id']:03d}.jpg"))
            .convert("RGB") for n in nodes]
    return imgs, nodes
