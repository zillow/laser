import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time

from model.render_kernel import rkernel
from model.fplocnet import persp2equir


def render_result(bases, bases_feat, loc_gt, loc_est, scale=30, border=1):
    n_bases = bases.shape[0]
    bases = np.copy(bases)
    affine = (border - bases.min(axis=0), scale)
    bases = (bases + affine[0]) * affine[1]
    loc_gt = (loc_gt + affine[0]) * affine[1]
    loc_est = (loc_est + affine[0]) * affine[1]

    W, H = np.ptp(bases, axis=0).astype(np.int) + int(2 * border * scale)
    canvas = np.zeros((H, W, 3), np.uint8)

    door_label = bases_feat[:, -2]
    window_label = bases_feat[:, -1]
    for i in range(n_bases):
        color = [255, 0, 0]
        if door_label[i] > 0.5:
            color[2] = 255
        if window_label[i] > 0.5:
            color[1] = 255
        cv2.circle(canvas, tuple(np.round(bases[i]).astype(np.int)), 1, tuple(color))

    cv2.circle(
        canvas,
        tuple(np.round(loc_gt).astype(np.int)),
        int(scale * 0.1),
        [0, 0, 255],
        -1,
    )  # 0.1 meter circle
    cv2.circle(
        canvas, tuple(np.round(loc_gt).astype(np.int)), int(scale * 0.5), [0, 0, 255]
    )  # 0.5 meter circle
    cv2.circle(
        canvas, tuple(np.round(loc_gt).astype(np.int)), int(scale), [0, 0, 255]
    )  # 1 meter circle
    cv2.drawMarker(
        canvas,
        tuple(np.round(loc_est).astype(np.int)),
        [255, 255, 255],
        cv2.MARKER_CROSS,
        int(scale * 0.2),
        1,
    )

    return canvas


def sampling_in_range(bases, interval=0.1, border=0.0):
    border = max(interval, border)
    bases_min = bases.min(dim=0)[0] - border
    bases_max = bases.max(dim=0)[0] + border + interval
    Iy, Ix = torch.meshgrid(
        torch.arange(bases_min[1], bases_max[1], interval),
        torch.arange(bases_min[0], bases_max[0], interval),
    )
    samples_loc = torch.stack([Ix, Iy], dim=-1).reshape(-1, 2)
    return samples_loc.to(bases.device), tuple(Ix.shape)


@torch.no_grad()
def sample_floorplan(data, model, cfg, sample_grid=0.1, batch_size=256):
    assert data["bases"].shape[0] == 1  # make sure batch_size=1
    model.eval()
    _, fp_feat = model(None, data["bases_feat"])
    samples_loc, samples_original_shape = sampling_in_range(
        data["bases"][0], interval=sample_grid
    )
    n_samples = samples_loc.shape[0]

    _start_time = time.perf_counter()
    samples_feat_subs = []
    for i in range(0, n_samples, batch_size):
        # if cfg['align_rot']
        samples_feat_subs.append(
            rkernel(
                samples_loc[i : i + batch_size].unsqueeze(0),
                None,
                data["bases"],
                data["bases_normal"],
                fp_feat,
                cfg,
            )
        )  # N,Q,V,D
    samples_feat = torch.cat(samples_feat_subs, dim=1)
    sampling_time = time.perf_counter() - _start_time

    return {
        "n_samples": n_samples,
        "samples_loc": samples_loc,
        "fp_feat": fp_feat,
        "samples_feat": samples_feat,
        "samples_original_shape": samples_original_shape,
        "sampling_fps": n_samples / sampling_time,
        "sampling_time": sampling_time,
    }


@torch.no_grad()
def match_images(
    sample_ret, data, model, cfg, mode="refine", sample_nrots=16, max_refine_its=3
):
    assert mode in ["match", "refine"]

    samples_loc, samples_feat = sample_ret["samples_loc"], sample_ret["samples_feat"]
    fp_feat = sample_ret["fp_feat"]
    samples_original_shape = sample_ret["samples_original_shape"]

    model.eval()
    n_images = data["query_image"].shape[0]
    _start_time = time.perf_counter()

    loc_gts = []
    loc_ests = []
    rot_gts = []
    rot_ests = []
    terrs = []
    rerrs = []
    score_maps = []
    rot_maps = []
    img_feats = []
    for i in range(n_images):

        if cfg["view_type"] == "pview":
            img_feat, _ = model(data["query_image"][i : i + 1], None)
            img_feat, img_feat_mask = persp2equir(
                img_feat, data["gt_fov"][i : i + 1], cfg["V"]
            )
            score_fun = (
                lambda x, y: (
                    F.cosine_similarity(x, y, dim=-1).sum(dim=-1)
                    / img_feat_mask.sum(dim=-1)
                    + 1
                )
                * 0.5
            )
        elif cfg["view_type"] == "eview":
            if cfg["V"] != 1:  # V=1 is disable circular feat
                V_fov = float(data["gt_fov"][i : i + 1]) / 360 * cfg["V"]
                assert V_fov % 1 == 0
                V_fov = int(V_fov)
            else:
                V_fov = 1
            img_feat, _ = model(data["query_image"][i : i + 1], None, V=V_fov)  # N,V,D
            img_feat = F.pad(img_feat.permute(0, 2, 1), (0, cfg["V"] - V_fov)).permute(
                0, 2, 1
            )
            score_fun = (
                lambda x, y: (F.cosine_similarity(x, y, dim=-1).sum(dim=-1) / V_fov + 1)
                * 0.5
            )
        else:
            raise "Unknown view_type"

        img_feats.append(img_feat.cpu().numpy())

        if mode in ["refine", "match"]:
            score_list = []
            rot_samples = torch.arange(sample_nrots).float() / sample_nrots * 360
            # Note bilinear interpolation cannot be applied to partially masked image-feat except V=sample_nrots
            # Decide to rotate image-feat(fast) or map-feat(a bit slower tiny bit slower)
            if data["gt_fov"][i] < 360 and sample_nrots != cfg["V"]:
                samples_feat_padded = F.pad(
                    samples_feat.squeeze(0).permute(0, 2, 1),
                    (cfg["V"], 0),
                    mode="circular",
                )  # N,D,V
                for r in rot_samples:
                    offset = r / 360 * cfg["V"]
                    offset_floor, offset_ceil = int(torch.floor(offset)), int(
                        torch.ceil(offset)
                    )
                    offset_floor_weight = offset_ceil - offset  # bilinear weight
                    Vidx = torch.arange(cfg["V"])
                    samples_feat_roted = samples_feat_padded[
                        ..., Vidx + offset_floor
                    ] * offset_floor_weight + samples_feat_padded[
                        ..., Vidx + offset_ceil
                    ] * (
                        1 - offset_floor_weight
                    )
                    samples_feat_roted = samples_feat_roted.permute(0, 2, 1).unsqueeze(
                        0
                    )  # N,Q,V,D
                    score_list.append(score_fun(img_feat, samples_feat_roted))
            else:
                img_feat_padded = F.pad(
                    img_feat.permute(0, 2, 1), (cfg["V"], 0), mode="circular"
                )  # N,D,V
                for r in rot_samples:
                    offset = r / 360 * cfg["V"]
                    offset_floor, offset_ceil = int(torch.floor(offset)), int(
                        torch.ceil(offset)
                    )
                    offset_floor_weight = offset_ceil - offset  # bilinear weight
                    Vidx = torch.arange(cfg["V"])
                    img_feat_roted = img_feat_padded[
                        ..., cfg["V"] + Vidx - offset_floor
                    ] * offset_floor_weight + img_feat_padded[
                        ..., cfg["V"] + Vidx - offset_ceil
                    ] * (
                        1 - offset_floor_weight
                    )
                    img_feat_roted = img_feat_roted.permute(0, 2, 1)  # N,V,D
                    score_list.append(
                        score_fun(img_feat_roted.unsqueeze(1), samples_feat)
                    )
            score_list = torch.stack(score_list, dim=-1)
            scores, matched_rot_idxs = score_list.max(dim=-1)
            loc_est = samples_loc[scores.argmax()].reshape(2)
            rot_est = matched_rot_idxs.reshape(-1)[scores.argmax()].reshape(1, 1, 1)
            rot_est = (rot_samples[rot_est] / 180 * np.pi).to(scores.device)

            if mode == "refine":
                score_cur = scores.max().item()
                feat_cur = rkernel(
                    loc_est.reshape(1, 1, 2),
                    rot_est,
                    data["bases"],
                    data["bases_normal"],
                    fp_feat,
                    cfg,
                )
                for it in range(max_refine_its):
                    t_offset, r_offset = model(img_feat, feat_cur, refine=True)
                    loc_refined = loc_est + t_offset
                    rot_refined = rot_est + r_offset
                    feat_cur = rkernel(
                        loc_refined.reshape(1, 1, 2),
                        rot_refined,
                        data["bases"],
                        data["bases_normal"],
                        fp_feat,
                        cfg,
                    )
                    score_refined = score_fun(img_feat.unsqueeze(1), feat_cur).item()

                    if (
                        score_refined > score_cur or it == 0
                    ):  # at least one refinement it to unquantize the est
                        score_cur = score_refined
                        loc_est = loc_refined
                        rot_est = rot_refined
                    else:
                        # print(it)
                        break

        loc_gt = data["gt_loc"][i].reshape(2).cpu().numpy()
        rot_gt = data["gt_rot"][i].reshape(1).cpu().numpy()
        loc_est = loc_est.reshape(2).cpu().numpy()
        rot_est = rot_est.reshape(1).cpu().numpy()
        terr = np.linalg.norm(loc_gt - loc_est)
        rerr = (np.abs(rot_est - rot_gt) * 180 / np.pi) % 360
        rerr = 360 - rerr if rerr > 180 else rerr
        score_map = (
            scores.reshape(samples_original_shape[0], samples_original_shape[1])
            .cpu()
            .numpy()
        )
        rot_map = (
            (
                matched_rot_idxs.reshape(
                    samples_original_shape[0], samples_original_shape[1]
                )
                .cpu()
                .numpy()
                / sample_nrots
            )
            * 2
            * np.pi
        )

        loc_gts.append(loc_gt)
        loc_ests.append(loc_est)
        rot_gts.append(rot_gt)
        rot_ests.append(rot_est)
        terrs.append(terr)
        rerrs.append(rerr)
        score_maps.append(score_map)
        rot_maps.append(rot_map)

    matching_time = time.perf_counter() - _start_time

    return {
        "n_images": n_images,
        "img_feats": np.stack(img_feats, axis=0),
        "loc_gts": np.stack(loc_gts, axis=0),
        "loc_ests": np.stack(loc_ests, axis=0),
        "rot_gts": np.stack(rot_gts, axis=0),
        "rot_ests": np.stack(rot_ests, axis=0),
        "terrs": np.stack(terrs, axis=0).reshape(-1),
        "rerrs": np.stack(rerrs, axis=0).reshape(-1),
        "score_maps": np.stack(score_maps, axis=0),
        "rot_maps": np.stack(rot_maps, axis=0),
        "matching_time": matching_time,
        "matching_fps": n_images / matching_time,
    }
