import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .resnet import ResNet50
from .pointnet import PointNet
from .render_kernel import rkernel


def outlier_sampling(bases, n_samples, border):
    N, B, _ = bases.shape  # N,B,2
    bmin = bases.min(dim=1, keepdim=True)[0] - border  # N,1,2
    bmax = bases.max(dim=1, keepdim=True)[0] + border  # N,1,2
    brng = bmax - bmin  # N,2
    trans = torch.rand(N, n_samples, 2, device=bases.device) * brng + bmin  # N,Q,2
    rot = torch.rand(N, n_samples, 1, device=bases.device) * np.pi * 2  # N,Q,1
    return trans, rot


def inlier_offset_sampling(N, n_samples, device, trans_radius, rot_radius):
    trans_offset = (
        torch.rand(N, n_samples, 2, device=device) * 2 - 1
    ) * trans_radius  # N,Q,2
    rot_offset = (torch.rand(N, n_samples, 1, device=device) * 2 - 1) * (
        rot_radius / 180 * np.pi
    )  # N,Q,1
    return trans_offset, rot_offset


def persp2equir(feat, FoV, V):
    N, Vp, D = feat.shape
    ring_anchors = (
        torch.arange(V, device=feat.device) * (360 / V) + 360 / V / 2 - 180
    )  # -180----0----180 V
    pview_anchors = (
        (torch.arange(Vp, device=feat.device) / Vp * 2 - 1 + 2 / Vp / 2)
        .reshape(1, Vp)
        .expand(N, -1)
    )  # -1---0---+1 N,Vp
    if torch.is_tensor(FoV):
        pview_anchors = torch.rad2deg(
            torch.atan(pview_anchors * torch.tan(torch.deg2rad(FoV / 2)))
        )  # N,Vp
    else:
        pview_anchors = torch.rad2deg(
            torch.atan(pview_anchors * np.tan(np.deg2rad(FoV / 2)))
        )  # N,Vp
    pview_groups = (
        (pview_anchors.reshape(N, Vp, 1) - ring_anchors.reshape(1, 1, V))
        .abs()
        .argmin(dim=-1)
    )
    pview_groups = (pview_groups - V // 2) % V  # N,Vp
    ret = torch.zeros((N, V, D), device=feat.device)
    ret_mask = torch.zeros((N, V), device=feat.device, dtype=torch.bool)
    for i in range(V):
        feat_tmp = feat.clone()
        group_mask = pview_groups == i
        feat_tmp[~group_mask] = 0
        total = group_mask.sum(dim=1)
        ret[:, i] = feat_tmp.sum(dim=1) / total.float().clamp(1e-8).reshape(N, 1)
        ret_mask[:, i] = total > 0
    return ret, ret_mask


class FpLocNet(nn.Module):
    def __init__(self, cfg):
        super(FpLocNet, self).__init__()
        self.cfg = cfg
        self.resnet = ResNet50(V=cfg["V"], D=cfg["D"])

        if "disable_pointnet" in cfg and cfg["disable_pointnet"]:
            self.use_shared_codebook = True
            self.codebook = nn.parameter.Parameter(
                torch.zeros((cfg["G"] + cfg["H"]) * cfg["D"], 3), requires_grad=True
            )
        else:
            self.use_shared_codebook = False
            self.pointnet = PointNet(in_chs=7, out_chs=(cfg["G"] + cfg["H"]) * cfg["D"])

        # at inference time, we are interested in running equir-fov images on a model trained with panorama
        # for correctly doing so, the refiner padding need be taken care of
        if cfg["view_type"] == "eview" and cfg["fov"] != 360:
            if cfg["V"] != 1:  # V=1 is disable circular feat
                V_fov = float(cfg["fov"]) / 360 * cfg["V"]
                assert V_fov % 1 == 0
                V_fov = int(V_fov)
            else:
                V_fov = 1
            self.V_fov = V_fov
            self.is_partial_eview = True
            if V_fov == 1:
                refiner_padding = "zeros"
            else:
                refiner_padding = "reflect"
        else:
            self.is_partial_eview = False
            refiner_padding = "circular"
        print(f"refiner_padding = {refiner_padding}")

        self.refiner = nn.Sequential(
            nn.BatchNorm1d(2 * cfg["D"]),
            nn.ReLU(),
            nn.Conv1d(
                2 * cfg["D"],
                128,
                3,
                1,
                padding=1,
                padding_mode=refiner_padding,
                bias=False,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(
                128, 128, 3, 1, padding=1, padding_mode=refiner_padding, bias=False
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 3),
        )

    def forward(self, x1, x2, refine=False, V=None):
        if not refine:
            img, fp = x1, x2
            img_feat, fp_feat = None, None
            if img is not None:
                img_feat = self.resnet(img, V=V)
            if fp is not None:
                if self.use_shared_codebook:
                    N, X, _ = fp.shape
                    fp_feat = self.codebook[:, 0].unsqueeze(0).expand(N * X, -1) + (
                        self.codebook[:, 1:].unsqueeze(0).expand(N * X, -1, -1)
                        * fp[..., -2:].reshape(N * X, 1, -1)
                    ).sum(-1)
                    fp_feat = fp_feat.reshape(N, X, -1)
                else:
                    fp_feat = self.pointnet(fp.permute(0, 2, 1)).permute(
                        0, 2, 1
                    )  # N,X,(G+H)*D
            return img_feat, fp_feat
        else:
            img_feat, render_feat = x1, x2
            N, Q, V, D = render_feat.shape
            x = torch.cat(
                [img_feat.reshape(N, 1, V, D).expand(-1, Q, -1, -1), render_feat],
                dim=-1,
            )  # N,Q,V,2D
            x = x.permute(0, 1, 3, 2).reshape(N * Q, 2 * D, V)
            if self.is_partial_eview:
                x = x[..., : self.V_fov]
            x = self.refiner(x)
            r_offset = x[:, 0:1].reshape(N, Q, 1)
            t_offset = x[:, 1:3].reshape(N, Q, 2)
            return t_offset, r_offset


def segmented_cosine_distance(a, b, m=None):
    # N,Q,V,D = a.shape, b.shape
    # N,Q,V,1 = m.shape
    if m is None:
        return 1.0 - F.cosine_similarity(a, b, dim=-1).mean(dim=-1)
    else:
        return ((1.0 - F.cosine_similarity(a, b, dim=-1)) * m.squeeze(dim=-1)).sum(
            dim=-1
        ) / m.squeeze(-1).sum(dim=-1)


def avg_reducer(x, m=None):
    # N,Q,V,D = x.shape
    # N,Q,V,1 = m.shape
    x = x / x.norm(dim=-1, keepdim=True).clamp(1e-6)  # normalize is important
    if m is None:
        x = x.mean(dim=-2, keepdim=True)
    else:
        x = (x * m).sum(dim=-2, keepdim=True) / m.sum(dim=-2, keepdim=True)
    return x


def quick_fplocnet_call(model, data, cfg, is_training=False, optimizer=None):
    if is_training:
        assert optimizer is not None
        model.train()
    else:
        model.eval()

    for k in data.keys():
        if torch.is_tensor(data[k]) and not data[k].is_cuda:
            data[k] = data[k].cuda()
    if cfg["disable_semantics"]:
        data["bases_feat"][..., -2:] = 0

    details = {}

    if is_training:
        optimizer.zero_grad()

    N = data["bases"].shape[0]
    V, D, Q = cfg["V"], cfg["D"], cfg["Q"]
    Q_refine = cfg["Q_refine"]
    G, H = cfg["G"], cfg["H"]

    outlier_sample_trans, outlier_sample_rot = outlier_sampling(
        data["bases"], n_samples=Q, border=0.5
    )  # N,Q,2, N,Q,1
    gt_trans = data["gt_loc"].reshape(N, 1, 2)  # N,1,2
    gt_rot = data["gt_rot"].reshape(N, 1, 1)  # N,1,1

    Q_seg = Q // 3
    outlier_sample_trans[
        :, 0:Q_seg, :
    ] = gt_trans  # first seg [0-Q_seg], random rotation, same translation
    outlier_sample_rot[
        :, Q_seg : 2 * Q_seg, :
    ] = gt_rot  # second seg [Qseg-2*Qseg], same rotation, random translation
    #   ------------------------------------------     third seg [2*Qseg-3*Qseg], random rotation, random translation

    img_feat, fp_feat = model(
        data["query_image"], data["bases_feat"]
    )  # N,V,D, N,(G+H)*D
    if cfg["view_type"] == "pview":
        img_feat, img_feat_mask = persp2equir(img_feat, data["gt_fov"], V)
        img_feat = img_feat.reshape(N, 1, V, D)
        img_feat_mask = img_feat_mask.reshape(N, 1, V, 1)
    elif cfg["view_type"] == "eview":
        img_feat = img_feat.reshape(N, 1, V, D)
        img_feat_mask = None
    else:
        raise "Unknown view_type"

    negative_feat = rkernel(
        outlier_sample_trans,
        outlier_sample_rot,
        data["bases"],
        data["bases_normal"],
        fp_feat,
        cfg,
    )  # N,Q,V,D
    positive_feat = rkernel(
        gt_trans, gt_rot, data["bases"], data["bases_normal"], fp_feat, cfg
    )  # N,1,V,D

    negative_feat_reduced = avg_reducer(
        negative_feat[:, Q_seg:, ...], img_feat_mask
    )  # exclude first seg (same translation, random rotation)
    positive_feat_reduced = avg_reducer(positive_feat, img_feat_mask)
    img_feat_reduced = avg_reducer(img_feat, img_feat_mask)

    triplet_loss = F.triplet_margin_with_distance_loss(
        img_feat,
        positive_feat,
        negative_feat,
        swap=True,
        reduction="mean",
        margin=1.0,
        distance_function=lambda a, b: segmented_cosine_distance(a, b, img_feat_mask),
    )

    triplet_loss_reduced = F.triplet_margin_with_distance_loss(
        img_feat_reduced,
        positive_feat_reduced,
        negative_feat_reduced,
        swap=True,
        reduction="mean",
        margin=1.0,
        distance_function=segmented_cosine_distance,
    )

    # refinement step
    inlier_sample_trans_offset, inlier_sample_rot_offset = inlier_offset_sampling(
        N, Q_refine, gt_trans.device, trans_radius=0.5, rot_radius=30
    )
    inlier_sample_trans = gt_trans - inlier_sample_trans_offset
    inlier_sample_rot = gt_rot - inlier_sample_rot_offset
    inlier_feat = rkernel(
        inlier_sample_trans,
        inlier_sample_rot,
        data["bases"],
        data["bases_normal"],
        fp_feat,
        cfg,
    )  # N,Q,V,D

    toffset_est, roffset_est = model(img_feat, inlier_feat, refine=True)

    refine_t_loss = (toffset_est - inlier_sample_trans_offset).norm(dim=-1).mean()
    refine_r_loss = (roffset_est - inlier_sample_rot_offset).abs().mean()

    # train & log
    loss_total = triplet_loss + refine_t_loss + refine_r_loss + triplet_loss_reduced

    if is_training:
        loss_total.backward()
        optimizer.step()

    details["triplet_loss"] = triplet_loss.item()
    details["triplet_loss_reduced"] = triplet_loss_reduced.item()
    details["refine_t_loss"] = refine_t_loss.item()
    details["refine_r_loss"] = refine_r_loss.item()
    details["loss_total"] = loss_total.item()

    return details
