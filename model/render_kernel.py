import numpy as np
import torch
import torch.nn.functional as F


def rkernel(queries, queries_rot, bases, bases_nvec, bases_value, cfg):
    # TODO (optimize): estimate visibility and filter rays first

    G, Vr, V, H, D = cfg["G"], cfg["Vr"], cfg["V"], cfg["H"], cfg["D"]
    dist_max = cfg["dist_max"]
    device = queries.device
    N, Q, _ = queries.shape
    _, B, _ = bases_value.shape
    assert bases_value.shape[-1] == (G + H) * D
    assert G + H > 0

    bases_codebook_rot = bases_value[..., : G * D].reshape(N, B, G, D)
    bases_codebook_dist = bases_value[..., G * D :].reshape(N, B, H, D)

    vec_qb = bases.reshape(N, 1, B, 2) - queries.reshape(N, Q, 1, 2)  # N,Q,B,2
    dist_qb = vec_qb.norm(dim=-1, keepdim=True)  # N,Q,B,1
    bases_dvec = torch.stack([-bases_nvec[..., 1], bases_nvec[..., 0]], dim=-1)

    i0_NQB = (
        torch.arange(N, device=device).reshape(N, 1, 1).expand(-1, Q, B).reshape(-1)
    )  # note torch.gather is slower and more memory-comsuming
    i1_NQB = (
        torch.arange(Q, device=device).reshape(1, Q, 1).expand(N, -1, B).reshape(-1)
    )
    i2_NQB = (
        torch.arange(B, device=device).reshape(1, 1, B).expand(N, Q, -1).reshape(-1)
    )

    # gather distance codebook
    if H > 0:
        anchors = torch.linspace(0, 1, H, device=device) * dist_max  # H

        anchor_dist = dist_qb - anchors.reshape(1, 1, 1, -1)  # N,Q,B,H
        anchor_closest_idx = anchor_dist.abs().min(dim=-1, keepdim=True)[1]  # N,Q,B,1
        anchor_closest_dist = anchor_dist[
            i0_NQB, i1_NQB, i2_NQB, anchor_closest_idx.reshape(-1)
        ].reshape(
            N, Q, B, 1
        )  # N,Q,B,1
        anchor_aux_idx = (
            anchor_closest_idx + torch.sign(anchor_closest_dist).long()
        ).clamp(
            0, H - 1
        )  # N,Q,B,1
        anchor_aux_dist = anchor_dist[
            i0_NQB, i1_NQB, i2_NQB, anchor_aux_idx.reshape(-1)
        ].reshape(
            N, Q, B, 1
        )  # N,Q,B,1

        value_closest = bases_codebook_dist[
            i0_NQB, i2_NQB, anchor_closest_idx.reshape(-1), :
        ].reshape(N, Q, B, D)
        value_aux = bases_codebook_dist[
            i0_NQB, i2_NQB, anchor_aux_idx.reshape(-1), :
        ].reshape(N, Q, B, D)
        value_closest_weight = anchor_aux_dist.abs() / (
            anchor_closest_dist.abs() + anchor_aux_dist.abs()
        ).clamp(min=1e-6)
        bases_value_dist = (
            value_closest_weight * value_closest
            + (1 - value_closest_weight) * value_aux
        )
    else:
        bases_value_dist = 0

    # gather incident-angle codebook
    if G > 0:
        inangle_sin = F.cosine_similarity(
            vec_qb, bases_nvec.reshape(N, 1, B, 2), dim=-1
        )  # N,Q,B
        inangle_cos = F.cosine_similarity(
            vec_qb, bases_dvec.reshape(N, 1, B, 2), dim=-1
        )  # N,Q,B
        inangle = torch.atan2(inangle_sin, inangle_cos)  # N,Q,B  [-pi~pi]

        group_floor = (
            torch.floor((inangle + np.pi) / (2 * np.pi / G)).long() % G
        )  # N,Q,B
        group_ceil = torch.ceil((inangle + np.pi) / (2 * np.pi / G)).long() % G  # N,Q,B
        ceil_weight = ((inangle + np.pi) / (2 * np.pi / G) % G % 1).unsqueeze(
            -1
        )  # N,Q,B,1
        value_floor = bases_codebook_rot[
            i0_NQB, i2_NQB, group_floor.reshape(-1), :
        ].reshape(N, Q, B, D)
        value_ceil = bases_codebook_rot[
            i0_NQB, i2_NQB, group_ceil.reshape(-1), :
        ].reshape(N, Q, B, D)
        bases_value_ang = value_ceil * ceil_weight + value_floor * (1 - ceil_weight)
    else:
        bases_value_ang = 0

    # merge dist/rot feat
    bases_value = bases_value_dist + bases_value_ang  # N,Q,B,D

    # visibility rendering
    outangle = torch.atan2(vec_qb[..., 1:2], vec_qb[..., 0:1])  # N,Q,B,1
    if queries_rot is not None:
        outangle += np.pi * 0.5 - queries_rot.reshape(
            N, Q, 1, 1
        )  # clock-wise, front zero degree, add offsets
    else:
        outangle += np.pi * 0.5  # clock-wise, front zero degree
    outangle_group = torch.floor(outangle / (2 * np.pi / Vr)).long() % Vr  # N,Q,B,1
    outangle_group_cdist = (
        outangle / (2 * np.pi / Vr) % Vr % 1 - 0.5
    )  # N,Q,B,1,  [-0.5...center...+0.5]

    visible_base_idx = []
    fake_base_mask = []
    for i in range(Vr):
        tmp_dist_pb = dist_qb.clone()
        tmp_dist_pb[outangle_group != i] = np.inf  # N,Q,B,1
        vmin, amin = torch.min(tmp_dist_pb, dim=2)  # N,Q,1
        visible_base_idx.append(amin)  # N,Q,1
        fake_base_mask.append(vmin == np.inf)

    visible_base_idx = torch.cat(visible_base_idx, dim=-1).unsqueeze(dim=-1)  # N,Q,Vr,1
    fake_base_mask = torch.cat(fake_base_mask, dim=-1)  # N,Q,Vr

    i0_NQVr = (
        torch.arange(N, device=device).reshape(N, 1, 1).expand(-1, Q, Vr).reshape(-1)
    )
    i1_NQVr = (
        torch.arange(Q, device=device).reshape(1, Q, 1).expand(N, -1, Vr).reshape(-1)
    )
    i2_NQVr = (
        torch.arange(Vr, device=device).reshape(1, 1, Vr).expand(N, Q, -1).reshape(-1)
    )

    ring_closest = bases_value[
        i0_NQVr, i1_NQVr, visible_base_idx.reshape(-1), :
    ].reshape(N, Q, Vr, D)
    ring_closest[fake_base_mask] = 0

    # bilinear interpolation on Vr
    closest_cdist = outangle_group_cdist[
        i0_NQVr, i1_NQVr, visible_base_idx.reshape(-1), :
    ].reshape(
        N, Q, Vr, 1
    )  # N,Q,Vr,1
    aux_ring_idx = (
        (i2_NQVr - torch.sign(closest_cdist).long().reshape(-1)) % Vr
    ).reshape(N, Q, Vr, 1)
    aux_cdist = (
        closest_cdist[i0_NQVr, i1_NQVr, aux_ring_idx.reshape(-1), :].reshape(
            N, Q, Vr, 1
        )
        + 1.0
    )

    ring_aux = ring_closest[i0_NQVr, i1_NQVr, aux_ring_idx.reshape(-1), :].reshape(
        N, Q, Vr, D
    )
    closest_weight = aux_cdist / (closest_cdist.abs() + aux_cdist)
    ring_feat = ring_closest * closest_weight + ring_aux * (
        1.0 - closest_weight
    )  # N,Q,Vr,D

    # pool Vr to V
    if V > 0:
        # NOTE: adaptive pooling has non-equal step that is bad for rotation
        assert Vr % V == 0
        pad_size = max(Vr // V // 2, 1)
        kernel_size = Vr // V + pad_size * 2
        stride = Vr // V
        ring_feat = ring_feat.reshape(N * Q, Vr, D).permute(0, 2, 1)  # N*Q,D,Vr
        ring_feat = F.pad(ring_feat, (pad_size, pad_size), mode="circular")
        ring_feat = F.avg_pool1d(ring_feat, kernel_size, stride)  # N*Q,D,V
        ring_feat = ring_feat.permute(0, 2, 1).reshape(N, Q, V, D)  # N,Q,V*D

    return ring_feat
