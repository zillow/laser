import cv2
import numpy as np
import torch


def render_fp(data, loc_viz, scale=30, border=1):
    bases, bases_feat = (
        data["bases"][0].cpu().numpy(),
        data["bases_feat"][0].cpu().numpy(),
    )
    bases_normal = data["bases_normal"][0].cpu().numpy()
    # loc_gt, loc_est =  est['loc_gt'], est['loc_est']
    n_bases = bases.shape[0]
    bases = np.copy(bases)
    affine = (border - bases.min(axis=0), scale)
    bases = (bases + affine[0]) * affine[1]
    loc_viz = (loc_viz + affine[0]) * affine[1]

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
        # color = [int(bases_normal[i,0]*128+128), int(bases_normal[i,1]*128+128), 0]
        cv2.circle(canvas, tuple(np.round(bases[i]).astype(np.int)), 1, tuple(color))

    cv2.circle(
        canvas,
        tuple(np.round(loc_viz).astype(np.int)),
        int(scale * 0.1),
        [0, 0, 255],
        -1,
    )

    return canvas


def find_interesting_fov(pano):
    H, W = 360, 720
    pano_mid = cv2.resize(pano, (W, H))[H // 4 : 3 * H // 4, :]  # crop mid 90 fov
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliencyMap = saliency.computeSaliency(pano_mid)
    if not success:
        return 0
    else:
        score = saliencyMap.mean(axis=0, keepdims=True)
        score = cv2.blur(
            score, (30, 1)
        )  # TODO: why borderType=cv2.BORDER_WRAP not supported???
        center = (score.reshape(-1).argmax() / W) * 360
        return center


def is_polygon_clockwise(lines):
    return np.sum(np.cross(lines[:, 0], lines[:, 1], axis=-1)) > 0


def poly_verts_to_lines(verts):
    n_verts = verts.shape[0]
    if n_verts == 0:
        return
    assert n_verts > 1
    lines = np.stack([verts[:-1], verts[1:]], axis=1)  # N,2,2
    return lines


def sample_points_from_lines(lines, interval):
    n_lines = lines.shape[0]
    lengths = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=1)
    n_samples_per_line = np.ceil(lengths / interval).astype(np.int)

    lines_normal = (lines[:, 0] - lines[:, 1]) / np.maximum(
        np.linalg.norm(lines[:, 0] - lines[:, 1], axis=-1, keepdims=True), 1e-8
    )  # N,2
    lines_normal = np.stack([lines_normal[:, 1], -lines_normal[:, 0]], axis=1)

    samples = []
    samples_normal = []
    for l in range(n_lines):
        if n_samples_per_line[l] == 0:
            continue
        p = np.arange(n_samples_per_line[l]).reshape(-1, 1) / n_samples_per_line[l] + (
            0.5 / n_samples_per_line[l]
        )  # uniform sampling
        # p = np.random.rand(n_samples_per_line[l]).reshape(-1,1) # random sampling
        samples.append(p * lines[l : l + 1, 0] + (1 - p) * lines[l : l + 1, 1])
        samples_normal.append(np.repeat(lines_normal[l].reshape(1, 2), p.size, axis=0))
    samples = np.concatenate(samples, axis=0)
    samples_normal = np.concatenate(samples_normal, axis=0)
    return samples, samples_normal


def points_on_lines(points, lines, eps=1e-3):
    N, _ = points.shape  # N,2
    X, _, _ = lines.shape  # X,2,2
    if X == 0:
        return np.zeros((N,), dtype=np.bool)
    side_a = np.linalg.norm(
        points.reshape(N, 1, 2) - lines[:, 0].reshape(1, X, 2), axis=-1
    )  # N,X
    side_b = np.linalg.norm(
        points.reshape(N, 1, 2) - lines[:, 1].reshape(1, X, 2), axis=-1
    )  # N,X
    side_c = np.linalg.norm(lines[:, 0] - lines[:, 1], axis=-1).reshape(1, X)  # 1,X
    residual_mat = side_a + side_b - side_c  # N,X
    best_idx = np.argmin(residual_mat, axis=1)  # N
    best_residual = residual_mat[np.arange(N), best_idx]  # N
    return best_residual < eps


def rot_verts(verts, rot):
    theta = np.deg2rad(rot)
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], float
    )
    org_shape = verts.shape
    return ((R @ verts.reshape(-1, 2).T).T).reshape(org_shape)


def rot_pano(pano, rot):
    pano_rot = np.zeros_like(pano)
    W = pano.shape[1]
    W_move = np.round(W * (rot % 360 / 360)).astype(np.int)
    pano_rot[:, : (W - W_move)] = pano[:, W_move:]
    pano_rot[:, (W - W_move) :] = pano[:, :W_move]
    return pano_rot


def nms_33(src, fill_value=0, eight_neighbor=True):
    assert len(src.shape) == 2
    h, w = src.shape
    if torch.is_tensor(src):
        dst = src.clone()
    elif isinstance(src, np.ndarray):
        dst = src.copy()
    dst[:, 0 : w - 1][src[:, 0 : w - 1] <= src[:, 1:w]] = fill_value  # l-r
    dst[:, 1:w][src[:, 1:w] <= src[:, 0 : w - 1]] = fill_value  # r-l
    dst[0 : h - 1, :][src[0 : h - 1, :] <= src[1:h, :]] = fill_value  # u-b
    dst[1:h, :][src[1:h, :] <= src[0 : h - 1, :]] = fill_value  # b-u
    if eight_neighbor:
        dst[0 : h - 1, 0 : w - 1][
            src[0 : h - 1, 0 : w - 1] <= src[1:h, 1:w]
        ] = fill_value  # lu-rb
        dst[1:h, 1:w][src[1:h, 1:w] <= src[0 : h - 1, 0 : w - 1]] = fill_value  # rb-lu
        dst[0 : h - 1, 1:w][
            src[0 : h - 1, 1:w] <= src[1:h, 0 : w - 1]
        ] = fill_value  # ru-lb
        dst[1:h, 0 : w - 1][
            src[1:h, 0 : w - 1] <= src[0 : h - 1, 1:w]
        ] = fill_value  # lb-ru
    return dst


# modified from https://github.com/fuenwang/Equirec2Perspec
def pano2persp(img, fov, yaw, pitch, roll, size, RADIUS=128):

    equ_h, equ_w = img.shape[:2]
    equ_cx = (equ_w - 1) / 2.0
    equ_cy = (equ_h - 1) / 2.0

    height, width = size
    wFOV = fov
    hFOV = float(height) / width * wFOV

    c_x = (width - 1) / 2.0
    c_y = (height - 1) / 2.0

    wangle = (180 - wFOV) / 2.0
    w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
    w_interval = w_len / (width - 1)

    hangle = (180 - hFOV) / 2.0
    h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
    h_interval = h_len / (height - 1)
    x_map = np.zeros([height, width], np.float32) + RADIUS
    y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
    z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
    D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = np.zeros([height, width, 3], np.float)
    xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
    xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
    xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(yaw - 180))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-pitch))
    [R3, _] = cv2.Rodrigues(np.dot(R2 @ R1, x_axis) * np.radians(-roll))

    xyz = xyz.reshape(height * width, 3).T
    xyz = ((R3 @ R2 @ R1) @ xyz).T
    lat = np.arcsin(xyz[:, 2] / RADIUS)
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])
    lon = ((lon / np.pi + 1) * equ_cx).reshape(height, width)
    lat = ((-lat / np.pi * 2 + 1) * equ_cy).reshape(height, width)

    persp = cv2.remap(
        img,
        lon.astype(np.float32),
        lat.astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_WRAP,
    )
    return persp  # , lon, lat, (lon/equ_cx*np.pi)[:width] #[0----pi-----2pi]
