import argparse
import os
import json
from functools import partial

parser = argparse.ArgumentParser()
parser.add_argument(
    "--save",
    action="store_true",
    help="If specified, the script will save the results under model folder. Otherwise, results are shown on fly.",
)
parser.add_argument("--gpu", type=str, default="0", help="Device ID of the GPU to use.")
parser.add_argument(
    "--sample_grid", type=float, default=0.1, help="Sampling grid resolution in meters."
)
parser.add_argument(
    "--sample_nrots", type=int, default=16, help="Number of rotations to sample."
)
parser.add_argument(
    "--sample_batchsize",
    type=int,
    default=256,
    help="Increase batch size to get higher speed.",
)
parser.add_argument(
    "--max_refine_its",
    type=int,
    default=3,
    help="Maximum # of iterations for the refinement branch",
)

parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Location of inference dataset on disk.",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="zind/s3df/s3ds/s3de. ZInD or S3D with full/simple/empty furnishing-level.",
    choices=["zind", "s3df", "s3ds", "s3de"],
)
parser.add_argument(
    "--mode",
    type=str,
    default="refine",
    help="match/refine. Use `match` mode to skip refinement.",
    choices=["match", "refine"],
)
parser.add_argument(
    "--fov",
    type=float,
    default=None,
    help="Overwrite testing fov. (default is same as training). If model trained using mixed-fov, need specify a single fov when eval",
)
parser.add_argument(
    "--interesting",
    type=int,
    default=0,
    help="0/1. Use `1` to indicate saliency-aware FoV sampling.",
)
parser.add_argument(
    "--log_dir",
    type=str,
    required=True,
    help="Directory where model is saved and logs written.",
)
parser.add_argument(
    "--eval_all",
    type=int,
    default=1,
    help="0/1. Use `0` to evaluate 1 sample for each floor map.",
)

parser.add_argument(
    "--suffix", type=str, default=None, help="Add suffix to result folder name"
)


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
opt.eval_all = bool(opt.eval_all)
opt.interesting = bool(opt.interesting)
print(opt)

import os
import pickle
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset.zind_dataset import ZindDataset
from dataset.s3d_dataset import S3dDataset
from model.fplocnet import FpLocNet
from eval_utils import *


def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        os.makedirs(path)


if __name__ == "__main__":
    with open(os.path.join(opt.log_dir, "cfg.json"), "r") as f:
        cfg = json.load(f)
    # overwrite some cfg if need
    if opt.fov is not None:
        cfg["fov"] = opt.fov
    assert not (
        isinstance(cfg["fov"], list) or isinstance(cfg["fov"], tuple)
    )  # if model trained using mixed-fov, need specify a single fov when eval
    cfg["find_interesting_fov"] = opt.interesting
    print(cfg)

    if opt.dataset == "zind":
        _dataset = ZindDataset
    elif opt.dataset == "s3df":
        _dataset = partial(S3dDataset, s3d_eval_furnishing="full")
    elif opt.dataset == "s3ds":
        _dataset = partial(S3dDataset, s3d_eval_furnishing="simple")
    elif opt.dataset == "s3de":
        _dataset = partial(S3dDataset, s3d_eval_furnishing="empty")
    else:
        raise "Unknown dataset"

    dataloader = DataLoader(
        _dataset(
            opt.dataset_path,
            is_training=False,
            n_sample_points=None,
            line_sampling_interval=0.1,
            return_empty_when_invalid=True,
            return_all_panos=opt.eval_all,
            crop_fov=cfg["fov"],
            view_type=cfg["view_type"],
            find_interesting_fov=cfg["find_interesting_fov"],
        ),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    model = FpLocNet(cfg).cuda()

    ckpt = torch.load(os.path.join(opt.log_dir, "ckpt"))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    global_step = ckpt["global_step"]
    print(f"Model with step={global_step} loaded.")

    if opt.save:
        if opt.suffix is None:
            folder_name = "results-{0}-fov{1:03d}-{2}-{3}".format(
                opt.mode,
                int(cfg["fov"]),
                "sal" if opt.interesting else "rnd",
                opt.dataset,
            )
        else:
            folder_name = "results-{0}-fov{1:03d}-{2}-{3}-{4}".format(
                opt.mode,
                int(cfg["fov"]),
                "sal" if opt.interesting else "rnd",
                opt.dataset,
                opt.suffix,
            )
        save_dir = os.path.join(opt.log_dir, folder_name)
        mkdir_if_not_exist(os.path.join(save_dir, "score_maps"))
        mkdir_if_not_exist(os.path.join(save_dir, "rot_maps"))
        mkdir_if_not_exist(os.path.join(save_dir, "results"))
        mkdir_if_not_exist(os.path.join(save_dir, "query_images"))
        mkdir_if_not_exist(os.path.join(save_dir, "terrs"))
        mkdir_if_not_exist(os.path.join(save_dir, "rerrs"))
        mkdir_if_not_exist(os.path.join(save_dir, "raws"))

    idx = 0
    for data in dataloader:
        if len(data.keys()) == 0:
            idx += 1
            continue
        for k in data.keys():
            if torch.is_tensor(data[k]) and not data[k].is_cuda:
                data[k] = data[k].cuda()
        if cfg["disable_semantics"]:
            data["bases_feat"][..., -2:] = 0

        if opt.eval_all:
            data["query_image"] = data["query_image"].squeeze(0)
            data["gt_loc"] = data["gt_loc"].squeeze(0)
            data["gt_fov"] = data["gt_fov"].squeeze(0)
            data["gt_rot"] = data["gt_rot"].squeeze(0)

        sample_ret = sample_floorplan(
            data,
            model,
            cfg,
            sample_grid=opt.sample_grid,
            batch_size=opt.sample_batchsize,
        )
        match_ret = match_images(
            sample_ret,
            data,
            model,
            cfg,
            mode=opt.mode,
            sample_nrots=opt.sample_nrots,
            max_refine_its=opt.max_refine_its,
        )

        fmt = {
            "idx": idx,
            "sampling_fps": sample_ret["sampling_fps"],
            "sampling_time": sample_ret["sampling_time"],
            "matching_fps": match_ret["matching_fps"],
            "matching_time": match_ret["matching_time"],
            "median_terr": np.median(match_ret["terrs"]),
            "median_rerr": np.median(match_ret["rerrs"]),
        }
        print(
            "{idx}, {sampling_fps:.0f} sampling_fps, {sampling_time:.2f} sampling_time, {matching_fps:.2f} matching_fps, {matching_time:.2f} matching_time, {median_terr:.4f} median_terr, {median_rerr:.4f} median_rerr".format(
                **fmt
            )
        )

        for i in range(match_ret["n_images"]):
            score_map = match_ret["score_maps"][i]
            rot_map = match_ret["rot_maps"][i]
            loc_gt = match_ret["loc_gts"][i]
            loc_est = match_ret["loc_ests"][i]

            score_map_viz = (
                cv2.resize(score_map, (0, 0), fx=2, fy=2) ** 2 * 255
            ).astype(np.uint8)
            rot_map_viz = (
                cv2.resize(rot_map / (2 * np.pi), (0, 0), fx=2, fy=2) * 255
            ).astype(np.uint8)
            result_viz = render_result(
                data["bases"][0].cpu().numpy(),
                data["bases_feat"][0].cpu().numpy(),
                loc_gt,
                loc_est,
            )
            query_image_viz = data["query_image"][i].permute(1, 2, 0).cpu().numpy() * (
                0.229,
                0.224,
                0.225,
            ) + (0.485, 0.456, 0.406)
            query_image_viz = cv2.cvtColor(
                (query_image_viz * 255).astype(np.uint8), cv2.COLOR_BGR2RGB
            )

            if opt.save:
                if i == 0:
                    np.savetxt(
                        os.path.join(save_dir, "terrs", f"terrs_{idx:04d}.txt"),
                        match_ret["terrs"],
                    )
                    np.savetxt(
                        os.path.join(save_dir, "rerrs", f"rerrs_{idx:04d}.txt"),
                        match_ret["rerrs"],
                    )
                    with open(
                        os.path.join(save_dir, "raws", f"raw_{idx:04d}.pkl"), "wb"
                    ) as f:
                        merged_raw = {**sample_ret, **match_ret, **data}
                        merged_raw["idx"] = idx
                        merged_raw[
                            "samples_feat"
                        ] = None  # too large to store, render in runtime
                        merged_raw["fp_feat"] = None
                        merged_raw["img_feats"] = None
                        for k in merged_raw:
                            if torch.is_tensor(merged_raw[k]):
                                merged_raw[k] = merged_raw[k].cpu().numpy()
                        pickle.dump(merged_raw, f)

                cv2.imwrite(
                    os.path.join(
                        save_dir, "score_maps", f"score_map_{idx:04d}_{i:03d}.png"
                    ),
                    score_map_viz,
                )
                cv2.imwrite(
                    os.path.join(
                        save_dir, "rot_maps", f"rot_map_{idx:04d}_{i:03d}.png"
                    ),
                    rot_map_viz,
                )
                cv2.imwrite(
                    os.path.join(save_dir, "results", f"result_{idx:04d}_{i:03d}.png"),
                    result_viz,
                )
                cv2.imwrite(
                    os.path.join(
                        save_dir, "query_images", f"query_image_{idx:04d}_{i:03d}.png"
                    ),
                    query_image_viz,
                )
            else:
                cv2.imshow("score_map_viz", score_map_viz)
                cv2.imshow("rot_map_viz", rot_map_viz)
                cv2.imshow("result_viz", result_viz)
                cv2.imshow("query_image_viz", query_image_viz)
                if cv2.waitKey(1 if opt.save else 0) == 113:
                    exit(0)
                pass
        idx += 1
