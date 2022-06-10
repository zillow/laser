import os, time, argparse, json

record_log = False
save_model = True
log_dir = "./logs"

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--its", type=int, default=0, help="Training iterations.")
parser.add_argument(
    "--fov", type=float, default=None, help="Training fov, left blank for panorama"
)
parser.add_argument(
    "--fov2", type=str, default=None, help="Train with mixed fovs. e.g. [--fov2 45,135]"
)
parser.add_argument("--dataset", type=str, default="zind", help="zind/s3d")
parser.add_argument("--dataset_path", type=str, default=None)

parser.add_argument("--D", type=int, default=128, help="Descriptor dimension")
parser.add_argument("--G", type=int, default=32, help="Incident-angle codebook size")
parser.add_argument("--H", type=int, default=32, help="Ray length codebook size")
parser.add_argument(
    "--dist_max", type=float, default=10, help="Maximum ray length in meters"
)
parser.add_argument("--V", type=int, default=16, help="Circular feature resolution")
parser.add_argument(
    "--Vr", type=int, default=64, help="Circular feature resolution at rendering time"
)
parser.add_argument(
    "--disable_semantics",
    action="store_true",
    help="Ignore semantic labels in the dataset",
)
parser.add_argument(
    "--disable_pointnet",
    action="store_true",
    help="Remove PointNet and use a shared static codebook",
)

opt = parser.parse_args()
assert opt.dataset in ["zind", "s3d"]

if opt.fov2 is not None:
    opt.fov = tuple([float(x) for x in opt.fov2.split(",")])
assert (
    (opt.fov is None)
    or (opt.fov2 is None and 180 > opt.fov > 0)
    or (
        len(opt.fov) == 2
        and 180 > opt.fov[0] > 0
        and 180 > opt.fov[1] > 0
        and opt.fov[1] > opt.fov[0]
    )
)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
if opt.log_dir is not None:
    record_log = True
    log_dir = opt.log_dir


def tic():
    globals()["tictoc"] = time.perf_counter()


def toc():
    return time.perf_counter() - globals()["tictoc"]


import torch
import torchvision
from dataset.zind_dataset import ZindDataset
from dataset.s3d_dataset import S3dDataset
from model.fplocnet import FpLocNet, quick_fplocnet_call
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tensorboardX


def log_instance(name, data, log_writer, global_step):
    if isinstance(data, np.ndarray):  # histogram data
        log_writer.add_histogram(name, data, global_step)
    elif isinstance(data, float):  # scalar data
        log_writer.add_scalar(name, data, global_step)
    elif torch.is_tensor(data):  # image data
        grid = torchvision.utils.make_grid(data, normalize=True)
        log_writer.add_image(name, grid, global_step)


def log_summary(suffix, details, log_writer, global_step):
    losses_to_record = details.keys()
    for log_name in losses_to_record:
        if isinstance(details[log_name], list):
            for i in range(len(details[log_name])):
                log_instance(
                    f"{log_name}/{suffix}-{i}",
                    details[log_name][i],
                    log_writer,
                    global_step,
                )
        else:
            log_instance(
                f"{log_name}/{suffix}", details[log_name], log_writer, global_step
            )


if __name__ == "__main__":

    cfg = {
        "Q": 100,
        "Q_refine": 20,
        "D": opt.D,
        "G": opt.G,
        "H": opt.H,
        "dist_max": opt.dist_max,
        "Vr": opt.Vr,
        "V": opt.V,
        "disable_semantics": opt.disable_semantics,
        "disable_pointnet": opt.disable_pointnet,
        "fov": 360 if opt.fov is None else opt.fov,
        "view_type": "eview" if opt.fov is None else "pview",
    }
    print(opt)
    print(cfg)

    # For eview, only support train with 360 pano, support crop FoV in eval time.
    # For pview, need train seperate model for different FoV.

    # dataset_root = 'D:/ZInD'
    dataset_root = opt.dataset_path
    if opt.dataset == "zind":
        _dataset = ZindDataset
        if opt.dataset_path is None:
            dataset_root = "/mnt/data/zhixiangm/ZInD"
    elif opt.dataset == "s3d":
        _dataset = S3dDataset
        if opt.dataset_path is None:
            dataset_root = "/mnt/data/zhixiangm/Structured3D"
    else:
        raise "Unknown dataset"

    model = FpLocNet(cfg).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    global_step = 0
    if record_log:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, "cfg.json"), "w") as f:
            json.dump(cfg, f)
        log_writer = tensorboardX.SummaryWriter(log_dir)
        if save_model:
            if os.path.isfile(os.path.join(log_dir, "ckpt")):
                ckpt = torch.load(os.path.join(log_dir, "ckpt"))
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                global_step = ckpt["global_step"]
                print(f"Model with step={global_step} loaded.")

    dataloader_train = DataLoader(
        _dataset(
            dataset_root,
            is_training=True,
            n_sample_points=2048,
            crop_fov=cfg["fov"],
            view_type=cfg["view_type"],
            find_interesting_fov=False,
        ),
        batch_size=8,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )

    dataloader_val = DataLoader(
        _dataset(
            dataset_root,
            is_training=False,
            n_sample_points=2048,
            crop_fov=cfg["fov"],
            view_type=cfg["view_type"],
            find_interesting_fov=False,
        ),
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )

    dataloader_vals = {"0": dataloader_val}
    dataloader_vals_it = {"0": iter(dataloader_val)}

    tic()

    while opt.its == 0 or global_step < opt.its:

        for data_train in dataloader_train:
            if opt.its > 0 and global_step >= opt.its:
                break

            train_details = quick_fplocnet_call(
                model, data_train, cfg=cfg, is_training=True, optimizer=optimizer
            )

            global_step += 1

            if global_step % 200 == 0:
                if record_log:
                    log_summary(f"train", train_details, log_writer, global_step)

                for key in dataloader_vals.keys():
                    try:
                        data_val = next(dataloader_vals_it[key])
                    except StopIteration:
                        dataloader_vals_it[key] = iter(dataloader_vals[key])
                        data_val = next(dataloader_vals_it[key])

                    with torch.no_grad():
                        val_details = quick_fplocnet_call(
                            model, data_val, cfg=cfg, is_training=False
                        )

                    if record_log:
                        log_summary(f"val-{key}", val_details, log_writer, global_step)

                print(
                    "step=%d,  time=%.1fmin,  loss=%.6f"
                    % (global_step, toc() / 60, train_details["loss_total"])
                )

            if record_log and global_step % 20000 == 0 and save_model:
                try:
                    torch.save(
                        {
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        os.path.join(log_dir, "ckpt"),
                    )
                    print("Model ckpt saved.")
                except:
                    print("Cannot save model.")

    if record_log:
        log_writer.close()
