import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import json
from .zind_utils import *
from .s3d_utils import *

# TODO: select floor based on n_images to balance the visits per floor in training time


class S3dDataset(Dataset):

    training_set = (0, 3000)
    testing_set = (3250, 3500)

    def __init__(
        self,
        dataset_dir,
        s3d_eval_furnishing="full",
        image_size=256,
        crop_fov=None,
        view_type="pview",
        find_interesting_fov=False,
        is_training=False,
        line_sampling_interval=0.1,
        n_sample_points=None,
        return_empty_when_invalid=False,
        return_all_panos=False,
    ):

        self.dataset_dir = dataset_dir
        self.s3d_eval_furnishing = s3d_eval_furnishing
        self.image_size = image_size
        self.crop_fov = crop_fov
        self.view_type = view_type
        assert self.view_type in ["pview", "eview"]
        self.find_interesting_fov = find_interesting_fov
        self.is_training = is_training
        self.line_sampling_interval = line_sampling_interval
        self.n_sample_points = n_sample_points
        self.return_empty_when_invalid = return_empty_when_invalid
        self.return_all_panos = return_all_panos

        if is_training:
            self.N = self.training_set[1] - self.training_set[0]
        else:
            self.N = self.testing_set[1] - self.testing_set[0]
            np.random.seed(123456789)

        print(
            f"{self.N} samples loaded from {type(self)} dataset. (is_training={is_training})"
        )

    def __len__(self):
        return self.N

    def fetch_another(self):
        if self.return_empty_when_invalid:
            return {}
        else:
            return self.__getitem__(np.random.randint(self.N))

    def __getitem__(self, idx):
        idx = idx + (self.training_set[0] if self.is_training else self.testing_set[0])

        instance_path = os.path.join(self.dataset_dir, f"scene_{idx:05d}")
        json_path = os.path.join(instance_path, "annotation_3d.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        global_rot = np.random.rand() * 360 if self.is_training else 0

        n_rooms, room_lines, door_lines, window_lines = read_s3d_floorplan(data)
        room_lines = rot_verts(room_lines, global_rot)
        door_lines = rot_verts(door_lines, global_rot)
        window_lines = rot_verts(window_lines, global_rot)

        if self.n_sample_points is not None:
            perimeter = np.linalg.norm(
                room_lines[:, 0, :] - room_lines[:, 1, :], axis=-1
            ).sum()
            bases, bases_normal = sample_points_from_lines(
                room_lines, 0.9 * perimeter / self.n_sample_points
            )
            rnd_sample_idx = np.random.permutation(bases.shape[0])[
                : self.n_sample_points
            ]
            bases = bases[rnd_sample_idx]
            bases_normal = bases_normal[rnd_sample_idx]
        else:
            bases, bases_normal = sample_points_from_lines(
                room_lines, self.line_sampling_interval
            )

        bases_door_mask = points_on_lines(bases, door_lines)
        bases_window_mask = points_on_lines(bases, window_lines)
        bases_feat = np.concatenate(
            [
                (bases - bases.mean(axis=0, keepdims=True))
                / 5.0,  # 2 ,normalize with 5.0
                np.zeros_like(bases[:, 0:1]),  # 1
                bases_normal,  # 2
                bases_door_mask.reshape(-1, 1),  # 1
                bases_window_mask.reshape(-1, 1),
            ],  # 1
            axis=1,
        )  # N,D

        pano_node_list = os.listdir(os.path.join(instance_path, "2D_rendering"))
        if not self.return_all_panos:
            if self.is_training:
                rnd_idx = np.random.randint(len(pano_node_list))
                pano_node_list = pano_node_list[rnd_idx : rnd_idx + 1]
            else:
                pano_node_list = pano_node_list[0:1]

        query_images = []
        gt_locs = []
        gt_rots = []
        gt_fovs = []
        for pano_node in pano_node_list:
            pano_loc = (
                np.loadtxt(
                    os.path.join(
                        instance_path,
                        "2D_rendering",
                        pano_node,
                        "panorama",
                        "camera_xyz.txt",
                    )
                )[:2]
                / 1000.0
            )
            pano_loc = rot_verts(pano_loc, global_rot)
            pano_rot = global_rot
            if self.is_training:
                render_type = ["empty", "simple", "full"][np.random.randint(3)]
            else:
                render_type = self.s3d_eval_furnishing
            pano_image_path = os.path.join(
                instance_path,
                "2D_rendering",
                pano_node,
                "panorama",
                render_type,
                "rgb_rawlight.png",
            )
            pano_image = cv2.imread(pano_image_path, cv2.IMREAD_COLOR)
            if pano_image is None:  # pnglib read error with 1753/972405/simple/rawlight
                if self.return_all_panos:
                    continue
                else:
                    return self.fetch_another()
            pano_image = cv2.flip(pano_image, 1)

            # random rotation augmentation
            # for eval time, we set fixed seed, besure only use 1 worker loader
            rnd_rot = np.random.rand() * 360
            pano_image = rot_pano(pano_image, rnd_rot)
            pano_rot = pano_rot + rnd_rot

            if self.crop_fov is not None:
                fov = (
                    np.random.rand() * (self.crop_fov[1] - self.crop_fov[0])
                    + self.crop_fov[0]
                    if isinstance(self.crop_fov, tuple)
                    else self.crop_fov
                )
                if self.find_interesting_fov:
                    yaw = find_interesting_fov(pano_image)
                    if self.view_type == "eview":
                        yaw = yaw - fov / 2
                else:
                    yaw = np.random.rand() * 360 if self.is_training else 0
                if self.view_type == "pview":
                    query_image = pano2persp(
                        pano_image, fov, yaw, 0, 0, (self.image_size, self.image_size)
                    )
                elif self.view_type == "eview":
                    query_image = cv2.resize(
                        pano_image, (self.image_size * 2, self.image_size)
                    )
                    query_image = rot_pano(query_image, yaw)[
                        :, : int(self.image_size * 2 * (fov / 360))
                    ]
                else:
                    raise "Unknown view_type"
                gt_fovs.append(np.array([fov]))
                gt_rots.append(np.array([pano_rot + yaw]))
            else:
                query_image = cv2.resize(
                    pano_image, (self.image_size * 2, self.image_size)
                )
                gt_fovs.append(np.array([360]))
                gt_rots.append(np.array([pano_rot]))

            query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB) / 255.0
            query_image -= (0.485, 0.456, 0.406)
            query_image /= (0.229, 0.224, 0.225)
            query_images.append(np.transpose(query_image, (2, 0, 1)))
            gt_locs.append(pano_loc)

        ret_fp = {
            "bases": bases.astype(np.float32),  # B,2
            "bases_normal": bases_normal.astype(np.float32),  # B,2
            "bases_feat": bases_feat.astype(np.float32),
        }  # B,D
        if self.return_all_panos:
            ret_img = {
                "gt_loc": np.stack(gt_locs, axis=0).astype(np.float32),  #
                "gt_rot": (np.stack(gt_rots, axis=0) % 360 / 180 * np.pi).astype(
                    np.float32
                ),  #
                "query_image": np.stack(query_images, axis=0).astype(np.float32),
                "gt_fov": np.stack(gt_fovs, axis=0).astype(np.float32),
            }
        else:
            ret_img = {
                "gt_loc": gt_locs[0].astype(np.float32),  # 2
                "gt_rot": (gt_rots[0] % 360 / 180 * np.pi).astype(np.float32),  # 1
                "query_image": query_images[0].astype(np.float32),
                "gt_fov": gt_fovs[0].astype(np.float32),
            }

        return {**ret_fp, **ret_img}

    def get_n_rooms(self, idx):
        idx = idx + (self.training_set[0] if self.is_training else self.testing_set[0])

        instance_path = os.path.join(self.dataset_dir, f"scene_{idx:05d}")
        json_path = os.path.join(instance_path, "annotation_3d.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        n_rooms, room_lines, door_lines, window_lines = read_s3d_floorplan(data)

        return n_rooms

    def get_area(self, idx):
        idx = idx + (self.training_set[0] if self.is_training else self.testing_set[0])

        instance_path = os.path.join(self.dataset_dir, f"scene_{idx:05d}")
        json_path = os.path.join(instance_path, "annotation_3d.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        n_rooms, room_lines, door_lines, window_lines = read_s3d_floorplan(data)
        ptp = np.ptp(room_lines.reshape(-1, 2), axis=0)
        return ptp[0] * ptp[1]
