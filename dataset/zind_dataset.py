import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import json
from .zind_utils import *

# TODO: select floor based on n_images to balance the visits per floor in training time


def eval_valid_floors(dataset_dir, set):
    valid_floors = []
    for idx in set:
        instance_path = os.path.join(dataset_dir, f"{idx:03d}")
        json_path = os.path.join(instance_path, "zind_data.json")
        if not os.path.isfile(json_path):
            continue
        with open(json_path, "r") as f:
            data = json.load(f)
        # if 'redraw' not in data:
        # continue
        scaled_floors = [
            k
            for k in data["scale_meters_per_coordinate"]
            if data["scale_meters_per_coordinate"][k] is not None
        ]
        if len(scaled_floors) == 0:
            continue
        for floor_name in scaled_floors:
            valid_floors.append((idx, floor_name))
    return valid_floors


class ZindDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
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

        with open("./dataset/zind_partition.json", "r") as f:
            partition = json.load(f)
        if is_training:
            self.valid_floors = eval_valid_floors(
                dataset_dir, np.array(partition["train"], int)
            )
            self.N = len(self.valid_floors)
        else:
            self.valid_floors = eval_valid_floors(
                dataset_dir, np.array(partition["test"], int)
            )
            self.N = len(self.valid_floors)
            np.random.seed(
                123456789
            )  # the default randomness in zind pano rotation is somehow non-uniform..

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
        zind_idx, floor_name = self.valid_floors[idx]
        instance_path = os.path.join(self.dataset_dir, f"{zind_idx:03d}")
        json_path = os.path.join(instance_path, "zind_data.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        # floor_node = data['merger'][floor_name]
        floor_scale = data["scale_meters_per_coordinate"][floor_name]

        global_rot = (
            np.random.rand() * 360
            if self.is_training
            else -data["floorplan_to_redraw_transformation"][floor_name]["rotation"]
        )

        room_lines = []
        door_lines = []
        window_lines = []
        for room_name in data["merger"][floor_name]:
            # pick complete geo from first pano of the complete_room
            room_node = data["merger"][floor_name][room_name]
            proom_node = room_node[list(room_node.keys())[0]]
            pano_node = proom_node[list(proom_node.keys())[0]]
            fp_trans = pano_node["floor_plan_transformation"]

            verts = np.array(pano_node["layout_complete"]["vertices"])
            verts = np.concatenate([verts, verts[0:1]], axis=0)
            verts = rot_verts(verts, fp_trans["rotation"]) * fp_trans[
                "scale"
            ] + np.array(fp_trans["translation"])
            verts *= floor_scale
            lines = poly_verts_to_lines(verts)
            if not is_polygon_clockwise(lines):
                lines = poly_verts_to_lines(np.flip(verts, axis=0))
            room_lines.append(lines)

            for sem_type, sem_list in zip(
                ["doors", "windows"], (door_lines, window_lines)
            ):
                verts = np.array(pano_node["layout_complete"][sem_type])
                if verts.size == 0:
                    continue
                verts = rot_verts(verts, fp_trans["rotation"]) * fp_trans[
                    "scale"
                ] + np.array(fp_trans["translation"])
                verts *= floor_scale
                sem_list.append(
                    np.array(
                        [verts[i] for i in range(verts.shape[0]) if i % 3 != 2]
                    ).reshape(-1, 2, 2)
                )

        room_lines = rot_verts(np.concatenate(room_lines, axis=0), global_rot)
        door_lines = (
            np.zeros((0, 2, 2), float)
            if len(door_lines) == 0
            else rot_verts(np.concatenate(door_lines, axis=0), global_rot)
        )
        window_lines = (
            np.zeros((0, 2, 2), float)
            if len(window_lines) == 0
            else rot_verts(np.concatenate(window_lines, axis=0), global_rot)
        )

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

        pano_node_list = []
        for room_name in data["merger"][floor_name]:
            for partial_room_name in data["merger"][floor_name][room_name]:
                for pano_name in data["merger"][floor_name][room_name][
                    partial_room_name
                ]:
                    pano_node_list.append(
                        data["merger"][floor_name][room_name][partial_room_name][
                            pano_name
                        ]
                    )
        if len(pano_node_list) == 0:
            return self.fetch_another()

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
                np.array(pano_node["floor_plan_transformation"]["translation"])
                * floor_scale
            )
            pano_loc = rot_verts(pano_loc, global_rot)
            pano_rot = pano_node["floor_plan_transformation"]["rotation"] + global_rot
            pano_image_path = os.path.join(instance_path, pano_node["image_path"])
            if not os.path.isfile(pano_image_path):
                if self.return_all_panos:
                    continue
                else:
                    return self.fetch_another()
            pano_image = cv2.imread(pano_image_path, cv2.IMREAD_COLOR)

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
        zind_idx, floor_name = self.valid_floors[idx]
        instance_path = os.path.join(self.dataset_dir, f"{zind_idx:03d}")
        json_path = os.path.join(instance_path, "zind_data.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        n_rooms = 0
        for room_name in data["merger"][floor_name]:
            is_closet = False
            for partial_room_name in data["merger"][floor_name][room_name]:
                for pano_name in data["merger"][floor_name][room_name][
                    partial_room_name
                ]:
                    pano_node = data["merger"][floor_name][room_name][
                        partial_room_name
                    ][pano_name]
                    if pano_node["is_primary"] and pano_node["label"] in ["closet"]:
                        is_closet = True
                        break
            if not is_closet:
                n_rooms += 1

        return n_rooms

    def get_area(self, idx):
        zind_idx, floor_name = self.valid_floors[idx]
        instance_path = os.path.join(self.dataset_dir, f"{zind_idx:03d}")
        json_path = os.path.join(instance_path, "zind_data.json")
        with open(json_path, "r") as f:
            data = json.load(f)
        floor_scale = data["scale_meters_per_coordinate"][floor_name]

        all_verts = []
        for room_name in data["merger"][floor_name]:
            # pick complete geo from first pano of the complete_room
            room_node = data["merger"][floor_name][room_name]
            proom_node = room_node[list(room_node.keys())[0]]
            pano_node = proom_node[list(proom_node.keys())[0]]
            fp_trans = pano_node["floor_plan_transformation"]

            verts = np.array(pano_node["layout_complete"]["vertices"])
            verts = np.concatenate([verts, verts[0:1]], axis=0)
            verts = rot_verts(verts, fp_trans["rotation"]) * fp_trans[
                "scale"
            ] + np.array(fp_trans["translation"])
            verts *= floor_scale
            all_verts.append(verts)
        all_verts = np.concatenate(all_verts, axis=0)
        ptp = np.ptp(all_verts, axis=0)
        return ptp[0] * ptp[1]
