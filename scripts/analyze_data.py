import numpy as np
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, required=True)
opt = parser.parse_args()

log_dir = opt.log_dir


def nms33(src):
    h, w = src.shape[0], src.shape[1]
    dst = src.copy()
    dst[:, 0 : w - 1][src[:, 0 : w - 1] <= src[:, 1:w]] = 0  # l-r
    dst[:, 1:w][src[:, 1:w] <= src[:, 0 : w - 1]] = 0  # r-l
    dst[0 : h - 1, :][src[0 : h - 1, :] <= src[1:h, :]] = 0  # u-b
    dst[1:h, :][src[1:h, :] <= src[0 : h - 1, :]] = 0  # b-u
    dst[0 : h - 1, 0 : w - 1][src[0 : h - 1, 0 : w - 1] <= src[1:h, 1:w]] = 0  # lu-rb
    dst[1:h, 1:w][src[1:h, 1:w] <= src[0 : h - 1, 0 : w - 1]] = 0  # rb-lu
    dst[0 : h - 1, 1:w][src[0 : h - 1, 1:w] <= src[1:h, 0 : w - 1]] = 0  # ru-lb
    dst[1:h, 0 : w - 1][src[1:h, 0 : w - 1] <= src[0 : h - 1, 1:w]] = 0  # lb-ru
    return dst


def read_all_raw(result_dir):
    raw_dir = os.path.join(result_dir, "raws")
    raw_fns = os.listdir(raw_dir)
    n_data = len(raw_fns)
    # n_data = 10
    data_list = []
    for i in range(n_data):
        with open(os.path.join(raw_dir, raw_fns[i]), "rb") as f:
            data = pickle.load(f)
        data["idx"] = int(raw_fns[i][4:8])  # raw_xxxx.pkl
        data_list.append(data)
        print(f"\rReading raw data {i} / {n_data} ...", end="")
    print()
    return data_list


def estimate_topk(data_list, maxk, min_score=0):
    n_data = len(data_list)
    topk_terrs = {k: [] for k in range(1, maxk + 1)}

    for i1 in range(n_data):
        print(f"\rEstimating top-k {i1} / {n_data} ...", end="")
        data = data_list[i1]
        n_images = data["n_images"]
        for i2 in range(n_images):
            score_map = data["score_maps"][i2]
            # score_map = cv2.GaussianBlur(score_map, (0,0), 1)
            score_map = nms33(score_map)
            scores_est = score_map[score_map > min_score].reshape(-1)
            locs_est = data["samples_loc"][np.where(score_map.reshape(-1) > min_score)]
            sort_idx = np.argsort(scores_est)[::-1]
            scores_est = scores_est[sort_idx]
            locs_est = locs_est[sort_idx]
            terrs = np.linalg.norm(locs_est - data["loc_gts"][i2], axis=-1)

            for k in range(1, maxk + 1):
                topk_terrs[k].append(terrs[:k].min())
    print()

    for k in range(1, maxk + 1):
        topk_terrs[k] = np.stack(topk_terrs[k])
        # print(k, (topk_terrs[k]<1).sum() / topk_terrs[k].size)

    return {"topk_terrs": topk_terrs}


def aggregate_data(data_list):
    n_data = len(data_list)
    ret = {
        "terrs": [],
        "rerrs": [],
        "matching_fpss": [],
        "sampling_fpss": [],
        "sampling_times": [],
    }
    for i1 in range(n_data):
        print(f"\rAggregating data {i1} / {n_data} ...", end="")
        data = data_list[i1]
        ret["terrs"].append(data["terrs"])
        ret["rerrs"].append(data["rerrs"])
        ret["matching_fpss"].append(np.array(data["matching_fps"]).reshape(1))
        ret["sampling_fpss"].append(np.array(data["sampling_fps"]).reshape(1))
        ret["sampling_times"].append(np.array(data["sampling_time"]).reshape(1))
    print()

    for k in ret:
        ret[k] = np.concatenate(ret[k])

    return ret


if __name__ == "__main__":

    result_list = [
        x
        for x in sorted(os.listdir(log_dir))
        if os.path.isdir(os.path.join(log_dir, x)) and x.startswith("results")
    ]
    n_results = len(result_list)
    for i in range(n_results):
        result_dir = os.path.join(log_dir, result_list[i])
        print(f"Processing {result_dir} ... ({i+1}/{n_results})")

        data_list = read_all_raw(result_dir)
        data_topk = estimate_topk(data_list, maxk=10)
        data_other = aggregate_data(data_list)
        data_world = {**data_topk, **data_other}

        with open(os.path.join(result_dir, "analytics_data.pkl"), "wb") as f:
            pickle.dump(data_world, f)

    exit(0)
