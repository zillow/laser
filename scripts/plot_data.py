import numpy as np
import os
from prettytable import PrettyTable
import sys, re
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", type=str, required=True)
opt = parser.parse_args()


log_dir = opt.log_dir

table_cdf = PrettyTable()
table_cdf_ths = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
table_cdf_th_in = (1, 30)
table_cdf.field_names = [
    "result_name",
    "1m+30deg (%)",
    "1cm (%)",
    "5cm (%)",
    "10cm (%)",
    "20cm (%)",
    "50cm (%)",
    "1m (%)",
]

table_topk = PrettyTable()
table_topk_ks = [1, 2, 3, 5, 7, 9]
table_topk.field_names = [
    "result_name",
    "top1_<1m (%)",
    "top2_<1m (%)",
    "top3_<1m (%)",
    "top5_<1m (%)",
    "top7_<1m (%)",
    "top9_<1m (%)",
]

table_err = PrettyTable()
table_err.field_names = [
    "result_name",
    "median_terr (cm)",
    "median_rerr (deg)",
    "<1m median_terr (cm)",
    "<1m median_rerr (deg)",
]

table_timing = PrettyTable()  # timing table only plot those ('timing' in result_name)
table_timing.field_names = [
    "result_name",
    "sampling_fps",
    "sampling_time (s)",
    "matching_fps",
]


n_results = 0

result_list = [
    x
    for x in sorted(os.listdir(log_dir))
    if os.path.isdir(os.path.join(log_dir, x)) and x.startswith("results")
]
for result_name in result_list:
    result_dir = os.path.join(log_dir, result_name)
    with open(os.path.join(result_dir, "analytics_data.pkl"), "rb") as f:
        data = pickle.load(f)

    # cdf
    inlier_rate = (
        (data["terrs"] < table_cdf_th_in[0]).reshape(-1)
        & (data["rerrs"] < table_cdf_th_in[1]).reshape(-1)
    ).sum() / data["terrs"].size
    row = [result_name, f"{inlier_rate*100:.2f}"]
    for th in table_cdf_ths:
        recall = (data["terrs"] < th).sum() / data["terrs"].size
        row.append(f"{recall*100:.2f}")
    table_cdf.add_row(row)

    # topk
    row = [result_name]
    for k in table_topk_ks:
        recall = (data["topk_terrs"][k] < 1).sum() / data["topk_terrs"][k].size
        row.append(f"{recall*100:.2f}")
    table_topk.add_row(row)

    # median err
    med_terr = np.median(data["terrs"])
    med_rerr = np.median(data["rerrs"])
    mask_1m = data["terrs"] < 1.0
    med_terr_1m = np.median(data["terrs"][mask_1m])
    med_rerr_1m = np.median(data["rerrs"][mask_1m])
    table_err.add_row(
        [
            result_name,
            f"{med_terr*100:.2f}",
            f"{med_rerr:.2f}",
            f"{med_terr_1m*100:.2f}",
            f"{med_rerr_1m:.2f}",
        ]
    )

    # timing
    sampling_fps_avg = np.mean(data["sampling_fpss"])
    sampling_fps_std = np.std(data["sampling_fpss"])
    sampling_time_avg = np.mean(data["sampling_times"])
    sampling_time_std = np.std(data["sampling_times"])
    matching_fps_avg = np.mean(data["matching_fpss"])
    matching_fps_std = np.std(data["matching_fpss"])
    table_timing.add_row(
        [
            result_name,
            f"{sampling_fps_avg:.2f} mp {sampling_fps_std:.2f}",
            f"{sampling_time_avg:.2f} mp {sampling_time_std:.2f}",
            f"{matching_fps_avg:.2f} mp {matching_fps_std:.2f}",
        ]
    )

    n_results += 1


print(table_cdf)
print(table_err)
print(table_topk)
print(table_timing)

print(f"{n_results} results loaded.")
