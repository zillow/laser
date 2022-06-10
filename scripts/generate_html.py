import os, argparse
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, required=True)
opt = parser.parse_args()
result_dir = opt.result_dir


def html_img_link(img_path):
    return rf'<img src="{img_path}" width=400px>'


result_fns = sorted(os.listdir(os.path.join(result_dir, "results")))
score_map_fns = sorted(os.listdir(os.path.join(result_dir, "score_maps")))
rot_map_fns = sorted(os.listdir(os.path.join(result_dir, "rot_maps")))
query_image_fns = sorted(os.listdir(os.path.join(result_dir, "query_images")))
terr_fns = sorted(os.listdir(os.path.join(result_dir, "terrs")))
rerr_fns = sorted(os.listdir(os.path.join(result_dir, "rerrs")))
terrs = []
rerrs = []
for terr_fn in terr_fns:
    terrs.append(np.loadtxt(os.path.join(result_dir, "terrs", terr_fn)).reshape(-1))
for rerr_fn in rerr_fns:
    rerrs.append(np.loadtxt(os.path.join(result_dir, "rerrs", rerr_fn)).reshape(-1))
terrs = np.concatenate(terrs, axis=0)
rerrs = np.concatenate(rerrs, axis=0)

assert (
    len(result_fns)
    == len(score_map_fns)
    == len(rot_map_fns)
    == len(query_image_fns)
    == len(rerrs)
    == len(terrs)
)
n_results = len(terrs)


print(f"{n_results} results loaded.")

# for rng in [(0,999), (0,0.2), (0.2,0.5), (0.5,1.0), (1.0, 999)]: # use this if you want multiple htmls of results of different accuracies
for rng in [(0, 999)]:
    n_rows = 0
    table = PrettyTable()
    table.field_names = [
        "idx",
        "trans_err(m)",
        "rot_err(degree)",
        "result",
        "score_map",
        "rot_map",
        "query_image",
    ]
    for i in range(n_results):
        if terrs[i] < rng[0] or terrs[i] > rng[1]:  # change condition here
            continue
        n_rows += 1
        result_path = os.path.join(".", "results", result_fns[i])
        score_map_path = os.path.join(".", "score_maps", score_map_fns[i])
        rot_map_path = os.path.join(".", "rot_maps", rot_map_fns[i])
        query_image_path = os.path.join(".", "query_images", query_image_fns[i])
        table.add_row(
            [
                i,
                f"{terrs[i]:.4f}",
                f"{rerrs[i]:.4f}",
                html_img_link(result_path),
                html_img_link(score_map_path),
                html_img_link(rot_map_path),
                html_img_link(query_image_path),
            ]
        )

    html_str = table.get_html_string()
    html_str = html_str.replace(r"&lt;", r"<")
    html_str = html_str.replace(r"&gt;", r">")
    html_str = html_str.replace(r"&quot;", r'"')

    with open(
        os.path.join(result_dir, f"results--{rng[0]:.1f}-{rng[1]:.1f}.html"), "w"
    ) as f:
        f.write(html_str)

# plot cdf
pdf, bins = np.histogram(
    np.clip(terrs, a_min=None, a_max=1.05), bins=21, range=(0, 1.05), density=True
)
pdf = pdf / sum(pdf)
cdf = np.cumsum(pdf)[:20]
rng = bins[1:-1]
print(cdf)
# np.savetxt(os.path.join(result_dir, 'cdf.txt'), np.stack([rng, cdf], axis=1), fmt='%.4f')
