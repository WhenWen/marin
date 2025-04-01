import matplotlib.pyplot as plt
import wandb
from pprint import pprint
import pandas as pd
from tqdm import tqdm
import argparse
import pickle
from scipy.interpolate import griddata
import numpy as np
import json
from scipy.optimize import curve_fit

plt.rcParams.update({
    "font.family": "Palatino Linotype"
})

parser = argparse.ArgumentParser()
parser.add_argument("--build_cache", action="store_true")
# parser.add_argument("--data1_name", type=str)
# parser.add_argument("--data2_name", type=str)
parser.add_argument("--model_size", type=str, default="150m")
args = parser.parse_args()

# data1_name, data2_name = args.data1_name, args.data2_name
data1_name, data2_name = "stack_dedup", "c4"

pretty_name_dict = {
    "stack_dedup": "Python",
    "stack_cpp": "C++",
    "c4": "C4",
}

wandb_key = {
    ("stack_dedup", "c4"): {
        "150m": "python-c4-0.005-varsched-cooldown-0.05-sweep",
    },
}[(data1_name, data2_name)][args.model_size]

def parse_run(run):
    run_dict = {}
    run_id = run.id

    run_json_config = json.loads(run.json_config)
    if "linear" in run_id:
        run_dict["schedule_type"] = "linear"
        run_dict["decay_ratio"] = run_json_config["optimizer"]["value"]["decay"]
    else:
        run_dict["schedule_type"] = "cosine"
        run_dict["decay_ratio"] = None

    run_id_entries = run_id.split("-")
    run_dict["fraction_data1_stage2"] = float(run_id_entries[run_id_entries.index("vs") + 1])

    stage2_stack_portion = run_json_config["data"]["value"]["train_weights"]["stack_dedup"]
    run_dict["stage2_duration"] = 0.005 * run_dict["fraction_data1_stage2"] / stage2_stack_portion

    history = run.history(keys=[f"eval/{data1_name}/loss", f"eval/{data2_name}/loss"])
    run_dict[f"final_{data1_name}_loss"] = history[f"eval/{data1_name}/loss"].iloc[-1]
    run_dict[f"final_{data2_name}_loss"] = history[f"eval/{data2_name}/loss"].iloc[-1]
    return run_dict

if args.build_cache:
    run_list = []
    runs = wandb.Api().runs("stanford-mercury/suhas-curriculum")
    for run in tqdm(runs):
        if wandb_key in run.tags and "stage2" in run.tags:
            run_dict = parse_run(run)
            run_list.append(run_dict)
    pickle.dump(run_list, open(f"cache/{wandb_key}_run_list.pkl", "wb"))
else:
    run_list = pickle.load(open(f"cache/{wandb_key}_run_list.pkl", "rb"))

decay_ratio_color_map = {
    0.25: 'orange',
    0.5: 'brown',
    0.75: 'teal',
    1.0: 'magenta',
}

plt.figure(figsize=(7, 5), dpi=600)
plt.grid(True, linestyle='--', alpha=0.4)
unique_fractions_data1_stage2 = list(set([run['fraction_data1_stage2'] for run in run_list]))
unique_fractions_data1_stage2.sort(key=lambda x: x)
fit_params = {}

for idx, fraction_data1_stage2 in enumerate(unique_fractions_data1_stage2):
    color = decay_ratio_color_map[fraction_data1_stage2]
    label = f"{fraction_data1_stage2:.2f} fraction of {pretty_name_dict[data1_name]} is allocated to stage 2"
    runs_with_decay = [run for run in run_list if run['fraction_data1_stage2'] == fraction_data1_stage2]
    runs_with_decay.sort(key=lambda x: x['stage2_duration'])
    
    x = np.array([1 - run['fraction_data1_stage2'] * 0.005 / run['stage2_duration'] for run in runs_with_decay])
    print(x)
    y = np.array([run[f"final_{data1_name}_loss"] for run in runs_with_decay])
    x_transformed = np.log(1 - x)

    # Fit a cubic polynomial
    z = np.polyfit(x_transformed, y, 3)
    p = np.poly1d(z)
    
    # Generate smooth points for plotting
    x_smooth = np.linspace(x.min(), x.max(), 100)
    x_smooth_transformed = np.log(1 - x_smooth)
    y_smooth = p(x_smooth_transformed)
    # Find minimum of the fitted curve
    min_idx = np.argmin(y_smooth)
    min_x = x_smooth[min_idx]
    min_y = y_smooth[min_idx]
    print(f"Minimum for {label}: x={min_x:.4f}, y={min_y:.4f}")
    
    # Plot star at minimum
    plt.plot(min_x, min_y, '*', markersize=10, color=color)
    plt.plot(x_smooth, y_smooth, '--', alpha=0.5, color=color)
    
    plt.scatter(x, y, label=label, marker='o', color=color)

plt.xlabel('Replay Ratio')
plt.ylabel(f'Final {data1_name} loss')
plt.xscale('function', functions=(lambda x: np.log(1-x), lambda x: 1 - np.exp(x)))
plt.title(f'Rare loss vs replay ratio\n\n{pretty_name_dict[data1_name]} (0.005) vs {pretty_name_dict[data2_name]} (0.995) with {args.model_size} parameters')
# plt.xticks([0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625], 
        #    [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.00625])
plt.xticks([0.99375, 0.9875, 0.975, 0.95, 0.9, 0.8, 0.6, 0.2],
           [0.99375, 0.9875, 0.975, 0.95, 0.9, 0.8, 0.6, 0.2])
if data1_name == "stack_dedup" and data2_name == "c4" and args.model_size == "150m":
    plt.ylim(ymax=3.6)
# plt.xlim(xmin=0.02)
# plt.xlim(xmax=0.85)
plt.plot([], [], '*', color='black', label='Minima from cubic fit', markersize=10)
plt.legend()
plt.tight_layout()
plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/brute_force_v2/{data1_name}_{data2_name}_{args.model_size}_rare_loss_curve_x_replay.png')
plt.close()
