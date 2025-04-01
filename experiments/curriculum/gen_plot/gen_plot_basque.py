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
# parser.add_argument("--model_size", type=str, default="150m")
# parser.add_argument("--rare_freq", type=float, default=0.1)
args = parser.parse_args()

# data1_name, data2_name = args.data1_name, args.data2_name
data1_name, data2_name = "latxa", "spj"

pretty_name_dict = {
    "latxa": "Basque",
    "spj": "SlimPajama",
}

acc_dict = {
    "latxa": "xcopa_eu",
}

# wandb_key = ""

def parse_run(run):
    run_dict = {}
    run_id = run.id
    run_dict["run_id"] = run_id

    if "latxa" not in run_id or "debug" in run_id or run_id.startswith("spj"):
        return None
    
    if "r4" in run_id:
        run_dict['repetition'] = 4
    else:
        run_dict['repetition'] = 1

    run_json_config = json.loads(run.json_config)
    if "linear" in run_id:
        run_dict["schedule_type"] = "linear"
        run_dict["decay_ratio"] = run_json_config["optimizer"]["value"]["decay"]
    else:
        run_dict["schedule_type"] = "cosine"
        run_dict["decay_ratio"] = None

    run_dict["num_steps"] = run_json_config["trainer"]["value"]["num_train_steps"]
    run_dict["rare_freq"] = run_json_config["data"]["value"]["train_weights"][-1][1][data1_name]
    run_dict["basque_steps"] = round(run_dict["num_steps"] * run_dict["rare_freq"] / run_dict["repetition"])

    run_history_loss_keys = [f"eval/{data1_name}/loss", f"eval/{data2_name}/loss"]
    run_history_acc_keys = [f"lm_eval/{acc_dict[data1_name]}/acc"]

    if run_dict["rare_freq"] == 1.0:
        run_history_loss_keys.remove(f"eval/{data2_name}/loss")

    history_loss = run.history(keys=run_history_loss_keys)
    history_acc = run.history(keys=run_history_acc_keys)

    run_dict["loss_history"] = history_loss
    run_dict["acc_history"] = history_acc

    if f"eval/{data1_name}/loss" in history_loss.columns:
        run_dict[f"final_{data1_name}_loss"] = history_loss[f"eval/{data1_name}/loss"].iloc[-1]
        # run_dict[f"final_{data2_name}_loss"] = history[f"eval/{data2_name}/loss"].iloc[-1]
        run_dict["finished"] = len(history_loss[f"eval/{data1_name}/loss"]) >= 20
    else:
        print(f"\033[91mNo {data1_name} loss in {run_id}\033[0m")
        return None

    if f"lm_eval/{acc_dict[data1_name]}/acc" in history_acc.columns:
        run_dict[f"final_{data1_name}_acc"] = history_acc[f"lm_eval/{acc_dict[data1_name]}/acc"].iloc[-1]
    else:
        print(f"\033[91mNo {data1_name} acc in {run_id}\033[0m")
        return None

    return run_dict

if args.build_cache:
    run_list = []
    runs = wandb.Api().runs("stanford-mercury/suhas-cpt")
    for run in tqdm(runs):
        run_dict = parse_run(run)
        if run_dict is not None:
            print(run_dict["run_id"])
            run_list.append(run_dict)
    pickle.dump(run_list, open(f"cache/basque_run_list.pkl", "wb"))
else:
    run_list = pickle.load(open(f"cache/basque_run_list.pkl", "rb"))

print("Total runs: ", len(run_list))
print("Finished runs: ", len([run for run in run_list if run['finished']]))


unique_basque_steps = list(set([(run['basque_steps'], run['repetition']) for run in run_list]))
unique_basque_steps.sort(key=lambda x: x[0])
fit_params = {}

rare_freq_to_color = {
    0.1: "red",
    0.25: "blue",
    0.75: "green",
    0.95: "purple",
    1.0: "black",
}

for idx, (basque_steps, repetition) in enumerate(unique_basque_steps):
    basque_tokens = int(basque_steps * 500000)
    plot_label = f"{int(basque_tokens / 1e6)}M Basque tokens ({repetition}x repetition)"
    runs_with_basque_steps = [run for run in run_list if run['basque_steps'] == basque_steps and run['repetition'] == repetition and run['finished']]
    runs_with_basque_steps.sort(key=lambda x: x['num_steps'])
    
    plt.figure(figsize=(7, 5), dpi=600)
    plt.grid(True, linestyle='--', alpha=0.4)
    # Plot loss history for each rare frequency
    for run in runs_with_basque_steps:
        history = run['loss_history']
        rare_freq = run['rare_freq']
        steps = history['_step'] / run['num_steps']
        plt.plot(steps, history[f"eval/{data1_name}/loss"], 
                label=f"Basque ratio {rare_freq}", 
                color=rare_freq_to_color[rare_freq])

        if run['rare_freq'] == 1.0:
            final_loss = run[f'final_{data1_name}_loss']
            plt.axhline(y=final_loss, color='black', linestyle=':', label='Final loss (Basque ratio 1.0)')

    plt.xlabel('Training duration')
    plt.ylabel(f'{pretty_name_dict[data1_name]} Loss')
    plt.title(f'Loss vs Training Duration\n{plot_label}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/basque/{data1_name}_{data2_name}_loss_history_{basque_steps}_{repetition}.png',
                bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(7, 5), dpi=600)
    plt.grid(True, linestyle='--', alpha=0.4)
    # Plot accuracy history for each rare frequency
    for run in runs_with_basque_steps:
        history = run['acc_history']
        rare_freq = run['rare_freq']
        steps = history['_step'] / run['num_steps']
        plt.plot(steps, history[f"lm_eval/{acc_dict[data1_name]}/acc"], 
                label=f"Basque ratio {rare_freq}", 
                color=rare_freq_to_color[rare_freq])

        if run['rare_freq'] == 1.0:
            final_acc = run[f'final_{data1_name}_acc']
            plt.axhline(y=final_acc, color='black', linestyle=':', label='Final accuracy (Basque ratio 1.0)')

    plt.xlabel('Training duration')
    plt.ylabel(f'{pretty_name_dict[data1_name]} Accuracy')
    plt.title(f'Accuracy vs Training Duration\n{plot_label}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/basque/{data1_name}_{data2_name}_accuracy_history_{basque_steps}_{repetition}.png',
                bbox_inches='tight')
    plt.close()

    # Add tradeoff plot
    plt.figure(figsize=(7, 5), dpi=600)
    plt.grid(True, linestyle='--', alpha=0.4)

    # Extract ratios and accuracies for this basque_steps
    ratios = [run['rare_freq'] for run in runs_with_basque_steps]
    accuracies = [run[f'final_{data1_name}_acc'] for run in runs_with_basque_steps]
    
    # Sort by ratio for line plot
    points = sorted(zip(ratios, accuracies))
    ratios, accuracies = zip(*points)
    
    # Fit line in semi-log space (log x, linear y)
    log_ratios = np.log(ratios)
    fit = np.polyfit(log_ratios, accuracies, 1)
    
    # Calculate R-squared and R
    y_fit_points = fit[0] * log_ratios + fit[1]
    residuals = np.array(accuracies) - y_fit_points
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((np.array(accuracies) - np.mean(accuracies)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    r = np.corrcoef(log_ratios, accuracies)[0,1]  # Pearson correlation coefficient
    
    # Generate points for the line of best fit
    x_fit = np.logspace(np.log10(min(ratios)), np.log10(max(ratios)), 100)
    y_fit = fit[0] * np.log(x_fit) + fit[1]
    
    plt.plot(ratios, accuracies, 'o-', color='blue')
    plt.plot(x_fit, y_fit, '--', color='gray', alpha=0.7, 
             label=f'Fit (R = {r:.3f}, R² = {r_squared:.3f})')
    
    plt.xscale('log')
    plt.xlabel('Basque Ratio')
    plt.xticks(ticks=[0.1, 0.25, 0.5, 0.75, 0.95, 1.0], labels=['0.1', '0.25', '0.5', '0.75', '0.95', '1.0'])
    plt.ylabel(f'{pretty_name_dict[data1_name]} Final Accuracy')
    plt.title(f'Accuracy vs Basque Ratio Tradeoff\n{plot_label}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/basque/{data1_name}_{data2_name}_tradeoff_{basque_steps}_{repetition}.png',
                bbox_inches='tight')
    plt.close()

# After the main for loop
# Filter for runs with repetition=1
runs_rep1 = [run for run in run_list if run['repetition'] == 1 and run['finished']]

# Get unique rare_freq values
rare_freqs = sorted(list(set(run['rare_freq'] for run in runs_rep1)))

# First get the fit for ratio 1.0 baseline
baseline_runs = [run for run in runs_rep1 if run['rare_freq'] == 1.0]
baseline_x = [run['basque_steps'] * 500000 for run in baseline_runs]
baseline_y = [run[f'final_{data1_name}_loss'] for run in baseline_runs]
baseline_log_x = np.log(baseline_x)
baseline_fit = np.polyfit(baseline_log_x, baseline_y, 1)

# Function to find intersection x value given a y value
def find_intersection_x(y_value, fit_params):
    # y = m*log(x) + b
    # solving for x: x = exp((y-b)/m)
    return np.exp((y_value - fit_params[1]) / fit_params[0])

print("\nEffective Data Multipliers:")
# Create a list of all multiplier results
multiplier_results = []
for rare_freq in rare_freqs:
    if rare_freq == 1.0:
        continue
        
    rare_freq_runs = [run for run in runs_rep1 if run['rare_freq'] == rare_freq]
    for run in rare_freq_runs:
        actual_tokens = run['basque_steps'] * 500000
        loss_value = run[f'final_{data1_name}_loss']
        
        # Find where this loss intersects with baseline curve
        effective_tokens = find_intersection_x(loss_value, baseline_fit)
        
        multiplier = effective_tokens / actual_tokens
        multiplier_results.append({
            'tokens': actual_tokens,
            'ratio': rare_freq,
            'multiplier': multiplier
        })

# Sort by tokens first, then by ratio
multiplier_results.sort(key=lambda x: (x['tokens'], x['ratio']))

# Print sorted results
for result in multiplier_results:
    print(f"Tokens {result['tokens']/1e6:.1f}M, Ratio {result['ratio']}: {result['multiplier']:.2f}x")

# Helper functions
def fit_and_plot_curve(x_values, y_values, color, label, ax):
    """Fit and plot both data points and best fit line"""
    points = sorted(zip(x_values, y_values))
    x_values, y_values = zip(*points)
    
    # Plot actual data points with lines
    ax.plot(x_values, y_values, 'o-', label=label, color=color)
    
    # Fit line in log-linear space
    log_x = np.log(x_values)
    fit = np.polyfit(log_x, y_values, 1)
    
    # Calculate R-squared and R
    y_fit_points = fit[0] * log_x + fit[1]
    residuals = np.array(y_values) - y_fit_points
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((np.array(y_values) - np.mean(y_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    r = np.corrcoef(log_x, y_values)[0,1]  # Pearson correlation coefficient
    
    # Generate and plot best fit line
    x_fit = np.logspace(np.log10(min(x_values)), np.log10(max(x_values)), 100)
    y_fit = fit[0] * np.log(x_fit) + fit[1]
    ax.plot(x_fit, y_fit, ':', color=color, alpha=0.7, 
            label=f'{label} fit (R = {r:.3f}, R² = {r_squared:.3f})')
    
    return fit

def calculate_multipliers(runs, baseline_fit, metric_name, higher_is_better=False):
    """Calculate effective data multipliers for a given metric"""
    multiplier_results = []
    for run in runs:
        actual_tokens = run['basque_steps'] * 500000
        metric_value = run[f'final_{data1_name}_{metric_name}']
        
        # Find where this value intersects with baseline curve
        # For accuracy (higher_is_better=True), we want the baseline tokens needed to reach this accuracy
        # For loss (higher_is_better=False), we want the baseline tokens needed to reach this loss
        effective_tokens = find_intersection_x(metric_value, baseline_fit)
        
        # For accuracy, if mixed training achieves higher value than baseline with fewer tokens, that's good
        # For loss, if mixed training achieves lower value than baseline with fewer tokens, that's good
        multiplier = effective_tokens / actual_tokens
        
        multiplier_results.append({
            'tokens': actual_tokens,
            'ratio': run['rare_freq'],
            'multiplier': multiplier,
            'metric_value': metric_value
        })
    
    return multiplier_results

# Create scaling law plots for both loss and accuracy
metrics = [('loss', False), ('acc', True)]  # (metric_name, higher_is_better)
fig, axes = plt.subplots(1, 2, figsize=(20, 6), dpi=600)

# Calculate and print multipliers for both metrics first
print("\nEffective Data Multipliers:")
non_baseline_runs = [run for run in runs_rep1 if run['rare_freq'] != 1.0]

for metric, higher_is_better in metrics:
    # Get baseline fit
    baseline_runs = [run for run in runs_rep1 if run['rare_freq'] == 1.0]
    baseline_x = [run['basque_steps'] * 500000 for run in baseline_runs]
    baseline_y = [run[f'final_{data1_name}_{metric}'] for run in baseline_runs]
    baseline_fit = np.polyfit(np.log(baseline_x), baseline_y, 1)
    
    # Calculate multipliers
    multiplier_results = calculate_multipliers(non_baseline_runs, baseline_fit, metric, higher_is_better)
    multiplier_results.sort(key=lambda x: (x['tokens'], x['ratio']))
    
    print(f"\nEffective Data Multipliers for {metric}:")
    for result in multiplier_results:
        print(f"Tokens {result['tokens']/1e6:.1f}M, Ratio {result['ratio']}: "
              f"multiplier={result['multiplier']:.2f}x, {metric}={result['metric_value']:.4f}")

# Now do the plotting
for idx, (metric, higher_is_better) in enumerate(metrics):
    ax = axes[idx]
    ax.grid(True, linestyle='--', alpha=0.4)
    
    # Get baseline fit first (we're calculating this again for plotting)
    baseline_runs = [run for run in runs_rep1 if run['rare_freq'] == 1.0]
    baseline_x = [run['basque_steps'] * 500000 for run in baseline_runs]
    baseline_y = [run[f'final_{data1_name}_{metric}'] for run in baseline_runs]
    baseline_fit = np.polyfit(np.log(baseline_x), baseline_y, 1)
    
    # Plot curves for each ratio
    for rare_freq in rare_freqs:
        rare_freq_runs = [run for run in runs_rep1 if run['rare_freq'] == rare_freq]
        x_values = [run['basque_steps'] * 500000 for run in rare_freq_runs]
        y_values = [run[f'final_{data1_name}_{metric}'] for run in rare_freq_runs]
        
        fit_and_plot_curve(x_values, y_values, 
                          rare_freq_to_color[rare_freq],
                          f"Basque ratio {rare_freq}", ax)
    
    ax.set_xscale('log')
    ax.set_xlabel('Total Basque Tokens')
    ax.set_ylabel(f'{pretty_name_dict[data1_name]} {metric.capitalize()}')
    ax.set_title(f'Scaling Law: {metric.capitalize()} vs Total Basque Tokens')
    ax.legend()

plt.tight_layout()
plt.savefig(f'/Users/Suhas/Desktop/SUHAS/Repos/marin/experiments/curriculum/plots/basque/{data1_name}_{data2_name}_scaling_laws.png',
            bbox_inches='tight')
plt.close()




    


