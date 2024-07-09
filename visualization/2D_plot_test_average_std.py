import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#from results_baseline1 import files as files
#from results_adapt3 import files as new_model_files
from results_adapt3 import files as files
from results_adapt3_focal import files as new_model_files

# Function to load JSON data
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load all data into a dictionary
data = {}
for organ, paths in files.items():
    data[organ] = {
        "avg": load_json(paths["avg"]),
        "std": load_json(paths["std"])
    }

# Prepare data for plotting
patients = set()
for organ_data in data.values():
    patients.update(organ_data["avg"].keys())

patients = sorted(list(patients))

plot_data = {
    "patient": [],
    "organ": [],
    "average": [],
    "std_dev": [],
    "model": []
}

for organ, organ_data in data.items():
    for patient in patients:
        if patient in organ_data["avg"]:
            plot_data["patient"].append(patient)
            plot_data["organ"].append(organ)
            plot_data["average"].append(organ_data["avg"][patient]["Dice"])
            plot_data["std_dev"].append(organ_data["std"][patient]["Dice"])
            plot_data["model"].append("baseline")
        else:
            plot_data["patient"].append(patient)
            plot_data["organ"].append(organ)
            plot_data["average"].append(None)
            plot_data["std_dev"].append(None)
            plot_data["model"].append("baseline")

# Load all data into a dictionary for the new model
new_model_data = {}
for organ, paths in new_model_files.items():
    new_model_data[organ] = {
        "avg": load_json(paths["avg"]),
        "std": load_json(paths["std"])
    }

# Combine old and new data
for organ, organ_data in new_model_data.items():
    for patient in patients:
        if patient in organ_data["avg"]:
            plot_data["patient"].append(patient)
            plot_data["organ"].append(organ)
            plot_data["average"].append(organ_data["avg"][patient]["Dice"])
            plot_data["std_dev"].append(organ_data["std"][patient]["Dice"])
            plot_data["model"].append("new_model")
        else:
            plot_data["patient"].append(patient)
            plot_data["organ"].append(organ)
            plot_data["average"].append(None)
            plot_data["std_dev"].append(None)
            plot_data["model"].append("new_model")

df = pd.DataFrame(plot_data)

# Calculate the mean for each model per organ
organ_means = {}
for organ in df["organ"].unique():
    organ_means[organ] = {
        "baseline_mean": df[(df["model"] == "baseline") & (df["organ"] == organ)]["average"].mean(),
        "new_model_mean": df[(df["model"] == "new_model") & (df["organ"] == organ)]["average"].mean()
    }

# Calculate the standard deviation for each model per organ
organ_stds = {}
for organ in df["organ"].unique():
    organ_stds[organ] = {
        "baseline_std": df[(df["model"] == "baseline") & (df["organ"] == organ)]["average"].std(),
        "new_model_std": df[(df["model"] == "new_model") & (df["organ"] == organ)]["average"].std()
    }

# Define color palette
color_palette = sns.color_palette("husl", 2)  # Get a color-friendly palette
baseline_color = color_palette[0]
new_model_color = color_palette[1]

# Plotting separate bar plots for each organ with updated colors, separate figures, shadowed regions, and horizontal lines spanning the entire width
for organ in df["organ"].unique():
    plt.figure(figsize=(14, 8))
    organ_data = df[df["organ"] == organ]
    
    baseline_data = organ_data[organ_data["model"] == "baseline"]
    new_model_data = organ_data[organ_data["model"] == "new_model"]
    
    width = 0.35  # Width of the bars
    x = range(len(baseline_data))
    
    plt.bar([p - width/2 for p in x], baseline_data["average"], width, yerr=baseline_data["std_dev"], label="Baseline", capsize=5, color=baseline_color)
    plt.bar([p + width/2 for p in x], new_model_data["average"], width, yerr=new_model_data["std_dev"], label="New Model", capsize=5, color=new_model_color, alpha=0.7)
    
    baseline_mean = organ_means[organ]["baseline_mean"]
    new_model_mean = organ_means[organ]["new_model_mean"]
    baseline_std = organ_stds[organ]["baseline_std"]
    new_model_std = organ_stds[organ]["new_model_std"]
    
    plt.axhline(y=baseline_mean, color=baseline_color, linestyle='--', label='Baseline Mean')
    plt.axhline(y=new_model_mean, color=new_model_color, linestyle='--', label='New Model Mean')
    
    plt.fill_between(range(-1, len(baseline_data)+1), baseline_mean - baseline_std, baseline_mean + baseline_std, color=baseline_color, alpha=0.1)
    plt.fill_between(range(-1, len(baseline_data)+1), new_model_mean - new_model_std, new_model_mean + new_model_std, color=new_model_color, alpha=0.1)
    
    plt.xlabel('Patient ID')
    plt.ylabel('Dice Score')
    plt.xticks(ticks=x, labels=baseline_data["patient"], rotation=45)
    plt.xlim(-1, len(baseline_data))
    plt.title(f'{organ.capitalize()} Segmentation - Average and Standard Deviation per Patient (Baseline vs New Model)')
    plt.legend()
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(f'{organ}_dice_comparison.png')
