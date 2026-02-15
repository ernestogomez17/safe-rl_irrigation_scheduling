import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict, OrderedDict
from scipy.signal import savgol_filter
from tensorboard.backend.event_processing import event_accumulator
from tqdm.auto import tqdm
from IPython.display import display

# %% --- PLOTTING SETUP FOR PUBLICATION ---
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"], # Or "Times New Roman"
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    # High-quality backend for saving figures
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': r'\usepackage[utf8x]{inputenc}\usepackage[T1]{fontenc}',
})

CONFIG = {
    "base_directory": "/scratch/egomez/irrigation_project_output/models",
    "smoothing_window": 21,
    "smoothing_polyorder": 5,
    "output_dir": "./plots",

    # Regex captures algo, chance, and days as separate named groups.
    "model_name_pattern": re.compile(
        r"exp\d+_(?P<algo>DDPG|SAC|DDPGLagrangian|SACLagrangian)_chance_"
        r"(?P<chance>[\d\.]+)_"
        r"(?P<days>days\d+)"
        r"(?:_s(?P<seed>\d+))?" # Seed is optional in the name itself
    ),

    # Tag mapping remains the same.
    "tag_mapping": {
        'Averageworker/EpRet': 'Return (Training)',
        'Averageworker/EpNumViolations': 'Number of Violations (Training)',
        
        'Averageeval/TestEpRet': 'Return (Evaluation)',
        'Averageeval/TestEpNumViolations': 'Number of Violations (Evaluation)',

        'Stdworker/EpRet': 'Return Std (Training)',
        'Stdworker/EpNumViolations': 'Number of Violations Std (Training)',

        'Stdeval/TestEpRet': 'Return Std (Evaluation)',
        'Stdeval/TestEpNumViolations': 'Number of Violations Std (Evaluation)',
    },

    # Style map uses different colors.
    "plot_styles": {
        'SAC':  {'color': '#1f77b4', 'linestyle': '-', 'label': 'SAC'},
        'DDPG': {'color': '#ff7f0e', 'linestyle': '-', 'label': 'DDPG'},

        'SACLagrangian': {
            '1.0':  {'color': '#2ca02c', 'linestyle': '-', 'label': r'SAC Lagrangian ($\alpha=1.0$)'},
            '0.95': {'color': '#d62728', 'linestyle': '-', 'label': r'SAC Lagrangian ($\alpha=0.95$)'},
            '0.85': {'color': '#9467bd', 'linestyle': '-', 'label': r'SAC Lagrangian ($\alpha=0.85$)'},
            '0.75': {'color': '#8c564b', 'linestyle': '-', 'label': r'SAC Lagrangian ($\alpha=0.75$)'},
        },
        'DDPGLagrangian': {
            '1.0':  {'color': '#e377c2', 'linestyle': '-', 'label': r'DDPG Lagrangian ($\alpha=1.0$)'},
            '0.95': {'color': '#7f7f7f', 'linestyle': '-', 'label': r'DDPG Lagrangian ($\alpha=0.95$)'},
            '0.85': {'color': '#bcbd22', 'linestyle': '-', 'label': r'DDPG Lagrangian ($\alpha=0.85$)'},
            '0.75': {'color': '#17becf', 'linestyle': '-', 'label': r'DDPG Lagrangian ($\alpha=0.75$)'},
        }
    }
}

# Create the output directory if it doesn't exist
os.makedirs(CONFIG['output_dir'], exist_ok=True)

def parse_model_info(dir_name: str) -> dict | None:
    """
    Parses a directory name using the detailed regex from CONFIG.
    """
    match = CONFIG["model_name_pattern"].search(dir_name)
    if not match:
        return None
    
    info = match.groupdict()
    
    # Create a unique name for this specific configuration
    experiment_name = f"{info['algo']}_chance{info['chance']}_{info['days']}"
    info['name'] = experiment_name
    
    return info


def process_scalar_data(tb_path: str) -> pd.DataFrame | None:
    """
    Loads scalar data from a single TensorBoard directory.
    """
    try:
        ea = event_accumulator.EventAccumulator(tb_path, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        if not ea.Tags().get('scalars'): return None

        ref_tag = 'Averageeval/TestEpRet'
        if ref_tag not in ea.Tags()['scalars']: return None
        
        data = {'TrainingSteps': [e.step for e in ea.Scalars(ref_tag)]}
        for tag, col_name in CONFIG["tag_mapping"].items():
            if tag in ea.Tags()['scalars']:
                values = [e.value for e in ea.Scalars(tag)]
                if len(values) == len(data['TrainingSteps']):
                    data[col_name] = values
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"✗ Error processing {tb_path}: {e}")
        return None

def load_all_runs(base_dir: str) -> dict:
    """
    Scans base directory, loads TensorBoard data, and groups it by experiment.
    
    Returns a dictionary where each entry contains the raw dataframes (dfs)
    and the parsed metadata (info) for that experiment.
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    # Use a defaultdict to simplify initialization
    grouped_data = defaultdict(lambda: {"dfs": [], "info": {}})
    
    tb_paths = [os.path.join(root, "tb") for root, dirs, _ in os.walk(base_dir) if "tb" in dirs]

    print(f"Found {len(tb_paths)} potential TensorBoard directories. Processing...")

    for tb_path in tqdm(tb_paths, desc="Loading runs"):
        model_dir_with_seed = os.path.basename(os.path.dirname(tb_path))
        parent_dir = os.path.basename(os.path.dirname(os.path.dirname(tb_path)))
        
        model_info = parse_model_info(parent_dir)
        if not model_info:
            model_info = parse_model_info(model_dir_with_seed)

        if not model_info:
            continue
            
        df = process_scalar_data(tb_path)
        if df is not None and not df.empty:
            exp_name = model_info['name']
            grouped_data[exp_name]["dfs"].append(df)
            grouped_data[exp_name]["info"] = model_info
            
    return dict(grouped_data)


# %% Load and process data from all TensorBoard logs
all_model_data = load_all_runs(CONFIG["base_directory"])

# Print a summary of the loaded data
print("\n--- Data Loading Summary ---")
if not all_model_data:
    print("✗ No data was loaded. Check your CONFIG settings and directory structure.")
else:
    for name, data in sorted(all_model_data.items()):
        print(f"- Experiment '{name}': Found {len(data['dfs'])} runs.")


def plot_single_metric(ax: plt.Axes, mean_metric_name: str, std_metric_name: str, day_data: dict, config: dict,
                       scale_factor: float = 1.0):
    """
    Helper function to plot data with improved legend ordering.
    Updated to group first by alpha value, then by algorithm.
    """
    # Create a function to manually sort the legend entries
    def get_legend_order(algo, chance=None):
        # Order: First regular algorithms (DDPG, SAC)
        if 'Lagrangian' not in algo:
            # Regular algorithms - DDPG first (0), then SAC (1)
            return 0 if algo == 'DDPG' else 1
        
        # For Lagrangian variants, first group by alpha value, then by algorithm
        # Start from position 2 (after regular algorithms)
        if chance == '0.75':
            base = 2
        elif chance == '0.85':
            base = 4
        elif chance == '0.95':
            base = 6
        elif chance == '1.0':
            base = 8
        else:
            base = 10  # Fallback
        
        # For each alpha group, DDPG first, then SAC
        if 'DDPG' in algo:
            return base
        else:  # SAC
            return base + 1
    
    # First sort the items by our custom order for plotting
    sorted_items = []
    
    # Extract info and sort by our custom ordering
    for exp_name, exp_data in day_data.items():
        info = exp_data['info']
        algo = info['algo']
        chance = info.get('chance', None)
        order = get_legend_order(algo, chance)
        sorted_items.append((order, (exp_name, exp_data)))
    
    # Sort by our custom order
    sorted_items.sort()
    
    # Now plot in the sorted order
    max_steps = 0
    for _, (exp_name, exp_data) in sorted_items:
        info, dfs = exp_data['info'], exp_data['dfs']
        if not dfs: continue

        all_runs_df = pd.concat(dfs).sort_values(by='TrainingSteps').reset_index(drop=True)
        if not all_runs_df.empty:
            max_steps = max(max_steps, all_runs_df['TrainingSteps'].max())

        # Check if the pre-calculated std data is available
        if std_metric_name in all_runs_df.columns:
            agg_df = all_runs_df.groupby('TrainingSteps')[[mean_metric_name, std_metric_name]].mean().reset_index()
            std_values = agg_df[std_metric_name]
            mean_values = agg_df[mean_metric_name]
        else:
            # Fallback if no pre-calculated std is found
            print(f"Warning: Std metric '{std_metric_name}' not found. Calculating std across seeds.")
            agg_df = all_runs_df.groupby('TrainingSteps')[mean_metric_name].agg(['mean', 'std']).reset_index()
            agg_df.columns = ['TrainingSteps', 'mean', 'std']
            std_values = agg_df['std'].fillna(0)
            mean_values = agg_df['mean']

        # Only smooth the mean values, not the standard deviation
        mean_smooth = savgol_filter(mean_values, config['smoothing_window'], config['smoothing_polyorder'])
        # Ensure std values are non-negative but DON'T smooth them
        std_values = np.maximum(std_values, 0)

        style_group = config['plot_styles'].get(info['algo'], {})
        style = style_group.get(info['chance'], {}) if 'Lagrangian' in info['algo'] else style_group
        color, linestyle, label = style.get('color', 'gray'), style.get('linestyle', '-'), style.get('label', exp_name)

        ax.plot(agg_df['TrainingSteps'], mean_smooth, label=label, color=color, linestyle=linestyle, linewidth=1.5)
        ax.fill_between(agg_df['TrainingSteps'], mean_smooth - std_values, mean_smooth + std_values, alpha=0.15, color=color)

    ax.set_ylabel(mean_metric_name)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: f'{x / scale_factor:.1f}'))
    return max_steps

# This function now displays each figure in the notebook before closing it.
def plot_publication_figures(data: dict, config: dict):
    """
    Generates a multi-page PDF with one page for Evaluation and one for Training data.
    """
    data_by_day = defaultdict(dict)
    for exp_name, exp_data in data.items():
        data_by_day[exp_data['info']['days']][exp_name] = exp_data

    if not data_by_day:
        print("No data to plot.")
        return

    save_path = os.path.join(config['output_dir'], 'performance_grid.pdf')
    with PdfPages(save_path) as pdf:
        print(f"\n--- Generating multi-page PDF: {save_path} ---")

        for metric_type in ["Evaluation", "Training"]:
            print(f"  -> Plotting page for {metric_type} data...")
            day_keys = sorted(data_by_day.keys())
            n_rows, n_cols = len(day_keys), 2

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows), squeeze=False)
            fig.suptitle(fr'\textbf{{Performance Metrics ({metric_type})}}', fontsize=16, y=0.96)

            for i, day in enumerate(day_keys):
                day_data = data_by_day[day]
                day_number = day.replace('days', '')
                ax_return, ax_violations = axes[i, 0], axes[i, 1]

                # Determine scale
                temp_max_steps = 0
                for exp_data in day_data.values():
                    for df in exp_data['dfs']:
                        if not df.empty:
                            temp_max_steps = max(temp_max_steps, df['TrainingSteps'].max())
                
                if temp_max_steps > 1.5e6: scale_factor, scale_label = 1e6, r'($\times 10^6$)'
                elif temp_max_steps > 1.5e3: scale_factor, scale_label = 1e3, r'($\times 10^3$)'
                else: scale_factor, scale_label = 1.0, ''

                # --- THIS IS THE FIX ---
                # Explicitly define all metric names to avoid errors.
                return_mean_metric = f'Return ({metric_type})'
                return_std_metric = f'Return Std ({metric_type})'
                violations_mean_metric = f'Number of Violations ({metric_type})'
                violations_std_metric = f'Number of Violations Std ({metric_type})'
                
                # Pass both mean and std names to the plotting function.
                plot_single_metric(ax_return, return_mean_metric, return_std_metric, day_data, config, scale_factor=scale_factor)
                plot_single_metric(ax_violations, violations_mean_metric, violations_std_metric, day_data, config, scale_factor=scale_factor)

                # Set labels and titles
                ax_return.set_ylabel(f'$d={day_number}$', fontsize=12)
                ax_violations.set_ylabel('')
                if i == 0:
                    ax_return.set_title(r'\textbf{Return}', fontsize=14)
                    ax_violations.set_title(r'\textbf{Number of Violations}', fontsize=14)

                row_xlabel = f'Training Steps {scale_label}'
                ax_return.set_xlabel(row_xlabel, fontsize=14)
                ax_violations.set_xlabel(row_xlabel, fontsize=14)

            # The legend order is now correct for BOTH plots because it's handled properly in plot_single_metric.
            handles, labels = axes[0, 0].get_legend_handles_labels()
            n_variants = len(handles) // 2 if len(handles) > 0 else 5
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
                    ncol=n_variants, frameon=False, title=r'\textbf{Model}', 
                    fontsize=10, title_fontsize=14)
            
            fig.tight_layout(rect=[0, 0.08, 1, 0.98], h_pad=2.0, w_pad=1.0)
            
            pdf.savefig(fig, bbox_inches='tight')
            display(fig)
            plt.close(fig)

    print(f"✅ Multi-page figure saved to: {save_path}")

# And finally, call the new function in your execution cell:
plot_publication_figures(all_model_data, CONFIG)

def plot_single_metric_chance95(ax: plt.Axes, mean_metric_name: str, std_metric_name: str, day_data: dict, config: dict,
                       scale_factor: float = 1.0):
    """
    A new version of the plotting function that ONLY plots regular models
    and Lagrangian models with chance=0.95.
    """

    filtered_day_data = {}
    for exp_name, exp_data in day_data.items():
        info = exp_data['info']
        algo = info['algo']
        chance = info.get('chance', None)
        
        # Define the conditions for which models to keep
        is_regular = 'Lagrangian' not in algo
        is_chance_95_lagrangian = 'Lagrangian' in algo and chance == '0.95'
        
        # If the model meets either condition, add it to our new dictionary
        if is_regular or is_chance_95_lagrangian:
            filtered_day_data[exp_name] = exp_data

    # Create a function to manually sort the legend entries
    def get_legend_order(algo, chance=None):
        # Order: First regular algorithms (DDPG, SAC)
        if 'Lagrangian' not in algo:
            # Regular algorithms - DDPG first (0), then SAC (1)
            return 0 if algo == 'DDPG' else 1
        
        # For Lagrangian variants, first group by alpha value, then by algorithm
        # Start from position 2 (after regular algorithms)
        if chance == '0.95':
            base = 2
        else:
            base = 4  # Fallback
        
        # For each alpha group, DDPG first, then SAC
        if 'DDPG' in algo:
            return base
        else:  # SAC
            return base + 1
    
    # First sort the items by our custom order for plotting
    sorted_items = []
    
    # Extract info and sort by our custom ordering
    for exp_name, exp_data in filtered_day_data.items():
        info = exp_data['info']
        algo = info['algo']
        chance = info.get('chance', None)
        order = get_legend_order(algo, chance)
        sorted_items.append((order, (exp_name, exp_data)))
    
    # Sort by our custom order
    sorted_items.sort()
    
    # Now plot in the sorted order
    max_steps = 0
    for _, (exp_name, exp_data) in sorted_items:
        info, dfs = exp_data['info'], exp_data['dfs']
        if not dfs: continue

        all_runs_df = pd.concat(dfs).sort_values(by='TrainingSteps').reset_index(drop=True)
        if not all_runs_df.empty:
            max_steps = max(max_steps, all_runs_df['TrainingSteps'].max())

        # Check if the pre-calculated std data is available
        if std_metric_name in all_runs_df.columns:
            agg_df = all_runs_df.groupby('TrainingSteps')[[mean_metric_name, std_metric_name]].mean().reset_index()
            std_values = agg_df[std_metric_name]
            mean_values = agg_df[mean_metric_name]
        else:
            # Fallback if no pre-calculated std is found
            print(f"Warning: Std metric '{std_metric_name}' not found. Calculating std across seeds.")
            agg_df = all_runs_df.groupby('TrainingSteps')[mean_metric_name].agg(['mean', 'std']).reset_index()
            agg_df.columns = ['TrainingSteps', 'mean', 'std']
            std_values = agg_df['std'].fillna(0)
            mean_values = agg_df['mean']

        # Only smooth the mean values, not the standard deviation
        mean_smooth = savgol_filter(mean_values, config['smoothing_window'], config['smoothing_polyorder'])
        # Ensure std values are non-negative but DON'T smooth them
        std_values = np.maximum(std_values, 0)

        style_group = config['plot_styles'].get(info['algo'], {})
        style = style_group.get(info['chance'], {}) if 'Lagrangian' in info['algo'] else style_group
        color, linestyle, label = style.get('color', 'gray'), style.get('linestyle', '-'), style.get('label', exp_name)

        ax.plot(agg_df['TrainingSteps'], mean_smooth, label=label, color=color, linestyle=linestyle, linewidth=1.5)
        ax.fill_between(agg_df['TrainingSteps'], mean_smooth - std_values, mean_smooth + std_values, alpha=0.15, color=color)

    ax.set_ylabel(mean_metric_name)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: f'{x / scale_factor:.1f}'))
    return max_steps

# This function now displays each figure in the notebook before closing it.
def plot_publication_figures_chance95(data: dict, config: dict):
    """
    Generates a multi-page PDF with one page for Evaluation and one for Training data.
    """
    data_by_day = defaultdict(dict)
    for exp_name, exp_data in data.items():
        data_by_day[exp_data['info']['days']][exp_name] = exp_data

    if not data_by_day:
        print("No data to plot.")
        return

    save_path = os.path.join(config['output_dir'], 'performance_grid_chance95.pdf')
    with PdfPages(save_path) as pdf:
        print(f"\n--- Generating multi-page PDF: {save_path} ---")

        for metric_type in ["Evaluation", "Training"]:
            print(f"  -> Plotting page for {metric_type} data...")
            day_keys = sorted(data_by_day.keys())
            n_rows, n_cols = len(day_keys), 2

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows), squeeze=False)
            fig.suptitle(fr'\textbf{{Performance Metrics ({metric_type})}}', fontsize=16, y=0.96)

            for i, day in enumerate(day_keys):
                day_data = data_by_day[day]
                day_number = day.replace('days', '')
                ax_return, ax_violations = axes[i, 0], axes[i, 1]

                # Determine scale
                temp_max_steps = 0
                for exp_data in day_data.values():
                    for df in exp_data['dfs']:
                        if not df.empty:
                            temp_max_steps = max(temp_max_steps, df['TrainingSteps'].max())
                
                if temp_max_steps > 1.5e6: scale_factor, scale_label = 1e6, r'($\times 10^6$)'
                elif temp_max_steps > 1.5e3: scale_factor, scale_label = 1e3, r'($\times 10^3$)'
                else: scale_factor, scale_label = 1.0, ''

                # --- THIS IS THE FIX ---
                # Explicitly define all metric names to avoid errors.
                return_mean_metric = f'Return ({metric_type})'
                return_std_metric = f'Return Std ({metric_type})'
                violations_mean_metric = f'Number of Violations ({metric_type})'
                violations_std_metric = f'Number of Violations Std ({metric_type})'
                
                # Pass both mean and std names to the plotting function.
                plot_single_metric_chance95(ax_return, return_mean_metric, return_std_metric, day_data, config, scale_factor=scale_factor)
                plot_single_metric_chance95(ax_violations, violations_mean_metric, violations_std_metric, day_data, config, scale_factor=scale_factor)

                # Set labels and titles
                ax_return.set_ylabel(f'$d={day_number}$', fontsize=12)
                ax_violations.set_ylabel('')
                if i == 0:
                    ax_return.set_title(r'\textbf{Return}', fontsize=14)
                    ax_violations.set_title(r'\textbf{Number of Violations}', fontsize=14)

                row_xlabel = f'Training Steps {scale_label}'
                ax_return.set_xlabel(row_xlabel, fontsize=14)
                ax_violations.set_xlabel(row_xlabel, fontsize=14)

            # The legend order is now correct for BOTH plots because it's handled properly in plot_single_metric.
            handles, labels = axes[0, 0].get_legend_handles_labels()
            n_variants = len(handles) // 2 if len(handles) > 0 else 5
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01),
                    ncol=n_variants, frameon=False, title=r'\textbf{Model}', 
                    fontsize=12, title_fontsize=14)
            
            fig.tight_layout(rect=[0, 0.08, 1, 0.98], h_pad=2.0, w_pad=1.0)
            
            pdf.savefig(fig, bbox_inches='tight')
            display(fig)
            plt.close(fig)

    print(f"✅ Multi-page figure saved to: {save_path}")

# And finally, call the new function in your execution cell:
plot_publication_figures_chance95(all_model_data, CONFIG)


SIMULATION_CONSTANTS = {
    's_star': 0.35, # Target Soil Moisture
    'sfc': 0.65,    # Field Capacity
    'sw': 0.3      # Wilting Point
}

def plot_simulation(df, title, save_path, constants):
    """
    Reads simulation data from a pandas DataFrame and creates an enhanced,
    publication-quality plot with correctly layered elements.
    """
    plt.style.use('seaborn-v0_8-paper')

    # --- 1. Define the Color Palette ---
    history_It = df["History It (before scaling)"] * 1000
    history_Rain = df["History Rainfall"]
    history_st = df["History Soil Moisture"]
    days = np.arange(len(history_st))

    color_moisture = "#0D7607"
    color_rain = "#1916ED"
    color_irrigation = "#4B3633"
    color_feasible_zone = "#B5E6B8"
    color_chance_zone = "#FFB46E"
    color_hard_zone = "#FD9286"

    # --- 2. Create the Plot and Axes ---
    fig, ax1 = plt.subplots(figsize=(14, 6))
    bar_width = 0.25
    ax2 = ax1.twinx()

    # --- 3. Add Shaded Background Zones with LOWER alpha ---
    ax2.axhspan(0, constants['sw'], color=color_hard_zone, alpha=0.3, label='Hard-Constrained Zone', zorder=0)
    ax2.axhspan(constants['sw'], constants['s_star'], color=color_chance_zone, alpha=0.3, label='Chance-Constrained Zone', zorder=0)
    ax2.axhspan(constants['s_star'], constants['sfc'], color=color_feasible_zone, alpha=0.3, label='Feasible Zone', zorder=0)
    ax2.axhspan(constants['sfc'], 1, color=color_chance_zone, alpha=0.3, label='Chance-Constrained Zone', zorder=0)

    # --- 4. Plot the Main Data with specific layer orders ---
    # Make sure bars are drawn on the correct axis (ax1) with higher zorder
    rain_bars = ax1.bar(days - bar_width/2, history_Rain, width=bar_width, color=color_rain, 
                        label='Rainfall (mm)', zorder=5)
    irrigation_bars = ax1.bar(days + bar_width/2, history_It, width=bar_width, color=color_irrigation, 
                              label='Irrigation (mm)', zorder=5)
    
    # Horizontal lines
    ax2.axhline(y=constants['sfc'], color="#BC850E", linestyle='--', 
                label=r'Field Capacity ($s_{fc}$)', linewidth=1.5, zorder=2)
    ax2.axhline(y=constants['s_star'], color="#7746D9", linestyle='--', 
                label=r'Target ($s^*$)', linewidth=1.5, zorder=2)
    ax2.axhline(y=constants['sw'], color="#9F4500", linestyle='--', 
                label=r'Wilting Point ($s_{w}$)', linewidth=1.5, zorder=2)
    
    # The main soil moisture line goes on top of everything
    moisture_line = ax2.plot(days, history_st, color=color_moisture, 
                             label='Soil Moisture', linewidth=1.5, zorder=6)[0]

    # --- 5. Formatting and Legend ---
    ax1.set_xlabel(r'Day', fontsize=14)
    ax1.set_ylabel(r'Precipitation \& Irrigation (mm)', fontsize=14, color=color_rain)  # Blue color to match rain
    ax2.set_ylabel(r'Soil Moisture Level', fontsize=14, color=color_moisture)
    ax1.set_ylim(0, max(np.max(history_Rain), np.max(history_It)) * 1.25 if len(history_It) > 0 else 10)
    ax2.set_ylim(0, 1)

    # Ensure ax1 (precipitation axis) is drawn in front of ax2 (moisture axis)
    ax1.set_zorder(10)
    ax1.patch.set_visible(False)  # This makes ax1's background transparent

    ax1.tick_params(axis='y', labelsize=12, colors=color_rain)  # Blue for rainfall axis
    ax1.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12, colors=color_moisture)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.title(title, fontsize=18)

    # Create THREE SEPARATE LEGEND ROWS with manual grouping
    # First row: Time series data
    legend1_items = [
        (moisture_line, 'Soil Moisture'),
        (rain_bars, 'Rainfall (mm)'),
        (irrigation_bars, 'Irrigation (mm)')
    ]
    
    # Second row: Threshold lines  
    legend2_items = [
        (ax2.get_lines()[0], r'Field Capacity ($s_{fc} = 0.65$)'),  # Field capacity line
        (ax2.get_lines()[1], r'Water Stress Point ($s^* = 0.35$)'),             # Target line
        (ax2.get_lines()[2], r'Permanent Wilting Point ($s_{w} = 0.3$)')     # Wilting point line
    ]
    
    # Third row: Zones
    legend3_items = [
        (plt.Rectangle((0, 0), 1, 1, fc=color_feasible_zone, alpha=0.3), 'Feasible Region'),
        (plt.Rectangle((0, 0), 1, 1, fc=color_chance_zone, alpha=0.3), 'Chance-Constrained Region'),
        (plt.Rectangle((0, 0), 1, 1, fc=color_hard_zone, alpha=0.3), 'Hard-Constrained Region')
    ]
    
    # Unpack each legend group
    handles1, labels1 = zip(*legend1_items)
    handles2, labels2 = zip(*legend2_items)
    handles3, labels3 = zip(*legend3_items)
    
    # Place three separate legends at different vertical positions
    fig.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=len(handles1), frameon=False, fontsize=12)
               
    fig.legend(handles2, labels2, loc='lower center', bbox_to_anchor=(0.5, -0.03), 
               ncol=len(handles2), frameon=False, fontsize=12)
               
    fig.legend(handles3, labels3, loc='lower center', bbox_to_anchor=(0.5, -0.08), 
               ncol=len(handles3), frameon=False, fontsize=12)

    # Adjust the bottom margin to make room for all three legend rows
    fig.tight_layout(rect=[0, 0.15, 1, 0.95])
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  -> Saved plot to {save_path}")
    plt.show()
    plt.close(fig)
    
def batch_plot_simulations(base_dir, episode_to_plot, config):
    """
    Finds and plots all simulation CSVs for a specific episode,
    saving them as PDFs to the main output directory.
    """
    print(f"\n--- Searching for simulation data for Episode {episode_to_plot} ---")
    
    # --- CHANGE #1: Use the output directory from your CONFIG ---
    output_plot_dir = config['output_dir']
    os.makedirs(output_plot_dir, exist_ok=True) # Ensure it exists
    
    # Use os.walk to search all subdirectories robustly
    for root, dirs, files in os.walk(base_dir):
        csv_filename = f"simulation_data_episode{episode_to_plot}.csv"
        
        if csv_filename in files:
            csv_path = os.path.join(root, csv_filename)
            model_dir_name = os.path.basename(root)
            
            print(f"\nFound data for: {model_dir_name}")
            
            try:
                df = pd.read_csv(csv_path)
                model_info = parse_model_info(model_dir_name)
                
                if model_info:
                    algo = model_info['algo']
                    chance = model_info['chance']
                    day_num = model_info['days'].replace('days', '')

                    display_algo = algo.replace('DDPGLagrangian', 'DDPG Lagrangian').replace('SACLagrangian', 'SAC Lagrangian')
                    
                    if 'Lagrangian' in algo:
                        title = fr'\textbf{{{display_algo} ($\alpha={chance}$, $d={day_num}$) | Evaluation Episode (Seed 59)}}'
                    else:
                        title = fr'\textbf{{{display_algo} ($d={day_num}$) | Evaluation Episode (Seed 59)}}'
                else:
                    title = fr'\textbf{{{model_dir_name} - Evaluation Episode (Seed 59)}}'

                # Save as .pdf ---
                save_filename = f"simulation_{model_dir_name}_e{episode_to_plot}.pdf"
                save_path = os.path.join(output_plot_dir, save_filename)
                
                plot_simulation(df, title, save_path, SIMULATION_CONSTANTS)

            except Exception as e:
                print(f"  -> ERROR processing {csv_path}: {e}")

# Choose which episode you want to generate plots for
EPISODE_TO_PLOT = 499 

batch_plot_simulations(CONFIG["base_directory"], episode_to_plot=EPISODE_TO_PLOT, config=CONFIG)


# %%
# =============================================================================
# CELL 2: CORE PLOTTING HELPER FUNCTION
# =============================================================================
def populate_simulation_axes(ax1, df, constants):
    """
    This is the core plotting logic, which draws all elements onto a given subplot axis (ax1).
    """
    # --- 1. Extract Data and Define Colors ---
    history_It = df["History It (before scaling)"] * 1000
    history_Rain = df["History Rainfall"]
    history_st = df["History Soil Moisture"]
    days = np.arange(len(history_st))

    color_moisture = "#0D7607"
    color_rain = "#1916ED"
    color_irrigation = "#4B3633"
    color_feasible_zone = "#B5E6B8"
    color_chance_zone = "#FFB46E"
    color_hard_zone = "#FD9286"
    
    bar_width = 0.25
    ax2 = ax1.twinx()

    # --- 2. Add Shaded Background Zones ---
    ax2.axhspan(0, constants['sw'], color=color_hard_zone, alpha=0.3, label='Hard-Constrained Region', zorder=0)
    ax2.axhspan(constants['sw'], constants['s_star'], color=color_chance_zone, alpha=0.3, label='Chance-Constrained Region', zorder=0)
    ax2.axhspan(constants['s_star'], constants['sfc'], color=color_feasible_zone, alpha=0.3, label='Feasible Region', zorder=0)
    ax2.axhspan(constants['sfc'], 1, color=color_chance_zone, alpha=0.3, label='Chance-Constrained Region', zorder=0)

    # --- 3. Plot the Main Data ---
    rain_bars = ax1.bar(days - bar_width/2, history_Rain, width=bar_width, color=color_rain, label='Rainfall (mm)', zorder=5)
    irrigation_bars = ax1.bar(days + bar_width/2, history_It, width=bar_width, color=color_irrigation, label='Irrigation (mm)', zorder=5)
    
    ax2.axhline(y=constants['sfc'], color="#BC850E", linestyle='--', label=r'Field Capacity ($s_{fc}$)', linewidth=1.5, zorder=2)
    ax2.axhline(y=constants['s_star'], color="#7746D9", linestyle='--', label=r'Target ($s^*$)', linewidth=1.5, zorder=2)
    ax2.axhline(y=constants['sw'], color="#9F4500", linestyle='--', label=r'Permanent Wilting Point ($s_{w}$)', linewidth=1.5, zorder=2)
    
    moisture_line = ax2.plot(days, history_st, color=color_moisture, label='Soil Moisture', linewidth=1, zorder=6)[0]

    # --- 4. Formatting ---
    ax1.set_ylabel(r'Precipitation \& Irrigation (mm)', fontsize=14, color=color_rain)
    ax2.set_ylabel(r'Soil Moisture Level', fontsize=14, color=color_moisture)
    ax1.set_ylim(0, max(np.max(history_Rain), np.max(history_It)) * 1.25 if len(history_It) > 0 else 10)
    ax2.set_ylim(0, 1)

    ax1.set_zorder(10)
    ax1.patch.set_visible(False)

    ax1.tick_params(axis='y', labelsize=12, colors=color_rain)
    ax1.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12, colors=color_moisture)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Return the handles and labels needed for the shared legend
    return moisture_line, rain_bars, irrigation_bars, ax2.get_lines()

# %%
# =============================================================================
# CELL 3: MAIN SIMULATION PLOTTING FUNCTION
# =============================================================================
def plot_all_days_comparison(base_dir, episode_to_plot, config):
    """
    Creates a multi-page PDF, where each page is a 2x2 grid comparing
    regular and chance=0.95 models for a specific day configuration (d=1, 3, 7).
    """
    # 1. Find all unique 'day_key's available in the data
    day_keys = set()
    for root, dirs, files in os.walk(base_dir):
        model_info = parse_model_info(os.path.basename(root))
        if model_info:
            day_keys.add(model_info['days'])

    if not day_keys:
        print("No model data found to plot.")
        return

    # 2. Initialize the multi-page PDF file
    save_path = os.path.join(config['output_dir'], f'simulation_comparison_all_days_e{episode_to_plot}.pdf')
    with PdfPages(save_path) as pdf:
        print(f"\n--- Creating multi-page comparison PDF: {save_path} ---")

        # 3. Loop through each day_key to create one page per day
        for day_key in sorted(list(day_keys)):
            print(f"  -> Plotting page for {day_key}...")

            # Find the four specific model data files for this day_key
            models_to_find = {'DDPG': None, 'SAC': None, 'DDPG Lagrangian': None, 'SAC Lagrangian': None}
            for root, dirs, files in os.walk(base_dir):
                if day_key not in root:
                    continue
                
                csv_filename = f"simulation_data_episode{episode_to_plot}.csv"
                if csv_filename in files:
                    model_info = parse_model_info(os.path.basename(root))
                    if not model_info: continue

                    is_regular = 'Lagrangian' not in model_info['algo']
                    is_chance_95 = 'Lagrangian' in model_info['algo'] and model_info['chance'] == '0.95'
                    key = model_info['algo'].replace('DDPGLagrangian', 'DDPG Lagrangian').replace('SACLagrangian', 'SAC Lagrangian')

                    if is_regular and key in models_to_find:
                        models_to_find[key] = (os.path.join(root, csv_filename), model_info['chance'])
                    elif is_chance_95 and key in models_to_find:
                        models_to_find[key] = (os.path.join(root, csv_filename), model_info['chance'])

            # Create the 2x2 figure grid for the current page
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
            plot_map = {(0, 0): 'DDPG', (0, 1): 'SAC', (1, 0): 'DDPG Lagrangian', (1, 1): 'SAC Lagrangian'}

            # Populate each subplot
            for (row, col), model_name in plot_map.items():
                ax = axes[row, col]
                found_data = models_to_find.get(model_name)
                
                if found_data and os.path.exists(found_data[0]):
                    csv_path, chance_val = found_data
                    df = pd.read_csv(csv_path)
                    moisture_line, rain_bars, irrigation_bars, ax2_lines = populate_simulation_axes(ax, df, SIMULATION_CONSTANTS)
                    
                    if model_name.endswith('Lagrangian'):
                        title_name = fr'{model_name} ($\alpha={chance_val}$)'
                    else:
                        title_name = f'{model_name} (Regular)'
                    ax.set_title(fr'\textbf{{{title_name}}}', fontsize=16)
                else:
                    ax.text(0.5, 0.5, f'Data not found for\n{model_name}', ha='center', va='center', fontsize=12)
                    ax.set_xticks([]); ax.set_yticks([])

            axes[1, 0].set_xlabel(r'Day', fontsize=14)
            axes[1, 1].set_xlabel(r'Day', fontsize=14)

            # Create the shared, three-row legend for the current page
            legend1_items = [(moisture_line, 'Soil Moisture'), (rain_bars, 'Rainfall (mm)'), (irrigation_bars, 'Irrigation (mm)')]
            legend2_items = [(ax2_lines[0], r'Field Capacity ($s_{fc} = 0.65$)'), (ax2_lines[1], r'Water Stress Point ($s^* = 0.35$)'), (ax2_lines[2], r'Permanent Wilting Point ($s_{w} = 0.3$)')]
            legend3_items = [(plt.Rectangle((0, 0), 1, 1, fc="#B5E6B8", alpha=0.3), 'Feasible Region'), (plt.Rectangle((0, 0), 1, 1, fc="#FFB46E", alpha=0.3), 'Chance-Constrained Region'), (plt.Rectangle((0, 0), 1, 1, fc="#FD9286", alpha=0.3), 'Hard-Constrained Region')]
            
            handles1, labels1 = zip(*legend1_items)
            handles2, labels2 = zip(*legend2_items)
            handles3, labels3 = zip(*legend3_items)
            
            # Place three separate legends at different vertical positions
            fig.legend(handles1, labels1, loc='lower center', bbox_to_anchor=(0.5, 0.03), 
                       ncol=len(handles1), frameon=False, fontsize=12)
                       
            fig.legend(handles2, labels2, loc='lower center', bbox_to_anchor=(0.5, 0.0), 
                       ncol=len(handles2), frameon=False, fontsize=12)
                       
            fig.legend(handles3, labels3, loc='lower center', bbox_to_anchor=(0.5, -0.03), 
                       ncol=len(handles3), frameon=False, fontsize=12)

            # Add overall title for the page
            day_number = day_key.replace('days', '')
            page_title = f"Model Performance During Evaluation, Seed 59 ($d={day_number}$)"
            fig.suptitle(fr'\textbf{{{page_title}}}', fontsize=18, y=0.98)
            
            # Adjust layout to make room for all three legend rows
            fig.tight_layout(rect=[0, 0.1, 1, 0.99], h_pad=1.5, w_pad=1.0)
            
            # Save to PDF and show in notebook
            pdf.savefig(fig, bbox_inches='tight')
            display(fig)
            plt.close(fig)

    print(f"✅ Multi-page figure saved to: {save_path}")


# Choose the episode you want to compare across all day configurations
EPISODE_TO_PLOT = 499

plot_all_days_comparison(CONFIG["base_directory"],
                         episode_to_plot=EPISODE_TO_PLOT,
                         config=CONFIG)


def calculate_simulation_stats(df, constants):
    """
    Calculate key statistics from a simulation dataframe.
    """
    # Convert the irrigation to mm (multiply by 1000)
    irrigation_mm = df["History It (before scaling)"] * 1000
    soil_moisture = df["History Soil Moisture"]
    
    # Calculate key statistics
    total_irrigation = irrigation_mm.sum()
    
    # Calculate probabilities
    total_days = len(soil_moisture)
    days_above_s_star = sum(soil_moisture >= constants['s_star'])
    days_below_s_star = sum(soil_moisture < constants['s_star'])
    days_below_sfc = sum(soil_moisture <= constants['sfc'])
    days_above_sfc = sum(soil_moisture > constants['sfc'])
    days_above_sw = sum(soil_moisture >= constants['sw'])
    days_below_sw = sum(soil_moisture < constants['sw'])
    
    # Convert to probabilities
    prob_above_s_star = days_above_s_star / total_days
    prob_below_sfc = days_below_sfc / total_days
    prob_above_sw = days_above_sw / total_days
    
    # Number of violations = absolute count of days with violations
    num_violations = days_below_s_star + days_below_sw + days_above_sfc
    
    # The return should be available in the dataframe if it was stored
    episode_return = df["Episode Return"].iloc[-1] if "Episode Return" in df.columns else None
    
    return {
        "total_irrigation_mm": total_irrigation,
        "prob_above_s_star": prob_above_s_star,
        "prob_below_sfc": prob_below_sfc,
        "prob_above_sw": prob_above_sw,
        "num_violations": num_violations,
        "episode_return": episode_return
    }

def print_all_simulation_stats(base_dir, episode_to_analyze, constants):
    """
    Finds all simulation CSV files for a specific episode and prints their statistics.
    """
    print(f"\n--- Statistics for Episode {episode_to_analyze} ---")
    # Update the table header
    print("| Model | Day | α | Total Irrigation (mm) | P(s ≥ s*) | P(s ≤ sfc) | P(s ≥ sw) | Violations | Return |")
    print("|-------|-----|---|---------------------|-----------|-----------|-----------|-----------|--------|")

    
    # Create a list to store all results for later DataFrame creation
    all_stats = []
    
    # Use os.walk to search all subdirectories
    for root, dirs, files in os.walk(base_dir):
        csv_filename = f"simulation_data_episode{episode_to_analyze}.csv"
        
        if csv_filename in files:
            csv_path = os.path.join(root, csv_filename)
            model_dir_name = os.path.basename(root)
            
            try:
                df = pd.read_csv(csv_path)
                model_info = parse_model_info(model_dir_name)
                
                if model_info:
                    algo = model_info['algo']
                    chance = model_info['chance']
                    day_num = model_info['days'].replace('days', '')
                    
                    display_algo = algo.replace('DDPGLagrangian', 'DDPG-L').replace('SACLagrangian', 'SAC-L')
                    
                    # Calculate statistics
                    stats = calculate_simulation_stats(df, constants)
                    
                    # If return is not found in the CSV, try to get it from TensorBoard
                    if stats['episode_return'] is None:
                        # Look for the tensorboard directory
                        tb_path = os.path.join(root, "tb")
                        if os.path.exists(tb_path):
                            try:
                                ea = event_accumulator.EventAccumulator(tb_path)
                                ea.Reload()
                                
                                # Look for evaluation return data
                                if 'Averageeval/TestEpRet' in ea.Tags().get('scalars', []):
                                    # Get the last evaluation return value
                                    stats['episode_return'] = ea.Scalars('Averageeval/TestEpRet')[-1].value
                                    print(f"  -> Found return from TensorBoard for {model_dir_name}: {stats['episode_return']:.1f}")
                            except Exception as e:
                                print(f"  -> Could not read TensorBoard data: {e}")
                    
                    # Add to results list
                    all_stats.append({
                        'Model': display_algo,
                        'Days': int(day_num),
                        'Alpha': chance,
                        'Total Irrigation (mm)': stats['total_irrigation_mm'],
                        'P(s ≥ s*)': stats['prob_above_s_star'],
                        'P(s ≤ sfc)': stats['prob_below_sfc'],
                        'P(s ≥ sw)': stats['prob_above_sw'],
                        'Violations': stats['num_violations'],
                        'Return': stats['episode_return']
                    })
                    
                    # Format the return value for display
                    if stats['episode_return'] is not None:
                        return_str = f"{stats['episode_return']:.1f}"
                    else:
                        return_str = "N/A"
                        
                    # Print as Markdown table row
                    print(f"| {display_algo} | {day_num} | {chance} | {stats['total_irrigation_mm']:.2f} | "
                        f"{stats['prob_above_s_star']*100:.2f}% | {stats['prob_below_sfc']*100:.2f}% |"
                        f"{stats['prob_above_sw']*100:.2f}% | {stats['num_violations']} | "
                        f"{return_str} |")
                    
            except Exception as e:
                print(f"  -> ERROR processing {csv_path}: {e}")
    
    # Create a DataFrame with all the statistics
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        # Sort by Days, then Alpha, then Model
        stats_df = stats_df.sort_values(by=['Days', 'Alpha', 'Model'])
        
        # Save to CSV
        stats_csv_path = os.path.join(CONFIG['output_dir'], f'simulation_stats_e{episode_to_analyze}.csv')
        stats_df.to_csv(stats_csv_path, index=False)
        print(f"\nStatistics saved to: {stats_csv_path}")
        
        return stats_df
    else:
        print("No statistics were calculated.")
        return None
    
# Choose which episode you want to analyze
EPISODE_TO_ANALYZE = 499

# Calculate and print statistics
stats_df = print_all_simulation_stats(CONFIG["base_directory"], 
                                     episode_to_analyze=EPISODE_TO_ANALYZE,
                                     constants=SIMULATION_CONSTANTS)

# Display the dataframe in the notebook (optional)
if stats_df is not None:
    display(stats_df)

def plot_results(data, plot_name, model_directory, episode_count):
    print(f"Data received for plotting: {data}")  # Debug print to check data integrity
    episodes = np.arange(1, len(data) + 1)

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(episodes, data, color='blue', label=plot_name.capitalize())

    ax.set_title(f'Total {plot_name.capitalize()}', fontsize=14)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(plot_name.capitalize())
    ax.grid(True)
    ax.legend()

    # Save the plot
    plot_file_name = os.path.join(model_directory, f'{plot_name}_plot_episode_{episode_count}.png')
    plt.savefig(plot_file_name)
    plt.close(fig)

    print(f"{plot_name} plot saved to {plot_file_name}")

def plot_simulation_data(history_It, history_Rain, history_st, history_ET_o, history_ETmax, history_Kc, history_rho, s_star, sfc, sw, model_directory, episode_count, model_name):
    # Scaling and preparing data
    original_history_It = np.array(history_It)
    history_It = original_history_It * 1000  # Scale irrigation data
    history_Rain = np.array(history_Rain)
    history_st = np.array(history_st)
    history_ET_o = np.array(history_ET_o)
    history_ETmax = np.array(history_ETmax)
    history_Kc = np.array(history_Kc)
    history_rho = np.array(history_rho)

    # Ensure consistent data length
    min_length = min(len(history_It), len(history_Rain), len(history_st), len(history_ET_o), len(history_ETmax), len(history_Kc), len(history_rho))
    if min_length == 0:
        print("No data available to plot.")
        return
    
    # Truncate data to minimum length
    history_It = history_It[:min_length]
    original_history_It = original_history_It[:min_length]
    history_Rain = history_Rain[:min_length]
    history_st = history_st[:min_length]
    history_ET_o = history_ET_o[:min_length]
    history_ETmax = history_ETmax[:min_length]
    history_Kc = history_Kc[:min_length]
    history_rho = history_rho[:min_length]
    days = np.arange(min_length)

    # Creating plot
    fig, ax1 = plt.subplots(figsize=(18, 10))
    color_rain = 'blue'
    color_irrigation = 'red'
    color_moisture = 'seagreen'

    # Set consistent bar width
    bar_width = 0.35

    # Plot rainfall with consistent width
    ax1.bar(days - bar_width/2, history_Rain, width=bar_width, color=color_rain, alpha=0.7, label='Rainfall (mm)')
    ax1.set_xlabel('Day', fontsize=16)
    ax1.set_ylabel('Precipitation and Irrigation (mm)', color='black', fontsize=16)
    ax1.tick_params(axis='y', labelsize=14, labelcolor='black')
    ax1.set_ylim(0, max(np.max(history_Rain), np.max(history_It)) * 1.2)

    # Create second y-axis for soil moisture
    ax2 = ax1.twinx()
    ax2.plot(days, history_st, color=color_moisture, label='Soil Moisture', linewidth=2)
    ax2.set_ylabel('Soil Moisture', color=color_moisture, fontsize=16)  # Increase font size for y-axis label on the second axis
    ax2.tick_params(axis='y', labelsize=14, labelcolor=color_moisture)

    # Add soil moisture target lines
    ax2.axhline(y=s_star, color='purple', linestyle='--', label='Target Soil Moisture ($s^*$)', linewidth=1.5)
    ax2.axhline(y=sfc, color='orange', linestyle='--', label='Field Capacity ($s_{fc}$)', linewidth=1.5)
    ax2.axhline(y=sw, color='brown', linestyle='--', label='Wilting Point ($s_{w}$)', linewidth=1.5)

    # Plot irrigation with consistent width on the same axis as rainfall
    ax1.bar(days + bar_width/2, history_It, width=bar_width, color=color_irrigation, alpha=0.7, label='Irrigation (mm)')
    
    # Combine legends from both y-axes and position closer to the plot
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles=handles1 + handles2, labels=labels1 + labels2, loc='upper right', bbox_to_anchor=(1, 1), fontsize=12)
    
    # Show plot with title including the model name
    plt.title(f'{model_name}: Daily Rainfall, Irrigation, and Soil Moisture Levels (Evaluation)', fontsize=18)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plot_simulation_name = os.path.join(model_directory, f'simulation_episode{episode_count}_plot.png')
    plt.savefig(plot_simulation_name)
    plt.close(fig)

    # Save data to CSV, including history_ET_o and history_ETmax
    csv_file_name = os.path.join(model_directory, f'simulation_data_episode{episode_count}.csv')
    data = {
        "History It (before scaling)": original_history_It,
        "History It (after scaling)": history_It,
        "History Rainfall": history_Rain,
        "History Soil Moisture": history_st,
        "History ET_o": history_ET_o,
        "History ETmax": history_ETmax,
        "History Kc": history_Kc,
        "History rho": history_rho
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file_name, index=False)
    print(f"Simulation data saved to {csv_file_name}")
    print(f"Simulation plot saved to {plot_simulation_name}")