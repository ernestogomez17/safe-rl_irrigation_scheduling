import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


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
    plt.show()
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
    plt.show()
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