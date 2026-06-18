"""Functions for plotting data points in a 2D diagram."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import csv

def plot_2d(data_dict, file_path="plots/2d_plot.png", title="", connect=False):
    """
    Creates one or more 2D scatter/line plots.

    Parameters
    ----------
    data_dict : dict
        The data to be plotted. Should contain the independent values at key 0 and
        arbitrarily many dependent data columns at keys 1, 2, ...
        Must furthermore contain a key "ax_titles" to specify the meaning of each
        column as strings.
    file_path : string
        The file path (incl. file ending) as which to save the plot.
    title : string
        An optional title to display at the top of the plot.
    connect : bool
        Whether the points scattered in the plot should be connected by lines.
    """
    fig, ax1 = plt.subplots()
    colors = list(mcolors.TABLEAU_COLORS)

    # Plot all y-values with uneven indices (1, 3, ...)
    for i in range(1, len(data_dict["ax_titles"]), 2):
        ax1.scatter(data_dict[0], data_dict[i], color=colors[i-1])
        if connect: ax1.plot(data_dict[0], data_dict[i], color=colors[i-1], alpha=0.5)
    ax1.set_xlabel(data_dict["ax_titles"][0])
    ax1.set_ylabel(data_dict["ax_titles"][1], color=colors[1-1])

    # If there is more than one x-y pair to plot, add a second y axis and plot all even indices:
    if len(data_dict["ax_titles"])>2:
        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
        for i in range(2, len(data_dict["ax_titles"]), 2):
            ax2.scatter(data_dict[0], data_dict[i], color=colors[i-1])
            if connect: ax2.plot(data_dict[0], data_dict[i], color=colors[i-1], alpha=0.5)
        ax2.set_ylabel(data_dict["ax_titles"][2], color=colors[2-1])


    if title != "":
        plt.title(title)
    plt.savefig(file_path)

def plot_csv(csv_path, out_path="plots/2d_plot.png", title="", delimiter=';', connect=False):
    """
    Reads a csv file and plot its data as one or more 2D scatter/line plots.

    Parameters
    ----------
    csv_path : string or list of dict
        Should be either a string specifying a single file as csv_path, which will 
        result in indices 0 and 1 to be plotted, or a single-element list of a dict 
        with keys "path" and "indices" (to be plotted). The list is to allow later 
        implementation of reading multiple files.
    out_path : string
        The file path (incl. file ending) as which to save the plot.
    title : string
        An optional title to display at the top of the plot.
    delimiter : string
        The character to use as delimiter for csv parsing.
    connect : bool
        Whether the points scattered in the plot should be connected by lines.
    """
    csv_paths_dicts = csv_path
    if not isinstance(csv_path, list):
        csv_paths_dicts = [{"path": csv_path, "indices": {0, 1}}]
    for data_dict in csv_paths_dicts:
        csv_path = data_dict["path"]
        with open(csv_path, mode='r') as file:
            reader = csv.DictReader(file, delimiter=delimiter)

            # Empty lists to store column values
            for i in range(len(data_dict["indices"])):
                data_dict[i] = []

            data_dict["ax_titles"] = None

            # Iterating over each row
            for row in reader:
                if data_dict["ax_titles"] == None: 
                    keys = list(row.keys())
                    data_dict["ax_titles"] = tuple([keys[i] for i in data_dict["indices"]])
                for i in range(len(data_dict["indices"])):
                    data_dict[i].append(float(row[data_dict["ax_titles"][i]].replace(",", ".")))

        plot_2d(data_dict, file_path=out_path, title=title, connect=connect)


plot_csv("output_files/attn_weight_tuning_grid_search_decaying_tuning_dot_or_comma.csv", delimiter=",", title="Tuning")
# plot_csv([{"path": "output_files/attn_weight_tuning_grid_search_user_and_assistant_fine_assessed.csv", "indices": [0, 1, 3]}], delimiter=",", title="Tuning every '.' and ','", connect=True)