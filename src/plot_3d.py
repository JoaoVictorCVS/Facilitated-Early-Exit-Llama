"""Functions for plotting data points in a 3D visualization."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import csv


def plot_3d(x, y, z, titles, file_path="plots/3d_plot.png", center=None, title=""):

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create the surface plot using triangulation
    surf = ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none', alpha=0.8)
    ax.scatter(x, y, z, c='blue', s=50, marker='o', linewidths=1.5, alpha=1)

    # Add labels
    ax.set_xlabel(titles[0])
    ax.set_ylabel(titles[1])
    ax.set_zlabel(titles[2])
    if title!="":
        ax.set_title(title)

    # Add a color bar
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if center!=None:
        max_x_diff = max(ax.get_xlim()[1] - center[0], center[1] - ax.get_xlim()[0])
        max_y_diff = max(ax.get_ylim()[1] - center[0], center[1] - ax.get_ylim()[0])
        ax.set_xlim(center[0]-max_x_diff, center[0]+max_x_diff)
        ax.set_ylim(center[1]-max_y_diff, center[1]+max_y_diff)

    plt.savefig(file_path)
    print("Done")

def plot_4_column(x, y, z, v, titles, file_path="plots/3d_plot.png", center=None, title=""):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(x, y, z, c=v, cmap='coolwarm', s=100, depthshade=False)

    ax.set_xlabel(titles[0])
    ax.set_ylabel(titles[1])
    ax.set_zlabel(titles[2])
    if title!="":
        ax.set_title(title)
    
    ax=plt.gca() # get current axis
    PCM=ax.get_children()[0] # get the mappable, the 1st and the 2nd are the x and y axes
    plt.colorbar(PCM, ax=ax)

    if center!=None:
        max_x_diff = max(ax.get_xlim()[1] - center[0], center[1] - ax.get_xlim()[0])
        max_y_diff = max(ax.get_ylim()[1] - center[0], center[1] - ax.get_ylim()[0])
        ax.set_xlim(center[0]-max_x_diff, center[0]+max_x_diff)
        ax.set_ylim(center[1]-max_y_diff, center[1]+max_y_diff)

    plt.show()

def plot_csv(csv_path, out_path="plots/3d_plot.png", center=None, four_columns=False, indices=[0, 1, 2, 3], title="", delimiter=";"):
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file, delimiter=delimiter)

        # Empty lists to store column values
        x = []
        y = []
        z = []
        v = []

        titles = None

        # Iterating over each row
        for row in reader:
            if titles == None: 
                keys = list(row.keys())
                if not four_columns:
                    titles = (keys[0], keys[1], keys[2])
                else:
                    titles = (keys[indices[0]], keys[indices[1]], keys[indices[2]], keys[indices[3]])
            x.append(float(row[titles[0]].replace(",", ".")))
            y.append(float(row[titles[1]].replace(",", ".")))
            z.append(float(row[titles[2]].replace(",", ".")))
            if four_columns:
                v.append(float(row[titles[3]].replace(",", ".")))

    if not four_columns:
        plot_3d(x, y, z, titles, file_path=out_path, center=center, title=title)
    else: 
        plot_4_column(x, y, z, v, titles, file_path=out_path, center=center, title=title)


# plot_csv("attn_weight_tuning_grid_search_start-header-id_end-header-id_full_dataset_merged.csv", center=(1.3, 2.5))
plot_csv("attn_weight_tuning_grid_search_decaying_tuning_start-header-id_end-header-id_assessed.csv", four_columns=True, indices=[0, 1, 2, 4], delimiter=",", title="", center=(1.0, 8.0))