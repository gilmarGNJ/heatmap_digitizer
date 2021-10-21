#!/usr/bin/env python3

"""This module uses the HeatmapDigitizer class, witch digitizes heatmap images."""

import itertools
import ast
import argparse
import sys
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import TextBox
import seaborn as sb
import cv2
import pkg_resources


class HeatmapDigitizer:
    """
    Digitize a Heatmap image.

    ...

    Attribute
    ----------
    file_name : str
        The heatmap file name image.

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """

    def __init__(self, file_name):
        """
        Construct all the necessary attributes for the HeatmapDigitizer object.

        file_name : str
            The heatmap file name image.
        """
        self.file_name = file_name

        if file_name == "example":
            self.image = load_example()
        else:
            self.image = cv2.imread(self.file_name)

        self.fig, self.ax = plt.subplots()
        self.imshow = self.ax.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.ax.axis('off')

        self.texts = itertools.chain(["select first point in the low left conner of heatmap",
                                      "select second point in the upper right conner of heatmap",
                                      "select third point in the bottom of color scale",
                                      "select fourth point in the top of color scale",
                                      "enter xy coordinates of the first point. Ex: (-4.91, -6.15)",
                                      "enter xy coordinates of the second point. Ex: (4.91, 6.15)",
                                      "enter the corresponding value of the third point. Ex: 6.51",
                                      "enter the corresponding value of the fourth point. Ex: 6.07. "
                                      "\nWait a few seconds after pressing enter!",
                                      "heatmap dataframe saved as a csv file!"])
        self.text = self.ax.text(0.5, 1.1, next(self.texts),
                                 bbox=dict(facecolor='red', alpha=0.5),
                                 horizontalalignment='center', verticalalignment='top', transform=self.ax.transAxes,
                                 fontsize=15)

        self.points_coord = []
        self.points_values = []

        self.cid_press = self.graph_box = self.text_box = None
        self.color_scale_image = self.color_scale_y_values = None
        self.heatmap_image = self.heatmap_y_values = self.heatmap_x_values = self.heatmap_df = None
        self.grid_interpolated = self.rgb_to_value = None

        self.heatmap_plot = False

    def show_update(self):
        """Show the update on the plot."""
        self.ax.figure.canvas.draw()  # alternatives: plt.draw(), plt.show(), fig.show(), fig.canvas.draw()

    def connect(self):
        """Connect to click events."""
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.show_update()
        plt.show()

    def disconnect(self):
        """Disconnect click callbacks."""
        self.fig.canvas.mpl_disconnect(self.cid_press)
        self.cid_press = None

    def connect_box(self):
        """Create box and connect to box events."""
        self.fig.subplots_adjust(bottom=0.2)
        self.graph_box = self.fig.add_axes([0.4, 0.05, 0.3, 0.075])
        self.text_box = TextBox(self.graph_box, "enter xy coordinates: ")
        self.text_box.on_submit(self.on_enter)
        self.show_update()

    def disconnect_box(self):
        """Remove box and disconnect press callbacks."""
        self.text_box.disconnect_events()
        self.graph_box.remove()
        self.fig.subplots_adjust(bottom=0.1)
        self.show_update()

    def on_click(self, event):
        """When user clicks, this function is called."""
        self.draw_text()
        self.points_coord.append((round(event.xdata), round(event.ydata)))

        print(f'click: x={event.xdata}, y={event.ydata}')

        if len(self.points_coord) == 2:
            self.draw_rectangle()
            self.heatmap_cut()

        if len(self.points_coord) == 4:
            self.draw_rectangle()
            self.color_scale_cut()
            self.connect_box()
            self.disconnect()

    def on_enter(self, event):
        """When user press enter, this function is called."""
        if event == "":
            return

        print(f'enter: BoxValue={event}')
        self.points_values.append(ast.literal_eval(event))
        self.draw_text()
        self.text_box.set_val("")

        if len(self.points_values) == 2:
            self.text_box.label.set_text("enter color values: ")

        self.show_update()

        if len(self.points_values) == 4:
            self.disconnect_box()
            self.heatmap_digitizer()

    def draw_rectangle(self):
        """Create a Rectangle patch."""
        point_1 = self.points_coord[-2]
        point_2 = self.points_coord[-1]

        xy = (point_1[0], point_1[1])
        width = (point_2[0] - point_1[0] if len(self.points_coord) != 4 else 0)
        height = point_2[1] - point_1[1]
        rect = patches.Rectangle(xy, width, height, linewidth=3, edgecolor='r', facecolor='none')

        self.ax.add_patch(rect)
        self.show_update()

    def draw_text(self):
        """Update text to the next action."""
        self.text.set_text(next(self.texts))
        self.show_update()

    def heatmap_cut(self):
        """Cut heatmap from image."""
        x_1, y_1, x_2, y_2 = self.get_points_xy(index_point_1=0, index_point_2=1)
        self.heatmap_image = self.image[y_2:y_1, x_1:x_2]

    def color_scale_cut(self):
        """Cut color_scale from image."""
        x_1, y_1, x_2, y_2 = self.get_points_xy(index_point_1=2, index_point_2=3)
        self.color_scale_image = self.image[y_2:y_1, x_1:x_1 + 1]
        self.color_scale_image = self.color_scale_image.reshape((-1, 3))

    def get_points_xy(self, index_point_1, index_point_2):
        """Store clicked points."""
        point_1 = self.points_coord[index_point_1]
        point_2 = self.points_coord[index_point_2]
        x_1 = point_1[0]
        y_1 = point_1[1]
        x_2 = point_2[0]
        y_2 = point_2[1]
        return x_1, y_1, x_2, y_2

    def heatmap_digitizer(self):
        """Digitize heatmap."""
        self.color_scale_values()
        self.color_scale_interpolate()
        self.heatmap_xy_values()
        self.heatmap_image_to_df()
        self.save_heatmap_df()

    def color_scale_values(self):
        """Set color scale values."""
        color_scale_x_1_value = self.points_values[2]
        color_scale_x_2_value = self.points_values[3]
        number_of_points = self.color_scale_image.shape[0]
        self.color_scale_y_values = np.linspace(color_scale_x_2_value, color_scale_x_1_value, number_of_points)

    def color_scale_interpolate(self):
        """Interpolate color scale RGB."""
        points = self.color_scale_image
        values = self.color_scale_y_values
        grid_x, grid_y, grid_z = np.mgrid[0:256, 0:256, 0:256]
        self.grid_interpolated = griddata(points, values, (grid_x, grid_y, grid_z), method='nearest')

    def heatmap_xy_values(self):
        """Set heatmap coordinates values."""
        heatmap_x_1_value, heatmap_y_1_value = self.points_values[0]
        heatmap_x_2_value, heatmap_y_2_value = self.points_values[1]
        self.heatmap_y_values = np.linspace(heatmap_y_2_value, heatmap_y_1_value, self.heatmap_image.shape[0])
        self.heatmap_x_values = np.linspace(heatmap_x_1_value, heatmap_x_2_value, self.heatmap_image.shape[1])

    def heatmap_image_to_df(self):
        """Convert heatmap image to dataframe."""
        self.rgb_to_value = np.apply_along_axis(self.function_interpolated, -1, self.heatmap_image)
        self.heatmap_df = pd.DataFrame(self.rgb_to_value, index=self.heatmap_y_values, columns=self.heatmap_x_values)

    def function_interpolated(self, xyz):
        """Create function from a grid."""
        return self.grid_interpolated[xyz[0], xyz[1], xyz[2]]

    def save_heatmap_df(self):
        """Save dataframe to csv file."""
        self.heatmap_df.to_csv(f"{self.file_name.split('.')[0]}.csv")
        if self.heatmap_plot:
            self.csv_plot_example()

    def csv_plot_example(self):
        """As an example, plot the generated csv file."""
        fig, ax = plt.subplots(1, figsize=(800 / 96, 800 / 96), dpi=96)

        # An example of how to plot the dataframe:
        example = self.heatmap_df.copy()
        example.columns = np.around(example.columns.astype(float), decimals=2)
        example.index = np.around(example.index, decimals=2)

        average = example.mean().mean()
        heat_map = sb.heatmap(example, center=average, annot=False, cbar=True,
                              cbar_kws={'label': 'values'}, ax=ax, square=True)

        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        ax.set_title('An example plot using the generated csv file.')
        plt.show()


def load_example():
    data_path = pkg_resources.resource_filename('heatmap_digitizer', 'data/example.png')
    return cv2.imread(data_path)


def main():
    """Run HeatmapDigitizer with command line"""
    parser = argparse.ArgumentParser(description='extract dataframe from heatmap image')
    parser.add_argument('-e', '--example', help='extract dataframe from an image example', action="store_true")
    parser.add_argument('-p', '--plot', help='plot the csv generated file at the end', action="store_true")

    if not ('--example' in sys.argv or '-e' in sys.argv):
        parser.add_argument('file_name', type=str, help='the heatmap image name to extract')

    args = parser.parse_args()

    if args.example:
        example1 = HeatmapDigitizer("example")
        example1.heatmap_plot = args.plot
        example1.connect()

    else:
        heatmap = HeatmapDigitizer(args.file_name)
        heatmap.heatmap_plot = args.plot
        heatmap.connect()
