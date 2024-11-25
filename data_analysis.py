# coding: utf-8
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__authors__ = "Grégoire Douillard-Jacq"
__contact__ = "NextStep-ns@outlook.com"
__copyright__ = "ARTEMIS, Côte d'Azur Observatory"
__date__ = "2024-06-17"
__version__ = "1.0.0"
__status__ = "Production"
__privacy__ = "Confidential"
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

"""
------------------------------------------------------------------------------------------------------------------------
Syntax rules [PEP484, pycodestyle]:
- ClassName; function_name(); variable_name; CONSTANT_NAME;
- def function():
    "Docstring."

    Parameters
    ----------
    param : int -> description

    Returns
    -------
    bool
        True if successful, False otherwise.
        
- separators: 
### : in between functions
=== : in between subject parts or inside function
--- :  all other separations
------------------------------------------------------------------------------------------------------------------------
"""

import seaborn as sns
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from typing import Optional, List, Any
from matplotlib.axes import Axes
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QTableWidget
from header_dictionary import execute_with_error_handling

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################

# Initialization of constants used by all the classes
AMOUNT_OF_GRAPHS = 3
AMOUNT_OF_GRAPHS_CALIBRATOR = 2


class DataAnalysis(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the axes used to plot
        self.axes_list: List[Axes] = []
        self.ax: Optional[Axes] = None

        # Initialize the graph environment
        self.canvas: Optional[FigureCanvas] = None
        self.figure: Figure = Figure()
        self.plot_id: int = 0
        self.tool_bar: Any = None

        # Initialize the different layout of the tab
        self.layout_graph_da: QVBoxLayout = QVBoxLayout()
        self.layout_da: QGridLayout = QGridLayout()

        # Call the initialization function with error handling
        execute_with_error_handling(self.setup_layout)
        execute_with_error_handling(self.initialize_ui)

    ####################################################################################################################
    ####################################################################################################################

    def initialize_ui(self):
        """
        Initialize components of the 'DataAnalysis' tab

        :return: int
        """

        # Create the plot's framework
        execute_with_error_handling(self.create_plot)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def setup_layout(self):
        """
        Set layout's parameters.

        :return: int
        """

        # Add the graph layout to the whole surface of the main layout
        self.layout_da.addLayout(self.layout_graph_da, 0, 1, 1, 1)
        self.setLayout(self.layout_da)
        self.layout_da.update()

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def create_plot(self):
        """
        Create the plot framework and its parameters.

        :return: int
        """
        self.plot_id = 0
        self.axes_list = []

        # Delete the previous canvas if there is already one displayed
        if self.canvas:
            self.layout_graph_da.removeWidget(self.canvas)
            self.layout_graph_da.removeWidget(self.tool_bar)
            self.canvas.deleteLater()

            # Clear all axes from the figure
            self.figure.clf()

        # Add the graph to the graph's layout
        self.canvas = FigureCanvas(self.figure)
        self.layout_graph_da.addWidget(self.canvas)
        self.layout_graph_da.update()

        # Adjust the figure's parameters for the different graphs to be well separated
        self.figure.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2, wspace=0.4, hspace=0.4)

        # Define the plot's style
        sns.set_style("ticks")
        sns.set_context("talk", font_scale=0.9)

        # Add a toolbar to ba able to move and zoom in each plot
        self.tool_bar = NavigationToolbar(self.canvas, self)
        self.layout_graph_da.addWidget(self.tool_bar)

        print("Graph framework displayed.")

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def compare_pos_plot(self, xdata: object, ydata: object, color: object, title: object, x_label: object,
                         y_label: object,
                         legend: object,
                         graph_id: object,
                         nb_or_rounds: object) -> object:
        """
        Plot statistics data.

        :param nb_or_rounds: Plot id.
        :param xdata: X-axis of the plot.
        :param ydata: Y-axis of the plot.
        :param color: List of colors to match the one of the Treatment tab.
        :param title: List of titles to match the one of the Treatment tab.
        :param x_label: Legend of the X-axis.
        :param y_label: Legend of the Y-axis.
        :param legend: List of legends to match the one of the Treatment tab.
        :param graph_id: ID of the graph to differentiate each subplot.
        """

        self.plot_id += 1

        # Add new axes only when the function is call again
        if self.plot_id <= graph_id:
            # Organize the different subplot automatically
            self.ax = self.figure.add_subplot(int(f"2{AMOUNT_OF_GRAPHS}{self.plot_id}"))

            # Store all the axes in a list to use the right one automatically
            self.axes_list.append(self.ax)

        # Call the right ax in function of the id that is calling the function
        self.ax = self.axes_list[graph_id - 1]

        # Set the different parameters to the figure
        self.ax.grid(True)
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_xscale('log')

        try:

            # Browse the new data to plot
            for data in range(int(nb_or_rounds - 1), len(xdata)):
                sns.scatterplot(x=xdata[data], y=ydata[data], ax=self.ax, color=color[data], label=legend[data])

            # Plot a vertical line at the noise floor mean value
            # self.ax.axvline(x=mean_noise_floor, color="black", linestyle="--", label="Mean floor value")

        except (len(xdata) != len(ydata) and len(xdata) == 0) or Exception as e:
            raise Exception(f"Unexpected Error: {e}") from e

        self.canvas.draw()

    ####################################################################################################################
    ####################################################################################################################

    def compare_pos_plot_calibrator(self, xdata: object, ydata: object, color: object, title: object, x_label: object,
                             y_label: object,
                             legend: object,
                             graph_id: object,
                             nb_or_rounds: object) -> object:
        """
        Plot statistics data.

        :param nb_or_rounds: Plot id.
        :param xdata: X-axis of the plot.
        :param ydata: Y-axis of the plot.
        :param color: List of colors to match the one of the Treatment tab.
        :param title: List of titles to match the one of the Treatment tab.
        :param x_label: Legend of the X-axis.
        :param y_label: Legend of the Y-axis.
        :param legend: List of legends to match the one of the Treatment tab.
        :param graph_id: ID of the graph to differentiate each subplot.
        """

        self.plot_id += 1

        # Add new axes only when the function is call again
        if self.plot_id <= graph_id:
            # Organize the different subplot automatically
            self.ax = self.figure.add_subplot(int(f"1{AMOUNT_OF_GRAPHS_CALIBRATOR}{self.plot_id}"))

            # Store all the axes in a list to use the right one automatically
            self.axes_list.append(self.ax)

        # Call the right ax in function of the id that is calling the function
        self.ax = self.axes_list[graph_id - 1]

        # Set the different parameters to the figure
        self.ax.grid(True)
        self.ax.set_title(title)
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)

        if graph_id == 1:
            self.ax.set_ylim([1 - 0.1 - 0.02, 1 + 0.1 + 0.02])
            # Ajouter deux lignes horizontales : 1+0.1 et 1-0.1
            self.ax.axhline(y=1 + 0.1, color='black', linewidth=2, linestyle='-',
                            label="validity limit")
            self.ax.axhline(y=1 - 0.1, color='black', linewidth=2, linestyle='-')

        else:
            self.ax.set_ylim([1 - 1e-4 - 0.0001, 1 + 1e-4 + 0.0001])
            # Ajouter deux lignes horizontales : 1+1e-4 et 1-1e-4
            self.ax.axhline(y=1 + 1e-4, color='black', linewidth=2, linestyle='-',
                            label="validity limit")
            self.ax.axhline(y=1 - 1e-4, color='black', linewidth=2, linestyle='-')

        try:

            # Browse the new data to plot
            for data in range(int(nb_or_rounds - 1), len(xdata)):
                sns.scatterplot(x=xdata[data], y=ydata[data], ax=self.ax, color=color[data], label=legend[data])

            # Plot a vertical line at the noise floor mean value
            # self.ax.axvline(x=mean_noise_floor, color="black", linestyle="--", label="Mean floor value")

        except (len(xdata) != len(ydata) and len(xdata) == 0) or Exception as e:
            raise Exception(f"Unexpected Error: {e}") from e

        self.canvas.draw()
