# coding: utf-8
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__authors__ = "Grégoire Douillard-Jacq"
__contact__ = "NextStep-ns@outlook.com"
__copyright__ = "ARTEMIS, Côte d'Azur Observatory"
__date__ = "2024-06-17"
__version__ = "1.0.0"
__status__ = "Production"
__privacy__ = "Confidential"

from pandas import DataFrame

from Code.SLS_Interface.graph_manager import GraphManager

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

import os
import sys
import json
import seaborn as sns
from typing import Optional, List
import pandas as pd
import qtawesome as qta
import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QCheckBox, QGroupBox, QLabel, QGridLayout, QLineEdit, QTextEdit, QComboBox,
    QHBoxLayout, QListWidget, QMessageBox, QDialog, QApplication)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from Data_Extraction import DataExtraction
from graph_manager import GraphManager
from header_dictionary import html_dialogbox, add_legends, execute_with_error_handling

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################

# Initialization of constants used by all the classes
LOW_OPD_BANDWIDTH_NOISE_MODEL = [0.0004, 2.5]
LOW_FREQ_BANDWIDTH_NOISE_MODEL = [0.007, 42.5]
LOW_OPD_BANDWIDTH = [0.0004, 0.25]
OPD_BANDWIDTH = [0.25, 12]
LOW_FREQ_BANDWIDTH = [0.007, 5.2]
FREQ_BANDWIDTH = [5.2, 200]
TIME_BANDWIDTH = [0, 100]


class VisualisationTab(QWidget):
    x_axis_scrollbox: QComboBox
    yaxis: DataFrame | None
    data_path: object
    low_yaxis: DataFrame | None
    xaxis: DataFrame | None
    low_xaxis: DataFrame | None

    def __init__(self, initialize_connections=True):
        super().__init__()
        """
        Optional = Union[X, None]
        Any = Any type
        @property: Used on functions to call them as a simple attribute

        :param initialize_connections: Flag to activate VisualisationTab connections only for the VisualisationTab
        """

        # --------------------------------------------------------------------------------------------------------------
        #                                              General configurations
        # --------------------------------------------------------------------------------------------------------------

        # Define the tab name
        self.tab_name: str = 'VisualisationTab'

        # Get screen size
        self.desktop = QApplication.primaryScreen().geometry()

        # Type of record connected to user choices from the UI
        self.record_type: str = 'None'

        # QPR ID connected to user choices from the UI
        self.qpr: Optional[str] = None

        # Pico choice connected to user choices from the UI
        self.pico_choice: Optional[str] = None

        # Canal choice connected to user choices from the UI
        self.canal: Optional[str] = None
        self.canal_calibrator: Optional[str] = None
        self.canal_scan0: Optional[str] = None
        self.canal_scan0P: Optional[str] = None

        # Name of measurements actually plotted
        self.measurement_name: Optional[str] = None

        # Flags
        self.error_flag: Optional[bool] = None
        self.title_measurement_name_flag: bool = False

        # --------------------------------------------------------------------------------------------------------------
        #                                              Data and analysis
        # --------------------------------------------------------------------------------------------------------------

        # VDC values of the selected signal
        self.data_vdc: Optional[pd.DataFrame] = None

        # Dataframe containing all the signals
        self.dataframes: Optional[dict[str, pd.DataFrame]] = None

        # Instance of the DataExtraction class
        self.analysis: Optional[DataExtraction] = None
        self.analysis_calibrator: Optional[DataExtraction] = None
        self.analysis_scan0: Optional[DataExtraction] = None
        self.analysis_scan0P: Optional[DataExtraction] = None

        # Axis that will be plotted
        self.xaxis: Optional[pd.DataFrame] = None
        self.yaxis: Optional[pd.DataFrame] = None
        self.low_xaxis: Optional[pd.DataFrame] = None
        self.low_yaxis: Optional[pd.DataFrame] = None
        self.xaxis_scan0: Optional[pd.DataFrame] = None
        self.yaxis_scan0: Optional[pd.DataFrame] = None
        self.low_xaxis_scan0: Optional[pd.DataFrame] = None
        self.low_yaxis_scan0: Optional[pd.DataFrame] = None
        self.xaxis_scan0P: Optional[pd.DataFrame] = None
        self.yaxis_scan0P: Optional[pd.DataFrame] = None
        self.low_xaxis_scan0P: Optional[pd.DataFrame] = None
        self.low_yaxis_scan0P: Optional[pd.DataFrame] = None
        self.xaxis_calibrator: Optional[pd.DataFrame] = None
        self.yaxis_calibrator: Optional[pd.DataFrame] = None
        self.low_xaxis_calibrator: Optional[pd.DataFrame] = None
        self.low_yaxis_calibrator: Optional[pd.DataFrame] = None
        self.xaxis_tot: Optional[pd.DataFrame] = None
        self.yaxis_tot: Optional[pd.DataFrame] = None

        # --------------------------------------------------------------------------------------------------------------
        #                                              Graphics / Plots
        # --------------------------------------------------------------------------------------------------------------

        # Graphs framework
        self.ax: Optional[Axes] = None
        self.figure: Optional[Figure] = None
        self.canvas: Optional[FigureCanvas] = None

        # Column name to extract from the dataframe
        self.x_axis_data: Optional[str] = None
        self.x_axis_data_spec_freq: Optional[str] = None
        self.x_axis_data_calibrator: Optional[str] = None
        self.x_axis_data_scan0: Optional[str] = None
        self.x_axis_data_scan0P: Optional[str] = None
        self.x_axis_datas: List[str] = ["Temps", "Freq", "DL 5,1", "DL 5,3"]

        # Axis legends in function of the plot parameters defined in the UI
        self.y_axis_legend: Optional[str] = None
        self.y_axis_legend_spec_freq: Optional[str] = None
        self.x_axis_legend: Optional[str] = None
        self.x_axis_legends: List[str] = ["Time (s)", "Freq (Hz)", "SL/nom OPD (m)", "SL/nom OPD (m)"]
        self.y_axis_legends: List[str] = ["Amplitude (mV)", "Amplitude (Vpk)", "SL/Snom amplitude ratio (Vpk/VDC)"]
        self.legends: List[str] = []

        # X-axis bandwidth
        self.xaxis_max_calibrator: Optional[float] = None
        self.xaxis_min_calibrator: Optional[float] = None
        self.low_xaxis_max_calibrator: Optional[float] = None
        self.low_xaxis_min_calibrator: Optional[float] = None
        self.xaxis_max: Optional[float] = None
        self.xaxis_min: Optional[float] = None
        self.low_xaxis_max: Optional[float] = None
        self.low_xaxis_min: Optional[float] = None
        self.xaxis_max_scan0: Optional[float] = None
        self.xaxis_min_scan0: Optional[float] = None
        self.low_xaxis_max_scan0: Optional[float] = None
        self.low_xaxis_min_scan0: Optional[float] = None
        self.xaxis_max_scan0P: Optional[float] = None
        self.xaxis_min_scan0P: Optional[float] = None
        self.low_xaxis_max_scan0P: Optional[float] = None
        self.low_xaxis_min_scan0P: Optional[float] = None

        # Plot of the graph
        self.title: Optional[str] = None
        self.titles: List[str] = []

        # --------------------------------------------------------------------------------------------------------------
        #                                             User Interface (GUI)
        # --------------------------------------------------------------------------------------------------------------

        # Main layout of the VisualisationTab
        self.layout: QGridLayout = QGridLayout()

        # Layout used to contain graphics
        self.layout_graph: QVBoxLayout = QVBoxLayout()

        # Part of the UI where the user select the parameters
        self.parameter_layout: QGridLayout = QGridLayout()
        self.frame_parameter: QGroupBox = QGroupBox("Parameters")

        # Measurement selection widget
        self.measurement_text: QLabel = QLabel("Measurement :")
        self.measurement_scrollbox: QComboBox = QComboBox()

        # Record selection widget
        self.record_text: QLabel = QLabel("Record :")
        self.record_scrollbox: QComboBox = QComboBox()

        # Pico selection widget
        self.pico_text: QLabel = QLabel("Pico :")
        self.pico_scrollbox_visualisationtab: QComboBox = QComboBox()

        # Optional graph's title personalisation widget
        self.title_text: QLabel = QLabel("Title :")
        self.text_scrollbox: QLineEdit = QLineEdit()

        # Canal selection widget
        self.canal_text: QLabel = QLabel("Canal :")
        self.canal_listwidget: QListWidget = QListWidget()

        # Y-axis selection widget
        self.yaxis_choice_text: QLabel = QLabel("Y axis :")
        self.yaxis_scrollbox: QComboBox = QComboBox()

        # X-axis selection widget
        self.x_axis_choice_text: QLabel = QLabel("X axis :")
        self.x_axis_scrollbox: QComboBox = QComboBox()

        # Optional commentary for the plot
        self.text_zone: QTextEdit = QTextEdit()

        # Horizontal layout for the user to select the logarithmic options
        self.hbox_layout: QHBoxLayout = QHBoxLayout()
        self.checkbox_log_text: QLabel = QLabel("Log options :")
        self.checkbox_logx: QCheckBox = QCheckBox("LogX")
        self.checkbox_logy: QCheckBox = QCheckBox("LogY")
        self.checkbox_no_log: QCheckBox = QCheckBox("None")

        # Main button of the VisualisationTab to laucnh the visualisation
        self.button_launch_measurement: QPushButton = QPushButton("Launch Visualisation")

        # Question mark shape's button to help the user
        self.help_button: QPushButton = QPushButton()
        self.help_button.setIcon(qta.icon('fa5s.question-circle'))
        self.help_button.setIconSize(QSize(40, 40))
        self.help_button.setToolTip("Click for help")
        self.help_button.setFixedSize(QSize(40, 40))
        self.help_button.setStyleSheet("QPushButton { background-color: #FFFFFF; border: none; }")

        # --------------------------------------------------------------------------------------------------------------
        #                                                 File paths
        # --------------------------------------------------------------------------------------------------------------

        if getattr(sys, 'frozen', False):
            # If the script is executed via an .exe
            self.base_path = os.path.dirname(sys.executable)  # Executable directory
        else:
            # If the script is run directly from the source code
            self.base_path: str = os.path.abspath(os.path.dirname(__file__))
            self.data_path: str = os.path.join(self.base_path, '..', '..', 'Data', 'Data-manip')

        # Path to JSON file in same directory as .exe file
        json_file_path = os.path.join(self.base_path, 'data_config.json')

        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as json_file:
                config = json.load(json_file)
                self.data_path = config.get("name_data_path")
        else:
            print(f"Error: The configuration file '{json_file_path}' does not exist.")

        if not os.path.exists(self.data_path):
            print(f"Warning: The data path '{self.data_path}' does not exist.")

        execute_with_error_handling(self.setup_ui)
        execute_with_error_handling(self.setup_layout)
        if initialize_connections:
            execute_with_error_handling(self.setup_connections)

    ####################################################################################################################
    ####################################################################################################################

    def setup_ui(self):
        """Set up the user interface elements."""

        # Define spacing ang design of the grid layouts
        self.setStyleSheet("""QGridLayout {background-color: #EEC4C9; color: #EEC4C9 }""")
        self.layout.setContentsMargins(40, 40, 40, 40)
        self.layout.setSpacing(10)
        self.parameter_layout.setContentsMargins(10, 10, 10, 10)

        # Set frame parameter size
        self.frame_parameter.setFixedWidth(self.desktop.width()//4)

        # Modify the button size
        self.button_launch_measurement.adjustSize()

        # Define scroll box and list widgets items
        self.canal_listwidget.setFixedHeight(80)
        self.canal_listwidget.addItems(["Canal A", "Canal B", "Canal C", "Canal D"])
        self.yaxis_scrollbox.addItems(["Fractional amplitude", "Amplitude", "Temporal amplitude"])
        self.x_axis_scrollbox.addItems(["Frequency (Hz)", "Optical Path Difference (OPD)", "Time (s)"])

    ####################################################################################################################
    ####################################################################################################################

    def setup_layout(self):
        """Set up all the widgets on their corresponding layout."""

        # --------------------------------------------------------------------------------------------------------------
        # Parameter layout

        # Add folder widgets to the parameter layout
        self.parameter_layout.addWidget(self.measurement_text, 0, 0)
        self.parameter_layout.addWidget(self.measurement_scrollbox, 0, 1)

        # Add sub folder widgets to the parameter layout
        self.parameter_layout.addWidget(self.record_text, 1, 0)
        self.parameter_layout.addWidget(self.record_scrollbox, 1, 1)

        # Add sub bis folder widgets to the parameter layout
        self.parameter_layout.addWidget(self.pico_text, 2, 0)
        self.parameter_layout.addWidget(self.pico_scrollbox_visualisationtab, 2, 1)

        # Add checkbox widgets to the hbox layout
        self.parameter_layout.addWidget(self.checkbox_log_text, 4, 0)
        self.hbox_layout.addWidget(self.checkbox_logx)
        self.hbox_layout.addWidget(self.checkbox_logy)
        self.hbox_layout.addWidget(self.checkbox_no_log)
        self.parameter_layout.addLayout(self.hbox_layout, 4, 1, 1, 3)

        # Add canal list widget to the parameter layout
        self.parameter_layout.addWidget(self.canal_text, 3, 0)
        self.parameter_layout.addWidget(self.canal_listwidget, 3, 1)

        # Add yaxis widgets to the parameter layout
        self.parameter_layout.addWidget(self.yaxis_choice_text, 5, 0)
        self.parameter_layout.addWidget(self.yaxis_scrollbox, 5, 1)

        # Add x_axis widgets to the parameter layout
        self.parameter_layout.addWidget(self.x_axis_choice_text, 6, 0)
        self.parameter_layout.addWidget(self.x_axis_scrollbox, 6, 1)

        # Add the comment ant title widgets to the parameter layout
        self.parameter_layout.addWidget(self.title_text, 7, 0)
        self.parameter_layout.addWidget(self.text_scrollbox, 7, 1)
        self.parameter_layout.addWidget(self.text_zone, 8, 0, 1, 2)

        # Set the parameter frame to the parameter layout
        self.frame_parameter.setLayout(self.parameter_layout)

        # --------------------------------------------------------------------------------------------------------------
        # Main layout

        # Add the parameter frame, graph layout and button to the layout
        self.layout.addWidget(self.frame_parameter, 0, 0)
        self.layout.addLayout(self.layout_graph, 0, 1, 3, 4)
        self.layout.addWidget(self.button_launch_measurement, 1, 0)
        self.layout.addWidget(self.help_button, 3, 0)

        self.setLayout(self.layout)

    ####################################################################################################################
    ####################################################################################################################

    def setup_connections(self):
        """Set up the signal-slot connections."""

        # --------------------------------------------------------------------------------------------------------------
        # Buttons

        # Connect the main button with the parent function
        self.button_launch_measurement.clicked.connect(self.ask_user_for_plotting_option)

        # Connect the question mark button to the corresponding help dialog box
        self.help_button.clicked.connect(self.help_dialog)

        # --------------------------------------------------------------------------------------------------------------
        # Scroll boxes

        # Update record_scrollbox when selection in measurement_scrollbox have changed
        self.measurement_scrollbox.currentIndexChanged.connect(
            lambda: self.update_data_scrollbox(self.measurement_scrollbox, self.record_scrollbox))

        # Apply these connections only to the visualisation page
        if self.pico_scrollbox_visualisationtab and self.tab_name == 'VisualisationTab':

            # Update log options when temporal signal is selected
            self.yaxis_scrollbox.currentIndexChanged.connect(lambda: self.temporal_signal_init)

            # Update pico_scrollbox_visualisationtab when selection in record_scrollbox have changed
            self.x_axis_scrollbox.currentIndexChanged.connect(
                lambda: self.update_data_scrollbox(self.record_scrollbox, self.pico_scrollbox_visualisationtab))

        # Update pico_scrollbox_visualisationtab when selection in record_scrollbox have changed
        self.record_scrollbox.currentIndexChanged.connect(
            lambda: self.update_data_scrollbox(self.record_scrollbox, self.pico_scrollbox_visualisationtab))

        self.add_subfolder_names_to_scrollbox(self.data_path, self.measurement_scrollbox)

        # --------------------------------------------------------------------------------------------------------------
        # Logarithmic check boxes

        # Connect checkbox to plot settings
        self.checkbox_logx.stateChanged.connect(self.update_checkboxes)
        self.checkbox_logy.stateChanged.connect(self.update_checkboxes)
        self.checkbox_no_log.stateChanged.connect(self.update_checkboxes)

    ####################################################################################################################
    ####################################################################################################################

    def help_dialog(self):
        """
        Creation of the help dialog box.
        :return: int
        """
        # Create a QDialog instance
        dialog = QDialog(self)
        dialog.setWindowTitle("Help")
        dialog.setGeometry(150, 150, 400, 300)

        layout = QVBoxLayout()

        # Add a label with help text
        help_text = html_dialogbox(self.tab_name)
        text_edit = QTextEdit()
        text_edit.setHtml(help_text)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        # Add a Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.setLayout(layout)
        dialog.exec()  # Show the dialog modally

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def update_data_scrollbox(self, initial_scroll_box, final_scroll_box):
        """
        Update the data scroll box based on the selected folder.

        :param initial_scroll_box: scrollbox who selection changed
        :param final_scroll_box: scrollbox to update

        :return: int
        """
        path = None

        # Get the initial scroll box current selection
        initial_scroll_box_selection = initial_scroll_box.currentText()

        # Check which scrollbox is called to adapt the path
        if initial_scroll_box == self.measurement_scrollbox:
            path = os.path.join(self.data_path, initial_scroll_box_selection)
        elif initial_scroll_box == self.record_scrollbox:
            path = os.path.join(self.data_path, self.measurement_scrollbox.currentText(), initial_scroll_box_selection)
        self.add_subfolder_names_to_scrollbox(path, final_scroll_box)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def update_checkboxes(self):
        """
        Define the rules of selection for checkboxes.

        :return: int
        """
        # While None is chosen, set the other options to False
        if self.checkbox_no_log.isChecked():
            self.checkbox_logx.setChecked(False)
            self.checkbox_logy.setChecked(False)
            self.checkbox_no_log.setChecked(True)
        elif not self.checkbox_no_log.isChecked():
            if not self.checkbox_logx.isChecked() and not self.checkbox_logy.isChecked():
                self.checkbox_no_log.setChecked(True)
            elif self.checkbox_logx.isChecked() and self.checkbox_logy.isChecked():
                self.checkbox_no_log.setChecked(False)
            elif self.checkbox_logx.isChecked():
                self.checkbox_logx.setChecked(True)
            elif self.checkbox_logy.isChecked():
                self.checkbox_logy.setChecked(True)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def temporal_signal_init(self):
        """
        Define logarithmic rules for a temporal plot.

        :return: int
        """

        # A temporal signal cannot be plot in logarithmic
        if self.yaxis_scrollbox.currentText() == "Temporal amplitude":
            self.checkbox_logx.setChecked(False)
            self.checkbox_logy.setChecked(False)
            self.checkbox_no_log.setChecked(True)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def add_subfolder_names_to_scrollbox(self, path, scrollbox):
        """
        Browse the OS path and add every file to a list to implement them in a scroll box.

        :param path: Path of the files to browse and add in the scroll box
        :param scrollbox: Output scroll box

        :return: list of the folder names found
        """
        folders_name = []

        # Clear to overwrite with new values
        scrollbox.clear()

        # Condition to know wether it's a temporal or frequential signal
        if self.x_axis_scrollbox.currentText() == "Time (s)":
            temporal_or_frequential = "RawData"
        else:
            temporal_or_frequential = "Processed"

        # If there is folder at path
        if os.path.isdir(path):

            # ----------------------------------------------------------------------------------------------------------
            # Measurement

            # Add every measurement folders to the measurement scrollbox
            if scrollbox == self.measurement_scrollbox:
                list_name_documents = os.listdir(path)
                folders_name = [name for name in list_name_documents if
                                os.path.isdir(os.path.join(path, name))]

            # ----------------------------------------------------------------------------------------------------------
            # Record

            # Add every record's type to the record scrollbox
            elif scrollbox == self.record_scrollbox:
                list_name_documents = os.listdir(path)

                # Access directly datas by adding the corresponding folder name to the path (RawData or Processed)
                if temporal_or_frequential in list_name_documents:
                    path = os.path.join(path, temporal_or_frequential)
                    if os.path.isdir(path):
                        list_name_documents = os.listdir(path)
                        folders_name = [name for name in list_name_documents if
                                        os.path.isfile(os.path.join(path, name))]
                else:
                    folders_name = [name for name in list_name_documents if
                                    os.path.isdir(os.path.join(path, name))]

            # ----------------------------------------------------------------------------------------------------------
            # Picoscope

            # Add the pico files to the pico scroll box
            elif scrollbox == self.pico_scrollbox_visualisationtab:
                path = os.path.join(path, temporal_or_frequential)
                if os.path.isdir(path):
                    list_name_documents = os.listdir(path)
                    folders_name = [name for name in list_name_documents if
                                    os.path.isfile(os.path.join(path, name))]

            # ----------------------------------------------------------------------------------------------------------
            # Scroll box filling

            # Add items to the corresponding output scroll box
            for name in folders_name:
                scrollbox.addItem(str(name))

            return folders_name

        else:
            print(f"The path {path} doesn't exist")
            scrollbox.addItem("This folder doesn't contain sub-folders.")
            return []

    ####################################################################################################################
    ####################################################################################################################

    def construct_final_path(self):
        """
        Construct the final path based on the scrollbox selections.
        :return: String corresponding to the plotted file path.
        """

        # Transform into a list to be able to use append() method
        path_parts = [self.data_path]

        # --------------------------------------------------------------------------------------------------------------
        # Condition to know wether it's a temporal or frequential signal

        if self.x_axis_scrollbox.currentText() == "Time (s)":
            temporal_or_frequential = "RawData"
        else:
            temporal_or_frequential = "Processed"

        # --------------------------------------------------------------------------------------------------------------
        # Measurement scroll box

        # Add the correct measurement to the final path
        if self.measurement_scrollbox.currentText() != "This folder doesn't exist":
            selection = self.measurement_scrollbox.currentText()

            # Check in case the csv file isn't at the right place (just after the measurement name)
            if selection.endswith('.csv'):
                path_parts.append(temporal_or_frequential)

            # Append the measurement name to the path
            path_parts.append(selection)

        # --------------------------------------------------------------------------------------------------------------
        # Record scroll box

        # Add the correct measurement to the final path
        if self.record_scrollbox.currentText() != "This folder doesn't exist":
            sub_selection = self.record_scrollbox.currentText()

            # Check in case the csv file isn't at the right place (just after the record name)
            if sub_selection.endswith('.csv'):
                path_parts.append(temporal_or_frequential)

            # Append the record name to the path
            path_parts.append(sub_selection)

        # --------------------------------------------------------------------------------------------------------------
        # Pico scroll box

        # Add the correct measurement to the final path
        if self.pico_scrollbox_visualisationtab.currentText() != "This folder doesn't exist":
            sub_bis_selection = self.pico_scrollbox_visualisationtab.currentText()

            # Check that the csv is store at the right place in the database
            if sub_bis_selection.endswith('.csv'):
                path_parts.append(temporal_or_frequential)

            # Append the pico file name to the path
            path_parts.append(sub_bis_selection)

        # --------------------------------------------------------------------------------------------------------------

        # Join all different parts of the path
        final_path = os.path.join(*path_parts)
        return final_path

    ####################################################################################################################
    ####################################################################################################################

    def link_vdc_to_canal(self, path: object) -> object:
        """
        Get the fractional amplitude.

        :param path: The path of the file that is plotted
        :return: The columns in FA which correspond to the canal and file selection
        """
        self.dataframes = {}

        # --------------------------------------------------------------------------------------------------------------
        # Get file names

        # Get the names of the pico files
        parent_path = os.path.join(path, '..')
        list_name_documents = os.listdir(parent_path)
        list_files = [file for file in list_name_documents if file.endswith('.csv')]

        # --------------------------------------------------------------------------------------------------------------
        # Dataframe creation

        # Read every pico CSV and create a df for each with 4 columns each
        for file in list_files:
            file_path = os.path.join(parent_path, file)
            if "RawData" in file_path:
                df = pd.read_csv(file_path, usecols=["Canal A", "Canal B", "Canal C", "Canal D"], delimiter=";")
                # Delete the first line
                df = df.iloc[1:].reset_index(drop=True)
                # Replace commas with dots
                df = df.replace(',', '.', regex=True)
                # Convert to numeric
                df = df.apply(pd.to_numeric, errors='coerce')

            elif "28092023-Testscancalib" in file_path :
                df = pd.read_csv(file_path, usecols=["Canal A"], delimiter=" ")
                self.canal = "Canal A"

            elif "20231011-TestScan0-All" in file_path:
                df = pd.read_csv(file_path, usecols=["Canal B"], delimiter=" ")
                self.canal = "Canal B"

            elif "20231017-TestScan0P-All" in file_path:
                df = pd.read_csv(file_path, usecols=["Canal B"], delimiter=" ")
                self.canal = "Canal B"

            else :
                df = pd.read_csv(file_path, usecols=["Canal A", "Canal B", "Canal C", "Canal D"], delimiter=" ")
            self.dataframes[file] = df

        # --------------------------------------------------------------------------------------------------------------
        # Get VDC

        # Read the corresponding VDC file
        self.data_vdc = pd.read_csv(os.path.join(parent_path, '..', 'VDC', 'VDC.csv'), delimiter=" ")

        # Get VDC values
        if 'VDC' not in self.data_vdc.columns:
            raise ValueError("There is not 'VDC' column in VDC.csv.")
        vdc_values = self.data_vdc['VDC'].values

        # --------------------------------------------------------------------------------------------------------------
        # Apply fractional amplitude calculation

        # Get every canal columns
        total_columns = sum(df.shape[1] for df in self.dataframes.values())
        if len(vdc_values) != total_columns:
            raise ValueError(f"The length of VDC {len(vdc_values)} must be equal to the number of columns in dataframes"
                              f"{total_columns}.")

        # Multiply every column of the dataframe with the corresponding VDC value
        vdc_index = 0
        for key, df in self.dataframes.items():
            for i, col in enumerate(df.columns):
                df[col] = df[col]/vdc_values[vdc_index]
                vdc_index += 1

        # ----------------------------------------------------------------------------------------------------------
        # Output files

        # Get the file name which is actually plot and search for the corresponding dataframe in the dictionary
        file_plotted = os.path.basename(path)
        file_vdc_df = self.dataframes[file_plotted][self.canal]

        return file_vdc_df

    ####################################################################################################################
    ####################################################################################################################

    def add_title(self, ax, scroll_box):
        """
        Add a title to the plot.

        :param scroll_box: Scroll where the date is displayed in the name of the selected folder.
        :param ax: Name of the figure subplot
        :return: int
        """
        personalised_title = self.text_scrollbox.text()

        # --------------------------------------------------------------------------------------------------------------

        # If overlay, display the measurement names of all measurement plotted together
        if self.title_measurement_name_flag and scroll_box.currentText()[4:8] not in self.measurement_name:
            self.measurement_name += " & " + scroll_box.currentText()[4:8]
        else:
            self.measurement_name = scroll_box.currentText()[4:8]

        # --------------------------------------------------------------------------------------------------------------

        # Update the title if function of the type of the plot and if an optional title have been added
        title_prefix = "Temporal representation" if self.x_axis_legend == "Time (s)" else "Frequency spectrum"
        title_suffix = personalised_title if personalised_title else ""

        self.title = ax.set_title(f"{title_prefix}, ZIFO, record: {self.measurement_name}{title_suffix}", fontsize=18)
        self.titles.append(self.title)

        # --------------------------------------------------------------------------------------------------------------

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_graph_parameters(self, final_path: object, pico: Optional = None) -> object:
        """
        Get graph's parameters from GraphManager class.

        :param final_path: Path of the plotted signal.
        :param pico: Pico chose by the user.

        :return: int
        """

        # --------------------------------------------------------------------------------------------------------------
        # Instance creation

        graph_manager: GraphManager = GraphManager(self.x_axis_scrollbox, self.yaxis_scrollbox, self.analysis,
                                                   self.canal_listwidget, self.x_axis_legends, self.x_axis_legend,
                                                   self.y_axis_legends, self.y_axis_legend, self.x_axis_datas,
                                                   self.x_axis_data, self.pico_choice, self.qpr, self.xaxis_min,
                                                   self.xaxis_max, self.low_xaxis_min, self.low_xaxis_max, self.xaxis,
                                                   self.yaxis, self.low_xaxis, self.low_yaxis, OPD_BANDWIDTH,
                                                   FREQ_BANDWIDTH, TIME_BANDWIDTH, LOW_OPD_BANDWIDTH,
                                                   LOW_FREQ_BANDWIDTH)

        # --------------------------------------------------------------------------------------------------------------
        # Get parameter values

        # Get canal value
        self.canal = graph_manager.get_canal(from_get_graph_parameters=True)

        # Get X-axis information
        (self.xaxis_min, self.xaxis_max, self.low_xaxis_min, self.low_yaxis_min, self.x_axis_legend,
         self.x_axis_data) = graph_manager.choose_x_axis(final_path)

        # --------------------------------------------------------------------------------------------------------------
        # Get pico information

        if self.tab_name == 'VisualisationTab':

            # Get pico value
            self.pico_choice = execute_with_error_handling(graph_manager.legend_pico_choice,
                                                           self.pico_scrollbox_visualisationtab)

            # Get qpr value by specifying the pico choice
            self.qpr = graph_manager.legend_qpr(final_path, pico=self.pico_choice)

        else:

            # Get qpr value without specifying the pico choice
            self.qpr = graph_manager.legend_qpr(final_path, pico)

        # --------------------------------------------------------------------------------------------------------------

        # Get Y-axis information
        file_vdc_df = self.link_vdc_to_canal(final_path)
        print("final_path", final_path)
        (self.xaxis, self.yaxis, self.low_xaxis, self.low_yaxis,
         self.y_axis_legend) = execute_with_error_handling(graph_manager.choose_y_axis, file_vdc_df)

        # --------------------------------------------------------------------------------------------------------------

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_graph_parameters_calibrator(self, final_path: object, pico: Optional = None, canal:str='Canal D') -> object:
        """
        Get graph's parameters from GraphManager class.

        :param final_path: Path of the plotted signal.
        :param pico: Pico chose by the user.
        :param x_axis_scrollbox

        :return: int
        """

        # --------------------------------------------------------------------------------------------------------------
        # Instance creation

        graph_manager: GraphManager = GraphManager(self.x_axis_scrollbox, self.yaxis_scrollbox, self.analysis_calibrator,
                                                   self.canal_listwidget, self.x_axis_legends,
                                                   self.x_axis_legend, self.y_axis_legends,
                                                   self.y_axis_legend, self.x_axis_datas,
                                                   self.x_axis_data_calibrator, self.pico_choice, self.qpr,
                                                   self.xaxis_min_calibrator, self.xaxis_max_calibrator,
                                                   self.low_xaxis_min_calibrator, self.low_xaxis_max_calibrator,
                                                   self.xaxis_calibrator, self.yaxis_calibrator,
                                                   self.low_xaxis_calibrator, self.low_yaxis_calibrator, OPD_BANDWIDTH,
                                                   FREQ_BANDWIDTH, TIME_BANDWIDTH, LOW_OPD_BANDWIDTH, LOW_FREQ_BANDWIDTH)

        # --------------------------------------------------------------------------------------------------------------
        # Get parameter values

        # Get canal value
        self.canal = canal

        # Get X-axis information
        (self.xaxis_min_calibrator, self.xaxis_max_calibrator, self.low_xaxis_min_calibrator,
         self.low_yaxis_min_calibrator, self.x_axis_legend, self.x_axis_data) = graph_manager.choose_x_axis(final_path)

        # --------------------------------------------------------------------------------------------------------------
        # Get pico information


        # Get Y-axis information
        file_vdc_df = self.link_vdc_to_canal(final_path)
        print("final_path", final_path)
        (self.xaxis_calibrator, self.yaxis_calibrator, self.low_xaxis_calibrator, self.low_yaxis_calibrator,
         self.y_axis_legend) = execute_with_error_handling(graph_manager.choose_y_axis, file_vdc_df)

        # --------------------------------------------------------------------------------------------------------------

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_graph_parameters_scan0(self, final_path: object, pico: Optional = None) -> object:
        """
        Get graph's parameters from GraphManager class.

        :param final_path: Path of the plotted signal.
        :param pico: Pico chose by the user.
        :param x_axis_scrollbox

        :return: int
        """

        # --------------------------------------------------------------------------------------------------------------
        # Instance creation

        graph_manager: GraphManager = GraphManager(self.x_axis_scrollbox, self.yaxis_scrollbox, self.analysis_scan0,
                                                   self.canal_listwidget, self.x_axis_legends,
                                                   self.x_axis_legend, self.y_axis_legends,
                                                   self.y_axis_legend, self.x_axis_datas,
                                                   self.x_axis_data_scan0, self.pico_choice, self.qpr,
                                                   self.xaxis_min_scan0, self.xaxis_max_scan0,
                                                   self.low_xaxis_min_scan0, self.low_xaxis_max_scan0,
                                                   self.xaxis_scan0, self.yaxis_scan0,
                                                   self.low_xaxis_scan0, self.low_yaxis_scan0, OPD_BANDWIDTH,
                                                   FREQ_BANDWIDTH, TIME_BANDWIDTH, LOW_OPD_BANDWIDTH, LOW_FREQ_BANDWIDTH)

        # --------------------------------------------------------------------------------------------------------------
        # Get parameter values

        # Get canal value
        self.canal_scan0 = execute_with_error_handling(graph_manager.get_canal)

        # Get X-axis information
        (self.xaxis_min_scan0, self.xaxis_max_scan0, self.low_xaxis_min_scan0,
         self.low_yaxis_min_scan0, self.x_axis_legend, self.x_axis_data) = graph_manager.choose_x_axis(final_path)

        # --------------------------------------------------------------------------------------------------------------
        # Get pico information


        # Get Y-axis information
        file_vdc_df = self.link_vdc_to_canal(final_path)
        print("final_path", final_path)
        (self.xaxis_scan0, self.yaxis_scan0, self.low_xaxis_scan0, self.low_yaxis_scan0,
         self.y_axis_legend) = execute_with_error_handling(graph_manager.choose_y_axis, file_vdc_df)


        # --------------------------------------------------------------------------------------------------------------

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_graph_parameters_scan0P(self, final_path: object, pico: Optional = None) -> object:
        """
        Get graph's parameters from GraphManager class.

        :param final_path: Path of the plotted signal.
        :param pico: Pico chose by the user.
        :param x_axis_scrollbox

        :return: int
        """

        # --------------------------------------------------------------------------------------------------------------
        # Instance creation

        graph_manager: GraphManager = GraphManager(self.x_axis_scrollbox, self.yaxis_scrollbox, self.analysis_scan0P,
                                                   self.canal_listwidget, self.x_axis_legends,
                                                   self.x_axis_legend, self.y_axis_legends,
                                                   self.y_axis_legend, self.x_axis_datas,
                                                   self.x_axis_data_scan0P, self.pico_choice, self.qpr,
                                                   self.xaxis_min_scan0P, self.xaxis_max_scan0P,
                                                   self.low_xaxis_min_scan0P, self.low_xaxis_max_scan0P,
                                                   self.xaxis_scan0P, self.yaxis_scan0P,
                                                   self.low_xaxis_scan0P, self.low_yaxis_scan0P, OPD_BANDWIDTH,
                                                   FREQ_BANDWIDTH, TIME_BANDWIDTH, LOW_OPD_BANDWIDTH, LOW_FREQ_BANDWIDTH)

        # --------------------------------------------------------------------------------------------------------------
        # Get parameter values

        # Get canal value
        self.canal_scan0P = execute_with_error_handling(graph_manager.get_canal)

        # Get X-axis information
        (self.xaxis_min_scan0P, self.xaxis_max_scan0P, self.low_xaxis_min_scan0P,
         self.low_yaxis_min_scan0P, self.x_axis_legend, self.x_axis_data) = graph_manager.choose_x_axis(final_path)

        # --------------------------------------------------------------------------------------------------------------
        # Get pico information


        # Get Y-axis information
        file_vdc_df = self.link_vdc_to_canal(final_path)
        print("final_path", final_path)
        (self.xaxis_scan0P, self.yaxis_scan0P, self.low_xaxis_scan0P, self.low_yaxis_scan0P,
         self.y_axis_legend) = execute_with_error_handling(graph_manager.choose_y_axis, file_vdc_df)

        # --------------------------------------------------------------------------------------------------------------

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def show_spectrum(self):
        """
        Plot the spectrum on the processing tab.
        Call every function to get the final plot

        :return: int
        """

        # Check possible errors before launching the treatement
        selected_items = self.canal_listwidget.selectedItems()

        # ==============================================================================================================
        # Error handling

        if not selected_items:
            QMessageBox.warning(self, "No Canal Selected", "Please select a canal before plotting.")
            return

        # --------------------------------------------------------------------------------------------------------------

        if ((self.x_axis_scrollbox.currentText() == "Time (s)" and
            self.yaxis_scrollbox.currentText() != "Temporal amplitude") or
                (self.x_axis_scrollbox.currentText() != "Time (s)" and
                 self.yaxis_scrollbox.currentText() == "Temporal amplitude")):

            QMessageBox.warning(self, "Wrong temporal parameters",
                                "To plot the temporal signal, select Xaxis=Time (s) & Yaxis=Temporal amplitude.")
            return

        # --------------------------------------------------------------------------------------------------------------

        if ((self.x_axis_scrollbox.currentText() == "Time (s)") and
                (self.checkbox_logx.isChecked() or self.checkbox_logy.isChecked())):
            QMessageBox.warning(self, "Logarithmic axis",
                                "You can't plot in temporal with logarithmic axis.")
            return

        # ==============================================================================================================
        # Function calls

        # Load and extract data
        final_path = execute_with_error_handling(self.construct_final_path)
        self.analysis = DataExtraction(final_path)
        execute_with_error_handling(self.analysis.load_data, final_path)
        execute_with_error_handling(self.analysis.extract_data)

        self.get_graph_parameters(final_path)
        self.record_type = execute_with_error_handling(add_legends, final_path)

        # Clear previous plot if exists
        if self.layout_graph.count() > 0:
            for i in reversed(range(self.layout_graph.count())):
                widget = self.layout_graph.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

        # ==============================================================================================================
        # Inside function

        # Intern function to create and add plot
        def add_plot(x_data, y_data, x_label, y_label, canal):
            """
            Create and add a plot to the layout.

            Parameters:
            x_data: Data for the x-axis.
            y_data: Data for the y-axis.
            x_label: Label for the x-axis.
            y_label: Label for the y-axis.
            canal: Canal information.
            """
            self.create_plot(x_data, y_data, x_label, y_label, canal)
            self.layout_graph.addWidget(NavigationToolbar(self.canvas, self))

        # ==============================================================================================================
        # Inside function's call

        # Plot amplitude spectrum
        if self.x_axis_scrollbox.currentText() == "Time (s)" :
            add_plot(self.xaxis, self.yaxis, self.x_axis_legend, self.y_axis_legend, self.canal)

        else :
            self.xaxis_tot = np.concatenate((self.low_xaxis, self.xaxis[1:]))
            self.yaxis_tot = np.concatenate((self.low_yaxis, self.yaxis[1:]))
            add_plot(self.xaxis_tot, self.yaxis_tot, self.x_axis_legend, self.y_axis_legend, self.canal)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def create_plot(self, xaxis, yaxis, xlabel, ylabel, canal):
        """
        Create the basic elements of a graph such as title, labels and axis...

        :param xaxis: x data to plot
        :param yaxis: y data to plot
        :param xlabel: x-axis legend
        :param ylabel: y-axis legend
        :param canal: Canal which is actually plotted
        :return: int
        """

        # --------------------------------------------------------------------------------------------------------------
        # Graph initialization

        # Create a new Figure instance
        self.figure = Figure(figsize=(20, 16))
        self.ax = self.figure.gca()
        self.canvas = FigureCanvas(self.figure)
        self.ax.grid(True)
        self.layout_graph.addWidget(self.canvas)
        self.ax.set_xlabel(xlabel, fontsize=14)
        self.ax.set_ylabel(ylabel, fontsize=14)
        if xlabel != "Time (s)":
            self.ax.set_ylim(bottom=10 ** -8)

        self.figure.subplots_adjust(bottom=0.2)
        self.add_title(self.ax, self.record_scrollbox)

        # --------------------------------------------------------------------------------------------------------------
        # Logarithmic initialization

        if self.checkbox_logy.isChecked():
            self.ax.set_yscale('log')
        if self.checkbox_logx.isChecked():
            self.ax.set_xscale('log')

        # --------------------------------------------------------------------------------------------------------------
        # Graph style

        sns.set_style("ticks")
        print(f"Desktop size : ({self.desktop.width()}; {self.desktop.height()})")

        if self.desktop.width() <= 1800 or self.desktop.height() <= 900:
            sns.set_context("notebook", font_scale=0.9)
        else:
            sns.set_context("talk", font_scale=0.9)

        # --------------------------------------------------------------------------------------------------------------
        # Plot data

        # Plot scatterplot with custom styling
        sns.lineplot(x=xaxis, y=yaxis, ax=self.ax, alpha=1, linewidth=0.7,
                     label=[self.record_type + ", " + self.qpr + ", " + canal + ", "
                            + self.record_scrollbox.currentText()[:8]])

        # Add the canvas to the layout_graph
        self.canvas.draw()

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def ask_user_for_plotting_option(self):
        """
        Display a dialog box asking the user whether to clear the existing plot or overlay a new plot.
        Returns True if the user chooses to clear the plot, False if the user chooses to overlay the plot.
        """

        # --------------------------------------------------------------------------------------------------------------

        if self.error_flag:
            QMessageBox.warning(self, "OPD parameters", "Can only plot Scan5.1 & 5.3 in function of OPD!")
            return

        # --------------------------------------------------------------------------------------------------------------
        # The n>1 time the visualisation is launched options

        if self.layout_graph.count() > 0:

            # Create an error message box
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle('Plotting Options')
            msg_box.setText("Do you want to clear the existing plot or overlay the new plot with a different color?")
            clear_plot_button = msg_box.addButton('Clear Plot', QMessageBox.YesRole)
            overlay_plot_button = msg_box.addButton('Overlay Plot', QMessageBox.NoRole)
            msg_box.exec()

            # If clear is selected, the new plot will overdraw the old one
            if msg_box.clickedButton() == clear_plot_button:
                self.show_spectrum()

            # If overlay is selected, the new plot will be overlaid with the old one
            elif msg_box.clickedButton() == overlay_plot_button:
                self.overlay_plot()

        # --------------------------------------------------------------------------------------------------------------
        # The n=0 time is calling show_spectrum() to simply plot the selected signal

        else:
            self.show_spectrum()

    ####################################################################################################################
    ####################################################################################################################

    def overlay_plot(self):
        """
        Overlay a new plot on the existing plot with a different color.
        :return: int
        """

        # --------------------------------------------------------------------------------------------------------------
        # Get parameters

        # Load and extract data
        final_path = execute_with_error_handling(self.construct_final_path)
        self.analysis = DataExtraction(final_path)
        execute_with_error_handling(self.analysis.load_data, final_path)
        execute_with_error_handling(self.analysis.extract_data)

        # Get the canal, linked data and legends
        self.get_graph_parameters(final_path)
        self.record_type = execute_with_error_handling(add_legends, final_path)

        # ==============================================================================================================

        def add_overlay_plot(x_data, y_data, canal):
            """
            Create and add an overlay plot to the existing plot.

            Parameters:
            x_data: Data for the x-axis.
            y_data: Data for the y-axis.
            canal: Canal information.
            """
            sns.lineplot(x=x_data, y=y_data, ax=self.ax, alpha=1, linewidth=0.7,
                         label=[self.record_type + ", " + self.qpr + ", " + canal + ", "
                                + self.record_scrollbox.currentText()[:8]])

            self.canvas.draw()

        # ==============================================================================================================

        # Concatenate low and high frequencies to obtain a complete x-axis.
        self.xaxis_tot = np.concatenate((self.low_xaxis, self.xaxis[1:]))
        # Concatenate the data associated with low and high frequencies to obtain the complete y-axis.
        self.yaxis_tot = np.concatenate((self.low_yaxis, self.yaxis[1:]))

        add_overlay_plot(self.xaxis_tot, self.yaxis_tot, self.canal)

        return 0
