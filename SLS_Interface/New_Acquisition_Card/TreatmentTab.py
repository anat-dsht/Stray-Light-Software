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
------------------------------------------------------------------------------------------------------------------------
"""
from numpy import ndarray
from statsmodels.nonparametric.bandwidths import bw_scott
from Code.SLS_Interface.Data_Extraction import DataExtraction
from Code.SLS_Interface.VisualisationTab import LOW_OPD_BANDWIDTH_NOISE_MODEL
from Code.SLS_Interface.comparison_5153 import ScansComparison
import pandas as pd
import os
import re
import seaborn as sns
import statsmodels.api as sm
import random
from scipy.signal import find_peaks, peak_widths
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter
from pathlib import Path
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import Signal
import platform
import subprocess
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from header_dictionary import removewidget, printing
from VisualisationTab import (VisualisationTab, add_legends, OPD_BANDWIDTH, FREQ_BANDWIDTH, LOW_OPD_BANDWIDTH,
                              LOW_FREQ_BANDWIDTH, LOW_FREQ_BANDWIDTH_NOISE_MODEL, LOW_OPD_BANDWIDTH_NOISE_MODEL)
from Data_Extraction import DataExtraction
from data_analysis import DataAnalysis
from peak_clustering import PeakClustering
from comparison_5153 import ScansComparison
from typing import List, Optional

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################

# Initialization of constants used by all the classes
POINTS_CALCUL_BARYCENTRE = 3
X_CONDITION_BARYCENTER_COMPARISON = 5e-4  #on aimerait plutôt 1e-4
Y_CONDITION_BARYCENTER_COMPARISON: float = 0.2

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


class StrayLightTreatment(VisualisationTab):
    record_type: str
    analysis: DataExtraction
    analysis_dark: DataExtraction
    analysis_noscan: DataExtraction
    analysis_scan0: DataExtraction
    analysis_scan0P: DataExtraction
    analysis_spec_freq: DataExtraction
    barycenter_processor: ScansComparison
    barycenter_processor_spec_freq: ScansComparison
    barycenters_list: list[ndarray]
    scans_done: object
    event_occurred: Signal = Signal()

    def __init__(self, initialize_connections=False):
        super().__init__(initialize_connections=initialize_connections)
        print("Initializing StrayLightTreatment class")

        # --------------------------------------------------------------------------------------------------------------
        #                                                 Class instances
        # --------------------------------------------------------------------------------------------------------------

        # Instance of the PeakClustering class
        self.peak_clustering: Optional[PeakClustering] = None
        self.peak_clustering_spec_freq: Optional[PeakClustering] = None
        self.peak_clustering_noscan: Optional[PeakClustering] = None
        self.peak_clustering_scan0: Optional[PeakClustering] = None
        self.peak_clustering_scan0P: Optional[PeakClustering] = None

        # Instance of the ScansComparison class
        self.barycenter_processor: Optional[ScansComparison] = None
        self.barycenter_processor_spec_freq: Optional[ScansComparison] = None

        # Instance of the DataAnalysis class
        self.data_analysis_tab: DataAnalysis = DataAnalysis()

        # --------------------------------------------------------------------------------------------------------------
        #                                                 Noise floor
        # --------------------------------------------------------------------------------------------------------------

        # Noise floor X&Y components
        self.noise_floor_values: Optional[np.ndarray] = None
        self.noise_floor_freqs: Optional[np.ndarray] = None
        self.log_fit_amplitude_optimal: Optional[np.ndarray] = None
        self.fit_amplitude_optimal: Optional[np.ndarray] = None
        self.noise_floor_values_dark: Optional[np.ndarray] = None
        self.noise_floor_freqs_dark: Optional[np.ndarray] = None
        self.log_fit_amplitude_optimal_dark: Optional[np.ndarray] = None
        self.fit_amplitude_optimal_dark: Optional[np.ndarray] = None
        self.noise_floor_values_noscan: Optional[np.ndarray] = None
        self.noise_floor_freqs_noscan: Optional[np.ndarray] = None
        self.log_fit_amplitude_optimal_noscan: Optional[np.ndarray] = None
        self.fit_amplitude_optimal_noscan: Optional[np.ndarray] = None
        self.noise_floor_values_scan0: Optional[np.ndarray] = None
        self.noise_floor_freqs_scan0: Optional[np.ndarray] = None
        self.log_fit_amplitude_optimal_scan0: Optional[np.ndarray] = None
        self.fit_amplitude_optimal_scan0: Optional[np.ndarray] = None
        self.noise_floor_values_scan0P: Optional[np.ndarray] = None
        self.noise_floor_freqs_scan0P: Optional[np.ndarray] = None
        self.log_fit_amplitude_optimal_scan0P: Optional[np.ndarray] = None
        self.fit_amplitude_optimal_scan0P: Optional[np.ndarray] = None

        # --------------------------------------------------------------------------------------------------------------
        #                                                Clusters
        # --------------------------------------------------------------------------------------------------------------

        # Clusters for barycenter calculation (3 points only)
        self.barycenter_clusters_list: List[np.ndarray] = []
        self.barycenter_clusters: Optional[np.ndarray] = None
        self.low_barycenter_clusters_list: List[np.ndarray] = []
        self.low_barycenter_clusters: Optional[np.ndarray] = None

        self.barycenter_clusters_list_dark: List[np.ndarray] = []
        self.barycenter_clusters_dark: Optional[np.ndarray] = None
        self.low_barycenter_clusters_list_dark: List[np.ndarray] = []
        self.low_barycenter_clusters_dark: Optional[np.ndarray] = None
        self.barycenter_clusters_list_noscan: List[np.ndarray] = []
        self.barycenter_clusters_noscan: Optional[np.ndarray] = None
        self.low_barycenter_clusters_list_noscan: List[np.ndarray] = []
        self.low_barycenter_clusters_noscan: Optional[np.ndarray] = None
        self.barycenter_clusters_list_scan0: List[np.ndarray] = []
        self.barycenter_clusters_scan0: Optional[np.ndarray] = None
        self.low_barycenter_clusters_list_scan0: List[np.ndarray] = []
        self.low_barycenter_clusters_scan0: Optional[np.ndarray] = None
        self.barycenter_clusters_list_scan0P: List[np.ndarray] = []
        self.barycenter_clusters_scan0P: Optional[np.ndarray] = None
        self.low_barycenter_clusters_list_scan0P: List[np.ndarray] = []
        self.low_barycenter_clusters_scan0P: Optional[np.ndarray] = None

        # Clusters to get whole peaks shapes
        self.full_clusters_list: List[np.ndarray] = []
        self.full_clusters_peaks: Optional[np.ndarray] = None
        self.low_full_clusters_list: List[np.ndarray] = []
        self.low_full_clusters_peaks: Optional[np.ndarray] = None
        self.full_clusters_list_dark: List[np.ndarray] = []
        self.full_clusters_peaks_dark: Optional[np.ndarray] = None
        self.low_full_clusters_list_dark: List[np.ndarray] = []
        self.low_full_clusters_peaks_dark: Optional[np.ndarray] = None
        self.full_clusters_list_noscan: List[np.ndarray] = []
        self.full_clusters_peaks_noscan: Optional[np.ndarray] = None
        self.low_full_clusters_list_noscan: List[np.ndarray] = []
        self.low_full_clusters_peaks_noscan: Optional[np.ndarray] = None
        self.full_clusters_list_scan0: List[np.ndarray] = []
        self.full_clusters_peaks_scan0: Optional[np.ndarray] = None
        self.low_full_clusters_list_scan0: List[np.ndarray] = []
        self.low_full_clusters_peaks_scan0: Optional[np.ndarray] = None
        self.full_clusters_list_scan0P: List[np.ndarray] = []
        self.full_clusters_peaks_scan0P: Optional[np.ndarray] = None
        self.low_full_clusters_list_scan0P: List[np.ndarray] = []
        self.low_full_clusters_peaks_scan0P: Optional[np.ndarray] = None

        # Full clusters of all Stray Light peaks
        self.sl_points_y: Optional[np.ndarray] = None
        self.sl_points_x: Optional[np.ndarray] = None
        self.low_sl_points_y: Optional[np.ndarray] = None
        self.low_sl_points_x: Optional[np.ndarray] = None
        self.indices_to_keep_53: Optional[np.ndarray] = None
        self.low_indices_to_keep_53: Optional[np.ndarray] = None
        self.indices_to_remove_53: Optional[np.ndarray] = None
        self.low_indices_to_remove_53: Optional[np.ndarray] = None

        # --------------------------------------------------------------------------------------------------------------
        #                                                Graph parameters
        # --------------------------------------------------------------------------------------------------------------

        # Get pico number from the file's name selected
        self.pico: Optional[str] = None

        # Signal's color actually plotted
        self.line_color: Optional[str] = None

        # List of all colors plotted on the graph
        self.plot_colors: List[str] = []

        # List of all the necessary labels in the graph. Uses for the DataAnalysis tab.
        self.list_labels: List[str] = []

        # Statistical variables for the DataAnalysis tab
        self.quality_x_SL: List[float] = []
        self.quality_y_SL: List[float] = []
        self.quality_x_perturbation: List[float] = []
        self.quality_y_perturbation: List[float] = []
        self.x_ratio_values: List[float] = []
        self.y_max_values: List[float] = []
        self.x_max_values: List[float] = []
        self.y_ratio_values: List[float] = []
        self.low_quality_x_SL: List[float] = []
        self.low_quality_y_SL: List[float] = []
        self.low_quality_x_perturbation: List[float] = []
        self.low_quality_y_perturbation: List[float] = []
        self.low_x_ratio_values: List[float] = []
        self.low_y_max_values: List[float] = []
        self.low_x_max_values: List[float] = []
        self.low_y_ratio_values: List[float] = []
        self.y_max_values_list: List[ndarray] = []
        self.x_max_values_list: List[ndarray] = []
        self.y_ratio_values_list: List[ndarray] = []
        self.quality_x_SL_list: List[ndarray] = []
        self.quality_y_SL_list: List[ndarray] = []
        self.quality_x_perturbation_list: List[ndarray] = []
        self.quality_y_perturbation_list: List[ndarray] = []
        self.pic_number: List[float] = []
        self.low_pic_number: List[float] = []

        # --------------------------------------------------------------------------------------------------------------
        #                                                Signal data
        # --------------------------------------------------------------------------------------------------------------

        # Selected signal data
        self.freqs: Optional[np.ndarray] = None
        self.data: Optional[np.ndarray] = None
        self.freqs_51: Optional[np.ndarray] = None
        self.data_51: Optional[np.ndarray] = None
        self.low_freqs: Optional[np.ndarray] = None
        self.low_data: Optional[np.ndarray] = None
        self.low_freqs_51: Optional[np.ndarray] = None
        self.low_data_51: Optional[np.ndarray] = None
        self.log_freqs: Optional[np.ndarray] = None
        self.log_data: Optional[np.ndarray] = None
        self.freqs_tot: Optional[np.ndarray] = None
        self.data_tot: Optional[np.ndarray] = None
        self.freqs_dark: Optional[np.ndarray] = None
        self.data_dark: Optional[np.ndarray] = None
        self.low_freqs_dark: Optional[np.ndarray] = None
        self.low_data_dark: Optional[np.ndarray] = None
        self.log_freqs_dark: Optional[np.ndarray] = None
        self.log_data_dark: Optional[np.ndarray] = None
        self.freqs_noscan: Optional[np.ndarray] = None
        self.data_noscan: Optional[np.ndarray] = None
        self.low_freqs_noscan: Optional[np.ndarray] = None
        self.low_data_noscan: Optional[np.ndarray] = None
        self.log_freqs_noscan: Optional[np.ndarray] = None
        self.log_data_noscan: Optional[np.ndarray] = None
        self.freqs_scan0: Optional[np.ndarray] = None
        self.data_scan0: Optional[np.ndarray] = None
        self.low_freqs_scan0: Optional[np.ndarray] = None
        self.low_data_scan0: Optional[np.ndarray] = None
        self.log_freqs_scan0: Optional[np.ndarray] = None
        self.log_data_scan0: Optional[np.ndarray] = None
        self.freqs_scan0P: Optional[np.ndarray] = None
        self.data_scan0P: Optional[np.ndarray] = None
        self.low_freqs_scan0P: Optional[np.ndarray] = None
        self.low_data_scan0P: Optional[np.ndarray] = None
        self.log_freqs_scan0P: Optional[np.ndarray] = None
        self.log_data_scan0P: Optional[np.ndarray] = None

        # --------------------------------------------------------------------------------------------------------------
        #                                                Peaks
        # --------------------------------------------------------------------------------------------------------------

        # Output treated signal of the Scan comparison process
        self.scans_comparison_treated_signal_y: List[List[float]] = []
        self.scans_comparison_treated_signal_x: List[List[float]] = []
        self.scans_comparison_treated_signal_y_51: List[List[float]] = []
        self.scans_comparison_treated_signal_x_51: List[List[float]] = []
        self.low_scans_comparison_treated_signal_y: List[List[float]] = []
        self.low_scans_comparison_treated_signal_x: List[List[float]] = []
        self.low_scans_comparison_treated_signal_y_51: List[List[float]] = []
        self.low_scans_comparison_treated_signal_x_51: List[List[float]] = []
        self.scans_comparison_treated_signal_y_spec_freq: List[List[float]] = []
        self.scans_comparison_treated_signal_x_spec_freq: List[List[float]] = []
        self.low_scans_comparison_treated_signal_y_spec_freq: List[List[float]] = []
        self.low_scans_comparison_treated_signal_x_spec_freq: List[List[float]] = []
        self.scans_comparison_treated_signal_x_tot: List[List[float]] = []
        self.scans_comparison_treated_signal_y_tot: List[List[float]] = []
        self.scans_comparison_treated_signal_x_tot_51: List[List[float]] = []
        self.scans_comparison_treated_signal_y_tot_51: List[List[float]] = []

        # Stray Light peaks barycenter
        self.list_straylight_barycenter51: List[List[float]] = []
        self.list_straylight_barycenter53: List[List[float]] = []
        self.low_list_straylight_barycenter51: List[List[float]] = []
        self.low_list_straylight_barycenter53: List[List[float]] = []
        self.list_straylight_barycenter51_spec_freq: List[List[float]] = []
        self.list_straylight_barycenter53_spec_freq: List[List[float]] = []
        self.low_list_straylight_barycenter51_spec_freq: List[List[float]] = []
        self.low_list_straylight_barycenter53_spec_freq: List[List[float]] = []
        self.list_straylight_max51: List[List[float]] = []
        self.list_straylight_max53: List[List[float]] = []
        self.list_straylight_max51_tot: List[List[float]] = []
        self.list_straylight_max53_tot: List[List[float]] = []
        self.low_list_straylight_max51: List[List[float]] = []
        self.low_list_straylight_max53: List[List[float]] = []

        # Perturbation peak barycenter
        self.list_perturbation_barycenter51: List[List[float]] = []
        self.list_perturbation_barycenter53: List[List[float]] = []
        self.low_list_perturbation_barycenter51: List[List[float]] = []
        self.low_list_perturbation_barycenter53: List[List[float]] = []
        self.list_perturbation_max51: List[List[float]] = []
        self.list_perturbation_max53: List[List[float]] = []
        self.list_perturbation_max51_tot: List[List[float]] = []
        self.list_perturbation_max53_tot: List[List[float]] = []
        self.low_list_perturbation_max51: List[List[float]] = []
        self.low_list_perturbation_max53: List[List[float]] = []
        self.list_straylight_barycenter51_spec_freq: List[List[float]] = []
        self.list_straylight_barycenter53_spec_freq: List[List[float]] = []
        self.low_list_straylight_barycenter51_spec_freq: List[List[float]] = []
        self.low_list_straylight_barycenter53_spec_freq: List[List[float]] = []
        self.list_perturbation_barycenter51_spec_freq: List[List[float]] = []
        self.list_perturbation_barycenter53_spec_freq: List[List[float]] = []
        self.low_list_perturbation_barycenter51_spec_freq: List[List[float]] = []
        self.low_list_perturbation_barycenter53_spec_freq: List[List[float]] = []
        self.list_perturbation_barycenter51_tot_spec_freq: List[List[float]] = []
        self.list_perturbation_barycenter53_tot_spec_freq: List[List[float]] = []
        self.list_straylight_barycenter51_tot_spec_freq: List[List[float]] = []
        self.list_straylight_barycenter53_tot_spec_freq: List[List[float]] = []
        self.list_straylight_barycenter51_tot: List[List[float]] = []
        self.list_straylight_barycenter53_tot: List[List[float]] = []

        # All barycenter storages
        self.barycenters: Optional[np.ndarray] = None
        self.barycenters_list: List[np.ndarray] = []
        self.low_barycenters: Optional[np.ndarray] = None
        self.low_barycenters_list: List[np.ndarray] = []
        self.barycenters_dark: Optional[np.ndarray] = None
        self.barycenters_list_dark: List[np.ndarray] = []
        self.low_barycenters_dark: Optional[np.ndarray] = None
        self.low_barycenters_list_dark: List[np.ndarray] = []
        self.barycenters_noscan: Optional[np.ndarray] = None
        self.barycenters_list_noscan: List[np.ndarray] = []
        self.low_barycenters_noscan: Optional[np.ndarray] = None
        self.low_barycenters_list_noscan: List[np.ndarray] = []
        self.barycenters_scan0: Optional[np.ndarray] = None
        self.barycenters_list_scan0: List[np.ndarray] = []
        self.low_barycenters_scan0: Optional[np.ndarray] = None
        self.low_barycenters_list_scan0: List[np.ndarray] = []
        self.barycenters_scan0P: Optional[np.ndarray] = None
        self.barycenters_list_scan0P: List[np.ndarray] = []
        self.low_barycenters_scan0P: Optional[np.ndarray] = None
        self.low_barycenters_list_scan0P: List[np.ndarray] = []
        self.barycenters_list_tot_dark = []
        self.barycenters_list_tot_noscan = []
        self.barycenters_list_tot_scan0 = []
        self.barycenters_list_tot_scan0P = []

        #Peak width
        self.peak_width_51 = []
        self.peak_width_53 = []
        self.low_peak_width_51 = []
        self.low_peak_width_53 = []

        # Stray Light peaks width
        self.widths_SL_51_tot = []
        self.widths_SL_53_tot = []

        # Perturbation peaks width
        self.widths_perturbation_51_tot = []
        self.widths_perturbation_53_tot = []

        # --------------------------------------------------------------------------------------------------------------
        #                                                User Interface widgets
        # --------------------------------------------------------------------------------------------------------------

        # Buttons
        self.button_treat_signal: QPushButton = QPushButton("Treat signal")
        self.button_access_data: QPushButton = QPushButton("Access data")

        # Interface user widgets
        self.folder_scrollbox_sl: QComboBox = QComboBox()

        # Plotting options
        self.plot_options_layout: QGridLayout = QGridLayout()
        self.checkbox_plot_options_text: QLabel = QLabel("Plot options:")
        self.checkbox_data_plotline: QCheckBox = QCheckBox("Scan 51/53 plot")
        self.checkbox_barycenters: QCheckBox = QCheckBox("Barycenter")
        self.checkbox_treated_signal: QCheckBox = QCheckBox("Treated signal")
        self.checkbox_noise_floor: QCheckBox = QCheckBox("Noise floor")

        # --------------------------------------------------------------------------------------------------------------
        #                                                 Others
        # --------------------------------------------------------------------------------------------------------------

        # Define which tab we are working on
        self.tab_name: str = "TreatmentTab"

        # Flag to identify once the treatment is over
        self.scans_done: bool = False
        self.low_scans_done: bool = False
        self.scans_done_spec_freq: bool = False
        self.low_scans_done_spec_freq: bool = False

        # Call the right function depends on the count value
        self.count: int = 0

        self.perturbation_list_51 = []
        self.perturbation_list_53 = []
        self.straylight_list_51 = []
        self.straylight_list_53 = []


        # --------------------------------------------------------------------------------------------------------------
        #                                                 File paths
        # --------------------------------------------------------------------------------------------------------------

        self.ref_file_path_scan0 = os.path.abspath(os.path.join(self.data_path, "20231011",
                                                                "20231011-TestScan0-All", "Processed"))

        self.ref_file_path_scan0P = os.path.abspath(os.path.join(self.data_path, "20231017",
                                                                 "20231017-TestScan0P-All", "Processed"))

        # --------------------------------------------------------------------------------------------------------------
        #                                                Specific calls
        # --------------------------------------------------------------------------------------------------------------

        self.setup_traitement_ui()

        # Update UI elements initialized in the Parent class
        self.yaxis_scrollbox.clear()
        self.x_axis_scrollbox.clear()
        self.yaxis_scrollbox.addItems(["Fractional amplitude", "Amplitude"])
        self.x_axis_scrollbox.addItems(["Frequency (Hz)", "Optical Path Difference (OPD)"])

        self.setup_connections_sl()

    ####################################################################################################################
    ####################################################################################################################

    def setup_traitement_ui(self):
        """Additional UI setup for the traitement tab."""

        # Add element to the tab layout
        self.layout.addWidget(self.button_treat_signal, 1, 0)
        self.layout.addWidget(self.button_access_data, 2, 0)
        self.layout.addWidget(self.help_button, 3, 0)

        # Add element to the parameter layout
        self.parameter_layout.addWidget(self.folder_scrollbox_sl, 0, 1)
        self.parameter_layout.addWidget(self.checkbox_plot_options_text, 8, 0)

        # Add element to the plotting options layout
        self.plot_options_layout.addWidget(self.checkbox_noise_floor, 0, 0)
        self.plot_options_layout.addWidget(self.checkbox_data_plotline, 0, 1)
        self.plot_options_layout.addWidget(self.checkbox_barycenters, 1, 0)
        self.plot_options_layout.addWidget(self.checkbox_treated_signal, 1, 1)
        self.parameter_layout.addLayout(self.plot_options_layout, 8, 1, 1, 2)

        # Initialize the plotting options checkboxes states
        self.checkbox_noise_floor.setChecked(True)
        self.checkbox_data_plotline.setChecked(True)
        self.checkbox_barycenters.setChecked(True)
        self.checkbox_treated_signal.setChecked(True)

        # Remove element which were defined in the Data Visualisation tab
        removewidget(self.parameter_layout, self.record_text)
        removewidget(self.parameter_layout, self.record_scrollbox)
        removewidget(self.layout, self.button_launch_measurement)
        removewidget(self.parameter_layout, self.measurement_scrollbox)
        removewidget(self.parameter_layout, self.text_zone)

    ####################################################################################################################
    ####################################################################################################################

    def setup_connections_sl(self):
        """Connect signals to slots."""
        print("Setting up connections")

        # Call the main function to process the signal
        self.button_treat_signal.clicked.connect(self.call_functions)

        # Access treated data
        self.button_access_data.clicked.connect(self.analyse_data)

        # User manual button
        self.help_button.clicked.connect(self.help_dialog)

        # Update checkbox states
        self.checkbox_logx.stateChanged.connect(self.update_checkboxes)
        self.checkbox_logy.stateChanged.connect(self.update_checkboxes)
        self.checkbox_no_log.stateChanged.connect(self.update_checkboxes)

        self.fill_scrollbox(self.folder_scrollbox_sl)

    ####################################################################################################################
    ####################################################################################################################

    def on_event_data_analyse(self):
        """
        Create a signal when the process is done to create and display the DataAnalysis tab.

        :return: int
        """
        self.event_occurred.emit()

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def call_functions(self):
        """
        Main function that call all other steps of the process.

        :return: int
        """

        printing("Starting processing of files", 50, "=")

        # Browse scans that fill user choices
        files = ["scan51", "scan53"]
        file_paths = self.process_files(files)

        # Access the dark acquisition corresponding to the user's choice
        new_file_path_dark, _ = self.extract_datas(file_paths)

        # Access the noscan acquisition corresponding to the user's choice
        _, new_file_path_noscan = self.extract_datas(file_paths)

        new_file_path_scan0 = []
        new_file_path_scan0P = []

        # Access the noscan acquisition corresponding to the user's choice
        for file_name in os.listdir(self.ref_file_path_scan0):
            new_file_path_scan0.append(os.path.join(self.ref_file_path_scan0, file_name))
            break

        # Access the noscan acquisition corresponding to the user's choice
        for file_name in os.listdir(self.ref_file_path_scan0P):
            new_file_path_scan0P.append(os.path.join(self.ref_file_path_scan0P, file_name))
            break

        # If it's not the first time the user click on Launch Process button
        if self.count != 0:

            # ----------------------------------------------------------------------------------------------------------
            # Message Box (clear or overlay)

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle('Plotting Options')
            msg_box.setText("Do you want to clear the existing plot or overlay the new plot with a different color?")
            clear_plot_button = msg_box.addButton('Treat a new record', QMessageBox.YesRole)
            overlay_plot_button = msg_box.addButton('Overlay treated records', QMessageBox.NoRole)
            msg_box.exec()

            # ----------------------------------------------------------------------------------------------------------
            # Clear previous plot

            if msg_box.clickedButton() == clear_plot_button:

                # Reset count value as if the user never clicked on Launch Process button before
                self.count = 0

                # Reset the DataAnalysis tab
                self.data_analysis_tab.create_plot()

                # Re-initialize elements when the user want to overdraw a new plot
                self.y_ratio_values = []
                self.low_y_ratio_values = []
                self.y_max_values = []
                self.low_y_max_values = []
                self.x_max_values = []
                self.low_x_max_values = []
                self.y_max_values_list = []
                self.x_max_values_list = []
                self.y_ratio_values_list = []
                self.list_labels = []
                self.low_quality_x_SL = []
                self.low_quality_y_SL = []
                self.quality_x_SL = []
                self.quality_y_SL = []
                self.quality_x_SL_list = []
                self.quality_y_SL_list = []
                self.barycenters_list = []
                self.low_barycenters_list = []
                self.barycenter_clusters_list = []
                self.low_barycenter_clusters_list = []
                self.full_clusters_list = []
                self.low_full_clusters_list = []
                self.plot_colors = []
                self.list_perturbation_barycenter53 = []
                self.low_list_perturbation_barycenter53 = []
                self.list_perturbation_barycenter51 = []
                self.low_list_perturbation_barycenter51 = []
                self.list_straylight_barycenter53 = []
                self.list_straylight_barycenter53_tot = []
                self.low_list_straylight_barycenter53 = []
                self.list_straylight_barycenter51 = []
                self.low_list_straylight_barycenter51 = []
                self.list_straylight_barycenter51_tot = []
                self.legends = []
                self.barycenters_list_dark = []
                self.barycenters_list_tot_dark = []
                self.low_barycenters_list_dark = []
                self.barycenters_list_noscan = []
                self.barycenters_list_tot_noscan = []
                self.low_barycenters_list_noscan = []
                self.barycenters_list_scan0 = []
                self.barycenters_list_tot_scan0 = []
                self.low_barycenters_list_scan0 = []
                self.barycenters_list_scan0 = []
                self.barycenters_list_tot_scan0P = []
                self.low_barycenters_list_scan0P = []
                self.list_perturbation_barycenter53_spec_freq = []
                self.list_perturbation_barycenter53_tot_spec_freq = []
                self.list_perturbation_barycenter51_tot_spec_freq = []
                self.list_straylight_barycenter53_tot_spec_freq = []
                self.list_straylight_barycenter51_tot_spec_freq = []
                self.low_list_perturbation_barycenter53_spec_freq = []
                self.list_perturbation_barycenter51_spec_freq = []
                self.low_list_perturbation_barycenter51_spec_freq = []
                self.list_straylight_barycenter53_spec_freq = []
                self.low_list_straylight_barycenter53_spec_freq = []
                self.list_straylight_barycenter51_spec_freq = []
                self.low_list_straylight_barycenter51_spec_freq = []
                self.perturbation_list_53 = []
                self.perturbation_list_51 = []
                self.straylight_list_51 = []
                self.straylight_list_53 = []
                self.widths_SL_51_tot = []
                self.widths_SL_53_tot = []
                self.widths_perturbation_51_tot = []
                self.widths_perturbation_53_tot = []
                self.list_straylight_max51_tot = []
                self.list_straylight_max53_tot = []

            # ----------------------------------------------------------------------------------------------------------
            # Overlay old and new plots

            elif msg_box.clickedButton() == overlay_plot_button:

                # Keep going for the count value
                self.count = self.count

        # If it's not the first time the user click on Launch Process button
        else:
            self.on_event_data_analyse()

        # ==============================================================================================================
        # Get scans data

        # Process each file and collect barycenter
        for file_path in file_paths:

            print("\n")
            printing(f"Processing file: {os.path.basename(file_path)}", 50, "-")

            # Extract data of the current scan treated
            self.analysis = DataExtraction(file_path)
            self.analysis.load_data(file_path)
            self.analysis.extract_data()

            # Get the canal, linked data and legends for the current scan treated
            self.get_graph_parameters(file_path)
            self.record_type = add_legends(file_path)

            # Apply the treatment method to the right type of data
            if "OPD" not in self.x_axis_legend:
                self.treatment_method(FREQ_BANDWIDTH) & self.low_treatment_method(LOW_FREQ_BANDWIDTH)
            else:
                self.treatment_method(OPD_BANDWIDTH) & self.low_treatment_method(LOW_OPD_BANDWIDTH)

            # Increment count
            self.count += 1

        # Call the function that starts processing the dark acquisition
        self.call_functions_other_spectrum(new_file_path_dark)
        # Call the function that starts processing the noscan acquisition
        self.call_functions_other_spectrum(new_file_path_noscan)
        # Call the function that starts processing the noscan acquisition
        self.call_functions_other_spectrum(new_file_path_scan0)
        # Call the function that starts processing the noscan acquisition
        self.call_functions_other_spectrum(new_file_path_scan0P)

        # ==============================================================================================================
        # Compare both scans
        # Call the scan comparison class
        self.scans_comparison()
        self.low_scans_comparison()

        list_straylight_max51_tot = np.concatenate((self.low_list_straylight_max51[-1],
                                                         self.list_straylight_max51[-1]), axis=0)
        self.list_straylight_max51_tot.append(list_straylight_max51_tot)

        list_straylight_max53_tot = np.concatenate((self.low_list_straylight_max53[-1],
                                                    self.list_straylight_max53[-1]), axis=0)
        self.list_straylight_max53_tot.append(list_straylight_max53_tot)

        list_perturbation_max51_tot = np.concatenate((self.low_list_perturbation_max51[-1],
                                                    self.list_perturbation_max51[-1]), axis=0)
        self.list_perturbation_max51_tot.append(list_perturbation_max51_tot)

        list_perturbation_max53_tot = np.concatenate((self.low_list_perturbation_max53[-1],
                                                    self.list_perturbation_max53[-1]), axis=0)
        self.list_perturbation_max53_tot.append(list_perturbation_max53_tot)

        # Plot the treated signal
        self.overlay_filtered_plot()
        self.scans_done = False
        self.low_scans_done = False

        # ==============================================================================================================
        # Compare both scans with dark acquisition and noscan acquisition
        self.perturbation_list_51, self.perturbation_list_53 = self.compare_barycenters_with_dark_and_noscan()
        self.straylight_list_51, self.straylight_list_53 = self.compare_barycenters_with_scan0_and_scan0P()

        widths_SL_51_tot = []
        widths_SL_53_tot = []
        widths_perturbation_51_tot = []
        widths_perturbation_53_tot = []

        widths_SL_51_tot = self.low_widths_SL_51[-1] + self.widths_SL_51[-1]
        self.widths_SL_51_tot.append(widths_SL_51_tot)

        widths_SL_53_tot = self.low_widths_SL_53[-1] + self.widths_SL_53[-1]
        self.widths_SL_53_tot.append(widths_SL_53_tot)

        widths_perturbation_51_tot = self.low_widths_perturbation_51[-1] + self.widths_perturbation_51[-1]
        self.widths_perturbation_51_tot.append(widths_perturbation_51_tot)

        widths_perturbation_53_tot = self.low_widths_perturbation_53[-1] + self.widths_perturbation_53[-1]
        self.widths_perturbation_53_tot.append(widths_perturbation_53_tot)

        print("\n")
        printing("Treatment successful!", 50, "=")

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def call_functions_other_spectrum(self, files):
        """
        Function that call all other steps of the process for the dark acquisition.

        :param files: List of access paths to dark spectra.

        :return: int
        """

        for file in files:
            print("\n")
            printing(f"Processing file: {os.path.basename(file)}", 50, "-")

            if 'dark' in file.lower():
                # Extract data of the dark acquisition corresponding to the current scan treated
                self.analysis_dark = DataExtraction(file)
                self.analysis_dark.load_data(file)
                self.analysis_dark.extract_data()

                self.file_vdc_df_dark = self.link_vdc_to_canal(file)

                # Apply the treatment method in frequencies
                self.treatment_method_dark(FREQ_BANDWIDTH) & self.low_treatment_method_dark(LOW_FREQ_BANDWIDTH)
                break

            elif 'noscan' in file.lower():
                # Extract data of the dark acquisition corresponding to the current scan treated
                self.analysis_noscan = DataExtraction(file)
                self.analysis_noscan.load_data(file)
                self.analysis_noscan.extract_data()

                self.file_vdc_df_noscan = self.link_vdc_to_canal(file)

                # Apply the treatment method to the right type of data
                self.treatment_method_noscan(FREQ_BANDWIDTH) & self.low_treatment_method_noscan(LOW_FREQ_BANDWIDTH)
                break

            elif 'scan0p' in file.lower():
                # Extract data of the dark acquisition corresponding to the current scan treated
                self.analysis_scan0P = DataExtraction(file)
                self.analysis_scan0P.load_data(file)
                self.analysis_scan0P.extract_data()

                # Get the canal, linked data and legends for the current scan treated
                self.get_graph_parameters_scan0P(file)
                self.record_type = add_legends(file)

                # Apply the treatment method to the right type of data
                if "OPD" not in self.x_axis_legend:
                    self.treatment_method_scan0P(FREQ_BANDWIDTH) & self.low_treatment_method_scan0P(LOW_FREQ_BANDWIDTH)
                else:
                    self.treatment_method_scan0P(OPD_BANDWIDTH) & self.low_treatment_method_scan0P(LOW_OPD_BANDWIDTH)
                break

            else:
                # Extract data of the dark acquisition corresponding to the current scan treated
                self.analysis_scan0 = DataExtraction(file)
                self.analysis_scan0.load_data(file)
                self.analysis_scan0.extract_data()

                # Get the canal, linked data and legends for the current scan treated
                self.get_graph_parameters_scan0(file)
                self.record_type = add_legends(file)

                # Apply the treatment method to the right type of data
                if "OPD" not in self.x_axis_legend:
                    self.treatment_method_scan0(FREQ_BANDWIDTH) & self.low_treatment_method_scan0(LOW_FREQ_BANDWIDTH)
                else:
                    self.treatment_method_scan0(OPD_BANDWIDTH) & self.low_treatment_method_scan0(LOW_OPD_BANDWIDTH)
                break

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def treatment_method_dark(self, bandwidth):
        """
        Process noise floor, clusters, peak detection, barycenter for dark acquisition.

        :param bandwidth: Bandwidth depends on the signal type.

        :return: int
        """

        printing("Starting treatment method dark", 50, "-")

        # Extract the dictionary containing all acquisition data
        data_columns = self.analysis_dark.data_columns

        # Extract frequency values from the dictionary
        self.freqs_dark = np.array(data_columns.get("Freq"))
        mask = (self.freqs_dark >= bandwidth[0]) & (self.freqs_dark <= bandwidth[1])
        self.freqs_dark = self.freqs_dark[mask]

        if self.yaxis_scrollbox.currentText() == "Fractional amplitude":
            # Filtrer les valeurs d'amplitudes dans la dataframe
            self.data_dark = self.file_vdc_df_dark[mask].reset_index(drop=True)

        else:
            # Select the appropriate canal values based on user choice and convert to NumPy array
            if self.canal == "Canal A":
                self.data_dark = np.array(data_columns.get("Canal A"))
            elif self.canal == "Canal B":
                self.data_dark = np.array(data_columns.get("Canal B"))
            elif self.canal == "Canal C":
                self.data_dark = np.array(data_columns.get("Canal C"))
            elif self.canal == "Canal D":
                self.data_dark = np.array(data_columns.get("Canal D"))

            # Apply the mask to self.data_dark
            self.data_dark = self.data_dark[mask]

        # Compute the noise floor of the signal
        self.noise_floor_freqs_dark, self.noise_floor_values_dark = self.riemann_method(bandwidth,
                                                                                        self.freqs_dark, self.data_dark)

        # Compute clusters and detect peaks
        self.get_custering_variables_dark(self.noise_floor_values_dark)

        # Calculate barycenter of the detected peaks
        self.barycenters_list_dark.append(self.barycenters_dark)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def low_treatment_method_dark(self, bandwidth):
        """
        Process noise floor, clusters, peak detection, barycenter for dark acquisition for low frequencies.

        :param bandwidth: Bandwidth depends on the signal type.

        :return: int
        """

        printing("Starting low treatment method dark", 50, "-")

        # Extract the dictionary containing all acquisition data
        data_columns = self.analysis_dark.data_columns

        # Extract frequency values from the dictionary
        self.low_freqs_dark = np.array(data_columns.get("Freq"))
        mask = (self.low_freqs_dark >= bandwidth[0]) & (self.low_freqs_dark <= bandwidth[1])
        self.low_freqs_dark = self.low_freqs_dark[mask]

        if self.yaxis_scrollbox.currentText() == "Fractional amplitude":
            # Filtrer les valeurs d'amplitudes dans la dataframe
            self.low_data_dark = self.file_vdc_df_dark[mask].reset_index(drop=True)

        else:
            if self.canal == "Canal A":
                self.low_data_dark = np.array(data_columns.get("Canal A"))
            if self.canal == "Canal B":
                self.low_data_dark = np.array(data_columns.get("Canal B"))
            if self.canal == "Canal C":
                self.low_data_dark = np.array(data_columns.get("Canal C"))
            if self.canal == "Canal D":
                self.low_data_dark = np.array(data_columns.get("Canal D"))

            self.low_data_dark = self.low_data_dark[mask]

        self.log_data_dark = np.log10(np.where(self.low_data_dark == 0, np.finfo(float).eps, self.low_data_dark))
        self.log_freqs_dark = np.log10(np.where(self.low_freqs_dark == 0, np.finfo(float).eps, self.low_freqs_dark))

        # Compute the noise floor of the signal
        self.log_fit_amplitude_optimal_dark, self.fit_amplitude_optimal_dark = self.create_noise_model(
            LOW_FREQ_BANDWIDTH_NOISE_MODEL, self.low_freqs_dark, self.low_data_dark,
            self.log_freqs_dark, self.log_data_dark)

        # Compute clusters and detect peaks
        self.get_custering_low_variables_dark(self.fit_amplitude_optimal_dark)

        # Calculate barycenter of the detected peaks
        self.low_barycenters_list_dark.append(self.low_barycenters_dark)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_custering_variables_dark(self, noise_floor_values):
        """
        Function linked to the PeakClustering class which create clusters for each peak detected in the dark spectrum.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """

        self.peak_clustering_dark = PeakClustering(self.freqs_dark, self.data_dark)

        xpeak, ypeak, _, _ = self.detect_peaks(noise_floor_values, self.freqs_dark, self.data_dark)
        self.barycenters_dark, self.barycenter_clusters_dark, self.full_clusters_peaks_dark = (
            self.peak_clustering_dark.group_points_into_peaks(xpeak, ypeak))

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_custering_low_variables_dark(self, noise_floor_values):
        """
        Function linked to the PeakClustering class which create clusters for each peak detected in the dark spectrum in
        low frequencies.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """

        self.peak_clustering_dark = PeakClustering(self.low_freqs_dark, self.low_data_dark)

        low_xpeak, low_ypeak, _, _ = self.detect_peaks_above_noise_model(noise_floor_values,
                                                                      self.low_freqs_dark, self.low_data_dark)
        self.low_barycenters_dark, self.low_barycenter_clusters_dark, self.low_full_clusters_peaks_dark = (
            self.peak_clustering_dark.group_points_into_peaks(low_xpeak, low_ypeak))

        return 0

    ####################################################################################################################
    ####################################################################################################################


    def compare_barycenters_with_dark_and_noscan(self):
        """
        Function that compares the position of the barycenters of the perturbation peaks of scan 5.1
        (respectively scan 5.3) with the barycenters of the dark and noscan spectra. If the x-axis deviation condition
        is met, a list storing the position of the dark and noscan barycenters is displayed.

        :returns:
        - self.perturbation_list_51: list of disturbance peak positions in scan 5.1, noscan and/or dark
        - self.perturbation_list_53: list of disturbance peak positions in scan 5.3, noscan and/or dark
        """

        # Initialize the temporary lists that store the barycenters of frequency perturbation peaks for scans 5.1 and 5.3
        list_perturbation_barycenter51_tot_spec_freq = []
        list_perturbation_barycenter53_tot_spec_freq = []

        # Initialize the temporary lists that store the barycenters of the dark and noscan spectra peaks
        barycenters_list_dark_tot = []
        barycenters_list_noscan_tot = []

        # Initialize temporary lists returned by the function
        perturbation_list_51_temp = []
        perturbation_list_53_temp = []

        # Store peak frequency perturbation from the 5.1 scan of the last treatment performed
        list_perturbation_barycenter51_tot_spec_freq = self.low_list_perturbation_barycenter51_spec_freq[-1] + \
                                             self.list_perturbation_barycenter51_spec_freq[-1]

        # Add these peaks to the list of all treatments performed
        self.list_perturbation_barycenter51_tot_spec_freq.append(list_perturbation_barycenter51_tot_spec_freq)

        # Store peak frequency perturbation from the 5.3 scan of the last treatment performed
        list_perturbation_barycenter53_tot_spec_freq = self.low_list_perturbation_barycenter53_spec_freq[-1] + \
                                             self.list_perturbation_barycenter53_spec_freq[-1]

        # Add these peaks to the list of all treatments performed
        self.list_perturbation_barycenter53_tot_spec_freq.append(list_perturbation_barycenter53_tot_spec_freq)

        # Store peak frequency straylight from the 5.1 scan of the last treatment performed
        list_straylight_barycenter51_tot_spec_freq = self.low_list_straylight_barycenter51_spec_freq[-1] + \
                                             self.list_straylight_barycenter51_spec_freq[-1]

        # Add these peaks to the list of all treatments performed
        self.list_straylight_barycenter51_tot_spec_freq.append(list_straylight_barycenter51_tot_spec_freq)

        # Store peak frequency straylight from the 5.3 scan of the last treatment performed
        list_straylight_barycenter53_tot_spec_freq = self.low_list_straylight_barycenter53_spec_freq[-1] + \
                                             self.list_straylight_barycenter53_spec_freq[-1]

        # Add these peaks to the list of all treatments performed
        self.list_straylight_barycenter53_tot_spec_freq.append(list_straylight_barycenter53_tot_spec_freq)

        # Repeat the same operations for the dark spectrum
        barycenters_list_dark_tot = self.low_barycenters_list_dark[-1] + self.barycenters_list_dark[-1]
        self.barycenters_list_tot_dark.append(barycenters_list_dark_tot)

        # Repeat the same operations for the noscan spectrum
        barycenters_list_noscan_tot = self.low_barycenters_list_noscan[-1] + self.barycenters_list_noscan[-1]
        self.barycenters_list_tot_noscan.append(barycenters_list_noscan_tot)

        # ==============================================================================================================
        # Start comparisons

        # Comparison with barycenters_list_tot_dark barycenters_list_tot_noscan for scan 5.1
        for idx, perturbation_tuple in enumerate(self.list_perturbation_barycenter51_tot_spec_freq[-1]):
            perturbation_value = perturbation_tuple[0]  # First tuple value
            perturbation_amplitude = perturbation_tuple[1]

            match_found = False
            for dark_tuple in self.barycenters_list_tot_dark[-1]:
                dark_value = dark_tuple[0]  # First tuple value in barycenters_list_tot_dark
                dark_amplitude = dark_tuple[1]

                # Conversion coefficient between OPD and frequency fo scan 5.1
                if abs(perturbation_value/dark_value - 1) <= (5e-3)/0.05878:
                    if abs(perturbation_amplitude/dark_amplitude - 1) <= 1:
                        perturbation_list_51_temp.append('D')
                        match_found = True
                        break  # Exit the loop as soon as a match is found
                    else:
                        perturbation_list_51_temp.append('D (amp diff)')
                        match_found = True
                        break  # Exit the loop as soon as a match is found

            if not match_found:
                perturbation_list_51_temp.append('0')  # No match found

            for noscan_tuple in self.barycenters_list_tot_noscan[-1]:
                noscan_value = noscan_tuple[0]
                noscan_amplitude = noscan_tuple[1]

                if abs(perturbation_value/noscan_value - 1) <= (5e-3)/0.05878:
                    # If this index already has a value, replace it with 'D&N'
                    if abs(perturbation_amplitude / noscan_amplitude - 1) <= 1 :
                        if perturbation_list_51_temp[idx] == 'D':
                            perturbation_list_51_temp[idx] = 'D & N'
                        elif perturbation_list_51_temp[idx] == 'D (amp diff)':
                            perturbation_list_51_temp[idx] = 'D (amp diff) & N'
                        elif perturbation_list_51_temp[idx] == '0':
                            perturbation_list_51_temp[idx] = 'N'
                        break
                    else:
                        if perturbation_list_51_temp[idx] == 'D':
                            perturbation_list_51_temp[idx] = 'D & N (amp diff)'
                        elif perturbation_list_51_temp[idx] == 'D (amp diff)':
                            perturbation_list_51_temp[idx] = 'D (amp diff) & N (amp diff)'
                        elif perturbation_list_51_temp[idx] == '0':
                            perturbation_list_51_temp[idx] = 'N (amp diff)'
                        break

        # Comparison with barycenters_list_tot_dark barycenters_list_tot_noscan for scan 5.3
        for idx, perturbation_tuple in enumerate(self.list_perturbation_barycenter53_tot_spec_freq[-1]):
            perturbation_value = perturbation_tuple[0]
            perturbation_amplitude = perturbation_tuple[1]

            match_found = False
            for dark_tuple in self.barycenters_list_tot_dark[-1]:
                dark_value = dark_tuple[0]
                dark_amplitude = dark_tuple[1]

                # Conversion coefficient between OPD and frequency fo scan 5.3
                if abs(perturbation_value/dark_value - 1) <= (5e-3)/0.05656:
                    if abs(perturbation_amplitude/dark_amplitude - 1) <= 1:
                        perturbation_list_53_temp.append('D')
                        match_found = True
                        break
                    else:
                        perturbation_list_53_temp.append('D (amp diff)')
                        match_found = True
                        break

            if not match_found:
                perturbation_list_53_temp.append('0')

            for noscan_tuple in self.barycenters_list_tot_noscan[-1]:
                noscan_value = noscan_tuple[0]
                noscan_amplitude = noscan_tuple[1]

                if abs(perturbation_value/noscan_value - 1) <= (5e-3)/0.05656:
                    if abs(perturbation_amplitude / noscan_amplitude - 1) <= 1 :
                        if perturbation_list_53_temp[idx] == 'D':
                            perturbation_list_53_temp[idx] = 'D & N'
                        elif perturbation_list_53_temp[idx] == 'D (amp diff)':
                            perturbation_list_53_temp[idx] = 'D (amp diff) & N'
                        elif perturbation_list_53_temp[idx] == '0':
                            perturbation_list_53_temp[idx] = 'N'
                        break
                    else:
                        if perturbation_list_53_temp[idx] == 'D':
                            perturbation_list_53_temp[idx] = 'D & N (amp diff)'
                        elif perturbation_list_53_temp[idx] == 'D (amp diff)':
                            perturbation_list_53_temp[idx] = 'D (amp diff) & N (amp diff)'
                        elif perturbation_list_53_temp[idx] == '0':
                            perturbation_list_53_temp[idx] = 'N (amp diff)'
                        break

        self.perturbation_list_51.append(perturbation_list_51_temp)
        self.perturbation_list_53.append(perturbation_list_53_temp)

        return self.perturbation_list_51, self.perturbation_list_53

    ####################################################################################################################
    ####################################################################################################################

    def compare_barycenters_with_scan0_and_scan0P(self):
        """
        Function that compares the position of the barycenters of the perturbation peaks of scan 5.1
        (respectively scan 5.3) with the barycenters of the dark and noscan spectra. If the x-axis deviation condition
        is met, a list storing the position of the dark and noscan barycenters is displayed.

        :returns:
        - self.perturbation_list_51: list of disturbance peak positions in scan 5.1, noscan and/or dark
        - self.perturbation_list_53: list of disturbance peak positions in scan 5.3, noscan and/or dark
        """

        # Initialize the temporary lists that store the barycenters of frequency perturbation peaks for scans 5.1 and 5.3
        list_straylight_barycenter51_tot = []
        list_sraylight_barycenter53_tot = []

        # Initialize the temporary lists that store the barycenters of the dark and noscan spectra peaks
        barycenters_list_scan0_tot = []
        barycenters_list_scan0P_tot = []

        # Initialize temporary lists returned by the function
        straylight_list_51_temp = []
        straylight_list_53_temp = []

        # Store peak frequency perturbation from the 5.1 scan of the last treatment performed
        list_straylight_barycenter51_tot = self.low_list_straylight_barycenter51[-1] + \
                                             self.list_straylight_barycenter51[-1]

        # Add these peaks to the list of all treatments performed
        self.list_straylight_barycenter51_tot.append(list_straylight_barycenter51_tot)

        # Store peak frequency perturbation from the 5.3 scan of the last treatment performed
        list_straylight_barycenter53_tot= self.low_list_straylight_barycenter53[-1] + \
                                             self.list_straylight_barycenter53[-1]

        # Add these peaks to the list of all treatments performed
        self.list_straylight_barycenter53_tot.append(list_straylight_barycenter53_tot)

        # Repeat the same operations for the scan0 spectrum
        barycenters_list_scan0_tot = self.low_barycenters_list_scan0[-1] + self.barycenters_list_scan0[-1]
        self.barycenters_list_tot_scan0.append(barycenters_list_scan0_tot)

        # Repeat the same operations for the scan0P spectrum
        barycenters_list_scan0P_tot = self.low_barycenters_list_scan0P[-1] + self.barycenters_list_scan0P[-1]
        self.barycenters_list_tot_scan0P.append(barycenters_list_scan0P_tot)

        # ==============================================================================================================
        # Start comparisons

        # Comparison with barycenters_list_tot_scan0 and barycenters_list_tot_scan0P for scan 5.1
        for idx, straylight_tuple in enumerate(self.list_straylight_barycenter51_tot[-1]):
            straylight_value = straylight_tuple[0]  # First tuple value
            straylight_amplitude = straylight_tuple[1]  # First tuple value

            match_found = False
            for scan0_tuple in self.barycenters_list_tot_scan0[-1]:
                scan0_value = scan0_tuple[0]  # First tuple value in barycenters_list_tot_dark
                scan0_amplitude = scan0_tuple[1]  # First tuple value in barycenters_list_tot_dark

                # Conversion coefficient between OPD and frequency fo scan 5.1
                if abs(straylight_value/scan0_value - 1) <= 5e-3:
                    if abs(straylight_amplitude/scan0_amplitude - 1) <= 1:
                        straylight_list_51_temp.append('S0')
                        match_found = True
                        break
                    else:
                        straylight_list_51_temp.append('S0 (amp diff)')
                        match_found = True
                        break  # Exit the loop as soon as a match is found

            if not match_found:
                straylight_list_51_temp.append('0') # No match found

            for scan0P_tuple in self.barycenters_list_tot_scan0P[-1]:
                scan0P_value = scan0P_tuple[0]
                scan0P_amplitude = scan0P_tuple[1]  # First tuple value in barycenters_list_tot_dark

                if abs(straylight_value/scan0P_value - 1) <= 5e-3:
                    # If this index already has a value, replace it with 'D&N'
                    if abs(straylight_amplitude/scan0P_amplitude - 1) <= 1:
                        if straylight_list_51_temp[idx] == 'S0':
                            straylight_list_51_temp[idx] = 'S0 & S0P'
                        elif straylight_list_51_temp[idx] == 'S0 (amp diff)':
                            straylight_list_51_temp[idx] = 'S0 (amp diff) & S0P'
                        elif straylight_list_51_temp[idx] == '0':
                            straylight_list_51_temp[idx] = 'S0P'
                        break
                    else:
                        if straylight_list_51_temp[idx] == 'S0':
                            straylight_list_51_temp[idx] = 'S0 & S0P (amp diff)'
                        elif straylight_list_51_temp[idx] == 'S0 (amp diff)':
                            straylight_list_51_temp[idx] = 'S0 (amp diff) & S0P (amp diff)'
                        elif straylight_list_51_temp[idx] == '0':
                            straylight_list_51_temp[idx] = 'S0P (amp diff)'
                        break

        # Comparison with barycenters_list_tot_scan0 and barycenters_list_tot_scan0P for scan 5.3
        for idx, straylight_tuple in enumerate(self.list_straylight_barycenter53_tot[-1]):
            straylight_value = straylight_tuple[0]
            straylight_amplitude = straylight_tuple[1]

            match_found = False
            for scan0_tuple in self.barycenters_list_tot_scan0[-1]:
                scan0_value = scan0_tuple[0]
                scan0_amplitude = scan0_tuple[1]


                # Conversion coefficient between OPD and frequency fo scan 5.3
                if abs(straylight_value/scan0_value - 1) <= 5e-3:
                    if abs(straylight_amplitude/scan0_amplitude - 1) <= 1:
                        straylight_list_53_temp.append('S0')
                        match_found = True
                        break
                    else:
                        straylight_list_53_temp.append('S0 (amp diff)')
                        match_found = True
                        break

            if not match_found:
                straylight_list_53_temp.append('0')

            for scan0P_tuple in self.barycenters_list_tot_scan0P[-1]:
                scan0P_value = scan0P_tuple[0]
                scan0P_amplitude = scan0P_tuple[1]

                # Conversion coefficient between OPD and frequency fo scan 5.3
                if abs(straylight_value/scan0P_value - 1) <= 5e-3:
                    if abs(straylight_amplitude / scan0P_amplitude - 1) <= 1 :
                        if straylight_list_53_temp[idx] == 'S0':
                            straylight_list_53_temp[idx] = 'S0 & S0P'
                        elif straylight_list_53_temp[idx] == 'S0 (amp diff)':
                            straylight_list_53_temp[idx] = 'S0 (amp diff) & S0P'
                        elif straylight_list_53_temp[idx] == '0':
                            straylight_list_53_temp[idx] = 'S0P'
                        break
                    else:
                        if straylight_list_53_temp[idx] == 'S0':
                            straylight_list_53_temp[idx] = 'S0 & S0P (amp diff)'
                        elif straylight_list_53_temp[idx] == 'S0 (amp diff)':
                            straylight_list_53_temp[idx] = 'S0 (amp diff) & S0P (amp diff)'
                        elif straylight_list_53_temp[idx] == '0':
                            straylight_list_53_temp[idx] = 'S0P (amp diff)'
                        break

        # Add new result lists to self.perturbation_dark_list_51 and self.perturbation_dark_list_53
        self.straylight_list_51.append(straylight_list_51_temp)
        self.straylight_list_53.append(straylight_list_53_temp)

        return self.straylight_list_51, self.straylight_list_53

    ####################################################################################################################
    ####################################################################################################################

    def treatment_method_noscan(self, bandwidth):
        """
        Process noise floor, clusters, peak detection, barycenter for noscan acquisition.

        :param bandwidth: Bandwidth depends on the signal type.

        :return: int
        """

        printing("Starting treatment method noscan", 50, "-")

        # Extract the dictionary containing all acquisition data
        data_columns = self.analysis_noscan.data_columns

        # Extract frequency values from the dictionary
        self.freqs_noscan = np.array(data_columns.get("Freq"))
        mask = (self.freqs_noscan >= bandwidth[0]) & (self.freqs_noscan <= bandwidth[1])
        self.freqs_noscan = self.freqs_noscan[mask]

        if self.yaxis_scrollbox.currentText() == "Fractional amplitude":
            # Filtrer les valeurs d'amplitudes dans la dataframe
            self.data_noscan = self.file_vdc_df_noscan[mask].reset_index(drop=True)

        else:
            if self.canal == "Canal A":
                self.data_noscan = np.array(data_columns.get("Canal A"))
            if self.canal == "Canal B":
                self.data_noscan = np.array(data_columns.get("Canal B"))
            if self.canal == "Canal C":
                self.data_noscan = np.array(data_columns.get("Canal C"))
            if self.canal == "Canal D":
                self.data_noscan = np.array(data_columns.get("Canal D"))

            self.data_noscan = self.data_noscan[mask]

        # Compute the noise floor of the signal
        self.noise_floor_freqs_noscan, self.noise_floor_values_noscan = self.riemann_method(bandwidth,
                                                                                    self.freqs_noscan, self.data_noscan)

        # Compute clusters and detect peaks
        self.get_custering_variables_noscan(self.noise_floor_values_noscan)

        # Calculate barycenter of the detected peaks
        self.barycenters_list_noscan.append(self.barycenters_noscan)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def treatment_method_scan0(self, bandwidth):
        """
        Process noise floor, clusters, peak detection, barycenter for noscan acquisition.

        :param bandwidth: Bandwidth depends on the signal type.

        :return: int
        """

        printing("Starting treatment method scan0", 50, "-")

        self.data_scan0 = np.array(self.yaxis_scan0)
        self.freqs_scan0 = np.array(self.xaxis_scan0)

        # Compute the noise floor of the signal
        self.noise_floor_freqs_scan0, self.noise_floor_values_scan0 = self.riemann_method(bandwidth, self.freqs_scan0, self.data_scan0)

        # Compute clusters and detect peaks
        self.get_custering_variables_scan0(self.noise_floor_values_scan0)

        # Calculate barycenter of the detected peaks
        self.barycenters_list_scan0.append(self.barycenters_scan0)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def treatment_method_scan0P(self, bandwidth):
        """
        Process noise floor, clusters, peak detection, barycenter for noscan acquisition.

        :param bandwidth: Bandwidth depends on the signal type.

        :return: int
        """

        printing("Starting treatment method scan0P", 50, "-")

        self.data_scan0P = np.array(self.yaxis_scan0P)
        self.freqs_scan0P = np.array(self.xaxis_scan0P)

        # Compute the noise floor of the signal
        self.noise_floor_freqs_scan0P, self.noise_floor_values_scan0P = self.riemann_method(bandwidth,
                                                                                    self.freqs_scan0P, self.data_scan0P)

        # Compute clusters and detect peaks
        self.get_custering_variables_scan0P(self.noise_floor_values_scan0P)

        # Calculate barycenter of the detected peaks
        self.barycenters_list_scan0P.append(self.barycenters_scan0P)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def low_treatment_method_noscan(self, bandwidth):
        """
        Process noise floor, clusters, peak detection, barycenter for noscan acquisition for low frequencies.

        :param bandwidth: Bandwidth depends on the signal type.

        :return: int
        """

        printing("Starting low treatment method noscan", 50, "-")

        # Extract the dictionary containing all acquisition data
        data_columns = self.analysis_noscan.data_columns

        # Extract frequency values from the dictionary
        self.low_freqs_noscan = np.array(data_columns.get("Freq"))
        mask = (self.low_freqs_noscan >= bandwidth[0]) & (self.low_freqs_noscan <= bandwidth[1])
        self.low_freqs_noscan = self.low_freqs_noscan[mask]

        if self.yaxis_scrollbox.currentText() == "Fractional amplitude":
            # Filtrer les valeurs d'amplitudes dans la dataframe
            self.low_data_noscan = self.file_vdc_df_noscan[mask].reset_index(drop=True)

        else:
            if self.canal == "Canal A":
                self.low_data_noscan = np.array(data_columns.get("Canal A"))
            if self.canal == "Canal B":
                self.low_data_noscan = np.array(data_columns.get("Canal B"))
            if self.canal == "Canal C":
                self.low_data_noscan = np.array(data_columns.get("Canal C"))
            if self.canal == "Canal D":
                self.low_data_noscan = np.array(data_columns.get("Canal D"))

            self.low_data_noscan = self.low_data_noscan[mask]

        self.log_data_noscan = np.log10(np.where(self.low_data_noscan == 0, np.finfo(float).eps, self.low_data_noscan))
        self.log_freqs_noscan = np.log10(np.where(self.low_freqs_noscan == 0, np.finfo(float).eps, self.low_freqs_noscan))

        # Compute the noise floor of the signal
        self.log_fit_amplitude_optimal_noscan, self.fit_amplitude_optimal_noscan = self.create_noise_model(
            LOW_FREQ_BANDWIDTH_NOISE_MODEL, self.low_freqs_noscan, self.low_data_noscan, self.log_freqs_noscan,
              self.log_data_noscan)

        # Compute clusters and detect peaks
        self.get_custering_low_variables_noscan(self.fit_amplitude_optimal_noscan)

        # Calculate barycenter of the detected peaks
        self.low_barycenters_list_noscan.append(self.low_barycenters_noscan)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def low_treatment_method_scan0(self, bandwidth):
        """
        Process noise floor, clusters, peak detection, barycenter for noscan acquisition for low frequencies.

        :param bandwidth: Bandwidth depends on the signal type.

        :return: int
        """

        printing("Starting low treatment method scan0", 50, "-")

        self.low_data_scan0 = np.array(self.low_yaxis_scan0)
        self.low_freqs_scan0 = np.array(self.low_xaxis_scan0)

        self.log_data_scan0 = np.log10(self.low_data_scan0)
        self.log_freqs_scan0 = np.log10(self.low_freqs_scan0)

        # Compute the noise floor of the signal
        self.log_fit_amplitude_optimal_scan0, self.fit_amplitude_optimal_scan0 = self.create_noise_model(
            LOW_OPD_BANDWIDTH_NOISE_MODEL, self.low_freqs_scan0, self.low_data_scan0, self.log_freqs_scan0,
             self.log_data_scan0)

        # Compute clusters and detect peaks
        self.get_custering_low_variables_scan0(self.fit_amplitude_optimal_scan0)

        # Calculate barycenter of the detected peaks
        self.low_barycenters_list_scan0.append(self.low_barycenters_scan0)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def low_treatment_method_scan0P(self, bandwidth):
        """
        Process noise floor, clusters, peak detection, barycenter for noscan acquisition for low frequencies.

        :param bandwidth: Bandwidth depends on the signal type.

        :return: int
        """

        printing("Starting low treatment method scan0P", 50, "-")

        self.low_data_scan0P = np.array(self.low_yaxis_scan0P)
        self.low_freqs_scan0P = np.array(self.low_xaxis_scan0P)

        self.log_data_scan0P = np.log10(self.low_data_scan0P)
        self.log_freqs_scan0P = np.log10(self.low_freqs_scan0P)

        # Compute the noise floor of the signal
        self.log_fit_amplitude_optimal_scan0P, self.fit_amplitude_optimal_scan0P = self.create_noise_model(
            LOW_OPD_BANDWIDTH_NOISE_MODEL, self.low_freqs_scan0P, self.low_data_scan0P, self.log_freqs_scan0P,
             self.log_data_scan0P)

        # Compute clusters and detect peaks
        self.get_custering_low_variables_scan0P(self.fit_amplitude_optimal_scan0P)

        # Calculate barycenter of the detected peaks
        self.low_barycenters_list_scan0P.append(self.low_barycenters_scan0P)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_custering_variables_noscan(self, noise_floor_values):
        """
        Function linked to the PeakClustering class which create clusters for each peak detected in the noscan spectrum.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """
        self.peak_clustering_noscan = PeakClustering(self.freqs_noscan, self.data_noscan)

        xpeak, ypeak, _, _ = self.detect_peaks(noise_floor_values, self.freqs_noscan, self.data_noscan)
        self.barycenters_noscan, self.barycenter_clusters_noscan, self.full_clusters_peaks_noscan = (
            self.peak_clustering_noscan.group_points_into_peaks(xpeak, ypeak))

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_custering_variables_scan0(self, noise_floor_values):
        """
        Function linked to the PeakClustering class which create clusters for each peak detected in the noscan spectrum.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """
        self.peak_clustering_scan0 = PeakClustering(self.freqs_scan0, self.data_scan0)

        xpeak, ypeak, _, _ = self.detect_peaks(noise_floor_values, self.freqs_scan0, self.data_scan0)
        self.barycenters_scan0, self.barycenter_clusters_scan0, self.full_clusters_peaks_scan0 = (
            self.peak_clustering_scan0.group_points_into_peaks(xpeak, ypeak))

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_custering_variables_scan0P(self, noise_floor_values):
        """
        Function linked to the PeakClustering class which create clusters for each peak detected in the noscan spectrum.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """
        self.peak_clustering_scan0P = PeakClustering(self.freqs_scan0P, self.data_scan0P)

        xpeak, ypeak, _, _ = self.detect_peaks(noise_floor_values, self.freqs_scan0P, self.data_scan0P)
        self.barycenters_scan0P, self.barycenter_clusters_scan0P, self.full_clusters_peaks_scan0P = (
            self.peak_clustering_scan0P.group_points_into_peaks(xpeak, ypeak))

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_custering_low_variables_noscan(self, noise_floor_values):
        """
        Function linked to the PeakClustering class which create clusters for each peak detected in the noscan spectrum
        in low frequencies.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """
        self.peak_clustering_noscan = PeakClustering(self.low_freqs_noscan, self.low_data_noscan)

        low_xpeak, low_ypeak, _, _ = self.detect_peaks_above_noise_model(noise_floor_values,
                                                                      self.low_freqs_noscan, self.low_data_noscan)
        self.low_barycenters_noscan, self.low_barycenter_clusters_noscan, self.low_full_clusters_peaks_noscan = (
            self.peak_clustering_noscan.group_points_into_peaks(low_xpeak, low_ypeak))

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_custering_low_variables_scan0(self, noise_floor_values):
        """
        Function linked to the PeakClustering class which create clusters for each peak detected in the noscan spectrum
        in low frequencies.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """
        self.peak_clustering_scan0 = PeakClustering(self.low_freqs_scan0, self.low_data_scan0)

        low_xpeak, low_ypeak, _, _ = self.detect_peaks_above_noise_model(noise_floor_values,
                                                                      self.low_freqs_scan0, self.low_data_scan0)
        self.low_barycenters_scan0, self.low_barycenter_clusters_scan0, self.low_full_clusters_peaks_scan0 = (
            self.peak_clustering_scan0.group_points_into_peaks(low_xpeak, low_ypeak))

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_custering_low_variables_scan0P(self, noise_floor_values):
        """
        Function linked to the PeakClustering class which create clusters for each peak detected in the noscan spectrum
        in low frequencies.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """
        self.peak_clustering_scan0P = PeakClustering(self.low_freqs_scan0P, self.low_data_scan0P)

        low_xpeak, low_ypeak, _, _ = self.detect_peaks_above_noise_model(noise_floor_values,
                                                                      self.low_freqs_scan0P, self.low_data_scan0P)
        self.low_barycenters_scan0P, self.low_barycenter_clusters_scan0P, self.low_full_clusters_peaks_scan0P = (
            self.peak_clustering_scan0P.group_points_into_peaks(low_xpeak, low_ypeak))

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def extract_datas(self, file_path):
        """
        Function that extracts the path to the noscan spectrum corresponding to the 5.1 and 5.3 scans currently being
        processed.

        :param file_path: List of path to scans 5.1 and 5.3.

        :return new_file_path: Noscan spectrum access path.
        """

        # Retrieve the last item in the path list for scans 5.1 and 5.3 and convert the path into a Path object
        path = Path(file_path[-1])
        # Move up the access path 4 levels
        parent_folder = path.parents[2]

        # Initialize variable for new path
        new_file_path_dark = []
        new_file_path_noscan = []

        # Browse subfolders to find the one containing “Dark” (case insensitive)
        for folder in parent_folder.iterdir():
            if folder.is_dir() and "noscan" in folder.name.lower():
                # Once in the “noscan” folder, enter the “processed” folder
                processed_folder = folder / "Processed"

                if processed_folder.exists() and processed_folder.is_dir():
                    # Get the list of files in the “processed” folder
                    processed_files = sorted(processed_folder.iterdir(), key=lambda f: f.name)
                    new_file_path_noscan.extend(str(file) for file in processed_files)

            if folder.is_dir() and "dark" in folder.name.lower():
                # Once in the “dark” folder, enter the “processed” folder
                processed_folder = folder / "Processed"

                if processed_folder.exists() and processed_folder.is_dir():
                    # Get the list of files in the “processed” folder
                    processed_files = sorted(processed_folder.iterdir(), key=lambda f: f.name)
                    new_file_path_dark.extend(str(file) for file in processed_files)

        return new_file_path_dark, new_file_path_noscan

    ####################################################################################################################
    ####################################################################################################################

    def treatment_method(self, bandwidth):
        """
        Process noise floor, clusters, peak detection, barycenter for scans 5.1 and 5.3.

        :param bandwidth: Bandwidth depends on the signal type.

        :return: int
        """

        printing("Starting treatment method", 50, "-")

        self.data = np.array(self.yaxis)
        self.freqs = np.array(self.xaxis)

        # Compute the noise floor of the signal
        self.noise_floor_freqs, self.noise_floor_values = self.riemann_method(bandwidth, self.freqs, self.data)

        # Compute clusters and detect peaks
        self.get_custering_variables(self.noise_floor_values)

        # Calculate barycenter of the detected peaks
        self.barycenters_list.append(self.barycenters)

        # Store output data
        self.barycenter_clusters_list.append(self.barycenter_clusters)
        self.full_clusters_list.append(self.full_clusters_peaks)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def low_treatment_method(self, bandwidth):
        """
        Process noise floor, clusters, peak detection, barycenter for scans 5.1 and 5.3 in low frequencies.

        :param bandwidth: Bandwidth depends on the signal type.

        :return: int
        """

        printing("Starting low treatment method", 50, "-")

        self.low_data = np.array(self.low_yaxis)
        self.low_freqs = np.array(self.low_xaxis)

        self.log_data = np.log10(self.low_data)
        self.log_freqs = np.log10(self.low_freqs)

        # Compute the noise floor of the signal
        self.log_fit_amplitude_optimal, self.fit_amplitude_optimal = self.create_noise_model(
            LOW_OPD_BANDWIDTH_NOISE_MODEL, self.low_freqs, self.low_data, self.log_freqs, self.log_data)

        # Compute clusters and detect peaks
        self.get_custering_low_variables(self.fit_amplitude_optimal)

        # Calculate barycenter of the detected peaks
        self.low_barycenters_list.append(self.low_barycenters)

        # Store output data
        self.low_barycenter_clusters_list.append(self.low_barycenter_clusters)
        self.low_full_clusters_list.append(self.low_full_clusters_peaks)

        # Display results
        if self.count == 0:
            self.show_filtered_spectrum()
        elif self.count % 2 != 0:
            self.overlay_filtered_plot()

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def analyse_data(self):
        """
        Store barycenter data in an excel file.
        :return: int
        """

        # Create dictionaries for each page of the Excel file
        data_page51_SL_peaks = {}
        data_page51_perturbation_peaks = {}
        data_page53_SL_peaks = {}
        data_page53_perturbation_peaks = {}
        data_treated_signal = {}
        data_treated_signal_51 = {}

        # Create columns in dataframes with lists values
        for i in range(len(self.barycenters_list) // 2):
            # ----------------------------------------------------------------------------------------------------------
            # Scan 5.1
            data_page51_SL_peaks[self.legends[i][15:] + ' Stray light peaks'] = [f"({float(v[0])}, {float(v[1])})" for v in
                                                                        self.list_straylight_barycenter51_tot[i]]

            data_page51_SL_peaks[self.legends[i][15:] + ' Stray light peaks real amplitude'] = (
                self.list_straylight_max51_tot)[i]

            data_page51_SL_peaks[self.legends[i][15:] + ' Stray light peaks FWMH'] = self.widths_SL_51_tot[i]

            data_page51_SL_peaks[self.legends[i][15:] + ' Origin of peaks with stray light signature '] = \
            self.straylight_list_51[i]

            data_page51_SL_peaks[self.legends[i][15:] + ' Quality percentage ratio x SL peaks'] = (
                self.quality_x_SL_list)[i]

            data_page51_SL_peaks[self.legends[i][15:] + ' Quality percentage ratio y SL peaks'] = (
                self.quality_y_SL_list)[i]

            data_page51_perturbation_peaks[self.legends[i][15:] + ' Perturbation peaks'] = [
                f"({float(v[0])}, {float(v[1])})" for v in
                self.low_list_perturbation_barycenter51[i] +
                self.list_perturbation_barycenter51[i]]

            if "OPD" in self.x_axis_legend:
                data_page51_perturbation_peaks[self.legends[i][15:] + ' Perturbation peaks in frequency'] = \
                [f"({float(v[0])}, {float(v[1])})" for v in self.list_perturbation_barycenter51_tot_spec_freq[i]]

            data_page51_perturbation_peaks[self.legends[i][15:] + ' Perturbation peaks real amplitude'] = (
                self.list_perturbation_max51_tot)[i]

            data_page51_perturbation_peaks[self.legends[i][15:] + ' Perturbation peaks FWMH'] = (
                self.widths_perturbation_51_tot)[i]

            data_page51_perturbation_peaks[self.legends[i][15:] + ' Origin of perturbation peaks'] = (
                self.perturbation_list_51)[i]

            # ----------------------------------------------------------------------------------------------------------
            # Scan 5.3
            data_page53_SL_peaks[self.legends[i][15:] + ' Stray light peaks'] = [f"({float(v[0])}, {float(v[1])})" for v in
                                                                        self.list_straylight_barycenter53_tot[i]]

            data_page53_SL_peaks[self.legends[i][15:] + ' Stray light peaks real amplitude'] = (
                self.list_straylight_max53_tot)[i]

            data_page53_SL_peaks[self.legends[i][15:] + ' Stray light peaks FWMH'] = self.widths_SL_53_tot[i]

            data_page53_SL_peaks[self.legends[i][15:] + ' Origin of peaks with stray light signature '] = \
                self.straylight_list_53[i]

            data_page53_SL_peaks[self.legends[i][15:] + ' Quality percentage ratio x SL peaks'] = (
                self.quality_x_SL_list)[i]

            data_page53_SL_peaks[self.legends[i][15:] + ' Quality percentage ratio y SL peaks'] = (
                self.quality_y_SL_list)[i]

            data_page53_perturbation_peaks[self.legends[i][15:] + ' Perturbation peaks'] = [f"({float(v[0])}, {float(v[1])})" for v in
                                                                         self.low_list_perturbation_barycenter53[i] +
                                                                         self.list_perturbation_barycenter53[i]]

            if "OPD" in self.x_axis_legend:
                data_page53_perturbation_peaks[self.legends[i][15:] + ' Perturbation peaks in frequency'] = \
                [f"({float(v[0])}, {float(v[1])})" for v in self.list_perturbation_barycenter53_tot_spec_freq[i]]

            data_page53_perturbation_peaks[self.legends[i][15:] + ' Perturbation peaks real amplitude'] = (
                self.list_perturbation_max53_tot)[i]

            data_page53_perturbation_peaks[self.legends[i][15:] + ' Perturbation peaks FWMH'] = \
            self.widths_perturbation_53_tot[i]

            data_page53_perturbation_peaks[self.legends[i][15:] + ' Origin of perturbation peaks'] = (
                self.perturbation_list_53)[i]

            # Treated signal
            # ----------------------------------------------------------------------------------------------------------
            # Scan 5.1
            data_treated_signal_51[self.legends[i] + " X-component Treated signal"] = (
                self.scans_comparison_treated_signal_x_tot_51)[i]
            data_treated_signal_51[self.legends[i] + " Y-component Treated signal"] = (
                self.scans_comparison_treated_signal_y_tot_51)[i]

            # ----------------------------------------------------------------------------------------------------------
            # Scan 5.3
            data_treated_signal[self.legends[i] + " X-component Treated signal"] = (
                self.scans_comparison_treated_signal_x_tot)[i]
            data_treated_signal[self.legends[i] + " Y-component Treated signal"] = (
                self.scans_comparison_treated_signal_y_tot)[i]

        # Transform dictionaries in dataframe
        df_51_SL_peaks = pd.DataFrame.from_dict(data_page51_SL_peaks, orient='index').transpose()
        df_51_perturbation_peaks = pd.DataFrame.from_dict(data_page51_perturbation_peaks, orient='index').transpose()
        df_53_SL_peaks = pd.DataFrame.from_dict(data_page53_SL_peaks, orient='index').transpose()
        df_53_perturbation_peaks = pd.DataFrame.from_dict(data_page53_perturbation_peaks, orient='index').transpose()
        df_treated_signal_51 = pd.DataFrame.from_dict(data_treated_signal_51, orient='index').transpose()
        df_treated_signal = pd.DataFrame.from_dict(data_treated_signal, orient='index').transpose()

        # --------------------------------------------------------------------------------------------------------------
        # Check exception errors
        try:

            # Save barycenter dataframe into Excel file
            file_path_barycenter = f'Barycenter_data_{self.qpr}.xlsx'
            with pd.ExcelWriter(file_path_barycenter, engine='openpyxl') as writer:
                df_51_SL_peaks.to_excel(writer, sheet_name='Scan51 SL peaks', index=False)
                df_51_perturbation_peaks.to_excel(writer, sheet_name='Scan51 perturbation peaks', index=False)
                df_53_SL_peaks.to_excel(writer, sheet_name='Scan53 SL peaks', index=False)
                df_53_perturbation_peaks.to_excel(writer, sheet_name='Scan53 perturbation peaks', index=False)
            print(f"Barycenter are now accessible in 'Barycenter_data_{self.qpr}.xlsx'.")

            # Save treated signal dataframe into Excel file
            file_path_treated_signal = f'Treated_signal_data_{self.qpr}.xlsx'
            with pd.ExcelWriter(file_path_treated_signal, engine='openpyxl') as writer:
                df_treated_signal_51.to_excel(writer, sheet_name='Treated signal Scan51', index=False)
                df_treated_signal.to_excel(writer, sheet_name='Treated signal', index=False)
            print(f"Treated signal's data are now accessible in 'Treated_signal_data_{self.qpr}.xlsx'.")

            # Open automatically the file depends on the computer's operator system
            if platform.system() == 'Windows':
                os.startfile(file_path_barycenter)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', file_path_barycenter))
            else:  # Linux
                subprocess.call(('xdg-open', file_path_barycenter))

        # --------------------------------------------------------------------------------------------------------------

        except Exception as e:
            print(f"An error occurred while saving the CSV: {e}")

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def fill_scrollbox(self, scrollbox):
        """
        Fill scroll  box in function of parent scroll box selection.

        :param scrollbox: Children scroll box to fill.

        :return: int
        """

        # Get a list containing all folders located at the computed path
        if scrollbox == self.folder_scrollbox_sl:
            list_name_documents = os.listdir(self.data_path)
            folders_name = [name for name in list_name_documents if
                            os.path.isdir(os.path.join(self.data_path, name))]

            # Fill the children scroll box
            for name in folders_name:
                scrollbox.addItem(str(name))

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def process_files(self, files):
        """
        Get file paths of signal to process.

        :param files: Type of files to browse.

        :return: List of computed file path: List[str]
        """

        # List to store the resulting file paths
        file_paths = []

        # Selected folder is retri
        folder_name = str(self.folder_scrollbox_sl.currentText())

        # --------------------------------------------------------------------------------------------------------------
        # File path creation

        # Iterate over the list of filenames provided
        for filename in files:

            # Construct the base directory path
            path = os.path.join(self.data_path, folder_name)

            # List all subdirectories in the path
            folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

            # Find the folder that matches the filename
            target_folder = None
            for folder in folders:
                if filename in folder.lower():
                    target_folder = folder
                    break

            # Construct the file path to the "Processed" subfolder within the target folder
            file_path = os.path.join(path, target_folder, 'Processed')

            # List all files in the "Processed" directory
            files = os.listdir(file_path)

            # Search for the specific file that matches the user's pico selection
            target_file = None
            for file in files:
                target_file = file
                break

            # Add the full path of the target file to the result list
            f = os.path.join(file_path, target_file)
            file_paths.append(f)

        # --------------------------------------------------------------------------------------------------------------

        return file_paths

    ####################################################################################################################
    ####################################################################################################################

    def create_noise_model(self, bandwidth, freqs, data, log_freqs, log_data):
        # Plage de fréquences entre les limites de 'bandwidth', avec un masque pour exclure les fréquences entre 8e-1 et 1.5

        if bandwidth == LOW_OPD_BANDWIDTH_NOISE_MODEL:
            mask = ((freqs >= np.array([0], dtype=np.float64)) &
                    (freqs <= np.array(bandwidth[1], dtype=np.float64)) &
                    ~((freqs >= np.array(7e-1, dtype=np.float64)) &
                      (freqs <= np.array(1.3, dtype=np.float64))))
        else:
            mask = ((freqs >= np.array(bandwidth[0], dtype=np.float64)) &
                    (freqs <= np.array(bandwidth[1], dtype=np.float64)) &
                    ~((freqs >= np.array(7e-1, dtype=np.float64)) &
                      (freqs <= np.array(1.3, dtype=np.float64))))

        valid_freq = freqs[mask].astype(np.float64)
        valid_data = data[mask].astype(np.float64)

        # Log-log transformation (log des fréquences et log des amplitudes) avec float64
        log_valid_freq = np.log10(valid_freq).astype(np.float64)
        log_valid_amplitude = np.log10(valid_data).astype(np.float64)

        # Subdivision de l'intervalle de fréquences
        n_sub_intervals = 10  # Nombre de sous-intervalles, tu peux ajuster
        sub_interval_bounds = np.linspace(log_valid_freq.min(), log_valid_freq.max(), n_sub_intervals + 1,
                                          dtype=np.float64)

        filtered_log_freq = []
        filtered_log_amplitude = []

        # Pour chaque sous-intervalle, filtrer les données à plus de 2 sigmas
        for i in range(n_sub_intervals):
            lower_bound = sub_interval_bounds[i]
            upper_bound = sub_interval_bounds[i + 1]

            # Sélectionner les données dans cet intervalle
            mask_sub_interval = (log_valid_freq >= lower_bound) & (log_valid_freq < upper_bound)
            sub_log_freq = log_valid_freq[mask_sub_interval].astype(np.float64)
            sub_log_amplitude = log_valid_amplitude[mask_sub_interval].astype(np.float64)

            # Calculer la moyenne et l'écart-type pour l'amplitude dans ce sous-intervalle
            mean_amplitude = np.mean(sub_log_amplitude)
            std_amplitude = np.std(sub_log_amplitude)

            # Filtrer les points qui sont dans les 2 sigmas autour de la moyenne
            mask_2sigma = (sub_log_amplitude >= mean_amplitude - 2 * std_amplitude) & (
                    sub_log_amplitude <= mean_amplitude + 2 * std_amplitude)
            filtered_log_freq.extend(sub_log_freq[mask_2sigma].astype(np.float64))
            filtered_log_amplitude.extend(sub_log_amplitude[mask_2sigma].astype(np.float64))

        # Convertir les listes filtrées en tableaux numpy
        filtered_log_freq = np.array(filtered_log_freq, dtype=np.float64)
        filtered_log_amplitude = np.array(filtered_log_amplitude, dtype=np.float64)

        # Diviser les données filtrées en ensemble d'entraînement et de test pour la validation croisée
        log_freq_train, log_freq_test, log_amplitude_train, log_amplitude_test = train_test_split(
            filtered_log_freq.astype(np.float64), filtered_log_amplitude.astype(np.float64), test_size=0.3,
            random_state=42)

        # Tester plusieurs degrés pour le fit polynomial
        max_degree = 10
        mse_scores = []

        for degree in range(1, max_degree + 1):
            # Fit polynomial avec float64
            coeffs = np.polyfit(log_freq_train, log_amplitude_train, degree).astype(np.float64)
            poly_fit = np.poly1d(coeffs)

            # Prédictions sur les données de test
            log_fit_amplitude_test = poly_fit(log_freq_test.astype(np.float64))

            # Calculer l'erreur quadratique moyenne (MSE)
            mse = mean_squared_error(log_amplitude_test.astype(np.float64), log_fit_amplitude_test.astype(np.float64))
            mse_scores.append(mse)

            # Trouver le degré avec la MSE la plus faible
        optimal_degree = np.argmin(mse_scores) + 1
        print(f"Optimum degree found : {optimal_degree}")

        # Refaire un fit polynomial avec le degré optimal sur toutes les données filtrées
        coeffs_optimal = np.polyfit(filtered_log_freq, filtered_log_amplitude, optimal_degree).astype(np.float64)
        poly_fit_optimal = np.poly1d(coeffs_optimal)

        # Générer la courbe ajustée sur toutes les fréquences en logarithmique
        log_fit_amplitude_optimal = poly_fit_optimal(log_freqs.astype(np.float64))
        fit_amplitude_optimal = np.power(10, log_fit_amplitude_optimal).astype(np.float64)

        return log_fit_amplitude_optimal.astype(np.float64), fit_amplitude_optimal.astype(np.float64)

    ####################################################################################################################
    ####################################################################################################################

    def detect_peaks_above_noise_model(self, fit_amplitude_optimal, low_freqs, low_data, threshold_factor=3,
                                       window_size=12):
        """
        Fonction pour détecter les pics au-dessus du modèle de bruit ajusté (amplitudes).

        :param filtered_log_freq: Fréquences (en échelle logarithmique).
        :param filtered_log_amplitude: Amplitudes réelles (en échelle logarithmique).
        :param log_fit_amplitude_optimal: Amplitudes prédites par le modèle ajusté.
        :param threshold_factor: Facteur multipliant l'écart-type pour définir le seuil de détection des pics.

        :return: Indices des pics et valeurs des pics.
        """

        # Calculer les écarts entre les amplitudes réelles et le modèle ajusté
        residuals = low_data - fit_amplitude_optimal.astype(np.float64)

        # Initialiser un tableau pour les seuils dynamiques locaux
        peak_threshold = np.zeros_like(low_data, dtype=np.float64)

        # Calculer le seuil pour chaque fréquence
        for i in range(len(low_data)):
            # Définir la fenêtre autour de la fréquence i
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(low_data), i + window_size // 2 + 1)

            # Extraire les résidus dans cette fenêtre locale
            local_residuals = residuals[start_idx:end_idx]

            # Calculer la médiane et l'IQR local
            local_median = np.median(local_residuals)
            local_iqr = np.percentile(local_residuals, 75) - np.percentile(local_residuals, 25)

            # Définir le seuil local : médiane locale + facteur * IQR local
            peak_threshold[i] = local_median + threshold_factor * local_iqr


        # Utiliser find_peaks pour détecter les pics au-dessus du seuil
        low_peaks, _ = find_peaks(low_data, height=peak_threshold.astype(np.float64), distance=2)

        # Retourner les fréquences et amplitudes correspondantes aux pics
        peak_frequencies = low_freqs[low_peaks].astype(np.float64)
        peak_amplitudes = low_data[low_peaks].astype(np.float64)

        # Calculer la largeur des pics (FWHM) pour chaque pic détecté
        results_half = peak_widths(low_data, low_peaks, rel_height=0.5)
        peak_widths_half = results_half[0]  # Largeur des pics à mi-hauteur

        return peak_frequencies, peak_amplitudes, peak_widths_half, peak_threshold.astype(np.float64)

    ####################################################################################################################
    ####################################################################################################################

    def riemann_method(self, bandwidth, freqs, data, percent_to_exclude=30):
        """
        Get a precise noise floor and interpolate every subpart to get the noise floor limit.

        :param bandwidth: Tuple indicating the frequency range.
        :param percent_to_exclude: Percentage of highest data points to exclude when calculating the noise floor.

        :returns:
        - interpolated_freqs: np.ndarray - Array of interpolated frequency values across the specified bandwidth.
        - interp_values: np.ndarray - Interpolated noise floor values corresponding to the interpolated frequencies.
        """

        # Set interval width based on bandwidth type
        if bandwidth == OPD_BANDWIDTH:
            interval_width = 0.25
        elif bandwidth == FREQ_BANDWIDTH:
            interval_width = 0.25 / 0.05656
        else:
            # Warn if the bandwidth specification is incorrect
            QMessageBox.warning(self, "Bandwidth issue", "The bandwidth specification went wrong.")
            return

        # Create frequency intervals for analysis
        freq_intervals = np.arange(bandwidth[0], bandwidth[-1] + interval_width, interval_width)

        # Initialize lists for storing maximum values and their corresponding center frequencies
        max_values = []
        centers = []

        # ==============================================================================================================
        # Apply the Riemann method

        # Process each frequency interval
        for i in range(len(freq_intervals) - 1):
            coefficient = 2.5
            low_freq = freq_intervals[i]
            high_freq = freq_intervals[i + 1]
            mask = (freqs >= low_freq) & (freqs < high_freq)

            if np.any(mask):

                # Filter data within the current interval
                interval_data = data[mask]

                # Sort and exclude the top percentage of data points
                sorted_indices = np.argsort(interval_data)
                num_points_to_exclude = int(len(interval_data) * (percent_to_exclude / 100))
                excluded_indices = sorted_indices[-num_points_to_exclude:]
                filtered_data = np.delete(interval_data, excluded_indices)

                if len(filtered_data) > 0:

                    while True:
                        # Adjust the max value based on a coefficient until it meets a noise threshold
                        max_value = np.max(filtered_data) * coefficient
                        if max_value > 1e-5:
                            coefficient -= 0.2
                        else:
                            break
                    center_freq = (low_freq + high_freq) / 2

                    max_values.append(max_value)
                    centers.append(center_freq)

        # Create an interpolation function based on the calculated max values
        interpolation_function = interp1d(centers, max_values, kind='slinear', fill_value="extrapolate")

        # Generate interpolated values across the full bandwidth
        interpolated_freqs = np.linspace(bandwidth[0], bandwidth[-1], num=len(freqs))
        interp_values = interpolation_function(interpolated_freqs)

        # Identify data points above the noise floor
        above_curve_mask = data > interp_values

        print(f"Noise floor fitted with a mean amplitude of: {np.mean(interp_values)}")

        return interpolated_freqs, interp_values

    ####################################################################################################################
    ####################################################################################################################
    
    def get_custering_variables(self, noise_floor_values):
        """
        Function linked to the PeakClustering class which create clusters for each peak detected in scans 5.1 and 5.3.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """

        self.peak_clustering = PeakClustering(self.freqs, self.data)

        if self.count % 2 == 0:
            xpeak, ypeak, peak_width_51, _ = self.detect_peaks(noise_floor_values, self.freqs, self.data)
            self.peak_width_51.append(5.878e-4 * peak_width_51)
            self.barycenters, self.barycenter_clusters, self.full_clusters_peaks = (
                self.peak_clustering.group_points_into_peaks(xpeak, ypeak))
            self.data_51 = self.data
            self.freqs_51 = self.freqs

        else:
            xpeak, ypeak, peak_width_53, _ = self.detect_peaks(noise_floor_values, self.freqs, self.data)
            self.peak_width_53.append(5.656e-4 * peak_width_53)
            self.barycenters, self.barycenter_clusters, self.full_clusters_peaks = (
                self.peak_clustering.group_points_into_peaks(xpeak, ypeak))
            self.low_data_51 = self.low_data
            self.low_freqs_51 = self.low_freqs

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_custering_low_variables(self, noise_floor_values):
        """
        Function linked to the PeakClustering class which create clusters for each peak detected in scans 5.1 and 5.3
        in low frequencies.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """

        self.peak_clustering = PeakClustering(self.low_freqs, self.low_data)

        if self.count % 2 == 0:
            low_xpeak, low_ypeak, low_peak_width_51, _ = self.detect_peaks_above_noise_model(noise_floor_values,
                                                                                        self.low_freqs, self.low_data)
            self.low_peak_width_51.append(5.878e-4 * low_peak_width_51)
            self.low_barycenters, self.low_barycenter_clusters, self.low_full_clusters_peaks = (
                self.peak_clustering.group_points_into_peaks(low_xpeak, low_ypeak))

        else:
            low_xpeak, low_ypeak, low_peak_width_53, _ = self.detect_peaks_above_noise_model(noise_floor_values,
                                                                                        self.low_freqs, self.low_data)
            self.low_peak_width_53.append(5.656e-4 * low_peak_width_53)
            self.low_barycenters, self.low_barycenter_clusters, self.low_full_clusters_peaks = (
                self.peak_clustering.group_points_into_peaks(low_xpeak, low_ypeak))


        return 0

    ####################################################################################################################
    ####################################################################################################################
        
    def scans_comparison(self):
        """
        Function linked to the ScansComparison class.

        :return: int
        """

        # Create a BarycenterProcessor instance
        self.barycenter_processor = ScansComparison(
            self.barycenters_list, self.barycenter_clusters_list, self.full_clusters_list, self.sl_points_y,
            self.sl_points_x, self.scans_comparison_treated_signal_x, self.scans_comparison_treated_signal_y,
            self.scans_comparison_treated_signal_x_51, self.scans_comparison_treated_signal_y_51,
            self.x_ratio_values, self.y_ratio_values, self.y_max_values, self.x_max_values, self.scans_done, self.freqs,
            self.freqs_51,
            self.data, self.data_51,
            self.list_straylight_barycenter51, self.list_perturbation_barycenter51, self.list_straylight_barycenter53,
            self.list_perturbation_barycenter53, self.pic_number, self.peak_width_51, self.peak_width_53,
            self.quality_x_SL, self.quality_y_SL, self.quality_x_perturbation, self.quality_y_perturbation,
            self.list_straylight_max51, self.list_straylight_max53,
            self.list_perturbation_max51, self.list_perturbation_max53)

        (list_y_max, self.indices_to_keep_53, self.indices_to_remove_53, self.widths_SL_51,
                self.widths_perturbation_51, self.widths_SL_53, self.widths_perturbation_53,
                self.sl_points_y, self.sl_points_x, self.scans_comparison_treated_signal_x,
                self.scans_comparison_treated_signal_y, self.scans_comparison_treated_signal_x_51,
                self.scans_comparison_treated_signal_y_51, self.x_ratio_values, self.y_ratio_values, self.y_max_values,
                self.x_max_values, self.scans_done, self.list_straylight_barycenter51, self.list_perturbation_barycenter51,
                self.list_straylight_barycenter53, self.list_perturbation_barycenter53, self.quality_x_SL,
                self.quality_y_SL, self.quality_x_perturbation, self.quality_y_perturbation, self.list_straylight_max51,
                self.list_straylight_max53, self.list_perturbation_max51, self.list_perturbation_max53) = (
         self.barycenter_processor.compare_barycenters(
         X_CONDITION_BARYCENTER_COMPARISON, Y_CONDITION_BARYCENTER_COMPARISON))


        self.list_perturbation_barycenter51_spec_freq = [
                [(val[0] / 5.878e-2, val[1]) for val in sublist]
                for sublist in self.list_perturbation_barycenter51
            ]

        self.list_perturbation_barycenter53_spec_freq = [
                [(val[0] / 5.656e-2, val[1]) for val in sublist]
                for sublist in self.list_perturbation_barycenter53
            ]

        self.list_straylight_barycenter51_spec_freq = [
                [(val[0] / 5.878e-2, val[1]) for val in sublist]
                for sublist in self.list_straylight_barycenter51
            ]

        self.list_straylight_barycenter53_spec_freq = [
                [(val[0] / 5.656e-2, val[1]) for val in sublist]
                for sublist in self.list_straylight_barycenter53
            ]

        print("\n")
        printing(
            f"Number of SL pics detected between 5.1 and 5.3 GHz/s scans : {len(list_y_max)}\n "
            f"Number of removed points: {len(self.indices_to_remove_53)}",
            100, "-")

        return 0
    
    ####################################################################################################################
    ####################################################################################################################
    
    def low_scans_comparison(self):
        """
        Function linked to the ScansComparison class for low frequencies.

        :return: int
        """

        # Create a BarycenterProcessor instance
        self.barycenter_processor = ScansComparison(
            self.low_barycenters_list, self.low_barycenter_clusters_list, self.low_full_clusters_list,
            self.low_sl_points_y, self.low_sl_points_x, self.low_scans_comparison_treated_signal_x,
            self.low_scans_comparison_treated_signal_y,  self.low_scans_comparison_treated_signal_x_51,
            self.low_scans_comparison_treated_signal_y_51, self.low_x_ratio_values, self.low_y_ratio_values,
            self.low_y_max_values, self.low_x_max_values, self.low_scans_done, self.low_freqs, self.low_freqs_51,
            self.low_data, self.low_data_51,
            self.low_list_straylight_barycenter51, self.low_list_perturbation_barycenter51,
            self.low_list_straylight_barycenter53, self.low_list_perturbation_barycenter53, self.low_pic_number,
            self.low_peak_width_51, self.low_peak_width_53, self.low_quality_x_SL, self.low_quality_y_SL,
            self.low_quality_x_perturbation, self.low_quality_y_perturbation, self.low_list_straylight_max51, self.low_list_straylight_max53,
            self.low_list_perturbation_max51, self.low_list_perturbation_max53)

        (low_list_y_max, self.low_indices_to_keep_53, self.low_indices_to_remove_53, self.low_widths_SL_51,
         self.low_widths_perturbation_51, self.low_widths_SL_53, self.low_widths_perturbation_53,
         self.sl_points_y, self.sl_points_x, self.low_scans_comparison_treated_signal_x,
         self.low_scans_comparison_treated_signal_y, self.low_scans_comparison_treated_signal_x_51,
         self.low_scans_comparison_treated_signal_y_51, self.low_x_ratio_values, self.low_y_ratio_values,
         self.low_y_max_values, self.low_x_max_values, self.low_scans_done, self.low_list_straylight_barycenter51,
         self.low_list_perturbation_barycenter51, self.low_list_straylight_barycenter53,
         self.low_list_perturbation_barycenter53, self.low_quality_x_SL, self.low_quality_y_SL,
         self.low_quality_x_perturbation, self.low_quality_y_perturbation, self.low_list_straylight_max51,
            self.low_list_straylight_max53, self.low_list_perturbation_max51, self.low_list_perturbation_max53) = (
         self.barycenter_processor.compare_barycenters(X_CONDITION_BARYCENTER_COMPARISON,
                                                          Y_CONDITION_BARYCENTER_COMPARISON))


        self.low_list_perturbation_barycenter51_spec_freq = [
                [(val[0] / 5.878e-2, val[1]) for val in sublist]
                for sublist in self.low_list_perturbation_barycenter51
            ]

        self.low_list_perturbation_barycenter53_spec_freq = [
                [(val[0] / 5.656e-2, val[1]) for val in sublist]
                for sublist in self.low_list_perturbation_barycenter53
            ]

        self.low_list_straylight_barycenter51_spec_freq = [
                [(val[0] / 5.878e-2, val[1]) for val in sublist]
                for sublist in self.low_list_straylight_barycenter51
            ]

        self.low_list_straylight_barycenter53_spec_freq = [
                [(val[0] / 5.656e-2, val[1]) for val in sublist]
                for sublist in self.low_list_straylight_barycenter53
            ]

        print("\n")
        printing(
            f"Number of SL pics detected between 5.1 and 5.3 GHz/s scans in low frequencies : "
            f"{len(low_list_y_max)}\n "
            f"Number of removed points in low frequencies: {len(self.low_indices_to_remove_53)}",
            100, "-")

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def detect_peaks(self, noise_floor_values: object, x, y, height_offset: object = 0) -> object:
        """
        Detect local maxima using find_peaks function.

        :param: noise_floor_values: Noise level threshold calculated using the Riemann method.
        :param: height_offset:

        :returns:
        - filtered_peaks_x: x-coordinates of detected peaks.
        - filtered_peaks_y: y-coordinates of detected peaks.
        - len(filtered_peaks_x): Number of detected peaks.
        """

        # Define the threshold based on the noise floor and a specified offset
        noise_floor_threshold = noise_floor_values + height_offset

        # Find peaks with the specified hieght
        peaks, _ = find_peaks(y, height=noise_floor_threshold, distance=2)

        # Filter peaks based on the noise floor threshold
        filtered_peaks_x = x[peaks]
        filtered_peaks_y = y[peaks]

        #new_y = y - noise_floor_threshold

        # Calculate the full width at half maximum (FWHM) for each detected peak
        results_half = peak_widths(y, peaks, rel_height=0.5)
        widths = results_half[0]  # Extract the FWHM for each peak

        print(f"Number of peaks found: {len(filtered_peaks_x)}")

        # Return the peaks and their heights
        return filtered_peaks_x, filtered_peaks_y, widths, len(filtered_peaks_x)

    ####################################################################################################################
    ####################################################################################################################

    def create_plot_sl(self, xaxis: object, yaxis: object, xlabel: object, ylabel: object, canal: object) -> object:
        """
        Create the basic elements of a graph such as title, labels and axis...

        :param xaxis: x data to plot
        :param yaxis: y data to plot
        :param xlabel: x-axis legend
        :param ylabel: y-axis legend
        :param canal: Canal which is actually plotted

        :return: void
        """

        # --------------------------------------------------------------------------------------------------------------
        # Graph style initialization

        sns.set_style("ticks")

        if self.desktop.width() <= 1800 or self.desktop.height() <= 900:
            sns.set_context("notebook", font_scale=0.9)
        else:
            sns.set_context("talk", font_scale=0.9)

        # --------------------------------------------------------------------------------------------------------------
        # Graph initialization

        # Initialize a Figure and Axes for plotting
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.gca()
        self.ax.grid(True)

        # Add a title to the graph using a helper method and the selected folder name
        self.add_title(self.ax, self.folder_scrollbox_sl)

        # Set the labels for the x and y axes
        self.ax.set_xlabel(xlabel, fontsize=14)
        self.ax.set_ylabel(ylabel, fontsize=14)

        # Adjust the layout of the figure to provide more space at the bottom
        self.figure.subplots_adjust(bottom=0.2)

        # Add the canvas to the layout_graph
        self.layout_graph.addWidget(self.canvas)

        # --------------------------------------------------------------------------------------------------------------
        # Logarithmic scale initialization

        if self.checkbox_logy.isChecked():
            self.ax.set_yscale('log')
        if self.checkbox_logx.isChecked():
            self.ax.set_xscale('log')

        # --------------------------------------------------------------------------------------------------------------
        # Plot labeling

        # Utilisation d'une expression régulière pour extraire "AI 54"
        match = re.search(r"AI \d+", canal)
        if match:
            canal = match.group(0)  # "AI 54"
            print(canal)  # Résultat : "AI 54"
        else:
            print("Aucun motif trouvé.")

        # Create a label for the plot legend using various instance attributes
        label = f'{self.record_type}, {self.qpr}, {canal}, {self.folder_scrollbox_sl.currentText()[:8]}'
        self.legends.append(label)

        # --------------------------------------------------------------------------------------------------------------
        # 5.1 Data plot

        handle = None

        # If the line plot option is checked, plot the x and y data
        if self.checkbox_data_plotline.isChecked():
            handle = sns.lineplot(x=xaxis, y=yaxis, ax=self.ax, linewidth=1.5, label=label)

            # Get the color of the last plotted line to maintain consistency
            line_handle = handle.get_lines()[-1]
            self.line_color = line_handle.get_color()

        else:
            self.line_color = None

        # --------------------------------------------------------------------------------------------------------------
        # 5.1 Barycenter plot

        if self.checkbox_barycenters.isChecked():

            barycenters = np.array(self.barycenters_list[self.count])
            low_barycenters = np.array(self.low_barycenters_list[self.count])
            # Concatenation of the two lists
            all_barycenters = np.concatenate((low_barycenters, barycenters), axis=0)

            if all_barycenters.size > 0:
                handle = sns.scatterplot(x=all_barycenters[:, 0], y=all_barycenters[:, 1], ax=self.ax,
                                         color=self.line_color,
                                         label=f"Barycenter ({POINTS_CALCUL_BARYCENTRE}pts) {self.record_type[:8]}")
            else:
                print("Not enough barycenter in the list to be plotted (<= 0).")

        # --------------------------------------------------------------------------------------------------------------
        # 5.1 Noise floor plot

        if self.checkbox_noise_floor.isChecked():

            sns.lineplot(x=self.noise_floor_freqs, y=self.noise_floor_values, ax=self.ax, color=self.line_color,
                         label=[f"Noise floor {self.record_type[:8]}"])

            sns.lineplot(x=self.low_freqs, y=self.fit_amplitude_optimal, ax=self.ax, color=self.line_color)

        # --------------------------------------------------------------------------------------------------------------
        # Finalize plot

        self.canvas.draw()

    ####################################################################################################################
    ####################################################################################################################

    def show_filtered_spectrum(self):
        """ Clear graph layout before plotting again

        :return: void
        """

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

            :param: x_data: Data for the x-axis.
            :param: y_data: Data for the y-axis.
            :param: x_label: Label for the x-axis.
            :param: y_label: Label for the y-axis.
            :param: canal: Canal information.

            :return: void
            """

            self.create_plot_sl(x_data, y_data, x_label, y_label, canal)
            self.layout_graph.addWidget(NavigationToolbar(self.canvas, self))

        # ==============================================================================================================
        # Call the inside function

        # Concatenate low and high frequencies to obtain a complete x-axis.
        self.freqs_tot = np.concatenate((self.low_xaxis, self.xaxis[1:]))
        # Concatenate the data associated with low and high frequencies to obtain the complete y-axis.
        self.data_tot = np.concatenate((self.low_yaxis, self.yaxis[1:]))

        # Check if freqs_tot is a list or a numpy array
        if isinstance(self.freqs_tot, (list, np.ndarray)):
            # Convert to pandas series
            self.freqs_tot = pd.Series(self.freqs_tot)
        else:
            # Convert to pandas series and encapsulate scalars in a list
            self.freqs_tot = pd.Series([self.freqs_tot])

        # Check if data_tot is a list or a numpy array
        if isinstance(self.data_tot, (list, np.ndarray)):
            # Convert to pandas series
            self.data_tot = pd.Series(self.data_tot)
        else:
            # Convert to pandas series and encapsulate scalars in a list
            self.data_tot = pd.Series([self.data_tot])

        add_plot(pd.Series(self.freqs_tot), pd.Series(self.data_tot), self.x_axis_legend, self.y_axis_legend,
                 self.canal)

    ####################################################################################################################
    ####################################################################################################################

    def overlay_filtered_plot(self):
        """
        Overlay a new plot on the existing plot with a different color.

        :return: void
        """

        # ==============================================================================================================

        def add_overlay_plot(x_data, y_data, canal):
            """
            Create and add an overlay plot to the existing plot.

            :param: x_data: Data for the x-axis.
            :param: y_data: Data for the y-axis.
            :param: canal: Canal information.

            :return: void
            """

            match = re.search(r"AI \d+", canal)
            if match:
                canal = match.group(0)  # "AI 54"
                print(canal)  # Résultat : "AI 54"
            else:
                print("Aucun motif trouvé.")

            # If no scans have been completed, create a new plot with relevant data.
            if not self.scans_done:

                # Construct a label for the current plot based on the record type, QPR, canal, and selected folder.
                label = f'{self.record_type}, {self.qpr}, {canal}, {self.folder_scrollbox_sl.currentText()[:8]}'
                self.list_labels.append(label)

                # If more than one label exists, append a legend indicating scan parameters.
                if len(self.list_labels) > 1:
                    self.legends.append(
                        f'Scan 5.1GHz/s, {self.qpr}, {canal}, {self.folder_scrollbox_sl.currentText()[:8]}')

                # Add a title to the plot using the current folder name.
                self.add_title(self.ax, self.folder_scrollbox_sl)

                # ------------------------------------------------------------------------------------------------------
                # Plot the selected signal in line

                if self.checkbox_data_plotline.isChecked():

                    # Plot the data with a specific label and store the color of the plot line.
                    handle = sns.lineplot(x=x_data, y=y_data, ax=self.ax, linewidth=1.5,
                                          label=label)
                    line_handle = handle.get_lines()[-1]
                    self.line_color = line_handle.get_color()

                else:

                    # If no plot line is added, set line color to None.
                    self.line_color = None

                # ------------------------------------------------------------------------------------------------------
                # Plot the pics' barycenter of the selected signal in line

                if self.checkbox_barycenters.isChecked():

                    barycenters = np.array(self.barycenters_list[self.count])
                    low_barycenters = np.array(self.low_barycenters_list[self.count])
                    # Concatenation of the two lists
                    all_barycenters = np.concatenate((low_barycenters, barycenters), axis=0)

                    if all_barycenters.size > 0:

                        # Scatter plot for barycenters with the same color as the signal line.
                        sns.scatterplot(x=all_barycenters[:, 0], y=all_barycenters[:, 1], ax=self.ax, color=self.line_color,
                                        label=f"Barycenter ({POINTS_CALCUL_BARYCENTRE}pts) "f"{self.record_type[:8]}")
                    else:
                        print("Not enough barycenter in the list to be plotted (<= 0).")

                # ------------------------------------------------------------------------------------------------------
                # Plot the noise floor of the selected signal in line

                if self.checkbox_noise_floor.isChecked():

                    sns.lineplot(x=self.noise_floor_freqs, y=self.noise_floor_values, ax=self.ax, color=self.line_color,
                                 label=[f"Noise floor {self.record_type[:8]}"])
                    sns.lineplot(x=self.low_freqs, y=self.fit_amplitude_optimal, ax=self.ax,
                                 color=self.line_color)

            # ==========================================================================================================
            # If scans have been completed, adjust the plot based on the treated signal.

            else:
                if self.checkbox_treated_signal.isChecked():

                    scans_comparison_treated_signal_x_tot = (self.low_scans_comparison_treated_signal_x[-1] +
                                                             self.scans_comparison_treated_signal_x[-1])

                    scans_comparison_treated_signal_y_tot = (self.low_scans_comparison_treated_signal_y[-1] +
                                                             self.scans_comparison_treated_signal_y[-1])

                    self.scans_comparison_treated_signal_x_tot.append(scans_comparison_treated_signal_x_tot)

                    self.scans_comparison_treated_signal_y_tot.append(scans_comparison_treated_signal_y_tot)

                    scans_comparison_treated_signal_x_tot_51 = (self.low_scans_comparison_treated_signal_x_51[-1] +
                                                             self.scans_comparison_treated_signal_x_51[-1])

                    scans_comparison_treated_signal_y_tot_51 = (self.low_scans_comparison_treated_signal_y_51[-1] +
                                                             self.scans_comparison_treated_signal_y_51[-1])

                    self.scans_comparison_treated_signal_x_tot_51.append(scans_comparison_treated_signal_x_tot_51)

                    self.scans_comparison_treated_signal_y_tot_51.append(scans_comparison_treated_signal_y_tot_51)

                    if self.line_color is not None:

                        # Modify the color for the treated signal by adjusting the RGB values of the existing line color
                        r, g, b = (int(self.line_color[1:3], 16), int(self.line_color[3:5], 16),
                               int(self.line_color[5:], 16))
                        r = min(max(r + int(100 / random.randint(1, 4)), 0), 255)
                        g = min(max(g - int(200 / random.randint(1, 4)), 0), 255)
                        b = min(max(b - int(200 / random.randint(1, 4)), 0), 255)

                        # Convert back to hexadecimal color code for plotting.
                        new_hex_color = f'#{r:02X}{g:02X}{b:02X}'

                        # Scatter plot the treated signal with the new color.
                        sns.scatterplot(x=self.scans_comparison_treated_signal_x_tot[-1],
                                    y=self.scans_comparison_treated_signal_y_tot[-1], ax=self.ax, s=75,
                                    color=new_hex_color,
                                    label=f"{self.record_type}, "f"{self.folder_scrollbox_sl.currentText()[:8]}, "
                                          f"{self.qpr}, treated signal")

                    else:
                        sns.scatterplot(x=self.scans_comparison_treated_signal_x_tot[-1],
                                    y=self.scans_comparison_treated_signal_y_tot[-1], ax=self.ax, s=75,
                                    label=f"{self.record_type}, "f"{self.folder_scrollbox_sl.currentText()[:8]}, "
                                          f"{self.qpr}, treated signal")

                        print("self.line_color is set to None.")

        # ==============================================================================================================
        # Call of functions to plot in the data_analysis tab

                # Get the current plot's collections to handle different plot elements.
                collections = self.ax.collections

                # Initialize list to store colors used in the plot.
                self.plot_colors = []

                # If there are plot collections, extract their colors.
                if collections:
                    for collection in collections:
                        facecolors = collection.get_facecolor()
                        if facecolors.size > 0:
                            for color in facecolors:
                                rgb_color = tuple(color[:3])
                                if rgb_color not in self.plot_colors:
                                    self.plot_colors.append(rgb_color)
                else:
                    print("No curves plotted.")

                # If a data plot line exists, remove its color from the list.
                if self.checkbox_data_plotline.isChecked():
                    self.plot_colors.pop(0)

                # ------------------------------------------------------------------------------------------------------
                # Compare y max and y ratio in the DataAnalysis tab

                # Store y max values from the 5.1 scan of the last treatment performed
                y_max_values_all = np.concatenate((self.low_y_max_values[-1], self.y_max_values[-1]), axis=0)

                # Add these values to the list of all treatments performed
                self.y_max_values_list.append(y_max_values_all)

                # Store y ratio values from the 5.1 scan of the last treatment performed
                y_ratio_values_all = np.concatenate((self.low_y_ratio_values[-1], self.y_ratio_values[-1]), axis=0)

                # Add these values to the list of all treatments performed
                self.y_ratio_values_list.append(y_ratio_values_all)

                # Call a function from DataAnalysis Class
                self.data_analysis_tab.compare_pos_plot(self.y_max_values_list, self.y_ratio_values_list,
                                                        self.plot_colors,
                                                        'Comparison of y_ratio and y_max (Barycentre over 3 pts)',
                                                        'y max values', '|y51/y53 - 1|',
                                                        self.list_labels, 1, self.count/2)

                # ------------------------------------------------------------------------------------------------------
                # Compare y max and quality percentage for x and y in the DataAnalysis tab

                # Store quality percentage from the 5.1 scan of the last treatment performed
                quality_x_SL_all = np.concatenate((self.low_quality_x_SL[-1],self.quality_x_SL[-1]), axis=0)

                quality_y_SL_all = np.concatenate((self.low_quality_y_SL[-1], self.quality_y_SL[-1]), axis=0)

                # Add these values to the list of all treatments performed
                self.quality_x_SL_list.append(quality_x_SL_all)
                self.quality_y_SL_list.append(quality_y_SL_all)

                # Store y max values from the 5.1 scan of the last treatment performed
                x_max_values_all = np.concatenate((self.low_x_max_values[-1], self.x_max_values[-1]), axis=0)

                # Add these values to the list of all treatments performed
                self.x_max_values_list.append(x_max_values_all)

                self.data_analysis_tab.compare_pos_plot(self.x_max_values_list, self.quality_x_SL_list,
                                                        self.plot_colors,
                                                        'Quality percentage for x in function of the position',
                                                        'x corresponding to y max values',
                                                        '1 - |x51/x53 - 1| * 100 (%)',
                                                        self.list_labels, 2, self.count/2)

                self.data_analysis_tab.compare_pos_plot(self.y_max_values_list, self.quality_y_SL_list,
                                                        self.plot_colors,
                                                        'Quality percentage for y in function of the amplitude (Vpk/VDC)',
                                                        'y max values',
                                                        '1 - |y51/y53 - 1| * 100 (%)',
                                                        self.list_labels, 3, self.count/2)

                self.data_analysis_tab.compare_pos_plot(self.y_max_values_list, self.quality_x_SL_list,
                                                        self.plot_colors,
                                                        'Quality percentage for x in function of the amplitude (Vpk/VDC)',
                                                        'y max values',
                                                        '1 - |x51/x53 - 1| * 100 (%)',
                                                        self.list_labels, 4, self.count / 2)

                self.data_analysis_tab.compare_pos_plot(self.x_max_values_list, self.quality_y_SL_list,
                                                        self.plot_colors,
                                                        'Quality percentage for y in function of the position',
                                                        'x corresponding to y max values',
                                                        '1 - |y51/y53 - 1| * 100 (%)',
                                                        self.list_labels, 5, self.count / 2)

        # ==============================================================================================================
        # Main function call

        # Call the add_overlay_plot function with the relevant data to create the overlay plot.
        # Concatenate low and high frequencies to obtain a complete x-axis.
        self.freqs_tot = np.concatenate((self.low_xaxis, self.xaxis[1:]))
        # Concatenate the data associated with low and high frequencies to obtain the complete y-axis.
        self.data_tot = np.concatenate((self.low_yaxis, self.yaxis[1:]))

        # Check if freqs_tot is a list or a numpy array
        if isinstance(self.freqs_tot, (list, np.ndarray)):
            # Convert to pandas series
            self.freqs_tot = pd.Series(self.freqs_tot)
        else:
            # Convert to pandas series and encapsulate scalars in a list
            self.freqs_tot = pd.Series([self.freqs_tot])  # Encapsulate scalars in a list

        # Check if data_tot is a list or a numpy array
        if isinstance(self.data_tot, (list, np.ndarray)):
            # Convert to pandas series
            self.data_tot = pd.Series(self.data_tot)
        else:
            # Convert to pandas series and encapsulate scalars in a list
            self.data_tot = pd.Series([self.data_tot])  # Encapsulate scalars in a list

        add_overlay_plot(pd.Series(self.freqs_tot), pd.Series(self.data_tot), self.canal)

        self.canvas.draw()