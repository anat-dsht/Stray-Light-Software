# coding: utf-8
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__authors__ = "Anatole Desouhant"
__contact__ = "anatole.desouhant@insa-lyon.fr"
__copyright__ = "ARTEMIS, Côte d'Azur Observatory"
__date__ = "2024-10-09"
__version__ = "1.0.0"
__status__ = "Production"
__privacy__ = "Confidential"

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

from PySide6.QtWidgets import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import os
import platform
import subprocess
import seaborn as sns
import numpy as np
from typing import Optional, List
import pandas as pd
from PySide6.QtCore import Signal
from scipy.signal import find_peaks
from numpy import ndarray
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from Code.SLS_Interface.header_dictionary import removewidget, printing
from TreatmentTab import (StrayLightTreatment, POINTS_CALCUL_BARYCENTRE, X_CONDITION_BARYCENTER_COMPARISON,
                          Y_CONDITION_BARYCENTER_COMPARISON)
from VisualisationTab import (add_legends, OPD_BANDWIDTH, FREQ_BANDWIDTH, LOW_OPD_BANDWIDTH_NOISE_MODEL,
                              LOW_FREQ_BANDWIDTH_NOISE_MODEL)
from Data_Extraction import DataExtraction
from peak_clustering import PeakClustering
from comparison_5153 import ScansComparison
from data_analysis import DataAnalysis

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################

# Initialization of constants used by all the classes
X_CONDITION_BARYCENTER_COMPARISON_CALIBRATOR = 0.0001
Y_CONDITION_BARYCENTER_COMPARISON_CALIBRATOR: float = 0.05

########################################################################################################################
########################################################################################################################
########################################################################################################################

class CalibratorTab(StrayLightTreatment):
    analysis_calibrator: DataExtraction
    event_occurred_calibrator = Signal()

    def __init__(self):
        super().__init__()
        print("Initializing CalibratorTab class")

        # --------------------------------------------------------------------------------------------------------------
        #                                                 Others
        # --------------------------------------------------------------------------------------------------------------

        # Define which tab we are working on
        self.tab_name: str = "CalibratorTab"

        # Flag to identify once the treatment is over
        self.scans_done_calibrator: bool = False
        self.low_scans_done_calibrator: bool = False

        # Call the right function depends on the count value
        self.count_calibrator: int = 0

        self.record_type_calibrator: str = 'None'

        # --------------------------------------------------------------------------------------------------------------
        #                                                 Noise floor
        # --------------------------------------------------------------------------------------------------------------

        # Noise floor X&Y components
        self.noise_floor_values_calibrator: Optional[np.ndarray] = None
        self.noise_floor_freqs_calibrator: Optional[np.ndarray] = None
        self.log_fit_amplitude_optimal_calibrator: Optional[np.ndarray] = None
        self.fit_amplitude_optimal_calibrator: Optional[np.ndarray] = None
        self.peak_threshold_calibrator: Optional[np.ndarray] = None

        # Noise floor's mean value
        self.mean_value_noise_floor_calibrator: Optional[float] = None

        # --------------------------------------------------------------------------------------------------------------
        #                                                Clusters
        # --------------------------------------------------------------------------------------------------------------

        # Clusters for barycenter calculation (3 points only)
        self.barycenter_clusters_list_calibrator: List[np.ndarray] = []
        self.barycenter_clusters_calibrator: Optional[np.ndarray] = None
        self.low_barycenter_clusters_list_calibrator: List[np.ndarray] = []
        self.low_barycenter_clusters_calibrator: Optional[np.ndarray] = None

        # Clusters to get whole peaks shapes
        self.full_clusters_list_calibrator: List[np.ndarray] = []
        self.full_clusters_peaks_calibrator: Optional[np.ndarray] = None
        self.low_full_clusters_list_calibrator: List[np.ndarray] = []
        self.low_full_clusters_peaks_calibrator: Optional[np.ndarray] = None

        # Full clusters of all Stray Light peaks
        self.sl_points_y_calibrator: Optional[np.ndarray] = None
        self.sl_points_x_calibrator: Optional[np.ndarray] = None
        self.low_sl_points_y_calibrator: Optional[np.ndarray] = None
        self.low_sl_points_x_calibrator: Optional[np.ndarray] = None

        # --------------------------------------------------------------------------------------------------------------
        #                                                Graph parameters
        # --------------------------------------------------------------------------------------------------------------

        # Statistical variables for the DataAnalysis tab
        self.x_ratio_values_calibrator: List[float] = []
        self.y_max_values_calibrator: List[float] = []
        self.y_ratio_values_calibrator: List[float] = []
        self.low_x_ratio_values_calibrator: List[float] = []
        self.low_y_max_values_calibrator: List[float] = []
        self.low_y_ratio_values_calibrator: List[float] = []
        self.y_max_values_list_calibrator: List[ndarray] = []
        self.y_ratio_values_list_calibrator: List[ndarray] = []
        self.x_ratio_values_list_calibrator: List[ndarray] = []
        self.quality_percentage_list_calibrator: List[ndarray] = []
        self.pic_number_calibrator: List[float] = []
        self.low_pic_number_calibrator: List[float] = []
        self.pic_number_list_calibrator: List[ndarray] = []
        self.peak_width_51_calibrator: List[ndarray] = []
        self.peak_width_53_calibrator: List[ndarray] = []
        self.low_peak_width_51_calibrator: List[ndarray] = []
        self.low_peak_width_53_calibrator: List[ndarray] = []

        # --------------------------------------------------------------------------------------------------------------
        #                                                Signal data
        # --------------------------------------------------------------------------------------------------------------

        # Selected signal data
        self.freqs_calibrator: Optional[np.ndarray] = None
        self.data_calibrator: Optional[np.ndarray] = None
        self.freqs_calibrator_51: Optional[np.ndarray] = None
        self.data_calibrator_51: Optional[np.ndarray] = None
        self.log_freqs_calibrator: Optional[np.ndarray] = None
        self.log_data_calibrator: Optional[np.ndarray] = None
        self.low_freqs_calibrator: Optional[np.ndarray] = None
        self.low_data_calibrator: Optional[np.ndarray] = None
        self.low_freqs_calibrator_51: Optional[np.ndarray] = None
        self.low_data_calibrator_51: Optional[np.ndarray] = None
        self.freqs_tot_calibrator: Optional[np.ndarray] = None
        self.data_tot_calibrator: Optional[np.ndarray] = None

        # Peaks detected
        self.peaks_x_calibrator: Optional[np.ndarray] = None
        self.peaks_y_calibrator: Optional[np.ndarray] = None
        self.peaks_low_x_calibrator: Optional[np.ndarray] = None
        self.peaks_low_y_calibrator: Optional[np.ndarray] = None

        # --------------------------------------------------------------------------------------------------------------
        #                                                Peaks
        # --------------------------------------------------------------------------------------------------------------

        # Output treated signal of the Scan comparison process
        self.scans_comparison_treated_signal_y_calibrator: List[List[float]] = []
        self.scans_comparison_treated_signal_x_calibrator: List[List[float]] = []
        self.scans_comparison_treated_signal_y_calibrator_51: List[List[float]] = []
        self.scans_comparison_treated_signal_x_calibrator_51: List[List[float]] = []
        self.low_scans_comparison_treated_signal_y_calibrator: List[List[float]] = []
        self.low_scans_comparison_treated_signal_x_calibrator: List[List[float]] = []
        self.low_scans_comparison_treated_signal_y_calibrator_51: List[List[float]] = []
        self.low_scans_comparison_treated_signal_x_calibrator_51: List[List[float]] = []
        self.scans_comparison_treated_signal_x_tot_calibrator: List[List[float]] = []
        self.scans_comparison_treated_signal_y_tot_calibrator: List[List[float]] = []

        # Stray Light peaks barycenter
        self.list_straylight_barycenter51_calibrator: List[List[float]] = []
        self.list_straylight_barycenter53_calibrator: List[List[float]] = []
        self.low_list_straylight_barycenter51_calibrator: List[List[float]] = []
        self.low_list_straylight_barycenter53_calibrator: List[List[float]] = []
        self.list_straylight_barycenter51_calibrator_tot: List[List[float]] = []
        self.list_straylight_barycenter53_calibrator_tot: List[List[float]] = []

        # Perturbation peak barycenter
        self.list_perturbation_barycenter51_calibrator: List[List[float]] = []
        self.list_perturbation_barycenter53_calibrator: List[List[float]] = []
        self.low_list_perturbation_barycenter51_calibrator: List[List[float]] = []
        self.low_list_perturbation_barycenter53_calibrator: List[List[float]] = []
        self.list_perturbation_barycenter51_calibrator_spec_freq: List[List[float]] = []
        self.list_perturbation_barycenter53_calibrator_spec_freq: List[List[float]] = []
        self.low_list_perturbation_barycenter51_calibrator_spec_freq: List[List[float]] = []
        self.low_list_perturbation_barycenter53_calibrator_spec_freq: List[List[float]] = []
        self.list_perturbation_barycenter51_calibrator_tot_spec_freq: List[List[float]] = []
        self.list_perturbation_barycenter53_calibrator_tot_spec_freq: List[List[float]] = []

        # All barycenter storages
        self.barycenters_calibrator: Optional[np.ndarray] = None
        self.barycenters_list_calibrator: List[np.ndarray] = []
        self.low_barycenters_calibrator: Optional[np.ndarray] = None
        self.low_barycenters_list_calibrator: List[np.ndarray] = []

        # --------------------------------------------------------------------------------------------------------------
        #                                                 File paths
        # --------------------------------------------------------------------------------------------------------------

        self.ref_file_path = os.path.abspath(os.path.join(self.data_path, "28092023-Testscancalib", "Processed"))

        if not os.path.exists(self.ref_file_path):
            raise FileNotFoundError(f"The path {self.ref_file_path} doesn't exist.")

        # --------------------------------------------------------------------------------------------------------------
        #                                                User Interface widgets
        # --------------------------------------------------------------------------------------------------------------

        #Texts
        self.record_calibrator_text: QLabel = QLabel("Record :")

        # Buttons
        self.treat_signal_button: QPushButton = QPushButton("Treat signal")
        self.access_data_button: QPushButton = QPushButton("Access data")

        # Interface user widgets
        self.folder_scrollbox: QComboBox = QComboBox()
        self.record_calibrator_scrollbox: QComboBox = QComboBox()

        # Plotting options
        self.checkbox_data_plotline_calibrator: QCheckBox = QCheckBox("Scan monitor/Reference monitor")
        self.checkbox_noise_floor: QCheckBox = QCheckBox("Noise floor")
        self.checkbox_barycenters: QCheckBox = QCheckBox("Barycenters")

        # --------------------------------------------------------------------------------------------------------------
        #                                                 Class instances
        # --------------------------------------------------------------------------------------------------------------

        # Instance of the PeakClustering class
        self.peak_clustering_calibrator: Optional[PeakClustering] = None

        # Instance of the ScansComparison class
        self.barycenter_processor_calibrator: Optional[ScansComparison] = None

        # Instance of the DataAnalysis class
        self.data_analysis_tab_calibrator: DataAnalysis = DataAnalysis()

        # --------------------------------------------------------------------------------------------------------------
        #                                                Specific calls
        # --------------------------------------------------------------------------------------------------------------

        self.setup_calibrator_ui()
        self.setup_connections_calibrator()

    ####################################################################################################################
    ####################################################################################################################

    def setup_calibrator_ui(self):

        # Add element to the tab layout
        self.layout.addWidget(self.treat_signal_button, 1, 0)
        self.layout.addWidget(self.access_data_button, 2, 0)
        self.layout.addWidget(self.help_button, 4, 0)

        # Add element to the parameter layout
        self.parameter_layout.addWidget(self.folder_scrollbox, 0, 1)

        # Add sub folder widgets to the parameter layout
        self.parameter_layout.addWidget(self.record_calibrator_text, 1, 0)
        self.parameter_layout.addWidget(self.record_calibrator_scrollbox, 1, 1)

        # Add element to the plotting options layout
        self.plot_options_layout.addWidget(self.checkbox_noise_floor, 1, 0)
        self.plot_options_layout.addWidget(self.checkbox_barycenters, 2, 0)
        self.plot_options_layout.addWidget(self.checkbox_data_plotline_calibrator, 3, 0)

        # Initialize the plotting options checkboxes states
        self.checkbox_data_plotline_calibrator.setChecked(True)
        self.checkbox_noise_floor.setChecked(True)
        self.checkbox_barycenters.setChecked(True)

        # Remove element which were defined in the Treatment tab
        removewidget(self.parameter_layout, self.folder_scrollbox_sl)
        removewidget(self.parameter_layout, self.checkbox_plot_options_text)
        removewidget(self.plot_options_layout, self.checkbox_data_plotline_53)
        removewidget(self.plot_options_layout, self.checkbox_data_plotline_51)
        removewidget(self.plot_options_layout, self.checkbox_treated_signal_53)
        removewidget(self.plot_options_layout, self.checkbox_treated_signal_51)
        removewidget(self.plot_options_layout, self.checkbox_barycenters_53)
        removewidget(self.plot_options_layout, self.checkbox_barycenters_51)
        removewidget(self.plot_options_layout, self.checkbox_noise_floor_53)
        removewidget(self.plot_options_layout, self.checkbox_noise_floor_51)
        removewidget(self.layout, self.button_access_data)
        removewidget(self.layout, self.button_treat_all_signals)

        # Remove element which were defined in the Data Visualisation tab
        removewidget(self.parameter_layout, self.canal_text)
        removewidget(self.parameter_layout, self.canal_listwidget)
        removewidget(self.parameter_layout, self.title_text)
        removewidget(self.parameter_layout, self.text_scrollbox)
        removewidget(self.parameter_layout, self.text_zone)

    ####################################################################################################################
    ####################################################################################################################

    def setup_connections_calibrator(self):
        """"""

        # --------------------------------------------------------------------------------------------------------------
        # Scroll boxes

        self.fill_scrollbox_calibrator(self.folder_scrollbox)

        # Update record_scrollbox when selection in measurement_scrollbox have changed
        self.folder_scrollbox.currentIndexChanged.connect(
            lambda: self.update_data_scrollbox_calibrator(self.folder_scrollbox, self.record_calibrator_scrollbox))

        # --------------------------------------------------------------------------------------------------------------
        # Buttons

        # Call the main function to process the signal
        self.treat_signal_button.clicked.connect(self.call_functions_calibrator)

        # Access treated data
        self.access_data_button.clicked.connect(self.analyse_data_calibrator)

        # User manual button
        self.help_button.clicked.connect(self.help_dialog)

    ####################################################################################################################
    ####################################################################################################################

    def fill_scrollbox_calibrator(self, scrollbox):
        """
        Fill scroll box in function of parent scroll box selection.

        :param scrollbox: Children scroll box to fill.

        :return: int
        """

        # Get a list containing all folders located at the computed path
        if scrollbox == self.folder_scrollbox:
            list_name_documents = os.listdir(self.data_path)
            folders_name = [name for name in list_name_documents if
                            os.path.isdir(os.path.join(self.data_path, name))]

            # Fill the children scroll box
            for name in folders_name:
                scrollbox.addItem(str(name))

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def update_data_scrollbox_calibrator(self, initial_scroll_box, final_scroll_box):
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
        if initial_scroll_box == self.folder_scrollbox:
            path = os.path.join(self.data_path, initial_scroll_box_selection)
        elif initial_scroll_box == self.record_calibrator_scrollbox:
            path = os.path.join(self.data_path, self.folder_scrollbox.currentText(), initial_scroll_box_selection)
        self.add_subfolder_names_to_scrollbox_calibrator(path, final_scroll_box)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def add_subfolder_names_to_scrollbox_calibrator(self, path, scrollbox):
        """
        Browse the OS path and add every file to a list to implement them in a scroll box.

        :param path: Path of the files to browse and add in the scroll box
        :param scrollbox: Output scroll box

        :return: list of the folder names found
        """

        folders_name = []
        pre_folders_name = []

        # Clear to overwrite with new values
        scrollbox.clear()

        # Condition to know wether it's a temporal or frequential signal
        temporal_or_frequential = "Processed"

        # If there is folder at path
        if os.path.isdir(path):

            # ----------------------------------------------------------------------------------------------------------
            # Measurement

            # Add every measurement folders to the measurement scrollbox
            if scrollbox == self.folder_scrollbox:
                list_name_documents = os.listdir(path)
                folders_name = [name for name in list_name_documents if
                                os.path.isdir(os.path.join(path, name))]

            # ----------------------------------------------------------------------------------------------------------
            # Record

            # Add every record's type to the record scrollbox
            elif scrollbox == self.record_calibrator_scrollbox:
                list_name_documents = os.listdir(path)

                # Filter folders that contain "Scan51" or "Scan53" in their name
                pre_folders_name = [name for name in list_name_documents  if
                                    os.path.isdir(os.path.join(path, name)) and ("Scan51" in name or "Scan53" in name)]

                # If there are folders matching the criteria
                if pre_folders_name:
                    for folder in pre_folders_name:
                        folder_path = os.path.join(path, folder)

                        # Check if "Processed" folder exists in the selected folder
                        processed_path = os.path.join(folder_path, temporal_or_frequential)
                        if os.path.isdir(processed_path):
                          # Add the folder name to the output scroll box
                          folders_name.append(folder)
                else:
                    print("No folders containing 'Scan51' or 'Scan53' found.")
                    scrollbox.addItem("No relevant folders found.")

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

    def call_functions_calibrator(self):
        """
        Main function that call all other steps of the process.

        :return: int
        """

        printing("Starting processing of files", 50, "=")

        file_paths = []

        # Browse scans that fill user choices
        file_path1 = self.process_files_calibrator()
        file_paths.append(file_path1)

        # If "Scan51" is in file_path1, define file_path2
        if "Scan51" in file_path1:
            # Search for a file containing ‘Scan51’ in the self.ref_file_path directory
            for file_name in os.listdir(self.ref_file_path):
                if "51" in file_name:
                    file_path2 = os.path.join(self.ref_file_path, file_name)
                    break  # Stop the loop as soon as the corresponding file is found

        # If "Scan53" is in file_path1, define file_path2
        elif "Scan53" in file_path1:
            # Search for a file containing ‘Scan53’ in the self.ref_file_path directory
            for file_name in os.listdir(self.ref_file_path):
                if "53" in file_name:
                    file_path2 = os.path.join(self.ref_file_path, file_name)
                    break  # Stop the loop as soon as the corresponding file is found

        else:
            for file_name in os.listdir(self.ref_file_path):
                file_path2 = os.path.join(self.ref_file_path, file_name)
                break

        file_paths.append(file_path2)

        # If it's not the first time the user click on Launch Process button
        if self.count_calibrator != 0:

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
                self.count_calibrator = 0

                # Reset the DataAnalysis tab
                self.data_analysis_tab_calibrator.create_plot()

                # Re-initialize elements when the user want to overdraw a new plot
                self.y_ratio_values_calibrator = []
                self.low_y_ratio_values_calibrator = []
                self.y_ratio_values_list_calibrator = []
                self.x_ratio_values_list_calibrator = []
                self.list_labels = []
                self.x_ratio_values_list_calibrator = []
                self.x_ratio_values_calibrator = []
                self.low_x_ratio_values_calibrator = []
                self.pic_number_list_calibrator = []
                self.pic_number_calibrator = []
                self.low_pic_number_calibrator = []
                self.barycenters_list_calibrator = []
                self.low_barycenters_list_calibrator = []
                self.barycenter_clusters_list_calibrator = []
                self.low_barycenter_clusters_list_calibrator = []
                self.full_clusters_list_calibrator = []
                self.low_full_clusters_list_calibrator = []
                self.plot_colors = []
                self.list_perturbation_barycenter53_calibrator = []
                self.low_list_perturbation_barycenter53_calibrator = []
                self.list_perturbation_barycenter51_calibrator = []
                self.low_list_perturbation_barycenter51_calibrator = []
                self.list_straylight_barycenter53_calibrator = []
                self.low_list_straylight_barycenter53_calibrator = []
                self.list_straylight_barycenter51_calibrator = []
                self.low_list_straylight_barycenter51_calibrator = []
                self.legends = []

            # ----------------------------------------------------------------------------------------------------------
            # Overlay old and new plots

            elif msg_box.clickedButton() == overlay_plot_button:

                # Keep going for the count value
                self.count_calibrator = self.count_calibrator

        # If it's the first time the user click on Launch Process button
        else:
            self.on_event_data_analyse_calibrator()

        # ==============================================================================================================
        # Get scans data

        # Process each file and collect barycenter
        for file_path in file_paths:
            print("\n")

            # Extract data of the current scan treated
            self.analysis_calibrator = DataExtraction(file_path)
            self.analysis_calibrator.load_data(file_path)
            self.analysis_calibrator.extract_data()

            # Get the canal, linked data and legends for the current scan treated
            self.get_graph_parameters_calibrator(file_path, self.count_calibrator)
            self.record_type_calibrator = add_legends(file_path)

            # Apply the treatment method to the right type of data
            if "OPD" not in self.x_axis_legend:
                self.treatment_method_calibrator(FREQ_BANDWIDTH) & self.low_treatment_method_calibrator(
                    LOW_FREQ_BANDWIDTH_NOISE_MODEL, file_path)
            else:
                self.treatment_method_calibrator(OPD_BANDWIDTH) & self.low_treatment_method_calibrator(
                    LOW_OPD_BANDWIDTH_NOISE_MODEL, file_path)

            # Increment count
            self.count_calibrator += 1

        # ==============================================================================================================
        # Compare both scans

        # Call the scan comparison class
        self.scans_comparison_calibrator()
        self.low_scans_comparison_calibrator()

        # Store peak frequency perturbation from the 5.1 scan of the last treatment performed
        list_straylight_barycenter51_calibrator_tot = self.low_list_straylight_barycenter51_calibrator[-1] + \
                                                      self.list_straylight_barycenter51_calibrator[-1]

        # Add these peaks to the list of all treatments performed
        self.list_straylight_barycenter51_calibrator_tot.append(list_straylight_barycenter51_calibrator_tot)

        # Store peak frequency perturbation from the 5.1 scan of the last treatment performed
        list_straylight_barycenter53_calibrator_tot = self.low_list_straylight_barycenter53_calibrator[-1] + \
                                                      self.list_straylight_barycenter53_calibrator[-1]

        # Add these peaks to the list of all treatments performed
        self.list_straylight_barycenter53_calibrator_tot.append(list_straylight_barycenter53_calibrator_tot)

        # Plot the treated signal
        self.overlay_filtered_plot_calibrator()
        self.scans_done_calibrator = False

        # ==============================================================================================================

        print("\n")
        printing("Treatment successful!", 50, "=")

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def on_event_data_analyse_calibrator(self):
        """
        Create a signal when the process is done to create and display the DataAnalysis tab.

        :return: int
        """

        self.event_occurred_calibrator.emit()

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def treatment_method_calibrator(self, bandwidth):
        """
        Process noise floor, clusters, peak detection, barycenter for scans 5.1 and 5.3.

        :param bandwidth: Bandwidth depends on the signal type.

        :return: int
        """

        printing("Starting treatment method calibrator", 50, "-")

        self.data_calibrator = np.array(self.yaxis_calibrator)
        self.freqs_calibrator = np.array(self.xaxis_calibrator)

        # Compute the noise floor of the signal
        self.noise_floor_freqs_calibrator, self.noise_floor_values_calibrator = self.riemann_method_calibrator(bandwidth)

        # Compute clusters and detect peaks
        self.get_custering_variables_calibrator(self.noise_floor_values_calibrator)

        # Calculate barycenter of the detected peaks
        self.barycenters_list_calibrator.append(self.barycenters_calibrator)

        # Store output data
        self.barycenter_clusters_list_calibrator.append(self.barycenter_clusters_calibrator)
        self.full_clusters_list_calibrator.append(self.full_clusters_peaks_calibrator)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def low_treatment_method_calibrator(self, bandwidth_noise_model, file_path):
        """
        Process noise floor, clusters, peak detection, barycenter for scans 5.1 and 5.3 in low frequencies.

        :param bandwidth_noise_model: Bandwidth depends on the signal type
        :param file_path

        :return: int
        """

        printing("Starting low treatment method calibrator", 50, "-")

        self.low_data_calibrator = np.array(self.low_yaxis_calibrator)
        self.low_freqs_calibrator = np.array(self.low_xaxis_calibrator)

        self.log_data_calibrator = np.log10(self.low_data_calibrator)
        self.log_freqs_calibrator = np.log10(self.low_freqs_calibrator)

        # Compute the noise floor of the signal
        self.log_fit_amplitude_optimal_calibrator, self.fit_amplitude_optimal_calibrator = (
            self.create_noise_model_calibrator(bandwidth_noise_model, self.low_freqs_calibrator,
                                               self.low_data_calibrator, self.log_freqs_calibrator))

        # Compute clusters and detect peaks
        self.get_custering_low_variables_calibrator(self.fit_amplitude_optimal_calibrator)

        # Calculate barycenter of the detected peaks
        self.low_barycenters_list_calibrator.append(self.low_barycenters_calibrator)

        # Store output data
        self.low_barycenter_clusters_list_calibrator.append(self.low_barycenter_clusters_calibrator)
        self.low_full_clusters_list_calibrator.append(self.low_full_clusters_peaks_calibrator)

        # Display results
        if self.count_calibrator == 0:
            self.show_filtered_spectrum_calibrator(file_path)
        elif self.count_calibrator != 0:
            self.overlay_filtered_plot_calibrator(file_path)

        return 0

    ####################################################################################################################
    ####################################################################################################################

    @staticmethod
    def create_noise_model_calibrator(bandwidth, freqs, data, log_freqs):
        """"""

        # Frequency range between bandwidth limits, with a mask to exclude frequencies between 8e-1 and 1.5
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
        print(f"Degré optimal trouvé : {optimal_degree}")

        # Refaire un fit polynomial avec le degré optimal sur toutes les données filtrées
        coeffs_optimal = np.polyfit(filtered_log_freq, filtered_log_amplitude, optimal_degree).astype(np.float64)
        poly_fit_optimal = np.poly1d(coeffs_optimal)

        # Générer la courbe ajustée sur toutes les fréquences en logarithmique
        log_fit_amplitude_optimal = poly_fit_optimal(log_freqs.astype(np.float64))
        fit_amplitude_optimal = np.power(10, log_fit_amplitude_optimal).astype(np.float64)

        return log_fit_amplitude_optimal.astype(np.float64), fit_amplitude_optimal.astype(np.float64)


    ####################################################################################################################
    ####################################################################################################################

    def riemann_method_calibrator(self, bandwidth, percent_to_exclude=30):
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
            mask = (self.freqs_calibrator >= low_freq) & (self.freqs_calibrator < high_freq)

            if np.any(mask):

                # Filter data within the current interval
                interval_data = self.data_calibrator[mask]

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
        interpolated_freqs = np.linspace(bandwidth[0], bandwidth[-1], num=len(self.freqs_calibrator))
        interp_values = interpolation_function(interpolated_freqs)

        print(f"Noise floor fitted with a mean amplitude of: {np.mean(interp_values)}")

        return interpolated_freqs, interp_values

    ####################################################################################################################
    ####################################################################################################################

    def get_custering_variables_calibrator(self, noise_floor_values):
        """
        Function linked to the PeakClustering class that creates clusters for each peak detected in scans 5.1 and 5.3
        during frequency processing.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """

        self.peak_clustering_calibrator = PeakClustering(self.freqs_calibrator, self.data_calibrator)

        xpeak, ypeak, _ = self.detect_peaks_calibrator(noise_floor_values)
        self.barycenters_calibrator, self.barycenter_clusters_calibrator, self.full_clusters_peaks_calibrator = (
            self.peak_clustering_calibrator.group_points_into_peaks(xpeak, ypeak))

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_custering_low_variables_calibrator(self, noise_floor_values):
        """
        Function linked to the PeakClustering class that creates clusters for each peak detected in scans 5.1 and 5.3
        during frequency processing in low frequencies.

        :param noise_floor_values: List of noise floor values get from the riemann method.

        :return: int
        """

        self.peak_clustering_calibrator = PeakClustering(self.low_freqs_calibrator, self.low_data_calibrator)

        low_xpeak, low_ypeak, self.peak_threshold_calibrator = (
            self.detect_peaks_above_noise_model_calibrator(noise_floor_values,
                                                           self.low_freqs_calibrator, self.low_data_calibrator))

        (self.low_barycenters_calibrator, self.low_barycenter_clusters_calibrator,
         self.low_full_clusters_peaks_calibrator) = \
            (self.peak_clustering_calibrator.group_points_into_peaks(low_xpeak, low_ypeak))

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def detect_peaks_calibrator(self, noise_floor_values: object, height_offset: object = 0) -> object:
        """
        Detect local maxima using find_peaks function.

        :param: noise_floor_values: Noise level threshold calculated using the Riemann method.
        :param: height_offset:

        :returns:
        - filtered_peaks_x: x-coordinates of detected peaks.
        - filtered_peaks_y: y-coordinates of detected peaks.
        - len(filtered_peaks_x): Number of detected peaks.
        """

        y = np.array(self.data_calibrator)
        x = np.array(self.freqs_calibrator)

        # Define the threshold based on the noise floor and a specified offset
        noise_floor_threshold = noise_floor_values + height_offset

        # Find peaks with the specified hieght
        peaks, _ = find_peaks(y, height=noise_floor_threshold, distance=2)

        # Filter peaks based on the noise floor threshold
        filtered_peaks_x = x[peaks]
        filtered_peaks_y = y[peaks]

        print(f"Number of peaks found: {len(filtered_peaks_x)}")

        # Return the peaks and their heights
        return filtered_peaks_x, filtered_peaks_y, len(filtered_peaks_x)

    ####################################################################################################################
    ####################################################################################################################

    @staticmethod
    def detect_peaks_above_noise_model_calibrator(fit_amplitude_optimal, low_freqs, low_data, threshold_factor=3,
                                                  window_size=60):
        """
        Fonction pour détecter les pics au-dessus du modèle de bruit ajusté (amplitudes).

        :param fit_amplitude_optimal
        :param low_freqs
        :param low_data
        :param threshold_factor: Facteur multipliant l'écart-type pour définir le seuil de détection des pics.
        :param window_size

        :return: Indices des pics et valeurs des pics.
        """

        # Calculate the differences between the actual amplitudes and the fitted model
        residuals = low_data - fit_amplitude_optimal.astype(np.float64)

        # Initialise a table for local dynamic thresholds
        peak_threshold = np.zeros_like(low_data, dtype=np.float64)

        # Calculate the threshold for each frequency
        for i in range(len(low_data)):
            # Define the window around frequency i
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(low_data)+60, i + window_size // 2 + 1)

            # Extract residuals in this local window
            local_residuals = residuals[start_idx:end_idx]

            # Calculate the median and local IQR
            local_median = np.median(local_residuals)
            local_iqr = np.percentile(local_residuals, 75) - np.percentile(local_residuals, 25)

            # Define the local threshold
            peak_threshold[i] = local_median + threshold_factor * local_iqr

        # Use find_peaks to detect peaks above the threshold
        low_peaks, _ = find_peaks(low_data, height=peak_threshold.astype(np.float64), distance=2)

        # Return the frequencies and amplitudes corresponding to the peaks
        peak_frequencies = low_freqs[low_peaks].astype(np.float64)
        peak_amplitudes = low_data[low_peaks].astype(np.float64)

        return peak_frequencies, peak_amplitudes, peak_threshold.astype(np.float64)

    ####################################################################################################################
    ####################################################################################################################

    def scans_comparison_calibrator(self):
        """
        Function linked to the ScansComparison class.

        :return: int
        """

        # Create a BarycenterProcessor instance
        self.barycenter_processor_calibrator = ScansComparison(
            self.barycenters_list_calibrator, self.barycenter_clusters_list_calibrator,
            self.full_clusters_list_calibrator, self.sl_points_y_calibrator, self.sl_points_x_calibrator,
            self.scans_comparison_treated_signal_x_calibrator, self.scans_comparison_treated_signal_y_calibrator,
            self.scans_comparison_treated_signal_x_calibrator_51, self.scans_comparison_treated_signal_y_calibrator_51,
            self.x_ratio_values_calibrator, self.y_ratio_values_calibrator, self.y_max_values_calibrator,
            self.x_max_values, self.scans_done_calibrator, self.freqs_calibrator, self.freqs_calibrator_51,
            self.data_calibrator, self.data_calibrator_51, self.list_straylight_barycenter51_calibrator,
            self.list_perturbation_barycenter51_calibrator, self.list_straylight_barycenter53_calibrator,
            self.list_perturbation_barycenter53_calibrator, self.pic_number_calibrator, self.peak_width_51_calibrator,
            self.peak_width_53_calibrator, self.quality_x_SL, self.quality_y_SL, self.quality_x_perturbation,
            self.quality_y_perturbation, self.list_straylight_max51, self.list_perturbation_max51,
            self.list_straylight_max53, self.list_perturbation_max53)

        (list_y_max, indices_to_remove_calibrator, self.sl_points_y_calibrator, self.sl_points_x_calibrator,
         self.scans_comparison_treated_signal_x_calibrator, self.scans_comparison_treated_signal_y_calibrator,
         self.x_ratio_values_calibrator, self.y_ratio_values_calibrator, self.y_max_values_calibrator,
         self.scans_done_calibrator, self.list_straylight_barycenter51_calibrator,
         self.list_perturbation_barycenter51_calibrator, self.list_straylight_barycenter53_calibrator,
         self.list_perturbation_barycenter53_calibrator, self.pic_number_calibrator) = (
            self.barycenter_processor_calibrator.compare_barycenters_calibrator(X_CONDITION_BARYCENTER_COMPARISON,
                                                                                Y_CONDITION_BARYCENTER_COMPARISON))

        self.list_perturbation_barycenter51_calibrator_spec_freq = [
                [(val[0] / 5.878e-2, val[1]) for val in sublist]
                for sublist in self.list_perturbation_barycenter51_calibrator
            ]

        self.list_perturbation_barycenter53_calibrator_spec_freq = [
                [(val[0] / 5.656e-2, val[1]) for val in sublist]
                for sublist in self.list_perturbation_barycenter53_calibrator
            ]

        print("\n")
        printing(
            f"Number of SL pics detected between 5.1 and 5.3 GHz/s scans : {len(list_y_max)}\n "
            f"Number of removed points: {len(indices_to_remove_calibrator)}",
            100, "-")

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def low_scans_comparison_calibrator(self):
        """
        Function linked to the ScansComparison class for low frequencies.

        :return: int
        """

        # Create a BarycenterProcessor instance
        self.barycenter_processor_calibrator = ScansComparison(
            self.low_barycenters_list_calibrator, self.low_barycenter_clusters_list_calibrator,
            self.low_full_clusters_list_calibrator, self.low_sl_points_y_calibrator, self.low_sl_points_x_calibrator,
            self.low_scans_comparison_treated_signal_x_calibrator,
            self.low_scans_comparison_treated_signal_y_calibrator,
            self.low_scans_comparison_treated_signal_x_calibrator_51,
            self.low_scans_comparison_treated_signal_y_calibrator_51, self.low_x_ratio_values_calibrator,
            self.low_y_ratio_values_calibrator, self.low_y_max_values_calibrator, self.low_x_max_values,
            self.low_scans_done_calibrator, self.low_freqs_calibrator, self.low_freqs_calibrator_51,
            self.low_data_calibrator, self.low_data_calibrator_51, self.low_list_straylight_barycenter51_calibrator,
            self.low_list_perturbation_barycenter51_calibrator, self.low_list_straylight_barycenter53_calibrator,
            self.low_list_perturbation_barycenter53_calibrator, self.low_pic_number_calibrator,
            self.low_peak_width_51_calibrator, self.low_peak_width_53_calibrator, self.low_quality_x_SL,
            self.low_quality_y_SL, self.low_quality_x_perturbation, self.low_quality_y_perturbation,
            self.low_list_straylight_max51, self.low_list_perturbation_max51, self.low_list_straylight_max53,
            self.low_list_perturbation_max53)

        (low_list_y_max, low_indices_to_remove_calibrator, self.low_sl_points_y_calibrator,
            self.low_sl_points_x_calibrator, self.low_scans_comparison_treated_signal_x_calibrator,
            self.low_scans_comparison_treated_signal_y_calibrator, self.low_x_ratio_values_calibrator,
            self.low_y_ratio_values_calibrator, self.low_y_max_values_calibrator,
            self.low_scans_done_calibrator, self.low_list_straylight_barycenter51_calibrator,
            self.low_list_perturbation_barycenter51_calibrator, self.low_list_straylight_barycenter53_calibrator,
            self.low_list_perturbation_barycenter53_calibrator, self.low_pic_number_calibrator) = (
            self.barycenter_processor_calibrator.compare_barycenters_calibrator(X_CONDITION_BARYCENTER_COMPARISON,
                                                                                Y_CONDITION_BARYCENTER_COMPARISON))

        self.low_list_perturbation_barycenter51_calibrator_spec_freq = [
                [(val[0] / 5.878e-2, val[1]) for val in sublist]
                for sublist in self.low_list_perturbation_barycenter51_calibrator
            ]

        self.low_list_perturbation_barycenter53_calibrator_spec_freq = [
                [(val[0] / 5.656e-2, val[1]) for val in sublist]
                for sublist in self.low_list_perturbation_barycenter53_calibrator
            ]

        print("\n")
        printing(
            f"Number of SL pics detected between 5.1 and 5.3 GHz/s scans in low frequencies : "
            f"{len(low_list_y_max)}\n "
            f"Number of removed points in low frequencies: {len(low_indices_to_remove_calibrator)}",
            100, "-")

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def create_plot_calibrator(self, xaxis: object, yaxis: object, xlabel: object, ylabel: object, file_path) -> object:
        """
        Create the basic elements of a graph such as title, labels and axis...

        :param xaxis: x data to plot
        :param yaxis: y data to plot
        :param xlabel: x-axis legend
        :param ylabel: y-axis legend
        :param file_path

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

        # Create a label for the plot legend using various instance attributes
        if "28092023-Testscancalib" in file_path:
            label = f'{self.record_type_calibrator}, {'reference monitor'}'
            self.legends.append(label)
        else:
            label = f'{self.record_type_calibrator}, {'monitor'}, {self.folder_scrollbox.currentText()[:8]}'
            self.legends.append(label)

        # --------------------------------------------------------------------------------------------------------------
        # 5.1 Data plot

        handle = None

        # If the line plot option is checked, plot the x and y data
        if self.checkbox_data_plotline_calibrator.isChecked():
            handle = sns.lineplot(x=xaxis, y=yaxis, ax=self.ax, linewidth=1.5, label=label)

            # Get the color of the last plotted line to maintain consistency
            line_handle = handle.get_lines()[-1]
            self.line_color = line_handle.get_color()

        else:
            self.line_color = None

        # --------------------------------------------------------------------------------------------------------------
        # 5.1 Barycenter plot

        if self.checkbox_barycenters.isChecked():

            barycenters = np.array(self.barycenters_list_calibrator[self.count_calibrator])
            low_barycenters = np.array(self.low_barycenters_list_calibrator[self.count_calibrator])
            # Concatenation of the two lists
            all_barycenters = np.concatenate((low_barycenters, barycenters), axis=0)

            if all_barycenters.size > 0:
                handle = sns.scatterplot(x=all_barycenters[:, 0], y=all_barycenters[:, 1], ax=self.ax,
                                             color=self.line_color,
                                             label=f"Barycenter ({POINTS_CALCUL_BARYCENTRE}pts) "
                                                   f"{self.record_type_calibrator[:8]}")
            else:
                print("Not enough barycenter in the list to be plotted (<= 0).")

        # --------------------------------------------------------------------------------------------------------------
        # 5.1 Noise floor plot

        if self.checkbox_noise_floor.isChecked():
            sns.lineplot(x=self.noise_floor_freqs_calibrator, y=self.noise_floor_values_calibrator, ax=self.ax,
                         color=self.line_color, label=[f"Noise floor {self.record_type_calibrator[:8]}"])

            sns.lineplot(x=self.low_freqs_calibrator, y=self.peak_threshold_calibrator,
                         ax=self.ax, color=self.line_color)

        # --------------------------------------------------------------------------------------------------------------
        # Finalize plot

        self.canvas.draw()

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def show_filtered_spectrum_calibrator(self, file_path):
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
        def add_plot_calibrator(x_data, y_data, x_label, y_label, file_path):
            """
            Create and add a plot to the layout.

            :param: x_data: Data for the x-axis.
            :param: y_data: Data for the y-axis.
            :param: x_label: Label for the x-axis.
            :param: y_label: Label for the y-axis.
            :param: file_path

            :return: void
            """

            self.create_plot_calibrator(x_data, y_data, x_label, y_label, file_path)
            self.layout_graph.addWidget(NavigationToolbar(self.canvas, self))

        # ==============================================================================================================
        # Call the inside function

        # Concatenate low and high frequencies to obtain a complete x-axis.
        self.freqs_tot_calibrator = np.concatenate((self.low_xaxis_calibrator, self.xaxis_calibrator[1:]))

        # Concatenate the data associated with low and high frequencies to obtain the complete y-axis.
        self.data_tot_calibrator = np.concatenate((self.low_yaxis_calibrator, self.yaxis_calibrator[1:]))

        # Check if freqs_tot is a list or a numpy array
        if isinstance(self.freqs_tot_calibrator, (list, np.ndarray)):
            # Convert to pandas series
            self.freqs_tot_calibrator = pd.Series(self.freqs_tot_calibrator)
        else:
            # Convert to pandas series and encapsulate scalars in a list
            self.freqs_tot_calibrator = pd.Series([self.freqs_tot_calibrator])

        # Check if data_tot is a list or a numpy array
        if isinstance(self.data_tot_calibrator, (list, np.ndarray)):
            # Convert to pandas series
            self.data_tot_calibrator = pd.Series(self.data_tot_calibrator)
        else:
            # Convert to pandas series and encapsulate scalars in a list
            self.data_tot_calibrator = pd.Series([self.data_tot_calibrator])

        add_plot_calibrator(pd.Series(self.freqs_tot_calibrator), pd.Series(self.data_tot_calibrator),
                            self.x_axis_legend, self.y_axis_legend, file_path)

    ####################################################################################################################
    ####################################################################################################################

    def overlay_filtered_plot_calibrator(self, file_path=None):
        """
        Overlay a new plot on the existing plot with a different color.

        :return: void
        """

        # ==============================================================================================================
        # Inside function

        def add_overlay_plot_calibrator(x_data, y_data, canal, file_path=None):
            """
            Create and add an overlay plot to the existing plot.

            :param: x_data: Data for the x-axis.
            :param: y_data: Data for the y-axis.
            :param: canal: Canal information.

            :return: void
            """


            # If no scans have been completed, create a new plot with relevant data.
            if not self.scans_done_calibrator:

                # Construct a label for the current plot based on the record type, QPR, canal, and selected folder.
                if "28092023-Testscancalib" in file_path:
                    label = f'{self.record_type_calibrator}, {'reference monitor'}'
                    self.list_labels.append(label)
                else:
                    label = f'{self.record_type_calibrator}, {'monitor'}, {self.folder_scrollbox.currentText()[:8]}'
                    self.list_labels.append(label)

                # If more than one label exists, append a legend indicating scan parameters.
                if len(self.list_labels) > 1 and "28092023-Testscancalib" not in file_path:
                    self.legends.append(
                        f'Scan 5.1GHz/s, {'monitor'}, {canal}, {self.folder_scrollbox.currentText()[:8]}')

                # ------------------------------------------------------------------------------------------------------
                # Plot the selected signal in line

                if self.checkbox_data_plotline_calibrator.isChecked():

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

                    barycenters = np.array(self.barycenters_list_calibrator[self.count_calibrator])
                    low_barycenters = np.array(self.low_barycenters_list_calibrator[self.count_calibrator])

                    # Concatenation of the two lists
                    all_barycenters = np.concatenate((low_barycenters, barycenters), axis=0)

                    if all_barycenters.size > 0:

                        # Scatter plot for barycenters with the same color as the signal line.
                        sns.scatterplot(x=all_barycenters[:, 0], y=all_barycenters[:, 1], ax=self.ax,
                                            color=self.line_color,
                                            label=f"Barycenter ({POINTS_CALCUL_BARYCENTRE}pts) "f"{self.record_type_calibrator[:8]}")
                    else:
                        print("Not enough barycenter in the list to be plotted (<= 0).")

                # ------------------------------------------------------------------------------------------------------
                # Plot the noise floor of the selected signal in line

                if self.checkbox_noise_floor.isChecked():
                    sns.lineplot(x=self.noise_floor_freqs_calibrator, y=self.noise_floor_values_calibrator, ax=self.ax,
                                     color=self.line_color,
                                     label=[f"Noise floor {self.record_type_calibrator[:8]}"])
                    sns.lineplot(x=self.low_freqs_calibrator, y=self.peak_threshold_calibrator, ax=self.ax,
                                     color=self.line_color)

            # ==========================================================================================================
            # If scans have been completed, adjust the plot based on the treated signal.

            else:

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
                if self.checkbox_data_plotline_calibrator.isChecked():
                    self.plot_colors.pop(0)

                # ------------------------------------------------------------------------------------------------------
                # Compare y max and y ratio in the DataAnalysis tab

                # Call a function from DataAnalysis Class
                self.data_analysis_tab_calibrator.compare_pos_plot_calibrator(self.pic_number_calibrator,
                                                                        self.y_ratio_values_calibrator,
                                                                        self.plot_colors,
                                                                        'y ratio for each calibrator peak',
                                                                        'Peak',
                                                                        'Y Ratio Values |y51/y53|',
                                                                        self.list_labels, 1,
                                                                        self.count_calibrator / 2)

                # ------------------------------------------------------------------------------------------------------
                # Compare y max and quality percentage in the DataAnalysis tab

                self.data_analysis_tab_calibrator.compare_pos_plot_calibrator(self.pic_number_calibrator,
                                                                        self.x_ratio_values_calibrator,
                                                                        self.plot_colors,
                                                                        'x ratio for each calibrator peak',
                                                                        'Peak',
                                                                        'X Ratio Values |x51/x53|',
                                                                        self.list_labels, 2,
                                                                        self.count_calibrator / 2)

                # ------------------------------------------------------------------------------------------------------

                table_data = [[x, y] for x, y in self.list_straylight_barycenter51_calibrator_tot[-1]]
                table_columns = ["X components", "Y components"]

                if table_data and table_columns:
                    ax = self.figure.add_axes([0.295, -0.02, 1, 0.2])
                    ax.axis("off")  # Turn off axis
                    the_table = ax.table(cellText=table_data, colLabels=table_columns, loc="center")
                    the_table.auto_set_font_size(False)
                    the_table.set_fontsize(11)
                    the_table.auto_set_column_width(col=list(range(len(table_columns))))
                    # Optionnel : ajuster la hauteur des lignes (plus grande hauteur)
                    for i, row in enumerate(the_table.get_celld().values()):
                        row.set_height(0.085)  # Augmenter la hauteur des cellules des données


        # ==============================================================================================================
        # Main function call

        # Call the add_overlay_plot function with the relevant data to create the overlay plot.
        # Concatenate low and high frequencies to obtain a complete x-axis.
        self.freqs_tot_calibrator = np.concatenate((self.low_xaxis_calibrator, self.xaxis_calibrator[1:]))
        # Concatenate the data associated with low and high frequencies to obtain the complete y-axis.
        self.data_tot_calibrator = np.concatenate((self.low_yaxis_calibrator, self.yaxis_calibrator[1:]))

        # Check if freqs_tot is a list or a numpy array
        if isinstance(self.freqs_tot_calibrator, (list, np.ndarray)):
            # Convert to pandas series
            self.freqs_tot_calibrator = pd.Series(self.freqs_tot_calibrator)
        else:
            # Convert to pandas series and encapsulate scalars in a list
            self.freqs_tot_calibrator = pd.Series([self.freqs_tot_calibrator])  # Encapsulate scalars in a list

        # Check if data_tot is a list or a numpy array
        if isinstance(self.data_tot_calibrator, (list, np.ndarray)):
            # Convert to pandas series
            self.data_tot_calibrator = pd.Series(self.data_tot_calibrator)
        else:
            # Convert to pandas series and encapsulate scalars in a list
            self.data_tot_calibrator = pd.Series([self.data_tot_calibrator])  # Encapsulate scalars in a list

        add_overlay_plot_calibrator(pd.Series(self.freqs_tot_calibrator), pd.Series(self.data_tot_calibrator),
                                    self.canal, file_path)

        self.canvas.draw()

    ####################################################################################################################
    ####################################################################################################################

    def process_files_calibrator(self):
        """
        Get file paths of signal to process.

        :return: List of computed file path: List[str]
        """

        # List to store the resulting file paths
        file_paths = None

        # Selected folder is retri
        folder_parent_name = str(self.folder_scrollbox.currentText())
        folder_name = str(self.record_calibrator_scrollbox.currentText())

        # --------------------------------------------------------------------------------------------------------------
        # File path creation

        # Construct the base directory path
        path = os.path.join(self.data_path, folder_parent_name)

        # List all subdirectories in the path
        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

        # Find the folder that matches the filename
        target_folder = None
        for folder in folders:
            if folder_name.lower() in folder.lower():
                target_folder = folder
                break

        if path is None:
            raise ValueError("La variable 'path' est None.")
        if target_folder is None:
            raise ValueError("La variable 'target_folder' est None.")

        # Construct the file path to the "Processed" subfolder within the target folder
        file_path = os.path.join(path, target_folder, 'Processed')

        # List all files in the "Processed" directory
        files = os.listdir(file_path)

        # Search for the specific file that matches the user's pico selection
        target_file = None
        target_file = files[-1]

        # Add the full path of the target file to the result list
        f = os.path.join(file_path, target_file)
        file_paths = f

        # --------------------------------------------------------------------------------------------------------------

        return file_paths

    ####################################################################################################################
    ####################################################################################################################

    def analyse_data_calibrator(self):
        """
        Store barycenter data in an excel file.

        :return: int
        """

        # Create dictionaries for each page of the Excel file
        data_pageScan_SL_peaks = {}
        data_pageRef_SL_peaks = {}


        # Create columns in dataframes with lists values
        for i in range(len(self.barycenters_list_calibrator) // 2):
            # ----------------------------------------------------------------------------------------------------------
            # Scan 5.1

            data_pageScan_SL_peaks[(f'{self.record_type_calibrator}, {'monitor'}, '
                                    f'{self.folder_scrollbox.currentText()[:8]}')] = [f"({float(v[0])}, {float(v[1])})"
                                                        for v in self.list_straylight_barycenter51_calibrator_tot[i]]

            # ----------------------------------------------------------------------------------------------------------
            # Scan 5.3

            data_pageRef_SL_peaks[f'{self.record_type_calibrator}, {'reference monitor'}'] = \
                [f"({float(v[0])}, {float(v[1])})" for v in self.list_straylight_barycenter53_calibrator_tot[i]]


        # Transform dictionaries in dataframe
        df_Scan_SL_peaks = pd.DataFrame.from_dict(data_pageScan_SL_peaks, orient='index').transpose()
        df_Ref_SL_peaks = pd.DataFrame.from_dict(data_pageRef_SL_peaks, orient='index').transpose()

        # --------------------------------------------------------------------------------------------------------------
        # Check exception errors

        try:

            # Save barycenter dataframe into Excel file
            file_path_barycenter = f'Barycenter_data_monitor_{self.record_calibrator_scrollbox.currentText()}_{self.folder_scrollbox.currentText()}.xlsx'
            with pd.ExcelWriter(file_path_barycenter, engine='openpyxl') as writer:
                df_Scan_SL_peaks.to_excel(writer, sheet_name='Scan monitor peaks', index=False)
                df_Ref_SL_peaks.to_excel(writer, sheet_name='Reference monitor peaks', index=False)
            print(f"Barycenter are now accessible in 'Barycenter_data_monitor_{self.record_calibrator_scrollbox.currentText()}_{self.folder_scrollbox.currentText()}.xlsx'.")

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