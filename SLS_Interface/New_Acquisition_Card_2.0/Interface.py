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

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox
from PySide6.QtGui import QIcon
from ConfigurationTab import ConfigurationTab
from VisualisationTab import VisualisationTab
from TreatmentTab import StrayLightTreatment
from header_dictionary import execute_with_error_handling
from CalibratorTab import CalibratorTab

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


class Interface(QMainWindow):
    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------------------------------------------------------
        #                                              Tabs creation
        # --------------------------------------------------------------------------------------------------------------

        # Create the tab main widget
        self.tabs: QTabWidget = QTabWidget()

        # Create tabs by calling instances of their corresponding classes
        #self.config_tab: ConfigurationTab = ConfigurationTab()
        self.visualization_tab: VisualisationTab = VisualisationTab()
        self.stray_light_tab: StrayLightTreatment = StrayLightTreatment()
        self.calibrator_tab: CalibratorTab = CalibratorTab()

        # Initialize UI components with error handling
        execute_with_error_handling(self.initialize_ui)

    ####################################################################################################################
    ####################################################################################################################

    def initialize_ui(self):
        """
        Initialize the user interface components and settings.
        """

        # Set the window title
        self.setWindowTitle("Stray Light Software")

        # Set the window geometry
        desktop = QApplication.primaryScreen().geometry()
        self.setGeometry(100, 100, int(desktop.width() * 0.6), int(desktop.height() * 0.6))

        # Add tabs to the tab widget
        #self.tabs.addTab(self.config_tab, "Configuration")
        self.tabs.addTab(self.visualization_tab, "Data Visualisation")
        self.tabs.addTab(self.stray_light_tab, "Stray Light Processing")
        self.tabs.addTab(self.calibrator_tab, "Calibrator Processing")

        # Show the 'Data Analysis' tab only when the treatment is over
        self.stray_light_tab.event_occurred.connect(self.show_data_analysis_tab)
        self.calibrator_tab.event_occurred_calibrator.connect(self.show_data_analysis_tab_calibrator)

        # Set the central widget of the main window
        self.setCentralWidget(self.tabs)

        # Function's call
        execute_with_error_handling(self.set_dark_theme)
        execute_with_error_handling(self.set_window_icon)

    ####################################################################################################################
    ####################################################################################################################

    def show_data_analysis_tab(self):
        """
        Add and display the Data Analysis tab once the treatment is over.

        """

        # ==============================================================================================================

        def inside_function():

            if not self.tabs.indexOf(self.stray_light_tab.data_analysis_tab) >= 0:
                self.tabs.addTab(self.stray_light_tab.data_analysis_tab, "Data Analysis")

                # Set the actual view of the user on the 'Data Analysis' tab
            self.tabs.setCurrentWidget(self.stray_light_tab.data_analysis_tab)

        # ==============================================================================================================

        # Call the function with the error handling
        execute_with_error_handling(inside_function)

    ####################################################################################################################
    ####################################################################################################################

    def show_data_analysis_tab_calibrator(self):
        """
        Add and display the Data Analysis tab once the treatment is over.

        """

        # ==============================================================================================================

        def inside_function_calibrator():

            if not self.tabs.indexOf(self.calibrator_tab.data_analysis_tab_calibrator) >= 0:
                self.tabs.addTab(self.calibrator_tab.data_analysis_tab_calibrator, "Data Analysis Calibrator")

                # Set the actual view of the user on the 'Data Analysis' tab
            self.tabs.setCurrentWidget(self.calibrator_tab.data_analysis_tab_calibrator)

        # ==============================================================================================================

        # Call the function with the error handling
        execute_with_error_handling(inside_function_calibrator)

    ####################################################################################################################
    ####################################################################################################################

    def set_dark_theme(self):
        """
        Change the interface's theme in dark using CSS.

        """

        self.setStyleSheet("""
                    QPushButton { background-color: #3C3F41; color: #FFFFFF; border-radius: 5px; padding: 10px; }
                    QGroupBox { border: 1px solid #3C3F41; border-radius: 5px; margin-top: 20px; padding: 10px; }
                    QLabel { font-size: 12px; color: #3C3F41; }
                    QDateEdit, QTimeEdit, QLineEdit, QTextEdit, QComboBox { background-color: #3C3F41; color: #FFFFFF; 
                                                border: 1px solid #555555; border-radius: 5px; padding: 5px; }
                    QDialog { background-color: #eeeeee; color: #FFFFFF; border: 1px solid #eeeeee; border-radius: 5px; 
                    margin-top: 20px; padding: 10px; }
                """)

    ####################################################################################################################
    ####################################################################################################################

    def set_window_icon(self):
        """
        Set the window icon.
        """

        icon_path = "logo_grand.ico"
        my_icon = QIcon(icon_path)
        self.setWindowIcon(my_icon)

    ####################################################################################################################
    ####################################################################################################################

    def show_error_message(self, title, message):
        """
        Display an error message dialog.
        """

        QMessageBox.warning(self, title, message)

########################################################################################################################
########################################################################################################################


if __name__ == "__main__":
    print(f"Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    app = QApplication(sys.argv)
    window = Interface()
    window.show()
    sys.exit(app.exec())
