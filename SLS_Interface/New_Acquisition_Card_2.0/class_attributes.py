import os
from typing import Optional
from PySide6.QtCore import QSize
import qtawesome as qta
from PySide6.QtWidgets import (
    QVBoxLayout, QPushButton, QCheckBox, QGroupBox, QLabel, QGridLayout, QLineEdit, QTextEdit, QComboBox,
    QHBoxLayout, QListWidget)


class VisualisationAttributes:
    def __init__(self):

        """self.attributes = VisualisationAttributes()
        for key, value in vars(self.attributes).items():
            setattr(self, key, value)"""

        # Define the graph layer and hbox layer
        self.tab_name: str = 'VisualisationTab'
        self.xaxis_max: Optional[float] = None
        self.xaxis_min: Optional[float] = None
        self.qpr = None
        self.pico_choice = None
        self.ax = None
        self.record_type = None
        self.x_axis_legends = ["Time (s)", "Freq (Hz)", "SL/nom OPD (m)", "SL/nom OPD (m)"]
        self.x_axis_datas = ["Temps", "Freq", "DL 5,1", "DL 5,3"]
        self.y_axis_legends = ["Amplitude (mV)", "Amplitude (Vpk)", "SL/Snom amplitude ratio (Vpk/VDC)"]
        self.x_axis_data = None
        self.y_axis_legend = None
        self.x_axis_legend = None

        self.mask = None
        self.xaxisDL = None
        self.current_data_scrollbox_selection = None
        self.canvas = None
        self.xaxis = None
        self.yaxis = None
        self.figure = None
        self.analysis = None
        self.canal = None
        self.data_vdc = None
        self.dataframes = None
        self.creation_csv_date = None
        self.error_flag = None
        self.title_record_name_flag = False
        self.record_name = None
        self.title = None

        self.layout_graph = QVBoxLayout()
        self.hbox_layout = QHBoxLayout()

        # Set the main layout in Grid mode
        self.layout = QGridLayout()

        # Define a frame on a parameter layer for the parameters choice
        self.frame_parameter = QGroupBox("Parameters")
        self.parameter_layout = QGridLayout()

        # Create the main button to launch the process
        self.button_launch_measurement = QPushButton("Launch Processing")
        self.button_divide_data = None

        # Set the labels for the parameter frame
        self.measurement_text = QLabel("Measurement :")
        self.record_text = QLabel("Record :")
        self.pico_text = QLabel("Pico :")
        self.title_text = QLabel("Title :")
        self.canal_text = QLabel("Canal :")
        self.checkbox_log_text = QLabel("Log options :")
        self.yaxis_choice_text = QLabel("Y axis :")
        self.x_axis_choice_text = QLabel("X axis :")

        # Create checkbox on a checkbox layout to get logarithmic axis for the plot
        self.checkbox_logx = QCheckBox("LogX")
        self.checkbox_logy = QCheckBox("LogY")
        self.checkbox_no_log = QCheckBox("None")

        self.help_button = QPushButton()
        self.help_button.setIcon(qta.icon('fa5s.question-circle'))  # Set the icon for the button
        self.help_button.setIconSize(QSize(40, 40))  # Set the size of the icon
        self.help_button.setToolTip("Click for help")  # Add a tooltip
        self.help_button.setFixedSize(QSize(40, 40))
        self.help_button.setStyleSheet("QPushButton { background-color: #FFFFFF; border: none; }")

        # Define other widgets of the parameter frame
        self.text_zone = QTextEdit()
        self.measurement_scrollbox = QComboBox()
        self.record_scrollbox = QComboBox()
        self.pico_scrollbox_visualisationtab = QComboBox()
        self.text_scrollbox = QLineEdit()
        self.listwidget = QListWidget()
        self.yaxis_scrollbox = QComboBox()
        self.x_axis_scrollbox = QComboBox()

        self.base_path = os.path.abspath(os.path.dirname(__file__))
        self.data_path = os.path.join(self.base_path, '..', '..', 'Data', 'Data-manip')