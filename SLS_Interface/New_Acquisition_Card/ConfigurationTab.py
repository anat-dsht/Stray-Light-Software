from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QMessageBox,
    QGroupBox, QLabel, QDateEdit, QGridLayout, QLineEdit,
    QTimeEdit, QTextEdit
)
from PySide6.QtGui import QFont
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ConfigurationTab(QWidget):
    def __init__(self):
        super().__init__()

        layout = QGridLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(10)

        # Create a parameter frame
        frame_parameter = QGroupBox("Parameters")

        # Create a layout for the parameter frame with inside margins and widgets
        parameter_layout = QGridLayout()
        parameter_layout.setContentsMargins(20, 20, 20, 20)
        parameter_layout.setSpacing(10)  # Spacing between widgets

        # Set a fixed width for the parameter frame
        frame_parameter.setFixedWidth(300)

        # Create a button and connect its clicked signal to the function
        button_launch_measurement = QPushButton("Launch Measurement")
        button_launch_measurement.clicked.connect(self.Show_Message)
        button_launch_measurement.adjustSize()  # Adjust size to fit the text

        # Create a text for scroll box
        date_text = QLabel("Date :")
        time_text = QLabel("Time :")
        title_text = QLabel("Title :")

        # Apply fonts
        font = QFont("Arial", 10)
        date_text.setFont(font)
        time_text.setFont(font)
        title_text.setFont(font)

        # Create a text zone for a description of the measurement
        text_zone = QTextEdit()

        # Create the scroll box
        date_scrollbox = QDateEdit()
        time_scrollbox = QTimeEdit()
        text_scrollbox = QLineEdit()

        # Add widgets to the parameter layout
        parameter_layout.addWidget(date_text, 0, 0)
        parameter_layout.addWidget(date_scrollbox, 0, 1)
        parameter_layout.addWidget(time_text, 1, 0)
        parameter_layout.addWidget(time_scrollbox, 1, 1)
        parameter_layout.addWidget(title_text, 2, 0)
        parameter_layout.addWidget(text_scrollbox, 2, 1)
        parameter_layout.addWidget(text_zone, 3, 0, 1, 2)

        # Set the layout to the parameter frame
        frame_parameter.setLayout(parameter_layout)

        # Create a layout for the graphs
        layout_graph = QVBoxLayout()

        # Add the parameter frame, button, and graphs to the main layout
        layout.addWidget(frame_parameter, 0, 0)
        layout.addLayout(layout_graph, 0, 1)
        layout.addWidget(button_launch_measurement, 1, 0)

        self.setLayout(layout)

    def Show_Message(self):
        """
        Show a pop-up message onde the button is clicked.

        :return:
        """
        # Show a message box when the button is clicked
        QMessageBox.information(self, "Message", "You clicked the button!")

    def Create_Plot(self):
        # Create a plot
        self.figure = Figure()
        canvas = FigureCanvas(self.figure)

        # Add a sample plot
        ax = self.figure.add_subplot(111)
        ax.plot([0, 1, 2, 3], [10, 1, 20, 5])
        ax.set_title("Sample Plot")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.set_facecolor('none')  # Remove background color
        self.figure.patch.set_facecolor('none')  # Remove the figure background

        return canvas
