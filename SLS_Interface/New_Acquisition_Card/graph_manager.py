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

from PySide6.QtWidgets import (QComboBox, QListWidget)
from typing import Any, Optional, List, Tuple, Union
import pandas as pd
import inspect
import re

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


class GraphManager:
    opd_bandwidth: list[float]
    low_opd_bandwidth: list[float]

    def __init__(self,  x_axis_scrollbox: Union[QComboBox, str],
                 yaxis_scrollbox: QComboBox,
                 analysis: Any,
                 canal_listwidget: QListWidget,
                 x_axis_legends: List[str],
                 x_axis_legend: Optional[str],
                 y_axis_legends: List[str],
                 y_axis_legend: Optional[str],
                 x_axis_datas: List[str],
                 x_axis_data: Optional[str],
                 #pico_choice: Optional[str],
                 qpr: Optional[str],
                 xaxis_min: Optional[float],
                 xaxis_max: Optional[float],
                 low_xaxis_min: Optional[float],
                 low_xaxis_max: Optional[float],
                 xaxis: Optional[pd.DataFrame],
                 yaxis: Optional[pd.DataFrame],
                 low_xaxis: Optional[pd.DataFrame],
                 low_yaxis: Optional[pd.DataFrame],
                 opd_bandwidth: List[float],
                 freq_bandwidth: List[float],
                 time_bandwidth: List[float],
                 low_opd_bandwith : List[float],
                 low_freq_bandwith : List[float]):

        # --------------------------------------------------------------------------------------------------------------
        #                                              Graph axis
        # --------------------------------------------------------------------------------------------------------------

        # Scroll box to get current user selection for the axis type
        self.x_axis_scrollbox = x_axis_scrollbox
        self.yaxis_scrollbox = yaxis_scrollbox

        # X-axis' legends
        self.x_axis_legends = x_axis_legends
        self.x_axis_legend = x_axis_legend

        # Y-axis' legends
        self.y_axis_legends = y_axis_legends
        self.y_axis_legend = y_axis_legend

        # X-axis' column names
        self.x_axis_datas = x_axis_datas
        self.x_axis_data = x_axis_data

        # Axis data
        self.yaxis = yaxis
        self.xaxis = xaxis
        self.low_yaxis = low_yaxis
        self.low_xaxis = low_xaxis

        # Bandwidth definition
        self.xaxis_min = xaxis_min
        self.xaxis_max = xaxis_max
        self.low_xaxis_min = low_xaxis_min
        self.low_xaxis_max = low_xaxis_max

        # --------------------------------------------------------------------------------------------------------------
        #                                              Parameters
        # --------------------------------------------------------------------------------------------------------------

        # Get signal parameters
        self.canal = []
        self.canal_index = None
        self.canal_listwidget = canal_listwidget
        self.qpr = qpr

        # Bandwidth definition in function of plot type
        self.opd_bandwidth = opd_bandwidth
        self.freq_bandwidth = freq_bandwidth
        self.time_bandwidth = time_bandwidth
        self.low_opd_bandwidth = low_opd_bandwith
        self.low_freq_bandwidth = low_freq_bandwith

        # Class instance recuperation
        self.analysis = analysis

    ####################################################################################################################
    ####################################################################################################################

    def get_canal(self, from_get_graph_parameters=False):
        """
        Get the canal choice.

        :return: Return the actually selected canal: Optional[str]
        """

        # Vérifie si la fonction qui a appelé get_canal() est fonction1
        if from_get_graph_parameters:
            # Canal current selection
            selected_canal = self.canal_listwidget.selectedItems()

            # Get the last canal selected
            self.canal = f"/'Group Name'/'{[item.text() for item in selected_canal][-1]}'"

            # Récupérer l'indice du canal dans la liste
            self.canal_index = self.canal_listwidget.row(selected_canal[-1])

        else :
            self.canal = 'Canal B'
            self.canal_index = 1

        return self.canal

    ####################################################################################################################
    ####################################################################################################################

    def legend_qpr(self, data_vdc):
        """
        Define the QPR legend based on the path and pico choice.

        :param path: File path used to determine the configuration.
        :param pico: Pico choice.
        :return: Return the corresponding qpr of the plotted signal: Optional[str]
        """
        # --------------------------------------------------------------------------------------------------------------
        # Vérifier que l'indice existe dans les données VDC
        vdc_values = data_vdc['Legends'].values
        if self.canal_index >= len(vdc_values):
            raise ValueError(f"L'indice {self.canal_index} est hors des limites des données VDC.")

        # Accéder à la légende
        self.qpr = vdc_values[self.canal_index]

        print(f'self.qpr={self.qpr}')
        return self.qpr

    ####################################################################################################################
    ####################################################################################################################

    def choose_x_axis(self, final_path: str) -> Tuple[float, float, float, float, str, any]:
        """
        Choose data and legends for the x-axis of the plot based on user choices.

        :param final_path: Path of the plotted file.
        :return: Tuple containing x-axis min, max values, legend, and data.
        """
        if isinstance(self.x_axis_scrollbox, QComboBox):
            # Si c'est un QComboBox, on utilise currentText()
            x_axis_type = self.x_axis_scrollbox.currentText()
        elif isinstance(self.x_axis_scrollbox, str):
            # Si c'est une chaîne de caractères (str), on l'utilise directement
            x_axis_type = self.x_axis_scrollbox
        else:
            # Gestion d'erreur si le type n'est ni QComboBox ni str
            raise TypeError("self.x_axis_scrollbox doit être soit un QComboBox, soit une chaîne de caractères.")

        # ==============================================================================================================
        # Frequency

        if x_axis_type == "Frequency (Hz)":
            self.xaxis_min = self.freq_bandwidth[0]
            self.xaxis_max = self.freq_bandwidth[1]
            self.low_xaxis_min = self.low_freq_bandwidth[0]
            self.low_xaxis_max = self.low_freq_bandwidth[1]
            self.x_axis_legend = self.x_axis_legends[1]
            self.x_axis_data = self.x_axis_datas[1]

        # ==============================================================================================================
        # Optical Path Difference

        elif x_axis_type == "Optical Path Difference (OPD)":

            self.xaxis_min = self.opd_bandwidth[0]
            self.xaxis_max = self.opd_bandwidth[1]
            self.low_xaxis_min = self.low_opd_bandwidth[0]
            self.low_xaxis_max = self.low_opd_bandwidth[1]

            # ----------------------------------------------------------------------------------------------------------
            # Scan 5.1 GHz/s

            if "51" in final_path:
                self.x_axis_legend = self.x_axis_legends[2]
                self.x_axis_data = self.x_axis_datas[2]

            # ----------------------------------------------------------------------------------------------------------
            # Scan 5.3 GHz/s

            elif "53" in final_path:
                self.x_axis_legend = self.x_axis_legends[3]
                self.x_axis_data = self.x_axis_datas[3]

            # ----------------------------------------------------------------------------------------------------------
            # Scan 0

            elif "Scan0" in final_path:
                self.x_axis_legend = self.x_axis_legends[3]
                self.x_axis_data = self.x_axis_datas[3]

            # ----------------------------------------------------------------------------------------------------------
            # Scan 0P

            elif "Scan0P" in final_path:
                self.x_axis_legend = self.x_axis_legends[3]
                self.x_axis_data = self.x_axis_datas[3]

            # ----------------------------------------------------------------------------------------------------------
            # Error handling

            else:
                raise ValueError("Cannot plot a Dark or No Scan based on Optical Path Difference.")

        # ==============================================================================================================
        # Time

        elif x_axis_type == "Time (s)":

            self.xaxis_min = self.time_bandwidth[0]
            self.xaxis_max = self.time_bandwidth[1]
            self.low_xaxis_min = self.time_bandwidth[0]
            self.low_xaxis_max = self.time_bandwidth[1]
            self.x_axis_legend = self.x_axis_legends[0]
            self.x_axis_data = self.x_axis_datas[0]

        else:
            raise ValueError(f"Unknown x-axis type: {x_axis_type}")

        return (self.xaxis_min, self.xaxis_max, self.low_xaxis_min, self.low_xaxis_max,
                self.x_axis_legend, self.x_axis_data)

    ####################################################################################################################
    ####################################################################################################################

    def choose_y_axis(self, file_vdc_df):
        """
        Choose data and legends for the y-axis of the plot based on user choices.

        :param file_vdc_df: DataFrame containing VDC information used for fractional amplitude calculation.
        :return: Tuple containing the x-axis data, y-axis data, and the y-axis legend.
        """

        # --------------------------------------------------------------------------------------------------------------
        # Get the type of y-axis from the user's selection

        y_axis_type = self.yaxis_scrollbox.currentText()

        # --------------------------------------------------------------------------------------------------------------
        # Determine the appropriate y-axis legend and data based on the selected y-axis type

        if y_axis_type == "Amplitude":
            # For "Amplitude", select the corresponding legend and compute the x and y axes
            self.y_axis_legend = self.y_axis_legends[1]
            self.xaxis, self.yaxis = self.analysis.get_axes(self.x_axis_data, self.canal, self.xaxis_min,
                                                            self.xaxis_max)
            self.low_xaxis, self.low_yaxis = self.analysis.get_axes(self.x_axis_data, self.canal, self.low_xaxis_min,
                                                                    self.low_xaxis_max)

        elif y_axis_type == "Fractional amplitude":
            # For "Fractional amplitude", select the corresponding legend
            self.y_axis_legend = self.y_axis_legends[2]
            self.xaxis, self.yaxis = self.analysis.get_axes(self.x_axis_data, self.canal, self.xaxis_min,
                                                            self.xaxis_max)
            self.low_xaxis, self.low_yaxis = self.analysis.get_axes(self.x_axis_data, self.canal, self.low_xaxis_min,
                                                                    self.low_xaxis_max)

            # Use the first and last indices of the x-axis to slice the VDC DataFrame and assign it to the y-axis
            xaxis_first_index = self.xaxis.index[0]
            xaxis_last_index = self.xaxis.index[-1]
            low_xaxis_first_index = self.low_xaxis.index[0]
            low_xaxis_last_index = self.low_xaxis.index[-1]
            self.yaxis = file_vdc_df[xaxis_first_index:xaxis_last_index + 1]
            self.low_yaxis = file_vdc_df[low_xaxis_first_index:low_xaxis_last_index + 1]

        elif y_axis_type == "Temporal amplitude":

            # For "Temporal amplitude", select the corresponding legend and compute the x and y axes
            self.y_axis_legend = self.y_axis_legends[0]
            self.xaxis, self.yaxis = self.analysis.get_axes(self.x_axis_data, self.canal, self.xaxis_min,
                                                            self.xaxis_max)
            self.low_xaxis, self.low_yaxis = self.analysis.get_axes(self.x_axis_data, self.canal, self.xaxis_min,
                                                            self.xaxis_max)

        # --------------------------------------------------------------------------------------------------------------
        # Return the calculated axes and the y-axis legend

        return self.xaxis, self.yaxis, self.low_xaxis, self.low_yaxis, self.y_axis_legend
