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
- ClassName; Function_Name(); variable_name; CONSTANT_NAME;
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

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


class DataExtraction:
    def __init__(self, path: str):
        """
        Initialize the DataExtraction object.

        :param path: Path to the data file.
        """

        self.datas: Any = None
        self.frequencies: Optional[List[float]] = None
        self.data_columns: Dict[str, List[float]] = {}
        self.fft_data: Dict[str, np.ndarray] = {}
        self.data_path: str = path

    ####################################################################################################################
    ####################################################################################################################

    def load_data(self, path):
        """
        Load data from a CSV file into a pandas DataFrame.

        :param path: Path to the CSV file.
        :return: int: Status code (0 for success).
        """
        if "Processed" in path:
            print(f'path =  {path}')
            self.datas = pd.read_csv(path, delimiter=" ")

        else:

            # Load CSV file with a semicolon delimiter and preprocess
            self.datas = pd.read_csv(path, delimiter=";", skiprows=[1, 2])

            # Replace commas with dots
            self.datas = self.datas.replace(',', '.', regex=True)

            # Convert to numeric
            self.datas = self.datas.apply(pd.to_numeric, errors='coerce')

            # Sort by 'Temps' column
            self.datas = self.datas.sort_values(by='Temps')
            self.datas.columns = self.datas.columns.str.strip()

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def extract_data(self) -> object:
        """
        Extract non nul columns of the dataframe and creates a dictionary (data_columns) with extracted values

        :return: int
        """
        columns = self.datas.columns.tolist()

        # Store values of each column in a dictionary
        for column in columns:
            column_values = self.datas[column].values
            self.data_columns[column] = column_values

        return 0

    ####################################################################################################################
    ####################################################################################################################

    def get_axes(self, column_namex, column_namey, minimum, maximum):
        """
        Get filtered data for specified x and y columns within a range.

        :param column_namex: Name of the x-axis column.
        :param column_namey: Name of the y-axis column.
        :param minimum: Minimum value for filtering.
        :param maximum: Maximum value for filtering.
        :return: Tuple of two arrays: x-axis and y-axis data within the specified range.
        """

        mask = (self.datas[column_namex] > minimum) & (self.datas[column_namex] <= maximum)
        if "Temps" in column_namex:

            # Return data for 'Temps' column
            return self.datas[column_namex][mask], self.datas[column_namey][mask]
        else:
            # Return absolute value of data for other columns
            return self.datas[column_namex][mask], np.abs(self.datas[column_namey][mask])

    ####################################################################################################################
    ####################################################################################################################

    def get_axes_bis(self, column_namex, column_namey, minimum, maximum):
        """
        Get filtered data for specified x and y columns within a range.

        :param column_namex: Name of the x-axis column.
        :param column_namey: Name of the y-axis column.
        :param minimum: Minimum value for filtering.
        :param maximum: Maximum value for filtering.
        :return: Tuple of two arrays: x-axis and y-axis data within the specified range.
        """

        y_axis = []

        mask = (self.datas[column_namex] > minimum) & (self.datas[column_namex] <= maximum)
        if "Temps" in column_namex:

            # Return data for 'Temps' column
            for i in range(len(column_namey)):
                column = self.datas.iloc[:, 0]
                filtered_values = column[mask]
                y_axis.append(filtered_values)
            return self.datas[column_namex][mask], y_axis
        else:
            # Return absolute value of data for other columns
            for i in range(len(column_namey)):
                column = self.datas.iloc[:, 0]
                filtered_values = column[mask]
                y_axis.append(filtered_values)
            return self.datas[column_namex][mask], np.abs(y_axis)

    ####################################################################################################################
    ####################################################################################################################

    @staticmethod
    def get_csv_file_modification_date(filepath):
        """
        Get the modification date of a CSV file.

        :param filepath: Full path to the CSV file.
        :return: str: Formatted modification date (YYYY-MM-DD HH:MM:SS) or an error message.
        """
        try:

            # Get file modification time and convert to formatted date
            modification_time = os.path.getmtime(filepath)
            modification_datetime = datetime.fromtimestamp(modification_time)
            return modification_datetime.strftime('%Y-%m-%d %H:%M:%S')
        except OSError:

            # Return error message if file is not found or inaccessible
            return "File not found or inaccessible"
