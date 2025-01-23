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

import math
import numpy as np
from sklearn.neighbors import KDTree
from typing import Union, List, Any
from tqdm import tqdm

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


class ScansComparison:
    def __init__(self, barycenters_list, barycenter_clusters_list, full_clusters_list, sl_points_y, sl_points_x,
                 scans_comparison_treated_signal_x, scans_comparison_treated_signal_y,
                 scans_comparison_treated_signal_x_51, scans_comparison_treated_signal_y_51, x_ratio_values, y_ratio_values,
                 y_max_values, x_max_values, scans_done, freqs, freqs_51, data, data_51, list_straylight_barycenter51,
                 list_perturbation_barycenter51, list_straylight_barycenter53, list_perturbation_barycenter53,
                 pic_number, peak_width_51, peak_width_53, quality_x_SL, quality_y_SL, quality_x_perturbation,
                 quality_y_perturbation, list_straylight_max51, list_perturbation_max51,
                 list_straylight_max53, list_perturbation_max53):

        # --------------------------------------------------------------------------------------------------------------
        # Barycenter

        self.barycenters_list: List[np.ndarray] = barycenters_list
        self.list_straylight_barycenter51: List[List[tuple[Any, Any]]] = list_straylight_barycenter51
        self.list_perturbation_barycenter51: List[List[tuple[Any, Any]]] = list_perturbation_barycenter51
        self.list_straylight_barycenter53: List[List[Union[float, int]]] = list_straylight_barycenter53
        self.list_perturbation_barycenter53: List[List[Union[float, int]]] = list_perturbation_barycenter53
        self.list_straylight_max51: List[List[tuple[Any, Any]]] = list_straylight_max51
        self.list_perturbation_max51: List[List[tuple[Any, Any]]] = list_perturbation_max51
        self.list_straylight_max53: List[List[Union[float, int]]] = list_straylight_max53
        self.list_perturbation_max53: List[List[Union[float, int]]] = list_perturbation_max53

        # --------------------------------------------------------------------------------------------------------------
        # Clusters
        self.barycenter_clusters_list: List[np.ndarray] = barycenter_clusters_list
        self.full_clusters_list: List[np.ndarray] = full_clusters_list
        self.sl_points_y: List[float] = sl_points_y
        self.sl_points_x: List[float] = sl_points_x
        self.scans_comparison_treated_signal_x: List[List[float]] = scans_comparison_treated_signal_x
        self.scans_comparison_treated_signal_y: List[List[float]] = scans_comparison_treated_signal_y
        self.scans_comparison_treated_signal_x_51: List[List[float]] = scans_comparison_treated_signal_x_51
        self.scans_comparison_treated_signal_y_51: List[List[float]] = scans_comparison_treated_signal_y_51

        # --------------------------------------------------------------------------------------------------------------
        # Data

        self.freqs: np.ndarray = freqs
        self.freqs_51: np.ndarray = freqs_51
        self.data: np.ndarray = data
        self.data_51: np.ndarray = data_51

        # --------------------------------------------------------------------------------------------------------------
        # Statistical values

        self.x_ratio_values: List[List[float]] = x_ratio_values
        self.y_ratio_values: List[List[float]] = y_ratio_values
        self.y_max_values: List[List[float]] = y_max_values
        self.x_max_values: List[List[float]] = x_max_values
        self.quality_x_SL: List[List[float]] = quality_x_SL
        self.quality_y_SL: List[List[float]] = quality_y_SL
        self.quality_x_perturbation: List[List[float]] = quality_x_perturbation
        self.quality_y_perturbation: List[List[float]] = quality_y_perturbation
        self.pic_number: List[List[float]] = pic_number
        self.peak_width_51: List[np.ndarray] = peak_width_51
        self.peak_width_53: List[np.ndarray] = peak_width_53

        # --------------------------------------------------------------------------------------------------------------
        # Flag

        self.scans_done: bool = scans_done
        self.indices_to_keep_53 = []
        self.indices_to_remove_53 = []
        self.indices_to_keep_51 = []
        self.indices_to_remove_51 = []
        self.widths_SL_51 = []
        self.widths_perturbation_51 = []
        self.widths_SL_53 = []
        self.widths_perturbation_53 = []

    ####################################################################################################################
    ####################################################################################################################

    @staticmethod
    def calculate_quality_percentage(x_ratio, y_ratio):
        """
        Calculate the quality percentage based on the x and y ratios.

        :param x_ratio: Value of the x_ratio for quality calculation
        :param y_ratio: Value of the y_ratio for quality calculation

        :return: quality score (float)
        """

        '''quality_percentage_x = (1 - x_ratio) * 100
        quality_percentage_y = (1 - y_ratio) * 100'''

        quality_percentage_x = x_ratio
        quality_percentage_y = y_ratio

        return quality_percentage_x, quality_percentage_y

    ####################################################################################################################
    ####################################################################################################################

    @staticmethod
    def find_closest_barycenter(tree, x, y, record):
        """
        Find the closest barycenter in record53 using KDTree.

        :param tree: Tree to compare the given point.
        :param x: Point's X-coordinate.
        :param y: Point's Y-coordinate.
        :param record: Get the number of points.

        :return: list of indices (list[int])
        """

        _, indices = tree.query([[x, y]], k=len(record), return_distance=True)
        
        return indices[0]

    ####################################################################################################################
    ####################################################################################################################

    def process_barycenter(self, x51: object, y51: object, record53: object, index: object, x_condition: object,
                           y_condition: object) -> object:
        """
        Process a barycenter and return the appropriate indices to keep or remove.

        :param x51: Point's X-coordinate.
        :param y51: Point's Y-coordinate.
        :param record53: Array containing the coordinates of the points in the 53rd record.
        :param index: Index of the point to compare within the 53rd record.
        :param x_condition: Threshold for X-coordinate comparison.
        :param y_condition: Threshold for Y-coordinate comparison.

        :return: A tuple containing:
            - indices_to_keep (List[np.ndarray]): Indices of the points to keep if the conditions are met.
            - indices_to_remove (List[np.ndarray]): Indices of the points to remove if the conditions are not met.
            - ratio_x (float): The absolute difference in the X-coordinates between the two points.
            - ratio_y (float): The relative difference in the Y-coordinates between the two points.
            - barycenter (Tuple[float, float]): The coordinates of the barycenter in the 53rd record.
        """
        # Extract coordinates of the corresponding barycenter in the 53rd record
        x53, y53 = record53[index]

        # Calcul des ratios
        try:
            ratio_y = abs(y51 / y53 - 1)
            ratio_x = abs(x51 / x53 - 1)
        except ZeroDivisionError:
            ratio_y, ratio_x = float('inf'), float('inf')

        # Initialize lists to store indices of points to keep or remove
        indices_to_keep = []
        indices_to_remove = []

        # Get the full set of associated points (masks) for the current barycenter
        mask_53_full = self.full_clusters_list[-1][index]
        mask53_full_set = set(map(tuple, mask_53_full))

        # Create a tuple for the current barycenter coordinates
        barycenter = (x53, y53)

        # --------------------------------------------------------------------------------------------------------------
        # Condition to decide whether to keep or remove points based on X and Y differences

        if (ratio_x <= x_condition) and (ratio_y <= y_condition):
            for mask_point in mask53_full_set:

                # Find the index of each point that meets the condition
                peak_index = np.where((self.freqs == mask_point[0]) & (self.data == mask_point[1]))[0]
                indices_to_keep.append(peak_index)

            # Return the indices to keep, empty list for removal, and the calculated ratios and barycenter
            return indices_to_keep, [], ratio_x, ratio_y, barycenter

        else:
            for mask_point in mask53_full_set:

                # Find the index of each point that doesn't meet the condition
                peak_index = np.where((self.freqs == mask_point[0]) & (self.data == mask_point[1]))[0]
                indices_to_remove.append(peak_index)

            # Return the empty list for keeping, indices to remove, and the calculated ratios and barycenter
            return [], indices_to_remove, ratio_x, ratio_y, barycenter

    ####################################################################################################################
    ####################################################################################################################

    def process_barycenter_51(self, x53: object, y53: object, record51: object, index: object, x_condition: object,
                           y_condition: object) -> object:
        """
        Process a barycenter and return the appropriate indices to keep or remove.

        :param x53: Point's X-coordinate.
        :param y53: Point's Y-coordinate.
        :param record51: Array containing the coordinates of the points in the 53rd record.
        :param index: Index of the point to compare within the 53rd record.
        :param x_condition: Threshold for X-coordinate comparison.
        :param y_condition: Threshold for Y-coordinate comparison.

        :return: A tuple containing:
            - indices_to_keep (List[np.ndarray]): Indices of the points to keep if the conditions are met.
            - indices_to_remove (List[np.ndarray]): Indices of the points to remove if the conditions are not met.
            - ratio_x (float): The absolute difference in the X-coordinates between the two points.
            - ratio_y (float): The relative difference in the Y-coordinates between the two points.
            - barycenter (Tuple[float, float]): The coordinates of the barycenter in the 53rd record.
        """
        # Extract coordinates of the corresponding barycenter in the 53rd record
        x51, y51 = record51[index]

        # Calculate the relative and absolute differences for Y and X coordinates
        ratio_y = abs(y51 / y53 - 1)
        ratio_x = abs(x51 / x53 - 1)

        # Initialize lists to store indices of points to keep or remove
        indices_to_keep_51 = []
        indices_to_remove_51 = []

        # Get the full set of associated points (masks) for the current barycenter
        mask_51_full = self.full_clusters_list[-2][index]
        mask51_full_set = set(map(tuple, mask_51_full))

        # --------------------------------------------------------------------------------------------------------------
        # Condition to decide whether to keep or remove points based on X and Y differences

        if (ratio_x <= x_condition) and (ratio_y <= y_condition):
            for mask_point in mask51_full_set:

                # Find the index of each point that meets the condition
                peak_index = np.where((self.freqs_51 == mask_point[0]) & (self.data_51 == mask_point[1]))[0]
                indices_to_keep_51.append(peak_index)

            # Return the indices to keep, empty list for removal, and the calculated ratios and barycenter
            return indices_to_keep_51, []

        else:
            for mask_point in mask51_full_set:

                # Find the index of each point that doesn't meet the condition
                peak_index = np.where((self.freqs_51 == mask_point[0]) & (self.data_51 == mask_point[1]))[0]
                indices_to_remove_51.append(peak_index)

            # Return the empty list for keeping, indices to remove, and the calculated ratios and barycenter
            return [], indices_to_remove_51

    ####################################################################################################################
    ####################################################################################################################

    def compare_barycenters(self, x_condition_barycenter_comparison: object, y_condition_barycenter_comparison: object) -> object:

        """
        Compare the last two barycenters and update the signal based on the comparison.
        :param x_condition_barycenter_comparison: Threshold for X-coordinate comparison.
        :param y_condition_barycenter_comparison: Threshold for Y-coordinate comparison.
        :return: Tuple containing the updated lists of Y maxima, indices to keep,
                 signal points, perturbation peaks, and other metrics.
        """

        # --------------------------------------------------------------------------------------------------------------
        # Ensure there are enough barycenters to perform the comparison

        if len(self.barycenters_list) < 2:
            print("Not enough barycenter to clean the signal (less than 2)")
            return

        # Retrieve the two most recent barycenter records
        record51 = self.barycenters_list[-2]
        record53 = self.barycenters_list[-1]

        # Create a KDTree for efficient barycenter comparison in record53
        tree = KDTree(record53)
        tree_51 = KDTree(record51)

        # Initialize lists to store indices and calculated ratios
        list_y_ratio_SL = []
        list_y_ratio_perturbation = []
        list_y_max = []
        list_x_max = []
        list_x_ratio_SL = []
        list_x_ratio_perturbation = []

        # Initialize lists to track straylight and perturbation barycenters
        list_straylight_barycenter51 = []
        list_perturbation_barycenter51 = []
        list_straylight_barycenter53 = []
        list_perturbation_barycenter53 = []

        peak_width_51 = self.peak_width_51[-1]
        peak_width_53 = self.peak_width_53[-1]
        widths_SL_51 = []
        widths_SL_53 = []
        widths_perturbation_51 = []
        widths_perturbation_53 = []
        iteration_count = 0

        list_straylight_max51 = []
        list_straylight_max53 = []
        list_perturbation_max51 = []
        list_perturbation_max53 = []

        # Set for tracking barycenters that have already been added to one of the lists
        processed_barycenters_straylight = set()
        processed_barycenters_perturbation = set()

        # --------------------------------------------------------------------------------------------------------------
        # Iterate over the barycenters in record51 and compare them with record53
        for x51, y51 in tqdm(record51, desc="Barycenter comparison"):
            iteration_count += 1  # Incrémenter le compteur à chaque itération
            indices = ScansComparison.find_closest_barycenter(tree, x51, y51, record53)
            for index in indices:
                keep, remove, ratio_x, ratio_y, barycenter = self.process_barycenter(
                    x51, y51, record53, index,
                    x_condition_barycenter_comparison,
                    y_condition_barycenter_comparison
                )
                if keep:
                    # If this barycenter was previously in the perturbation list, move it to the straylight list
                    if barycenter in processed_barycenters_perturbation:
                        # Remove from perturbation list
                        idx = list_perturbation_barycenter53.index(barycenter)
                        list_perturbation_barycenter53.pop(idx)
                        widths_perturbation_53.pop(idx)

                    if (x51, y51) not in list_straylight_barycenter51:
                        list_straylight_barycenter51.append((x51, y51))

                    if barycenter not in list_straylight_barycenter53:
                        list_straylight_barycenter53.append(barycenter)
                        list_straylight_barycenter53 = sorted(list_straylight_barycenter53, key=lambda b: b[0])

                    # Retrieve the corresponding width for each barycenter
                    try:
                        widths_SL_51.append(peak_width_51[iteration_count-1])
                    except IndexError:
                        print(f"Failed to append width for index {iteration_count-1} because it is out of bounds.")
                    widths_SL_53.append(peak_width_53[index])

                    list_x_ratio_SL.append(ratio_x)
                    list_y_ratio_SL.append(ratio_y)

                    # Find the maximum Y value for the current cluster and add it to the list
                    cluster_points = self.barycenter_clusters_list[-1][index]
                    max_point = max(cluster_points, key=lambda point: point[1])
                    pic_y_value = max_point[1]
                    list_y_max.append(pic_y_value)

                    # Retrieve the first value (associated with Y max) and add it to list_x_max
                    list_x_max.append(max_point[0])

                    # Add the indices to keep
                    self.indices_to_keep_53.extend(keep)

                    # Mark this barycenter as processed for straylight
                    processed_barycenters_straylight.add(barycenter)

                    break

                else:
                    if (x51, y51) not in list_perturbation_barycenter51:
                        list_perturbation_barycenter51.append((x51, y51))
                        try:
                            widths_perturbation_51.append(peak_width_51[iteration_count-1])
                        except IndexError:
                            print(f"Failed to append width for index {iteration_count-1} because it is out of bounds.")

                    if barycenter not in processed_barycenters_perturbation:
                        if barycenter not in list_perturbation_barycenter53:
                            list_perturbation_barycenter53.append(barycenter)
                            list_perturbation_barycenter53 = sorted(list_perturbation_barycenter53, key=lambda b: b[0])
                            widths_perturbation_53.append(peak_width_53[index])
                        # Mark this barycenter as processed for perturbation
                        processed_barycenters_perturbation.add(barycenter)

                    list_x_ratio_perturbation.append(ratio_x)
                    list_y_ratio_perturbation.append(ratio_y)

                    # Add the indices to remove
                    self.indices_to_remove_53.extend(remove)

        # Iterate over the barycenters in record51 and compare them with record53
        for x53, y53 in record53:
            indices_51 = ScansComparison.find_closest_barycenter(tree_51, x53, y53, record51)
            for index_51 in indices_51:
                keep_51, remove_51 = self.process_barycenter_51(
                    x53, y53, record51, index_51,
                    x_condition_barycenter_comparison,
                    y_condition_barycenter_comparison
                )
                if keep_51:
                    # Add the indices to keep
                    self.indices_to_keep_51.extend(keep_51)
                    break

                else:
                    self.indices_to_remove_51.extend(remove_51)

        duplicates_51 = set(list_straylight_barycenter51).intersection(set(list_perturbation_barycenter51))

        if duplicates_51:
            print(
                f"Attention : des doublons existent entre les listes de lumière parasite et de perturbations pour le scan 51 : {duplicates_51}")
        else:
            print("Aucun doublon entre les listes de lumière parasite et de perturbations pour le scan 51.")

        duplicates_53 = set(list_straylight_barycenter53).intersection(set(list_perturbation_barycenter53))

        if duplicates_53:
            print(
                f"Attention : des doublons existent entre les listes de lumière parasite et de perturbations pour le scan 53 : {duplicates_53}")
        else:
            print("Aucun doublon entre les listes de lumière parasite et de perturbations pour le scan 53.")

        self.widths_SL_51.append(widths_SL_51)
        self.widths_SL_53.append(widths_SL_53)
        self.widths_perturbation_51.append(widths_perturbation_51)
        self.widths_perturbation_53.append(widths_perturbation_53)

        # --------------------------------------------------------------------------------------------------------------
        # Finalize the lists of indices to keep and remove by converting them to sets
        if len(self.indices_to_keep_53) > 0:
            self.indices_to_keep_53 = set(np.concatenate(self.indices_to_keep_53))
        else:
            self.indices_to_keep_53 = set()  # Si la liste est vide, on initialise un ensemble vide

        # Finaliser les listes d'indices à supprimer
        if len(self.indices_to_remove_53) > 0:
            self.indices_to_remove_53 = set(np.concatenate(self.indices_to_remove_53))
        else:
            self.indices_to_remove_53 = set()  # Si la liste est vide, on initialise un ensemble vide

        # Filter out any indices that are in both the keep and remove lists
        self.indices_to_remove_53 = [
            index for index in tqdm(self.indices_to_remove_53, desc="Checking obtained results")
            if index not in self.indices_to_keep_53
        ]

        if len(self.indices_to_keep_51) > 0:
            self.indices_to_keep_51 = set(np.concatenate(self.indices_to_keep_51))
        else:
            self.indices_to_keep_51 = set()  # Si la liste est vide, on initialise un ensemble vide

        # Finaliser les listes d'indices à supprimer
        if len(self.indices_to_remove_51) > 0:
            self.indices_to_remove_51 = set(np.concatenate(self.indices_to_remove_51))
        else:
            self.indices_to_remove_51 = set()  # Si la liste est vide, on initialise un ensemble vide

        # Filter out any indices that are in both the keep and remove lists
        self.indices_to_remove_51 = [
            index for index in tqdm(self.indices_to_remove_51, desc="Checking obtained results")
            if index not in self.indices_to_keep_51
        ]

        # --------------------------------------------------------------------------------------------------------------
        # Update the signal points based on the indices to keep

        self.sl_points_y = self.data[list(self.indices_to_keep_53)]
        self.sl_points_x = self.freqs[list(self.indices_to_keep_53)]

        freqs = self.freqs.tolist()
        data = self.data.tolist()

        freqs_51 = self.freqs_51.tolist()
        data_51 = self.data_51.tolist()

        epsilon = 1e-9

        # Génération des signaux traités avec remplacement des données par des valeurs proches de 0 pour les indices à supprimer
        scan_comparison_treated_signal_x, scan_comparison_treated_signal_y = zip(*[
            (freqs[i], data[i] if i not in self.indices_to_remove_53 else epsilon)
            for i in tqdm(range(len(freqs)), desc="Generation of the treated signal for scan 53")
        ])

        scan_comparison_treated_signal_x_51, scan_comparison_treated_signal_y_51 = zip(*[
            (freqs_51[i], data_51[i] if i not in self.indices_to_remove_51 else epsilon)
            for i in tqdm(range(len(freqs_51)), desc="Generation of the treated signal for scan 51")
        ])


        # --------------------------------------------------------------------------------------------------------------
        # Normalize the ratios and calculate quality percentages

        x_ratios_normalized_SL = np.array(list_x_ratio_SL) / x_condition_barycenter_comparison
        y_ratios_normalized_SL = np.array(list_y_ratio_SL) / y_condition_barycenter_comparison

        x_ratios_normalized_perturbation = np.array(list_x_ratio_perturbation) / x_condition_barycenter_comparison
        y_ratios_normalized_perturbation = np.array(list_y_ratio_perturbation) / y_condition_barycenter_comparison

        qualities_SL = [ScansComparison.calculate_quality_percentage(x_ratio, y_ratio)
                               for x_ratio, y_ratio in zip(x_ratios_normalized_SL, y_ratios_normalized_SL)]

        if qualities_SL:  # Vérifie si la liste n'est pas vide
            quality_x_SL, quality_y_SL = zip(*qualities_SL)
        else:  # Liste vide : affecter des valeurs par défaut ou passer
            quality_x_SL, quality_y_SL = [], []

        qualities_perturbation = [ScansComparison.calculate_quality_percentage(x_ratio, y_ratio)
                               for x_ratio, y_ratio in zip(x_ratios_normalized_perturbation,
                                                           y_ratios_normalized_perturbation)]

        quality_x_perturbation, quality_y_perturbation = zip(*qualities_perturbation)

        # --------------------------------------------------------------------------------------------------------------
        # Append the calculated values to their respective lists

        self.x_ratio_values.append(list_x_ratio_SL)
        self.y_ratio_values.append(list_y_ratio_SL)
        self.y_max_values.append(list_y_max)
        self.x_max_values.append(list_x_max)
        self.quality_x_SL.append(quality_x_SL)
        self.quality_y_SL.append(quality_y_SL)
        self.quality_x_perturbation.append(quality_x_perturbation)
        self.quality_y_perturbation.append(quality_y_perturbation)

        self.list_perturbation_barycenter53.append(list_perturbation_barycenter53)
        self.list_perturbation_barycenter51.append(list_perturbation_barycenter51)
        self.list_straylight_barycenter53.append(list_straylight_barycenter53)
        self.list_straylight_barycenter51.append(list_straylight_barycenter51)

        # S'assurer que les listes de barycentres n'ont pas de doublons
        list_perturbation_barycenter51 = list(set(list_perturbation_barycenter51))
        list_perturbation_barycenter53 = list(set(list_perturbation_barycenter53))
        list_straylight_barycenter51 = list(set(list_straylight_barycenter51))
        list_straylight_barycenter53 = list(set(list_straylight_barycenter53))

        for i, couple in enumerate(list_straylight_barycenter51):
            try:
                second_value = couple[1]
                amp_elec = second_value / 0.654093
                # Effectuer le calcul et ajouter le résultat
                list_straylight_max51.append(amp_elec / 2)
            except Exception as e:
                print(f"Erreur lors du traitement de list_straylight_barycenter51 à l'indice {i} : {e}")

        for i, couple in enumerate(list_straylight_barycenter53):
            try:
                second_value = couple[1]
                amp_elec = second_value / 0.654093
                list_straylight_max53.append(amp_elec / 2)
            except Exception as e:
                print(f"Erreur lors du traitement de list_straylight_barycenter53 à l'indice {i} : {e}")

        for i, couple in enumerate(list_perturbation_barycenter51):
            try:
                second_value = couple[1]
                amp_elec = second_value / 0.654093
                list_perturbation_max51.append(amp_elec / 2)
            except Exception as e:
                print(f"Erreur lors du traitement de list_perturbation_barycenter51 à l'indice {i} : {e}")

        for i, couple in enumerate(list_perturbation_barycenter53):
            try:
                second_value = couple[1]
                amp_elec = second_value / 0.654093
                list_perturbation_max53.append(amp_elec / 2)
            except Exception as e:
                print(f"Erreur lors du traitement de list_perturbation_barycenter53 à l'indice {i} : {e}")

        print(f'len(list_perturbation_max51) = {len(list_perturbation_max51)}')
        print(f'len(list_perturbation_barycenter51) = {len(list_perturbation_barycenter51)}')

        # Vérifier les tailles des listes après traitement
        assert len(list_perturbation_max51) == len(list_perturbation_barycenter51), \
            f"Taille incohérente : {len(list_perturbation_max51)} != {len(list_perturbation_barycenter51)}"
        assert len(list_perturbation_max53) == len(list_perturbation_barycenter53), \
            f"Taille incohérente : {len(list_perturbation_max53)} != {len(list_perturbation_barycenter53)}"

        # Parcourir chaque couple de valeurs dans list_straylight_barycenter51
        '''for couple in list_straylight_barycenter51:
            # Récupérer la seconde valeur du couple (index 1)
            second_value = couple[1]
            amp_elec = second_value / 0.654093
            # Effectuer le calcul et ajouter le résultat à list_straylight_max51
            list_straylight_max51.append(amp_elec / 2)

        for couple in list_straylight_barycenter53:
            second_value = couple[1]
            amp_elec = second_value / 0.654093
            list_straylight_max53.append(amp_elec / 2)

        for couple in list_perturbation_barycenter51:
            second_value = couple[1]
            amp_elec = second_value / 0.654093
            list_perturbation_max51.append(amp_elec / 2)

        for couple in list_perturbation_barycenter53:
            second_value = couple[1]
            amp_elec = second_value / 0.654093
            list_perturbation_max53.append(amp_elec / 2)'''

        self.list_straylight_max51.append(list_straylight_max51)
        self.list_straylight_max53.append(list_straylight_max53)
        self.list_perturbation_max51.append(list_perturbation_max51)
        self.list_perturbation_max53.append(list_perturbation_max53)

        self.scans_comparison_treated_signal_x.append(list(scan_comparison_treated_signal_x))
        self.scans_comparison_treated_signal_y.append(list(scan_comparison_treated_signal_y))

        self.scans_comparison_treated_signal_x_51.append(list(scan_comparison_treated_signal_x_51))
        self.scans_comparison_treated_signal_y_51.append(list(scan_comparison_treated_signal_y_51))

        self.scans_done = True

        # --------------------------------------------------------------------------------------------------------------
        # Return all relevant values for further analysis

        return (list_y_max, self.indices_to_keep_53, self.indices_to_remove_53, self.widths_SL_51,
                self.widths_perturbation_51, self.widths_SL_53, self.widths_perturbation_53,
                self.sl_points_y, self.sl_points_x, self.scans_comparison_treated_signal_x,
                self.scans_comparison_treated_signal_y, self.scans_comparison_treated_signal_x_51,
                self.scans_comparison_treated_signal_y_51, self.x_ratio_values, self.y_ratio_values, self.y_max_values,
                self.x_max_values, self.scans_done, self.list_straylight_barycenter51, self.list_perturbation_barycenter51,
                self.list_straylight_barycenter53, self.list_perturbation_barycenter53, self.quality_x_SL,
                self.quality_y_SL, self.quality_x_perturbation, self.quality_y_perturbation, self.list_straylight_max51,
                self.list_straylight_max53, self.list_perturbation_max51, self.list_perturbation_max53)

    ####################################################################################################################
    ####################################################################################################################

    def process_barycenter_calibrator(self, x51: object, y51: object, record53: object, index: object, x_condition: object,
                           y_condition: object) -> object:
        """
        Process a barycenter and return the appropriate indices to keep or remove.

        :param x51: Point's X-coordinate.
        :param y51: Point's Y-coordinate.
        :param record53: Array containing the coordinates of the points in the 53rd record.
        :param index: Index of the point to compare within the 53rd record.
        :param x_condition: Threshold for X-coordinate comparison.
        :param y_condition: Threshold for Y-coordinate comparison.

        :return: A tuple containing:
            - indices_to_keep (List[np.ndarray]): Indices of the points to keep if the conditions are met.
            - indices_to_remove (List[np.ndarray]): Indices of the points to remove if the conditions are not met.
            - ratio_x (float): The absolute difference in the X-coordinates between the two points.
            - ratio_y (float): The relative difference in the Y-coordinates between the two points.
            - barycenter (Tuple[float, float]): The coordinates of the barycenter in the 53rd record.
        """
        # Extract coordinates of the corresponding barycenter in the 53rd record
        x53, y53 = record53[index]

        # Calculate the relative and absolute differences for Y and X coordinates
        ratio_y = abs(y51 / y53 - 1)
        ratio_x = abs(x51 / x53 - 1)

        ratio_y_calibrator = abs(y51 / y53)
        ratio_x_calibrator = abs(x51 / x53)

        # Initialize lists to store indices of points to keep or remove
        indices_to_keep = []
        indices_to_remove = []

        # Get the full set of associated points (masks) for the current barycenter
        mask_53_full = self.full_clusters_list[-1][index]
        mask53_full_set = set(map(tuple, mask_53_full))

        # Create a tuple for the current barycenter coordinates
        barycenter = (x53, y53)

        # --------------------------------------------------------------------------------------------------------------
        # Condition to decide whether to keep or remove points based on X and Y differences

        if (ratio_x <= x_condition) and (ratio_y <= y_condition):
            for mask_point in mask53_full_set:

                # Find the index of each point that meets the condition
                peak_index = np.where((self.freqs == mask_point[0]) & (self.data == mask_point[1]))[0]
                indices_to_keep.append(peak_index)

            # Return the indices to keep, empty list for removal, and the calculated ratios and barycenter
            return indices_to_keep, [], ratio_x_calibrator, ratio_y_calibrator, barycenter

        else:
            for mask_point in mask53_full_set:

                # Find the index of each point that doesn't meet the condition
                peak_index = np.where((self.freqs == mask_point[0]) & (self.data == mask_point[1]))[0]
                indices_to_remove.append(peak_index)

            # Return the empty list for keeping, indices to remove, and the calculated ratios and barycenter
            return [], indices_to_remove, ratio_x_calibrator, ratio_y_calibrator, barycenter

    ####################################################################################################################
    ####################################################################################################################

    def compare_barycenters_calibrator(self, x_condition_barycenter_comparison: object, y_condition_barycenter_comparison: object) -> object:

        """
        Compare the last two barycenters and update the signal based on the comparison.
        :param x_condition_barycenter_comparison: Threshold for X-coordinate comparison.
        :param y_condition_barycenter_comparison: Threshold for Y-coordinate comparison.
        :return: Tuple containing the updated lists of Y maxima, indices to keep,
                 signal points, perturbation peaks, and other metrics.
        """

        # --------------------------------------------------------------------------------------------------------------
        # Ensure there are enough barycenters to perform the comparison

        if len(self.barycenters_list) < 2:
            print("Not enough barycenter to clean the signal (less than 2)")
            return

        # Retrieve the two most recent barycenter records
        record51 = self.barycenters_list[-2]
        record53 = self.barycenters_list[-1]


        # Create a KDTree for efficient barycenter comparison in record53
        tree = KDTree(record53)

        # Initialize lists to store indices and calculated ratios
        indices_to_keep = []
        indices_to_remove = []
        list_y_ratio = []
        list_y_max = []
        list_x_ratio = []
        i = 0
        pic_number =[]

        # Initialize lists to track straylight and perturbation barycenters
        list_straylight_barycenter51 = []
        list_perturbation_barycenter51 = []
        list_straylight_barycenter53 = []
        list_perturbation_barycenter53 = []

        # Dictionnaire pour stocker les barycentres par pic
        barycenters_by_pic = {}

        # --------------------------------------------------------------------------------------------------------------
        # Iterate over the barycenters in record51 and compare them with record53

        for x51, y51 in tqdm(record51, desc="Barycenter comparison"):
            indices = ScansComparison.find_closest_barycenter(tree, x51, y51, record53)
            for index in indices:
                keep, remove, ratio_x, ratio_y, barycenter = self.process_barycenter_calibrator(
                    x51, y51, record53, index,
                    x_condition_barycenter_comparison,
                    y_condition_barycenter_comparison
                )
                if keep:
                    # Vérification si x51 est un multiple de 1.40
                    if abs(x51) > 0.3 and (math.isclose(x51 % 1.40, 0, abs_tol=0.2) or math.isclose(x51 % 1.40, 1.40, abs_tol=0.2)) and y51 >= 2e-6:
                        pic_key = round((x51+0.2) / 1.40)  # Crée une clé unique pour le pic multiple de 1.40

                        # Si un barycentre pour ce pic existe déjà, comparer y51 pour ne garder que le plus élevé
                        if pic_key in barycenters_by_pic:
                            existing_barycenter = barycenters_by_pic[pic_key]
                            if y51 > existing_barycenter['y51']:
                                # Remplacer par le nouveau barycentre avec un y51 plus élevé
                                barycenters_by_pic[pic_key] = {'x51': x51, 'y51': y51, 'barycenter': barycenter,
                                                               'ratio_x': ratio_x, 'ratio_y': ratio_y, 'index': index}
                        else:
                            # Ajouter un nouveau barycentre pour ce pic
                            barycenters_by_pic[pic_key] = {'x51': x51, 'y51': y51, 'barycenter': barycenter,
                                                           'ratio_x': ratio_x, 'ratio_y': ratio_y, 'index': index}

                else:
                    if (x51, y51) not in list_perturbation_barycenter51:
                        list_perturbation_barycenter51.append((x51, y51))
                    if barycenter not in list_perturbation_barycenter53:
                        list_perturbation_barycenter53.append(barycenter)
                    indices_to_remove.extend(remove)

        # --------------------------------------------------------------------------------------------------------------
        # Once all valid barycenters have been traversed, the lists are filled in.
        for pic_key, barycenter_data in barycenters_by_pic.items():
            list_straylight_barycenter51.append((barycenter_data['x51'], barycenter_data['y51']))
            list_straylight_barycenter53.append(barycenter_data['barycenter'])

            list_x_ratio.append(barycenter_data['ratio_x'])
            list_y_ratio.append(barycenter_data['ratio_y'])

            # Trouver la valeur Y max pour le cluster en cours
            pic_y_value = max(point[1] for point in self.barycenter_clusters_list[-1][barycenter_data['index']])
            list_y_max.append(pic_y_value)

            # Ajout des indices à garder
            indices_to_keep.extend(keep)

            i += 1
            pic_number.append(i)

        # --------------------------------------------------------------------------------------------------------------
        # Finalize the lists of indices to keep and remove by converting them to sets
        if len(indices_to_keep) > 0:
            indices_to_keep = set(np.concatenate(indices_to_keep))
        else:
            indices_to_keep = set()  # Si la liste est vide, on initialise un ensemble vide

        # Finaliser les listes d'indices à supprimer
        if len(indices_to_remove) > 0:
            indices_to_remove = set(np.concatenate(indices_to_remove))
        else:
            indices_to_remove = set()  # Si la liste est vide, on initialise un ensemble vide

        # Filter out any indices that are in both the keep and remove lists
        indices_to_remove = [
            index for index in tqdm(indices_to_remove, desc="Checking obtained results")
            if index not in indices_to_keep
        ]

        # --------------------------------------------------------------------------------------------------------------
        # Update the signal points based on the indices to keep

        self.sl_points_y = self.data[list(indices_to_keep)]
        self.sl_points_x = self.freqs[list(indices_to_keep)]

        freqs = self.freqs.tolist()
        data = self.data.tolist()

        # Generate the updated perturbation peaks by excluding removed indices
        scan_comparison_treated_signal_x, scan_comparison_treated_signal_y = zip(*[(freqs[i], data[i])
                                                                                   for i in
                                                                                   tqdm(range(len(freqs)),
                                                                                        desc="Generation of"
                                                                                             " the treated "
                                                                                             "signal") if i
                                                                                   not in indices_to_remove]
                                                                                 )

        # --------------------------------------------------------------------------------------------------------------
        # Append the calculated values to their respective lists

        self.x_ratio_values.append(list_x_ratio)
        self.y_ratio_values.append(list_y_ratio)
        self.y_max_values.append(list_y_max)
        self.pic_number.append(pic_number)

        self.list_perturbation_barycenter53.append(list_perturbation_barycenter53)
        self.list_perturbation_barycenter51.append(list_perturbation_barycenter51)
        self.list_straylight_barycenter53.append(list_straylight_barycenter53)
        self.list_straylight_barycenter51.append(list_straylight_barycenter51)

        self.scans_comparison_treated_signal_x.append(list(scan_comparison_treated_signal_x))
        self.scans_comparison_treated_signal_y.append(list(scan_comparison_treated_signal_y))

        self.scans_done = True

        # --------------------------------------------------------------------------------------------------------------
        # Return all relevant values for further analysis

        return (list_y_max, indices_to_remove, self.sl_points_y, self.sl_points_x, self.scans_comparison_treated_signal_x,
                self.scans_comparison_treated_signal_y, self.x_ratio_values, self.y_ratio_values, self.y_max_values,
                self.scans_done, self.list_straylight_barycenter51, self.list_perturbation_barycenter51,
                self.list_straylight_barycenter53, self.list_perturbation_barycenter53, self.pic_number)




