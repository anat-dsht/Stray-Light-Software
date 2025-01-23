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

import numpy as np
from typing import List, Dict

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


class PeakClustering:
    def __init__(self, freqs, data):

        # --------------------------------------------------------------------------------------------------------------
        # Data

        self.freqs: np.ndarray = freqs
        self.data: np.ndarray = data

        # --------------------------------------------------------------------------------------------------------------
        # Initialize containers

        self.barycenters: List[List[float]] = []
        self.barycenter_clusters: Dict[int, List[float]] = {}
        self.full_clusters_peaks: Dict[int, List[float]] = {}

    ####################################################################################################################
    ####################################################################################################################

    def group_points_into_peaks(self, xpeak: list, ypeak: list) -> object:
        """
        Call the functions to calculate clusters and their barycenter.

        :param xpeak: X-coordinate of the peaks.
        :param ypeak: Y-coordinate of the peaks.

        :return:
        A tuple containing:
            - barycenters (List[List[float]]): A list of barycenters for the clusters found.
            - barycenter_clusters (Dict[int, List[float]]): A dictionary of clusters with their respective data points.
            - full_clusters_peaks (Dict[int, List[float]]): A dictionary of full clusters with their peaks.
        """

        # Get signal's data
        data = np.array(list(zip(self.freqs, self.data)))

        # Call clustering function
        clusters_3pt, clusters_full = PeakClustering.find_all_peaks_points(data, xpeak, ypeak)

        # Store clusters
        self.barycenter_clusters = clusters_3pt
        self.full_clusters_peaks = clusters_full

        # Call barycenter function
        self.barycenters = PeakClustering.calculate_barycenters(clusters_3pt)

        print(f"Number of barycenter found: {len(self.barycenters)}")

        return self.barycenters, self.barycenter_clusters, self.full_clusters_peaks

    ####################################################################################################################
    ####################################################################################################################

    @staticmethod
    def find_all_peaks_points(peaks, x_peak, y_peak):
        """
        Identify and group peak points based on given peak coordinates.

        This method analyzes peak points by clustering them into two dictionaries:
        - `clusters_3pt` contains clusters with more than one point above the noise floor.
        - `clusters_full` contains clusters of all peak points above the noise floor.

        Parameters:
            peaks (np.ndarray): An array of peak data points, where each entry represents a peak.
            x_peak (List[float]): A list of x-coordinates of detected peaks.
            y_peak (List[float]): A list of y-coordinates of detected peaks.

        Returns:
            Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
                - clusters_3pt (Dict[int, np.ndarray]): A dictionary where keys are cluster IDs and values are arrays
                of clustered points (more than one point above the noise floor).

                - clusters_full (Dict[int, np.ndarray]): A dictionary where keys are cluster IDs and values are arrays
                of all peak points (including single points) above the noise floor.
        """
        # --------------------------------------------------------------------------------------------------------------
        # Initialize dictionaries and cluster IDs

        clusters_3pt = {}
        clusters_full = {}
        cluster_id_full = 0
        cluster_id = 0

        # --------------------------------------------------------------------------------------------------------------
        # Iterate over the peak points to classify them into clusters

        for i, peak_point in enumerate(zip(x_peak, y_peak)):
            x_val, y_val = peak_point

            # Find the cluster points and peaks points related to the current peak point
            cluster_points, peaks_points = PeakClustering.find_peaks_points(peaks, x_val, y_val)

            # ----------------------------------------------------------------------------------------------------------
            # Add to clusters_3pt if there are more than one point in the cluster

            # Only consider clusters with more than one point
            if len(cluster_points) > 1:
                clusters_3pt[cluster_id] = np.array(cluster_points)
                cluster_id += 1

            # ----------------------------------------------------------------------------------------------------------
            # Add to clusters_full if there are any peaks points

            # Only consider peaks with at least one point
            if len(peaks_points) > 0:
                clusters_full[cluster_id_full] = np.array(peaks_points)
                cluster_id_full += 1

        # --------------------------------------------------------------------------------------------------------------
        # Return the dictionaries of clusters

        return clusters_3pt, clusters_full

    ####################################################################################################################
    ####################################################################################################################

    @staticmethod
    def find_peaks_points(peaks, x_val, y_val):
        """
        Find cluster points and all peak points for a given peak point.

        :param peaks: Array of peaks with x and y coordinates.
        :param x_val: X-coordinate of the peak point.
        :param y_val: Y-coordinate of the peak point.

        :return:
            Tuple containing:
            - cluster_points: List of points used to calculate the barycenter.
            - peaks_points: List of all points of the peak.
        """

        # --------------------------------------------------------------------------------------------------------------
        # Initialize lists for cluster points and peak points

        cluster_points = [(x_val, y_val)]
        peaks_points = [(x_val, y_val)]

        # Find the index of the corresponding peak in the peaks array
        peak_index = np.where((peaks[:, 0] == x_val) & (peaks[:, 1] == y_val))[0]

        if peak_index.size > 0:
            peak_index = peak_index[0]

            # --------------------------------------------------------------------------------------------------------------
            # Add neighboring points to cluster and peak points lists

            for direction in [-1, 1]:
                index = peak_index + direction

                # Add points to the cluster if they meet the condition
                if 0 <= index < len(peaks) and peaks[index - direction, 1] >= peaks[index, 1]:
                    cluster_points.append((peaks[index, 0], peaks[index, 1]))

                # Add points to the peak points list
                while 0 <= index < len(peaks) and peaks[index - direction, 1] >= peaks[index, 1]:
                    peaks_points.append((peaks[index, 0], peaks[index, 1]))
                    index += direction

        return cluster_points, peaks_points

    ####################################################################################################################
    ####################################################################################################################

    @staticmethod
    def calculate_barycenters(cluster):
        """
        Calculate the weighted barycenters for the given clusters.

        :param cluster: Dictionary of clusters where each key is a cluster ID and each value is an array of points.

        :return:
            List of barycenters for each cluster.
        """

        barycenters = []

        # --------------------------------------------------------------------------------------------------------------
        # Calculate barycenters for each cluster

        for label, points in cluster.items():
            x_coord = points[:, 0]
            y_coord = points[:, 1]

            # Calculate the weighted barycenter
            weighted_x = np.sum(x_coord * y_coord)
            total_weight = np.sum(y_coord)
            weighted_barycenter_x = weighted_x / total_weight
            barycenter = (weighted_barycenter_x, np.mean(y_coord))

            barycenters.append(barycenter)

        return barycenters

    ####################################################################################################################
    ####################################################################################################################

    def calculate_clustering_quality(self, clusters, peaks_x, num_clusters, num_peaks):
        """
        Calculate the clustering quality based on the number of clusters versus the number of detected peaks.

        :param clusters: Dictionary of clusters where each key is a cluster ID and each value is an array of points.
        :param peaks_x: List of x-coordinates of detected peaks.
        :param num_clusters: Total number of clusters detected.
        :param num_peaks: Total number of detected peaks.

        :return:
            quality_score: A measure of clustering quality (higher values indicate better quality).
        """

        # --------------------------------------------------------------------------------------------------------------
        # Calculate the mean x-coordinates for cluster centers

        cluster_centers_x = [np.mean(cluster[:, 0]) for cluster in clusters.values()]

        # --------------------------------------------------------------------------------------------------------------
        # Check the distance between cluster centers and peaks

        peak_positions = set(peaks_x)  # Remove duplicates
        distance_sum = 0

        for center_x in cluster_centers_x:
            if not any(abs(center_x - peak_x) <= 0.2 for peak_x in peak_positions):
                distances = [abs(center_x - peak_x) for peak_x in peak_positions]
                min_distance = min(distances) if distances else 0
                distance_sum += min_distance

        # --------------------------------------------------------------------------------------------------------------
        # Normalize distance and calculate distance quality

        max_distance = max(self.freqs) - min(self.freqs)
        distance_quality = 1 - (distance_sum / (len(cluster_centers_x) * max_distance))

        # --------------------------------------------------------------------------------------------------------------
        # Check that the number of clusters matches the number of peaks

        quality_score = 1 / (1 + abs(num_clusters - num_peaks))

        # Calculate combined quality
        combined_quality = (1 * quality_score + 1 * distance_quality)

        return combined_quality
