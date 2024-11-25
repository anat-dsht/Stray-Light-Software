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

from typing import Callable, Optional
import numpy as np
import os

########################################################################################################################
# -------------------------------------------------------------------------------------------------------------------- #
########################################################################################################################


def define_harm_sect():
    fond_sect_freq = int(50)
    return fond_sect_freq

########################################################################################################################
########################################################################################################################


def html_dialogbox(tab):

    # ==================================================================================================================
    # Visualisation Tab

    if tab == "VisualisationTab":
        text = """
            <head>
                <style>
                    body {
                        line-height: 1.6;
                    }
                    h2 {
                    }
                    h3 {
                    }
                    ul {
                        margin: 0;
                        padding: 0;
                        list-style-type: disc;
                        padding-left: 20px;
                    }
                    li {
                        margin-bottom: 10px;
                    }
                </style>
            </head>
            <body>
                <h2>Welcome to the <b>Visualisation Tab</b>!</h2>
                <p>In this tab, you'll be able to display and overlay all the signals you want from the database.</p>
                
                <h3>Data Selection:</h3>
                <ul>
                    <li><b>Measurement:</b> Choose the measurement you want to plot.</li>
                    <li><b>Record:</b> Choose the measurement type.</li>
                    <li><b>Pico:</b> Choose the pico from which the signal got out.</li>
                    <li><b>Canal:</b> Select the canal of the pico.</li>
                </ul>
                
                <h3>Plotting Options:</h3>
                <ul>
                    <li><b>Log Options:</b> Select which axis to display in logarithmic (by default None).</li>
                    
                    <li><b>Y-axis:</b></li>
                    <ul>
                    <li><b>Fractional amplitude:</b> Given by Vpk/VDC.</li>
                    <li><b>Amplitude:</b> Given in Vpk.</li>
                    <li><b>Temporal amplitude:</b> Given in mV.</li>
                    </ul>
                    
                    <li><b>X-axis:</b></li>
                    <ul>
                    <li><b>Frequency:</b> The X-axis unit will in Hertz.</li>
                    <li><b>Optical Path Difference:</b> The X-axis unit will in Meter.</li>
                    <li><b>Time:</b> The X-axis unit will in Second.</li>
                    </ul>
                    
                    <li><b>Title:</b> Customize the plot's title (Optional).</li>
                    <li><b>Commentary:</b> Add a commentary which will be shown below the title.</li>
                </ul>
                
                <h3>Visualisation description:</h3>
                <ul>
                    <li><b>Plot:</b> By pressing the button, the selected signal will be plotted and selected plot 
                    parameters will be applied.</li>
                    <li><b>Clear plot:</b> The signal currently displayed will be overdraw by the new signal.</li>
                    <li><b>Overlay plot:</b> The new signal will overlay with the currently displayed signal 
                    (It must have the same plot parameters).</li>
                </ul>
            </body>
            """

    # ==================================================================================================================
    # Treatment Tab

    else:
        text = """
                    <head>
                        <style>
                            body {
                                line-height: 1.6;
                            }
                            h2 {
                            }
                            h3 {
                            }
                            ul {
                                margin: 0;
                                padding: 0;
                                list-style-type: disc;
                                padding-left: 20px;
                            }
                            li {
                                margin-bottom: 10px;
                            }
                        </style>
                    </head>
                    <body>
                        <h2>Welcome to the <b>Stray Light Processing Tab</b>!</h2>
                        <p>In this tab, you'll be able to clean all the perturbations from a chosen signal.</p>

                        <h3>Data Selection:</h3>
                        <ul>
                            <li><b>Measurement:</b> Choose the measurement you want to clean.</li>
                            <li><b>Pico:</b> Choose the pico from which you want to clean the signal.</li>
                            <li><b>Canal:</b> Select the canal of the pico, which will be cleaned.</li>
                        </ul>

                        <h3>Plotting Options:</h3>
                        <ul>
                            <li><b>Log Options:</b> Select which axis to display in logarithmic (by default None).</li>
                            <li><b>Y Axis:</b> Choose the Y-axis' unit (Select 'Fractional Amplitude').</li>
                            <li><b>X Axis:</b> Choose the X-axis' unit (Select 'Optical Path Difference (OPD)').</li>
                            <li><b>Title:</b> Customize the plot's title (Optional).</li>
                            <li><b>Plot options:</b></li>
                            <ul>
                            <li><b>Noise floor:</b> Display the noise floor approximation for both scans.</li>
                            <li><b>Scan 51/53 plot:</b> Display data from scan5.1 and scan 5.3.</li>
                            <li><b>Barycenter:</b> Display the barycenter calculated from each peak detected.</li>
                            <li><b>Treated signal:</b> Display the output signal in which remains the SL peaks only.
                            </li>
                            </ul>
                        </ul>

                        <h3>Process description:</h3>
                        <ul>
                            <li><b>Step 1:</b> Link Folder, Pico and Canal choices to get the Scan 5.1 and the Scan 5.3 
                            of the 
                            selected signal.</li>
                            <li><b>Step 2:</b> Extract the signal's data in function of the X axis and Y axis selection.
                            </li>
                            <li><b>Step 3:</b> Identify the noise floor for the whole signal bandwidth.</li>
                            <li><b>Step 4:</b> Detect peaks in function of the noise floor values found..</li>
                            <li><b>Step 5:</b> For each peak, create cluster with the points that compose the peak.</li>
                            <li><b>Step 6:</b> Compute the barycenter of each 3 points' cluster found .</li>
                            <li><b>Step 7:</b> Compare barycenter positions to complete a first treatment to get rid of 
                            most of the 
                            perturbation peaks.</li>
                            <li><b>Step 8:</b> Extract statistical data to fill the data analysis tab..</li>
                        </ul>
                    </body>
                    """

    # ==================================================================================================================
    # Return

    return text

########################################################################################################################
########################################################################################################################


def execute_with_error_handling(func: Callable, argument: Optional = None) -> object:
    """
    Execute a function with error handling and raise an exception with a detailed message if an error occurs.
    """
    try:
        if argument is not None:
            output = func(argument)
        else:
            output = func()
    except AttributeError as e:
        # QMessageBox.warning(self, "No Canal Selected", "Please select a canal before plotting.")
        raise AttributeError(f"Attribute Error: {e}") from e
    except TypeError as e:
        raise TypeError(f"Type Error: {e}") from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File Not Found Error: {e}") from e
    except ValueError as e:
        raise ValueError(f"Value Error: {e}") from e
    except IndexError as e:
        raise IndexError(f"Index Error: {e}") from e
    except KeyError as e:
        raise KeyError(f"Key Error: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected Error: {e}") from e

    if output:
        return output

########################################################################################################################
########################################################################################################################


def add_legends(path):
    """
    Choose legends for the type of record of the plot in function of the user choices.

    :param path: Path to the file for which legend is to be determined.
    :return: Appropriate legend string based on the file name and its parent directories.
    """
    # Get the absolute path of the given path
    abs_path = os.path.abspath(path)

    # Get the parent directory of the file
    parent_path = os.path.dirname(abs_path)

    # Get the grandparent directory of the file
    grandparent_path = os.path.dirname(parent_path)

    # Get the base name of the grandparent directory
    grandparent_foldername = os.path.basename(grandparent_path)

    # Get the base name of the file
    file_name = os.path.basename(abs_path)

    # Determine the legend based on the file name and its parent directories
    if 'dark' in file_name.lower():
        return "Dark"
    if 'noscan' in file_name.lower() or 'noscan' in grandparent_foldername.lower():
        return "No Scan"
    if 'scan51' in file_name.lower():
        return "Scan 5,1GHz/s"
    if 'scan53' in file_name.lower():
        return "Scan 5,3GHz/s"

########################################################################################################################
########################################################################################################################


def removewidget(lay, widget):
    lay.removeWidget(widget)
    widget.deleteLater()

########################################################################################################################
########################################################################################################################


def are_colors_similar(color1, color2):
    return np.all(np.array(color1[:3]) - np.array(color2[:3]))

########################################################################################################################
########################################################################################################################


def printing(object_to_print: object, line_size: object, line_type: object) -> object:
    print(line_type * line_size)
    print(object_to_print)
    print(line_type * line_size)

########################################################################################################################
########################################################################################################################


'''def fill_table_with_data(data, table):

    num_points = len(data[0])

    table.setRowCount(num_points+1)
    table.setColumnCount(len(data))

    headers = ["Scan 5.1 GHz/s", "Scan 5.3 GHz/s"]
    sub_headers = " Barycenter :[X, Y]"

    table.setHorizontalHeaderLabels(headers)
    table.setItem(0, 0, QTableWidgetItem(sub_headers))
    table.setItem(0, 1, QTableWidgetItem(sub_headers))

    for col in range(len(data)):
        for row in range(num_points):
            value = []
            for i in range(2):
                v = format(data[col][row][i], ".3e")
                value.append(v)
            table.setItem(row+1, col, QTableWidgetItem(str(value)))

    column_width = 250  # Définissez la largeur souhaitée pour chaque colonne
    for col in range(num_points):
        table.setColumnWidth(col, column_width)'''


"""def show_histogram(self, hist, bin_edges, noise_floor_range):
        '''Plot the histogram and indicate the noise floor range.'''
        if self.layout_graph.count() > 0:
            for i in reversed(range(self.layout_graph.count())):
                widget = self.layout_graph.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        ax = self.figure.add_subplot(111)

        # Plot the histogram
        ax.hist(bin_edges[:-1], bin_edges, weights=hist, color='blue', alpha=0.7, label='Amplitude Histogram')

        # Highlight the noise floor range
        ax.axvspan(noise_floor_range[0], noise_floor_range[1], color='red', alpha=0.3, label='Noise Floor Range')

        ax.set_xlabel('Amplitude')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid()

        self.layout_graph.addWidget(self.canvas)
        self.layout_graph.addWidget(NavigationToolbar(self.canvas, self))"""

'''def separate_peaks_with_dbscan(self, ax, noise_floor_values, n_trial=100, eps_range=(0.1, 5),
                                   min_samples_range=(1, 10)):
        """
        Séparer les pics en groupes horizontaux et verticaux en utilisant DBSCAN.

        Parameters:
        eps: Distance maximale entre deux échantillons pour qu'ils soient considérés comme voisins.
        min_samples: Nombre minimum d'échantillons pour qu'un point soit considéré comme un noyau d'un cluster.
        """

        best_eps = eps_range[0]
        best_min_samples = min_samples_range[0]
        best_clusters = None
        best_score = -1
        best_quality = -1

        xpeak, ypeak, num_peak = self.detect_peaks(ax, noise_floor_values)
        peaks = np.array(list(zip(self.peaks_x, self.peaks_y)))

        # Phase de recherche des meilleurs paramètres
        for _ in range(n_trial):
            eps = np.random.uniform(max(best_eps - 0.1, eps_range[0]), min(best_eps + 0.1, eps_range[1]))
            min_samples = np.random.randint(max(best_min_samples - 2, min_samples_range[0]),
                                            min(best_min_samples + 2, min_samples_range[1]))

            # Appliquer DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            labels = dbscan.fit_predict(peaks)

            # Compter le nombre de clusters
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if num_clusters > 1:
                # Regarde à quel point les points sont proches dans le même cluster & à quel point les clusters 
                sont éloignés
                clusters = {label: peaks[labels == label] for label in np.unique(labels) if label != -1}
                score = silhouette_score(peaks, labels)
                quality = self.calculate_clustering_quality(clusters, xpeak, num_clusters, num_peak)

                # Vérifie si le score de silhouette ou la qualité sont meilleurs que les meilleurs précédents
                if score > best_score:
                    best_eps = eps
                    best_min_samples = min_samples
                    best_score = score
                    best_clusters = clusters

                if quality > best_quality:
                    best_eps = eps
                    best_min_samples = min_samples
                    best_quality = quality
                    best_clusters = clusters

        # Affichage des résultats de la phase de recherche des meilleurs paramètres
        print("Phase de recherche terminée.")
        print("Meilleurs paramètres trouvés : eps =", best_eps, ", min_samples =", best_min_samples)
        print(f"Meilleur score de silhouette : {best_score}")
        print(f"Meilleure qualité des clusters : {best_quality}")

        # Phase d'ajustement des clusters
        eps_final, min_samples_final = best_eps, best_min_samples
        dbscan_final = DBSCAN(eps=eps_final, min_samples=min_samples_final, metric='euclidean')
        labels_final = dbscan_final.fit_predict(peaks)
        final_clusters = {label: peaks[labels_final == label] for label in np.unique(labels_final) if label != -1}
        final_score = silhouette_score(peaks, labels_final)
        final_quality = self.calculate_clustering_quality(final_clusters, xpeak, num_clusters, num_peak)

        # Ajustement des points des clusters pour améliorer le score et la qualité
        iteration = 0
        max_iterations = 50  # Limiter le nombre d'itérations pour éviter une boucle infinie

        while (final_score < best_score or final_quality < best_quality) and iteration < max_iterations:
            iteration += 1

            # Appliquer une légère perturbation sur les points de chaque cluster
            new_peaks = []
            for label, points in final_clusters.items():
                # Calculer le centre de chaque cluster
                center = np.mean(points, axis=0)
                # Ajouter un bruit aléatoire autour du centre du cluster
                noise = np.random.normal(loc=0.0, scale=0.05, size=points.shape)
                new_peaks.extend(points + noise)

            new_peaks = np.array(new_peaks)

            # Réappliquer DBSCAN avec les nouveaux points perturbés
            dbscan_final = DBSCAN(eps=eps_final, min_samples=min_samples_final, metric='euclidean')
            labels_final = dbscan_final.fit_predict(new_peaks)
            final_clusters = {label: new_peaks[labels_final == label] for label in np.unique(labels_final) if
                              label != -1}
            final_score = silhouette_score(new_peaks, labels_final)
            final_quality = self.calculate_clustering_quality(final_clusters, xpeak, len(final_clusters), num_peak)

            if final_score > best_score and final_quality > best_quality:
                best_eps, best_min_samples = eps_final, min_samples_final
                best_score, best_quality = final_score, final_quality
                best_clusters = final_clusters

        # Affichage des résultats de la phase d'ajustement
        print("Phase d'ajustement terminée.")
        print("Meilleurs paramètres ajustés : eps =", best_eps, ", min_samples =", best_min_samples)
        print(f"Score de silhouette final : {final_score}")
        print(f"Qualité des clusters finale : {final_quality}")

        # Visualisation des clusters finaux
        for label, points in best_clusters.items():
            ax.scatter(points[:, 0], points[:, 1], label=f'Cluster {label}', alpha=0.5)
        ax.legend()

        return best_eps, best_min_samples, best_clusters'''

'''def detect_noise_floor(self):
        """Detect and print the noise floor level."""
        if self.yaxis is not None:
            data = np.array(self.yaxis)

            # Calculate the histogram of the data
            hist, bin_edges = np.histogram(data, bins=500, range=(0, 1e-5))

            # Identify the bin with the highest frequency
            noise_floor_bin = np.argmax(hist)
            noise_floor_range = (bin_edges[noise_floor_bin], bin_edges[noise_floor_bin + 1])

            QMessageBox.information(self, "Noise Floor Detected",
                                    f"Noise floor is approximately between {noise_floor_range[0]:.2e} and 
                                    {noise_floor_range[1]:.2e}")
            # self.show_histogram(hist, bin_edges, noise_floor_range)

            return [noise_floor_range[0], noise_floor_range[1]]
        else:
            QMessageBox.warning(self, "No Data", "No data to analyze. Please load data first.")
            return 1

    def apply_filter(self):
        """Apply the notch filters to reduce the amplitude of the sectorial harmonics"""

        if self.yaxis is not None:

            base_freq = define_harm_sect()
            delta_freq = 0.3
            noise_floor_range = self.detect_noise_floor()
            harmonic_indices = []

            # Identify points belonging to the noise floor and to pics
            self.noise_floor_indices = np.where((self.data >= noise_floor_range[0]) & 
            (self.data <= noise_floor_range[1]))[0]
            self.pics_indices = np.where(self.data > noise_floor_range[1])[0]
            self.pics_freq = self.freqs[self.pics_indices]

            for n in range(1, 6):
                low_freq = base_freq*n-delta_freq
                high_freq = base_freq * n + delta_freq
                indices = np.where((self.freqs >= low_freq) & (self.freqs <= high_freq))[0]
                harmonic_indices.append(indices)
                if n > 1:
                    self.freqs[indices] = self.freqs[indices]/n

            # Flatten the list of indices and remove potential duplicates
            harmonic_indices = np.unique(np.concatenate(harmonic_indices))

            """for indice in harmonic_indices:
                if self.data[indice] >= noise_floor_range[1]*2:
                    self.data[indice] = noise_floor_range[1]/2"""

            self.yaxis = self.data

        else:
            QMessageBox.warning(self, "No Data", "No data to divide. Please load data first.")
            
            
    
    def weighted_weiszfeld(points, weights, epsilon=1e-5):
        points = np.array(points)
        weights = np.array(weights)
    
        # Initial guess: weighted average of points
        P = np.average(points, axis=0, weights=weights)
    
        while True:
            P_new = np.zeros_like(P)
            denom = 0
            for point, weight in zip(points, weights):
                distance = np.linalg.norm(P - point)
                if distance > epsilon:
                    P_new += weight * point / distance
                    denom += weight / distance
    
            P_new /= denom
    
            # Check convergence
            if np.linalg.norm(P_new - P) < epsilon:
                break
    
            P = P_new
    
        return P'''
