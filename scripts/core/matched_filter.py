"""
Matched filter utilities: clustering, statistics, filter computation, and
optional plotting. Shared between PRISMA and EnMAP processing pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def k_means_hyperspectral(image, k):
    """

    Per eseguire una classificazione non supervisionata utilizzando il metodo K-means su un'immagine iperspettrale, dovrai seguire alcuni passaggi chiave. Prima di tutto, avrai bisogno dell'immagine iperspettrale e della scelta del numero di cluster (K) per il K-means. Gli step includono:

    1. **Caricare l'Immagine Iperspettrale**: Questa immagine sarà un array tridimensionale con dimensioni (n_righe, n_colonne, n_bande), dove ogni "banda" rappresenta un diverso spettro di luce catturato.

    2. **Rimodellare l'Immagine per il Clustering**: Il K-means lavora su dati bidimensionali, quindi devi rimodellare l'immagine da (n_righe, n_colonne, n_bande) a (n_righe*n_colonne, n_bande). In questo modo, ogni pixel (ora una riga nel nuovo array) sarà rappresentato dal suo spettro (le bande).

    3. **Applicare il K-means Clustering**: Esegui l'algoritmo K-means sui dati rimodellati. Ciò raggrupperà i pixel in K gruppi basati sulle loro proprietà spettrali.

    4. **Rimodellare l'Output per l'Immagine Classificata**: Dopo il clustering, avrai le etichette dei cluster per ogni pixel. Queste etichette devono essere rimodellate di nuovo in un formato di immagine (n_righe, n_colonne) per visualizzare l'immagine classificata.

    5. **Visualizzare l'Immagine Classificata**: Ogni pixel dell'immagine può ora essere colorato in base al cluster di appartenenza, permettendoti di visualizzare i diversi cluster.

    Se hai l'immagine iperspettrale e vuoi procedere con la classificazione, posso aiutarti a scrivere e eseguire il codice necessario. Fammi sapere se hai l'immagine e il numero di cluster desiderato.

    Applica il K-means clustering a un'immagine iperspettrale.

    Parametri:
    image (numpy array): L'immagine iperspettrale con dimensioni (n_righe, n_colonne, n_bande).
    k (int): Il numero di cluster da utilizzare nel K-means.

    Ritorna:
    numpy array: Immagine classificata con dimensioni (n_righe, n_colonne).
    """

    # Ottieni le dimensioni dell'immagine
    n_righe, n_colonne, n_bande = image.shape

    # Rimodella l'immagine per il clustering
    reshaped_image = image.reshape((n_righe * n_colonne, n_bande))

    # Applica il K-means clustering
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(reshaped_image)

    # Ottieni le etichette dei cluster e rimodella per l'immagine classificata
    clustered_image = kmeans.labels_.reshape((n_righe, n_colonne))

    return clustered_image


def plot_classified_image(classified_image, title="Classified Image"):
    """
    Visualizza l'immagine classificata.

    Parametri:
    classified_image (numpy array): Immagine classificata con dimensioni (n_righe, n_colonne).
    title (str): Titolo del grafico.
    """

    plt.figure(figsize=(8, 6))
    plt.imshow(classified_image, cmap="jet")
    plt.colorbar()
    plt.title(title)
    plt.show(block=False)
    plt.show()


def calculate_statistics(image, classified_image, k):
    """
    Calcola la radianza media e la matrice di covarianza per ogni cluster.

    Parametri:
    image (numpy array): Immagine iperspettrale originale con dimensioni (n_righe, n_colonne, n_bande).
    classified_image (numpy array): Immagine classificata con dimensioni (n_righe, n_colonne).
    k (int): Numero di cluster.

    Ritorna:
    mean_radiance (list of numpy arrays): Rad
    """
    n_rows, n_columns, n_bands = image.shape
    mean_radiance = []
    covariance_matrices = []

    for cls in range(k):
        class_mask = classified_image == cls
        class_pixels = image[class_mask]
        if class_pixels.size == 0:
            # Handle empty clusters
            mean_radiance.append(np.zeros(n_bands))
            covariance_matrices.append(np.zeros((n_bands, n_bands)))
            continue
        mean_radiance.append(np.mean(class_pixels, axis=0))
        covariance_matrices.append(np.cov(class_pixels, rowvar=False))
    mean_radiance = np.array(mean_radiance)
    covariance_matrices = np.array(covariance_matrices)
    return mean_radiance, covariance_matrices


def calculate_matched_filter(rads_array, classified_image, mean_radiance, covariance_matrices, target_spectra, k):
    """
    Applies the matched filter to the hyperspectral data according to the formulation in the paper.

    Parameters:
    - rads_array: The hyperspectral radiance data cube (rows x cols x bands).
    - classified_image: The classification map from k-means clustering.
    - mean_radiance: List or array of mean radiance vectors for each class.
    - covariance_matrices: List or array of covariance matrices for each class.
    - target_spectra: The unit absorption spectrum (k) for methane.
    - k: Number of clusters/classes.

    Returns:
    - concentration_map: The CH4 concentration enhancement map.
    """
    n_rows, n_columns, n_bands = rads_array.shape
    concentration_map = np.zeros((n_rows, n_columns))
    regularization = 1e-6

    for cls in range(k):
        # Inverse of the regularized covariance matrix
        inv_cov_matrix_cls = np.linalg.inv(covariance_matrices[cls] + regularization * np.eye(n_bands))

        # Mean radiance for the current class
        mean_radiance_cls = mean_radiance[cls]

        # Target signature including mean radiance (t = mu * k)
        target_spc_cls = mean_radiance_cls * target_spectra

        # Pixels belonging to the current class
        class_mask = classified_image == cls
        pixels = rads_array[class_mask]

        # Numerator and denominator of the concentration enhancement formula
        numerator = (pixels - mean_radiance_cls) @ inv_cov_matrix_cls @ target_spc_cls
        denominator = target_spc_cls.T @ inv_cov_matrix_cls @ target_spc_cls

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            concentration_map[class_mask] = numerator / denominator

    return concentration_map


def calculate_matched_filter_columnwise(
    rads_array, classified_image, mean_radiance, covariance_matrices, target_spectra, k
):
    """
    Column-wise matched filter used for PRISMA spectral smile mitigation.
    """
    n_rows, n_columns, n_bands = rads_array.shape
    concentration_map = np.zeros((n_rows, n_columns))
    regularization = 1e-6

    for w in range(n_columns):
        # Precompute adapted filters for each class for this column
        adapted_filters = []
        target_normalizations = []
        for cls in range(k):
            # Column-wise target spectrum
            column_unit_spectrum = target_spectra[:, w]
            mean_radiance_cls = mean_radiance[cls]

            # Construct the target spectrum t by scaling the unit absorption spectrum k by μ:
            target_spectrum = mean_radiance_cls * column_unit_spectrum

            # Retrieve and regularize the covariance matrix for numerical stability
            cov_matrix_cls = covariance_matrices[cls] + regularization * np.eye(n_bands)

            # Invert the covariance matrix
            inv_cov_matrix_cls = np.linalg.inv(cov_matrix_cls)

            # Compute the denominator of the matched filter for normalization:
            target_normalization = target_spectrum.T @ inv_cov_matrix_cls @ target_spectrum

            # Compute the adapted filter for the target:
            adapted_filter = inv_cov_matrix_cls @ target_spectrum

            # Store results for this class
            adapted_filters.append(adapted_filter)
            target_normalizations.append(target_normalization)

        # Apply the matched filter to each pixel in this column
        for i in range(n_rows):
            # Identify the class of the current pixel
            pixel_class = classified_image[i, w]
            mean_radiance_cls = mean_radiance[pixel_class]

            # Retrieve the adapted filter and normalization for this pixel's class
            adapted_filter = adapted_filters[pixel_class]
            target_normalization = target_normalizations[pixel_class]

            # Extract the pixel's measured radiance spectrum
            pixel_spectrum = rads_array[i, w, :]

            # Compute the numerator of the matched filter:
            numerator = (pixel_spectrum - mean_radiance_cls).T @ adapted_filter

            # Compute the concentration value:
            concentration_value = numerator / target_normalization if target_normalization != 0 else 0.0

            # Store the concentration result
            concentration_map[i, w] = concentration_value

    return concentration_map


def plot_matched_filter_scores(scores, title="Matched Filter Scores"):
    """
    Visualizza i punteggi del filtro adattato utilizzando la palette turbo.

    Parametri:
    scores (numpy array): Array dei punteggi del filtro adattato.
    title (str): Titolo del grafico.
    """

    plt.figure(figsize=(10, 10))
    plt.imshow(scores, cmap="turbo")
    plt.colorbar()
    plt.title(title)
    plt.show(block=False)
    plt.show()

