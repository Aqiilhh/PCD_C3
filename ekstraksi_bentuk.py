# --- 1. Import library ---
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import measure, filters
from skimage.color import rgb2gray
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

# --- 2. Fungsi ekstraksi fitur bentuk ---
def extract_shape_features(roi):
    import numpy as np
    from skimage import measure, filters
    from skimage.color import rgb2gray
    from skimage.morphology import closing, square
    from skimage.segmentation import clear_border

    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return {k: 0 for k in ['area', 'perimeter', 'eccentricity', 'solidity',
                               'extent', 'major_axis', 'minor_axis', 'orientation',
                               'compactness', 'aspect_ratio',
                               'hu_moment_0', 'hu_moment_1', 'hu_moment_2',
                               'hu_moment_3', 'hu_moment_4']}

    gray = rgb2gray(roi)

    try:
        thresh = filters.threshold_otsu(gray)
        binary = gray > thresh
        binary = clear_border(binary)
        binary = closing(binary, square(3))
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)

        if not regions:
            binary = np.ones_like(gray, dtype=bool)
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled)
            if not regions:
                raise ValueError("Tidak ada region ditemukan bahkan setelah fallback.")
        largest_region = max(regions, key=lambda x: x.area)
    except Exception:
        binary = np.ones_like(gray, dtype=bool)
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        if not regions:
            return {k: 0 for k in ['area', 'perimeter', 'eccentricity', 'solidity',
                                   'extent', 'major_axis', 'minor_axis', 'orientation',
                                   'compactness', 'aspect_ratio',
                                   'hu_moment_0', 'hu_moment_1', 'hu_moment_2',
                                   'hu_moment_3', 'hu_moment_4']}
        largest_region = max(regions, key=lambda x: x.area)

    shape_features = {
        'area': largest_region.area,
        'perimeter': largest_region.perimeter if largest_region.perimeter > 0 else 0,
        'eccentricity': largest_region.eccentricity if largest_region.eccentricity is not None else 0,
        'solidity': largest_region.solidity if largest_region.solidity is not None else 0,
        'extent': largest_region.extent if largest_region.extent is not None else 0,
        'major_axis': largest_region.major_axis_length if largest_region.major_axis_length is not None else 0,
        'minor_axis': largest_region.minor_axis_length if largest_region.minor_axis_length is not None else 0,
        'orientation': largest_region.orientation if largest_region.orientation is not None else 0,
    }

    perimeter_sq = largest_region.perimeter ** 2 if largest_region.perimeter > 0 else 1e-7
    minor_axis_len = largest_region.minor_axis_length if largest_region.minor_axis_length and largest_region.minor_axis_length > 0 else 1e-7
    major_axis_len = largest_region.major_axis_length if largest_region.major_axis_length and largest_region.major_axis_length > 0 else 1e-7

    shape_features['compactness'] = (4 * np.pi * largest_region.area) / (perimeter_sq + 1e-7)
    shape_features['aspect_ratio'] = major_axis_len / minor_axis_len

    try:
        moments = measure.moments_hu(measure.moments(binary.astype(np.uint8)))
        for i, moment in enumerate(moments[:5]):
            shape_features[f'hu_moment_{i}'] = moment if not np.isnan(moment) and not np.isinf(moment) else 0
    except Exception:
        for i in range(5):
            shape_features[f'hu_moment_{i}'] = 0

    return shape_features

    # <-- fungsi yang sudah kamu tulis sebelumnya, masukkan di sini secara utuh -->

# --- 3. Kode utama: Loop folder dan gambar ---
dataset_path = r"D:\KULIAH\SEMESTER 4\PCD\PROJEK REGULER\dataset_sampah"

for kategori in os.listdir(dataset_path):
    kategori_path = os.path.join(dataset_path, kategori)
    if os.path.isdir(kategori_path):
        for filename in os.listdir(kategori_path):
            file_path = os.path.join(kategori_path, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = imread(file_path)
                    fitur_bentuk = extract_shape_features(img)

                    plt.imshow(img)
                    plt.title(f"{kategori} - {filename}")
                    plt.axis('off')
                    plt.show()

                    print(f"Kategori: {kategori}, Nama File: {filename}")
                    print("Fitur Bentuk:")
                    for key, value in fitur_bentuk.items():
                        print(f"  {key}: {value}")
                    print("\n" + "-"*50 + "\n")

                except Exception as e:
                    print(f"Error saat memproses {file_path}: {e}")
