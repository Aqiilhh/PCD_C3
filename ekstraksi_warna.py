import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv, gray2rgb
from sklearn.cluster import KMeans

# Fungsi ekstraksi warna
def extract_color_features(roi):
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return {k: 0 for k in ['mean_r', 'mean_g', 'mean_b', 'std_r', 'std_g', 'std_b',
                               'mean_h', 'mean_s', 'mean_v',
                               'dominant_r_0', 'dominant_g_0', 'dominant_b_0',
                               'dominant_r_1', 'dominant_g_1', 'dominant_b_1',
                               'dominant_r_2', 'dominant_g_2', 'dominant_b_2',
                               'hist_r_0', 'hist_r_1', 'hist_r_2', 'hist_r_3', 'hist_r_4',
                               'hist_g_0', 'hist_g_1', 'hist_g_2', 'hist_g_3', 'hist_g_4',
                               'hist_b_0', 'hist_b_1', 'hist_b_2', 'hist_b_3', 'hist_b_4']}

    hsv = rgb2hsv(roi)
    color_features = {
        'mean_r': np.mean(roi[:, :, 0]), 'mean_g': np.mean(roi[:, :, 1]), 'mean_b': np.mean(roi[:, :, 2]),
        'std_r': np.std(roi[:, :, 0]), 'std_g': np.std(roi[:, :, 1]), 'std_b': np.std(roi[:, :, 2]),
        'mean_h': np.mean(hsv[:, :, 0]), 'mean_s': np.mean(hsv[:, :, 1]), 'mean_v': np.mean(hsv[:, :, 2])
    }

    pixels = roi.reshape(-1, 3)
    if len(pixels) >= 3:
        try:
            n_clusters_actual = min(3, len(np.unique(pixels, axis=0)))
            if n_clusters_actual > 0:
                kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
                kmeans.fit(pixels)
                dominant_colors = kmeans.cluster_centers_
                for i in range(3):
                    if i < len(dominant_colors):
                        color_features[f'dominant_r_{i}'] = dominant_colors[i][0]
                        color_features[f'dominant_g_{i}'] = dominant_colors[i][1]
                        color_features[f'dominant_b_{i}'] = dominant_colors[i][2]
                    else:
                        color_features[f'dominant_r_{i}'] = 0
                        color_features[f'dominant_g_{i}'] = 0
                        color_features[f'dominant_b_{i}'] = 0
        except:
            for i in range(3):
                color_features[f'dominant_r_{i}'] = 0
                color_features[f'dominant_g_{i}'] = 0
                color_features[f'dominant_b_{i}'] = 0
    else:
        for i in range(3):
            color_features[f'dominant_r_{i}'] = 0
            color_features[f'dominant_g_{i}'] = 0
            color_features[f'dominant_b_{i}'] = 0

    hist_r = np.histogram(roi[:, :, 0], bins=16, range=[0, 256])[0]
    hist_g = np.histogram(roi[:, :, 1], bins=16, range=[0, 256])[0]
    hist_b = np.histogram(roi[:, :, 2], bins=16, range=[0, 256])[0]
    hist_r = hist_r / (np.sum(hist_r) + 1e-7)
    hist_g = hist_g / (np.sum(hist_g) + 1e-7)
    hist_b = hist_b / (np.sum(hist_b) + 1e-7)

    for i in range(5):
        color_features[f'hist_r_{i}'] = hist_r[i]
        color_features[f'hist_g_{i}'] = hist_g[i]
        color_features[f'hist_b_{i}'] = hist_b[i]

    return color_features

# Path dataset
base_dataset_path = r"D:\KULIAH\SEMESTER 4\PCD\PROJEK REGULER\dataset_sampah"
subfolders = ['b3', 'non organik', 'organik']

data = []

for label in subfolders:
    folder_path = os.path.join(base_dataset_path, label)
    if not os.path.exists(folder_path):
        print(f"❌ Folder tidak ditemukan: {folder_path}")
        continue

    # Buat folder hasil_output di dalam folder kategori
    output_category_dir = os.path.join(folder_path, "hasil_output")
    os.makedirs(output_category_dir, exist_ok=True)

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, file_name)
            try:
                img = imread(img_path)
                if len(img.shape) == 2:
                    img = gray2rgb(img)

                features = extract_color_features(img)
                features['label'] = label
                features['filename'] = file_name
                data.append(features)

                # Visualisasi warna dominan disimpan ke folder hasil_output kategori masing2
                plt.figure(figsize=(6, 3))

                dominant_colors = []
                rgb_texts = []
                for i in range(3):
                    r = features.get(f'dominant_r_{i}', 0)
                    g = features.get(f'dominant_g_{i}', 0)
                    b = features.get(f'dominant_b_{i}', 0)
                    dominant_colors.append([r/255, g/255, b/255])
                    rgb_texts.append(f"RGB({int(r)}, {int(g)}, {int(b)})")

                for i, color in enumerate(dominant_colors):
                    plt.bar(i, 1, color=color)
                    # Tambahkan teks RGB di bawah bar
                    plt.text(i, -0.15, rgb_texts[i], ha='center', va='top', fontsize=9)

                plt.xticks(range(3), [f"Warna {i+1}" for i in range(3)])
                plt.yticks([])
                plt.title(f"Warna Dominan: {label} - {file_name}")
                plt.ylim(-0.3, 1)  # beri ruang untuk teks di bawah bar
                plt.tight_layout()

                save_path = os.path.join(output_category_dir, f"{os.path.splitext(file_name)[0]}_dominant_colors.png")
                plt.savefig(save_path)
                plt.close()

                print(f"✅ Diproses dan visualisasi disimpan: {save_path}")

            except Exception as e:
                print(f"⚠️ Error saat memproses {img_path}: {e}")

# Simpan data fitur warna ke CSV
df = pd.DataFrame(data)
output_csv = os.path.join(base_dataset_path, "fitur_warna_dataset.csv")
df.to_csv(output_csv, index=False)
print(f"\n✅ Ekstraksi selesai. Data disimpan ke: {output_csv}")
