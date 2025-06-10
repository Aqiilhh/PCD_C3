import cv2
import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from matplotlib import pyplot as plt

class SampahClassifier:
    def __init__(self, folder_dataset):
        self.folder_dataset = folder_dataset
        self.data = []

    def ekstrak_fitur(self, image):
        image = cv2.resize(image, (100, 100))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def load_dataset(self):
        print("\n🔍 Memuat dataset dari:", self.folder_dataset)
        for kategori in os.listdir(self.folder_dataset):
            path_kategori = os.path.join(self.folder_dataset, kategori)
            if not os.path.isdir(path_kategori):
                continue
            for nama_file in os.listdir(path_kategori):
                path_gambar = os.path.join(path_kategori, nama_file)
                if not path_gambar.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                image = cv2.imread(path_gambar)
                if image is None:
                    print(f"⚠️ Gagal membaca gambar: {path_gambar}")
                    continue
                fitur = self.ekstrak_fitur(image)
                self.data.append((fitur, kategori, path_gambar))
        print(f"✅ Dataset berhasil dimuat: {len(self.data)} gambar\n")

    def ekstrak_visual_output(self, image, label, filename_out):
        img_resized = cv2.resize(image, (200, 200))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Warna (mean)
        mean_color = np.mean(img_resized, axis=(0, 1)).astype(np.uint8)
        color_patch = np.zeros_like(img_resized)
        color_patch[:, :] = mean_color

        # Kontur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        kontur = img_resized.copy()
        cv2.drawContours(kontur, contours, -1, (0, 255, 0), 1)

        # GLCM (tekstur)
        glcm_input = gray
        glcm = graycomatrix(glcm_input, [1], [0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]

        # Plot 4 kotak
        fig, axs = plt.subplots(2, 2, figsize=(6, 6))
        axs[0, 0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title("Asli\nPred: {}".format(label))

        axs[0, 1].imshow(cv2.cvtColor(color_patch, cv2.COLOR_BGR2RGB))
        axs[0, 1].set_title("Warna (Mean)")

        axs[1, 0].imshow(cv2.cvtColor(kontur, cv2.COLOR_BGR2RGB))
        axs[1, 0].set_title("Bentuk (Kontur)")

        axs[1, 1].imshow(glcm_input, cmap='gray')
        axs[1, 1].set_title("Tekstur (Input GLCM)")

        for ax in axs.flat:
            ax.axis('off')

        plt.tight_layout()

        # Simpan
        output_dir = os.path.join(self.folder_dataset, f"{label}_output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(filename_out))
        plt.savefig(output_path)
        plt.close()

    def klasifikasikan_semua(self):
        if len(self.data) < 2:
            print("Dataset terlalu sedikit untuk klasifikasi!")
            return

        print("📊 Memulai klasifikasi semua gambar dalam dataset...\n")
        for idx, (fitur_uji, label_asli, path_gambar) in enumerate(self.data):
            min_jarak = float('inf')
            label_prediksi = "Tidak Diketahui"

            for j, (fitur_data, label_data, _) in enumerate(self.data):
                if fitur_data is fitur_uji:
                    continue
                jarak = np.linalg.norm(fitur_uji - fitur_data)
                if jarak < min_jarak:
                    min_jarak = jarak
                    label_prediksi = label_data

            print(f"[{idx+1:02}] File: {os.path.basename(path_gambar)}")
            print(f"     Label Asli     : {label_asli}")
            print(f"     Prediksi       : {label_prediksi}")
            print(f"     Akurat?        : {'✅' if label_prediksi == label_asli else '❌'}\n")

            image = cv2.imread(path_gambar)
            if image is not None:
                self.ekstrak_visual_output(image, label_prediksi, path_gambar)

# ===== MAIN =====
if __name__ == "__main__":
    folder_dataset = r"D:\KULIAH\SEMESTER 4\PCD\PROJEK REGULER\dataset_sampah"
    classifier = SampahClassifier(folder_dataset)
    classifier.load_dataset()
    classifier.klasifikasikan_semua()