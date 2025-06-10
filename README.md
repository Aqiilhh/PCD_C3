# 🎓 Tugas Besar - Pengolahan Citra Digital

## 📌 Judul Proyek
**Ekstraksi Fitur Citra RGB untuk Klasifikasi Sampah (Plastik, Kertas, Organik)**

## 👥 Anggota Kelompok
  - Rizky Aqil Hibatullah (152023052)
  - Ni Made Dwi Salsa Anggraeni (152023012)
  - Hilda Nuraeni (152023008)

## 📖 Deskripsi Proyek
Proyek ini bertujuan untuk mengembangkan program pengolahan citra digital yang mampu mengekstraksi **fitur warna, bentuk, dan tekstur** dari citra RGB berisi objek sampah untuk tiga kategori utama: **Plastik**, **Kertas**, dan **Organik**.

## 🎯 Tujuan Proyek
- Membangun sistem ekstraksi fitur citra RGB dari dataset gambar.
- Mengimplementasikan tiga metode ekstraksi: warna (mean RGB), bentuk    (kontur), dan tekstur (GLCM).
- Mengklasifikasikan gambar berdasarkan fitur menggunakan pendekatan sederhana (perbandingan jarak).
- Menyediakan visualisasi hasil ekstraksi fitur.
- Mendokumentasikan hasil dan proses dalam laporan serta video demo.

## 🧠 Fitur yang Diekstraksi
1. Warna – Rata-rata warna dari citra RGB.
2. Bentuk – Kontur objek yang dideteksi dari citra grayscale.
3. Tekstur – Menggunakan metode GLCM untuk menghitung nilai kontras citra.

🗃️ Kategori Sampah
♻️ Anorganik
⚠️ B3 (Bahan Berbahaya & Beracun)
🍃 Organik

## 📦 Output Proyek
- ✅ Kode Program
- ✅ Dataset Citra
- ✅ Laporan Tertulis
- ✅ Video Demo
- ✅ Visualisasi hasil fitur setiap gambar

## 🗂 Struktur Proyek
📁 PCD_C3/
├── dataset_sampah/
│   ├── anorganik/
│   ├── b3/
│   └── organik/
├── ekstraksi_bentuk.py        
├── ekstraksi_teksture.py      
├── ekstraksi_warna.py        
├── PROJEK.PY                 
└── README.md     
📄 laporan.pdf
📄 demo_video.mp4

🔍 Cara Kerja
1. Gambar dimuat dari masing-masing kategori folder.
2. Gambar diresize dan dikonversi ke grayscale.
3. Fitur warna, bentuk, dan tekstur diekstraksi per gambar.
4. Fitur dibandingkan menggunakan jarak Euclidean untuk klasifikasi.
5. Visualisasi hasil disimpan ke dalam folder output per label.
