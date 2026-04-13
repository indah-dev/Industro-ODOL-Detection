**Analisis Komparasi Algoritma YOLO, SSD, dan Faster R-CNN untuk Deteksi Objek Pelanggaran Truk ODOL (Over Dimension Overload)
**
Repositori ini berisi kode sumber pendukung untuk penelitian komparatif algoritma Deep Learning (YOLOv8 Nano, SSD, dan Faster R-CNN) dalam mendeteksi pelanggaran truk muatan berlebih (ODOL). Penelitian ini ditujukan untuk memenuhi standar transparansi dan reproducibility (keterulangan eksperimen) dalam publikasi jurnal Q1.


**📂 Struktur Repositori**
YOLOv8n.ipynb : Skrip pelatihan dan evaluasi untuk model YOLOv8 Nano.

SSD.ipynb : Skrip pelatihan dan evaluasi untuk model Single Shot MultiBox Detector (SSD).

R_CNN.ipynb : Skrip pelatihan dan evaluasi untuk model Faster R-CNN.

app.py : (Opsional) Antarmuka purwarupa untuk pengujian deteksi secara visual.

**📥 Pengaturan Bobot Model (Model Weights)**
Untuk menjaga agar repositori GitHub tetap ringan, file hasil pelatihan (bobot model) yang berukuran besar tidak diunggah secara langsung ke repositori ini.

Seluruh model final telah dicadangkan secara permanen dan dapat diakses publik melalui Hugging Face.

Tautan Unduhan Model:
👉 [MASUKKAN LINK REPOSITORI HUGGING FACE ANDA DI SINI]

Di dalam tautan tersebut, Anda akan menemukan tiga file:

best_yolo.pt (Untuk YOLOv8 Nano)

best_ssd.pth (Untuk SSD)

best_frcnn.pth (Untuk Faster R-CNN)

**⚙️ Petunjuk Eksekusi (Panduan untuk Reviewer / Dosen Penguji)**
Jika Bapak/Ibu ingin menjalankan atau mengevaluasi kode di Google Colab tanpa melakukan pelatihan ulang (training dari awal), mohon ikuti langkah-langkah berikut agar skrip berjalan tanpa error pathing:

Unduh Model: Unduh ketiga file model dari tautan Hugging Face di atas.

Siapkan Google Drive: Buka akun Google Drive Anda.

Buat Struktur Direktori: Buatlah folder secara berurutan di dalam Google Drive Anda agar path di dalam notebook terbaca oleh sistem. Buat struktur persis seperti ini:
MyDrive/Evaluasi_ODOL/YOLOv8_Hasil/weights/
(Catatan: Anda bisa menyesuaikan nama folder di dalam Google Drive Anda sesuai dengan path yang tertera pada Cell eksekusi di masing-masing notebook).

Unggah Model: Masukkan file best_yolo.pt, best_ssd.pth, dan best_frcnn.pth ke dalam folder yang baru saja dibuat.

Eksekusi: Buka file .ipynb di Google Colab, jalankan Cell Mount Google Drive, lalu Anda dapat langsung menuju Cell Evaluasi atau Prediksi.

**Dikembangkan oleh Indah Lestari untuk pengembangan sistem cerdas pemantauan infrastruktur jalan raya.**
