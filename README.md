# Analisis Sentimen Ulasan Restoran dengan Granite

## Deskripsi Tujuan

Proyek ini bertujuan untuk menganalisis sentimen (positif atau negatif) dari ulasan restoran menggunakan model bahasa besar Granite dari IBM. Analisis meliputi prapemrosesan data teks, klasifikasi sentimen menggunakan model AI, visualisasi hasilnya, dan evaluasi kinerja model.

## Langkah Pengerjaan

* **Muat Data**: Memuat dataset ulasan restoran dari file TSV ke dalam Pandas DataFrame.

* **Prapemrosesan Data**: Membersihkan data teks dengan menghapus duplikat, menangani nilai yang hilang (jika ada), melakukan tokenisasi, menghapus stopword, dan melakukan lemmatization.

* **Analisis Sentimen dengan Granite**: Menggunakan model Granite yang diinisialisasi untuk memprediksi sentimen setiap ulasan.

* **Evaluasi Model**: Mengevaluasi kinerja model Granite dengan membandingkan sentimen yang diprediksi dengan label sentimen asli menggunakan metrik seperti akurasi, presisi, recall, dan F1-score.

* **Visualisasi**: Membuat visualisasi seperti grafik distribusi sentimen dan wordcloud untuk menyajikan hasil analisis.

## Hasil & Insight

* Dataset berisi 1000 ulasan restoran yang berhasil dimuat dan diprapemrosesan.

* Tidak ditemukan duplikat atau nilai yang hilang dalam data ulasan.

* Model Granite berhasil mengklasifikasikan sentimen dengan akurasi sekitar 90%, menunjukkan kemampuannya dalam tugas ini.

* Visualisasi distribusi sentimen menunjukkan keseimbangan antara ulasan positif dan negatif.

* Wordcloud menyoroti kata-kata yang paling sering muncul, dengan fokus pada "food", "service", dan "place", menunjukkan aspek-aspek utama yang diperhatikan pelanggan.

* Insight utama termasuk pentingnya fokus pada kualitas makanan dan layanan, perlunya pemantauan ulasan negatif, dan potensi penggunaan wordcloud untuk mengidentifikasi isu spesifik.

## Cara Menjalankan Kode

* Pastikan Anda memiliki akses ke Google Colab.

* Unggah dataset Restaurant_Reviews.tsv ke lingkungan Colab Anda.

* Instal pustaka yang diperlukan (replicate, pandas, langchain_community, nltk, wordcloud, matplotlib, seaborn) menggunakan pip.

* Setel token API Replicate Anda sebagai variabel lingkungan atau gunakan Secrets Manager di Colab.

* Jalankan setiap sel kode secara berurutan di notebook Colab.

* Pastikan untuk mengganti your_token_here dengan token Replicate API Anda yang sebenarnya.

## Link Collab 

* https://colab.research.google.com/drive/1bRhZcuiL7IyXCEA49PSuXcklSXB7oHGV?usp=sharing


