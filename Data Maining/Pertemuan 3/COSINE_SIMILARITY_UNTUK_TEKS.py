#pertemuan 3 COSINE SIMILARITY UNTUK TEKS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Kumpulan dokumen (Dataset Teks)
documents = [
    "data mining machine learning",
    "machine learning data mining",
    "artificial intelligence",
    "deep learning neural network"
]

# 2. Konversi teks ke dalam bentuk angka (Vektor TF-IDF)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 3. Hitung Cosine Similarity antar semua dokumen
# Hasilnya adalah matriks simetris N x N
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Cosine Similarity Matrix:")
print(similarity_matrix.round(3))

print("\nFitur (kata-kata yang terdeteksi):")
print(vectorizer.get_feature_names_out())

# 4. Mencari dokumen yang paling mirip dengan dokumen pertama (indeks 0)
# Kita gunakan argsort()[-2] karena nilai tertinggi (1.0) pasti adalah dirinya sendiri
most_similar_idx = similarity_matrix[0].argsort()[-2]
print(f"\nDokumen 0 paling mirip dengan dokumen {most_similar_idx}")
print(f"Isi Dokumen 0: {documents[0]}")
print(f"Isi Dokumen {most_similar_idx}: {documents[most_similar_idx]}")