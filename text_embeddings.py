sentences = [
    "The Great Wall of China stretches over 21,000 kilometers.",
    "The Amazon rainforest produces about 20% of the world’s oxygen.",
    "The Eiffel Tower was completed in 1889 for the Paris Exposition.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "The human brain contains approximately 86 billion neurons.",
    "Global oil prices rose sharply after supply disruptions in the Middle East.",
    "India’s Chandrayaan-3 mission successfully landed near the lunar south pole.",
    "The United Nations warned about rising sea levels threatening coastal cities.",
    "The stock market rallied after positive quarterly earnings reports.",
    "Scientists discovered microplastics in human blood samples.",
    "Python remains one of the most popular programming languages in data science.",
    "Cloud computing enables businesses to scale resources on demand.",
    "Artificial intelligence is transforming healthcare diagnostics.",
    "Blockchain technology underpins cryptocurrencies like Bitcoin and Ethereum.",
    "Cybersecurity threats are increasing with the rise of remote work.",
    "Yoga originated in ancient India and promotes physical and mental well-being.",
    "The Oscars are awarded annually to honor achievements in film.",
    "Cricket is considered a national passion in India.",
    "Social media platforms influence consumer behavior worldwide.",
    "Climate change activism has gained momentum among younger generations."
]

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print("Embeddings shape:", embeddings.shape)
from sklearn.metrics.pairwise import cosine_similarity

# Create all possible sentence pairs
similarity_matrix = cosine_similarity(embeddings)

# Display similarity scores for a few examples
for i in range(5):
    for j in range(i + 1, 5):
        print(f"Similarity between:\n  '{sentences[i]}'\n  and\n  '{sentences[j]}'\n  → {similarity_matrix[i][j]:.3f}\n")

# Save results
np.savetxt("similarity_scores.csv", similarity_matrix, delimiter=",")
import heapq

pairs = []
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = similarity_matrix[i][j]
        pairs.append((score, i, j))

# Get top 5 pairs by similarity score
top_pairs = heapq.nlargest(5, pairs, key=lambda x: x[0])

print("Top 5 most similar sentence pairs:\n")
for score, i, j in top_pairs:
    print(f"Pair: ({i}, {j})")
    print(f"Sentence 1: {sentences[i]}")
    print(f"Sentence 2: {sentences[j]}")
    print(f"Similarity Score: {score:.3f}\n")
    import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix, xticklabels=sentences, yticklabels=sentences,
            cmap="YlGnBu", annot=False, cbar=True)

plt.title("Sentence Similarity Heatmap", fontsize=16)
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
from sklearn.cluster import KMeans

num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(embeddings)

print("\nSentence Clusters:\n")
for cluster_id in range(num_clusters):
    print(f"Cluster {cluster_id + 1}:")
    for i, label in enumerate(labels):
        if label == cluster_id:
            print(f" - {sentences[i]}")
    print()
from collections import defaultdict

cluster_summary = defaultdict(list)
for i, label in enumerate(labels):
    cluster_summary[label].append(sentences[i])

print("\nCluster Insights:\n")
for cluster_id, items in cluster_summary.items():
    print(f"Cluster {cluster_id + 1} ({len(items)} sentences):")
    for s in items:
        print(f" - {s}")
    print()
# Save embeddings with sentences
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.insert(0, "Sentence", sentences)
embeddings_df.to_csv("embeddings.csv", index=False)

# Save 10 selected pairs (example: first 10 pairs from 'pairs' list)
selected_pairs = pairs[:10]
selected_df = pd.DataFrame([
    {"Sentence1": sentences[i], "Sentence2": sentences[j], "Similarity": score}
    for score, i, j in selected_pairs
])
selected_df.to_csv("selected_pairs_similarity.csv", index=False)

# Save top 5 most similar pairs
top5_df = pd.DataFrame([
    {"Sentence1": sentences[i], "Sentence2": sentences[j], "Similarity": score}
    for score, i, j in top_pairs
])
top5_df.to_csv("top5_pairs_similarity.csv", index=False)
