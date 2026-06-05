import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv("jobs.csv")

print("=" * 60)
print("AI TECH STACK RECOMMENDER")
print("=" * 60)

print("\nEnter at least 3 skills")

user_skills = input(
    "\nSkills (comma separated): "
)

user_profile = " ".join(
    [skill.strip().lower()
     for skill in user_skills.split(",")]
)

# Combine user profile with dataset
documents = [user_profile] + data["Skills"].tolist()

# TF-IDF
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    documents
)

# Cosine Similarity
similarity_scores = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
)

scores = similarity_scores.flatten()

data["Match Score"] = scores

# Sort
recommendations = data.sort_values(
    by="Match Score",
    ascending=False
)

print("\nTop Recommendations\n")

for index, row in recommendations.head(3).iterrows():

    print(
        f"{row['Role']} "
        f"({row['Match Score']*100:.2f}%)"
    )