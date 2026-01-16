# resumeranker
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


def load_resumes(folder_path):
    resumes = []
    filenames = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                resumes.append(clean_text(f.read()))
                filenames.append(file)

    return resumes, filenames


def rank_resumes(job_desc, resumes, filenames):
    documents = [job_desc] + resumes

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    ranked = sorted(
        zip(filenames, similarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


if __name__ == "__main__":
    with open("job_description.txt", "r", encoding="utf-8") as jd_file:
        job_description = clean_text(jd_file.read())

    resumes, filenames = load_resumes("resumes")

    results = rank_resumes(job_description, resumes, filenames)

    print("\nResume Ranking Based on Job Description:\n")
    for rank, (name, score) in enumerate(results, start=1):
        print(f"{rank}. {name}  -->  Match Score: {round(score * 100, 2)}%")
