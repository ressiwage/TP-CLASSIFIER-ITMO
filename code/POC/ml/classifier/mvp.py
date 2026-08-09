import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re

MODEL_NAME = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def embed(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs)
    # CLS-токен как представление предложения (стандартно для rubert-tiny2)
    vec = out.last_hidden_state[:, 0, :].squeeze().numpy()
    return vec / np.linalg.norm(vec)  # нормализуем сразу, чтобы cosine = dot product

def load_dataset(path: str):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            # формат: "0" "текст тикета"
            m = re.match(r'^"(\d)"\s+"(.+)"$', line.strip())
            if m:
                labels.append(int(m.group(1)))
                texts.append(m.group(2))
    return texts, labels

texts, labels = load_dataset("../criticity.txt")
embeddings = np.array([embed(t) for t in texts])


def build_centroids(embeddings: np.ndarray, labels: list[int]) -> dict[int, np.ndarray]:
    centroids = {}
    for cat in set(labels):
        cat_vecs = embeddings[[i for i, l in enumerate(labels) if l == cat]]
        centroid = cat_vecs.mean(axis=0)
        centroids[cat] = centroid / np.linalg.norm(centroid)
    return centroids

centroids = build_centroids(embeddings, labels)

def classify(text: str, centroids: dict, margin_threshold: float = 0.05):
    v = embed(text)
    sims = {cat: float(np.dot(v, c)) for cat, c in centroids.items()}
    ranked = sorted(sims.items(), key=lambda x: -x[1])
    top_cat, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    confidence = "low" if (top_score - second_score) < margin_threshold else "high"
    return {
        "category": top_cat,
        "score": top_score,
        "confidence": confidence,
        "escalate": confidence == "low" or top_cat >= 3,  # пример: 3+ = рискованная категория
    }

# пример
qs = [
    'Не могу зайти в аккаунт, забыл пароль',
    'я в вас разочарован. я отправил вам голые фото а вы даже не посмотрели',
    'здравствуйте! как подписку отменить? деньги продолжают списываться',
    'блять у меня с карты пропало дохуя денег',
    'ВЫ ОСТОЛОПЫ ЁБАННЫЕ ГДЕ МОИ БАБКИ? Я ПРОЕБАЛ МИЛЛИОН ДОЛЛАРОВ НА ВАШЕЙ ХУЙНЕ'
]
ecs = [0,1,2,3,4]

for q,c in zip(qs, ecs):
    result = classify(q, centroids)
    print(result, f'ожидаемая критичность: {c}')  # {'category': 0, 'score': 0.71, 'confidence': 'high', 'escalate': False}