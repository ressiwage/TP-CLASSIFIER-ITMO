from typing import Any
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re

class Classifier:
    def __init__(self, model='cointegrated/rubert-tiny2', data_path="../criticity.txt"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)

        texts, labels = self._load_dataset(data_path)
        embeddings = np.array([self._embed(t) for t in texts])

        self.centroids = self._build_centroids(embeddings, labels)


    def _embed(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = self.model(**inputs)
        # CLS-токен как представление предложения (стандартно для rubert-tiny2)
        vec = out.last_hidden_state[:, 0, :].squeeze().numpy()
        return vec / np.linalg.norm(vec)  # нормализуем сразу, чтобы cosine = dot product

    def _load_dataset(self, path: str):
        texts, labels = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                # формат: "0" "текст тикета"
                m = re.match(r'^"(\d)"\s+"(.+)"$', line.strip())
                if m:
                    labels.append(int(m.group(1)))
                    texts.append(m.group(2))
        return texts, labels

    def _build_centroids(self, embeddings: np.ndarray, labels: list[int]) -> dict[int, np.ndarray]:
        centroids = {}
        for cat in set(labels):
            cat_vecs = embeddings[[i for i, l in enumerate(labels) if l == cat]]
            centroid = cat_vecs.mean(axis=0)
            centroids[cat] = centroid / np.linalg.norm(centroid)
        return centroids
    
    def __call__(self, text: str, margin_threshold: float = 0.05):
        '''возвращает dict из category:int, score:float, confidence:'low'|high'''
        v = self._embed(text)
        sims = {cat: float(np.dot(v, c)) for cat, c in self.centroids.items()}
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
if __name__=='__main__':
    C = Classifier()

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
        result = C(q)
        print(result, f'ожидаемая критичность: {c}')  # {'category': 0, 'score': 0.71, 'confidence': 'high', 'escalate': False}
