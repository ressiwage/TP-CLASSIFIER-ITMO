from typing import Any
from sentence_transformers import SentenceTransformer
import numpy as np
import time as t
import re

class Searcher:
    def __init__(self, data_path="../questions.txt", model_name="cointegrated/rubert-tiny2", THRESHOLD=0.7):
        self.data = self._load_dataset(data_path)
        self.model = SentenceTransformer(model_name)
        questions = [d["question"] for d in self.data]
        self.embeddings = self.model.encode(questions, normalize_embeddings=True)  # (N, dim)
        self.threshold=THRESHOLD

    def _load_dataset(self, path):
        """
        Формат строки: "T" "вопрос" "ответ"
        """
        pattern = re.compile(r'"([TN])"\s+"([^"]+)"\s+"([^"]+)"')
        data = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = pattern.match(line.strip())
                if m:
                    label, question, answer = m.groups()
                    data.append({"label": label, "question": question, "answer": answer})
        return data
    
    def __call__(self, query, top_k=3):
        '''выдаёт: [(question:str, answer:str, confidence:str),...], confidence>THRESHOLD'''
        q_emb = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ q_emb  # т.к. векторы нормированы, dot = cosine
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [(self.data[i]["question"], self.data[i]["answer"], sims[i]) for i in top_idx if  sims[i]>self.threshold]


if __name__=='__main__':
    S = Searcher()
    print(S("забыл пароль"))