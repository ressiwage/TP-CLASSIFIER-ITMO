# parse
import time as t
import re

def load_dataset(path):
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

data = load_dataset("../questions.txt")
print(len(data), data[0])

# embed
s = t.time()
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("cointegrated/rubert-tiny2")

questions = [d["question"] for d in data]
embeddings = model.encode(questions, normalize_embeddings=True)  # (N, dim)
print(f'[LOG] время на инициализацию: {t.time()-s:.3f}s')
s=t.time()
# search
def search(query, top_k=3):
    q_emb = model.encode([query], normalize_embeddings=True)[0]
    sims = embeddings @ q_emb  # т.к. векторы нормированы, dot = cosine
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [(data[i]["question"], data[i]["answer"], sims[i]) for i in top_idx]

q_1 = "подскажите как поменять пароль от аккаунта"
q_2 = "лее ежжий балят я пароль от аккаунта забыл"

print(q_1)
results = search(q_1)
for q, a, score in results:
    print(f"{score:.3f} | {q}")

print(q_2)
results = search(q_2)
for q, a, score in results:
    print(f"{score:.3f} | {q}")
print(f'[LOG] время на 2 вопроса: {t.time()-s:.3f}s')
# search w THRESHOLD

THRESHOLD = 0.75  # калибруется на валидации ниже

def route(query):
    top_q, top_a, score = search(query, top_k=1)[0]
    if score >= THRESHOLD:
        return {"action": "auto_answer", "answer": top_a, "score": float(score)}
    else:
        return {"action": "escalate_low_confidence", "score": float(score)}

