from fastapi import Form
# main.py
from fastapi import FastAPI
from pydantic import BaseModel

from ml.classifier.main import Classifier
from ml.searcher.main import Searcher

app = FastAPI()

T = 3  # порог критичности для эскалации

# грузим модели один раз при старте приложения, не на каждый запрос
classifier = Classifier(data_path='ml/criticity.txt')
searcher_broad = Searcher(THRESHOLD=0.5,  data_path='ml/questions.txt')   # для R: широкий поиск, K=5
searcher_strict = Searcher(THRESHOLD=0.9, data_path='ml/questions.txt')  # для R1: строгий поиск для reply_static


class TicketRequest(BaseModel):
    text: str


def send_to_operator(reason: str = "") -> dict:
    print(f"[ESCALATE] reason={reason}")
    return {
        "action": "send_to_operator",
        "message": "Ваш запрос отправлен на рассмотрение оператору",
    }


def reply_static(answer_text: str) -> dict:
    return {"action": "reply_static", "message": answer_text}


def reply_llm(found: list, query: str) -> dict:
    context = "\n\n".join(
        f"[Похожий вопрос]: {q}\n[Ответ]: {a}\n[Similarity]: {sim:.3f}"
        for q, a, sim in found
    )

    prompt = f"""Ты — ассистент поддержки крупного онлайн-сервиса. Отвечай вежливо, кратко и по делу, на русском языке.

Ниже приведены похожие вопросы из базы знаний и ответы на них. Используй их как контекст для ответа, но не копируй дословно, если по смыслу не подходит.

КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{query}

Сформулируй ответ пользователю на основе контекста выше. Не придумывай факты (точные суммы, даты, номера заказов, сроки), которых нет в контексте — если контекст их не содержит, предложи пользователю уточнить детали или дождаться оператора."""

    print("=== ЗАПРОС В LLM (заглушка, реальный вызов не выполняется) ===")
    print(prompt)
    print("================================================================")

    return {"action": "reply_llm", "prompt_sent_to_llm": prompt}


@app.post("/ticket")
def process_ticket(req: TicketRequest):
    text = req.text
    log = {"text": text}

    criticity = classifier(text)
    log["criticity"] = criticity

    # шаг 1: фильтр по критичности
    if criticity["category"] > T:
        result = send_to_operator(reason=f"criticity={criticity['category']} > T={T}")
        log["result"] = result
        return log

    # шаг 2: поиск похожих вопросов
    R = searcher_broad(text, top_k=5)
    R1 = searcher_strict(text, top_k=5)

    log["R_count"] = len(R)
    log["R1_count"] = len(R1)

    if len(R) == 0:
        result = send_to_operator(reason="нет похожих вопросов в базе знаний (R пуст)")
    elif len(R1) != 0:
        # R1[i] = (question, answer, similarity) -> берём answer первого элемента
        result = reply_static(R1[0][1])
    else:
        result = reply_llm(R, text)

    log["result"] = result
    return log


# main.py — добавить к существующему коду
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


@app.post("/", response_class=HTMLResponse)
def index_submit(request: Request, text: str = Form(...)):
    result = process_ticket(TicketRequest(text=text))
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "text": text})