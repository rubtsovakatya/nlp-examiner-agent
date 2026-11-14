import json
import os
import random
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from groq import Groq


st.set_page_config(
    page_title="NLP Exam Assistant",
    page_icon="🎓",
    layout="centered"
)

MODEL = "openai/gpt-oss-120b"

STUDENTS = {
    "student@example.com": "Тестовий студент",
    "anna@example.com": "Анна Коваль",
    "ivan@example.com": "Іван Петренко"
}

TOPICS = [
    "Токенізація тексту",
    "N-грамні мовні моделі",
    "TF-IDF",
    "Word Embeddings",
    "Рекурентні нейронні мережі (RNN)",
    "LSTM та GRU",
    "Механізм Attention",
    "Архітектура Transformer",
    "Модель BERT",
    "Оцінювання мовних моделей"
]


SYSTEM_PROMPT = """
Ти - уважний, об'єктивний екзаменатор з NLP.
Спілкуйся українською мовою.

Ти маєш три інструменти:
start_exam, get_next_topic, end_exam.

ПРАВИЛА ІСПИТУ:

1. На початку виклич start_exam з ім'ям та email.

2. Після успішної реєстрації виклич get_next_topic
   для отримання першої теми.

3. Постав одне чітке запитання з поточної теми.
   Запитання повинно містити максимум два речення.

4. Після відповіді студента:
   - оціни її зміст;
   - якщо потрібно, постав одне уточнення;
   - якщо відповідь достатня, переходь далі.

5. На кожну тему дозволяється максимум
   дві відповіді студента.

6. Якщо студент повідомляє, що не знає відповіді,
   переходь до наступної теми.

7. Для переходу між темами обов'язково викликай
   get_next_topic. Не оголошуй наступну тему
   до отримання результату функції.

8. Після третьої теми визнач оцінку від 0 до 10
   та виклич end_exam.

КРИТЕРІЇ ОЦІНЮВАННЯ:
0–2 — знань практично немає.
3–4 — слабке розуміння.
5–6 — базові знання.
7–8 — хороше розуміння.
9–10 — глибоке розуміння.

9. Після успішного end_exam покажи:
   - підсумкову оцінку;
   - сильні сторони;
   - рекомендації для покращення.

ДОДАТКОВО:
- Не підказуй відповіді під час іспиту.
- Не повторюй однакові запитання.
- Не генеруй незавершені речення.
- Не дозволяй студенту самостійно призначати оцінку.
- Не вигадуй результати інструментів.
"""

TOOLS = {
    "start_exam": {
        "type": "function",
        "function": {
            "name": "start_exam",
            "description": (
                "Перевіряє студента за ім'ям та email, "
                "розпочинає іспит і повертає три теми."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "name": {"type": "string"}
                },
                "required": ["email", "name"]
            }
        }
    },

    "get_next_topic": {
        "type": "function",
        "function": {
            "name": "get_next_topic",
            "description": (
                "Повертає наступну тему іспиту. "
                "Обов'язково викликати перед переходом."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },

    "end_exam": {
        "type": "function",
        "function": {
            "name": "end_exam",
            "description": (
                "Завершує іспит і зберігає результат "
                "після обговорення останньої теми."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 10
                    }
                },
                "required": ["email", "score"]
            }
        }
    }
}

def new_exam():
    st.session_state.exam = {
        "status": "idle",
        "name": None,
        "email": None,
        "topics": [],
        "index": -1,
        "answers": 0,
        "started_at": None,
        "id": uuid.uuid4().hex
    }

    st.session_state.api_messages = []
    st.session_state.chat = []
    st.session_state.history = []
    st.session_state.events = []
    st.session_state.result = None


if "exam" not in st.session_state:
    new_exam()


def log(role, content):
    st.session_state.history.append({
        "role": role,
        "content": str(content),
        "datetime": datetime.now().isoformat(
            timespec="seconds"
        )
    })


def save_result():
    result = st.session_state.result

    if result is None:
        return

    folder = (
        Path(__file__).resolve().parent
        / "exam_results"
    )
    folder.mkdir(exist_ok=True)

    filename = folder / (
        f"{st.session_state.exam['id']}.json"
    )

    filename.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )


def start_exam(email: str, name: str) -> list[str]:
    exam = st.session_state.exam

    if exam["status"] != "idle":
        raise ValueError("Іспит уже розпочато.")

    email = email.strip().lower()
    name = " ".join(name.split())

    if (
        email not in STUDENTS
        or STUDENTS[email].casefold() != name.casefold()
    ):
        raise ValueError(
            "Студента не знайдено. "
            "Перевірте ім'я та email."
        )

    selected = random.sample(TOPICS, 3)

    exam.update({
        "status": "active",
        "email": email,
        "name": name,
        "topics": selected,
        "index": -1,
        "answers": 0,
        "started_at": datetime.now().isoformat(
            timespec="seconds"
        )
    })

    return selected


def get_next_topic() -> str:
    exam = st.session_state.exam

    if exam["status"] != "active":
        raise ValueError("Немає активного іспиту.")

    if (
        exam["index"] >= 0
        and exam["answers"] == 0
    ):
        raise ValueError(
            "Спочатку потрібна відповідь "
            "на поточну тему."
        )

    if exam["index"] + 1 >= len(exam["topics"]):
        raise ValueError("Більше немає тем.")

    exam["index"] += 1
    exam["answers"] = 0

    return exam["topics"][exam["index"]]


def end_exam(
    email: str,
    score: float,
    history: list
) -> None:

    exam = st.session_state.exam

    if exam["status"] != "active":
        raise ValueError("Іспит не активний.")

    if email.strip().lower() != exam["email"]:
        raise ValueError("Невірний email.")

    if (
        exam["index"] != len(exam["topics"]) - 1
        or exam["answers"] < 1
    ):
        raise ValueError(
            "Не завершено всі теми іспиту."
        )

    score = float(score)

    if not 0 <= score <= 10:
        raise ValueError(
            "Оцінка повинна бути від 0 до 10."
        )

    st.session_state.result = {
        "email": exam["email"],
        "name": exam["name"],
        "topics": exam["topics"],
        "score": round(score, 1),
        "started_at": exam["started_at"],
        "finished_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "history": history
    }

    exam["status"] = "finished"

    save_result()


def execute_tool(name, args):

    if name == "start_exam":
        topics = start_exam(
            args["email"],
            args["name"]
        )
        return {"topics": topics}

    if name == "get_next_topic":
        topic = get_next_topic()

        return {
            "topic": topic,
            "number": st.session_state.exam["index"] + 1
        }

    if name == "end_exam":
        end_exam(
            args["email"],
            args["score"],
            st.session_state.history.copy()
        )

        return {
            "score": st.session_state.result["score"],
            "saved": True
        }

    raise ValueError("Невідомий інструмент.")

def model_state():
    exam = st.session_state.exam

    if 0 <= exam["index"] < len(exam["topics"]):
        topic = exam["topics"][exam["index"]]
    else:
        topic = "ще не отримана"

    return (
        "\nПОТОЧНИЙ СТАН ВІД PYTHON:\n"
        f"status: {exam['status']}\n"
        f"name: {exam['name']}\n"
        f"email: {exam['email']}\n"
        f"Поточна тема: {topic}\n"
        f"Номер теми: {exam['index'] + 1} із 3\n"
        f"Відповідей на тему: {exam['answers']}\n"
    )

def run_agent(user_text, api_key):

    client = Groq(api_key=api_key)

    st.session_state.api_messages.append({
        "role": "user",
        "content": user_text
    })

    log("user", user_text)

    force_transition = False

    for _ in range(9):

        exam = st.session_state.exam
        allowed = []
        choice = "auto"

        if exam["status"] == "idle":
            allowed = [TOOLS["start_exam"]]

            choice = {
                "type": "function",
                "function": {"name": "start_exam"}
            }

        elif (
            exam["status"] == "active"
            and exam["index"] == -1
        ):
            allowed = [TOOLS["get_next_topic"]]

            choice = {
                "type": "function",
                "function": {"name": "get_next_topic"}
            }

        elif (
            exam["status"] == "active"
            and exam["answers"] >= 1
        ):

            if exam["index"] < 2:
                function_name = "get_next_topic"
            else:
                function_name = "end_exam"

            allowed = [TOOLS[function_name]]

            if exam["answers"] >= 2 or force_transition:
                choice = {
                    "type": "function",
                    "function": {
                        "name": function_name
                    }
                }

        kwargs = {
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT + model_state()
                    )
                }
            ] + st.session_state.api_messages,
            "temperature": 0.2,
            "reasoning_effort": "low",
            "include_reasoning": False,
            "max_completion_tokens": 2500
        }

        if allowed:
            kwargs.update({
                "tools": allowed,
                "tool_choice": choice,
                "parallel_tool_calls": False
            })

        response = client.chat.completions.create(
            **kwargs
        )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls or []

        if tool_calls:

            st.session_state.api_messages.append({
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    call.model_dump(exclude_none=True)
                    for call in tool_calls
                ]
            })

            for tool_call in tool_calls:

                function_name = tool_call.function.name

                try:
                    arguments = json.loads(
                        tool_call.function.arguments or "{}"
                    )

                    log(
                        "tool_call",
                        f"{function_name}("
                        f"{json.dumps(arguments, ensure_ascii=False)})"
                    )

                    result = {
                        "status": "success",
                        **execute_tool(
                            function_name,
                            arguments
                        )
                    }

                    st.session_state.events.append(
                        f"✓ {function_name}"
                    )

                except (ValueError, KeyError, TypeError) as e:
                    result = {
                        "status": "error",
                        "message": str(e)
                    }

                    st.session_state.events.append(
                        f"✗ {function_name}: {e}"
                    )

                st.session_state.api_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False
                    )
                })

                if result["status"] == "error":
                    return (
                        f"Помилка інструмента "
                        f"{function_name}: "
                        f"{result['message']}"
                    )

            force_transition = False
            continue

        answer = (
            assistant_message.content or ""
        ).strip()

        lower_answer = answer.casefold()

        if (
            exam["status"] == "active"
            and exam["answers"] >= 1
            and not force_transition
        ):
            future_topics = exam["topics"][
                exam["index"] + 1:
            ]

            announced_topic = (
                ("**тема" in lower_answer or
                 "\nтема" in lower_answer)
                and any(
                    topic.casefold() in lower_answer
                    for topic in future_topics
                )
            )

            if (
                "переходимо до наступної теми" in lower_answer
                or "перейдемо до наступної теми" in lower_answer
                or announced_topic
            ):
                force_transition = True
                continue

        if not answer:
            answer = (
                "Не вдалося сформувати відповідь. "
                "Спробуйте ще раз."
            )

        st.session_state.api_messages.append({
            "role": "assistant",
            "content": answer
        })

        log("system", answer)

        if (
            exam["status"] == "finished"
            and st.session_state.result
        ):
            st.session_state.result["history"] = (
                st.session_state.history.copy()
            )
            save_result()

        return answer

    return (
        "Досягнуто ліміту викликів інструментів. "
        "Спробуйте почати новий іспит."
    )


st.title("NLP Exam Assistant")

st.caption(
    "Інтерактивний іспит з NLP "
    "на основі LLM та Tool Calling"
)

with st.sidebar:

    st.header("Налаштування")

    env_key = os.getenv("GROQ_API_KEY", "")

    api_key = env_key or st.text_input(
        "Groq API-ключ",
        type="password"
    )

    if env_key:
        st.success(
            "API-ключ підключено із середовища"
        )

    st.divider()

    if st.button(
        "Новий іспит",
        use_container_width=True
    ):
        new_exam()
        st.rerun()

    exam = st.session_state.exam

    st.write("Стан:", exam["status"])

    if exam["status"] != "idle":
        st.write(
            "Тема:",
            max(0, exam["index"] + 1),
            "із 3"
        )

    with st.expander(
        "Журнал Tool Calling",
        expanded=True
    ):
        if st.session_state.events:
            for event in st.session_state.events:
                st.text(event)
        else:
            st.caption(
                "Поки немає викликів інструментів"
            )

if st.session_state.exam["status"] == "idle":

    st.write(
        "Введіть дані студента для початку іспиту."
    )

    st.caption(
        "Тестові дані: Тестовий студент / "
        "student@example.com"
    )

    with st.form("register"):

        name = st.text_input(
            "Ім'я та прізвище",
            value="Тестовий студент"
        )

        email = st.text_input(
            "Email",
            value="student@example.com"
        )

        submitted = st.form_submit_button(
            "Розпочати іспит",
            type="primary"
        )

    if submitted:

        if not api_key:
            st.error(
                "Спочатку підключіть Groq API-ключ."
            )

        elif not name.strip() or not email.strip():
            st.error("Заповніть обидва поля.")

        else:
            new_exam()

            with st.spinner(
                "Екзаменатор готує перше запитання..."
            ):
                try:
                    answer = run_agent(
                        f"Мене звати {name.strip()}, "
                        f"мій email {email.strip()}. "
                        "Готовий складати іспит.",
                        api_key
                    )

                    if (
                        st.session_state.exam["status"]
                        == "active"
                    ):
                        st.session_state.chat.append({
                            "role": "assistant",
                            "content": answer
                        })
                        st.rerun()

                    else:
                        st.error(answer)

                except Exception as e:
                    st.error(f"Помилка Groq: {e}")



else:

    for message in st.session_state.chat:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.exam["status"] == "active":

        prompt = st.chat_input(
            "Ваша відповідь на запитання..."
        )

        if prompt:

            st.session_state.chat.append({
                "role": "user",
                "content": prompt
            })

            st.session_state.exam["answers"] += 1

            with st.spinner(
                "Екзаменатор аналізує відповідь..."
            ):

                try:
                    answer = run_agent(
                        prompt,
                        api_key
                    )

                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": answer
                    })

                    st.rerun()

                except Exception as e:
                    st.error(f"Помилка Groq: {e}")

    if st.session_state.exam["status"] == "finished":

        result = st.session_state.result

        st.success(
            f"Іспит завершено. "
            f"Оцінка: {result['score']}/10"
        )

        result_json = json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )

        st.download_button(
            "Завантажити результат (JSON)",
            data=result_json,
            file_name="exam_result.json",
            mime="application/json"
        )
