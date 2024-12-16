import glob
import yaml
import streamlit as st

from dotenv import load_dotenv
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import loading
from langchain_core.prompts.base import BasePromptTemplate

load_dotenv()

def load_prompt(file_path, encoding="utf8") -> BasePromptTemplate:
    with open(file_path, "r", encoding=encoding) as f:
        config = yaml.safe_load(f)

    return loading.load_prompt_from_config(config)


# 제목 표시
st.title("Chat GPT")

# 처음 1번만 실행위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("초기화")

    prompt_files = glob.glob("prompts/*.yaml")
    selected_prompt = st.selectbox("프롬프트를 선택해주세요.", prompt_files, index=0)
    task_input = st.text_input("TASK INPUT", "")


# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# chain 생성
def create_chain(prompt_filepath, task: ""):

    prompt = load_prompt(prompt_filepath)
    if task:
        # partial 을 통해서 변수 추가
        prompt = prompt.partial(task=task)

    llm = ChatOpenAI(
        model_name="gpt-4o-mini", temperature=0
    )
    out_parser = StrOutputParser()

    chain = prompt | llm | out_parser
    return chain

# 이전 내용 초기화
if clear_btn:
    st.session_state["messages"] = []

# 이전 내용 출력
print_messages()

# 사용자 입력 폼
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 만약 사용자가 입력이 들어오면
if user_input:
    # 웹에 노출
    st.chat_message("user").write(user_input)

    # 체인 생성
    chain = create_chain(selected_prompt, task=task_input)

    # 스트리밍 시작
    response = chain.stream({"question": user_input})

    # 스트리밍 출력
    with st.chat_message("assistant"):
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.write(ai_answer)

    # 대화기록 저장
    add_message("user", user_input)
    add_message("assistant", ai_answer)
