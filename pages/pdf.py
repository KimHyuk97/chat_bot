import os
import streamlit as st

from langchain_core.messages import ChatMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_teddynote import logging

logging.langsmith("[PROJECT] PDF-RAG_STREAMLIT")

from main import load_prompt

if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# 제목 표시
st.title("PDF 기반 QA")

# 처음 1번만 실행위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("초기화")
    uploaded_file = st.file_uploader("파일 업로드", type="pdf")
    selected_model = st.selectbox("LLM 선택", ["gpt-4o-mini", "gpt-4o", "gpt-4-trubo"], index=0)

# 파일이 업로드 되었을 때 캐시 저장
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다.....")
def embed_file(file):
    # 업로드할 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever


# chain 생성
def create_chain(retriever, model_name="gpt-4o-mini"):

    # 프롬프트 생성(Create Prompt)
    prompt = load_prompt("prompts/pdf-rag.yaml")

    # 언어모델(LLM) 생성
    llm = ChatOpenAI(model_name=model_name, temperature=0.1)

    # 체인(Chain) 생성
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return chain

# 파일이 업로드 되었을 때
if uploaded_file:
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, selected_model)
    st.session_state["chain"] = chain

# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

# 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 이전 내용 초기화
if clear_btn:
    st.session_state["messages"] = []

# 이전 내용 출력
print_messages()

# 사용자 입력 폼
user_input = st.chat_input("궁금한 내용을 물어보세요!")\

# 경고 메시지를 띄우기 위함
warning_msg = st.empty()

# 만약 사용자가 입력이 들어오면
if user_input:

    # 체인 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 웹에 노출
        st.chat_message("user").write(user_input)

        # 스트리밍 시작
        response = chain.stream(user_input)

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
    else:
        warning_msg.error("파일을 업로드 해주세요.")
