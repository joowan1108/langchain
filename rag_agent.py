from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

"""
특정한 정보 source을 사용하여 QA을 하는 것이 RAG이다.
"""


model = ChatGoogleGenerativeAI(model = "google_genai:gemini-3-flash-preview")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
embedding_dim = len(embeddings.embed_query("c"))
index = faiss.IndexFlatL2(embedding_dim)

"""
index_to_docstore_id 는 raw vector index와 실제 document data를 mapping해주는 mapping dictionary이다.
FAISS는 숫자와 벡터만 이해할 수 있고 text는 이해하지 못한다.
따라서 document store와 vector index 간의 mapping을 도와주는 mapping dictionary가 필요한 것이다.

Step A: Query를 vector로

Step B: FAISS 검색 결과 index 42가 가장 relevant

Step C: LangChain -> index_to_docstore_id[42] -> document id

Step D: LangChain with document id -> 실제 document 가져옴
"""
vector_store = FAISS(
    embedding_function=embeddings,
    index = index,
    docstore = InMemoryDocstore(),
    index_to_docstore_id={},
)

"""
Indexing의 과정은 보통 다음과 같다.
1: document loading
2: text splitting으로 너무 긴 document들을 chunk으로 쪼갠다
3: 이 pslit들을 저장소에 저장하여 나중에 검색될 수 있도록 한다
"""

# Loading Documents

'''
Blog 안에 있는 내용으로 RAG QA을 진행할 것이라서 WebBaseLoader을 사용하였다.

urlib으로 web URL으로부터 HTML을 가져와 BeautifulSoup으로 text으로 parse해준다.
'''

bs4_strainer = bs4.SoupStrainer(class_ = ("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()

#print(docs[0].page_content[:50])


#Splitting Documents 

'''
모델이 감당 가능한 context 길이만큼 주면서 이 fragment들의 의미를 잃으면 안된다.
'''

"""
TextSplitter = Document 객체 list을 return
"""
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True,
)

all_splits = text_splitter.split_documents(docs)

document_ids = vector_store.add_documents(documents=all_splits)

#LLM에 input으로 text만 줄 수 있도록 하는 방법
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Query에 도움되는 문서들만 retrieve하는 함수"""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata} \nContent: {doc.page_content}" for doc in retrieved_docs)
    )
    return serialized, retrieved_docs


"""
Agent 초기화
"""
# tools = [retrieve_context]
# prompt = (
#     "You have access to a tool that retrieves context from a blog post. "
#     "Use the tool to help answer user queries."
# )
# agent = create_agent(model = "google_genai:gemini-3-flash-preview", tools=tools, system_prompt=prompt)

# query = (
#     "What is the standard method for Task Decomposition?\n\n"
#     "Once you get the answer, look up common extensions of that method."
# )

#Stream으로 LLM의 생각 과정을 볼 수 있음
# for event in agent.stream(
#     {"messages": [{"role": "user", "content": query}]},
#     stream_mode="values",
# ):
#     event["messages"][-1].pretty_print()


# response = agent.invoke({"messages": [{"role": "user", "content": query}]})

# 2. 결과에서 마지막 메시지(AI의 최종 답변)의 내용만 출력
# final_answer = response["messages"][-1].content

# print("-" * 50)
# print("최종 답변:")
# print(final_answer)
# print("-" * 50)

"""
Tool call은 언제든지 필요할 때만 사용할 수 있게 해주고 LLM이 query을 받을 때, query을 tool에 맞게 변환시켜서 처리하기 때문에 contextual 정보가 담긴 query가 가능한다. + Tool을 여러번 쓸 수 있다.

하지만 tool call로 search을 하면 tool에 적합한 query로 만드는 과정에서 llm api call 한번 + 최종 답변을 만드는 과정에서 api call 한 번, 총 2번이 요구된다.
또, 특정 tool이 필요한 경우에도 잘못 판단하여 tool을 사용하지 않을 수 있다.

--> 이런 문제를 해결하기 위해 tools 중에서 사용한 만한 것을 고르도록 하는 것이 아니라 tool 사용이 무조건 되도록 tool을 chain으로 만들어 two-step chain으로 만들면 된다.
Tool call을 하는 과정에서 LLM이 query 재구성을 하는 과정을 거치지 않고 raw query 그대로 항상 search을 하게 만든 뒤, 그 결과를 LLM에 들어가는 query에 붙이면 한 번의 inference call만으로 tool + LLM을 거칠 수 있게 된다. 
"""

#이 annotation이 붙은 함수는 모델이 실행되기 직전에 호출된다. State에 따라 system message가 달라지도록 한다.
@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state message"""
    last_query = request.state['messages'][-1].content
    retrieve_docs = vector_store.similarity_search(last_query)

    docs_content = '\n\n'.join(doc.page_content for doc in retrieve_docs) #search 결과를 하나의 context으로

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message

query = "What is task decomposition?"

agent = create_agent(model="google_genai:gemini-3-flash-preview", tools=[], middleware=[prompt_with_context])

result = agent.invoke({"messages": [{"role": "user", "content": query}]})

print(result["messages"][-1])
