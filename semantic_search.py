from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import time
from typing import List
from langchain_core.runnables import chain


load_dotenv()

'''
document loading
'''
file_path = "cats.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

'''
PyPDFLoader는 PDF 별로 나오는 document object을 하나씩 load한다. 이를 통해 각 page의 string 내용과 metadata (file name, page #) 등을 알 수 있다.
'''

# print(f"{docs[0].page_content[:200]}\n")
# print(docs[0].metadata)

"""
Splitting: Page 단위로 retrieve한다고 했을 떄, 그 내용이 너무 방대해서 query와 직접적인 대응이 어려울 수 있다. 따라서, splitting을 거쳐서 가져온 page을 내용을 잃지 않으면섣호 잘게 쪼개서 
더 밀도있는 내용만 retrieve할 수 있도록 하는 것이다.
"""

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200, chunk_overlap=20, add_start_index = True
)

all_splits = text_splitter.split_documents(docs)

print(len(all_splits))

"""
Embedding: Split된 document들을 벡터화해야지 query vector와 유사도를 비교하면서 최적의 document을 찾을 수 있다.
"""

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

"""
Vector store: 사용한 embedding model을 바탕으로 document object들을 저장하고 similarity metrics을 기반으로 query와 비교까지 도와준다.
"""
 
#InMemoryVectorStore는 프로그램 종료되면 안에 있는 정보를 잃음
vector_store = InMemoryVectorStore(embeddings) #초기화를 document을 embedding할 떄 사용한 model로 해야 한다.

i=1
for split in all_splits:
    print(f"{i}-th done")
    i+=1
    vector_store.add_documents([split])
    time.sleep(0.01)

#String query
results = vector_store.similarity_search(
    "고양이의 눈동자 모양은 어때?"
)

print(results[0]) #results에 유사도 점수를 기반으로 이미 정렬되어있음

# 유사도 점수도 반환할 수 있음
# results = vector_store.similarity_search_with_score("고양이의 눈동자 모양은 어때?")
# doc, score = results[0]

"""
Retrievers: VectorStore 클래스는 runnable object가 아니기 때문에 invoke나 batch 함수들이 구현되어있지 않다. 반면, Langchain Retriever들은 Runnable이기에 이 함수들을 호출할 수 있다.
runnable retriever을 만드는 방법은 vector store으로부터 직접 만들 수 있고 아니면 @chain을 통해 직접 구성할 수 있다.
"""

#chain annotation을 통해 runnable object을 만듦
@chain
def retriever(query:str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

#runnable으로 바뀌어서 VectorStore에서 runnable의 invoke 함수를 사용할 수 있게 된다
retriever.invoke("고양이의 눈동자 모양은 어때")

#VectorStore으로부터 직접 runnable retriever (VectorStoreRetriever)으로도 만들 수 있다.
# search_type, search_kwargs parameter으로 VectorStore의 어떤 함수를 사용하여 retriever을 만들 것인지, 얼마나 많은 문서를 반환할 것인지 정할 수 있다.
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":1},
)
retriever.invoke("고양이의 눈동자 모양은 어때")

# https://docs.langchain.com/oss/python/langchain/knowledge-base#google-gemini









