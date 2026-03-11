import os
import requests, pathlib
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()

#모델 정의
model_name = os.getenv("MODEL_NAME")
model = init_chat_model(model_name)


#SQL-lite data 가져오기

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"{local_path} already exists, skipping download.")
else:
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"Downloaded {local_path}")
    else:
        print("Failed")


# SQL database wrapper: 사용하기 쉬운 인터페이스라고 생각하면 된다.

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(f"Dialect: {db.dialect}")
# print(f"Available tables: {db.get_usable_table_names()}")
# print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')

toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# for tool in tools:
#     print(f"{tool.name}: {tool.description}")

#Prompt
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)


'''
agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
)

question = "Which genre on average has the longest tracks?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
'''

#Human in the loop review : 사람의 review을 받기 위해 middleware을 통해 특정 작업을 수행하기 전에 agent을 정지시키는 것

'''
SQLDatabaseToolkit에서 sql_db_query tool을 수행한 후, 사람의 검증을 받을 수 있도록 하기
'''


agent = create_agent(
    model_name,
    tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=InMemorySaver(), #멈추고 다시 진행하기 위해서 이전 대화 기록이 필요하기에 memory 생성
)


question = "Which genre on average has the longest tracks?"
config = {"configurable": {"thread_id": "1"}}

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    config,
    stream_mode="values",
):
    if "__interrupt__" in step:
        print("Interrupted")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])

        #사용자 입력 받기
        user_choice = input("\n이 작업을 실행하시겠습니까? (yes/no): ").strip().lower()

        if user_choice.lower() == 'yes':
            decision_type = "approve"

        else:
            decision_type = "reject"
        
        print(f"유저의 결정은 {decision_type}입니다.")

        for next_step in agent.stream(
            Command(resume={"decisions": [{"type": decision_type}]}),
            config,
            stream_mode="values",
        ):
            if "messages" in next_step:
                next_step["messages"][-1].pretty_print()
        break

    elif "messages" in step:
        step["messages"][-1].pretty_print()
    else:
        pass
