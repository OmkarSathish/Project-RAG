from openai import AsyncOpenAI
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from typing_extensions import TypedDict
from typing import List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
import asyncio

load_dotenv()

llm = AsyncOpenAI()


class State(TypedDict):
    messages: List[dict]
    context: str
    relevance_score: int | None
    run_count: int


embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="nodejs",
    embedding=embedding_model,
    force_disable_check_same_thread=True,
)


async def search_node(state: State):
    user_query = state["messages"][-1]["content"]
    search_results = await vector_db.asimilarity_search(query=user_query, k=10)
    context = "\n\n".join(
        [
            f"Page Content: {result.page_content}\n Page Number: {result.metadata['page_label']}\nFile Location:{result.metadata['source']}"
            for result in search_results
        ]
    )
    state["context"] = context
    return state


async def llm_node(state: State):
    SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant who answers user queries based on the available context retrieved from a PDF file
        along with page contents and page number.
        You should only answer the user based on the following context and navigate the user to open the right page number
        to know more.
        Context:
        {state["context"]}
    """
    state["messages"].append({"role": "system", "content": SYSTEM_PROMPT})
    llm_response = await llm.chat.completions.create(
        model="gpt-4.1-mini", messages=state["messages"]
    )
    system_msg = {"role": "system", "content": llm_response.choices[0].message.content}
    state["messages"].append(system_msg)
    return state


async def validate_result(state: State):
    VALIDATE_RESULT_PROMPT = f"""
    You are an AI Assistant which helps in analyzing the user query and the respective llm response.
    Your goal is to check how relevant the answer is and grade it from 0-10. A grade of 0 signifies
    no relevance between the user query and llm response while 10 signifies total/complete relevance
    between the user query and llm response.

    Your response should only be a single digit and that single digit should be the rating.

    CONTEXT:

    user query: {state["messages"][-2]["content"]}
    llm response: {state["messages"][-1]["content"]}
    """
    llm_response = await llm.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": VALIDATE_RESULT_PROMPT}],
    )
    score_str = llm_response.choices[0].message.content.strip()
    try:
        score = int(score_str)
    except Exception:
        score = 0
    state["relevance_score"] = score
    state["run_count"] += 1
    return state


def validate_conditional_edge(state: State):
    score = state.get("relevance_score", 0)
    run_count = state.get("run_count", 0)
    if run_count >= 3:
        return END
    if score < 5:
        return "search"
    return "llm"


def create_memory_graph(graph, checkpointer):
    memory_graph = graph.compile(checkpointer=checkpointer)
    return memory_graph


graph = StateGraph(State)
graph.add_node("search", search_node)
graph.add_node("llm", llm_node)
graph.add_node("validate", validate_result)
graph.add_edge("search", "llm")
graph.add_edge("llm", "validate")
graph.add_conditional_edges("validate", validate_conditional_edge)
graph.add_edge("llm", END)
graph.set_entry_point("search")

URI = "mongodb://:27017"
config = {"configurable": {"thread_id": "1"}}


async def main():
    """Main async function to run the RAG system"""
    with MongoDBSaver.from_conn_string(URI) as checkpointer:
        rag_chain = create_memory_graph(graph, checkpointer=checkpointer)

        while True:
            user_query = input(">> ")
            user_msg = {"role": "user", "content": user_query}
            rag_state: State = {
                "messages": [user_msg],
                "context": "",
                "run_count": 0,
                "relevance_score": None,
            }
            final_state = await rag_chain.ainvoke(rag_state, config)
            if final_state["run_count"] >= 3 and final_state["relevance_score"] < 5:
                print(
                    "Could not generate a highly relevant answer after 3 attempts. Here is the best attempt:"
                )
            print(final_state["messages"][-1]["content"])


if __name__ == "__main__":
    asyncio.run(main())
