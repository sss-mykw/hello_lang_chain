import operator
from typing import TypedDict, Annotated

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage

from langchain_ollama import ChatOllama

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph

class GraphState(TypedDict):
    # messagesがキー、list[AnyMessage]がバリュー
    # messagesキーには、HumanMessageやAIMessageなど、任意のメッセージオブジェクトのリストが格納される
    # list[AnyMessage]バリューには、messagesキーに新しい値（メッセージリスト）が渡された場合、既存のリストにその値を追加（add）する
    messages: Annotated[list[AnyMessage], operator.add]


def create_langgraph(chain):
    def call_llm(state: GraphState):
        response = chain.invoke({"messages": state["messages"]})
        return {"messages": [response]}

    workflow = StateGraph(state_schema=GraphState)
    # nodeの追加
    workflow.add_node("model", call_llm)
    # edgeの追加
    workflow.add_edge(START, "model")
    workflow.add_edge("model", END)
    # グラフの状態をインメモリーに保存
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    return graph

def conversation(graph):
    while True:
        query = input("質問を入力してください: ")

        if query.lower() in ["exit", "quit"]:
            print("終了します。")
            break

        print("=================================")
        print("質問:", query)

        input_query = [HumanMessage(
            [
                {
                    "type": "text",
                    "text": f"{query}"
                },
            ]
        )]

        # 同じスレッドIDでinvokeが繰り返されることで、会話履歴が引き継がれる
        response = graph.invoke({"messages": input_query}, config={"configurable": {"thread_id": "12345"}})

        print("=================================")
        print("AIの回答", response["messages"][-1].content)

def main():
    model = ChatOllama(
        model="gemma3:27b-it-qat",
        temperature=0.2,
        top_p=0.95,
    )

    # messageを作成する
    message = [
        SystemMessage(
            content="あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。また考える過程も出力してください。"),
        # メッセージのリスト（会話履歴）を動的に挿入するためのplaceholder
        MessagesPlaceholder("messages"),
    ]

    # messageからプロンプトを作成
    prompt = ChatPromptTemplate.from_messages(message)

    # chainとgraphを作成
    chain = prompt | model
    graph = create_langgraph(chain)

    conversation(graph)

if __name__ == "__main__":
    main()
