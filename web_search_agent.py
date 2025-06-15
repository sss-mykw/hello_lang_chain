import operator
from typing import TypedDict, Annotated

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_tavily import TavilySearch, TavilyExtract

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

class GraphState(TypedDict):
    # messagesがキー、list[AnyMessage]がバリュー
    # messagesキーには、HumanMessageやAIMessageなど、任意のメッセージオブジェクトのリストが格納される
    # list[AnyMessage]バリューには、messagesキーに新しい値（メッセージリスト）が渡された場合、既存のリストにその値を追加（add）する
    messages: Annotated[list[AnyMessage], operator.add]


def create_langgraph(chain, tools):
    def call_llm(state: GraphState):
        response = chain.invoke({"messages": state["messages"]})
        print("====response====")
        print(response)
        return {"messages": [response]}

    def should_continue(state: GraphState):
        messages = state["messages"]
        last_message = messages[-1]
        # LLMがツール呼び出しを要求したかどうか
        if last_message.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(state_schema=GraphState)
    # nodeの追加
    node_name_agent = "agent"
    node_name_tools = "tools"
    workflow.add_node(node_name_agent, call_llm)
    workflow.add_node(node_name_tools, ToolNode(tools))
    # edgeの追加
    workflow.add_edge(START, node_name_agent)
    workflow.add_conditional_edges(node_name_agent, path=should_continue, path_map=[node_name_tools, END])
    # toolsノードの結果をLLMに返し、再びツールを使用する必要があるかどうかを判断する
    workflow.add_edge(node_name_tools, node_name_agent)
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

        # デバック用
        print("response: ", response)

        print("=================================")
        print("AIの回答", response["messages"][-1].content)

def main():
    # orieg/gemma3-tools:27b-it-qatだと性能不足でtoolを活用することが出来なかった
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-04-17",
        temperature=0,
        max_retries=0,
    )

    tavily_search_tool = TavilySearch(
        max_results=10,
        topic="general",
    )

    tavily_extract_tool = TavilyExtract()

    tools = [
        tavily_search_tool,
        tavily_extract_tool,
    ]

    system_prompt = """
    あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。また考える過程も出力してください。
    私たちは「tavily_search_tool」と「tavily_extract_tool」という2つのツールを持っています。
    tavily_search_toolは、Google検索を行い、上位5件のURLや概要を取得するツールです。どんなwebサイトがあるかを浅く拾う場合にはこちらを利用します
    tavily_extract_toolは、URLを指定して、ページの内容を抽出するツールです。特定のWebサイトのURLがわかっており、詳細に内容を取得する場合はこちらを利用します。
    適切に利用してユーザからの質問に回答してください。
    """

    # messageを作成する
    message = [
        SystemMessage(content=system_prompt),
        # メッセージのリスト（会話履歴）を動的に挿入するためのplaceholder
        MessagesPlaceholder("messages"),
    ]

    # messageからプロンプトを作成
    prompt = ChatPromptTemplate.from_messages(message)

    # chainとgraphを作成
    chain = prompt | llm.bind_tools(tools)
    graph = create_langgraph(chain, tools)

    conversation(graph)

if __name__ == "__main__":
    main()
