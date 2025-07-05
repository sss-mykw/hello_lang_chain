from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

from prompts import web_search_system_prompt
from state import GraphState
from tools import tavily_tools

# orieg/gemma3-tools:27b-it-qatだと性能不足でtoolを活用することが出来なかった
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    temperature=0,
    max_retries=0,
)

tools = []
tools.extend(tavily_tools)

# messageを作成する
message = [
    SystemMessage(content=web_search_system_prompt),
    # メッセージのリスト（会話履歴）を動的に挿入するためのplaceholder
    MessagesPlaceholder("messages"),
]

# messageからプロンプトを作成
prompt = ChatPromptTemplate.from_messages(message)

# chainとgraphを作成
chain = prompt | llm.bind_tools(tools)


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

def create_lang_graph():
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
