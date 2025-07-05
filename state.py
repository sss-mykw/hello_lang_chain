import operator
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage


class GraphState(TypedDict):
    # messagesがキー、list[AnyMessage]がバリュー
    # messagesキーには、HumanMessageやAIMessageなど、任意のメッセージオブジェクトのリストが格納される
    # list[AnyMessage]バリューには、messagesキーに新しい値（メッセージリスト）が渡された場合、既存のリストにその値を追加（add）する
    messages: Annotated[list[AnyMessage], operator.add]
