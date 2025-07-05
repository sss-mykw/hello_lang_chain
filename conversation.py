from langchain_core.messages import HumanMessage

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
        response = graph.invoke(
            {"messages": input_query},
            config={"configurable": {"thread_id": "12345"}}
        )

        # デバック用
        print("response: ", response)

        print("=================================")
        print("AIの回答", response["messages"][-1].content)
