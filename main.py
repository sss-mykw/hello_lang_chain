from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


def make_prompt(user_input: str) -> list:
    return [
        ("system", "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。また考える過程も出力してください。"),
        ("human", f"{user_input}")
    ]


# 保守的（一貫性が高く、堅実な出力）になるようにtemperatureとtop_pを設定
model = ChatOllama(
    model="gemma3:27b-it-qat",
    temperature=0.2,
    top_p=0.95,
)

# プロンプトテンプレート ＋ モデル ＋ 出力パーサー
chain = make_prompt | model | StrOutputParser()

def main():
    for chunk in chain.stream("日本で一番高い山は何ですか？"):
        print(chunk, end="", flush=True)

if __name__ == "__main__":
    main()
