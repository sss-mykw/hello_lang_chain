from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import  SystemMessage

from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field


class OutputModel(BaseModel):
    reasoning: str = Field(..., description='問題を解く上で必要な全ての思考内容と、最終的な結果を出力する')
    conclusion: str = Field(..., description='これまでの思考結果から最終的な結論のみを出力する')


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

def answer_math_question():
    system_prompt = """
    回答する際は下記のように構造化して出力してください
    - reasoning: ユーザの質問をステップバイステップで検討し、最終的な回答を推論し、全ての思考をこのフィールドに出力してください。
    - conclusion: 最終的な回答のみを出力してください。
    """

    question = """
        公園と学校の距離は1200mです。
        A君が公園から、B君は学校から向かい合って同時に出発すると8分で出会いました。
        また別の日、B君が学校から出た5分後にA君が追いかけるとA君が出発して10分後に追いつきました。
        A君の速さは、「分速何m」でしょうか？
        """
    message = [
        SystemMessage(content=system_prompt),
        HumanMessagePromptTemplate.from_template(
            [
                {
                    "type": "text",
                    "text": "{question}"
                },
            ]
        )
    ]
    prompt = ChatPromptTemplate.from_messages(message)

    # memo モデルgemma3:27b-it-qatだと性能が足りず、正しい回答が得られない
    model = ChatOllama(
        model="gemma3:27b-it-qat",
        temperature=0,
        top_p=0.2,
    )

    chain = prompt | model.with_structured_output(OutputModel)
    result = chain.invoke({"question": question})
    print("====modelの出力====")
    print("===思考過程===")
    print(result.reasoning)
    print("===結論===")
    print(result.conclusion)


def main():
    for chunk in chain.stream("日本で一番高い山は何ですか？"):
        print(chunk, end="", flush=True)

    answer_math_question()

if __name__ == "__main__":
    main()
