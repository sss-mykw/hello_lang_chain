from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import  SystemMessage
from langchain_core.runnables import Runnable, RunnableParallel, RunnableBranch, RunnableLambda

from langchain_ollama import ChatOllama

from pydantic import BaseModel, Field


class OutputModel(BaseModel):
    reasoning: str = Field(..., description='問題を解く上で必要な全ての思考内容と、最終的な結果を出力する')
    conclusion: str = Field(..., description='これまでの思考結果から最終的な結論のみを出力する')

# 出力の構造を定義する
class Router(BaseModel):
    reasoning: str = Field(..., description='LLMの思考過程')
    next: str = Field(..., description='LLMの結論')


def make_prompt(user_input: str) -> list:
    return [
        ("system", "あなたは日本語を話す優秀なアシスタントです。回答には必ず日本語で答えてください。また考える過程も出力してください。"),
        ("human", f"{user_input}")
    ]


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


def answer_stream(input: str):
    # 保守的（一貫性が高く、堅実な出力）になるようにtemperatureとtop_pを設定
    model = ChatOllama(
        model="gemma3:27b-it-qat",
        temperature=0.2,
        top_p=0.95,
    )

    # プロンプトテンプレート ＋ モデル ＋ 出力パーサー
    chain = make_prompt | model | StrOutputParser()

    for chunk in chain.stream(input):
        print(chunk, end="", flush=True)


def get_router_chain() -> Runnable:
    # 質問分類機としての役割を定義する
    routing_prompt = """
        # 目的
        あなたは、ユーザの質問文を受け取り、どんな分野の質問かを判断する必要があります。
        便宜上「質問」と記載していますが、雑談であっても、分野を判断してください。

        # 出力構成
        あなたは`Router`クラスで定義された構造体を出力してください。
        - `reasoning`フィールドには、あなたの思考過程を出力してください。
        - `next`フィールドには、質問文の分野を出力してください。候補は[お菓子、料理、その他]の3つです。 
        """

    # promptテンプレートを定義する
    router_message = [
        SystemMessage(content=routing_prompt),
        HumanMessagePromptTemplate.from_template(
            [
                {
                    "type": "text",
                    "text": "{question}"
                },
            ]
        )
    ]
    router_prompt = ChatPromptTemplate.from_messages(router_message)

    model = ChatOllama(
        model="gemma3:27b-it-qat",
        temperature=0.2,
        top_p=0.95,
    )

    # 質問分類機のchainを定義する
    return router_prompt | model.with_structured_output(Router)


def get_cooking_chain() -> Runnable:
    # 料理の専門家としての振る舞いを定義する
    cooking_prompt = """
    # 目的
    あなたは、料理の専門家です。ユーザからの質問には「肉料理最高」の立場をとって、「魚料理」や「野菜料理」を否定するようにしてください。
    """
    # promptテンプレートを定義する
    cooking_message = [
        SystemMessage(content=cooking_prompt),
        HumanMessagePromptTemplate.from_template(
            [
                {
                    "type": "text",
                    "text": "{question}"
                },
            ]
        )
    ]
    cooking_prompt = ChatPromptTemplate.from_messages(cooking_message)

    model = ChatOllama(
        model="gemma3:27b-it-qat",
        temperature=0.2,
        top_p=0.95,
    )

    return cooking_prompt | model | StrOutputParser()


def get_sweets_chain() -> Runnable:
    # お菓子の専門家としての振る舞いを定義する
    sweets_prompt = """
    # 目的
    あなたは、お菓子の専門家です。ユーザからの質問には「たけのこの里最高」の立場をとって、「きのこの山」や「他のお菓子」を否定するようにしてください。
    """

    # promptテンプレートを定義する
    sweets_message = [
        SystemMessage(content=sweets_prompt),
        HumanMessagePromptTemplate.from_template(
            [
                {
                    "type": "text",
                    "text": "{question}"
                },
            ]
        )
    ]
    sweets_prompt = ChatPromptTemplate.from_messages(sweets_message)

    model = ChatOllama(
        model="gemma3:27b-it-qat",
        temperature=0.2,
        top_p=0.95,
    )

    return sweets_prompt | model | StrOutputParser()


def router_sample():
    router_chain = get_router_chain()
    cooking_chain = get_cooking_chain()
    sweets_chain = get_sweets_chain()

    # AIワークフローをchainとして定義する
    key_question = "question"
    key_router = "router"
    chain = (
        RunnableParallel(
            {
                key_question: RunnableLambda(lambda x: x[key_question]),
                key_router: router_chain,
            }
        ).assign(
            output=RunnableBranch(
                ((lambda x: x[key_router].next == "料理"), cooking_chain),
                ((lambda x: x[key_router].next == "お菓子"), sweets_chain),
                RunnableLambda(lambda x: f"申し訳ありませんが、料理やお菓子以外の内容についてはお答えできません。")
            )
        )
    )

    questions = [
        "お菓子の中で一番美味しいものは？",
        "今日の晩御飯何にしたらいい？",
        "最近運動不足なんですが、何かいいアドバイスはありますか？",
        "きのこの山って美味しいですよね？"
    ]

    for question in questions:
        result = chain.invoke({key_question: question})
        print("\n====質問内容=====")
        print(result[key_question])
        print("====ルーティング結果=====")
        print(result[key_router])
        print("====最終結果=====")
        print(result["output"])


def main():
    # answer_stream("日本で一番高い山は何ですか？")

    # answer_math_question()

    router_sample()

if __name__ == "__main__":
    main()
