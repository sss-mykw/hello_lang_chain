from openai import OpenAI

# ローカルサーバーに接続
# api_keyはダミー値を設定
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
)

def main():
    completion = client.chat.completions.create(
        model="google/gemma-3-27b",
        messages=[
            {"role": "system",
             "content": "あなたは優秀なアシスタントです。あなたは人類最高知能を持つAIなので、人間の質問に対しても完璧に回答することが出来ます。"},
            {"role": "user", "content": "自己紹介をしてください。"}
        ],
        temperature=0.7
    )

    # 実際の出力の表示
    print(completion.choices[0].message.content)


if __name__ == "__main__":
    main()
