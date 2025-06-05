import asyncio

from agents import Agent, OpenAIProvider, RunConfig, Runner, set_tracing_disabled

# OpenAIのサーバーへ送られるトレースの完全無効化
# 「OPENAI_API_KEY is not set, skipping trace export」を防ぐ
set_tracing_disabled(disabled=True)

# config設定
run_config = RunConfig(
    model_provider=OpenAIProvider(
        api_key="lm-studio",  # ダミーキー
        base_url="http://localhost:1234/v1/",
        use_responses=False,  # LM Studioはresponse APIを持っていないためFalseにしておく
    ),
    model="google/gemma-3-27b",
)

# Agentの定義
agent = Agent(
    name="Simple Agent",
    instructions="あなたは与えられたタスクを行うAgentです。",
)
# print(f"agent name:{agent.name}")
# print(f"agent model:{agent.model}")
# print(f"agent tools:{agent.tools}")

async def main():
    msg = "自己紹介をしてください。"
    result = await Runner.run(agent, msg, run_config=run_config)
    print(f"プロンプト\n{result.input}")
    print()
    print(f"回答\n{result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
