from langchain_tavily import TavilySearch, TavilyExtract

tavily_search_tool = TavilySearch(
    max_results=10,
    topic="general",
)

tavily_extract_tool = TavilyExtract()

tavily_tools = [
    tavily_search_tool,
    tavily_extract_tool,
]
