from conversation import conversation
from web_search_graph import create_lang_graph


def main():
    graph = create_lang_graph()
    conversation(graph)

if __name__ == "__main__":
    main()
