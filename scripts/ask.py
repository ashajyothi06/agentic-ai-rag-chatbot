import argparse
import json

from app.service import get_rag_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG chatbot from the terminal.")
    parser.add_argument("question", help="Question to ask")
    args = parser.parse_args()

    result = get_rag_service().answer(args.question)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
