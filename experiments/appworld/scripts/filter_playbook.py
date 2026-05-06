import argparse
import json
from pathlib import Path


def is_answer_guideline(item):
    content = item.get("content", "").lower()
    kill_keywords = [
        "answer",
        "complete_task",
        "summary",
        "summarize",
        "narrative",
    ]
    return any(keyword in content for keyword in kill_keywords)


def filter_items(items, name):
    original_count = len(items)
    filtered = []
    removed = []

    for item in items:
        if is_answer_guideline(item):
            removed.append(item)
        else:
            filtered.append(item)

    print(f"[{name}] Original: {original_count}")
    print(f"[{name}] Removed: {len(removed)}")
    print(f"[{name}] Remaining: {len(filtered)}")
    return filtered, removed


def build_default_output_path(source_file):
    return source_file.with_name(f"{source_file.stem}_no_answer_guidelines{source_file.suffix}")


def main():
    parser = argparse.ArgumentParser(description="Filter answer-format guidelines from an AppWorld playbook.")
    parser.add_argument("source_file", type=Path)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()

    output_file = args.output_file or build_default_output_path(args.source_file)

    with args.source_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    filtered_strategies, removed_strategies = filter_items(data.get("strategies", []), "Strategies")
    filtered_apis, removed_apis = filter_items(data.get("apis", []), "APIs")
    data["strategies"] = filtered_strategies
    data["apis"] = filtered_apis

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved to: {output_file}")

    print("\n--- Examples of Removed Strategies ---")
    for strategy in removed_strategies[:5]:
        print(f"ID: {strategy.get('id')}\nContent: {strategy.get('content')}\n")

    print("\n--- Examples of Removed APIs ---")
    for api in removed_apis[:5]:
        print(f"ID: {api.get('id')}\nContent: {api.get('content')}\n")


if __name__ == "__main__":
    main()
