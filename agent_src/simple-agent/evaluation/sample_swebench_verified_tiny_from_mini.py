import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def sample_per_repo(entries, per_repo: int, seed: int):
    rng = random.Random(seed)
    repo2entries = defaultdict(list)
    for entry in entries:
        repo2entries[entry["repo"]].append(dict(entry))

    selected_entries = []
    for repo in sorted(repo2entries):
        repo_entries = sorted(repo2entries[repo], key=lambda entry: entry["instance_id"])
        rng.shuffle(repo_entries)
        selected_entries.extend(repo_entries[:per_repo])

    return selected_entries


def main():
    parser = argparse.ArgumentParser(
        description="Sample a reproducible per-repo tiny split from SWE-bench Verified mini."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("swebench_verified_mini.json"),
        help="Mini split JSON to sample from.",
    )
    parser.add_argument("--per-repo", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("swebench_verified_tiny.json"),
    )
    parser.add_argument(
        "--ids-output",
        type=Path,
        default=Path("swebench_verified_tiny_ids.txt"),
    )
    args = parser.parse_args()

    if args.per_repo <= 0:
        raise ValueError("--per-repo must be positive")

    entries = json.load(open(args.input))
    selected_entries = sample_per_repo(entries, args.per_repo, args.seed)

    with open(args.output, "w") as f:
        json.dump(selected_entries, f, indent=2)
        f.write("\n")

    if args.ids_output:
        with open(args.ids_output, "w") as f:
            for entry in selected_entries:
                f.write(entry["instance_id"] + "\n")

    repo_counts = defaultdict(int)
    source_repo_counts = defaultdict(int)
    for entry in entries:
        source_repo_counts[entry["repo"]] += 1
    for entry in selected_entries:
        repo_counts[entry["repo"]] += 1

    print(f"Wrote {len(selected_entries)} entries to {args.output}")
    if args.ids_output:
        print(f"Wrote instance ids to {args.ids_output}")
    print(f"Seed: {args.seed}")
    print("Repo distribution:")
    for repo in sorted(source_repo_counts):
        print(f"  {repo}: {repo_counts[repo]} of {source_repo_counts[repo]}")


if __name__ == "__main__":
    main()
