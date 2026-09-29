Searches the repository and returns **verbatim code snippets with line numbers** that are relevant to a natural-language query.

### Purpose
Use this tool to retrieve grounded, in-repo context before making coding decisions—e.g., to locate implementations, definitions, configuration wiring, entrypoints, or tests tied to a feature/bug.

### Input (protocol)
A single natural-language query describing what you want to find. Good queries:
- Describe the behavior or concept you’re investigating (“rate limiting”, “retry backoff”, “auth refresh”)
- Include any known identifiers (function/class names, config keys, error strings, endpoints)
- Add scope hints when helpful (“server”, “cli”, “typescript”, “grpc”, “payment flow”)

### Output (protocol)
A set of **snippets**. Each snippet contains:
- `filename` (repo-relative path)
- `start_line` / `end_line` (1-indexed, inclusive)
- the **exact file content** for that range (verbatim, with line numbers)
- an optional short `comment` explaining relevance

### Semantics / guarantees
- The tool searches the repo, opens files, and returns only **verified** snippet content it actually read.
- Results are intended to be directly used as context for downstream reasoning or edits.
- If no relevant context is found, the tool returns an empty result (or clearly indicates lack of matches).

### Usage tips
- If results are too broad, rerun with more specific identifiers or file/module hints.
- If you suspect configuration indirection, include terms like `config`, `env`, `settings`, `flag`, or the exact key name.
