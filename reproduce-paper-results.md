# Reproduce the paper results

Run these commands from the repository root after extracting the data archive. Each command reads only
`data/processed/`; generated tables, summaries, and PDFs go to
`results/`. The released inputs are processed
intermediates, so these commands do not rerun the agents or extract features
from `data/raw_traj/`. Install the dependencies with `python3 -m pip install -r scripts/requirements.txt`.

Tables 12–15 share one script, so each point command also regenerates the other
tables in that set.

## 1. Atomic is more consistent and has fewer interaction errors

**Results:** Tables 2, 4, 9, 10, and 12; Figure 6.

```bash
python3 scripts/reproduce_from_processed.py --point atomic
```

Tables 2, 4, and 12 count resolved attempts for each task and calculate
consistency at *k* as `C(successes, k) / C(attempts, k)`, then average across
tasks; unfinished attempts count as failures. Table 12 reads the
additional-task outcomes directly from `data/processed/tables12_15/evaluation_outcomes/`.
Figure 6 averages classified environment-interaction errors across repeats
and tasks using `data/processed/error_labels/`. Table 9 summarizes parsed
BashOnly operations per step, and Table 10 compares BashOnly and Atomic
trajectory medians for steps and token use. The command writes the reproduced
tables and Figure 6 PDF under `results/`.

## 2. NLSearch explores more diverse context

**Results:** Figure 3; Tables 5, 7, and 13.

```bash
python3 scripts/reproduce_from_processed.py --point nlsearch
```

Figure 3 averages pairwise Jaccard distances between file-read sets across
repeats for exploration diversity and uses processed CodeBLEU patch distances
for final-solution diversity. Table 5 averages processed task-level Jaccard and
normalized Levenshtein distances for early search actions. Table 7 computes
per-trajectory read precision and recall against the task-level relevant-file
sets in `data/processed/high_relevant_files/`. Table 13 compares processed
additional-task group means of read-set Jaccard diversity as a percentage
change from BashOnly. The command writes the reproduced figure, tables, and
metric summaries under `results/`.

## 3. The Python interface is more efficient

**Results:** Figure 4; Tables 8, 14, and 15.

```bash
python3 scripts/reproduce_from_processed.py --point python
```

Figure 4 groups processed trajectories by model and setup, then plots mean
interaction steps against mean cumulative input tokens with evaluated resolve
rates. Table 8 averages per-trajectory steps, total output tokens, total
unique observation tokens, and the corresponding per-step ratios for BashOnly
and Python. Tables 14 and 15 compare Python with BashOnly on processed group
means of input tokens and steps in the additional tasks. Each displayed
difference is calculated before taking the equal-weight Overall average. The
command writes the Figure 4 PDF and table summaries under
`results/`.

To reproduce all supported results at once, run
`python3 scripts/reproduce_from_processed.py`.
