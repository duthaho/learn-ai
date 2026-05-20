# Project: Hybrid Retrieval + Reranker

Build a 5-stage RAG pipeline (dense + BM25 → RRF fusion → cross-encoder rerank → grounded answer) on a small embedded corpus, and prove with a labeled eval set that each added stage actually helps.

## What you'll build

- A 5-stage retrieval pipeline (dense FAISS + BM25 → RRF fusion → cross-encoder rerank → LLM answer with `[Source N]` citations).
- A 5-doc, ~33-chunk embedded corpus across Auth / Cache / Observability / Networking / Database topics.
- A 15-query labeled eval set + a `recall@k` / MRR scorer that compares 4 strategies side-by-side.
- A 5-mode CLI: `--ask`, `--explain`, `--eval`, `--list`, `--flush`.

## Prerequisites

- [Module 07 (RAG)](../../07-rag/) — basic pipeline this upgrades.
- [Module 03 (Embeddings & Vector Search)](../../03-embeddings-vector-search/) — `sentence-transformers` bi-encoder and FAISS inner-product index are reused; the cross-encoder is new.
- [Module 15 (Evaluation & Testing)](../../15-evaluation-testing/) — recall@k and MRR are the same flavor of labeled-set scoring; this module is a domain-specific instance.
- [Module 17 (Caching & Cost Optimization)](../../17-caching-cost-optimization/) — the `.rag_index/` cache uses M17's atomic-write-and-rename pattern.

## Setup

`.env` at the repo root supplies your API key. The script resolves it three levels up (`parent.parent.parent / ".env"`), so you can run it from any working directory.

Set `LLM_MODEL` in `.env` (or your shell environment) to pick the model. Default is `openai/gpt-4o-mini` if unset.

Install the dependencies if they are not already in your environment:

```
pip install litellm pydantic python-dotenv sentence-transformers faiss-cpu rank_bm25 numpy
```

First execution downloads two Hugging Face models (~80MB each): `all-MiniLM-L6-v2` (bi-encoder) and `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker). They cache under `~/.cache/huggingface/` and never re-download.

### Project layout

```text
project/
├── README.md         this file
├── solution.py       pipeline + CLI + corpus + eval set (~700 lines)
└── .rag_index/       created at runtime, gitignored
    ├── meta.json
    ├── chunks.json
    ├── embeddings.npy
    └── bm25.pkl
```

Read `solution.py` end-to-end before running it. The corpus, the index builder, the retrieval stages, and the CLI are independently usable — you can import the pipeline class into any script without touching `argparse`. The four files in `.rag_index/` carry different costs to rebuild: `meta.json` is trivial, `chunks.json` is the corpus text and is regenerated from the in-script corpus, `bm25.pkl` is a tokenized index that rebuilds in milliseconds, and `embeddings.npy` is the only expensive artifact — recomputing it on a CPU takes a few seconds per 33 chunks but would scale linearly with corpus size, which is why the cache exists in the first place.

## Walkthrough

Run these four steps in order. Each one builds on the previous.

### Step A — Build the index and confirm

```
python solution.py --list
```

The cold first run prints `building index for 33 chunks…` followed by `saved index to .rag_index/`, then the corpus listing. The second run reads from cache and is instant — embeddings are reused, BM25 is unpickled, and FAISS rebuilds from the saved vectors in milliseconds.

Expected shape of `--list`:

```
Corpus: 5 documents, 33 chunks
  auth.md            7 chunks
  cache.md           6 chunks
  observability.md   7 chunks
  networking.md      7 chunks
  database.md        6 chunks

Eval set: 15 queries (5 keyword, 5 semantic, 5 mixed)
```

If the second run is not instant, the cache did not write — check that `.rag_index/` contains all four files. If `chunks.json` is present but `embeddings.npy` is missing, delete the directory and re-run; the loader treats partial caches as invalid and rebuilds from scratch rather than guessing at half-written state.

The `meta.json` file records the corpus hash and the model names. If you change the corpus or swap the bi-encoder, the hash mismatch triggers a rebuild automatically — you do not need to remember to `--flush` after editing the in-script corpus.

### Step B — Inspect a query with `--explain`

```
python solution.py --explain "what hashing algorithm should I use for passwords"
```

This prints four side-by-side rankings — `DENSE` and `BM25` in the top panel, `RRF FUSED` and `RERANKED` in the bottom. No LLM call is made. The interesting reading is across panels, not down them:

- Which `chunk_id` is rank #1 in each panel?
- Where do `DENSE` and `BM25` agree? Where do they disagree?
- When they disagree, what does `RRF FUSED` do — does it pick one side, or surface a compromise candidate that ranked moderately well in both?
- Compare the top-3 of `RRF FUSED` against `RERANKED`. The cross-encoder usually reshuffles within that top-3, sometimes promoting a chunk from rank 8-10 in the fused list into the final top-3.

That last move — a chunk jumping from rank 9 to rank 2 after reranking — is the moment the cross-encoder earns its keep.

### Step C — Get an answer with `--ask`

```
python solution.py --ask "what hashing algorithm should I use for passwords"
```

Expected shape:

```
retrieval: dense=12ms bm25=3ms fuse=1ms rerank=84ms total=100ms

Use bcrypt or argon2id for password hashing [Source 1]. Both are
deliberately slow algorithms designed to resist brute-force attacks,
unlike general-purpose hashes like SHA-256 [Source 2].

Sources:
  [Source 1] auth.md#chunk_03  (rerank=8.42)
  [Source 2] auth.md#chunk_05  (rerank=4.11)

tokens: in=412 out=58  cost: $0.0001
```

The `[Source N]` citations in the answer match the entries in the sources block — that mapping is enforced by the prompt template, not a post-processing step. If the LLM omits a citation, the answer is still shown but a warning is printed.

### Step D — Run the eval

```
python solution.py --eval
```

This runs all 15 labeled queries through 4 strategies and prints a metrics table:

```
strategy         recall@3   recall@10   MRR@10
dense-only         0.467       0.733     0.521
bm25-only          0.533       0.800     0.589
rrf-fused          0.733       0.933     0.704
rrf+rerank         0.867       0.933     0.812

Per-kind recall@3:
                 dense   bm25   rrf   rrf+rerank
  keyword         0.40   0.80   0.80     0.80
  semantic        0.60   0.20   0.60     0.80
  mixed           0.40   0.60   0.80     1.00
```

Two numbers do the work. The headline number is the gap between `dense-only` and `rrf+rerank` on recall@3 — that is the cumulative lift from every stage you added, and on this corpus it should land somewhere around +0.40 absolute (roughly doubling recall). The per-kind table is the proof that the lift comes from the *right* places: BM25 lifts `keyword`, dense holds `semantic`, RRF picks up `mixed` by combining the two, and rerank tightens all three by promoting the chunks the bi-encoders agreed were *relevant* into the chunks the cross-encoder agrees are *answers*.

MRR@10 falls between the recall numbers in usefulness. Recall@k answers "did we find it"; MRR@k answers "did we find it near the top". A retriever that always puts the answer at rank 1 has MRR = 1.0; one that always puts it at rank 10 has MRR = 0.1; one that misses it has MRR = 0. Reranking moves MRR more than it moves recall — it does not add new chunks, it just reorders the chunks the fuser already found.

## Worked exercise A: Find where rerank changes top-3

1. Run `python solution.py --eval` and scan the per-query output (printed under the table when run with no flags suppressing it). Pick a query of `kind=mixed` where `rrf+rerank` got recall@3 = 1.0 and `rrf-fused` got recall@3 = 0.
2. Run `python solution.py --explain "<that query>"` and put the `RRF FUSED` and `RERANKED` columns next to each other.
3. Identify the `chunk_id` that jumped into the top-3 after reranking. Read its text in the `--list` output, or inspect the chunk file directly:
   ```
   python -c "import json; print(json.load(open('.rag_index/chunks.json'))[10]['text'])"
   ```
   (Replace `10` with the index of the chunk you want.)
4. Explain to yourself *why* the cross-encoder boosted it. Usually it is a chunk that mentions both the keyword from the query AND the surrounding context the query implies — something a bi-encoder, which embeds the query and the chunk independently, cannot model.

The lesson in one sentence: the cross-encoder is doing something a bi-encoder cannot — modeling the joint (query, doc) signal, with both halves attending to each other token-by-token.

## Worked exercise B: Find where BM25 wins

1. Run `python solution.py --eval`. Look at the per-kind breakdown.
2. Find a query of `kind=keyword` where `bm25-only` recall@3 is higher than `dense-only` recall@3. (There are several — `keyword` is BM25's home turf.)
3. Run `python solution.py --explain "<that query>"`. Confirm `BM25` has the relevant chunk near the top of its panel and `DENSE` does not have it in its top-10 at all.
4. Read the relevant chunk in the corpus listing. Notice that the query and the chunk share a rare term — a specific algorithm name, a config flag, an HTTP status code — that BM25 weights heavily (high IDF) and the bi-encoder smears across the broader topic.

The lesson in one sentence: rare-keyword recall is BM25's home turf, and dropping BM25 in favor of "modern embeddings only" silently loses these queries — you would only catch the loss with an eval set that contains keyword-flavored questions.

## Live-test commands

Run these from the repo root:

```
python 19-advanced-rag/project/solution.py --list
python 19-advanced-rag/project/solution.py --explain "what hashing algorithm should I use for passwords"
python 19-advanced-rag/project/solution.py --ask "what hashing algorithm should I use for passwords"
python 19-advanced-rag/project/solution.py --eval
python 19-advanced-rag/project/solution.py --flush
```

`--explain` and `--eval` make zero LLM calls — they are free to run as often as you like, and they are the modes you will iterate on while developing. Only `--ask` spends money, and at gpt-4o-mini prices a single ask costs around $0.0001. `--flush` deletes the `.rag_index/` directory; the next run rebuilds it from scratch and re-downloads nothing (the Hugging Face model cache lives elsewhere).

## Where to go next

- [Module 20 (Deployment Patterns)](../../20-deployment-patterns/) — running this pipeline in production: latency budgets (cross-encoder rerank is the slowest stage and the first to cut under load), fallback chains (rerank → fused → dense-only), and A/B testing of retrieval strategies against live user queries.
- Module 19 README Section 7 ("The Wider Toolbox") — concept overview of HyDE, parent-document retrieval, multi-query rewriting, self-querying, and contextual compression. Each is a one-paragraph idea with a clear shape. Pick one and try wedging it into this pipeline as an exercise — most slot in cleanly as a sixth stage or as a query-time preprocessor, and the eval harness you already have will tell you instantly whether the addition helps or hurts.
