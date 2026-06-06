# ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦
# ✦ Author  : Sobaha
# ✦ Created : 2026-01-04 12:49:34
# ✦ Life Is a Series of Events
# ✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦✦

#!/usr/bin/env python3
"""

Usage:
  pip install openai pandas numpy

  export OPENAI_API_KEY="..."

  python auto_fix_train_from_mismatches.py \
    --train train.csv \
    --mismatch mismatch.csv \
    --out-train train.fixed.csv \
    --out-log fixes.log.csv \
    --train-question-col question \
    --train-tag-col tag \
    --mm-test-question-col question \
    --mm-mapped-train-question-col similar_question \
    --mm-expected-tag-col expected_tag \
    --mm-predicted-tag-col predicted_tag

Notes:
- Applies only high-confidence decisions (configurable).
- Never deletes rows by default (safer). It can MOVE tags and/or ADD new rows.
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from openai import OpenAI


# -----------------------------
# Config
# -----------------------------

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def normalize_text(s: str) -> str:
    # light normalization to reduce duplicate drift
    return " ".join(str(s).strip().split())


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def batched(iterable, n: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def safe_str(x) -> str:
    return "" if pd.isna(x) else str(x)


# -----------------------------
# Embedding cache (optional but helpful)
# -----------------------------

class EmbeddingCache:
    """
    Very simple disk cache: JSONL { "text": "...", "vec": [...] }
    Good enough for avoiding repeated embedding calls across runs.
    """
    def __init__(self, path: str):
        self.path = path
        self.map: Dict[str, np.ndarray] = {}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    self.map[obj["text"]] = np.array(obj["vec"], dtype=np.float32)

    def get(self, text: str) -> Optional[np.ndarray]:
        return self.map.get(text)

    def put_many(self, items: List[Tuple[str, np.ndarray]]):
        # append-only
        with open(self.path, "a", encoding="utf-8") as f:
            for text, vec in items:
                if text in self.map:
                    continue
                obj = {"text": text, "vec": vec.tolist()}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                self.map[text] = vec


# -----------------------------
# LLM structured decision schema
# -----------------------------

JUDGE_SCHEMA = {
    "name": "train_fix_decision",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "KEEP_NO_CHANGE",
                    "ADD_TEST_AS_VARIATION",
                    "MOVE_MAPPED_TRAIN_QUESTION",
                    "MOVE_AND_ADD",
                    "AMBIGUOUS_REVIEW",
                    "EXPECTED_TAG_LIKELY_WRONG"
                ]
            },
            "add_tag": {"type": ["string", "null"]},
            "move_tag": {"type": ["string", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
            "extra_notes": {"type": "string"}
        },
        "required": ["action", "add_tag", "move_tag", "confidence", "rationale", "extra_notes"]
    }
}


@dataclass
class CandidateIntent:
    tag: str
    sim: float
    examples: List[str]


# -----------------------------
# Core functions
# -----------------------------

def embed_texts(
    client: OpenAI,
    texts: List[str],
    model: str,
    cache: Optional[EmbeddingCache] = None,
    batch_size: int = 128
) -> np.ndarray:
    # Return vectors in same order as texts
    vectors: List[np.ndarray] = [None] * len(texts)  # type: ignore
    to_embed = []
    to_embed_idx = []

    for i, t in enumerate(texts):
        t_norm = normalize_text(t)
        if cache:
            v = cache.get(t_norm)
            if v is not None:
                vectors[i] = v
                continue
        to_embed.append(t_norm)
        to_embed_idx.append(i)

    if to_embed:
        new_items_for_cache = []
        for chunk, idx_chunk in zip(batched(to_embed, batch_size), batched(to_embed_idx, batch_size)):
            resp = client.embeddings.create(model=model, input=chunk)
            for j, item in enumerate(resp.data):
                vec = np.array(item.embedding, dtype=np.float32)
                i = idx_chunk[j]
                vectors[i] = vec
                new_items_for_cache.append((normalize_text(texts[i]), vec))
        if cache:
            cache.put_many(new_items_for_cache)

    return np.vstack(vectors)


def build_tag_prototypes(
    client: OpenAI,
    train_df: pd.DataFrame,
    qcol: str,
    tcol: str,
    embed_model: str,
    cache: Optional[EmbeddingCache],
    max_examples_per_tag: int = 30,
) -> Tuple[List[str], np.ndarray, Dict[str, List[str]]]:
    """
    Build centroid embedding per tag + keep example texts per tag.
    """
    tag2examples: Dict[str, List[str]] = {}
    for tag, g in train_df.groupby(tcol):
        ex = g[qcol].dropna().astype(str).map(normalize_text).tolist()
        # deterministic selection: take first N unique
        uniq = []
        seen = set()
        for s in ex:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
            if len(uniq) >= max_examples_per_tag:
                break
        tag2examples[str(tag)] = uniq

    tags = sorted(tag2examples.keys())
    centroids = []

    for tag in tags:
        ex = tag2examples[tag]
        if not ex:
            # should not happen, but keep stable
            centroids.append(np.zeros((1536,), dtype=np.float32))
            continue
        vecs = embed_texts(client, ex, embed_model, cache=cache)
        centroids.append(vecs.mean(axis=0))

    return tags, np.vstack(centroids), tag2examples


def shortlist_candidate_intents(
    query_vec: np.ndarray,
    tags: List[str],
    centroids: np.ndarray,
    tag2examples: Dict[str, List[str]],
    include_tags: List[str],
    top_k: int = 6,
    examples_per_tag: int = 4,
) -> List[CandidateIntent]:
    sims = [cosine_sim(query_vec, centroids[i]) for i in range(len(tags))]
    order = np.argsort(sims)[::-1]

    chosen = []
    chosen_set = set()

    # always include expected/predicted first (if exists)
    for t in include_tags:
        if t and t in tag2examples and t not in chosen_set:
            idx = tags.index(t) if t in tags else None
            sim = sims[idx] if idx is not None else 0.0
            chosen.append(CandidateIntent(t, float(sim), tag2examples[t][:examples_per_tag]))
            chosen_set.add(t)

    # fill with top centroids
    for i in order:
        t = tags[i]
        if t in chosen_set:
            continue
        chosen.append(CandidateIntent(t, float(sims[i]), tag2examples[t][:examples_per_tag]))
        chosen_set.add(t)
        if len(chosen) >= top_k:
            break

    return chosen


def judge_fix(
    client: OpenAI,
    judge_model: str,
    test_question: str,
    expected_tag: str,
    predicted_tag: str,
    mapped_train_question: str,
    mapped_train_current_tag: Optional[str],
    candidates: List[CandidateIntent],
) -> dict:
    system = (
        "You are a strict dataset labeling auditor for an intent/tag classification dataset.\n"
        "Goal: decide safe train-set fixes that will likely improve accuracy.\n"
        "You will be given:\n"
        "- a test question (mismatched case)\n"
        "- expected_tag and predicted_tag from evaluation\n"
        "- the train question that the system mapped to (mapped_train_question) and its current train tag\n"
        "- candidate intents with a few example questions from train\n\n"
        "Rules:\n"
        "- Choose MOVE_MAPPED_TRAIN_QUESTION only if the mapped_train_question is clearly mislabeled in train.\n"
        "- Choose ADD_TEST_AS_VARIATION only if the test_question is clearly a valid variation of add_tag.\n"
        "- Choose MOVE_AND_ADD if BOTH are true.\n"
        "- Choose EXPECTED_TAG_LIKELY_WRONG if expected_tag does not match the meaning, and predicted_tag (or another candidate) does.\n"
        "- If unclear/overlapping intents, return AMBIGUOUS_REVIEW.\n"
        "- Output MUST follow the JSON schema exactly.\n"
        "- Prefer conservative actions if confidence is low.\n"
    )

    payload = {
        "test_question": normalize_text(test_question),
        "expected_tag": expected_tag,
        "predicted_tag": predicted_tag,
        "mapped_train_question": normalize_text(mapped_train_question),
        "mapped_train_current_tag": mapped_train_current_tag,
        "candidate_intents": [
            {
                "tag": c.tag,
                "similarity": round(c.sim, 4),
                "examples": c.examples
            }
            for c in candidates
        ],
    }

    resp = client.responses.create(
        model=judge_model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
        text={
            "format": {
                "type": "json_schema",
                "json_schema": JUDGE_SCHEMA
            }
        }
    )

    return json.loads(resp.output_text)


def find_train_rows_by_question(train_df: pd.DataFrame, qcol: str, q: str) -> List[int]:
    qn = normalize_text(q)
    matches = train_df.index[train_df[qcol].astype(str).map(normalize_text) == qn].tolist()
    return [int(i) for i in matches]


def add_row_if_missing(train_df: pd.DataFrame, qcol: str, tcol: str, question: str, tag: str) -> Tuple[pd.DataFrame, bool]:
    qn = normalize_text(question)
    exists = (train_df[qcol].astype(str).map(normalize_text) == qn).any()
    if exists:
        return train_df, False
    new_row = {qcol: question, tcol: tag}
    return pd.concat([train_df, pd.DataFrame([new_row])], ignore_index=True), True


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--mismatch", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-log", required=True)

    ap.add_argument("--train-question-col", default="question")
    ap.add_argument("--train-tag-col", default="tag")

    ap.add_argument("--mm-test-question-col", default="question")
    ap.add_argument("--mm-mapped-train-question-col", default="similar_question")
    ap.add_argument("--mm-expected-tag-col", default="expected_tag")
    ap.add_argument("--mm-predicted-tag-col", default="predicted_tag")

    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)

    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--examples-per-tag", type=int, default=4)

    ap.add_argument("--apply-threshold", type=float, default=0.85)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all mismatches")

    ap.add_argument("--cache-path", default=".emb_cache.jsonl")

    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY is not set in environment.")

    client = OpenAI()
    cache = EmbeddingCache(args.cache_path) if args.cache_path else None

    train_df = pd.read_csv(args.train)
    mm_df = pd.read_csv(args.mismatch)

    # validate columns
    for col in [args.train_question_col, args.train_tag_col]:
        if col not in train_df.columns:
            raise SystemExit(f"ERROR: train CSV missing column: {col}")

    for col in [args.mm_test_question_col, args.mm_mapped_train_question_col, args.mm_expected_tag_col, args.mm_predicted_tag_col]:
        if col not in mm_df.columns:
            raise SystemExit(f"ERROR: mismatch CSV missing column: {col}")

    # Normalize core fields
    train_df[args.train_question_col] = train_df[args.train_question_col].astype(str)
    train_df[args.train_tag_col] = train_df[args.train_tag_col].astype(str)

    mm_df = mm_df.copy()
    mm_df[args.mm_test_question_col] = mm_df[args.mm_test_question_col].astype(str)
    mm_df[args.mm_mapped_train_question_col] = mm_df[args.mm_mapped_train_question_col].astype(str)
    mm_df[args.mm_expected_tag_col] = mm_df[args.mm_expected_tag_col].astype(str)
    mm_df[args.mm_predicted_tag_col] = mm_df[args.mm_predicted_tag_col].astype(str)

    # Keep only real mismatches
    mm_df = mm_df[mm_df[args.mm_expected_tag_col] != mm_df[args.mm_predicted_tag_col]].reset_index(drop=True)
    if args.max_rows and args.max_rows > 0:
        mm_df = mm_df.head(args.max_rows)

    if len(mm_df) == 0:
        print("No mismatches found (expected_tag == predicted_tag for all rows). Nothing to do.")
        train_df.to_csv(args.out_train, index=False)
        pd.DataFrame([]).to_csv(args.out_log, index=False)
        return

    # Build tag centroids from train
    tags, centroids, tag2examples = build_tag_prototypes(
        client=client,
        train_df=train_df,
        qcol=args.train_question_col,
        tcol=args.train_tag_col,
        embed_model=args.embed_model,
        cache=cache,
        max_examples_per_tag=30,
    )

    # Embed all mismatched test questions once
    test_questions = mm_df[args.mm_test_question_col].astype(str).map(normalize_text).tolist()
    test_vecs = embed_texts(client, test_questions, args.embed_model, cache=cache)

    logs = []
    applied_moves = 0
    applied_adds = 0

    for i in range(len(mm_df)):
        row = mm_df.iloc[i]
        test_q = safe_str(row[args.mm_test_question_col])
        mapped_q = safe_str(row[args.mm_mapped_train_question_col])
        expected = safe_str(row[args.mm_expected_tag_col])
        predicted = safe_str(row[args.mm_predicted_tag_col])

        # find mapped train row tag (exact match)
        mapped_idxs = find_train_rows_by_question(train_df, args.train_question_col, mapped_q)
        mapped_current_tag = None
        if mapped_idxs:
            mapped_current_tag = str(train_df.loc[mapped_idxs[0], args.train_tag_col])

        qvec = test_vecs[i]

        candidates = shortlist_candidate_intents(
            query_vec=qvec,
            tags=tags,
            centroids=centroids,
            tag2examples=tag2examples,
            include_tags=[expected, predicted],
            top_k=args.top_k,
            examples_per_tag=args.examples_per_tag,
        )

        decision = judge_fix(
            client=client,
            judge_model=args.judge_model,
            test_question=test_q,
            expected_tag=expected,
            predicted_tag=predicted,
            mapped_train_question=mapped_q,
            mapped_train_current_tag=mapped_current_tag,
            candidates=candidates,
        )

        action = decision["action"]
        conf = float(decision["confidence"])
        add_tag = decision["add_tag"]
        move_tag = decision["move_tag"]

        applied = False
        add_done = False
        move_done = False

        if conf >= args.apply_threshold:
            # Apply safe actions
            if action in ("MOVE_MAPPED_TRAIN_QUESTION", "MOVE_AND_ADD") and mapped_idxs and move_tag:
                # move all exact-matching mapped rows
                for idx in mapped_idxs:
                    train_df.loc[idx, args.train_tag_col] = move_tag
                applied_moves += 1
                move_done = True

            if action in ("ADD_TEST_AS_VARIATION", "MOVE_AND_ADD") and add_tag:
                train_df, added = add_row_if_missing(
                    train_df,
                    qcol=args.train_question_col,
                    tcol=args.train_tag_col,
                    question=test_q,
                    tag=add_tag
                )
                if added:
                    applied_adds += 1
                add_done = added

            applied = move_done or add_done

        logs.append({
            "row_i": i,
            "test_question": normalize_text(test_q),
            "mapped_train_question": normalize_text(mapped_q),
            "expected_tag": expected,
            "predicted_tag": predicted,
            "mapped_train_current_tag": mapped_current_tag,
            "llm_action": action,
            "llm_add_tag": add_tag,
            "llm_move_tag": move_tag,
            "llm_confidence": conf,
            "applied": applied,
            "applied_move": move_done,
            "applied_add": add_done,
            "rationale": decision["rationale"],
            "extra_notes": decision["extra_notes"],
        })

    # Save outputs
    train_df.to_csv(args.out_train, index=False)
    pd.DataFrame(logs).to_csv(args.out_log, index=False)

    print(f"Done. Saved updated train: {args.out_train}")
    print(f"Saved fix log: {args.out_log}")
    print(f"Applied moves: {applied_moves}, applied adds: {applied_adds}")


if __name__ == "__main__":
    main()
