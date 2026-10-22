# Lab 5: A semantic cache, and why the most similar questions are the most dangerous

After lesson 3 · about 35 minutes · offline: the local Nomic encoder, no API key, no Docker

## The problem

Many users ask the same thing in different words. A **semantic cache** stores validated
answers with the question's embedding; when a new question is close enough, it reuses
the answer and skips retrieval and the LLM. It's the first building block of the
planned `c06_system1_rag` lesson. The risk is a **false hit**: serving a confident
answer to a question that only *looks* the same.

## Background

- Cosine similarity of normalized vectors is their dot product.
- A threshold decides "close enough". Higher means safer and fewer reuses.
- **Store only validated answers.** A refusal or a clarification depends on the
  conversation; reusing it for someone else is unsafe.
- Embeddings blur exact details: "24 S, **12**-month contract" and "24 S, **18**-month
  contract" are almost the same vector with different prices. A **guard** can veto a
  match for reasons the vectors can't see.

## Your tasks

Edit [`starter.py`](starter.py): `normalize`, `cosine`, `numbers_match`,
`SemanticCache.add` and `SemanticCache.lookup` (including falling through to the next
entry when the guard vetoes the best one).

## Check and measure

```bash
uv run --all-extras pytest exercises/lab05_semantic_cache
uv run --all-extras python -m exercises.lab05_semantic_cache.measure
```

The measure script caches six answered questions, then tries six paraphrases that should
reuse an answer and five near-misses that must not. Reference results with the course
encoder:

| Probe | Closest cached question | Similarity |
| --- | --- | ---: |
| How can I sign up for Magenta Moments? | How do I join Magenta Moments? ✓ | 0.908 |
| When do Hearts expire? | *Can I transfer my Hearts…?* ✗ (should be the Greek expiry question) | 0.473 |
| **24 S, 18-month contract** price? | 24 S, 12-month contract price ✗ | **0.931** |
| **24 XL with 420 minutes**? | 24 XL with 500 minutes ✗ | **0.897** |
| How do I **leave** Magenta Moments? | How do I **join**…? ✗ | 0.826 |

| Guard | Threshold | Correct reuse | Wrong answer served |
| --- | ---: | ---: | ---: |
| none | 0.70 | 3/6 | 3 |
| none | 0.90 | 1/6 | 1 |
| none | 0.95 | 0/6 | 0 |
| numbers | 0.70 | 3/6 | 1 |
| numbers | 0.85 | 1/6 | **0** |

The near-misses score *higher* than most true paraphrases. Without a guard, the only
threshold that never serves a wrong answer reuses nothing. The numbers guard blocks the
420/500 and 12/18 cases; "leave" vs "join" still gets through below 0.83.

## Discuss

1. Why do near-misses score higher than paraphrases? (Think about what changes between
   them: one word versus every word.)
2. "When do Hearts expire?" doesn't even match the Greek "Πότε λήγουν οι Καρδιές;". Is
   that a reason to cache per language?
3. Write a guard for "leave" vs "join". Is a word list a good idea? What else could veto
   a match (a classifier, the answer's cited clauses)?
4. Each wrong answer served is worse than a miss. With these numbers, would you turn the
   cache on for this course? At what threshold?
5. What has to happen to the cache when the PDF or the answer prompt changes?
