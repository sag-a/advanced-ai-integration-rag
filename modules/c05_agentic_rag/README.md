# Lesson 5: Agentic RAG

Suggested teaching time: 50 minutes. Teach from [lesson.html](lesson.html); press
**P** to present one section per screen.

## Concept

Retrieval becomes a bounded research job. A Deep Agents coordinator reads a topic
skill, searches, reads exact clauses, records what is still missing, may delegate
one topic to a constrained helper, and finally selects passage IDs. It never
writes the answer; the shared generator from lesson 3 does. `ResearchGuard`
enforces every budget and tool permission in code.

Code: `architectures/agentic_rag/retrieval.py` (agent setup), `tools.py`,
`guard.py`, `options.py`, `skills/*/SKILL.md`, `templates/research.j2`.

## Three commands

```bash
./rag ask --process --architecture agentic_rag "Πότε λήγουν οι Καρδιές;"
./rag ask --architecture agentic_rag "How do I join Magenta Moments, and can I transfer my Hearts?"
./rag ask --architecture agentic_rag --option max_searches=1 "How do I join Magenta Moments, and can I transfer my Hearts?"
```

## What to look for

- The order of events: skill read, searches, clause reads, assessment, selection.
- The two-topic question reads two skills and searches at least twice. It may delegate
  one topic to the helper; that is the model's choice and varies between runs.
- With `max_searches=1`, the two-topic question's second search exhausts the budget and
  the turn is rejected. No partial answer is accepted. A one-topic question would
  finish within the budget.
- It selects fewer passages than Base RAG's fixed six, at the cost of more model
  calls and latency. For simple questions Base RAG may be the better choice.

## Instructor notes

- [MEASUREMENTS.md](instructor_notes/MEASUREMENTS.md): the initial failure (refusing
  without selecting evidence), the one-reminder fix, and why abstentions lower the
  faithfulness and citation-support scores.
