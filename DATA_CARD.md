# Data card: repository synthetic samples

## Purpose and source

The JSONL files under `data/` were authored as small synthetic examples to test schemas,
training, evaluation, negative mining, and retrieval. They do not contain collected users,
personal information, licensed third-party corpora, or representative natural language.
They are available under the repository Apache-2.0 license.

## Schemas and preprocessing

Samples include positive pairs, triplets, similarity-scored pairs, retrieval records,
queries with known positive IDs, and documents. Readers normalize repeated whitespace,
reject null/blank values, reject duplicate record IDs, and reject positives listed as
negatives. Splits and shuffles are seeded.

## Bias, privacy, and limitations

The examples are short English sentences about benign topics. They omit nearly every
language, dialect, domain, demographic, long-document pattern, ambiguity, and adversarial
case. Their simplicity can create misleadingly optimistic behavior. They must not be used
to measure quality, fairness, privacy, or deployment readiness.

## Production data guidance

Document provenance, consent, license, retention, deletion, sensitive attributes,
deduplication, leakage controls, poisoning defenses, annotator process, domain distribution,
and versioned train/validation/test splits. Remove secrets and unnecessary personal data.
Maintain every known positive per query so miners cannot create false negatives.
