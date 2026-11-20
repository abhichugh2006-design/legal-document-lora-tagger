# Evaluation: GPT-2 + LoRA legal clause tagger

Test set: 234 clauses written from **held-out templates** (phrasings never seen in training).

## Accuracy

| Model | Accuracy | Macro-F1 |
|---|---|---|
| GPT-2, tag tokens added, no fine-tuning (baseline) | 52.6% | 0.431 |
| GPT-2 + LoRA adapter (unmerged) | 93.2% | 0.930 |
| GPT-2 with LoRA merged into weights | 93.2% | 0.930 |

Merged and unmerged models agree on 100.0% of test clauses.

## Per-tag results

| Tag | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `termination` | 1.00 | 1.00 | 1.00 | 24 |
| `confidentiality` | 1.00 | 0.83 | 0.91 | 24 |
| `indemnification` | 0.67 | 1.00 | 0.80 | 24 |
| `governing_law` | 1.00 | 1.00 | 1.00 | 24 |
| `payment` | 1.00 | 1.00 | 1.00 | 24 |
| `limitation_of_liability` | 1.00 | 0.50 | 0.67 | 24 |
| `public_records` | 0.86 | 1.00 | 0.92 | 24 |
| `insurance` | 1.00 | 1.00 | 1.00 | 24 |
| `non_discrimination` | 1.00 | 1.00 | 1.00 | 24 |
| `force_majeure` | 1.00 | 1.00 | 1.00 | 18 |

## Confusion matrix (rows = true tag, columns = predicted)

| true \ pred | termin | confid | indemn | govern | paymen | limita | public | insura | non_di | force_ |
|---|---|---|---|---|---|---|---|---|---|---|
| `termination` | 24 | . | . | . | . | . | . | . | . | . |
| `confidentiality` | . | 20 | . | . | . | . | 4 | . | . | . |
| `indemnification` | . | . | 24 | . | . | . | . | . | . | . |
| `governing_law` | . | . | . | 24 | . | . | . | . | . | . |
| `payment` | . | . | . | . | 24 | . | . | . | . | . |
| `limitation_of_liability` | . | . | 12 | . | . | 12 | . | . | . | . |
| `public_records` | . | . | . | . | . | . | 24 | . | . | . |
| `insurance` | . | . | . | . | . | . | . | 24 | . | . |
| `non_discrimination` | . | . | . | . | . | . | . | . | 24 | . |
| `force_majeure` | . | . | . | . | . | . | . | . | . | 18 |

## Efficiency

| | LoRA | Full fine-tuning |
|---|---|---|
| Trainable parameters | 302,592 (0.243%) | 124,447,488 (100%) |
| Saved weights on disk | 1.22 MB | 498 MB |

LoRA trains **411x fewer parameters**: 294,912 in the rank-8 adapters on attn.c_attn plus 7,680 in the new tag embeddings.

| Inference | ms per clause |
|---|---|
| Unmerged (extra LoRA matmuls) | 6.19 |
| Merged | 5.88 |

Device: mps:0.

## Mistakes

Errors by held-out template (tag/template number): `limitation_of_liability/9` 12, `confidentiality/9` 4

5 of 16 mistakes fall below the 60% review threshold, so the `tag` command would flag them for a clerk.

- true `confidentiality`, predicted `public_records` (61%): Any non-public material furnished by the Public Works Department shall be treated as proprietary, kept secure, and shared only with personnel who need it to carry out this Lease.
- true `confidentiality`, predicted `public_records` (58%): Any non-public material furnished by the City shall be treated as proprietary, kept secure, and shared only with personnel who need it to carry out this Grant Agreement.
- true `confidentiality`, predicted `public_records` (65%): Any non-public material furnished by the Public Works Department shall be treated as proprietary, kept secure, and shared only with personnel who need it to carry out this Interlocal Agreement.
- true `confidentiality`, predicted `public_records` (57%): Any non-public material furnished by the School District shall be treated as proprietary, kept secure, and shared only with personnel who need it to carry out this Ordinance.
- true `limitation_of_liability`, predicted `indemnification` (73%): Recovery by the Supplier for any breach by the Municipality is restricted to direct damages, and the maximum exposure of either party is the contract value.
- true `limitation_of_liability`, predicted `indemnification` (70%): Recovery by the Consultant for any breach by the Water District is restricted to direct damages, and the maximum exposure of either party is the contract value.
- true `limitation_of_liability`, predicted `indemnification` (75%): Recovery by the Grantee for any breach by the Water District is restricted to direct damages, and the maximum exposure of either party is the contract value.
- true `limitation_of_liability`, predicted `indemnification` (67%): Recovery by the Service Provider for any breach by the Public Works Department is restricted to direct damages, and the maximum exposure of either party is the contract value.
- true `limitation_of_liability`, predicted `indemnification` (55%): Recovery by the Supplier for any breach by the Authority is restricted to direct damages, and the maximum exposure of either party is the contract value.
- true `limitation_of_liability`, predicted `indemnification` (60%): Recovery by the Grantee for any breach by the Housing Authority is restricted to direct damages, and the maximum exposure of either party is the contract value.
- true `limitation_of_liability`, predicted `indemnification` (91%): Recovery by the Licensee for any breach by the Public Works Department is restricted to direct damages, and the maximum exposure of either party is the contract value.
- true `limitation_of_liability`, predicted `indemnification` (54%): Recovery by the Service Provider for any breach by the State Agency is restricted to direct damages, and the maximum exposure of either party is the contract value.
- true `limitation_of_liability`, predicted `indemnification` (66%): Recovery by the Supplier for any breach by the School District is restricted to direct damages, and the maximum exposure of either party is the contract value.
- true `limitation_of_liability`, predicted `indemnification` (61%): Recovery by the Service Provider for any breach by the City is restricted to direct damages, and the maximum exposure of either party is the contract value.
- true `limitation_of_liability`, predicted `indemnification` (67%): Recovery by the Grantee for any breach by the Transit Authority is restricted to direct damages, and the maximum exposure of either party is the contract value.
