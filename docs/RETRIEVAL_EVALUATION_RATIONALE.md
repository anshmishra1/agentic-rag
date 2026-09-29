# Retrieval evaluation: reasoning, evidence, and decision record

This explains what was tested, why each test was run, what it established,
and what it did **not** establish. It is intended to make the project work
auditable and easy to discuss in an interview. The application retrieves
passages from one selected document using dense search plus document-local
BM25, fuses them with RRF, reranks with a local cross-encoder, then routes
to generation, semantic grading, or query rewriting. Answer verification
checks claims against cited evidence.

## The questions we separated

| Question | Measurement | Why it matters |
| --- | --- | --- |
| Can the selected document answer the question? | Human-reviewed document/question label. | Tests routing and abstention without assuming every same-topic document is relevant. |
| Did retrieval find a sufficient passage? | Source passage or chunk label and rank. | Document-level labels cannot prove that the chunks sent to the answer model contain the answer. |
| Did reranking help? | The same candidates' ranks before and after the cross-encoder. | A high final score alone cannot attribute success or failure to reranking. |
| Did routing choose a useful next step? | Generate, grade, rewrite, or abstain per case. | A relevant passage can be wasted by an unnecessary rewrite; a weak passage can cause an unsupported answer. |
| Is the answer supported? | Claim/citation review and grounding outcome. | A `grounded` flag does not by itself establish that cited passages support every claim. |

The offline unit tests check contracts and failure paths. Live retrieval
and answer checks measure system behavior. Neither substitutes for the other.

## Evaluation sequence and what we learned

1. **Constructed reviewed document pairs.** The 80-question legacy set was
   narrowed to 46 reviewed question/document pairs across six exact local
   PDFs (35 development, 11 holdout). Source and alternate-document pairs
   were checked against the PDFs; two proposed negatives were corrected to
   positives because the alternate book also answered them. We excluded
   ambiguous and cross-document cases. This established credible
   document-level labels, not gold chunk IDs.

2. **Measured retrieval scores without answer-model calls.** The six-PDF
   snapshot showed that a development-only candidate top-score floor of
   about 0.0696 separated these document labels, and that fixed candidate
   classified the 11 holdout document labels correctly (6 positive, 5
   negative). We did **not** adopt the floor: the positive `q033-other`
   document had a 0.1313 top score, but its retrieved passage did not clearly
   answer the question. Score separation at document level is not proof of
   passage sufficiency. Cross-encoder scores are ranking signals, not
   calibrated probabilities.

3. **Replayed routing decisions.** Four of 25 document-positive cases were
   rewritten on the first pass. For three (`q001-source`, `q031-source`, and
   `q038-source`), inspected passages supported the question but similar
   high scores made the score-gap rule look weak. We changed that
   above-floor, close-candidate route to semantic grading. The 21 negative
   cases and the weak-passage `q033-other` retained their prior route. This
   corrected a policy decision without changing numeric thresholds.

4. **Checked the full graph with bounded live questions.** `q031-source`
   reached grading and generated an answer, but its first live answer was
   called grounded despite uncited claims; the verifier had seen uncited
   retrieved passages. `q051-other` safely abstained. We then changed
   verification to examine only cited passages and fail closed on missing
   citations or uncited factual list items. The 87-test offline suite and a
   saved-answer probe covered that rule. In a later authorized single-case
   rerun, `q031-source` ended with a supported, cited answer after an empty
   first verifier response triggered the one permitted correction. This
   confirmed the fail-closed path, but did not live-test semantic rejection
   of uncited claims; that specific behavior remains offline-tested.

5. **Reviewed final passages.** A deliberately selected 15-case diagnostic
   review confirmed sufficient content in ten positive cases, found no
   support in the inspected content for two negative cases, and left three
   unresolved. Because selection followed observed results, these counts
   are examples, **not** recall or accuracy estimates. The saved snapshot
   held only final ranks, so it could not show whether the reranker improved
   the original RRF ordering.

6. **Investigated ID mismatches before further tuning.** Rebuilding chunks
   from exact PDF bytes matched 155 of 230 final retrieved content hits; 75
   did not match. Four sampled unmatched IDs existed in Pinecone and matched
   hashes of their stored text. One stored Hands-On LLM passage extended a
   current local chunk (1,432 versus 1,278 characters). A read-only audit of
   every evaluation PDF then showed that all current content IDs exist, but
   five documents also retain many older IDs. Source history changed
   chunking from character-based to token-based; the ingestion pipeline
   upserted new IDs without pruning prior IDs. The exact ingestion event
   that created each old vector is not recorded, but the stale supersets and
   upsert-only code explain the contamination.

## Read-only index alignment audit (2026-09-29)

`Current content` counts unique IDs from the current PDF loader/chunker.
Repeated identical chunks can make this smaller than the raw chunk count.
The exact-PDF hashes, local chunk counts, Pinecone ID lists, and registry
chunk counts were checked. No index mutation occurred.

| PDF | Current content IDs | Indexed content IDs | Extra content IDs | Extra overview/legacy IDs | Planned keep | Planned delete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AI Engineering | 384 | 1,358 | 974 | 4 | 385 | 978 |
| CS229 | 699 | 1,983 | 1,284 | 3 | 700 | 1,287 |
| Explainable AI | 173 | 569 | 396 | 3 | 174 | 399 |
| ML eBook | 79 | 79 | 0 | 0 | 80 | 0 |
| DevOps Troubleshooting | 587 | 1,763 | 1,176 | 3 | 588 | 1,179 |
| Hands-On LLM | 931 | 2,668 | 1,737 | 3 | 932 | 1,740 |
| **Total** | **2,853** | **8,420** | **5,567** | **16** | **2,859** | **5,583** |

The active overview for each document was identified by comparing its stored
BM25 sparse vector to the current PostgreSQL BM25 state. The maintenance
plan is local and ignored by Git at
`logs/evaluation/20260929T111320Z/cleanup_plan.json`; its SHA-256 is
`4f43b0444cfc8621c64ac794f198d001c338af0e61064375974cd59091b78bc0`.
The plan contains IDs but no passage text. Its default command is read-only.
**No vectors have been deleted.** One NLTK tokenizer/stopword download
occurred inside a disposable diagnostic container; there were no Groq calls
or ingestion requests during this audit.

## Repair and completion criteria

The code now keeps the IDs from a completed upsert and prunes older IDs only
after the matching BM25 registry state is saved. Tests cover deletion
batching, document isolation, and ordering when the overview, vector upsert,
or registry step fails. The full offline suite passed 95 tests with Docker
networking disabled and no private `.env` mount. The separate maintenance
command checks exact PDF hashes,
registry chunk counts, all current IDs, and a unique active overview. Its
apply mode additionally requires the reviewed plan digest and recomputes
the entire plan before any deletion. Apply remains pending review and
specific authorization because it deletes 5,583 existing Pinecone IDs.

After cleanup, confirm one current overview and the expected current
content IDs per document. Then make a small, independent passage-label set
from PDF source spans, capture RRF and cross-encoder ranks for the **same**
candidates, and compare hit@k and reciprocal rank on development cases.
Keep holdout questions untouched until the choice is fixed. If reranking
helps, retain it and move on; if it demonstrably hurts, make one targeted
change and repeat the comparison. A final bounded app check should verify
that routing and cited answers still behave safely. This is the exit gate
for the current retrieval-quality stage, not a claim that all future PDFs
will behave identically.

## Interview explanation

The key lesson is to isolate failure layers. A low or high similarity score
does not prove answer quality. The first evaluation exposed a routing issue;
the live answer exposed a citation-verification issue; passage review then
revealed an index-consistency issue that could distort reranker results.
We fixed or bounded each issue at its own layer, used development data for
decisions and holdout data for checking them, and did not tune thresholds
until passage-level evidence could justify it.
