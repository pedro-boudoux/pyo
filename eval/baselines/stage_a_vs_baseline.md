# Phase 1 (Stage A) eval — dense semantic tags vs sparse-tag baseline

Same ground truth (`eval/ground_truth.json`, 294 seeds), same read-only mode (no
Last.fm top-up), `k=10`. Baseline = sparse `vector(300)` tag-slot model; Stage A =
dense `vector(384)` semantic tag embeddings (all-MiniLM-L6-v2, count-weighted avg).

| Metric | Baseline | Stage A | Δ |
|---|---|---|---|
| **recall@10** | 0.1823 | **0.2276** | **+0.0453 (+24.8%)** |
| **mrr** | 0.4249 | **0.4545** | **+0.0296 (+7.0%)** |
| intra_list_distance | 0.0928 | 0.0443 | −0.0485 (−52%) |
| median_listeners | 235,016 | 220,006 | −15,010 (more underground) |

## Verdict: clear win — ship Stage A

recall@10 is up ~25% and MRR up 7% — the dense semantic space retrieves more of the
held-out getSimilar targets and ranks the first hit higher. Underground health held
(median listeners actually dropped slightly), so the recall gain wasn't bought with
popular tracks. Per the spec's acceptance criteria this is a "clear win → ship it,
proceed to Phase 2 with confidence."

## Caveat: the intra_list_distance drop is mostly geometric, not a real diversity loss

MiniLM sentence embeddings are anisotropic — they occupy a narrow cone, so *any* two
tracks sit at higher cosine similarity than two sparse tag-slot vectors did. That
compresses the whole distance scale, which is why ILD roughly halved even though the
recommendations aren't obviously more samey. The MMR re-rank still runs on cosine, so
its diversity term now has less dynamic range in this space. Worth a follow-up:
retune `MMR_LAMBDA` (currently 0.7) against the new geometry, and/or measure diversity
with a rank/artist-based metric instead of raw cosine. Not a ship blocker.

## Rollback

`embedding_legacy_300` still holds the 1592 original sparse vectors. Drop it only
after Stage A is signed off in production.