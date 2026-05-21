# OVIR Brainstorm Synthesis

## Resolution

OVIR should be framed as an amortized inference runtime, not as a general transformer replacement. The correct wedge is repeated short-query inference over stable long contexts: diligence data rooms, contract review, support archives, large repositories, compliance corpora, scientific literature sets, and financial evidence packs.

## Source Integration

1. The research plan supplies the implementable spine: BigBird-style sparse attention, PEER-style product-key expert routing, ONNX graph identity, ETS caching, FragmentWorker GenServers, Supervisors, fallbacks, receipts, spot checks, and multi-node BEAM execution.

2. The landing page had the right runtime thesis, but it underweighted the workload wedge. The updated page moves workloads above architecture and makes the acceptance rule explicit: sparse answer only when confidence, citation coverage, and conflict checks clear the threshold.

3. The runtime simulation is useful, but its cost model needed correction. Fresh compute should sum per-fragment work rather than square all missed tokens together. That better matches a fragment-worker runtime.

## Positioning

OVIR.net: a runtime network for inference that repeats.

- It fragments one stable long context.
- It routes each short query to likely evidence fragments.
- It runs sparse expert or attention work only on selected fragments.
- It caches by content hash and model graph hash.
- It escalates low-confidence cases.
- It records receipts for audit and recomputation.

## Build Priorities

1. Make the toy benchmark falsifiable.

2. Keep the first benchmark deliberately small: 200 tokens, 20 fragments, 64 experts, top-2 product-key routing, 100 repeated queries.

3. Treat every shortcut as conditional. The core tests should track cache hit rate, fallback rate, expert forward-call reduction, output shape, and receipt validity.

4. Do not add a P2P protocol, GPU backend, durable cache, or real embedding model until the Week 5 integration test passes.

## Site Changes Implemented

1. Reframed the hero around repeated inference rather than generic long-context inference.

2. Added workload-first section with six concrete wedge categories.

3. Added runtime loop with ingest, fragment, route, sparse compute, cache, fallback, and verify.

4. Connected BigBird, PEER, ONNX, and Elixir/OTP as separate architecture roles.

5. Streamlined references.

6. Corrected simulation math and added fallback receipt state.
