# OVIR Research Plan
## Extending BigBird + PEER to Distributed Cached Inference with Elixir + Nx

**Target:** early undergraduate. No Elixir or ML framework experience assumed.  
**Duration:** 5 weeks + a short orientation, 3–5 sessions per week, ~1–2 hours each.  
**Goal:** build working intuition for how sparse attention (BigBird) and sparse expert retrieval
(PEER) can be composed into the OVIR.net pattern — many short queries against one stable long
context — using Elixir/Nx for numerics and OTP for distribution, caching, and fault tolerance.

---

## Why Elixir instead of Python + Ray

Python simulates distribution. Elixir is built on it. The mapping is direct:

| OVIR concept | Elixir / OTP primitive |
|---|---|
| Fragment worker | `GenServer` process |
| Worker pool | `Supervisor` + `Registry` |
| Content-addressed cache | ETS table (concurrent, shared memory) |
| Fallback escalation | supervisor-managed fallback worker |
| Receipt log | append-only ETS / `Logger` |
| Multi-machine distribution | connected BEAM nodes (built in) |
| Fault recovery | supervisor restart strategy (built in) |

Nx provides tensor operations (dot products, softmax, matrix multiply) with a JIT-compilable
`defn` macro that can target CPU via EXLA. The math is the same as in the NumPy version;
the execution and distribution model is qualitatively better.

---

## Orientation — Elixir and Nx Basics (3–4 sessions before Week 1)

### O.1 — Set up the project

```bash
mix new ovir --module OVIR
cd ovir
```

Add to `mix.exs`:

```elixir
{:nx, "~> 0.7"},
{:exla, "~> 0.7"}   # XLA backend for compiled defn
```

```bash
mix deps.get
```

Set the default backend in `config/config.exs`:

```elixir
config :nx, default_backend: EXLA.Backend
```

Open `iex -S mix` and verify:

```elixir
Nx.tensor([1, 2, 3]) |> Nx.multiply(2)
# #Nx.Tensor<s32[3] [2, 4, 6]>
```

---

### O.2 — Nx tensor basics

Key operations you will use every week. Run each in `iex` and read the output shape.

```elixir
# Create tensors
a = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
b = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])

# Dot product (matrix multiply)
Nx.dot(a, b)

# Element-wise ops
Nx.add(a, b)
Nx.multiply(a, 2.0)

# Softmax over last axis
Nx.softmax(Nx.tensor([1.0, 2.0, 3.0]))

# Shape and type
Nx.shape(a)     # {2, 2}
Nx.type(a)      # {:f, 32}

# Reduction
Nx.mean(a, axes: [0])   # mean over rows → shape {2}
```

**Checkpoint:** you should be able to write a function that takes a 2D tensor and returns the
row-wise softmax without looking anything up.

---

### O.3 — `defn` for compiled numerical code

`defn` is Nx's version of a compiled numerical function. Write it in a module, call it like
a normal function. The JIT compilation happens on first call.

```elixir
defmodule OVIR.Math do
  import Nx.Defn

  defn softmax(x) do
    e = Nx.exp(x - Nx.reduce_max(x))
    e / Nx.sum(e)
  end

  defn dot_scores(queries, keys) do
    # queries: {n, d}   keys: {m, d}
    # returns: {n, m}
    Nx.dot(queries, [1], keys, [1])
  end
end
```

Test in `iex`:

```elixir
OVIR.Math.softmax(Nx.tensor([1.0, 2.0, 3.0]))
# should sum to 1.0
```

**Unit test (ExUnit):**

```elixir
test "softmax sums to 1" do
  result = OVIR.Math.softmax(Nx.tensor([1.0, 2.0, 3.0]))
  assert_in_delta Nx.to_number(Nx.sum(result)), 1.0, 1.0e-5
end
```

---

### O.5 — ONNX and Ortex (one session)

ONNX is the portable graph format that makes expert fragments distributable. An ONNX model
is a file — it can be hashed, versioned, sent over the network, and executed by any
ONNX-compatible runtime regardless of what framework produced it. In OVIR.net, this is
what makes two workers with identical expert weights *verifiably identical*: same ONNX
bytes → same SHA-256 → same cache key.

In Elixir, ONNX models run via `Ortex`.

Add to `mix.exs`:

```elixir
{:ortex, "~> 0.1"}
```

**Exercise O.5.1 — Export a tiny MLP to ONNX from Python, load it in Elixir.**

In Python (run once, outside the Mix project):

```python
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
dummy = torch.randn(1, 4)
torch.onnx.export(model, dummy, "tiny_expert.onnx",
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}})
```

In Elixir:

```elixir
model = Ortex.load("tiny_expert.onnx")
input = Nx.random_normal({1, 4})
{output} = Ortex.run(model, {input})
Nx.shape(output)   # {1, 4}
```

**Unit test:** output shape must be `{1, 4}`. Running the same input twice must return
identical output (ONNX Runtime is deterministic on CPU).

---

**Exercise O.5.2 — Hash the model file as its identity.**

```elixir
defmodule OVIR.OnnxModel do
  defstruct [:path, :model, :content_hash]

  def load(path) do
    content_hash =
      File.read!(path)
      |> then(&:crypto.hash(:sha256, &1))
      |> Base.encode16(case: :lower)
    %__MODULE__{path: path, model: Ortex.load(path), content_hash: content_hash}
  end

  def run(%__MODULE__{model: m}, input_tensor) do
    {output} = Ortex.run(m, {input_tensor})
    output
  end
end
```

**Key insight:** the content hash is derived from the *model graph bytes*, not from any
runtime state. Two workers that load the same `.onnx` file on different machines will have
the same `content_hash`. That hash is what OVIR.net uses as the stable identity of a
fragment — it is what goes into receipts and cache keys.

---

### O.4 — GenServer basics (two sessions)

This is the structural primitive for every worker in the plan. Build a counter to get the
pattern down before attaching any ML logic.

```elixir
defmodule OVIR.Counter do
  use GenServer

  def start_link(init \\ 0), do: GenServer.start_link(__MODULE__, init, name: __MODULE__)
  def increment, do: GenServer.cast(__MODULE__, :increment)
  def value, do: GenServer.call(__MODULE__, :value)

  @impl true
  def init(n), do: {:ok, n}

  @impl true
  def handle_cast(:increment, n), do: {:noreply, n + 1}

  @impl true
  def handle_call(:value, _from, n), do: {:reply, n, n}
end
```

Run it:

```elixir
{:ok, _} = OVIR.Counter.start_link()
OVIR.Counter.increment()
OVIR.Counter.increment()
OVIR.Counter.value()   # 2
```

**Checkpoint:** you understand the call/cast distinction and can trace a message through
`handle_call` / `handle_cast` back to the caller.

---

## Week 1 — BigBird: Sparse Attention with Nx

**Problem:** full attention over N tokens costs O(N²) memory and compute. BigBird cuts that
to O(N) by restricting each token to attend only to: a local window, a random sample, and
a small set of global tokens.

### Day 1 — Full attention in `defn`

**Exercise 1.1 — Compute a 5×5 attention matrix by hand.**

Given 5 tokens represented as 2D vectors, work out the scaled dot-product attention matrix
on paper:

```
score(i, j) = dot(Q_i, K_j) / sqrt(d_k)
attn_row_i  = softmax(score_row_i)
output_i    = sum_j attn(i, j) * V_j
```

Write out the full 5×5 score grid. Every entry is nonzero. That is the problem PEER and
BigBird both address from different angles.

---

**Exercise 1.2 — Implement full attention in `defn`.**

```elixir
defmodule OVIR.Attention do
  import Nx.Defn

  defn full_attention(q, k, v) do
    # q: {n, d}  k: {n, d}  v: {n, dv}
    d_k = Nx.axis_size(k, 1) |> Nx.tensor() |> Nx.sqrt()
    scores = Nx.dot(q, [1], k, [1]) / d_k    # {n, n}
    weights = Nx.softmax(scores, axis: 1)     # {n, n}
    Nx.dot(weights, [1], v, [0])              # {n, dv}
  end
end
```

**Unit tests:**

```elixir
test "attention weights sum to 1 per row" do
  {n, d} = {8, 4}
  q = Nx.random_normal({n, d})
  k = Nx.random_normal({n, d})
  v = Nx.random_normal({n, d})
  {weights, _output} = OVIR.Attention.full_attention_with_weights(q, k, v)
  row_sums = Nx.sum(weights, axes: [1])
  assert Nx.all_close(row_sums, Nx.broadcast(1.0, {n}), atol: 1.0e-5) |> Nx.to_number() == 1
end
```

---

### Day 2 — BigBird sparsity mask

**Exercise 1.3 — Build a BigBird mask as an Nx boolean tensor.**

The mask is a `{n, n}` boolean tensor. `mask[i][j] == 1` means token i attends to token j.
Three components: local window, random, global.

```elixir
defmodule OVIR.Attention.Mask do
  def bigbird(n, window \\ 3, n_random \\ 2, n_global \\ 1) do
    base = Nx.broadcast(0, {n, n})

    # Local window
    local = for i <- 0..(n-1), j <- 0..(n-1),
                abs(i - j) <= div(window, 2), do: {i, j}

    # Global: first n_global rows and columns
    global = for i <- 0..(n-1), g <- 0..(n_global-1) do
      [{i, g}, {g, i}]
    end |> List.flatten()

    # Random: for each row, pick n_random non-local, non-global positions
    random = for i <- 0..(n-1) do
      candidates =
        for j <- 0..(n-1),
            abs(i - j) > div(window, 2),
            j >= n_global,
            do: {i, j}
      Enum.take_random(candidates, n_random)
    end |> List.flatten()

    all_ones = (local ++ global ++ random) |> Enum.uniq()
    indices = Nx.tensor(all_ones)
    # set those positions to 1
    Nx.indexed_put(base, indices, Nx.broadcast(1, {length(all_ones)}))
    |> Nx.as_type({:u, 8})
  end
end
```

**Unit tests:**

```elixir
test "diagonal is always 1 (self-attention)" do
  mask = OVIR.Attention.Mask.bigbird(20)
  diag = for i <- 0..19, do: mask[i][i] |> Nx.to_number()
  assert Enum.all?(diag, &(&1 == 1))
end

test "global rows are fully attended" do
  mask = OVIR.Attention.Mask.bigbird(20, 3, 2, 1)
  row_0_sum = Nx.sum(mask[0]) |> Nx.to_number()
  assert row_0_sum == 20
end

test "mask is genuinely sparse" do
  mask = OVIR.Attention.Mask.bigbird(20)
  total = Nx.sum(mask) |> Nx.to_number()
  assert total < 20 * 20
end
```

---

### Day 3 — Sparse attention and the approximation gap

**Exercise 1.4 — Apply mask, compare to full attention.**

Set masked positions to -∞ before softmax:

```elixir
defn sparse_attention(q, k, v, mask) do
  d_k = Nx.axis_size(k, 1) |> Nx.tensor() |> Nx.sqrt()
  scores = Nx.dot(q, [1], k, [1]) / d_k
  neg_inf = Nx.broadcast(-1.0e9, Nx.shape(scores))
  masked_scores = Nx.select(mask, scores, neg_inf)
  weights = Nx.softmax(masked_scores, axis: 1)
  Nx.dot(weights, [1], v, [0])
end
```

Run both `full_attention` and `sparse_attention` on a sequence of length 20. Compute the
mean absolute difference per token. Plot (or print) the difference. Write the answer to:
which token positions lose the most? (Answer: those with no global token in their attention
set and whose relevant context falls outside the window.)

---

### Day 4 — Complexity measurement

**Exercise 1.5 — Count nonzero entries vs. N.**

For `n` in `[16, 64, 256, 1024]`, compute `Nx.sum(bigbird(n))` and compare to `n*n`.
Print the ratio. At what N does sparse attention use less than 10% of the memory of full
attention with these settings?

Write a property test: for all N in the above list, `sum(mask) / N` should be strictly
less than `N` (sub-quadratic growth).

---

## Week 2 — PEER: Product-Key Expert Retrieval with Nx

**Problem:** a dense feedforward layer activates all H neurons for every input. PEER replaces
this with sparse retrieval over E tiny experts, each with its own key. Only the top-k
experts (by key similarity) run for any given input.

### Day 5 — Toy expert pool

**Exercise 2.1 — Build 8 experts as ONNX models, route one query.**

Each expert is a tiny MLP exported to ONNX (use the Python snippet from O.5.1, varying
the random seed per expert). Load each via `OVIR.OnnxModel.load/1`. The expert's key
vector is stored separately — it is used for routing only, not baked into the ONNX graph.

```elixir
defmodule OVIR.Expert do
  defstruct [:id, :onnx, :key]

  def load(id, onnx_path) do
    %__MODULE__{
      id:   id,
      onnx: OVIR.OnnxModel.load(onnx_path),
      key:  Nx.random_normal({4}, seed: id)   # stable per-expert key
    }
  end

  def forward(%__MODULE__{onnx: m}, x) do
    OVIR.OnnxModel.run(m, Nx.new_axis(x, 0))
    |> Nx.squeeze()
  end

  def content_hash(%__MODULE__{onnx: m}), do: m.content_hash
end
```

Route function is unchanged — the scoring is still dot-product over key vectors:

```elixir
defmodule OVIR.ExpertRouter do
  def route(x, experts, query_proj, k \\ 2) do
    query = Nx.dot(query_proj, x)
    keys  = experts |> Enum.map(& &1.key) |> Nx.stack()
    scores = Nx.dot(keys, query)
    top_k_indices = Nx.argsort(scores, direction: :desc) |> Nx.slice([0], [k])
    top_k_scores  = Nx.softmax(Nx.take(scores, top_k_indices))

    output =
      top_k_indices
      |> Nx.to_flat_list()
      |> Enum.with_index()
      |> Enum.reduce(Nx.broadcast(0.0, {4}), fn {idx, i}, acc ->
           expert = Enum.at(experts, idx)
           w = top_k_scores[i] |> Nx.to_number()
           Nx.add(acc, Nx.multiply(OVIR.Expert.forward(expert, x), w))
         end)

    {output, Nx.to_flat_list(top_k_indices)}
  end
end
```

**Unit tests:**
1. `top_k_indices` has exactly `k` elements, all in `0..E-1`.
2. `OVIR.Expert.content_hash(expert)` is a 64-character hex string.
3. Two `Expert` structs loaded from the *same* `.onnx` file have equal `content_hash`
   values, even if loaded in separate calls.

---

### Day 6 — Product-key indexing

**Exercise 2.2 — Scale to 64 experts without brute force.**

Split each d-dimensional key into two halves. Build two sub-codebooks of √E vectors each.
Top-k lookup costs 2 × √E dot products instead of E.

```elixir
defmodule OVIR.ProductKeyIndex do
  import Nx.Defn

  defn lookup(query, sub_keys_1, sub_keys_2, k) do
    # query: {d}   sub_keys_*: {sqrt_e, d/2}
    d = Nx.axis_size(query, 0)
    half = div(d, 2)
    q1 = query[0..(half - 1)]
    q2 = query[half..(d - 1)]

    s1 = Nx.dot(sub_keys_1, q1)   # {sqrt_e}
    s2 = Nx.dot(sub_keys_2, q2)   # {sqrt_e}

    # outer sum: combined[i][j] = s1[i] + s2[j]
    combined = Nx.add(Nx.new_axis(s1, 1), Nx.new_axis(s2, 0))  # {sqrt_e, sqrt_e}
    flat = Nx.reshape(combined, {:auto})                         # {E}
    Nx.argsort(flat, direction: :desc) |> Nx.slice([0], [k])
  end
end
```

**Unit test:** run 100 random queries. Compare `ProductKeyIndex.lookup` against brute-force
argsort. For random Gaussian keys and queries, top-1 agreement should be > 80%.

---

**Exercise 2.3 — Complexity table.**

Number of dot products for brute-force vs. product-key, for E = 100, 10_000, 1_000_000:

| E | Brute force | Product key | Speedup |
|---|---|---|---|
| 100 | 100 | 20 | 5× |
| 10,000 | 10,000 | 200 | 50× |
| 1,000,000 | 1,000,000 | 2,000 | 500× |

Formula: speedup = E / (2√E) = √E / 2. Verify for E = 10,000: √10000 / 2 = 50. ✓

---

### Day 7 — ETS cache for expert outputs

**Exercise 2.4 — Cache expert outputs in ETS, keyed by model + input.**

ETS is Elixir's built-in in-memory key/value store. It is concurrent, shared across
processes, and requires no extra dependency. This is the native OVIR caching primitive.

The cache key is a tuple of `{onnx_content_hash, input_binary}`. This means two workers
that loaded the same ONNX file will share cached results — the model identity is part of
the key, not just the input.

```elixir
defmodule OVIR.ExpertCache do
  @table :expert_cache

  def start do
    :ets.new(@table, [:named_table, :set, :public])
  end

  def get(key), do: :ets.lookup(@table, key) |> List.first()

  def put(key, value), do: :ets.insert(@table, {key, value})

  def stats do
    %{size: :ets.info(@table, :size)}
  end
end

defmodule OVIR.CachedRouter do
  def forward(x, experts, query_proj, k \\ 2) do
    # Key combines the set of expert content hashes + the input tensor
    expert_hashes = experts |> Enum.map(&OVIR.Expert.content_hash/1) |> Enum.join("|")
    cache_key = {expert_hashes, Nx.to_binary(x)}
    case OVIR.ExpertCache.get(cache_key) do
      {^cache_key, result} ->
        {result, :hit}
      nil ->
        result = OVIR.ExpertRouter.route(x, experts, query_proj, k)
        OVIR.ExpertCache.put(cache_key, result)
        {result, :miss}
    end
  end
end
```

**Unit tests:**
1. Send the same 10 input vectors 5 times each. After the first round, all subsequent
   calls return `:hit`. Assert `OVIR.ExpertCache.stats().size == 10`.
2. Swap one expert for a different ONNX model (different content hash). Assert the cache
   misses again — the key changed because the computation graph changed.
3. Two separate `CachedRouter` instances sharing the same ETS table and loading the same
   ONNX models will share cached results. Assert the second router's first call on a
   pre-warmed input returns `:hit`.

---

## Week 3 — Connecting Sparse Attention + Sparse Experts for OVIR's Target Problem

**Problem:** 100-token document (long context), 5-token query (short). Answer the query by
looking only at relevant fragments — not the whole document.

### Day 8 — Fragment the document

**Exercise 3.1 — Chunk a document tensor into named fragments.**

```elixir
defmodule OVIR.Document do
  defstruct [:tokens, :chunks]

  def from_tensor(tokens, chunk_size \\ 10) do
    n = Nx.axis_size(tokens, 0)
    chunks =
      0..(n - 1)
      |> Enum.take_every(chunk_size)
      |> Enum.map(fn start ->
           stop = min(start + chunk_size, n) - 1
           {start, Nx.slice(tokens, [start, 0], [stop - start + 1, Nx.axis_size(tokens, 1)])}
         end)
    %__MODULE__{tokens: tokens, chunks: chunks}
  end

  def fragment_hash({_start, chunk_tensor}) do
    :crypto.hash(:sha256, Nx.to_binary(chunk_tensor))
    |> Base.encode16(case: :lower)
  end
end
```

**Unit tests:**
1. Concatenating all chunks recovers the original document.
2. Every chunk has at most `chunk_size` rows.
3. Start indices are non-overlapping and cover `0..N-1`.

---

### Day 9 — Route a query to relevant fragments

**Exercise 3.2 — Query-to-fragment scoring.**

Summarise each fragment as the mean of its token embeddings. Score the query (also mean-
pooled) against each summary via dot product. Return the top-k fragment indices. This is
BigBird-style sparse routing at fragment granularity.

```elixir
defmodule OVIR.FragmentRouter do
  import Nx.Defn

  def top_k_fragments(query_tokens, doc, k \\ 3) do
    query_vec = Nx.mean(query_tokens, axes: [0])
    scored =
      doc.chunks
      |> Enum.map(fn {start, chunk} ->
           summary = Nx.mean(chunk, axes: [0])
           score = Nx.dot(query_vec, summary) |> Nx.to_number()
           {start, score}
         end)
      |> Enum.sort_by(&elem(&1, 1), :desc)
      |> Enum.take(k)
    scored
  end
end
```

**Unit test:** construct a document where fragment at index 20 is very similar to the query
(cosine similarity > 0.9). Assert that fragment 20 appears in `top_k_fragments` results.

---

### Day 10 — Run experts only on selected fragments

**Exercise 3.3 — Process selected fragments, merge results.**

```elixir
defmodule OVIR.FragmentProcessor do
  def process(fragment_tokens, cached_router, experts, query_proj) do
    fragment_tokens
    |> Nx.to_batched(1)
    |> Enum.map(fn token ->
         token = Nx.squeeze(token)
         {{output, _experts_used}, _hit_or_miss} =
           OVIR.CachedRouter.forward(token, experts, query_proj)
         output
       end)
    |> Nx.stack()
    |> Nx.mean(axes: [0])   # fragment-level summary
  end
end
```

Chain: query → top-k fragments → per-fragment expert processing → weighted merge.

**Unit test:** processing the same fragment twice must return an identical tensor and must
increase the ETS cache hit count by `chunk_size` (one hit per token).

---

### Day 11 — End-to-end toy pipeline

**Exercise 3.4 — Wire everything together.**

```elixir
defmodule OVIR.Pipeline do
  def query(query_tokens, doc, experts, query_proj,
            chunk_size \\ 10, top_k \\ 3) do
    scored_fragments = OVIR.FragmentRouter.top_k_fragments(query_tokens, doc, top_k)

    results =
      for {start, score} <- scored_fragments do
        {_start, chunk} = Enum.find(doc.chunks, fn {s, _} -> s == start end)
        summary = OVIR.FragmentProcessor.process(chunk, experts, query_proj)
        {score, summary}
      end

    weights = results |> Enum.map(&elem(&1, 0)) |> Nx.tensor() |> Nx.softmax()

    answer =
      results
      |> Enum.with_index()
      |> Enum.reduce(fn {{_, vec}, i}, acc ->
           Nx.add(acc, Nx.multiply(vec, weights[i]))
         end)

    answer
  end
end
```

Run 20 different queries against the same 100-token document. Assert:
1. Cache hit rate increases monotonically across the 20 queries.
2. Output shape is `{d_out}` every time.
3. Total expert `forward` calls < `top_k * chunk_size * 20`.

---

## Week 4 — OTP Workers, Supervision, and Receipts

This is where Elixir separates from the Python + Ray approach. Distribution is not simulated
— it is native to the BEAM.

### Day 12 — FragmentWorker as a GenServer

**Exercise 4.1 — One GenServer per fragment.**

```elixir
defmodule OVIR.FragmentWorker do
  use GenServer

  defstruct [:fragment_id, :fragment_hash, :expert_hash, :chunk, :experts, :query_proj,
             :hits, :misses]

  def start_link({id, chunk, experts, query_proj}) do
    GenServer.start_link(__MODULE__, {id, chunk, experts, query_proj},
                         name: via(id))
  end

  def process(id, query_tokens) do
    GenServer.call(via(id), {:process, query_tokens})
  end

  defp via(id), do: {:via, Registry, {OVIR.WorkerRegistry, id}}

  @impl true
  def init({id, chunk, experts, query_proj}) do
    # Content address = hash of input tokens + hash of ONNX model graph.
    # Two workers with the same ONNX files and the same fragment will
    # compute the same cache key and share results across the cluster.
    expert_hash =
      experts
      |> Enum.map(&OVIR.Expert.content_hash/1)
      |> Enum.join("|")
      |> then(&:crypto.hash(:sha256, &1))
      |> Base.encode16(case: :lower)

    state = %__MODULE__{
      fragment_id:   id,
      fragment_hash: OVIR.Document.fragment_hash({id, chunk}),
      expert_hash:   expert_hash,
      chunk:         chunk,
      experts:       experts,
      query_proj:    query_proj,
      hits:          0,
      misses:        0
    }
    {:ok, state}
  end

  @impl true
  def handle_call({:process, _query_tokens}, _from, state) do
    # Full content address: what was processed (tokens) + how (ONNX graph)
    cache_key = {state.fragment_hash, state.expert_hash}
    case OVIR.ExpertCache.get(cache_key) do
      {^cache_key, result} ->
        {:reply, {result, :hit}, %{state | hits: state.hits + 1}}
      nil ->
        result = OVIR.FragmentProcessor.process(state.chunk, state.experts, state.query_proj)
        OVIR.ExpertCache.put(cache_key, result)
        {:reply, {result, :miss}, %{state | misses: state.misses + 1}}
    end
  end
end
```

**Unit tests:**
1. Starting two workers with different IDs and sending `process/2` to each returns correctly
   shaped tensors.
2. Sending the same query to the same worker twice: second call returns `:hit`.
3. Two workers initialized with the same ONNX paths and the same chunk produce the same
   `cache_key` tuple — verify by comparing `{state.fragment_hash, state.expert_hash}` via
   `:sys.get_state/1` on each worker PID.
4. Killing a worker with `Process.exit(pid, :kill)` and then calling `process/2` should
   raise a meaningful error (the supervisor will restart it — test that next).

---

### Day 13 — Supervision tree

**Exercise 4.2 — Put workers under a Supervisor.**

```elixir
defmodule OVIR.WorkerSupervisor do
  use Supervisor

  def start_link(worker_specs) do
    Supervisor.start_link(__MODULE__, worker_specs, name: __MODULE__)
  end

  @impl true
  def init(worker_specs) do
    children =
      [
        {Registry, keys: :unique, name: OVIR.WorkerRegistry}
        | Enum.map(worker_specs, fn spec ->
            {OVIR.FragmentWorker, spec}
          end)
      ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
```

Test fault tolerance:

```elixir
test "crashed worker restarts and can still process queries" do
  worker_pid = GenServer.whereis({:via, Registry, {OVIR.WorkerRegistry, 0}})
  Process.exit(worker_pid, :kill)
  Process.sleep(50)   # give supervisor time to restart
  new_pid = GenServer.whereis({:via, Registry, {OVIR.WorkerRegistry, 0}})
  assert new_pid != nil
  assert new_pid != worker_pid
  {result, _} = OVIR.FragmentWorker.process(0, some_query_tokens())
  assert Nx.shape(result) == {d_out()}
end
```

This test has no Python + Ray equivalent without significant extra scaffolding. In Elixir it
is 6 lines.

---

### Day 14 — Receipt generation

**Exercise 4.3 — Issue a verifiable receipt for each query.**

A receipt is an Elixir map. It records everything needed to audit or recompute the result.

```elixir
defmodule OVIR.Receipt do
  defstruct [
    :query_hash,
    :document_version,
    :fragments_used,     # list of fragment hashes
    :cache_hits,         # list of :hit | :miss atoms
    :worker_pids,        # PIDs that handled each fragment
    :answer_hash,
    :timestamp_utc
  ]

  def new(query_tokens, doc, fragment_results, answer) do
    %__MODULE__{
      query_hash:       hash(query_tokens),
      document_version: hash(doc.tokens),
      fragments_used:   Enum.map(fragment_results, & &1.fragment_hash),
      cache_hits:       Enum.map(fragment_results, & &1.hit_or_miss),
      worker_pids:      Enum.map(fragment_results, & &1.worker_pid),
      answer_hash:      hash(answer),
      timestamp_utc:    DateTime.utc_now()
    }
  end

  defp hash(tensor_or_binary) do
    b = if is_struct(tensor_or_binary, Nx.Tensor),
          do: Nx.to_binary(tensor_or_binary),
          else: tensor_or_binary
    :crypto.hash(:sha256, b) |> Base.encode16(case: :lower)
  end
end
```

**Unit tests:**
1. Re-running the same query produces a receipt where all `cache_hits` are `:hit`.
2. Every receipt has exactly `top_k` entries in `fragments_used`.
3. `answer_hash` is deterministic: same inputs → same hash.

---

## Week 5 — Confidence, Fallback, Spot-Check, and Distribution

### Day 15 — Confidence scoring

**Exercise 5.1 — When to trust the sparse path.**

Cosine similarity between the query vector and the merged answer:

```elixir
defmodule OVIR.Confidence do
  import Nx.Defn

  defn score(query_vec, answer_vec) do
    q_norm = Nx.sqrt(Nx.dot(query_vec, query_vec))
    a_norm = Nx.sqrt(Nx.dot(answer_vec, answer_vec))
    Nx.dot(query_vec, answer_vec) / (q_norm * a_norm + 1.0e-9)
  end
end
```

Run 50 random queries against the same document. Print the distribution of confidence
scores. Pick a threshold (try 0.3) below which the system escalates to a fallback.

---

### Day 16 — Fallback worker

**Exercise 5.2 — Full-attention fallback as its own GenServer.**

```elixir
defmodule OVIR.FallbackWorker do
  use GenServer

  def start_link(_), do: GenServer.start_link(__MODULE__, %{}, name: __MODULE__)

  def run(query_tokens, doc_tokens) do
    GenServer.call(__MODULE__, {:run, query_tokens, doc_tokens})
  end

  @impl true
  def init(state), do: {:ok, state}

  @impl true
  def handle_call({:run, query_tokens, doc_tokens}, _from, state) do
    # Full attention over entire document — expensive but correct
    q = Nx.mean(query_tokens, axes: [0]) |> Nx.new_axis(0)
    result = OVIR.Attention.full_attention(q, doc_tokens, doc_tokens)
             |> Nx.squeeze()
    {:reply, result, state}
  end
end
```

Wire into the pipeline:

```elixir
def query_with_fallback(query_tokens, doc, workers, threshold \\ 0.3) do
  answer = OVIR.Pipeline.query(query_tokens, doc, workers)
  conf   = OVIR.Confidence.score(Nx.mean(query_tokens, axes: [0]), answer)
           |> Nx.to_number()
  if conf >= threshold do
    {answer, :sparse, conf}
  else
    fallback = OVIR.FallbackWorker.run(query_tokens, doc.tokens)
    {fallback, :fallback, conf}
  end
end
```

Test: a query orthogonal to the document must return `:fallback`. A query closely matching
fragment 3 must return `:sparse`.

---

### Day 17 — Spot-check verification

**Exercise 5.3 — Recompute a random 10% of cached results.**

```elixir
defmodule OVIR.SpotChecker do
  def check(fragment_hash, chunk, experts, query_proj, p \\ 0.1) do
    if :rand.uniform() > p do
      :skipped
    else
      fresh = OVIR.FragmentProcessor.process(chunk, experts, query_proj)
      case OVIR.ExpertCache.get(fragment_hash) do
        {^fragment_hash, cached} ->
          if Nx.all_close(fresh, cached, atol: 1.0e-5) |> Nx.to_number() == 1 do
            :ok
          else
            {:mismatch, fragment_hash}
          end
        nil ->
          :not_cached
      end
    end
  end
end
```

For a deterministic router (fixed seed), `:mismatch` must never occur. Write an assertion
that fails the test if it does.

---

### Day 18 — Distributed nodes (the OVIR.net endgame)

**Exercise 5.4 — Run workers on two connected BEAM nodes.**

On your machine, start two named nodes in separate terminals:

```bash
# Terminal 1
iex --sname worker1 --cookie ovir_secret -S mix

# Terminal 2
iex --sname worker2 --cookie ovir_secret -S mix
```

Connect them:

```elixir
# in worker1's iex
Node.connect(:"worker2@hostname")
Node.list()   # should show [:worker2@hostname]
```

Spawn a `FragmentWorker` on the remote node:

```elixir
Node.spawn(:"worker2@hostname", fn ->
  OVIR.FragmentWorker.start_link({5, some_chunk(), experts(), query_proj()})
end)
```

Call it from worker1:

```elixir
OVIR.FragmentWorker.process(5, some_query_tokens())
```

This is real distribution. The fragment lives on a different OS process (potentially a
different machine). No extra framework needed — it is built into the BEAM.

---

### Day 19 — Final integration test

**Exercise 5.5 — End-to-end benchmark.**

Setup:
- Document: 200 tokens, 20 fragments of 10
- Experts: 64, product-key top-2
- Workers: 20 `FragmentWorker` GenServers under `WorkerSupervisor`
- FallbackWorker: running under the same supervisor
- Queries: 100 total — 10 unique queries × 10 repetitions

Assert all five:

```elixir
test "end-to-end OVIR pipeline" do
  results = run_all_queries(...)

  cache_hit_rate = count_hits(results) / 100
  assert cache_hit_rate > 0.80,   "cache should hit > 80% after warm-up"

  fallback_rate = count_fallbacks(results) / 100
  assert fallback_rate < 0.30,    "sparse path should handle > 70% of queries"

  expert_calls = total_expert_forward_calls()
  assert expert_calls < 100 * 10 * 2,   "caching must save expert calls"

  assert Enum.all?(results, fn {answer, _} -> Nx.shape(answer) == {d_out()} end)
  assert Enum.all?(results, fn {_, receipt} -> receipt_valid?(receipt) end)
end
```

If all five assertions pass, the toy OVIR pipeline is working correctly on Elixir + OTP.

---

## What you have at the end of Week 5

| Component | Implementation |
|---|---|
| BigBird sparse attention mask | `OVIR.Attention.Mask` — Nx boolean tensor |
| Full and sparse attention | `OVIR.Attention` — `defn` |
| PEER product-key expert retrieval | `OVIR.ProductKeyIndex` — `defn` |
| ETS fragment cache | `OVIR.ExpertCache` — `:ets` |
| Fragment routing | `OVIR.FragmentRouter` — dot-product scoring |
| Fragment workers | `OVIR.FragmentWorker` — `GenServer` |
| Fault tolerance | `OVIR.WorkerSupervisor` — `Supervisor` |
| Verifiable receipts | `OVIR.Receipt` — SHA-256 hashed maps |
| Confidence + fallback | `OVIR.Confidence` + `OVIR.FallbackWorker` |
| Spot-check verification | `OVIR.SpotChecker` |
| Real distribution | `Node.spawn` + `Node.connect` |

---

## What this does NOT cover (next steps)

- Real token embeddings (plug in `Bumblebee` for HuggingFace models on Elixir).
- EXLA GPU backend (replace `EXLA.Backend` CPU with CUDA when hardware is available).
- Persistent cache (swap ETS for `Cachex` or Mnesia for durability across restarts).
- Broadway or Flow for pipeline parallelism over large query batches.
- The PEER fine-grained MoE scaling law experiments (Section 4 of the paper).
- BigBird's Turing completeness proof (Section 3) — worth reading, not implementing.
- P2P receipt verification across untrusted nodes (the Gensyn REE direction).

Do not start these until Exercise 5.5 passes cleanly.

---

## References

[1] Zaheer et al. (2020). "Big Bird: Transformers for Longer Sequences." NeurIPS 2020.
    https://arxiv.org/abs/2007.14062  
[2] He (2024). "Mixture of A Million Experts." arXiv 2407.04153.
    https://arxiv.org/abs/2407.04153  
[3] OVIR.net project. https://github.com/metonymize-kripa/ovir  
[4] Nx documentation. https://hexdocs.pm/nx  
[5] EXLA documentation. https://hexdocs.pm/exla  
[6] Ortex (ONNX Runtime for Elixir). https://hexdocs.pm/ortex  
[7] ONNX specification. https://onnx.ai/  
[8] Erlang distributed systems. https://www.erlang.org/doc/system/distributed.html
