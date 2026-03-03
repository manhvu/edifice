# Future Directions — Edifice Roadmap Research

*Generated 2026-03-03. Project state: 263 architectures, 26 families, 56 CUDA kernels, pretrained hub, 2500+ tests.*

## Project Maturity Assessment

Edifice has comprehensive **breadth** (263 architectures across every major ML family) and solid **depth** (CUDA kernel fusion through P10, pretrained weight loading, 22 guides, 13 notebooks). The TODO is effectively cleared.

The next phase should shift from "implement more architectures" to **depth, usability, and real-world applicability**.

---

## Direction 1: Applied Task Benchmarks (Priority: High)

### Problem
Edifice has extensive micro-benchmarks (latency, memory, throughput) but no downstream task evaluations. Users can't answer: "Which architecture should I pick for my sequence classification task?"

### Proposal
Create a `bench/tasks/` suite that evaluates architectures on small standardized tasks:

| Task | Dataset | Architectures | Metric |
|------|---------|---------------|--------|
| Sequence classification | Synthetic (length generalization) | LSTM, Mamba, GQA, MinGRU, RetNet | Accuracy + latency |
| Image classification | MNIST / FashionMNIST subset | MLP, ResNet, ViT, ConvNeXt, EfficientViT | Accuracy + params |
| Graph classification | Synthetic (community detection) | GCN, GAT, GIN, EGNN, GPS | Accuracy |
| Autoregressive generation | Char-level Shakespeare | Decoder-only, Mamba, RWKV, Hyena | Perplexity + throughput |
| Copy/recall task | Synthetic | LSTM, Mamba, Titans, SSM variants | Accuracy vs sequence length |

**Why**: Transforms Edifice from a "zoo" into an opinionated guide. The results become content for guides and notebooks.

**Effort**: Medium. Small datasets, short training loops (Axon.Loop), ~500 lines per task.

---

## Direction 2: Inference Serving Layer (Priority: High)

### Problem
Users can build and train models but there's no path to deploy them. No batched inference, no streaming, no KV cache management for autoregressive generation.

### Proposal
`Edifice.Serving` module with:

1. **Batched inference server** — GenServer wrapping `Axon.build(model, compiler: EXLA)` with request batching and timeout. Think `Nx.Serving` but architecture-aware.
2. **KV cache integration** — `Edifice.Blocks.KVCache` already exists. Wire it into a stateful generation loop for decoder-only, Mamba, and hybrid architectures.
3. **Autoregressive generation** — `Edifice.Serving.generate(model, params, prompt, max_tokens: 100)` with temperature, top-k, top-p sampling. Support speculative decoding via existing `Edifice.Inference.Medusa` and `Edifice.Inference.SpeculativeDecoding`.
4. **Streaming output** — Stream tokens via `Stream` or Phoenix.PubSub integration.

**Why**: Closes the gap between "I loaded pretrained weights" and "I'm serving predictions."

**Effort**: High. KV cache generation loop is the hard part. Start with greedy decoding, add sampling later.

---

## Direction 3: Architecture Search / AutoML (Priority: Medium)

### Problem
263 architectures is overwhelming. Users need help choosing and configuring architectures for their problem.

### Proposal
`Edifice.AutoML` module:

1. **Architecture recommender** — Given task type (classification, generation, detection) and constraints (latency budget, param budget, GPU/CPU), recommend top-3 architectures with suggested hyperparameters.
2. **Hyperparameter search** — Wrap Axon.Loop with Bayesian optimization over architecture opts (hidden_size, num_layers, num_heads, etc.). Small search space, few epochs.
3. **Architecture morphisms** — Given a trained small model, grow it into a larger architecture (Net2Net-style). Leverage pretrained weight loading infrastructure.

**Why**: Makes the library accessible to non-experts. Differentiator vs PyTorch ecosystem.

**Effort**: High. Recommender is medium (rule-based + benchmark data). Hyperparameter search is medium (wrap existing Axon.Loop). Architecture morphisms are research-grade.

---

## Direction 4: Numerical Correctness Suite (Priority: High)

### Problem
Edifice implements 263 architectures but only ViT and Whisper have numerical validation against PyTorch reference outputs. Users need confidence that the implementations are correct beyond "shapes match and no NaNs."

### Proposal
Expand `test/edifice/pretrained/numerical_validation_test.exs` pattern:

1. **Reference output fixtures** — For 10-15 key architectures, generate PyTorch outputs on fixed random inputs via `scripts/generate_numerical_fixtures.py`. Store as `.safetensors` fixtures.
2. **Cross-framework validation** — Compare Edifice forward pass against fixtures at `atol=1e-4`.
3. **Gradient validation** — Compare `value_and_grad` outputs against PyTorch `autograd` for architectures with CUDA kernels (ensures backward kernels are correct beyond the existing analytical gradient smoke tests).

**Target architectures**: LSTM, Mamba, GQA, ViT, DETR, MinGRU, DeltaNet, DiT, ResNet, GAT.

**Why**: Essential for trust. Users won't use Edifice for real work without numerical validation.

**Effort**: Medium. The infrastructure exists (fixture generator, safetensors round-trip). Mainly need PyTorch reference scripts.

---

## Direction 5: New Architecture Candidates (Priority: Medium)

### What's Hot (ICLR 2026 / NeurIPS 2025)

Already implemented (skip):
- Gated Attention, NSA, Differential Transformer, Mamba-3, RWKV-7, Titans, Coconut, Memory Layers, FoX, LASER, MoBA, DeltaProduct, GSA, Samba, Huginn, LLaDA, CaDDi, DeepFlow, Meissonic, VeRA, Kron-LoRA, MoT

**Not yet implemented — worth considering:**

| Architecture | Source | Key Idea | Complexity |
|---|---|---|---|
| **GoldFinch** | RWKV team | Hybrid GOLD transformer + enhanced RWKV-6, compressed KV-Cache | Medium |
| **Artificial Hippocampus Network (AHN)** | 2025 | Dual memory: sliding-window KV-cache + fixed-size RNN for long context | Medium-High |
| **Contextual Priority Attention (CPA)** | Nature 2025 | Priority scoring + contextual gating for dynamic attention allocation | Medium |
| **Latent Recurrent Depth** | OpenReview 2026 | Adaptive-depth recurrent block for test-time compute scaling | High |
| **MaxScore MoE Routing** | 2025 | Constrained optimization routing, replaces token-dropping | Low |
| **Causal MoE** | 2025 | Embeds causal structure discovery into expert routing | High |
| **Saguaro (SSD)** | 2025 | Parallelized speculative + verification for 5x AR speedup | Medium |
| **Online Speculative Decoding** | 2025 | Adaptive draft model via online knowledge distillation | Medium |
| **DSKD (Dual-Space KD)** | 2025 | Cross-architecture distillation with vocabulary mismatch handling | Medium |

### Recommendation
Don't chase architecture count further. Instead, add only architectures that:
1. Won best paper awards (already done for NeurIPS 2025 / ACL 2025)
2. Are deployed in production frontier models
3. Fill a genuinely new architectural pattern not covered by existing 263

The library's value is now in **quality and usability**, not quantity.

---

## Direction 6: Training Recipes (Priority: Medium)

### Problem
Edifice has architectures and a training loop (Axon.Loop) but no pre-built training recipes. Users must figure out loss functions, optimizers, learning rate schedules, and data loading themselves.

### Proposal
`Edifice.Recipes` module with ready-to-run training configurations:

1. **Classification recipe** — `Edifice.Recipes.classify(model, train_data, opts)`. Handles cross-entropy loss, Adam optimizer, cosine LR schedule, early stopping.
2. **Sequence modeling recipe** — `Edifice.Recipes.language_model(model, tokenized_data, opts)`. Causal LM loss, gradient clipping, warmup schedule.
3. **Contrastive recipe** — `Edifice.Recipes.contrastive(model, augmented_pairs, opts)`. InfoNCE loss, projection head, EMA target network.
4. **Fine-tuning recipe** — `Edifice.Recipes.fine_tune(model, params, data, opts)`. Freeze base, train head. Support LoRA/DoRA via existing meta modules.

**Why**: Reduces time-to-first-result from hours to minutes. Pairs naturally with pretrained weight loading.

**Effort**: Medium. Thin wrappers around Axon.Loop with sensible defaults.

---

## Direction 7: Interactive Model Explorer (Priority: Low-Medium)

### Problem
`mix edifice.viz` exists but is CLI-only. Users want to interactively explore architectures, compare them, and understand their structure.

### Proposal
A Livebook Smart Cell or Phoenix LiveView app:

1. **Architecture browser** — Filter/search 263 architectures by family, input type, param count.
2. **Side-by-side comparison** — Pick 2-3 architectures, see layer structure, param counts, and benchmark results.
3. **Interactive builder** — Configure opts via form, see model structure update in real-time.
4. **Benchmark dashboard** — Visualize latency/memory/accuracy tradeoffs from task benchmarks (Direction 1).

**Why**: Makes the library discoverable and approachable. Great for talks/demos.

**Effort**: High for LiveView, Medium for Livebook Smart Cell.

---

## Proposed Prioritization

### Phase 1 — Trust & Usability (next)
1. **Numerical correctness suite** — 10 architectures validated against PyTorch
2. **Applied task benchmarks** — 5 standardized tasks with leaderboard
3. **Verify new notebooks** — existing TODO item

### Phase 2 — Production Path
4. **Inference serving layer** — batched inference + KV cache generation
5. **Training recipes** — 4 pre-built training configurations

### Phase 3 — Discovery & Polish
6. **Interactive model explorer** — Livebook Smart Cell
7. **AutoML / recommender** — architecture selection assistant
8. **Selective new architectures** — only best-paper / production-deployed

---

## Decision Points

- **Should Edifice stay a pure architecture library or grow into a framework?** Training recipes and serving push toward framework territory. Could conflict with Bumblebee/Axon ecosystem.
- **How much PyTorch reference testing is feasible?** Requires maintaining Python fixtures. CI can't run PyTorch — fixtures must be pre-generated and committed.
- **Is the audience researchers or practitioners?** Researchers want breadth + correctness. Practitioners want recipes + serving. Current trajectory leans researcher, but serving/recipes would broaden appeal.
