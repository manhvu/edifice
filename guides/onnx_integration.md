# ONNX Integration
> How to export Edifice models to ONNX format for deployment in other frameworks, and how to import ONNX models into Edifice for inference or fine-tuning in Elixir.

## What This Guide Covers

[ONNX](https://onnx.ai) (Open Neural Network Exchange) is an interchange format supported by
PyTorch, TensorFlow, ONNX Runtime, and dozens of other frameworks. Exporting an Edifice model
to ONNX lets you deploy it in production environments that don't run Elixir -- mobile apps,
edge devices, web browsers (via ONNX.js), or C++/Python services.

This guide walks through:

1. **Exporting** -- Edifice model to `.onnx` file via `axon_onnx`
2. **Importing** -- ONNX model to Edifice/Axon for Elixir inference
3. **Running ONNX directly** -- using `ortex` (ONNX Runtime bindings) without conversion
4. **Limitations** -- which ops convert cleanly and which need workarounds
5. **Troubleshooting** -- common export/import failures and how to fix them

**Prerequisites:** Familiarity with `Edifice.build/2`, `Axon.build/2`, and `Axon.predict/3`.
You should know how to build a model and run inference in Elixir.

## Option A: Export and Import via axon_onnx

### 1. Add the dependency

`axon_onnx` is a separate package that converts between Axon models and ONNX format.
Add it to your `mix.exs`:

```elixir
defp deps do
  [
    {:edifice, "~> 0.3.0"},
    {:axon_onnx, "~> 0.4.0"}
  ]
end
```

Then `mix deps.get`. Note: `axon_onnx` requires `protoc` (Protocol Buffers compiler, >= 3.0)
to be available on your system for protobuf code generation.

### 2. Export an Edifice model to ONNX

Every Edifice architecture returns a standard Axon model, so the export workflow is:

```elixir
# 1. Build the Edifice model
model = Edifice.build(:mamba, input_size: 32, hidden_size: 64, output_size: 10)

# 2. Initialize parameters
{init_fn, predict_fn} = Axon.build(model)
params = init_fn.(Nx.template({1, 16, 32}, :f32), Axon.ModelState.empty())

# 3. Define input templates (must match the model's expected input shape)
templates = %{"input" => Nx.template({1, 16, 32}, :f32)}

# 4. Export to ONNX file
AxonOnnx.export(model, templates, params, path: "mamba_model.onnx")
```

The resulting `mamba_model.onnx` can be loaded in Python, C++, JavaScript, or any
ONNX-compatible runtime.

### 3. Import an ONNX model into Edifice

To load a model trained in PyTorch or TensorFlow:

```elixir
# Import returns an Axon model + parameters
{model, params} = AxonOnnx.import("resnet50.onnx")

# Run inference
input = Nx.random_uniform({1, 3, 224, 224}, type: :f32)
output = Axon.predict(model, params, input, compiler: EXLA)
```

For models with dynamic dimensions (batch size, sequence length), pass dimension hints:

```elixir
{model, params} = AxonOnnx.import("transformer.onnx", batch: 1, sequence: 128)
```

### 4. Running the exported model in Python

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("mamba_model.onnx")

# Check expected inputs
for inp in session.get_inputs():
    print(f"{inp.name}: {inp.shape} ({inp.type})")

# Run inference
input_data = np.random.randn(1, 16, 32).astype(np.float32)
results = session.run(None, {"input": input_data})
print(results[0].shape)
```

## Option B: Run ONNX Models Directly with Ortex

If you don't need to convert the model into Axon's graph representation -- you just want
to run an ONNX model in Elixir -- use `ortex`. It wraps ONNX Runtime via Rust NIFs and
supports the full ONNX spec (every op, every version).

### 1. Add the dependency

```elixir
defp deps do
  [
    {:ortex, "~> 0.1.10"}
  ]
end
```

### 2. Load and run

```elixir
model = Ortex.load("resnet50.onnx")

# Prepare input as an Nx tensor
input = Nx.random_uniform({1, 3, 224, 224}, type: :f32)

# Run inference (returns Nx tensors)
{output} = Ortex.run(model, input)
```

### 3. Production deployment with Nx.Serving

Ortex integrates with `Nx.Serving` for concurrent, batched inference:

```elixir
serving =
  Nx.Serving.new(Ortex.Serving, model)
  |> Nx.Serving.batch_size(8)
  |> Nx.Serving.client_preprocessing(fn input ->
    {Nx.Batch.stack([input]), :ok}
  end)

# Start as a supervised process
{:ok, _pid} = Nx.Serving.start_link(serving: serving, name: MyApp.Model)

# Call from anywhere
result = Nx.Serving.batched_run(MyApp.Model, input_tensor)
```

## When to Use Which

| Scenario | Tool | Why |
|----------|------|-----|
| Export Edifice model for deployment elsewhere | `axon_onnx` | Converts Axon graph to ONNX format |
| Import a simple ONNX model for Elixir training | `axon_onnx` | Gives you an Axon model you can train with `Axon.Loop` |
| Run any ONNX model in production Elixir | `ortex` | Full ONNX spec support via ONNX Runtime |
| Deploy with GPU acceleration (non-EXLA) | `ortex` | ONNX Runtime has CUDA/TensorRT/DirectML providers |
| Convert between frameworks (PyTorch <-> Elixir) | `axon_onnx` | Bidirectional conversion |

**Rule of thumb:** Use `axon_onnx` when you need to modify the model (train, fine-tune,
inspect layers). Use `ortex` when you just need fast inference on a pre-built ONNX model.

## Limitations

### axon_onnx limitations

`axon_onnx` supports a **subset** of the ONNX operator specification. Models using standard
layers (linear, conv, attention, normalization, activations) typically convert cleanly. Models
using exotic ops may fail.

**Ops that typically work:**
- Dense/Linear, Conv1D/2D, BatchNorm, LayerNorm, GroupNorm
- ReLU, GELU, Sigmoid, Tanh, Softmax
- MatMul, Add, Mul, Concat, Reshape, Transpose
- MaxPool, AveragePool, GlobalAveragePool
- LSTM, GRU (standard recurrent layers)

**Ops that may not convert:**
- Custom CUDA kernels (fused scans, flash attention) -- these are Edifice-specific
- Dynamic control flow (if/while inside the model graph)
- Sparse operations, scatter/gather patterns
- Framework-specific ops without ONNX equivalents

### Edifice-specific considerations

Edifice architectures that use custom fused CUDA kernels (Mamba selective scan, DeltaNet,
flash attention, etc.) will export using their **Elixir fallback** implementations. The
exported ONNX model is mathematically equivalent but won't have the fused kernel performance.
This is expected -- ONNX is a portable format, and the target runtime provides its own
optimizations.

**Architectures that export cleanly** (standard ops only):
- MLP, LSTM, GRU, Transformer (decoder-only), ViT, ConvNeXt, Swin
- Any architecture built entirely from Axon's built-in layers

**Architectures that export via fallback** (custom kernels replaced by equivalent Nx ops):
- Mamba, Mamba-3, SSD, S4, Hyena (SSM family)
- DeltaNet, GatedDeltaNet, sLSTM, TTT (matrix-state recurrences)
- Flash/LASER/FoX attention (replaced by standard attention)
- Titans, MIRAS, Reservoir (custom scan kernels)

The fallback path is correct -- it produces identical numerical results. The model will
simply run slower on the target runtime than it does with Edifice's CUDA kernels.

### ortex limitations

`ortex` runs the full ONNX spec, so there are no op-level limitations. The main constraints
are:
- Inference only (no training/gradient computation)
- Model must already be in `.onnx` format
- Some ONNX Runtime execution providers (CUDA, TensorRT) require specific system libraries

## Troubleshooting

### "Unsupported ONNX op: CustomOp"

The ONNX model uses an operator that `axon_onnx` doesn't support. Options:
1. Check if a newer version of `axon_onnx` added support
2. Use `ortex` instead (supports all ops via ONNX Runtime)
3. Simplify the source model before ONNX export (e.g., replace custom ops with standard equivalents in PyTorch before `torch.onnx.export`)

### Export produces empty or wrong-shaped output

Ensure your input templates match exactly what the model expects:

```elixir
# Wrong: template shape doesn't match model input
templates = %{"input" => Nx.template({1, 32}, :f32)}

# Right: check model input shape first
Axon.get_inputs(model)
# => %{"input" => {nil, 16, 32}}

templates = %{"input" => Nx.template({1, 16, 32}, :f32)}
```

### "protoc not found" during compilation

`axon_onnx` needs Protocol Buffers compiler. Install it:

```bash
# Ubuntu/Debian
sudo apt install protobuf-compiler

# macOS
brew install protobuf

# NixOS / nix-shell
nix-shell -p protobuf
```

### Import produces NaN outputs

Check the model's expected input normalization. Many vision models expect ImageNet-normalized
inputs (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

```elixir
mean = Nx.tensor([0.485, 0.456, 0.406]) |> Nx.reshape({1, 3, 1, 1})
std = Nx.tensor([0.229, 0.224, 0.225]) |> Nx.reshape({1, 3, 1, 1})
normalized = Nx.divide(Nx.subtract(input, mean), std)
```

### Exported model is much slower than Edifice

This is expected for architectures with custom CUDA kernels. The ONNX export uses the
Elixir fallback path (pure Nx operations), which is correct but not optimized. The target
runtime (ONNX Runtime, PyTorch, etc.) may apply its own graph optimizations, but won't
match Edifice's hand-tuned kernels for specialized operations like fused scans.

For maximum deployment performance, consider:
1. Using `ortex` with CUDA execution provider for GPU inference
2. ONNX Runtime's graph optimization passes (`ort.SessionOptions` with `ORT_ENABLE_ALL`)
3. TensorRT execution provider for NVIDIA GPUs (automatic kernel fusion)
4. The GGUF export path (`Edifice.Export.GGUF`) for LLM deployment in llama.cpp

## Further Reading

- [axon_onnx documentation](https://hexdocs.pm/axon_onnx/AxonOnnx.html)
- [Ortex documentation](https://hexdocs.pm/ortex/Ortex.html)
- [Axon's ONNX guide](https://hexdocs.pm/axon/onnx_to_axon.html)
- [ONNX operator specification](https://onnx.ai/onnx/operators/)
- [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/)
- [Edifice GGUF export](../lib/edifice/export/gguf.ex) -- alternative export for LLM deployment
