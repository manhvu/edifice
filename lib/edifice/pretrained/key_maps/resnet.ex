defmodule Edifice.Pretrained.KeyMaps.ResNet do
  @moduledoc """
  Key map for loading HuggingFace ResNet checkpoints into Edifice ResNet models.

  Maps parameter names from the `microsoft/resnet-50` SafeTensors format
  to the Axon layer names used by `Edifice.Convolutional.ResNet`.

  ## Key Differences

  - **Conv2d layout**: PyTorch stores conv weights as `{out, in, kH, kW}` (OIHW),
    while Axon uses `{kH, kW, in, out}` (HWIO). All conv weights are permuted.
  - **No conv bias**: HuggingFace ResNet uses `bias=False` on all Conv2d layers.
    Axon's default zero-initialized bias is harmless since BatchNorm follows each conv.
  - **BatchNorm naming**: HuggingFace uses `normalization.{weight,bias,running_mean,running_var}`,
    Edifice uses `{gamma,beta,mean,var}`.
  - **Classifier indexing**: HuggingFace wraps the classifier in `nn.Sequential` with
    `[0]=Flatten, [1]=Linear`, so the Linear weight is at `classifier.1.weight`.
  - **Bottleneck sublayer indexing**: HuggingFace uses `layer.{0,1,2}` for the three
    convs in a bottleneck block, Edifice uses `conv{1,2,3}` (1-indexed).
  - **Double embedder**: HuggingFace nests the stem conv inside
    `resnet.embedder.embedder.convolution`, with the outer embedder also containing
    a parameter-less MaxPool.

  ## Usage

      model_state = Edifice.Pretrained.load(
        Edifice.Pretrained.KeyMaps.ResNet,
        "model.safetensors"
      )

  """

  @behaviour Edifice.Pretrained.KeyMap

  # HF: resnet.embedder.embedder.{convolution|normalization}.{param}
  @stem_re ~r/^resnet\.embedder\.embedder\.(\w+)\.(\w+)$/

  # HF: resnet.encoder.stages.{s}.layers.{l}.layer.{k}.{convolution|normalization}.{param}
  @block_re ~r/^resnet\.encoder\.stages\.(\d+)\.layers\.(\d+)\.layer\.(\d+)\.(\w+)\.(\w+)$/

  # HF: resnet.encoder.stages.{s}.layers.{l}.shortcut.{convolution|normalization}.{param}
  @shortcut_re ~r/^resnet\.encoder\.stages\.(\d+)\.layers\.(\d+)\.shortcut\.(\w+)\.(\w+)$/

  # HF: classifier.1.{weight|bias}
  @classifier_re ~r/^classifier\.1\.(\w+)$/

  # --- Stem ---
  @impl true
  def map_key(key) do
    cond do
      match = Regex.run(@stem_re, key) ->
        [_, module, param] = match
        map_stem(module, param)

      match = Regex.run(@block_re, key) ->
        [_, stage, layer, sublayer, module, param] = match
        map_block(stage, layer, sublayer, module, param)

      match = Regex.run(@shortcut_re, key) ->
        [_, stage, layer, module, param] = match
        map_shortcut(stage, layer, module, param)

      match = Regex.run(@classifier_re, key) ->
        [_, param] = match
        map_classifier(param)

      true ->
        :unmapped
    end
  end

  # --- Stem mapping ---

  defp map_stem("convolution", "weight"), do: "stem_conv.kernel"

  defp map_stem("normalization", param), do: map_bn_param("stem_bn", param)

  defp map_stem(_module, _param), do: :unmapped

  # --- Bottleneck block mapping ---
  # HF layer.{k} → Edifice conv{k+1}/bn{k+1}

  defp map_block(s, l, k, "convolution", "weight") do
    conv_idx = String.to_integer(k) + 1
    "stage#{s}_block#{l}_conv#{conv_idx}.kernel"
  end

  defp map_block(s, l, k, "normalization", param) do
    bn_idx = String.to_integer(k) + 1
    map_bn_param("stage#{s}_block#{l}_bn#{bn_idx}", param)
  end

  defp map_block(_s, _l, _k, _module, _param), do: :unmapped

  # --- Shortcut (projection) mapping ---

  defp map_shortcut(s, l, "convolution", "weight"),
    do: "stage#{s}_block#{l}_skip_proj.kernel"

  defp map_shortcut(s, l, "normalization", param),
    do: map_bn_param("stage#{s}_block#{l}_skip_bn", param)

  defp map_shortcut(_s, _l, _module, _param), do: :unmapped

  # --- Classifier ---

  defp map_classifier("weight"), do: "classifier.kernel"
  defp map_classifier("bias"), do: "classifier.bias"
  defp map_classifier(_), do: :unmapped

  # --- BatchNorm parameter mapping ---
  # HF: weight → gamma, bias → beta, running_mean → mean, running_var → var

  defp map_bn_param(prefix, "weight"), do: "#{prefix}.gamma"
  defp map_bn_param(prefix, "bias"), do: "#{prefix}.beta"
  defp map_bn_param(prefix, "running_mean"), do: "#{prefix}.mean"
  defp map_bn_param(prefix, "running_var"), do: "#{prefix}.var"
  defp map_bn_param(_prefix, "num_batches_tracked"), do: :skip
  defp map_bn_param(_prefix, _param), do: :unmapped

  @impl true
  def tensor_transforms do
    [
      # Conv kernels: PyTorch {O, I, H, W} → Axon {H, W, I, O}
      # Classifier kernel: PyTorch {out, in} → Axon {in, out}
      {~r/\.kernel$/, &transform_kernel/1}
    ]
  end

  defp transform_kernel(tensor) do
    case Nx.rank(tensor) do
      4 -> Nx.transpose(tensor, axes: [2, 3, 1, 0])
      2 -> Nx.transpose(tensor)
      _ -> tensor
    end
  end
end
