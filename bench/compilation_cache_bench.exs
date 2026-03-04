# XLA Compilation Cache Benchmark
#
# Measures cold-start vs warm-start compilation and inference times to
# quantify the speedup from EXLA disk cache + XLA autotune cache.
#
# Usage:
#   # GPU (recommended — autotune cache only helps on GPU):
#   EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/compilation_cache_bench.exs
#
#   # CPU:
#   EXLA_TARGET=host mix run bench/compilation_cache_bench.exs

Nx.default_backend(EXLA.Backend)
Logger.configure(level: :warning)

defmodule CompilationCacheBench do
  @moduledoc false

  @embed 128
  @seq_len 16
  @batch 1
  @cache_dir ".cache/exla_bench"

  @architectures [
    {:min_gru, 2, "MinGRU 2L"},
    {:griffin, 4, "Griffin 4L"},
    {:lstm, 3, "LSTM 3L"},
    {:gqa, 6, "GQA 6L"}
  ]

  defp model_opts(num_layers) do
    [
      embed_dim: @embed,
      hidden_size: @embed,
      state_size: min(@embed, 16),
      num_layers: num_layers,
      seq_len: @seq_len,
      window_size: @seq_len,
      head_dim: max(div(@embed, 8), 8),
      num_heads: max(div(@embed, max(div(@embed, 8), 8)), 1),
      dropout: 0.0
    ]
  end

  defp input_template, do: %{"state_sequence" => Nx.template({@batch, @seq_len, @embed}, :f32)}
  defp rand_input, do: %{"state_sequence" => Nx.Random.normal(Nx.Random.key(42), shape: {@batch, @seq_len, @embed}) |> elem(0)}

  defp time_ms(fun) do
    {us, result} = :timer.tc(fun)
    {us / 1_000, result}
  end

  def run do
    IO.puts("=" |> String.duplicate(72))
    IO.puts("XLA Compilation Cache Benchmark")
    IO.puts("embed=#{@embed}, seq_len=#{@seq_len}, batch=#{@batch}")
    IO.puts("=" |> String.duplicate(72))
    IO.puts("")

    # Phase 0: GPU warmup (absorb one-time cuDNN/BFC costs)
    IO.puts("## Phase 0: GPU Runtime Warmup")
    {arch0, layers0, _} = hd(@architectures)
    model0 = Edifice.build(arch0, model_opts(layers0))
    {init_fn, predict_fn} = Axon.build(model0, compiler: EXLA)
    params = init_fn.(input_template(), Axon.ModelState.empty())
    predict_fn.(params, rand_input())
    IO.puts("  Warmup done (#{arch0})")
    IO.puts("")

    # Clear bench cache
    File.rm_rf(@cache_dir)
    File.mkdir_p!(@cache_dir)

    IO.puts("## Cold vs Warm Compilation")
    IO.puts("-" |> String.duplicate(72))
    IO.puts(
      "  #{pad("Architecture", 20)}" <>
      "#{pad("Cold (ms)", 14)}" <>
      "#{pad("Warm (ms)", 14)}" <>
      "Speedup"
    )
    IO.puts("  " <> String.duplicate("-", 52))

    results =
      for {arch, num_layers, label} <- @architectures do
        cache_key = "bench_#{arch}_#{num_layers}L"
        cache_path = Path.join(@cache_dir, "#{cache_key}.exla")
        model = Edifice.build(arch, model_opts(num_layers))

        # Cold compile (no cache)
        File.rm(cache_path)
        {cold_ms, {init_fn, predict_fn}} =
          time_ms(fn ->
            Axon.build(model, compiler: EXLA, cache: cache_path)
          end)

        # Initialize params once
        params = init_fn.(input_template(), Axon.ModelState.empty())

        # Cold predict (triggers kernel autotuning)
        {cold_predict_ms, _} = time_ms(fn -> predict_fn.(params, rand_input()) end)

        # Warm compile (cache exists)
        {warm_ms, {_init_fn2, predict_fn2}} =
          time_ms(fn ->
            Axon.build(model, compiler: EXLA, cache: cache_path)
          end)

        # Warm predict (autotune cache hit)
        {warm_predict_ms, _} = time_ms(fn -> predict_fn2.(params, rand_input()) end)

        compile_speedup = if warm_ms > 0, do: cold_ms / warm_ms, else: 0.0

        IO.puts(
          "  #{pad(label, 20)}" <>
          "#{pad(fmt(cold_ms), 14)}" <>
          "#{pad(fmt(warm_ms), 14)}" <>
          "#{Float.round(compile_speedup, 1)}x"
        )

        {label, cold_ms, warm_ms, cold_predict_ms, warm_predict_ms}
      end

    IO.puts("")
    IO.puts("## Cold vs Warm Inference (first call)")
    IO.puts("-" |> String.duplicate(72))
    IO.puts(
      "  #{pad("Architecture", 20)}" <>
      "#{pad("Cold (ms)", 14)}" <>
      "#{pad("Warm (ms)", 14)}" <>
      "Speedup"
    )
    IO.puts("  " <> String.duplicate("-", 52))

    for {label, _cold_ms, _warm_ms, cold_predict_ms, warm_predict_ms} <- results do
      predict_speedup = if warm_predict_ms > 0, do: cold_predict_ms / warm_predict_ms, else: 0.0

      IO.puts(
        "  #{pad(label, 20)}" <>
        "#{pad(fmt(cold_predict_ms), 14)}" <>
        "#{pad(fmt(warm_predict_ms), 14)}" <>
        "#{Float.round(predict_speedup, 1)}x"
      )
    end

    # Report cache artifacts
    IO.puts("")
    IO.puts("## Cache Artifacts")
    IO.puts("-" |> String.duplicate(72))

    case File.ls(@cache_dir) do
      {:ok, files} ->
        exla_files = Enum.filter(files, &String.ends_with?(&1, ".exla"))
        IO.puts("  #{length(exla_files)} cached executables in #{@cache_dir}/")
        for f <- Enum.sort(exla_files) do
          size = File.stat!(Path.join(@cache_dir, f)).size
          IO.puts("    #{f} (#{div(size, 1024)} KB)")
        end
      {:error, _} ->
        IO.puts("  No cache directory found")
    end

    IO.puts("")

    # Cleanup
    File.rm_rf(@cache_dir)
    IO.puts("  Bench cache cleaned up.")
    IO.puts("")
    IO.puts("Done.")
  end

  defp pad(str, width), do: String.pad_trailing(to_string(str), width)

  defp fmt(ms) when ms < 0.01, do: "#{Float.round(ms * 1000, 1)} us"
  defp fmt(ms) when ms < 1, do: "#{Float.round(ms, 3)} ms"
  defp fmt(ms) when ms < 100, do: "#{Float.round(ms, 2)} ms"
  defp fmt(ms), do: "#{Float.round(ms, 0)} ms"
end

CompilationCacheBench.run()
