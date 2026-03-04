# Autoregressive Generation Benchmark
#
# Compares generate (position-aware) vs generate_simple (full-recompute)
# across model sizes to measure the benefit of position-aware decoding.
#
# Usage:
#   EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/generation_bench.exs
#   EXLA_TARGET=host mix run bench/generation_bench.exs

Logger.configure(level: :warning)

defmodule GenerationBench do
  @moduledoc false

  alias Edifice.Serving.Generate

  @vocab_size 256
  @seq_len String.to_integer(System.get_env("BENCH_SEQ_LEN", "64"))
  @embed_dim String.to_integer(System.get_env("BENCH_EMBED", "32"))
  @max_tokens String.to_integer(System.get_env("BENCH_MAX_TOKENS", "16"))
  @warmup_runs 1
  @bench_runs String.to_integer(System.get_env("BENCH_RUNS", "3"))

  def run do
    IO.puts("=== Autoregressive Generation Benchmark ===")
    IO.puts("seq_len=#{@seq_len} embed=#{@embed_dim} vocab=#{@vocab_size} max_tokens=#{@max_tokens}")
    IO.puts("")

    # Build model: simple dense layers (architecture-agnostic)
    model =
      Axon.input("state_sequence", shape: {nil, @seq_len, @embed_dim})
      |> Axon.dense(@embed_dim, name: "hidden", activation: :relu)
      |> Axon.dense(@vocab_size, name: "lm_head")

    template = %{"state_sequence" => Nx.template({1, @seq_len, @embed_dim}, :f32)}
    {init_fn, predict_fn} = Axon.build(model)
    params = init_fn.(template, Axon.ModelState.empty())

    # Embedding table
    {embed_table, _key} = Nx.Random.uniform(Nx.Random.key(0), shape: {@vocab_size, @embed_dim})
    embed_fn = fn ids -> Nx.take(embed_table, ids) end

    prompt = Nx.tensor([[1, 2, 3, 4, 5]])

    shared_opts = [
      prompt: prompt,
      embed_fn: embed_fn,
      max_tokens: @max_tokens,
      seq_len: @seq_len,
      temperature: 0.0
    ]

    # Warmup
    IO.puts("Warming up...")
    for _ <- 1..@warmup_runs do
      Generate.generate(predict_fn, params, shared_opts)
      Generate.generate_simple(predict_fn, params, shared_opts)
    end
    IO.puts("")

    # Benchmark generate (position-aware)
    IO.puts("--- generate (position-aware) ---")
    times_cached = bench(fn -> Generate.generate(predict_fn, params, shared_opts) end)
    report("generate", times_cached)

    # Benchmark generate_simple (full recompute)
    IO.puts("--- generate_simple (full recompute) ---")
    times_simple = bench(fn -> Generate.generate_simple(predict_fn, params, shared_opts) end)
    report("generate_simple", times_simple)

    # Compare
    mean_cached = Enum.sum(times_cached) / length(times_cached)
    mean_simple = Enum.sum(times_simple) / length(times_simple)

    if mean_cached > 0 do
      speedup = mean_simple / mean_cached
      IO.puts("")
      IO.puts("Speedup (position-aware vs recompute): #{Float.round(speedup, 2)}x")
    end

    # Validate outputs match for greedy
    result_cached = Generate.generate(predict_fn, params, shared_opts)
    result_simple = Generate.generate_simple(predict_fn, params, shared_opts)
    IO.puts("")
    IO.puts("Cached output:  #{inspect(Nx.to_flat_list(result_cached))}")
    IO.puts("Simple output:  #{inspect(Nx.to_flat_list(result_simple))}")
  end

  defp bench(fun) do
    Enum.map(1..@bench_runs, fn _ ->
      {usec, _result} = :timer.tc(fun)
      usec / 1_000.0
    end)
  end

  defp report(label, times_ms) do
    mean = Enum.sum(times_ms) / length(times_ms)
    min_t = Enum.min(times_ms)
    max_t = Enum.max(times_ms)

    IO.puts(
      "  #{label}: mean=#{Float.round(mean, 1)}ms min=#{Float.round(min_t, 1)}ms max=#{Float.round(max_t, 1)}ms (#{length(times_ms)} runs)"
    )
  end
end

GenerationBench.run()
