# XLA Compiler Flags Sweep Benchmark
#
# Tests the impact of various XLA_FLAGS on compilation time and
# inference latency for representative Edifice architectures.
#
# Usage (GPU required):
#   EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/xla_flags_sweep.exs
#
# Note: XLA_FLAGS are read at process start and cannot be changed at runtime.
# This script measures the current flag configuration and documents
# recommended flag combinations.
#
# To test different flags, restart with:
#   XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true" \
#     EXLA_TARGET=cuda XLA_TARGET=cuda12 mix run bench/xla_flags_sweep.exs

Logger.configure(level: :warning)

defmodule XlaFlagsSweep do
  @moduledoc false

  @embed String.to_integer(System.get_env("BENCH_EMBED", "64"))
  @seq_len String.to_integer(System.get_env("BENCH_SEQ_LEN", "32"))
  @batch String.to_integer(System.get_env("BENCH_BATCH", "1"))
  @warmup 3
  @iterations String.to_integer(System.get_env("BENCH_ITERS", "10"))

  # Representative architectures across families
  @archs [
    :min_gru,
    :mamba,
    :decoder_only,
    :s4d,
    :hyena
  ]

  @xla_flags_doc """
  ## Key XLA_FLAGS for GPU Performance

  | Flag | Effect | Recommended |
  |------|--------|-------------|
  | --xla_gpu_enable_latency_hiding_scheduler=true | Overlaps compute/memory ops | Yes |
  | --xla_gpu_graph_level=3 | Max graph-level optimization | Try 1-3 |
  | --xla_gpu_all_reduce_combine_threshold_bytes=N | AllReduce fusion threshold | 33554432 (32MB) |
  | --xla_gpu_enable_triton_gemm=false | Disable Triton for GEMM | Try both |
  | --xla_gpu_enable_command_buffer=true | Command buffer reuse | Yes |
  | --xla_gpu_enable_custom_fusions=true | Enable custom fusion | Yes |
  | --xla_dump_to=/tmp/xla_dump | Dump HLO for inspection | Debug only |

  ## EXLA Client Options (Elixir-level)

  | Option | Values | Default |
  |--------|--------|---------|
  | :allocator | :bfc, :cuda_async, :default | :bfc (upstream), :cuda_async (recommended for training) |
  | :memory_fraction | 0.0-1.0 | 0.9 |
  | :preallocate | true/false | true |
  """

  def run do
    IO.puts("=== XLA Compiler Flags Sweep ===")
    IO.puts("embed=#{@embed} seq_len=#{@seq_len} batch=#{@batch}")
    IO.puts("warmup=#{@warmup} iterations=#{@iterations}")
    IO.puts("")

    # Report current flags
    xla_flags = System.get_env("XLA_FLAGS", "(none)")
    tf_xla_flags = System.get_env("TF_XLA_FLAGS", "(none)")
    exla_target = System.get_env("EXLA_TARGET", "(none)")
    IO.puts("Current environment:")
    IO.puts("  EXLA_TARGET:  #{exla_target}")
    IO.puts("  XLA_FLAGS:    #{xla_flags}")
    IO.puts("  TF_XLA_FLAGS: #{tf_xla_flags}")
    IO.puts("")

    # Attach telemetry to capture compilation details
    attach_compilation_telemetry()

    IO.puts("Profiling #{length(@archs)} architectures...\n")

    results =
      Enum.map(@archs, fn arch ->
        IO.write("  #{arch}...")
        result = profile_arch(arch)

        case result do
          {:ok, data} ->
            IO.puts(" compile=#{data.compile_ms}ms, infer=#{data.mean_ms}ms")
            Map.put(data, :arch, arch)

          {:error, reason} ->
            IO.puts(" FAILED: #{reason}")
            %{arch: arch, error: reason}
        end
      end)

    IO.puts("")
    print_results(results)
    print_recommendations()

    detach_compilation_telemetry()
  end

  defp profile_arch(arch) do
    try do
      build_opts =
        case arch do
          :decoder_only ->
            [embed_dim: @embed, hidden_size: @embed, seq_len: @seq_len,
             num_layers: 2, num_heads: max(div(@embed, 8), 1),
             num_kv_heads: max(div(@embed, 8), 1)]

          _ ->
            [embed_dim: @embed, hidden_size: @embed, seq_len: @seq_len, num_layers: 2]
        end

      model = Edifice.build(arch, build_opts)

      template = %{
        "state_sequence" => Nx.template({@batch, @seq_len, @embed}, :f32)
      }

      # Measure compilation
      {compile_us, {init_fn, predict_fn}} = :timer.tc(fn -> Axon.build(model) end)
      compile_ms = compile_us / 1_000.0

      params = init_fn.(template, Axon.ModelState.empty())

      {input, _key} = Nx.Random.uniform(Nx.Random.key(42), shape: {@batch, @seq_len, @embed})
      input_map = %{"state_sequence" => input}

      # Warmup
      for _ <- 1..@warmup, do: predict_fn.(params, input_map)

      # Timed iterations
      times_ms =
        Enum.map(1..@iterations, fn _ ->
          {us, _} = :timer.tc(fn -> predict_fn.(params, input_map) end)
          us / 1_000.0
        end)

      mean = Enum.sum(times_ms) / length(times_ms)
      min_t = Enum.min(times_ms)
      p99 = Enum.sort(times_ms) |> Enum.at(round(length(times_ms) * 0.99))

      {:ok,
       %{
         compile_ms: Float.round(compile_ms, 1),
         mean_ms: Float.round(mean, 3),
         min_ms: Float.round(min_t, 3),
         p99_ms: Float.round(p99 || mean, 3)
       }}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp print_results(results) do
    header =
      String.pad_trailing("Architecture", 20) <>
        String.pad_leading("Compile(ms)", 13) <>
        String.pad_leading("Mean(ms)", 11) <>
        String.pad_leading("Min(ms)", 10) <>
        String.pad_leading("P99(ms)", 10)

    IO.puts(header)
    IO.puts(String.duplicate("-", String.length(header)))

    Enum.each(results, fn result ->
      if Map.has_key?(result, :error) do
        IO.puts(String.pad_trailing(to_string(result.arch), 20) <> "  ERROR: #{result.error}")
      else
        IO.puts(
          String.pad_trailing(to_string(result.arch), 20) <>
            String.pad_leading(to_string(result.compile_ms), 13) <>
            String.pad_leading(to_string(result.mean_ms), 11) <>
            String.pad_leading(to_string(result.min_ms), 10) <>
            String.pad_leading(to_string(result.p99_ms), 10)
        )
      end
    end)
  end

  defp print_recommendations do
    IO.puts("")
    IO.puts("=== Recommended Flag Combinations ===")
    IO.puts("")
    IO.puts("# Latency-optimized (real-time inference):")
    IO.puts(~s|XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_graph_level=3"|)
    IO.puts("")
    IO.puts("# Throughput-optimized (batch training):")
    IO.puts(~s|XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_all_reduce_combine_threshold_bytes=33554432"|)
    IO.puts("")
    IO.puts("# Debug (dump HLO graphs for analysis):")
    IO.puts(~s|XLA_FLAGS="--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text"|)
    IO.puts("")
    IO.puts("Re-run this script with different XLA_FLAGS to compare.")
    IO.puts(@xla_flags_doc)
  end

  # Telemetry for EXLA compilation events
  defp attach_compilation_telemetry do
    :telemetry.attach(
      "xla-flags-sweep",
      [:exla, :compilation],
      fn [:exla, :compilation], measurements, _meta, _config ->
        total_ms = measurements[:total_time] / 1_000
        if total_ms > 100 do
          IO.puts("    [EXLA] compilation took #{Float.round(total_ms, 1)}ms")
        end
      end,
      nil
    )
  rescue
    _ -> :ok
  end

  defp detach_compilation_telemetry do
    :telemetry.detach("xla-flags-sweep")
  rescue
    _ -> :ok
  end
end

XlaFlagsSweep.run()
