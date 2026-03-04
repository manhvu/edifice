# Autoregressive Generation Benchmark
#
# Task: Next-token prediction on a simple repeating grammar (0,1,2,...,7,0,1,...).
# Input is one-hot encoded tokens, model predicts the next token.
# Tests autoregressive / pattern learning capability.
#
# Input:  [batch, seq_len=31, vocab_size=8] one-hot encoded tokens
# Target: next token → [batch, 8] one-hot
#
# Usage:
#   mix run bench/tasks/autoregressive.exs
#
# Environment variables:
#   TASK_EPOCHS  - Training epochs (default: 30)
#   TASK_LR      - Learning rate (default: 0.01)
#   TASK_BATCH   - Batch size (default: 32)

Nx.default_backend(EXLA.Backend)
Logger.configure(level: :warning)

Code.require_file("bench/tasks/task_helpers.exs")

defmodule Autoregressive do
  @epochs String.to_integer(System.get_env("TASK_EPOCHS", "30"))
  @lr String.to_float(System.get_env("TASK_LR", "0.01"))
  @batch String.to_integer(System.get_env("TASK_BATCH", "32"))

  @vocab_size 8
  @embed @vocab_size
  @hidden 32
  @seq_len 31
  @num_layers 2
  @num_train 192
  @num_eval 64

  @shared_opts [
    embed_dim: @embed,
    hidden_size: @hidden,
    num_layers: @num_layers,
    seq_len: @seq_len,
    window_size: @seq_len,
    head_dim: 8,
    num_heads: 4,
    num_kv_heads: 2,
    head_size: 8,
    state_size: 8,
    dropout: 0.0
  ]

  @architectures [
    {:decoder_only, :transformer, []},
    {:mamba, :ssm, []},
    {:rwkv, :attention, []},
    {:min_gru, :recurrent, []}
  ]

  def run do
    IO.puts("Generating data...")
    {train_batches, eval_batches} = generate_data()

    TaskHelpers.print_header(
      "Autoregressive — Next Token Prediction (Repeating Grammar)",
      "vocab=#{@vocab_size}, hidden=#{@hidden}, seq_len=#{@seq_len}, layers=#{@num_layers}, " <>
        "epochs=#{@epochs}, lr=#{@lr}, batch=#{@batch}",
      [
        {"Architecture", 20},
        {"Category", 12},
        {"Train Acc", 12},
        {"Eval Acc", 12},
        {"Final Loss", 12},
        {"Time", 12}
      ]
    )

    results =
      for {arch, category, extra_opts} <- @architectures do
        run_arch(arch, category, extra_opts, train_batches, eval_batches)
      end

    TaskHelpers.print_summary(results, :eval_accuracy, "Eval Accuracy")
  end

  defp generate_data do
    # Grammar: repeating pattern 0,1,2,3,4,5,6,7,0,1,...
    # Each sample starts at a random offset in the pattern.
    # Input: tokens[0:seq_len] as one-hot, Target: token[seq_len] as one-hot

    train_data = build_samples(@num_train, 42)
    eval_data = build_samples(@num_eval, 999)

    train_batches = TaskHelpers.make_batches(
      %{"state_sequence" => train_data.inputs},
      train_data.targets,
      @batch
    )

    eval_batches = TaskHelpers.make_batches(
      %{"state_sequence" => eval_data.inputs},
      eval_data.targets,
      @batch
    )

    {train_batches, eval_batches}
  end

  defp build_samples(num_samples, seed) do
    # Random starting offsets
    offsets = TaskHelpers.random_integers({num_samples}, @vocab_size, seed)

    # Build token sequences: offset, offset+1, ..., offset+seq_len (mod vocab_size)
    # Shape: [num_samples, seq_len + 1]
    positions = Nx.iota({1, @seq_len + 1})
    offset_col = Nx.reshape(offsets, {num_samples, 1})
    tokens = Nx.remainder(Nx.add(offset_col, positions), @vocab_size)

    # Input: first seq_len tokens, one-hot encoded → [num_samples, seq_len, vocab_size]
    input_tokens = Nx.slice_along_axis(tokens, 0, @seq_len, axis: 1)
    inputs = one_hot_sequence(input_tokens)

    # Target: last token, one-hot → [num_samples, vocab_size]
    target_tokens = tokens[[.., @seq_len]]
    targets = TaskHelpers.one_hot(Nx.reshape(target_tokens, {:auto}), @vocab_size)

    %{inputs: inputs, targets: targets}
  end

  defp one_hot_sequence(tokens) do
    # tokens: [batch, seq_len] integers
    # output: [batch, seq_len, vocab_size] float
    expanded = Nx.reshape(tokens, {elem(Nx.shape(tokens), 0), @seq_len, 1})
    classes = Nx.iota({1, 1, @vocab_size})
    Nx.equal(expanded, classes) |> Nx.as_type(:f32)
  end

  defp run_arch(arch, category, extra_opts, train_batches, eval_batches) do
    try do
      opts = Keyword.merge(@shared_opts, extra_opts)
      base = Edifice.build(arch, opts)
      model = base |> Axon.dense(@vocab_size)

      {init_fn, predict_fn} = Axon.build(model)
      template = %{"state_sequence" => Nx.template({@batch, @seq_len, @embed}, :f32)}
      model_state = init_fn.(template, Axon.ModelState.empty())

      {train_us, {final_state, loss_history}} =
        :timer.tc(fn ->
          TaskHelpers.train(predict_fn, model_state, train_batches,
            epochs: @epochs,
            lr: @lr,
            loss_fn: &TaskHelpers.cross_entropy_loss/2
          )
        end)

      train_ms = train_us / 1_000
      train_acc = TaskHelpers.accuracy(predict_fn, final_state, train_batches)
      eval_acc = TaskHelpers.accuracy(predict_fn, final_state, eval_batches)
      final_loss = List.last(loss_history) || 0.0

      TaskHelpers.print_row([
        {arch, 20},
        {category, 12},
        {TaskHelpers.fmt_pct(train_acc), 12},
        {TaskHelpers.fmt_pct(eval_acc), 12},
        {Float.round(final_loss, 4), 12},
        {TaskHelpers.fmt_ms(train_ms), 12}
      ])

      %{
        arch: arch,
        category: category,
        train_accuracy: train_acc,
        eval_accuracy: eval_acc,
        final_loss: final_loss,
        train_ms: train_ms,
        status: :ok
      }
    rescue
      e ->
        msg = Exception.message(e) |> String.slice(0, 80)

        TaskHelpers.print_row([
          {arch, 20},
          {category, 12},
          {"FAIL: #{msg}", 60}
        ])

        %{arch: arch, category: category, status: :fail, error: msg}
    end
  end
end

Autoregressive.run()
