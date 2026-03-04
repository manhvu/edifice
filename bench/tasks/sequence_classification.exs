# Sequence Classification Benchmark
#
# Task: Predict the sign of the total sum of a random sequence.
# Tests long-range dependency tracking — the model must aggregate
# information across all timesteps.
#
# Input:  [batch, seq_len, embed_dim] random normal
# Target: sign(sum) → binary one-hot [batch, 2]
#
# Usage:
#   mix run bench/tasks/sequence_classification.exs
#
# Environment variables:
#   TASK_EPOCHS  - Training epochs (default: 20)
#   TASK_LR      - Learning rate (default: 0.01)
#   TASK_BATCH   - Batch size (default: 32)

Nx.default_backend(EXLA.Backend)
Logger.configure(level: :warning)

Code.require_file("bench/tasks/task_helpers.exs")

defmodule SequenceClassification do
  @epochs String.to_integer(System.get_env("TASK_EPOCHS", "20"))
  @lr String.to_float(System.get_env("TASK_LR", "0.01"))
  @batch String.to_integer(System.get_env("TASK_BATCH", "32"))

  @embed 32
  @hidden 32
  @seq_len 64
  @num_layers 2
  @num_classes 2
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
    dropout: 0.0
  ]

  @architectures [
    {:lstm, :recurrent, []},
    {:mamba, :ssm, [state_size: 8]},
    {:gqa, :attention, []},
    {:min_gru, :recurrent, []},
    {:retnet, :attention, []}
  ]

  def run do
    IO.puts("Generating data...")
    {train_batches, eval_batches} = generate_data()

    TaskHelpers.print_header(
      "Sequence Classification — Cumulative Sum Sign",
      "embed=#{@embed}, hidden=#{@hidden}, seq_len=#{@seq_len}, layers=#{@num_layers}, " <>
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
    # Training data
    train_data = TaskHelpers.random_normal({@num_train, @seq_len, @embed}, 42)
    train_targets = compute_targets(train_data)
    train_batches = TaskHelpers.make_batches(%{"state_sequence" => train_data}, train_targets, @batch)

    # Eval data (different seed)
    eval_data = TaskHelpers.random_normal({@num_eval, @seq_len, @embed}, 123)
    eval_targets = compute_targets(eval_data)
    eval_batches = TaskHelpers.make_batches(%{"state_sequence" => eval_data}, eval_targets, @batch)

    {train_batches, eval_batches}
  end

  defp compute_targets(data) do
    # Target: sign of total sum across all timesteps and features
    sums = Nx.sum(data, axes: [1, 2])
    labels = Nx.greater(sums, 0) |> Nx.as_type(:s64) |> Nx.reshape({:auto})
    TaskHelpers.one_hot(labels, @num_classes)
  end

  defp run_arch(arch, category, extra_opts, train_batches, eval_batches) do
    try do
      opts = Keyword.merge(@shared_opts, extra_opts)
      base = Edifice.build(arch, opts)
      model = base |> Axon.dense(@num_classes)

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

SequenceClassification.run()
