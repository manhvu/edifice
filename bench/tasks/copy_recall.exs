# Copy/Recall Benchmark
#
# Task: Given a sequence where the first half contains one of N template
# patterns and the second half is zeros, classify which template was shown.
# Tests memory capacity — the model must retain early input information.
#
# Input:  [batch, seq_len=32, embed=8] — first 16 steps are a template, last 16 are zeros
# Target: which template (1 of 4) → [batch, 4] one-hot
#
# Usage:
#   mix run bench/tasks/copy_recall.exs
#
# Environment variables:
#   TASK_EPOCHS  - Training epochs (default: 30)
#   TASK_LR      - Learning rate (default: 0.01)
#   TASK_BATCH   - Batch size (default: 32)

Nx.default_backend(EXLA.Backend)
Logger.configure(level: :warning)

Code.require_file("bench/tasks/task_helpers.exs")

defmodule CopyRecall do
  @epochs String.to_integer(System.get_env("TASK_EPOCHS", "30"))
  @lr String.to_float(System.get_env("TASK_LR", "0.01"))
  @batch String.to_integer(System.get_env("TASK_BATCH", "32"))

  @embed 8
  @hidden 32
  @seq_len 32
  @half_len 16
  @num_layers 2
  @num_templates 4
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
    state_size: 8,
    dropout: 0.0
  ]

  @architectures [
    {:lstm, :recurrent, []},
    {:mamba, :ssm, []},
    {:min_gru, :recurrent, []},
    {:retnet, :attention, []},
    {:delta_net, :recurrent, []},
    {:titans, :recurrent, [memory_size: 16]}
  ]

  def run do
    IO.puts("Generating data...")
    {templates, train_batches, eval_batches} = generate_data()
    _ = templates

    TaskHelpers.print_header(
      "Copy/Recall — Template Memory Classification",
      "embed=#{@embed}, hidden=#{@hidden}, seq_len=#{@seq_len}, layers=#{@num_layers}, " <>
        "templates=#{@num_templates}, epochs=#{@epochs}, lr=#{@lr}, batch=#{@batch}",
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
    # Create N fixed template patterns [num_templates, half_len, embed]
    templates =
      for i <- 0..(@num_templates - 1) do
        TaskHelpers.random_normal({@half_len, @embed}, 100 + i)
      end
      |> Nx.stack()

    # Generate training data
    {train_data, train_targets} = build_samples(@num_train, templates, 42)
    train_batches = TaskHelpers.make_batches(%{"state_sequence" => train_data}, train_targets, @batch)

    # Generate eval data
    {eval_data, eval_targets} = build_samples(@num_eval, templates, 999)
    eval_batches = TaskHelpers.make_batches(%{"state_sequence" => eval_data}, eval_targets, @batch)

    {templates, train_batches, eval_batches}
  end

  defp build_samples(num_samples, templates, seed) do
    # Assign each sample a random template
    labels = TaskHelpers.random_integers({num_samples}, @num_templates, seed)

    # Build sequences: template in first half, zeros in second half
    # Add small noise to templates to create variation
    noise = TaskHelpers.random_normal({num_samples, @half_len, @embed}, seed + 1)
    noise = Nx.multiply(noise, 0.1)

    # Gather template for each sample
    label_indices = Nx.reshape(labels, {num_samples, 1, 1})
    label_indices = Nx.broadcast(label_indices, {num_samples, @half_len, @embed})
    first_half = Nx.take(templates, Nx.reshape(labels, {num_samples}), axis: 0)
    first_half = Nx.add(first_half, noise)

    # Second half is zeros
    second_half = Nx.broadcast(Nx.tensor(0.0), {num_samples, @half_len, @embed})

    data = Nx.concatenate([first_half, second_half], axis: 1)
    targets = TaskHelpers.one_hot(Nx.reshape(labels, {:auto}), @num_templates)

    {data, targets}
  end

  defp run_arch(arch, category, extra_opts, train_batches, eval_batches) do
    try do
      opts = Keyword.merge(@shared_opts, extra_opts)
      base = Edifice.build(arch, opts)
      model = base |> Axon.dense(@num_templates)

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

CopyRecall.run()
