# Shared utilities for applied task benchmarks.
#
# Provides: manual SGD training loop, evaluation metrics, data generation,
# and formatted output. Follows the existing bench pattern (value_and_grad,
# no Axon.Loop).
#
# Usage: Code.require_file("bench/tasks/task_helpers.exs")

defmodule TaskHelpers do
  @doc """
  Train with manual SGD using Nx.Defn.value_and_grad.

  Arguments:
    - predict_fn: compiled prediction function from Axon.build
    - init_model_state: initial %Axon.ModelState{} from init_fn
    - batches: list of {input_map, target} tuples
    - opts: [epochs: N, lr: float, loss_fn: fn(output, target) -> scalar]

  Returns {final_model_state, loss_history}
  """
  def train(predict_fn, init_model_state, batches, opts \\ []) do
    lr = opts[:lr] || 0.01
    epochs = opts[:epochs] || 10
    loss_fn = opts[:loss_fn] || &mse_loss/2

    Enum.reduce(1..epochs, {init_model_state, []}, fn _epoch, {ms, losses} ->
      {ms, epoch_loss} =
        Enum.reduce(batches, {ms, 0.0}, fn {input, target}, {ms, acc} ->
          # backend_copy all captured tensors to BinaryBackend so they can
          # be inlined as constants during defn tracing (avoids EXLA.Backend
          # vs Nx.Defn.Expr incompatibility in value_and_grad closures)
          input_bc = backend_copy_map(input)
          target_bc = Nx.backend_copy(target, Nx.BinaryBackend)
          state_bc = backend_copy_state(ms.state)
          ms_bc = %{ms | state: state_bc}

          {loss, grads} =
            Nx.Defn.value_and_grad(ms.data, fn params ->
              state = %{ms_bc | data: params}
              output = predict_fn.(state, input_bc)
              loss_fn.(output, target_bc)
            end)

          new_data = sgd_step(ms.data, grads, lr)
          {%{ms | data: new_data}, acc + Nx.to_number(loss)}
        end)

      avg_loss = epoch_loss / max(length(batches), 1)
      {ms, losses ++ [avg_loss]}
    end)
  end

  @doc "Classification accuracy (argmax match)"
  def accuracy(predict_fn, model_state, batches) do
    {correct, total} =
      Enum.reduce(batches, {0, 0}, fn {input, target}, {c, t} ->
        output = predict_fn.(model_state, input)
        preds = Nx.argmax(output, axis: -1)
        labels = Nx.argmax(target, axis: -1)
        matches = Nx.equal(preds, labels) |> Nx.sum() |> Nx.to_number()
        batch_size = output |> Nx.shape() |> elem(0)
        {c + matches, t + batch_size}
      end)

    if total > 0, do: correct / total, else: 0.0
  end

  @doc "Mean squared error evaluation"
  def eval_mse(predict_fn, model_state, batches) do
    {total, count} =
      Enum.reduce(batches, {0.0, 0}, fn {input, target}, {acc, n} ->
        output = predict_fn.(model_state, input)
        err = Nx.subtract(output, target) |> Nx.pow(2) |> Nx.mean() |> Nx.to_number()
        {acc + err, n + 1}
      end)

    if count > 0, do: total / count, else: 0.0
  end

  # --- Loss functions ---

  def mse_loss(output, target) do
    Nx.subtract(output, target) |> Nx.pow(2) |> Nx.mean()
  end

  def cross_entropy_loss(logits, target) do
    # Numerically stable: -mean(sum(target * log_softmax(logits)))
    max_l = Nx.reduce_max(logits, axes: [-1], keep_axes: true)
    shifted = Nx.subtract(logits, max_l)
    log_sum_exp = shifted |> Nx.exp() |> Nx.sum(axes: [-1], keep_axes: true) |> Nx.log()
    log_probs = Nx.subtract(shifted, log_sum_exp)
    Nx.multiply(target, log_probs) |> Nx.sum(axes: [-1]) |> Nx.negate() |> Nx.mean()
  end

  # --- Data generation (deterministic, Nx.Random.key based) ---

  def random_normal(shape, key \\ 42) do
    k = if is_integer(key), do: Nx.Random.key(key), else: key
    {t, _} = Nx.Random.normal(k, shape: shape)
    t
  end

  def random_uniform(shape, min_val \\ 0.0, max_val \\ 1.0, key \\ 42) do
    k = if is_integer(key), do: Nx.Random.key(key), else: key
    {t, _} = Nx.Random.uniform(k, min_val, max_val, shape: shape)
    t
  end

  def random_integers(shape, max_val, key \\ 42) do
    k = if is_integer(key), do: Nx.Random.key(key), else: key
    {t, _} = Nx.Random.randint(k, 0, max_val, shape: shape)
    t
  end

  def one_hot(indices, num_classes) do
    Nx.equal(
      Nx.reshape(indices, {Nx.size(indices), 1}),
      Nx.iota({1, num_classes})
    )
    |> Nx.as_type(:f32)
  end

  # --- Batching ---

  @doc "Split a map of tensors + target tensor into batches"
  def make_batches(input_map, targets, batch_size) do
    total = targets |> Nx.shape() |> elem(0)
    num_batches = div(total, batch_size)

    for i <- 0..(num_batches - 1) do
      s = i * batch_size

      batch_input =
        Map.new(input_map, fn {k, v} ->
          {k, Nx.slice_along_axis(v, s, batch_size, axis: 0)}
        end)

      batch_target = Nx.slice_along_axis(targets, s, batch_size, axis: 0)
      {batch_input, batch_target}
    end
  end

  # --- Formatting (matches existing bench pattern) ---

  def fmt_ms(ms) when ms < 0.01, do: "#{Float.round(ms * 1000, 1)} us"
  def fmt_ms(ms) when ms < 1, do: "#{Float.round(ms, 3)} ms"
  def fmt_ms(ms) when ms < 100, do: "#{Float.round(ms, 2)} ms"
  def fmt_ms(ms), do: "#{Float.round(ms, 0)} ms"

  def fmt_pct(pct), do: "#{Float.round(pct * 100, 1)}%"

  def print_header(title, config_str, columns) do
    IO.puts(String.duplicate("=", 80))
    IO.puts(title)
    IO.puts(config_str)
    IO.puts(String.duplicate("=", 80))
    IO.puts("")

    header =
      Enum.map_join(columns, fn {name, width} ->
        String.pad_trailing(name, width)
      end)

    IO.puts("  " <> header)
    IO.puts("  " <> String.duplicate("-", 78))
  end

  def print_row(values) do
    row =
      Enum.map_join(values, fn {val, width} ->
        String.pad_trailing(to_string(val), width)
      end)

    IO.puts("  " <> row)
  end

  def print_summary(results, metric_key, metric_label, opts \\ []) do
    higher_better = Keyword.get(opts, :higher_better, true)
    IO.puts("")
    IO.puts("  " <> String.duplicate("-", 78))

    {ok, failed} = Enum.split_with(results, &(&1.status == :ok))
    IO.puts("  #{length(ok)}/#{length(results)} succeeded")

    if failed != [] do
      IO.puts("")
      IO.puts("  FAILURES:")
      Enum.each(failed, fn r -> IO.puts("    #{r.arch}: #{r.error}") end)
    end

    if ok != [] do
      IO.puts("")
      IO.puts("## Ranking by #{metric_label}")
      IO.puts(String.duplicate("-", 60))

      sorted =
        if higher_better,
          do: Enum.sort_by(ok, &(-Map.get(&1, metric_key))),
          else: Enum.sort_by(ok, &Map.get(&1, metric_key))

      sorted
      |> Enum.with_index(1)
      |> Enum.each(fn {r, rank} ->
        val = Map.get(r, metric_key)

        formatted =
          if metric_key in [:accuracy, :eval_accuracy],
            do: fmt_pct(val),
            else: "#{Float.round(val * 1.0, 4)}"

        IO.puts(
          "  #{String.pad_trailing("#{rank}", 6)}" <>
            "#{String.pad_trailing("#{r.arch}", 25)}" <>
            "#{formatted}"
        )
      end)
    end

    IO.puts("")
    IO.puts("Done.")
  end

  def count_params(model_state) do
    model_state.data |> collect_tensors() |> Enum.reduce(0, fn t, acc -> acc + Nx.size(t) end)
  end

  defp collect_tensors(%Nx.Tensor{} = t), do: [t]
  defp collect_tensors(%{} = map), do: Enum.flat_map(map, fn {_, v} -> collect_tensors(v) end)
  defp collect_tensors(_), do: []

  @doc """
  Replace RNN hidden state init descriptors (["key"]) with zero tensors.
  Must be called before training any architecture that uses Axon.rnn_state
  (LSTM, GRU, etc.) to avoid Nx.Random.split failures inside value_and_grad.
  """
  def materialize_rnn_states(model_state, batch_size, hidden_size) do
    new_state =
      Map.new(model_state.state, fn
        {k, ["key"]} ->
          {k, Nx.broadcast(Nx.tensor(0.0), {batch_size, hidden_size})}

        {k, v} ->
          {k, v}
      end)

    %{model_state | state: new_state}
  end

  defp backend_copy_map(map) do
    Map.new(map, fn {k, v} -> {k, Nx.backend_copy(v, Nx.BinaryBackend)} end)
  end

  defp backend_copy_state(state) do
    Map.new(state, fn
      {k, %Nx.Tensor{} = t} -> {k, Nx.backend_copy(t, Nx.BinaryBackend)}
      {k, v} -> {k, v}
    end)
  end

  defp sgd_step(params, grads, lr) do
    Map.new(params, fn {key, value} ->
      grad = Map.get(grads, key)

      case {value, grad} do
        {%Nx.Tensor{} = p, %Nx.Tensor{} = g} ->
          {key, Nx.subtract(p, Nx.multiply(g, lr))}

        {%{} = p_map, %{} = g_map} when map_size(g_map) > 0 ->
          {key, sgd_step(p_map, g_map, lr)}

        _ ->
          {key, value}
      end
    end)
  end
end
