# Graph Classification Benchmark
#
# Task: Classify random graphs into 4 classes based on edge density.
# Tests message passing / graph representation learning.
#
# Input:  nodes [batch, 8, 16] + adjacency [batch, 8, 8]
# Target: density class → [batch, 4] one-hot
#
# Usage:
#   mix run bench/tasks/graph_classification.exs
#
# Environment variables:
#   TASK_EPOCHS  - Training epochs (default: 20)
#   TASK_LR      - Learning rate (default: 0.01)
#   TASK_BATCH   - Batch size (default: 32)

Nx.default_backend(EXLA.Backend)
Logger.configure(level: :warning)

Code.require_file("bench/tasks/task_helpers.exs")

defmodule GraphClassification do
  @epochs String.to_integer(System.get_env("TASK_EPOCHS", "20"))
  @lr String.to_float(System.get_env("TASK_LR", "0.01"))
  @batch String.to_integer(System.get_env("TASK_BATCH", "32"))

  @num_nodes 8
  @node_dim 16
  @num_classes 4
  @num_train 192
  @num_eval 64

  def run do
    IO.puts("Generating data...")
    {train_data, eval_data} = generate_data()

    TaskHelpers.print_header(
      "Graph Classification — Edge Density",
      "nodes=#{@num_nodes}, node_dim=#{@node_dim}, classes=#{@num_classes}, " <>
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

    architectures = [
      {:gcn, :spectral, &build_gcn/0, &make_batches_2input/2},
      {:gat, :attention, &build_gat/0, &make_batches_2input/2},
      {:gin, :isomorphism, &build_gin/0, &make_batches_2input/2},
      {:gin_v2, :isomorphism, &build_ginv2/0, &make_batches_3input/2}
    ]

    results =
      for {arch, category, build_fn, batch_fn} <- architectures do
        run_arch(arch, category, build_fn, batch_fn, train_data, eval_data)
      end

    TaskHelpers.print_summary(results, :eval_accuracy, "Eval Accuracy")
  end

  # --- Model builders ---

  # GCN: per-node output → mean pool → dense(4)
  defp build_gcn do
    base = Edifice.build(:gcn, input_dim: @node_dim, hidden_dims: [32, 32], dropout: 0.0)
    pooled = Axon.nx(base, &Nx.mean(&1, axes: [1]), name: "graph_pool")
    Axon.dense(pooled, @num_classes, name: "classifier")
  end

  # GAT: per-node output [batch, nodes, num_classes] → mean pool
  defp build_gat do
    base =
      Edifice.build(:gat,
        input_dim: @node_dim,
        num_classes: 32,
        hidden_size: 8,
        num_heads: 4,
        num_layers: 2,
        dropout: 0.0
      )

    pooled = Axon.nx(base, &Nx.mean(&1, axes: [1]), name: "graph_pool")
    Axon.dense(pooled, @num_classes, name: "classifier")
  end

  # GIN: built-in pool → dense head via num_classes
  defp build_gin do
    Edifice.build(:gin,
      input_dim: @node_dim,
      hidden_dims: [32, 32],
      pool: :mean,
      num_classes: @num_classes,
      dropout: 0.0
    )
  end

  # GINv2: like GIN but with edge features
  defp build_ginv2 do
    Edifice.build(:gin_v2,
      input_dim: @node_dim,
      edge_dim: 1,
      hidden_dims: [32, 32],
      pool: :mean,
      num_classes: @num_classes,
      dropout: 0.0
    )
  end

  # --- Data generation ---

  defp generate_data do
    train = build_samples(@num_train, 42)
    eval = build_samples(@num_eval, 999)
    {train, eval}
  end

  defp build_samples(num_samples, seed) do
    # Assign density class to each sample
    labels = TaskHelpers.random_integers({num_samples}, @num_classes, seed)
    targets = TaskHelpers.one_hot(Nx.reshape(labels, {:auto}), @num_classes)

    # Generate node features (random, not class-dependent)
    nodes = TaskHelpers.random_normal({num_samples, @num_nodes, @node_dim}, seed + 1)

    # Generate adjacency matrices with class-dependent density
    # Class 0: ~15% edges, Class 1: ~35%, Class 2: ~60%, Class 3: ~85%
    thresholds = Nx.tensor([0.85, 0.65, 0.40, 0.15])
    per_sample_thresh =
      Nx.take(thresholds, Nx.reshape(labels, {num_samples}))
      |> Nx.reshape({num_samples, 1, 1})

    rand_adj = TaskHelpers.random_uniform({num_samples, @num_nodes, @num_nodes}, 0.0, 1.0, seed + 2)
    adjacency = Nx.greater(rand_adj, per_sample_thresh) |> Nx.as_type(:f32)

    # Make symmetric and zero diagonal
    adj_upper = Nx.triu(adjacency, k: 1)
    adjacency = Nx.add(adj_upper, Nx.transpose(adj_upper, axes: [0, 2, 1]))

    # Edge features for GINv2: adjacency value as 1-dim feature
    edge_features = Nx.reshape(adjacency, {num_samples, @num_nodes, @num_nodes, 1})

    %{nodes: nodes, adjacency: adjacency, edge_features: edge_features, targets: targets}
  end

  # --- Batching helpers ---

  defp make_batches_2input(data, split) do
    {nodes, adj, targets} = select_split(data, split)
    num = elem(Nx.shape(targets), 0)
    num_batches = div(num, @batch)

    for i <- 0..(num_batches - 1) do
      s = i * @batch

      input = %{
        "nodes" => Nx.slice_along_axis(nodes, s, @batch, axis: 0),
        "adjacency" => Nx.slice_along_axis(adj, s, @batch, axis: 0)
      }

      target = Nx.slice_along_axis(targets, s, @batch, axis: 0)
      {input, target}
    end
  end

  defp make_batches_3input(data, split) do
    {nodes, adj, targets} = select_split(data, split)
    edge_feat = if split == :train,
      do: Nx.slice_along_axis(data.edge_features, 0, @num_train, axis: 0),
      else: Nx.slice_along_axis(data.edge_features, 0, elem(Nx.shape(data.edge_features), 0), axis: 0)

    num = elem(Nx.shape(targets), 0)
    num_batches = div(num, @batch)

    for i <- 0..(num_batches - 1) do
      s = i * @batch

      input = %{
        "nodes" => Nx.slice_along_axis(nodes, s, @batch, axis: 0),
        "adjacency" => Nx.slice_along_axis(adj, s, @batch, axis: 0),
        "edge_features" => Nx.slice_along_axis(edge_feat, s, @batch, axis: 0)
      }

      target = Nx.slice_along_axis(targets, s, @batch, axis: 0)
      {input, target}
    end
  end

  defp select_split(data, :train) do
    {data.nodes, data.adjacency, data.targets}
  end

  defp select_split(data, :eval) do
    {data.nodes, data.adjacency, data.targets}
  end

  # --- Per-architecture runner ---

  defp run_arch(arch, category, build_fn, batch_fn, train_data, eval_data) do
    try do
      model = build_fn.()

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template_base = %{
        "nodes" => Nx.template({@batch, @num_nodes, @node_dim}, :f32),
        "adjacency" => Nx.template({@batch, @num_nodes, @num_nodes}, :f32)
      }

      template =
        if arch == :gin_v2,
          do: Map.put(template_base, "edge_features", Nx.template({@batch, @num_nodes, @num_nodes, 1}, :f32)),
          else: template_base

      model_state = init_fn.(template, Axon.ModelState.empty())

      train_batches = batch_fn.(train_data, :train)
      eval_batches = batch_fn.(eval_data, :eval)

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

GraphClassification.run()
