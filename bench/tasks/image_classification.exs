# Image Classification Benchmark
#
# Task: Classify 16x16 synthetic images by which quadrant has the brightest
# pixels. Tests spatial feature extraction.
#
# Input:  [batch, C, H, W] (NCHW) or [batch, H, W, C] (NHWC for ResNet)
# Target: quadrant class → [batch, 4] one-hot
#
# Usage:
#   mix run bench/tasks/image_classification.exs
#
# Environment variables:
#   TASK_EPOCHS  - Training epochs (default: 20)
#   TASK_LR      - Learning rate (default: 0.01)
#   TASK_BATCH   - Batch size (default: 32)

Nx.default_backend(EXLA.Backend)
Logger.configure(level: :warning)

Code.require_file("bench/tasks/task_helpers.exs")

defmodule ImageClassification do
  @epochs String.to_integer(System.get_env("TASK_EPOCHS", "20"))
  @lr String.to_float(System.get_env("TASK_LR", "0.01"))
  @batch String.to_integer(System.get_env("TASK_BATCH", "32"))

  @image_size 16
  @in_channels 3
  @num_classes 4
  @half 8
  @num_train 192
  @num_eval 64

  # Each architecture has its own build opts and input format
  @architectures [
    {:resnet, :conv,
     [
       input_shape: {nil, @image_size, @image_size, @in_channels},
       num_classes: @num_classes,
       block_sizes: [1, 1],
       initial_channels: 16,
       dropout: 0.0
     ], :nhwc, "input"},
    {:vit, :attention,
     [
       image_size: @image_size,
       patch_size: 4,
       in_channels: @in_channels,
       embed_dim: 32,
       depth: 2,
       num_heads: 4,
       num_classes: @num_classes,
       dropout: 0.0
     ], :nchw, "image"},
    {:convnext, :conv,
     [
       image_size: @image_size,
       patch_size: 2,
       in_channels: @in_channels,
       depths: [1, 1],
       dims: [32, 64],
       num_classes: @num_classes,
       dropout: 0.0
     ], :nchw, "image"},
    {:mlp_mixer, :mixer,
     [
       image_size: @image_size,
       patch_size: 4,
       in_channels: @in_channels,
       hidden_size: 32,
       num_layers: 2,
       token_mlp_dim: 32,
       channel_mlp_dim: 64,
       num_classes: @num_classes,
       dropout: 0.0
     ], :nchw, "image"},
    {:efficient_vit, :attention,
     [
       image_size: @image_size,
       patch_size: 4,
       in_channels: @in_channels,
       embed_dim: 32,
       depths: [1, 1],
       num_heads: [4, 4],
       num_classes: @num_classes,
       dropout: 0.0
     ], :nchw, "image"}
  ]

  def run do
    IO.puts("Generating data...")
    {images_nchw, images_nhwc, targets} = generate_data(@num_train + @num_eval)
    train_targets = Nx.slice_along_axis(targets, 0, @num_train, axis: 0)
    eval_targets = Nx.slice_along_axis(targets, @num_train, @num_eval, axis: 0)

    TaskHelpers.print_header(
      "Image Classification — Quadrant Brightness",
      "image=#{@image_size}x#{@image_size}, channels=#{@in_channels}, classes=#{@num_classes}, " <>
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
      for {arch, category, opts, layout, input_name} <- @architectures do
        {train_imgs, eval_imgs} = select_images(images_nchw, images_nhwc, layout)

        train_batches =
          TaskHelpers.make_batches(
            %{input_name => Nx.slice_along_axis(train_imgs, 0, @num_train, axis: 0)},
            train_targets,
            @batch
          )

        eval_batches =
          TaskHelpers.make_batches(
            %{input_name => Nx.slice_along_axis(eval_imgs, 0, @num_eval, axis: 0)},
            eval_targets,
            @batch
          )

        run_arch(arch, category, opts, input_name, layout, train_batches, eval_batches)
      end

    TaskHelpers.print_summary(results, :eval_accuracy, "Eval Accuracy")
  end

  defp generate_data(num_samples) do
    # Assign each sample a quadrant (0=TL, 1=TR, 2=BL, 3=BR)
    labels = TaskHelpers.random_integers({num_samples}, @num_classes, 42)
    targets = TaskHelpers.one_hot(Nx.reshape(labels, {:auto}), @num_classes)

    # Generate base dim images (low intensity)
    base = TaskHelpers.random_uniform({num_samples, @in_channels, @image_size, @image_size}, 0.0, 0.3, 43)

    # For each quadrant class, create a bright region
    # Build masks for each quadrant [1, 1, H, W]
    bright = TaskHelpers.random_uniform({num_samples, @in_channels, @image_size, @image_size}, 0.7, 1.0, 44)

    # Create per-sample quadrant masks
    images_nchw =
      Enum.reduce(0..(@num_classes - 1), base, fn q, acc ->
        # Mask: 1 where this sample's label == q AND pixel is in quadrant q
        is_this_class = Nx.equal(labels, q) |> Nx.reshape({num_samples, 1, 1, 1})
        is_this_class = Nx.broadcast(is_this_class, {num_samples, @in_channels, @image_size, @image_size})

        {row_start, col_start} = quadrant_offsets(q)
        # Create spatial mask
        rows = Nx.iota({@image_size, @image_size}, axis: 0)
        cols = Nx.iota({@image_size, @image_size}, axis: 1)
        in_rows = Nx.logical_and(Nx.greater_equal(rows, row_start), Nx.less(rows, row_start + @half))
        in_cols = Nx.logical_and(Nx.greater_equal(cols, col_start), Nx.less(cols, col_start + @half))
        spatial = Nx.logical_and(in_rows, in_cols) |> Nx.reshape({1, 1, @image_size, @image_size})
        spatial = Nx.broadcast(spatial, {num_samples, @in_channels, @image_size, @image_size})

        mask = Nx.logical_and(is_this_class, spatial) |> Nx.as_type(:f32)
        # Where mask=1, use bright; else keep acc
        Nx.add(Nx.multiply(acc, Nx.subtract(1.0, mask)), Nx.multiply(bright, mask))
      end)

    # NHWC version for ResNet
    images_nhwc = Nx.transpose(images_nchw, axes: [0, 2, 3, 1])

    {images_nchw, images_nhwc, targets}
  end

  defp quadrant_offsets(0), do: {0, 0}
  defp quadrant_offsets(1), do: {0, @half}
  defp quadrant_offsets(2), do: {@half, 0}
  defp quadrant_offsets(3), do: {@half, @half}

  defp select_images(nchw, _nhwc, :nchw), do: {nchw, nchw}
  defp select_images(_nchw, nhwc, :nhwc), do: {nhwc, nhwc}

  defp run_arch(arch, category, opts, input_name, layout, train_batches, eval_batches) do
    try do
      # Vision models already include classification head via num_classes
      model = Edifice.build(arch, opts)

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template_shape =
        case layout do
          :nchw -> {input_name, Nx.template({@batch, @in_channels, @image_size, @image_size}, :f32)}
          :nhwc -> {input_name, Nx.template({@batch, @image_size, @image_size, @in_channels}, :f32)}
        end

      template = Map.new([template_shape])
      model_state = init_fn.(template, Axon.ModelState.empty())
      params = TaskHelpers.count_params(model_state)

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
        params: params,
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

ImageClassification.run()
