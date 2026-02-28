defmodule Edifice.Recurrent.MinGRU do
  @moduledoc """
  Minimal GRU (MinGRU) - A simplified GRU with a single gate.

  Implements the MinGRU from "Were RNNs All We Needed?" (Feng et al., 2024).
  MinGRU strips the GRU down to its essential component: a single forget/update
  gate. This makes it parallel-scannable during training while preserving the
  core gating mechanism that makes GRUs effective.

  ## Key Innovations

  - **Single gate**: Only one gate `z_t` controls interpolation (vs 3 in standard GRU)
  - **No hidden-to-hidden**: Gate depends only on input, not previous hidden state
  - **Parallel scannable**: The simplified recurrence admits a parallel prefix scan
  - **~30 lines of core logic**: Drastically simpler than standard GRU

  ## Equations

  ```
  z_t = sigmoid(linear_z(x_t))           # Update gate (input-only)
  candidate_t = linear_h(x_t)            # Candidate (no hidden dependency)
  h_t = (1 - z_t) * h_{t-1} + z_t * candidate_t  # Interpolation
  ```

  ## Architecture

  ```
  Input [batch, seq_len, embed_dim]
        |
        v
  [Input Projection] -> hidden_size
        |
        v
  +---------------------------+
  |     MinGRU Layer          |
  |  z = sigmoid(W_z * x)    |
  |  c = W_h * x             |
  |  h = (1-z)*h + z*c       |
  +---------------------------+
        | (repeat num_layers)
        v
  [Layer Norm] -> [Last Timestep]
        |
        v
  Output [batch, hidden_size]
  ```

  ## Usage

      model = MinGRU.build(
        embed_dim: 287,
        hidden_size: 256,
        num_layers: 4,
        dropout: 0.1
      )

  ## References
  - Paper: https://arxiv.org/abs/2410.01201
  """

  # ============================================================================
  # Default Hyperparameters
  # ============================================================================

  @doc "Default hidden dimension"
  @spec default_hidden_size() :: pos_integer()
  def default_hidden_size, do: 256

  @doc "Default number of layers"
  @spec default_num_layers() :: pos_integer()
  def default_num_layers, do: 4

  @doc "Default dropout rate"
  @spec default_dropout() :: float()
  def default_dropout, do: 0.1

  # ============================================================================
  # Model Building
  # ============================================================================

  @doc """
  Build a MinGRU model for sequence processing.

  ## Options
    - `:embed_dim` - Size of input embedding per frame (required)
    - `:hidden_size` - Internal hidden dimension (default: 256)
    - `:num_layers` - Number of MinGRU layers (default: 4)
    - `:dropout` - Dropout rate between layers (default: 0.1)
    - `:window_size` - Expected sequence length (default: 60)

  ## Returns
    An Axon model that processes sequences and outputs the last hidden state.
  """
  @typedoc "Options for `build/1`."
  @type build_opt ::
          {:dropout, float()}
          | {:embed_dim, pos_integer()}
          | {:hidden_size, pos_integer()}
          | {:num_layers, pos_integer()}
          | {:seq_len, pos_integer()}
          | {:window_size, pos_integer()}

  @spec build([build_opt()]) :: Axon.t()
  def build(opts \\ []) do
    embed_dim = Keyword.fetch!(opts, :embed_dim)
    hidden_size = Keyword.get(opts, :hidden_size, default_hidden_size())
    num_layers = Keyword.get(opts, :num_layers, default_num_layers())
    dropout = Keyword.get(opts, :dropout, default_dropout())
    window_size = Keyword.get(opts, :window_size, 60)
    seq_len = Keyword.get(opts, :seq_len, window_size)

    input_seq_dim = if seq_len, do: seq_len, else: nil

    # Input: [batch, seq_len, embed_dim]
    input = Axon.input("state_sequence", shape: {nil, input_seq_dim, embed_dim})

    # Project to hidden dimension if needed
    x =
      if embed_dim != hidden_size do
        Axon.dense(input, hidden_size, name: "input_projection")
      else
        input
      end

    # Stack MinGRU layers
    output =
      Enum.reduce(1..num_layers, x, fn layer_idx, acc ->
        layer = build_min_gru_layer(acc, hidden_size, "min_gru_#{layer_idx}")

        if dropout > 0 and layer_idx < num_layers do
          Axon.dropout(layer, rate: dropout, name: "dropout_#{layer_idx}")
        else
          layer
        end
      end)

    # Final layer norm
    output = Axon.layer_norm(output, name: "final_norm")

    # Extract last timestep: [batch, seq_len, hidden] -> [batch, hidden]
    Axon.nx(
      output,
      fn tensor ->
        seq = Nx.axis_size(tensor, 1)
        Nx.slice_along_axis(tensor, seq - 1, 1, axis: 1) |> Nx.squeeze(axes: [1])
      end,
      name: "last_timestep"
    )
  end

  # ============================================================================
  # MinGRU Layer
  # ============================================================================

  defp build_min_gru_layer(input, hidden_size, name) do
    # Pre-norm for stability
    normed = Axon.layer_norm(input, name: "#{name}_norm")

    # Gate projection: z = sigmoid(W_z * x)
    gate_proj = Axon.dense(normed, hidden_size, name: "#{name}_gate")

    # Candidate projection: candidate = W_h * x
    candidate_proj = Axon.dense(normed, hidden_size, name: "#{name}_candidate")

    # Apply MinGRU recurrence via two-input layer
    # Dispatches to fused CUDA kernel on GPU, falls back to Elixir scan on CPU
    recurrence_output =
      Axon.layer(
        fn gates, candidates, _opts ->
          Edifice.CUDA.FusedScan.mingru(gates, candidates)
        end,
        [gate_proj, candidate_proj],
        name: "#{name}_recurrence"
      )

    # Residual connection
    Axon.add(input, recurrence_output, name: "#{name}_residual")
  end

  @doc false
  # Sequential scan for MinGRU.
  #
  # Interface designed to match the fused CUDA kernel signature:
  #   gates:      [batch, seq_len, hidden] — raw gate logits (sigmoid applied here)
  #   candidates: [batch, seq_len, hidden] — candidate values
  #
  # Returns: [batch, seq_len, hidden] — all hidden states
  def min_gru_scan(gates, candidates) do
    batch_size = Nx.axis_size(gates, 0)
    seq_len = Nx.axis_size(gates, 1)
    hidden_size = Nx.axis_size(gates, 2)

    # Apply sigmoid to gate logits
    z = Nx.sigmoid(gates)

    # Sequential scan: h_t = (1 - z_t) * h_{t-1} + z_t * candidate_t
    h_init = Nx.broadcast(0.0, {batch_size, hidden_size})

    {_, h_list} =
      Enum.reduce(0..(seq_len - 1), {h_init, []}, fn t, {h_prev, acc} ->
        z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
        c_t = Nx.slice_along_axis(candidates, t, 1, axis: 1) |> Nx.squeeze(axes: [1])

        h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
        {h_t, [h_t | acc]}
      end)

    h_list |> Enum.reverse() |> Nx.stack(axis: 1)
  end

  # ============================================================================
  # Utilities
  # ============================================================================

  @doc """
  Get the output size of a MinGRU model.
  """
  @spec output_size(keyword()) :: non_neg_integer()
  def output_size(opts \\ []) do
    Keyword.get(opts, :hidden_size, default_hidden_size())
  end
end
