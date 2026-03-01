defmodule Edifice.SSM.GSSTest do
  use ExUnit.Case, async: true
  @moduletag :ssm

  alias Edifice.SSM.GSS

  @batch 2
  @seq_len 8
  @embed_dim 32
  @hidden_size 32

  @opts [
    embed_dim: @embed_dim,
    hidden_size: @hidden_size,
    state_size: 8,
    num_layers: 2,
    dropout: 0.0,
    seq_len: @seq_len
  ]

  defp random_input do
    key = Nx.Random.key(42)
    {input, _key} = Nx.Random.uniform(key, shape: {@batch, @seq_len, @embed_dim})
    input
  end

  describe "build/1" do
    test "produces correct output shape [batch, hidden_size]" do
      model = GSS.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.shape(output) == {@batch, @hidden_size}
    end

    test "output is finite" do
      model = GSS.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      output = predict_fn.(params, %{"state_sequence" => random_input()})
      assert Nx.all(Nx.is_nan(output) |> Nx.logical_not()) |> Nx.to_number() == 1
    end

    test "deterministic with same params and input" do
      model = GSS.build(@opts)
      {init_fn, predict_fn} = Axon.build(model)

      params =
        init_fn.(
          %{"state_sequence" => Nx.template({@batch, @seq_len, @embed_dim}, :f32)},
          Axon.ModelState.empty()
        )

      input = random_input()
      out1 = predict_fn.(params, %{"state_sequence" => input})
      out2 = predict_fn.(params, %{"state_sequence" => input})
      assert Nx.all_close(out1, out2, atol: 1.0e-5) |> Nx.to_number() == 1
    end
  end

  describe "output_size/1" do
    test "returns hidden_size" do
      assert GSS.output_size(@opts) == @hidden_size
    end
  end

  describe "recommended_defaults/0" do
    test "returns keyword list with expected keys" do
      defaults = GSS.recommended_defaults()
      assert Keyword.has_key?(defaults, :hidden_size)
      assert Keyword.has_key?(defaults, :state_size)
      assert Keyword.has_key?(defaults, :num_layers)
    end
  end
end
