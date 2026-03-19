defmodule Edifice.Serving.BatchTest do
  use ExUnit.Case, async: true

  alias Edifice.Serving.Batch

  @embed_dim 4

  describe "new/3 inline usage" do
    test "single tensor input" do
      predict_fn = fn _params, input ->
        # Simple model: input["state_sequence"] * 2
        Nx.multiply(input["state_sequence"], 2)
      end

      serving = Batch.new(predict_fn, %{})

      input = Nx.tensor([[1.0, 2.0, 3.0, 4.0]])
      result = Nx.Serving.run(serving, input)

      expected = Nx.tensor([[2.0, 4.0, 6.0, 8.0]])

      assert Nx.all_close(result, expected) |> Nx.to_number() == 1
    end

    test "batched tensor input" do
      predict_fn = fn _params, input ->
        Nx.multiply(input["state_sequence"], 3)
      end

      serving = Batch.new(predict_fn, %{})

      input = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      result = Nx.Serving.run(serving, input)

      expected = Nx.tensor([[3.0, 6.0], [9.0, 12.0]])

      assert Nx.all_close(result, expected) |> Nx.to_number() == 1
    end

    test "list of tensors stacked into batch" do
      predict_fn = fn _params, input ->
        Nx.add(input["state_sequence"], 10)
      end

      serving = Batch.new(predict_fn, %{})

      inputs = [Nx.tensor([1.0, 2.0]), Nx.tensor([3.0, 4.0])]
      result = Nx.Serving.run(serving, inputs)

      expected = Nx.tensor([[11.0, 12.0], [13.0, 14.0]])

      assert Nx.all_close(result, expected) |> Nx.to_number() == 1
    end

    test "custom input_key" do
      predict_fn = fn _params, input ->
        Nx.multiply(input["image"], 2)
      end

      serving = Batch.new(predict_fn, %{}, input_key: "image")

      input = Nx.tensor([[1.0, 2.0]])
      result = Nx.Serving.run(serving, input)

      expected = Nx.tensor([[2.0, 4.0]])
      assert Nx.all_close(result, expected) |> Nx.to_number() == 1
    end

    test "nil input_key passes tensor directly" do
      predict_fn = fn _params, input ->
        Nx.multiply(input, 5)
      end

      serving = Batch.new(predict_fn, %{}, input_key: nil)

      input = Nx.tensor([[1.0, 2.0]])
      result = Nx.Serving.run(serving, input)

      expected = Nx.tensor([[5.0, 10.0]])
      assert Nx.all_close(result, expected) |> Nx.to_number() == 1
    end

    test "uses model params from closure" do
      predict_fn = fn params, input ->
        Nx.add(input["state_sequence"], params["bias"])
      end

      params = %{"bias" => Nx.tensor([100.0, 200.0])}
      serving = Batch.new(predict_fn, params)

      input = Nx.tensor([[1.0, 2.0]])
      result = Nx.Serving.run(serving, input)

      expected = Nx.tensor([[101.0, 202.0]])
      assert Nx.all_close(result, expected) |> Nx.to_number() == 1
    end
  end

  describe "new/3 with Axon model" do
    test "works with a simple Axon model" do
      model =
        Axon.input("state_sequence", shape: {nil, @embed_dim})
        |> Axon.dense(@embed_dim, name: "out")

      {init_fn, predict_fn} = Axon.build(model, mode: :inference)

      template = %{"state_sequence" => Nx.template({1, @embed_dim}, :f32)}
      params = init_fn.(template, Axon.ModelState.empty())

      serving = Batch.new(predict_fn, params)

      input = Nx.iota({2, @embed_dim}, type: :f32)
      result = Nx.Serving.run(serving, input)

      # Just verify shape and type — values depend on random init
      assert Nx.shape(result) == {2, @embed_dim}
      assert Nx.type(result) == {:f, 32}
    end
  end

  describe "composability" do
    test "composes with FP8 quantization" do
      predict_fn = fn params, input ->
        # Dequantize params and use them
        kernel = params["kernel"]

        actual_kernel =
          case kernel do
            %{tensor: t, scale: s, original_type: ot} ->
              t |> Nx.as_type({:f, 32}) |> Nx.multiply(s) |> Nx.as_type(ot)

            t ->
              t
          end

        Nx.dot(input["state_sequence"], actual_kernel)
      end

      kernel = Nx.tensor([[1.0, 0.0], [0.0, 1.0]])
      q_kernel = Edifice.Quantization.FP8.quantize(%{"kernel" => kernel})
      serving = Batch.new(predict_fn, q_kernel)

      input = Nx.tensor([[3.0, 4.0]])
      result = Nx.Serving.run(serving, input)

      # Should be approximately [3.0, 4.0] (identity kernel)
      assert Nx.all_close(result, Nx.tensor([[3.0, 4.0]]), atol: 0.5) |> Nx.to_number() == 1
    end
  end
end
