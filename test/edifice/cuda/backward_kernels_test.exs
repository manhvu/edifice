defmodule Edifice.CUDA.BackwardKernelsTest do
  @moduledoc """
  Gradient correctness tests for backward CUDA kernels.

  Tests run on BinaryBackend (no GPU required) — they validate the Elixir
  backward fallback functions against Nx autodiff of the forward scan.

  Test strategy:
  1. Backward fallback matches Nx autodiff (forward scan differentiated by Nx)
  2. Numerical gradient check via finite differences
  3. Various shapes (batch=1, seq=1, large hidden, large seq)
  """
  use ExUnit.Case, async: true

  # ============================================================================
  # Linear scan backward tests
  # ============================================================================

  describe "linear_scan_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {a_vals, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {b_vals, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      # Forward scan to get output
      forward_out = Edifice.CUDA.FusedScan.linear_scan_fallback(a_vals, b_vals)

      # Use Nx autodiff to get reference gradients
      # We compute grad of sum(forward(a,b)) w.r.t. a and b
      grad_fn = fn a, b ->
        Edifice.CUDA.FusedScan.linear_scan_fallback(a, b) |> Nx.sum()
      end

      {ref_grad_a, ref_grad_b} = Nx.Defn.grad({a_vals, b_vals}, fn {a, b} -> grad_fn.(a, b) end)

      # Our backward fallback with ones gradient (since we took sum)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})
      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      {grad_a, grad_b, _grad_h0} =
        Edifice.CUDA.FusedScan.linear_scan_backward_fallback(a_vals, h0, forward_out, grad_output)

      assert_all_close(grad_a, ref_grad_a, atol: 1.0e-5)
      assert_all_close(grad_b, ref_grad_b, atol: 1.0e-5)
    end

    test "numerical gradient check via finite differences" do
      batch = 1
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(99)
      {a_vals, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {b_vals, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      forward_out = Edifice.CUDA.FusedScan.linear_scan_fallback(a_vals, b_vals)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_a, grad_b, _} =
        Edifice.CUDA.FusedScan.linear_scan_backward_fallback(a_vals, h0, forward_out, grad_output)

      # Finite difference for a_vals
      eps = 1.0e-3
      numerical_grad_a = finite_diff_grad(a_vals, fn a -> Edifice.CUDA.FusedScan.linear_scan_fallback(a, b_vals) |> Nx.sum() end, eps)
      numerical_grad_b = finite_diff_grad(b_vals, fn b -> Edifice.CUDA.FusedScan.linear_scan_fallback(a_vals, b) |> Nx.sum() end, eps)

      assert_all_close(grad_a, numerical_grad_a, atol: 1.0e-2)
      assert_all_close(grad_b, numerical_grad_b, atol: 1.0e-2)
    end

    test "handles batch=1 and seq=1" do
      a = Nx.tensor([[[0.5]]], type: {:f, 32})
      b = Nx.tensor([[[1.0]]], type: {:f, 32})
      h0 = Nx.tensor([[0.0]], type: {:f, 32})

      forward_out = Edifice.CUDA.FusedScan.linear_scan_fallback(a, b)
      grad_output = Nx.tensor([[[1.0]]], type: {:f, 32})

      {grad_a, grad_b, grad_h0} =
        Edifice.CUDA.FusedScan.linear_scan_backward_fallback(a, h0, forward_out, grad_output)

      # h_0 = 0, h_1 = 0.5 * 0 + 1.0 = 1.0
      # da = dh * h_{-1} = 1.0 * 0.0 = 0.0
      # db = dh = 1.0
      # dh0 = dh * a = 1.0 * 0.5 = 0.5
      assert_all_close(grad_a, Nx.tensor([[[0.0]]], type: {:f, 32}), atol: 1.0e-6)
      assert_all_close(grad_b, Nx.tensor([[[1.0]]], type: {:f, 32}), atol: 1.0e-6)
      assert_all_close(grad_h0, Nx.tensor([[0.5]], type: {:f, 32}), atol: 1.0e-6)
    end

    test "larger hidden dimension" do
      batch = 2
      seq_len = 8
      hidden = 32

      key = Nx.Random.key(123)
      {a_vals, key} = Nx.Random.uniform(key, 0.3, 0.7, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {b_vals, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      forward_out = Edifice.CUDA.FusedScan.linear_scan_fallback(a_vals, b_vals)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})
      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      {ref_grad_a, ref_grad_b} =
        Nx.Defn.grad({a_vals, b_vals}, fn {a, b} ->
          Edifice.CUDA.FusedScan.linear_scan_fallback(a, b) |> Nx.sum()
        end)

      {grad_a, grad_b, _} =
        Edifice.CUDA.FusedScan.linear_scan_backward_fallback(a_vals, h0, forward_out, grad_output)

      assert_all_close(grad_a, ref_grad_a, atol: 1.0e-4)
      assert_all_close(grad_b, ref_grad_b, atol: 1.0e-4)
    end
  end

  # ============================================================================
  # MinGRU backward tests
  # ============================================================================

  describe "mingru_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {z, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {candidates, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      # Forward
      forward_fn = fn z, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(z)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
            z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
            {h_t, [h_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(z, candidates)

      # Reference grads via Nx autodiff
      {ref_grad_z, ref_grad_cand} =
        Nx.Defn.grad({z, candidates}, fn {z, cand} ->
          forward_fn.(z, cand) |> Nx.sum()
        end)

      # Our backward
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_z, grad_cand, _grad_h0} =
        Edifice.CUDA.FusedScan.mingru_backward_fallback(z, candidates, h0, forward_out, grad_output)

      assert_all_close(grad_z, ref_grad_z, atol: 1.0e-5)
      assert_all_close(grad_cand, ref_grad_cand, atol: 1.0e-5)
    end

    test "numerical gradient check via finite differences" do
      batch = 1
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(77)
      {z, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {candidates, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})

      forward_fn = fn z, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(z)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {h_prev, acc} ->
            z_t = Nx.slice_along_axis(z, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            h_t = Nx.add(Nx.multiply(Nx.subtract(1.0, z_t), h_prev), Nx.multiply(z_t, c_t))
            {h_t, [h_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(z, candidates)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_z, grad_cand, _} =
        Edifice.CUDA.FusedScan.mingru_backward_fallback(z, candidates, h0, forward_out, grad_output)

      eps = 1.0e-3
      numerical_z = finite_diff_grad(z, fn z -> forward_fn.(z, candidates) |> Nx.sum() end, eps)
      numerical_cand = finite_diff_grad(candidates, fn c -> forward_fn.(z, c) |> Nx.sum() end, eps)

      assert_all_close(grad_z, numerical_z, atol: 1.0e-2)
      assert_all_close(grad_cand, numerical_cand, atol: 1.0e-2)
    end

    test "handles seq_len=1" do
      z = Nx.tensor([[[0.3, 0.7]]], type: {:f, 32})
      cand = Nx.tensor([[[2.0, -1.0]]], type: {:f, 32})
      h0 = Nx.tensor([[0.0, 0.0]], type: {:f, 32})
      forward_out = Nx.tensor([[[0.6, -0.7]]], type: {:f, 32})
      grad_output = Nx.tensor([[[1.0, 1.0]]], type: {:f, 32})

      {grad_z, grad_cand, grad_h0} =
        Edifice.CUDA.FusedScan.mingru_backward_fallback(z, cand, h0, forward_out, grad_output)

      # dz = dh * (c - h_prev) = 1 * (2-0, -1-0) = (2, -1)
      # dc = dh * z = 1 * (0.3, 0.7) = (0.3, 0.7)
      # dh0 = dh * (1-z) = 1 * (0.7, 0.3) = (0.7, 0.3)
      assert_all_close(grad_z, Nx.tensor([[[2.0, -1.0]]], type: {:f, 32}), atol: 1.0e-6)
      assert_all_close(grad_cand, Nx.tensor([[[0.3, 0.7]]], type: {:f, 32}), atol: 1.0e-6)
      assert_all_close(grad_h0, Nx.tensor([[0.7, 0.3]], type: {:f, 32}), atol: 1.0e-6)
    end
  end

  # ============================================================================
  # MinLSTM backward tests
  # ============================================================================

  describe "minlstm_backward_fallback" do
    test "matches Nx autodiff for basic case" do
      batch = 2
      seq_len = 4
      hidden = 3

      key = Nx.Random.key(42)
      {f_gate, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {i_gate, key} = Nx.Random.uniform(key, 0.1, 0.9, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {candidates, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      norm_eps = 1.0e-6

      # Forward function matching the kernel's behavior
      forward_fn = fn f, i, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(f)
        gate_sum = Nx.add(f, Nx.add(i, norm_eps))
        f_norm = Nx.divide(f, gate_sum)
        i_norm = Nx.divide(i, gate_sum)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {c_prev, acc} ->
            f_t = Nx.slice_along_axis(f_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            i_t = Nx.slice_along_axis(i_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            cand_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.add(Nx.multiply(f_t, c_prev), Nx.multiply(i_t, cand_t))
            {c_t, [c_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(f_gate, i_gate, candidates)

      # Reference grads
      {ref_grad_f, ref_grad_i, ref_grad_cand} =
        Nx.Defn.grad({f_gate, i_gate, candidates}, fn {f, i, cand} ->
          forward_fn.(f, i, cand) |> Nx.sum()
        end)

      # Our backward
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_f, grad_i, grad_cand, _grad_h0} =
        Edifice.CUDA.FusedScan.minlstm_backward_fallback(f_gate, i_gate, candidates, h0, forward_out, grad_output)

      assert_all_close(grad_f, ref_grad_f, atol: 1.0e-4)
      assert_all_close(grad_i, ref_grad_i, atol: 1.0e-4)
      assert_all_close(grad_cand, ref_grad_cand, atol: 1.0e-4)
    end

    test "numerical gradient check via finite differences" do
      batch = 1
      seq_len = 3
      hidden = 2

      key = Nx.Random.key(55)
      {f_gate, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {i_gate, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {candidates, _key} = Nx.Random.uniform(key, -0.5, 0.5, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      norm_eps = 1.0e-6

      forward_fn = fn f, i, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(f)
        gate_sum = Nx.add(f, Nx.add(i, norm_eps))
        f_norm = Nx.divide(f, gate_sum)
        i_norm = Nx.divide(i, gate_sum)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {c_prev, acc} ->
            f_t = Nx.slice_along_axis(f_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            i_t = Nx.slice_along_axis(i_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            cand_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.add(Nx.multiply(f_t, c_prev), Nx.multiply(i_t, cand_t))
            {c_t, [c_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(f_gate, i_gate, candidates)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {grad_f, grad_i, grad_cand, _} =
        Edifice.CUDA.FusedScan.minlstm_backward_fallback(f_gate, i_gate, candidates, h0, forward_out, grad_output)

      eps = 1.0e-3
      numerical_f = finite_diff_grad(f_gate, fn f -> forward_fn.(f, i_gate, candidates) |> Nx.sum() end, eps)
      numerical_i = finite_diff_grad(i_gate, fn i -> forward_fn.(f_gate, i, candidates) |> Nx.sum() end, eps)
      numerical_cand = finite_diff_grad(candidates, fn c -> forward_fn.(f_gate, i_gate, c) |> Nx.sum() end, eps)

      assert_all_close(grad_f, numerical_f, atol: 1.0e-2)
      assert_all_close(grad_i, numerical_i, atol: 1.0e-2)
      assert_all_close(grad_cand, numerical_cand, atol: 1.0e-2)
    end

    test "larger shapes" do
      batch = 2
      seq_len = 8
      hidden = 16

      key = Nx.Random.key(200)
      {f_gate, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {i_gate, key} = Nx.Random.uniform(key, 0.2, 0.8, shape: {batch, seq_len, hidden}, type: {:f, 32})
      {candidates, _key} = Nx.Random.uniform(key, -1.0, 1.0, shape: {batch, seq_len, hidden}, type: {:f, 32})

      h0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {batch, hidden})
      norm_eps = 1.0e-6

      forward_fn = fn f, i, cand ->
        {_batch, seq_len, _hidden} = Nx.shape(f)
        gate_sum = Nx.add(f, Nx.add(i, norm_eps))
        f_norm = Nx.divide(f, gate_sum)
        i_norm = Nx.divide(i, gate_sum)

        {_, h_list} =
          Enum.reduce(0..(seq_len - 1), {h0, []}, fn t, {c_prev, acc} ->
            f_t = Nx.slice_along_axis(f_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            i_t = Nx.slice_along_axis(i_norm, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            cand_t = Nx.slice_along_axis(cand, t, 1, axis: 1) |> Nx.squeeze(axes: [1])
            c_t = Nx.add(Nx.multiply(f_t, c_prev), Nx.multiply(i_t, cand_t))
            {c_t, [c_t | acc]}
          end)

        h_list |> Enum.reverse() |> Nx.stack(axis: 1)
      end

      forward_out = forward_fn.(f_gate, i_gate, candidates)
      grad_output = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {batch, seq_len, hidden})

      {ref_grad_f, ref_grad_i, ref_grad_cand} =
        Nx.Defn.grad({f_gate, i_gate, candidates}, fn {f, i, cand} ->
          forward_fn.(f, i, cand) |> Nx.sum()
        end)

      {grad_f, grad_i, grad_cand, _} =
        Edifice.CUDA.FusedScan.minlstm_backward_fallback(f_gate, i_gate, candidates, h0, forward_out, grad_output)

      assert_all_close(grad_f, ref_grad_f, atol: 1.0e-3)
      assert_all_close(grad_i, ref_grad_i, atol: 1.0e-3)
      assert_all_close(grad_cand, ref_grad_cand, atol: 1.0e-3)
    end
  end

  # ============================================================================
  # Helpers
  # ============================================================================

  defp assert_all_close(actual, expected, opts) do
    atol = Keyword.get(opts, :atol, 1.0e-5)
    diff = Nx.abs(Nx.subtract(actual, expected))
    max_diff = Nx.to_number(Nx.reduce_max(diff))
    assert max_diff < atol, "max diff #{max_diff} exceeds tolerance #{atol}"
  end

  # Compute numerical gradient via central finite differences
  defp finite_diff_grad(tensor, f, eps) do
    flat = Nx.reshape(tensor, {Nx.size(tensor)})
    n = Nx.size(tensor)

    grads =
      for idx <- 0..(n - 1) do
        # +eps
        delta = Nx.indexed_put(Nx.broadcast(0.0, {n}), Nx.tensor([[idx]]), Nx.tensor([eps]))
        plus = Nx.reshape(Nx.add(flat, delta), Nx.shape(tensor))
        f_plus = Nx.to_number(f.(plus))

        # -eps
        minus = Nx.reshape(Nx.subtract(flat, delta), Nx.shape(tensor))
        f_minus = Nx.to_number(f.(minus))

        (f_plus - f_minus) / (2 * eps)
      end

    Nx.tensor(grads, type: {:f, 32}) |> Nx.reshape(Nx.shape(tensor))
  end
end
