defmodule Edifice.CUDA.NIF do
  @moduledoc false
  # Low-level NIF bindings for fused CUDA scan kernels.
  #
  # All functions take raw uint64 device pointer addresses and integer dimensions.
  # Use Edifice.CUDA.FusedScan for the high-level API that handles Nx tensors.
  #
  # The NIF is loaded lazily — if the .so doesn't exist (e.g. no CUDA),
  # all functions return :erlang.nif_error(:not_loaded).

  @on_load :load_nif

  def load_nif do
    path = :filename.join(:code.priv_dir(:edifice), ~c"libedifice_cuda_nif")

    case :erlang.load_nif(path, 0) do
      :ok -> :ok
      {:error, {:load_failed, _}} -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  @doc false
  def fused_mingru_scan(_gates_ptr, _cand_ptr, _h0_ptr, _batch, _seq_len, _hidden),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def fused_minlstm_scan(_forget_ptr, _input_ptr, _cand_ptr, _h0_ptr, _batch, _seq_len, _hidden),
    do: :erlang.nif_error(:not_loaded)

  @doc false
  def cuda_free(_device_ptr),
    do: :erlang.nif_error(:not_loaded)
end
