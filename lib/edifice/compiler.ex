defmodule Edifice.Compiler do
  @moduledoc """
  Thin wrapper around `Axon.build/2` that enables EXLA disk caching with
  deterministic cache paths.

  Caches compiled XLA executables to disk so subsequent BEAM sessions skip
  XLA compilation entirely (cold-start drops from seconds to milliseconds).
  """

  @cache_dir ".cache/exla"

  @doc """
  Build an Axon model with EXLA compilation and persistent disk cache.

  Equivalent to `Axon.build(model, compiler: EXLA)` but caches the compiled
  executable to disk so subsequent runs skip XLA compilation entirely.

  ## Options
    - `:cache_dir` - Base directory for cache files (default: `.cache/exla`)
    - `:cache_key` - Unique key for this model config (required)
    - `:mode` - `:inference` or `:train` (default: `:inference`)
    - All other opts forwarded to `Axon.build/2`

  ## Returns
    `{init_fn, predict_fn}` — same as `Axon.build/2`
  """
  def build(model, opts \\ []) do
    {cache_key, opts} = Keyword.pop!(opts, :cache_key)
    {cache_dir, opts} = Keyword.pop(opts, :cache_dir, @cache_dir)

    cache_path = Path.join([cache_dir, "#{cache_key}.exla"])
    File.mkdir_p!(Path.dirname(cache_path))

    opts = Keyword.merge([compiler: EXLA, cache: cache_path], opts)
    Axon.build(model, opts)
  end

  @doc "List cached compilation artifacts."
  def list_cached(cache_dir \\ @cache_dir) do
    case File.ls(cache_dir) do
      {:ok, files} -> Enum.filter(files, &String.ends_with?(&1, ".exla"))
      {:error, _} -> []
    end
  end

  @doc "Clear all cached compilations."
  def clear_cache(cache_dir \\ @cache_dir) do
    File.rm_rf(cache_dir)
    :ok
  end

  @doc "Clear a specific cached compilation."
  def clear_cached(cache_key, cache_dir \\ @cache_dir) do
    Path.join(cache_dir, "#{cache_key}.exla") |> File.rm()
  end
end
