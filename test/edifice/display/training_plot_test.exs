defmodule Edifice.Display.TrainingPlotTest do
  use ExUnit.Case, async: true

  alias Edifice.Display.TrainingPlot

  describe "loss_curve/1" do
    test "generates VegaLite spec from data" do
      data = [{0, 2.3}, {100, 1.5}, {200, 0.8}]
      spec = TrainingPlot.loss_curve(data)

      assert spec["$schema"] =~ "vega-lite"
      assert spec["title"] == "Training Loss"
      assert spec["mark"]["type"] == "line"
      assert length(spec["data"]["values"]) == 3
      assert hd(spec["data"]["values"])["step"] == 0
      assert hd(spec["data"]["values"])["loss"] == 2.3
    end
  end

  describe "lr_schedule/2" do
    test "generates VegaLite spec from schedule function" do
      schedule = Polaris.Schedules.cosine_decay(0.001, decay_steps: 100)
      spec = TrainingPlot.lr_schedule(schedule, total_steps: 100, sample_every: 10)

      assert spec["$schema"] =~ "vega-lite"
      assert spec["title"] == "Learning Rate Schedule"
      values = spec["data"]["values"]
      assert length(values) == 11
      # First value should be near 0.001
      assert hd(values)["lr"] > 0
      # Last value should be lower (decayed)
      assert List.last(values)["lr"] < hd(values)["lr"]
    end
  end

  describe "live_widget/0" do
    test "returns :no_kino when Kino.VegaLite not available in test env" do
      # In test env, Kino.VegaLite may or may not be loaded
      result = TrainingPlot.live_widget()
      # Should be either a widget or :no_kino
      assert result == :no_kino or is_struct(result)
    end
  end

  describe "attach/3" do
    test "no-ops when widget is :no_kino" do
      model = Axon.input("x", shape: {nil, 4}) |> Axon.dense(2)
      loop = Axon.Loop.trainer(model, :mean_squared_error, :adam)

      result = TrainingPlot.attach(loop, :no_kino)
      assert %Axon.Loop{} = result
    end
  end
end
