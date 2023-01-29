Mix.install(
  [
    {:exla, "~> 0.4"},
    {:nx, "~> 0.4"},
    {:axon, "~> 0.4.1"},
    {:kino, "~> 0.8.0"},
    {:kino_vega_lite, "~> 0.1.7"},
    {:vega_lite, "~> 0.1.6"},
    {:table_rex, "~> 3.1.1"}
  ],
  config: [nx: [default_backend: EXLA.Backend]]
)

defmodule C17.EchidnaDataset do
  import Nx.Defn

  @filename Path.join(["..", "data", "echidna.txt"])

  @doc """
  Loads the echidna dataset and returns the input `x` and label `y` tensors.

  - the dataset has been shuffled
  - the input tensor is already normalized
  """
  def load() do
    with {:ok, binary} <- read_file() do
      # seed the random algorithm
      :rand.seed(:exsss, {1, 2, 3})

      tensor =
        binary
        |> parse()
        |> Enum.shuffle()
        |> Nx.tensor()

      # all the rows, only first 2 columns
      x = tensor[[0..-1//1, 0..1//1]] |> normalize_inputs()

      # all the rows, only 3rd column
      y =
        tensor[[0..-1//1, 2]]
        |> Nx.reshape({:auto, 1})
        |> Nx.as_type(:u8)

      %{x: x, y: y}
    end
  end

  def parse(binary) do
    binary
    |> String.split("\n", trim: true)
    |> Enum.slice(1..-1)
    |> Enum.map(fn row ->
      row
      |> String.split(" ", trim: true)
      |> Enum.map(&parse_float/1)
    end)
  end

  # Normalization (Min-Max Scalar)
  #
  # In this approach, the data is scaled to a fixed range — usually 0 to 1.
  # In contrast to standardization, the cost of having this bounded range
  # is that we will end up with smaller standard deviations,
  # which can suppress the effect of outliers.
  # Thus MinMax Scalar is sensitive to outliers.
  defnp normalize_inputs(x_raw) do
    # Compute the min/max over the first axe
    min = Nx.reduce_min(x_raw, axes: [0])
    max = Nx.reduce_max(x_raw, axes: [0])

    # After MinMaxScaling, the distributions are not centered
    # at zero and the standard deviation is not 1.
    # Thefore, subtract 0.5 to rescale data between -0.5 and 0.5
    (x_raw - min) / (max - min) - 0.5
  end

  # to handle both integer and float numbers
  defp parse_float(stringified_float) do
    {float, ""} = Float.parse(stringified_float)
    float
  end

  def read_file() do
    if File.exists?(@filename) do
      File.read(@filename)
    else
      {:error, "The file #{@filename} is missing!"}
    end
  end
end

%{x: x_all, y: y_all} = C17.EchidnaDataset.load()

size = (elem(Nx.shape(x_all), 0) / 3) |> ceil()

[x_train, x_validation, x_test] = Nx.to_batched(x_all, size) |> Enum.to_list()
[y_train, y_validation, y_test] = Nx.to_batched(y_all, size) |> Enum.to_list()

data = %{
  x_train: x_train,
  x_validation: x_validation,
  x_test: x_test,
  y_train: y_train,
  y_validation: y_validation,
  y_test: y_test
}

x_train = data.x_train
x_validation = data.x_validation

# One-hot encode the labels
y_train = Nx.equal(data.y_train, Nx.tensor(Enum.to_list(0..1)))
y_validation = Nx.equal(data.y_validation, Nx.tensor(Enum.to_list(0..1)))

# Prepare batches
batch_size = 25

train_inputs = Nx.to_batched(x_train, batch_size)
train_labels = Nx.to_batched(y_train, batch_size)
train_batches = Stream.zip(train_inputs, train_labels)

validation_data = [{x_validation, y_validation}]

model =
  Axon.input("data")
  |> Axon.dense(100, activation: :sigmoid)
  |> Axon.dense(30, activation: :sigmoid)
  |> Axon.dense(2, activation: :softmax)

# `output_transform/1` applies a transformation on the final accumulated loop state.
#
# At the moment `axon` does not provide a clean API to override/set it,
# therefore we use an "hack" (`Map.update`) to override its value in the Loop's state.
#
# https://hexdocs.pm/axon/Axon.Loop.html#loop/3
# https://github.com/elixir-nx/axon/blob/d180f074c33cf841fcbaf44c8e66d677c364d713/test/axon/loop_test.exs#L1073-L1080
output_transform = fn %{step_state: step_state, metrics: metrics} ->
  %{params: step_state[:model_state], metrics: metrics}
end

epochs = 15_000

loop =
  model
  |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.rmsprop(0.001))
  |> Axon.Loop.validate(model, validation_data)
  |> Map.update(:output_transform, nil, fn _original_output_transform ->
    fn state -> output_transform.(state) end
  end)

{microsec, %{params: params, metrics: metrics}} = :timer.tc(fn ->
  Axon.Loop.run(loop, train_batches, %{}, epochs: epochs, compiler: EXLA)
end)

IO.inspect("TRAINING CONCLUDED IN #{ceil(microsec / (60 * 1_000_000))} minutes.")

sorted_metrics =
  metrics
  |> Enum.to_list()
  |> Enum.sort_by(& elem(&1, 0))
  |> Enum.map(& elem(&1, 1))

losses = Enum.map(sorted_metrics, & Map.fetch!(&1, "loss") |> Nx.to_number())
validation_losses = Enum.map(sorted_metrics, & Map.fetch!(&1, "validation_loss") |> Nx.to_number())

File.write!("./losses_1", Enum.join(losses, "\n"))
File.write!("./validation_losses_1", Enum.join(validation_losses, "\n"))

{:ok, model_params} = File.open("./model_params_1.term", [:write])
Nx.serialize(params) |> then(& IO.binwrite(model_params, &1))
:ok = File.close(model_params)
