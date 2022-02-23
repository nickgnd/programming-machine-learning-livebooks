# Based on sonar_classifier

Mix.install([
  {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
  {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true}
])

# Set the backend
Nx.Defn.global_default_options(compiler: EXLA)

defmodule C7.SonarDataset do
  @moduledoc """
  Use this Module to load the Sonar database (test, train, and labels).

  Sonar dataset specifications can be found here: https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)
  The documentation of the dataset can be found in the `sonar.names` file.
  """

  @sonar_all_data_filename "../data/sonar/sonar.all-data"

  @type t :: %__MODULE__{
          x_train: Nx.Tensor.t(),
          x_test: Nx.Tensor.t(),
          y_train: Nx.Tensor.t(),
          y_test: Nx.Tensor.t()
        }
  defstruct [:x_train, :x_test, :y_train, :y_test]

  @doc """
  Load the Sonar database and return the train and test data.

  The function accepts the argument `rnd_seed` to initialize the Random algorithm
  before shuffling the list of patterns.
  The given number should consider that the are in total 208 patterns (111 from metal,
  97 from rocks). The default value is 48, which is ~23% of the whole dataset.
  """
  @spec load(rnd_seed :: integer()) :: t()
  def load(rnd_seed) do
    # The file "sonar.all-data" contains 208 patterns:
    # - 111 patterns obtained by bouncing sonar signals off a
    # metal cylinder at various angles and under various conditions.
    # - 97 patterns obtained from rocks under similar conditions.
    #
    # - The patterns are ordered: first the 97 rocks ones and then the 111 metals ones.
    # - Each pattern is a set of 60 numbers in the range 0.0 to 1.0, followed by either
    # `M` or `R` depending if it has been obtained from a metal or rock.
    #

    :rand.seed(:exsss, rnd_seed)

    with {:ok, binary} <- File.read(@sonar_all_data_filename) do
      data =
        binary
        |> parse()
        |> Enum.shuffle()

      # Keep 48 examples for testing
      {train_data, test_data} = Enum.split(data, length(data) - 48)

      {x_train, y_train} = split_inputs_and_labels(train_data)
      {x_test, y_test} = split_inputs_and_labels(test_data)

      %__MODULE__{
        x_train: prepend_bias(Nx.tensor(x_train)),
        x_test: prepend_bias(Nx.tensor(x_test)),
        y_train: Nx.tensor(y_train) |> Nx.reshape({:auto, 1}),
        y_test: Nx.tensor(y_test) |> Nx.reshape({:auto, 1})
      }
    end
  end

  defp split_inputs_and_labels(data) do
    Enum.reduce(data, {[], []}, fn [pattern, label], {x, y} = _acc ->
      {x ++ [pattern], y ++ [label]}
    end)
  end

  defp pattern_type_to_label("M"), do: 1
  defp pattern_type_to_label("R"), do: 0

  defp parse(binary) do
    binary
    |> String.split("\n", trim: true)
    |> Enum.map(&String.split(&1, ",", trim: true))
    |> Enum.map(fn row ->
      {pattern_type, pattern} = List.pop_at(row, -1)

      [
        Enum.map(pattern, &String.to_float/1),
        pattern_type_to_label(pattern_type)
      ]
    end)
  end

  @doc """
  One-hot encode the given tensor (classes: either 0 or 1).
  """
  @spec one_hot_encode(y :: Nx.Tensor.t()) :: Nx.Tensor.t()
  def one_hot_encode(y) do
    Nx.equal(y, Nx.tensor([0, 1]))
  end

  @doc """
  Prepend a the bias, an extra column of 1s, to
  the given tensor.
  """
  @spec prepend_bias(Nx.Tensor.t()) :: Nx.Tensor.t()
  def prepend_bias(x) do
    bias = Nx.broadcast(1, {elem(Nx.shape(x), 0), 1})

    # Insert a column of 1s in the position 0 of x.
    # ("axis: 1" stands for: "insert a column, not a row")
    # in python: `np.insert(X, 0, 1, axis=1)`
    Nx.concatenate([bias, x], axis: 1)
  end
end

defmodule C7.Classifier do
  import Nx.Defn

  @doc """
  A sigmoid function is a mathematical function having
  a characteristic "S"-shaped curve or sigmoid curve.

  A sigmoid function:
  - is monotonic
  - has no local minimums
  - has a non-negative derivative for each point

  More here https://en.wikipedia.org/wiki/Sigmoid_function
  """
  @spec sigmoid(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn sigmoid(z) do
    Nx.divide(1, Nx.add(1, Nx.exp(Nx.negate(z))))
  end

  @doc """
  Return the prediction tensor ŷ (y_hat) given the inputs and weight.
  The returned tensor is a matrix with the same dimensions as
  the weighted sum: one row per example, and one column.
  Each element in the matrix is now constrained between 0 and 1.
  """
  @spec forward(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn forward(x, weight) do
    weighted_sum = Nx.dot(x, weight)
    sigmoid(weighted_sum)
  end

  @doc """
  Return the prediction rounded to forecast a value between 0 and 9.
  """
  @spec classify(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn classify(x, weight) do
    y_hat = forward(x, weight)

    # Get the index of the maximum value in each row of y_hat
    # (the value that’s closer to 1).
    # NOTE: in case of MNIST dataset, the returned index is also the
    # decoded label (0..9).
    labels = Nx.argmax(y_hat, axis: 1)

    Nx.reshape(labels, {:auto, 1})
  end

  @doc """
  Log loss function.
  """
  @spec loss(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn loss(x, y, weight) do
    y_hat = forward(x, weight)

    # Each label in the matrix `y_hat` is either `0` or `1`.
    # - `first_term` disappears when `y_hat` is 0
    # - `second_term` disappears when `y_hat` is 1
    first_term = y * Nx.log(y_hat)
    second_term = Nx.subtract(1, y) * Nx.log(Nx.subtract(1, y_hat))

    # Corrected version (Chapter 7)
    Nx.add(first_term, second_term)
    |> Nx.sum()
    |> Nx.divide(elem(Nx.shape(x), 0))
    |> Nx.negate()
  end

  @doc """
  Returns the derivate of the loss curve.
  """
  @spec gradient(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn gradient(x, y, weight) do
    # in python:
    # np.matmul(X.T, (predict(X, w) - Y)) / X.shape[0]

    predictions = forward(x, weight)
    errors = Nx.subtract(predictions, y)
    n_examples = elem(Nx.shape(x), 0)

    Nx.transpose(x)
    |> Nx.dot(errors)
    |> Nx.divide(n_examples)
  end

  @typep report_item :: {
           iteration :: integer(),
           training_loss :: float(),
           matches_percentage :: float()
         }

  @doc """
  Utility to compute training loss and matches % per iteration,
  use this to report the result.
  """
  @spec report(
          integer(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t()
        ) :: report_item()
  def report(iteration, x_train, y_train, x_test, y_test, weight) do
    matches = matches(x_test, y_test, weight) |> Nx.to_number()
    n_test_examples = elem(Nx.shape(y_test), 0)
    matches = matches * 100.0 / n_test_examples
    training_loss = loss(x_train, y_train, weight) |> Nx.to_number()

    # Commented to don't freeze the browser
    # IO.inspect("Iteration #{iteration} => Loss: #{training_loss}, #{matches}%")
    {iteration, training_loss, matches}
  end

  defnp matches(x_test, y_test, weight) do
    classify(x_test, weight)
    |> Nx.equal(y_test)
    |> Nx.sum()
  end

  @doc """
  Computes the weight by training the system
  with the given inputs and labels, by iterating
  over the examples the specified number of times.

  It returns a tuple with the final weight and the
  reports of the loss per iteration.
  """
  @spec train(
          train_inputs :: Nx.Tensor.t(),
          train_labels :: Nx.Tensor.t(),
          test_inputs :: Nx.Tensor.t(),
          test_labels :: Nx.Tensor.t(),
          iterations :: integer(),
          learning_rate :: float()
        ) :: {weight :: Nx.Tensor.t(), reports :: [report_item()]}
  def train(x_train, y_train, x_test, y_test, iterations, lr) do
    init_weight = init_weight(x_train, y_train)
    init_reports = [report(0, x_train, y_train, x_test, y_test, init_weight)]

    {final_weight, reversed_reports} =
      Enum.reduce(1..(iterations - 1), {init_weight, init_reports}, fn i, {weight, reports} ->
        new_weight = step(x_train, y_train, weight, lr)
        updated_reports = [report(i, x_train, y_train, x_test, y_test, new_weight) | reports]

        {new_weight, updated_reports}
      end)

    {final_weight, Enum.reverse(reversed_reports)}
  end

  defnp step(x, y, weight, lr) do
    Nx.subtract(weight, Nx.multiply(gradient(x, y, weight), lr))
  end

  # Returns a tensor of shape `{n, m}`, where
  # `n` is the number of columns in `x` (input variables) and
  # `m` is the number of columns in `y` (classes).
  # Each element in the tensor is initialized to 0.
  defnp init_weight(x, y) do
    n_input_variables = elem(Nx.shape(x), 1)
    n_classes = elem(Nx.shape(y), 1)
    Nx.broadcast(0, {n_input_variables, n_classes})
  end
end

seed_range = 0..100

results =
  Enum.reduce(seed_range, [], fn seed, acc ->
    IO.puts "Seed #{seed} in progress..."

    %{x_train: x_train, x_test: x_test, y_train: y_train, y_test: y_test} =
      C7.SonarDataset.load(seed)

    y_train = C7.SonarDataset.one_hot_encode(y_train)

    {weight, reports} =
      C7.Classifier.train(x_train, y_train, x_test, y_test, iterations = 100_000, lr = 0.01)

    max_match_report = Enum.max_by(reports, &elem(&1, 2))

    acc ++ [{seed, Enum.at(reports, -1), max_match_report}]
  end)

content =
  Enum.map(results, fn {seed, last_report, max_report} ->
    "seed: #{seed}, last report: {#{elem(last_report, 0)}, #{elem(last_report, 2)}%}, max report: {#{elem(max_report, 0)}, #{elem(max_report, 2)}%}"
  end)
  |> Enum.join("\n")

File.write!("./seed_results.txt", content)
