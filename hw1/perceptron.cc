// Colton Riedel
// Texas A&M ECEN 649
//
// Assignment #1, Problem #2
//
// 9/23/2018

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

size_t read_data(std::vector<std::vector<double>>& data, std::ifstream& file)
{
  // For reading bytes from file
  char c;
  // Array for buffering chars read from file
  char input[5];
  // Value control variable (0..2)
  size_t i = 0;
  // Input char control variable (0..4)
  size_t j = 0;

  // Throw away the first line (headers)
  file.ignore(std::numeric_limits<std::streamsize>::max(), '\r');

  while (file.get(c))
  {
    // End of value in CSV file
    if (c == ',' || c == '\r')
    {
      // Set terminating character
      input[j] = '\0';

      // Reset j for next value
      j = 0;

      // Parse value and push into storage and increment control variable
      data[i++].push_back(std::atof(input));

      // If c was \r (end of line), reset value control variable
      if (c == '\r')
        i = 0;
    }
    // Else still reading part of value
    else
      input[j++] = c;
  }

  // Check that all features have the same size
  size_t num_records = data[0].size();
  for (const auto& feature : data)
    if (feature.size() != num_records)
      return 0;

  return data[0].size();
}

size_t train(std::array<double, 3>& weight,
    std::vector<std::vector<double>>& training_data)
{
  size_t iterations = 0;

  while (true)
  {
    size_t errors = 0;

    // Evaluate every record in training data
    for (size_t i = 0; i < training_data[0].size(); ++i)
    {
      // Make prediction based on current weights
      double prediction = weight[0] * training_data[0][i]
        + weight[1] * training_data[1][i]
        + weight[2];

      // If incorrect prediction (different signs), update weights
      if ((prediction < 0 && training_data[2][i] > 0)
          || (prediction > 0 && training_data[2][i] < 0))
      {
        errors++;

        weight[0] += training_data[2][i] * training_data[0][i];
        weight[1] += training_data[2][i] * training_data[1][i];
        weight[2] += training_data[2][i];
      }
    }

    // If there were no errors, exit the loop
    if (errors == 0)
      break;
    else
      iterations++;
  }

  return iterations;
}

void evaluate(std::array<double, 3>& weight,
    std::vector<std::vector<double>>& test_set_data,
    std::vector<double>& prediction)
{
  for (size_t i = 0; i < test_set_data[0].size(); ++i)
    prediction[i] = weight[0] * test_set_data[0][i]
      + weight[1] * test_set_data[1][i] + weight[2];
}

int main(int argc, char* argv[])
{
  if (argc != 3)
  {
    std::cout << "Usage: " << argv[0]
      << " train_set.csv test_set.csv" << std::endl;

    exit(1);
  }

  size_t num_features = 2;
  size_t training_records;
  size_t test_set_records;

  std::array<double, 3> weight = {0.01, 0.01, 0.01};

  std::string training_filename(argv[1]);
  std::string test_set_filename(argv[2]);

  // Vector (|test set|)
  std::vector<double> prediction;

  // 2D matrix (|features| + 1) * (|training set|)
  std::vector<std::vector<double>> training_data;
  training_data.resize(1 /* class */ + num_features);

  // 2D matrix (|features| + 1) * (|test set|)
  std::vector<std::vector<double>> test_set_data;
  test_set_data.resize(1 /* class */ + num_features);

  // Open training and test set CSVs
  std::ifstream training_file(training_filename, std::ios::binary);
  std::ifstream test_set_file(test_set_filename, std::ios::binary);

  // Parse training data CSV
  auto start = std::chrono::high_resolution_clock::now();
  training_records = read_data(training_data, training_file);
  auto stop = std::chrono::high_resolution_clock::now();

  if (training_records == 0)
  {
    std::cout << "Parsing training data set failed, aborting" << std::endl;

    // Close files, if open
    if (training_file.is_open())
      training_file.close();

    if (test_set_file.is_open())
      test_set_file.close();

    exit(1);
  }
  else
    std::cout << "Parsed " << training_records << " training set records in "
      << std::chrono::duration_cast<
        std::chrono::duration<double>>(stop - start).count()
      << " seconds" << std::endl;

  // Parse test data CSV
  start = std::chrono::high_resolution_clock::now();
  test_set_records = read_data(test_set_data, test_set_file);
  stop = std::chrono::high_resolution_clock::now();

  if (test_set_records == 0)
  {
    std::cout << "Parsing test set data set failed, aborting" << std::endl;

    // Close files, if open
    if (training_file.is_open())
      training_file.close();

    if (test_set_file.is_open())
      test_set_file.close();

    exit(1);
  }
  else
    std::cout << "Parsed " << test_set_records << " test set records in "
      << std::chrono::duration_cast<
        std::chrono::duration<double>>(stop - start).count()
      << " seconds" << std::endl;

  prediction.resize(test_set_records);

  // Train the perceptron
  start = std::chrono::high_resolution_clock::now();
  size_t iter = train(weight, training_data);
  stop = std::chrono::high_resolution_clock::now();

  std::cout << "\nTrained perceptron with " << iter << " iterations in "
    << std::chrono::duration_cast<
      std::chrono::duration<double>>(stop - start).count()
    << " seconds" << std::endl;

  // Make predictions for the test set
  start = std::chrono::high_resolution_clock::now();
  evaluate(weight, test_set_data, prediction);
  stop = std::chrono::high_resolution_clock::now();

  // Determine number of correct predictions
  size_t correct = 0;
  for (size_t i = 0; i < test_set_records; ++i)
    if ((prediction[i] < 0 && test_set_data[2][i] < 0)
        || (prediction[i] > 0 && test_set_data[2][i] > 0))
      correct++;

  std::cout << "\nEvaluated " << test_set_records << " test records in "
    << std::chrono::duration_cast<
      std::chrono::duration<double>>(stop - start).count()
    << " seconds with "
    << (correct * 100.0) / test_set_records
    << "\% accuracy" << std::endl;

  std::cout << "Weights: y = x_1 * (" << weight[0] << ") + x_2 * (" << weight[1]
   << ") + " << weight[2] << std::endl;

  // Close files, if open
  if (training_file.is_open())
    training_file.close();

  if (test_set_file.is_open())
    test_set_file.close();
}
