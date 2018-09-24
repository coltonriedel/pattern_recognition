// Colton Riedel
// Texas A&M ECEN 649
//
// Assignment #1, Problem #1
//
// 9/23/2018

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

size_t read_data(std::vector<std::vector<uint8_t>>& data, std::ifstream& file)
{
  // For reading bytes from file
  char c;
  // Array for buffering chars read from file, length 4 because maximum
  // expected value is 3 digits (255), with extra byte for terminating char
  char input[4];
  // Value control variable (0..724)
  size_t i = 0;
  // Input char control variable (0..3)
  size_t j = 0;

  while (file.get(c))
  {
    // End of value in CSV file
    if (c == ',' || c == '\n')
    {
      // Set terminating character
      input[j] = '\0';

      // Reset j for next value
      j = 0;

      // Parse value and push into storage and increment control variable
      data[i++].push_back(std::atoi(input));

      // If c was \n (end of line), reset value control variable
      if (c == '\n')
        i = 0;
    }
    // Else still reading part of value
    else
      input[j++] = c;
  }

  // Check that all features have the same size
  size_t num_records = data[0].size();
  for (auto feature : data)
    if (feature.size() != num_records)
      return 0;

  return data[0].size();
}

void train(std::vector<double>& priors,
    std::vector<std::vector<std::vector<double>>>& parameters,
    std::vector<std::vector<uint8_t>>& training_data)
{
  std::vector<double> class_counts;
  class_counts.resize(priors.size());

  // Calculate prior probabilities
  #pragma omp parallel for
  for (size_t i = 0; i < priors.size(); ++i)
  {
    class_counts[i] = std::count(training_data[0].begin(),
        training_data[0].end(), i);
    priors[i] = class_counts[i] / training_data[0].size();
  }

  // Calculate probabilities
  // For each feature (i = 0 .. num_features)
  #pragma omp parallel for
  for (size_t i = 0; i < parameters.size(); ++i)
    // For each feature value (j = 0 .. 255)
    for (size_t j = 0; j < parameters[i].size(); ++j)
      // For each class (k = 0 .. 9)
      for (size_t k = 0; k < parameters[i][j].size(); ++k)
      {
        size_t count = 1;

        // For each record
        for (size_t l = 0; l < training_data[0].size(); ++l)
          if (training_data[0][l] == k
              && training_data[i + 1][l] == j)
            count++;

        parameters[i][j][k] = count / (class_counts[k] + parameters[i].size());
      }
}

void evaluate(std::vector<double>& priors,
    std::vector<std::vector<std::vector<double>>>& parameters,
    std::vector<std::vector<uint8_t>>& test_set_data,
    std::vector<size_t>& prediction)
{
  // For each digit to predict
  #pragma omp parallel for
  for (size_t i = 0; i < test_set_data[0].size(); ++i)
  {
    // Set probability equal to log of prior probability
    std::vector<double> probability;
    for (auto prior : priors)
      probability.push_back(std::log(prior));

    // Determine probabiltiy of each class given feature value
    for (size_t feature = 1; feature < test_set_data.size(); ++feature)
      for (size_t classs = 0; classs < probability.size(); ++classs)
        probability[classs] +=
          std::log(parameters[feature - 1][test_set_data[feature][i]][classs]);

    // Pick highest probability class
    for (size_t classs = 0; classs < probability.size(); ++classs)
      if (probability[classs] > probability[prediction[i]])
        prediction[i] = classs;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3)
  {
    std::cout << "Usage: " << argv[0]
      << " train_set.csv test_set.csv" << std::endl;

    exit(1);
  }

  // Assume mnist datasets, consider generalizing later
  size_t class_domain = 10;
  size_t feature_domain = 256;
  size_t num_features = 784;
  size_t training_records;
  size_t test_set_records;

  std::string training_filename(argv[1]);
  std::string test_set_filename(argv[2]);

  // Vector (|test set|)
  std::vector<size_t> prediction;

  // 2D matrix (|features| + 1) * (|training set|)
  std::vector<std::vector<uint8_t>> training_data;
  training_data.resize(1 /* class */ + num_features);

  // 2D matrix (|features| + 1) * (|test set|)
  std::vector<std::vector<uint8_t>> test_set_data;
  test_set_data.resize(1 /* class */ + num_features);

  // Vector (|class_domain|)
  std::vector<double> priors;
  priors.resize(class_domain);

  // 3D matrix (|features| * |feature_domain| * |class_domain|)
  std::vector<std::vector<std::vector<double>>> parameters;
  parameters.resize(num_features);
  for (size_t i = 0; i < parameters.size(); ++i)
  {
    parameters[i].resize(feature_domain);

    for (size_t j = 0; j < parameters[i].size(); ++j)
      parameters[i][j].resize(class_domain);
  }

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

  // Train the model
  start = std::chrono::high_resolution_clock::now();
  train(priors, parameters, training_data);
  stop = std::chrono::high_resolution_clock::now();

  std::cout << "Trained model in " << std::chrono::duration_cast<
      std::chrono::duration<double>>(stop - start).count()
    << " seconds" << std::endl;

  // Evaluate the test data
  start = std::chrono::high_resolution_clock::now();
  evaluate(priors, parameters, test_set_data, prediction);
  stop = std::chrono::high_resolution_clock::now();

  // Determine number of correct predictions
  size_t correct = 0;
  for (size_t i = 0; i < test_set_records; ++i)
    if (prediction[i] == test_set_data[0][i])
      correct++;

  std::cout << "\nEvaluated " << test_set_records << " test records in "
    << std::chrono::duration_cast<
      std::chrono::duration<double>>(stop - start).count()
    << " seconds with "
    << (correct * 100.0) / test_set_records
    << "\% accuracy" << std::endl;

  std::cout << "\nDigit\t Precision\t Recall" << std::endl;
  // Compute precision and recall for all digits
  for (size_t i = 0; i < class_domain; ++i)
  {
    size_t tp = 0;
    size_t fp = 0;
    size_t fn = 0;

    for (size_t j = 0; j < test_set_records; ++j)
      if (prediction[j] == i && test_set_data[0][j] == i)
        tp++;
      else if (prediction[j] == i && test_set_data[0][j] != i)
        fp++;
      else if (prediction[j] != i && test_set_data[0][j] == i)
        fn++;

    std::cout << i << "\t " << (tp / static_cast<double>(tp + fn))
      << "\t " << (tp / static_cast<double>(tp + fp)) << std::endl;
  }

  // Close files, if open
  if (training_file.is_open())
    training_file.close();

  if (test_set_file.is_open())
    test_set_file.close();
}
