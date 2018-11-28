// Colton Riedel & Ryan Vrecenar
// Texas A&M ECEN 649
//
// Course project
//
// Implementation of the Gecko clustering algorithm, RIPPER classification
//   algorithm, and definition of state transition logic to construct a finite
//   state atomaton for the detection of time series anomalies. Applied to GNSS
//   recevier clock drift rate data
//
// 11/28/2018

#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

/* Class to represent the graph of the time series data
 *
 * Because of the nature of how the graph will be used, we store only the nodes,
 * corresponding to time series values, and edge cut values (i.e. individual
 * edges - which are parametric, and edges values - which can be calculated as
 * a function of node values but aren't used beyond making edge cuts, are not
 * stored)
 */
class graph
{
public:
  // Vector to represent the nodes in the graph, size n
  //   Each index represents a time series data point
  //   e.g.: node[5] is the value of the time series at t=6
  std::vector<int32_t> node;

  // Vector to represent the edge cut values in the graph, size n-1
  //   Each index represents the edge cut between two time series points
  //
  //   e.g.: let min cluster size s = 5, then k = 2*s = 10
  //
  //         node[20] (t=21) has edges to its k=10 nearest neighbors, where we
  //         assume the nearest neighbors are determined using temporal
  //         locality
  //
  //         so then the node representing t=21 has edges to the k/2 previous
  //         time series points, and the k/2 future time series points
  //
  //         egdes: 21-20, 21-19, 21-18, 21-17, 21-16
  //                21-22, 21-23, 21-24, 21-25, 21-26
  //
  //         to reduce space and simplify later calculation, we store all edge
  //         weights only once, so edge[20] contains the following weights:
  //           21-22, 21-23, 21-24, 21-25, 21-26
  //
  //         plus, the applicable weights from edges which would be included in
  //         this cut, but do not involve the node for t=21, such as:
  //           20-22, 20-23, 20-24, 20-25
  //           19-22, 19-23, 19-24
  //           18-22, 19-23
  //           17-22
  //
  //         Note that only one edge exists between any pair of nodes, so the
  //         edge 17-22 and 22-17 is not repeated, and only included once in the
  //         edge-cut sum
  std::vector<double> edge;
};

/* Class to represent a cluster within the graph, including the first and last
 * time series points and representative values
 */
class cluster
{
public:
  // Index of the first and last nodes in the cluster
  size_t start;
  size_t end;

  // Representative value of the cluster
  double value;

  // Pointers to any sub clusters which might exist
  cluster* left;
  cluster* right;

  cluster(size_t start, size_t end, double value, cluster* left, cluster* right)
    : start(start), end(end), value(value), left(left), right(right)
  { }

  cluster(size_t start, size_t end, graph g)
    : start(start), end(end), left(nullptr), right(nullptr)
  {
    // Compute value of cluster
    //   need to regress over points
  }

  // Return the number of nodes in the cluster (equivalent to duration as
  // clusters are temporal)
  size_t size()
  {
    return end - start;
  }

  // Overload operator+ to represent merging two clusters, input is rhs cluster,
  // output is a new cluster composed of the two sub clusters
  cluster operator+(cluster& rhs)
  {
    if (end+1 != rhs.start)
      throw std::runtime_error("Error: attempt to join non-adjacent clusters");

    // TODO finish - need to calculate value

    return { start, rhs.end, value, this, &rhs };
  }
};

/* Function to read a CSV file of time series data and generate the
 *   corresponding graph
 *
 * INPUT:
 *   csv_filename : filename
 *      field_num : the column number of interest within the CSV file
 *     has_header : indicates if the CSV file has a header (i.e. the first line
 *                    is text and should be discarded)
 * OUTPUT:
 *   Returns an edge(cut)-less graph of the time series data
 */
graph read_graph_nodes(const std::string& csv_filename, size_t field_num,
    bool has_header=true)
{
  // Create the graph
  graph g;

  // Open the CSV file
  std::ifstream file(csv_filename, std::ios::binary);

  // For reading bytes from file
  char c;
  // Array for buffering chars read from file
  // Length set to 11 because maximum value of int32_t is 2147483647 (10 digits
  //   plus \0 character)
  char input[11];
  // Field control variable
  size_t i = 0;
  // Input char control variable (0..3)
  size_t j = 0;

  // If file has a header, throw it away
  if (has_header)
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  while (file.get(c))
  {
    // End of value in CSV file
    if (c == ',' || c == '\n')
    {
      // If we're actually in the field of interest
      if (i == field_num)
      {
        // Set terminating character
        input[j] = '\0';

        // Reset j for next time series value
        j = 0;

        // Parse value and push into storage and increment control variable
        g.node.push_back(std::atoi(input));
      }

      // If c was , increment the field number
      if (c == ',')
        i++;
      // If c was \n (end of line), reset field control variable
      else if (c == '\n')
        i = 0;
    }
    // If not end of field or line, and within field of interest, buffer digit
    else if (i == field_num)
      input[j++] = c;
    // Else within field we weren't interested in, throw away value
  }

  // Close file, if open
  if (file.is_open())
    file.close();
  else // File wasn't open for some reason
  {
    std::cerr << "File wasn't open, aborting" << std::endl;

    exit(1);
  }

  return g;
}

/* Function to determine the edge cut weights for the time series graph
 *
 * INPUT:
 *   graph : the graph containing the nodes and their values
 *       s : the minimum cluster size
 * OUTPUT:
 *   Updates edge member of graph passed in by reference
 */
void calc_edge_cuts(graph& g, size_t s)
{
  // Resize edge cut weight vector
  g.edge.resize(g.node.size() - 1);

  // For each node in the graph
  // (skip the last node because there are no future time points for it to
  //  connect to)
  for (size_t n = 0; n < g.node.size() - 1; ++n)
  {
    // For the k nearest neighbors (equivalent to k/2 = s future time points
    //   since we don't want to double count edges)
    for (size_t e = 1; e < s; ++e)
    {
      // Don't attempt to connect to nodes that don't exist
      if (n + e < g.node.size())
      {
        double d = g.node[n] - g.node[n + e];

        // Calculate the euclidian distance between points and add it to the
        // running sum of edge cut weights
        g.edge[n] += std::log(1.0 / (std::sqrt(e * e + 128 * d * d) + 1));
        g.edge[n + e] += std::log(1.0 / (std::sqrt(e * e + 128 * d * d) + 1));
      }
    }
  }

  // Print out value and edge cut weight for visualization
  //std::cout << "index,value,cut_weight" << std::endl;
  //for (size_t i = 0; i < g.edge.size(); ++i)
  //  std::cout << i << "," << g.node[i] << "," << g.edge[i] << std::endl;

  // At this point we have all the edge cuts completed with the exception of the
  // s-1 first time series points and s-1 last points, but technically we don't
  // need to have those values calculated anyways since the smallest cluster
  // permissible is size s (i.e. we're not allowed to select those edges for
  // cutting anyways)
}

int main(int argc, char* argv[])
{
  // If incorrect number of args are supplied, abort
  if (argc != 4)
  {
    std::cout << "Usage: " << argv[0] << " filename field_number s"
      << std::endl;

    exit(1);
  }

  // Parse filename, csv field number (clkd), gecko parameter s
  std::string csv_filename(argv[1]);
  size_t field_num = std::atol(argv[2]);
  size_t s = std::atol(argv[3]);

  // Read in CSV file and populate graph
  auto start = std::chrono::high_resolution_clock::now();
  graph g = read_graph_nodes(csv_filename, field_num, true);
  auto stop = std::chrono::high_resolution_clock::now();

  std::cout << g.node.size() << " time series points were loaded in "
    << std::chrono::duration_cast<
      std::chrono::duration<double>>(stop - start).count()
    << " seconds" << std::endl;

  // If min cluster size is so large that we wouldn't do anything, don't
  if (g.node.size() < s*3)
  {
    std::cout << "S is too large, nothing meaningful to do" << std::endl;

    exit(1);
  }

  // Finish constructing graph by setting edge (cut) values
  start = std::chrono::high_resolution_clock::now();
  calc_edge_cuts(g, s);
  stop = std::chrono::high_resolution_clock::now();

  std::cout << "Determined edge cut weights in "
    << std::chrono::duration_cast<
      std::chrono::duration<double>>(stop - start).count()
    << " seconds" << std::endl;

  if (g.node.size() != g.edge.size() + 1)
  {
    std::cerr << "Incorrect number of edge cut weights given graph size"
      << std::endl;

    exit(1);
  }

  // Bisect graph recursively into clusters
  // form_clusters(g, s);

  // Recursively merge clusters
  // merge_clusters(g);

  // Classify using RIPPER algorithm
  //

  // Dump values? (add tag so states can be graphed easily)
  // When to determine and apply state transition logic
}
