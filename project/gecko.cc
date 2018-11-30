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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
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

  // Indices of myself, any sub clusters, and parent
  int self;
  int left;
  int right;
  int parent = -1;

  cluster(size_t start, size_t end, double value, int left, int right)
    : start(start), end(end), value(value), left(left), right(right)
  { }

  cluster(size_t start, size_t end, int index, graph& g)
    : start(start), end(end), self(index), left(-1), right(-1)
  {
    double sum = 0;

    for (size_t i = start; i < end - 1; ++i)
      sum += (g.node[i] - g.node[i + 1]);

    value = sum / (end - start);
  }

  // Return the number of nodes in the cluster (equivalent to duration as
  // clusters are temporal)
  size_t size()
  {
    return end - start;
  }

  // Funtion to return if the cluster has a parent (in effect, if it is eligible
  // to be merged with some other cluster
  bool mergeable() const
  {
    return parent == -1;
  }

  // Overload operator+ to represent merging two clusters, input is rhs cluster,
  // output is a new cluster composed of the two sub clusters
  cluster operator+(cluster& rhs)
  {
    if (end+1 != rhs.start)
      throw std::runtime_error("Error: attempt to join non-adjacent clusters");

    double new_value = (value * size() + rhs.value * rhs.size())
      / (rhs.end - start);

    return { start, rhs.end, new_value, self, rhs.self };
  }

  // Overload operator< to enable sorting of clusters by starting vertex
  bool operator< (const cluster& rhs) const
  {
    return start < rhs.start;
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

/* Function to recursively bisect the graph along the min cuts to form as many
 * clusters as possible of minimum size s
 *
 * INPUT:
 *       g : graph
 *       s : minimum cluster size
 *       c : list of clusters
 *   start : the starting node available for clustering
 *     end : the end node available for clustering
 *
 * OUTPUT:
 *   Populated vector of clusters sorted by starting node
 */
void form_clusters(graph& g, size_t s, std::vector<cluster>& c, size_t start,
    size_t end)
{
  // Check if there are enough nodes in the given sub-graph to form multiple
  // clusters, if not we're finished. Create a cluster and return
  if ((end - start + 1) < 2 * s)
    c.push_back({ start, end, static_cast<int>(c.size()), g });
  else // There are enough nodes to generate sub-clusters
  {
    // Find the minimum edge cut value in the eligible region
    //   The s first and s last points are excluded from the range to avoid
    //   generating sub-clusters which would not meet minimum size requirements
    auto cut = std::min_element((g.edge.begin() + start + s - 2),
        (g.edge.begin() + end - s + 1));

    // Recursively call form_clusters for the sub-graphs on either side of the
    // new cut
    form_clusters(g, s, c, start, std::distance(g.edge.begin(), cut));
    form_clusters(g, s, c, std::distance(g.edge.begin(), cut + 1), end);
  }
}

/* Function to merge clusters until all clusters have been sucessfully merged
 * into a single tree
 *
 * Paper says this should be done recursively, but given the way things are
 * already organized its probably too much trouble to change so we'll just do it
 * iteratively - doesn't matter anyways
 *
 * INPUT:
 *  cluster_list : a vector of clusters
 *
 * OUTPUT:
 *          root : the index of the root cluster (clusters have members to store
 *                   pointers to leaves)
 */
void merge_clusters(std::vector<cluster>& cluster_list)
{
  // Repeat process until there is a single root cluster remaining
  while (std::count_if(cluster_list.begin(), cluster_list.end(),
        [](cluster c){ return c.mergeable(); }) != 1)
  {
    // Select two adjacent clusters with the minimum representative slope
    // difference
    int lhs_cluster = -1;
    int rhs_cluster = -1;
    double min_dissimilarity = 1e6;

    // For each cluster
    for (size_t i = 0; i < cluster_list.size(); ++i)
    {
      // If i isn't mergeable then there's no sense in trying to merge it
      if (!cluster_list[i].mergeable())
        continue;

      // For each other cluster (if i is one of the original clusters, i.e. has
      //   no children, then we only need to search subsequent clusters since we
      //   know all the original clusters were ordered, and any new clusters are
      //   appended to the list at creation
      for (size_t j = i + 1; j < cluster_list.size()
          && cluster_list[i].left == -1 && cluster_list[i].right == -1; ++j)
      {
        // If both clusters are mergeable and adjacent, and the dissimilarity is
        //   less than the currently identified minimum, nominate clusters for
        //   merging
        if (cluster_list[i].mergeable() && cluster_list[j].mergeable()
            && cluster_list[i].end + 1 == cluster_list[j].start
            && std::abs(cluster_list[i].value - cluster_list[j].value)
              < min_dissimilarity)
        {
          lhs_cluster = i;
          rhs_cluster = j;

          min_dissimilarity =
            std::abs(cluster_list[i].value - cluster_list[j].value);
        }
      }

      // For each other cluster (if i is a newly created cluster, i.e. is a
      //   result of the merger of two other clusters and therefore has
      //   children, then we must search all clusters since i is not in any
      //   order
      for (size_t j = 0; j < cluster_list.size()
          && (cluster_list[i].left != -1 || cluster_list[i].right != -1); ++j)
      {
        // If both clusters are mergeable and adjacent, and the dissimilarity is
        //   less than the currently identified minimum, nominate clusters for
        //   merging
        if (cluster_list[i].mergeable() && cluster_list[j].mergeable()
            && cluster_list[i].end + 1 == cluster_list[j].start
            && std::abs(cluster_list[i].value - cluster_list[j].value)
              < min_dissimilarity)
        {
          lhs_cluster = i;
          rhs_cluster = j;

          min_dissimilarity =
            std::abs(cluster_list[i].value - cluster_list[j].value);
        }
      }
    }

    // Double check that two valid clusters were selected
    if (lhs_cluster == -1 || rhs_cluster == -1
        || lhs_cluster > static_cast<int>(cluster_list.size())
        || rhs_cluster > static_cast<int>(cluster_list.size())
        || cluster_list[lhs_cluster].end + 1 != cluster_list[rhs_cluster].start)
    {
      std::cerr << "ERROR: failed to select clusters for merging" << std::endl;

      //std::cout << "LHS: " << lhs_cluster
      //  << " RHS: " << rhs_cluster
      //  << " size: " << cluster_list.size()
      //  << " dis: " << min_dissimilarity
      //  << " l_end: " << cluster_list[lhs_cluster].end
      //  << " r_start: " << cluster_list[rhs_cluster].start
      //  << std::endl;

      exit(1);
    }

    // Create a new cluster which contains the lhs and rhs clusters as children
    auto new_cluster = cluster_list[lhs_cluster] + cluster_list[rhs_cluster];

    // Mark the lhs and rhs clusters as having a parent
    cluster_list[lhs_cluster].parent = cluster_list.size();
    cluster_list[rhs_cluster].parent = cluster_list.size();

    // Mark the new cluster's index
    new_cluster.self = cluster_list.size();

    // Append to cluster list
    cluster_list.push_back(new_cluster);
  }

  // Debug output
  //std::cout << "Resulting clusters:" << std::endl;
  //for (auto const& cluster : cluster_list)
  //  std::cout << "ID: " << cluster.self
  //    << " LHS: " << cluster.left
  //    << " RHS: " << cluster.right
  //    << " PAR: " << cluster.parent
  //    << " MER: " << (int)cluster.mergeable()
  //    << " start: " << cluster.start
  //    << " end: " << cluster.end
  //    << std::endl;
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

  std::vector<cluster> cluster_list;

  // Bisect graph recursively into clusters
  start = std::chrono::high_resolution_clock::now();
  form_clusters(g, s, cluster_list, 0, g.edge.size()+1);
  stop = std::chrono::high_resolution_clock::now();

  std::cout << "Formed " << cluster_list.size() << " clusters in "
    << std::chrono::duration_cast<
      std::chrono::duration<double>>(stop - start).count()
    << " seconds" << std::endl;

  // Recursively merge clusters
  start = std::chrono::high_resolution_clock::now();
  merge_clusters(cluster_list);
  stop = std::chrono::high_resolution_clock::now();

  std::cout << "Merged clusters based on similarity, resulting in a tree of "
    << cluster_list.size() << " total clusters in "
    << std::chrono::duration_cast<
      std::chrono::duration<double>>(stop - start).count()
    << " seconds" << std::endl;

  // Classify using RIPPER algorithm
  //

  // Dump values? (add tag so states can be graphed easily)
  // When to determine and apply state transition logic
}
