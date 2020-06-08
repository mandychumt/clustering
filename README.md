# clustering

Implement K-Means and Bradley-Fayyad-Reina (BFR) algorithms from scratch. K-Means is the main-memory clustering algorithm used in BFR. My program iteratively load the data points from a file and process these data points with the BFR algorithm. 

Execution Command:
python3 bfr.py <input_path> <n_cluster> <out_file1> <out_file2>
<input_path>: the folder containing the files of data points <n_cluster>: the number of clusters
<out_file1>: the output file of cluster results (JSON format, key = data point index, value = clsuter index)
<out_file2>: the output file of intermediate results (CSV file with colums: round id, number of clusters in the discard set, total number of the discarded points, number of clusters in the compression set, total number of the compressed points, and number of points in the retained set)

General Processes:
Suppose the number of clusters is K and the number of dimensions is d.
a. Load the data points from one file.
b. Run K-Means on a small random sample of the data points to initialize the K centroids using the Euclidean distance as the similarity measurement.
c. Use the K-Means result from b to generate the DS clusters (i.e., discard points and generate statistics).
d. The initialization of DS has finished, so far, you have K clusters in DS.
e. Run K-Means on the rest of the data points with a large number of clusters (e.g., 8 times of K) to generate CS (clusters with more than one points) and RS (clusters with only one point).
f. Load the data points from next file.
g. For the new points, compare them to the clusters in DS using the Mahalanobis Distance and assign them to the nearest DS cluster if the distance is < 𝛼√𝑑.
h. For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and assign the points to the nearest CS cluster if the distance is < 𝛼√𝑑.
i. For the new points that are not assigned to any clusters in DS or CS, assign them to RS.
j. Merge the data points in RS by running K-Means with a large number of clusters (e.g., 8 times of K) to generate CS (clusters with more than one points) and RS (clusters with only one point).
k. Merge clusters in CS that have a Mahalanobis Distance < 𝛼√𝑑.
l. Repeat the steps f – k until all the files are processed.
m. If this is the last round (after processing the last chunk of data points), merge clusters in CS with the clusters in DS that have a Mahalanobis Distance < 𝛼√𝑑.
(𝛼 is a hyper-parameter, I choose 4 in this case)
