import random
import math
import json
import sys
import time

def load(file_name):
    # data(list of list): [[index, dimensions], [.., ..], ...]
    data = []              
    fh = open(file_name)
    for line in fh:
        line = line.strip().split(',')
        temp = [int(line[0])]
        for feature in line[1:]:
            temp.append(float(feature))
        data.append(temp)  
    return data

def initialize_centroids(data, dimension, k):
    centroids = [[0 for _ in range(dimension)] for _ in range(k)]
    max_feature_vals = [0 for _ in range(dimension)]
    min_feature_vals = [float('inf') for _ in range(dimension)]
    for point in data:
        for i in range(dimension):
            max_feature_vals[i] = max(max_feature_vals[i], point[i + 1])
            min_feature_vals[i] = min(min_feature_vals[i], point[i + 1])
    for i in range(dimension):
        min_feature_val = min_feature_vals[i]
        max_feature_val = max_feature_vals[i]
        diff = max_feature_val - min_feature_val
        for j in range(k):
            centroids[j][i] = min_feature_val + diff * random.uniform(1e-5, 1)
    return centroids

def get_euclidean_distance(p1, p2, p1_with_index, p2_with_index):
    i1 = 0
    i2 = 0
    if p1_with_index:
        i1 = 1
    if p2_with_index:
        i2 = 1
    sd_sum = 0
    for d in range(len(p1) - i1):
        sd_sum += (p1[d + i1] - p2[d + i2]) ** 2
    return math.sqrt(sd_sum)

def get_sample(data):
    length = len(data)
    sample_size = int(length * 0.01)
    random_nums = set()
    sample_data = []

    for i in range(sample_size):
        random_index = random.randint(0, length - 1)
        while random_index in random_nums:
            random_index = random.randint(0, length - 1)
        random_nums.add(random_index)
        sample_data.append(data[random_index])
    return sample_data


def kmeans(data, dimension, k):
    
    centroids = initialize_centroids(data, dimension, k)
    cluster_affiliation = [[tuple(features), None] for features in data]
    flag = 1

    while flag:

        flag=0

        for i, point in enumerate(data):
            min_distance = float('inf')
            min_distance_index = None

            #find closest centroids for each data points
            for cluster_index, centroid in enumerate(centroids):
                if centroid[0] == None:
                    continue
                distance = get_euclidean_distance(centroid, point, False, True)
                if distance < min_distance:
                    min_distance = distance
                    min_distance_index = cluster_index

            #record or update cluster for each data points
            if cluster_affiliation[i][1] != min_distance_index:
                flag = 1
                cluster_affiliation[i][1] = min_distance_index
        #recompute centroids
        centroids = [[0 for _ in range(dimension)] for _ in range(k)]
        clutser_point_count = [0 for _ in range(k)]
        for i, point in enumerate(data):
            cluster_index = cluster_affiliation[i][1]
            clutser_point_count[cluster_index] += 1
            for d in range(dimension):
                centroids[cluster_index][d] += point[d + 1]
        for cluster_index, centroid in enumerate(centroids):
            point_count = clutser_point_count[cluster_index]
            for d in range(dimension):
                if point_count == 0:
                    centroids[cluster_index][d] = None
                else:
                    centroids[cluster_index][d] /= point_count

    return (centroids, cluster_affiliation)

def gather_clusters_info(centroids, cluster_affiliation):
    clusters_temp = [[tuple(centroid), []] for centroid in centroids]
    for a in cluster_affiliation:
        features, cluster_index = a[0], a[1]
        clusters_temp[cluster_index][1].append(features)
    clusters = []
    for cluster in clusters_temp:
        if cluster[0][0] != None:
            clusters.append(cluster)
    return clusters

def delete_redundant_cluster(clusters, final_count):
    index_count = []
    for index, cluster in enumerate(clusters):
        index_count.append((len(cluster[1]), index))
    index_count.sort()
    for i in range(len(clusters) - final_count):
        clusters.pop(0)
    return clusters

def initialize_stat(clusters, dimension, cluster_min_size):
    stats = []
    set_point_index = []
    remaining_points = []
    for centroid, points in clusters:
        if len(points) >= cluster_min_size:
            stat = [0, [0 for _ in range(dimension)], [0 for _ in range(dimension)]]
            point_index = set()
            for point in points:
                point_index.add(point[0])
                stat[0] += 1
                for d in range(dimension):
                    stat[1][d] += point[d + 1]
                    stat[2][d] += point[d + 1] ** 2
            stats.append(stat)
            set_point_index.append(point_index)
        else:
            remaining_points.extend(points)
    return (stats, set_point_index, remaining_points)

def get_centroids_sd(stat, dimension):
    centroids = []
    cluster_sd = []
    for N, SUM, SUMSQ in stat:
        centroid = []
        sd = []
        for d in range(dimension):
            centroid.append(SUM[d] / N)
            sd.append(math.sqrt(SUMSQ[d] / N - (SUM[d] / N) ** 2))
        centroids.append(centroid)
        cluster_sd.append(sd)
    return (centroids, cluster_sd)

def get_mahalanobis_distance(p1, p2, p1_with_index, p2_with_index, sd, dimension):
    i1 = 0
    i2 = 0
    if p1_with_index:
        i1 = 1
    if p2_with_index:
        i2 = 1
    sum_sq = 0
    for d in range(dimension):
        sum_sq += ((p1[d + i1] - p2[d + i2]) / sd[d]) ** 2
    return math.sqrt(sum_sq)

def update_stat(data, stat, set_point_index, dimension, threshold, first_load):
    
    centroids, cluster_sd = get_centroids_sd(stat, dimension)

    remaining_points = []

    for point in data:
        point = tuple(point)

        if first_load:
            point_exist = False
            for point_index in set_point_index:
                if point[0] in point_index:
                    point_exist = True
                    break
            if point_exist:
                continue

        min_mahalanobis_distance = float('inf')
        for index, centroid in enumerate(centroids):
            mahalanobis_distance = get_mahalanobis_distance(point, centroid, True, False, cluster_sd[index], dimension)     
            if mahalanobis_distance < min_mahalanobis_distance:
                min_mahalanobis_distance = mahalanobis_distance
                min_index = index
        if min_mahalanobis_distance < threshold:
            set_point_index[min_index].add(point[0])
            stat[min_index][0] += 1
            for d in range(dimension):
                stat[min_index][1][d] += point[d + 1]
                stat[min_index][2][d] += point[d + 1] ** 2
        else:
            remaining_points.append(point)
    return (stat, set_point_index, remaining_points)

def merge_clusters(stat1, point_index1, stat2, point_index2, i, j, dimension, combined_cs):
    s1 = []
    s2 = []
    for n in range(3):
        s1.append(stat1[i][n])
        s2.append(stat2[j][n])
    combined_stat = [s1[0] + s2[0]]
    v1 = []
    v2 = []
    for d in range(dimension):
        v1.append(s1[1][d] + s2[1][d])
        v2.append(s1[2][d] + s2[2][d])
    combined_stat.append(v1)
    combined_stat.append(v2)
    
    temp = point_index1[i]
    for point in point_index2[j]:
        temp.add(point)
    if combined_cs:
        del stat2[max(i, j)]
        del stat2[min(i, j)]
        del point_index2[max(i, j)]
        del point_index2[min(i, j)]
    else:
        del stat1[i]
        del stat2[j]
        del point_index1[i]
        del point_index2[j]
    stat2.append(combined_stat)
    point_index2.append(temp)

    return (stat1, point_index1, stat2, point_index2)

def check_merge_clusters(stat1, point_index1, stat2, point_index2, dimension, combined_cs, threshold):
    while True:
        merged = False
        
        if not combined_cs:
            if stat1:
                centroids1, cluster_sd1 = get_centroids_sd(stat1, dimension)
                centroids2, cluster_sd2 = get_centroids_sd(stat2, dimension)
                for i in range(len(stat1)):
                    if merged:
                        break
                    for j in range(len(stat2)):
                        mahalanobis_distance = get_mahalanobis_distance(centroids1[i], centroids2[j], False, False, cluster_sd2[j], dimension)
                        if mahalanobis_distance < threshold:
                            stat1, point_index1, stat2, point_index2 = merge_clusters(stat1, point_index1, stat2, point_index2, i, j, dimension, False)
                            merged = True
                            break
            if not merged:
                return (stat1, point_index1, stat2, point_index2)
            
        else:
            cs = stat2
            cs_length = len(cs)
            if cs_length > 1:
                centroids, cluster_sd = get_centroids_sd(cs, dimension)
                for i in range(cs_length - 1):
                    if merged:
                        break
                    for j in range(i + 1, cs_length):
                        mahalanobis_distance = get_mahalanobis_distance(centroids[i], centroids[j], False, False, cluster_sd[j], dimension)
                        if mahalanobis_distance < threshold:
                            stat1, point_index1, stat2, point_index2 = merge_clusters(cs, point_index2, cs, point_index2, i, j, dimension, True)
                            merged = True
                            break
            if not merged:
                return (stat2, point_index2)

def main():
	start = time.time()

	inputpath = sys.argv[1]
	K = int(sys.argv[2])
	output1 = sys.argv[3]
	output2 = sys.argv[4]

	data_num = 0
	data = load(inputpath + '/data' + str(data_num) + '.txt')
	dimension = len(data[0]) - 1
	threshold = 4 * math.sqrt(dimension)
	sample_data = get_sample(data)
	centroids, cluster_affiliation = kmeans(sample_data, dimension, K * 8)
	# if not enough clusters, run kmeans again
	while True:
	    centroid_count = 0
	    for centroid in centroids:
	        if centroid[0] != None:
	            centroid_count += 1
	    if centroid_count < K:
	        centroids, cluster_affiliation = kmeans(sample_data, dimension, K * 8)
	    else:
	        break
	# clusters: [(centroid1, [(point1 features), (point2 features), ...]), (centroid2, ...)]
	clusters = gather_clusters_info(centroids, cluster_affiliation)
	# delete redundant clutser
	clusters = delete_redundant_cluster(clusters, K)

	# ds: [(N, [SUM], [SUMSQ], [point_indexes]), ...]
	ds, ds_point_index, temp = initialize_stat(clusters, dimension, 1)

	first_load = True
	cs = []
	rs = []
	cs_point_index = []
	inter_results = []

	while True:

		try:
		    data = load(inputpath + '/data' + str(data_num) + '.txt')
		except:
		    cs, cs_point_index, ds, ds_point_index = check_merge_clusters(cs, cs_point_index, ds, ds_point_index, dimension, False, threshold)
		    break

		# assign points to ds
		ds, ds_point_index, remaining_points = update_stat(data, ds, ds_point_index, dimension, threshold, first_load)
		if first_load:
		    first_load = False  
		if remaining_points:    
		    if cs:
		        # merge cs if needed
		        cs, cs_point_index = check_merge_clusters(cs, cs_point_index, cs, cs_point_index, dimension, True, threshold)
		        # assign points to cs
		        cs, cs_point_index, remaining_points = update_stat(remaining_points, cs, cs_point_index, dimension, threshold, False)
		    centroids, cluster_affiliation = kmeans(remaining_points + rs, dimension, 3 * K)
		    clusters = gather_clusters_info(centroids, cluster_affiliation)
		    cs_temp, cs_point_index_temp, rs = initialize_stat(clusters, dimension, 2)
		    cs.extend(cs_temp)
		    cs_point_index.extend(cs_point_index_temp)
		data_num += 1
		ds_point_count = 0
		cs_point_count = 0
		inter_results.append((data_num, len(ds), sum([len(points) for points in ds_point_index]), len(cs), sum([len(points) for points in cs_point_index]), len(rs)))


	results = {}
	for index, points in enumerate(ds_point_index):
	    for point in points:
	        results[str(point)] = index
	for points in cs_point_index:
	    for point in points:
	        results[str(point)] = -1
	for point in rs:
	    results[str(point[0])] = -1
	fh = open(output1, 'w')
	json.dump(results, fh)
	fh.close()
	
	fh = open(output2, 'w')
	fh.write('round_id,nof_cluster_discard,nof_point_discard,nof_cluster_compression,nof_point_compression,nof_point_retained')
	for line in inter_results:
	    fh.write('\n')
	    fh.write(str(line).strip('()'))
	fh.close()	

	print('Duration: %s' % (time.time() - start))

if __name__ == "__main__":
	while True:
		try:
			main()
			break
		except:
			print('********* except + 1 ***********')
			continue
