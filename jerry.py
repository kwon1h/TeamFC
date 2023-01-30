from sweeppy import Sweep
import numpy as np
import matplotlib.pyplot as plt
import time

class myDBSCAN():
    
    cluster_label = 0
    
    def __init__(self, data_x, data_y, eps, min_samples):
        self.data_x = data_x
        self.data_y = data_y
        self.data_XY = np.array([data_x, data_y])
        self.eps = eps
        self.min_samples = min_samples
        self.labels = [0]*len(data_x)

    def adaptive_eps(lidar_data):
        n = lidar_data.shape[0]
        distances = np.zeros(n)
        for i in range(n):
            point = lidar_data[i]
            diff = lidar_data - point
            dist = np.linalg.norm(diff, axis=1)
            dist[i] = np.inf
            nearest_neighbor_dist = np.min(dist)
            distances[i] = nearest_neighbor_dist
        eps = np.max(distances)
        return eps    

    def k_distance(data, k):

        n = data.shape[0]
        k_dist = np.zeros(n)
        for i in range(n):
            point = data[i]
            distances = np.linalg.norm(data - point, axis=1)
            distances = np.sort(distances)
            k_dist[i] = distances[k] * 0.75
        return k_dist

    # INPUT : x_values, y_values nparray
    def run_dbscan(self):
        
        for point in range(len(self.data_x)):
            
            # pass if already checked
            if not (self.labels[point] == 0):
                continue
            
            # Find all of P's neighboring point
            neighbors = self.get_neighbors(point)
            
            # if len below min_samples, this point is noise
            # it can be border point!! but it will be covered by griowCluster func
            if len(neighbors) < self.min_samples:
                self.labels[point] = -1
            else:
                self.cluster_label += 1
                self.growCluster(point, neighbors)
                
        return self.labels
            
    # OUTPUT - POINT INDEX
    # [[1,3,5,7], [2,4,6,8]...] 
    
    def test_print(self):
        print(self.data_XY)
        print(self.labels)
        print(self.eps)
        print(self.min_samples)
        
    def get_neighbors(self, point):
        neighbors = []
        
        for point_next in range(len(self.data_x)):
            
            # distance threshold below neighbor
            # not-squared for fast calculation
            if abs(self.data_x[point] - self.data_x[point_next]) + abs(self.data_y[point] - self.data_y[point_next]) < self.eps:
                neighbors.append(point_next)
        
        return neighbors
    
    def growCluster(self, point, neighbors):
        self.labels[point] = self.cluster_label
        
        i = 0
        while i < len(neighbors):
            
            # get next point from queue
            point_next = neighbors[i]
            
            # noise or border => border point check
            if self.labels[point_next] == -1:
                self.labels[point_next] = self.cluster_label
            
            elif self.labels[point_next] == 0:
                self.labels[point_next] = self.cluster_label
                
                # Find all neighbors of Pn
                point_next_neighbors = self.get_neighbors(point_next)
                
                if len(point_next_neighbors) >= self.min_samples:
                    neighbors = neighbors + point_next_neighbors
            i += 1

# lidar_cluster = myDBSCAN(x_values, y_values, eps=10, min_samples=3)
# labels = lidar_cluster.run_dbscan()
# lidar_cluster.test_print()

# colors = sns.color_palette('bright', len(labels))

# for i in range(1, max(labels)+1):
#     for j in range(len(labels)):
#         if i == labels[j]:
#             plt.scatter(x_values[j], y_values[j], color=colors[i])
    
# plt.show()

plot_count = 0

with Sweep('/dev/ttyUSB0') as sweep:
    sweep.set_motor_speed(10)            #모터속도 설정
    sweep.set_sample_rate(1000)         #샘플레이트 설정
    speed = sweep.get_motor_speed()
    rate  = sweep.get_sample_rate()
    ready = sweep.get_motor_ready()
    print("Moter speed is " + str(speed))
    print("Sample rate is " + str(rate))
    print("Is device ready? --> " + str(ready))
    sweep.start_scanning()

    for scan in sweep.get_scans():

        # first  index  []  scan_count, 
        # second index  []  sample_count,
        # third  index  []  angle, distance, strength
        # print(scan[0][0][0])

        X = []
        Y = []

        plot_count += 1
        print(plot_count)

        for i in range(len(scan[0])):
                
            # angle : scan[0][i][0]
            # distance : scan[0][i][1]
                      
            # Get X,Y value 
            X.append(np.cos(np.deg2rad(scan[0][i][0]/1000)) * scan[0][i][1])
            Y.append(np.sin(np.deg2rad(scan[0][i][0]/1000)) * scan[0][i][1])
                
            
        data_XY = []
        sort_XY = []

        for i in range(len(X)):
            data_XY.append(X[i])
            data_XY.append(Y[i])
            sort_XY.append(list(data_XY))
            data_XY = []
        #print(sort_XY)
        np_sort_XY = np.array(sort_XY)
        my_eps = myDBSCAN.adaptive_eps(np_sort_XY)
        print(my_eps)

        if(my_eps<15):
            my_eps = 15
        elif(my_eps>30):
            my_eps = 30

        #lidar_cluster = myDBSCAN(X, Y, eps=25, min_samples=2)
        #labels = lidar_cluster.run_dbscan()
        lidar_cluster_labels = myDBSCAN(X, Y, eps=25, min_samples=4).run_dbscan()
        np_labels = np.array(lidar_cluster_labels)

        #print(" start scanning !!")

        # labels = i 인 점들의 인덱스를 전부 모아서 dataX[인덱스]의 평균, dataY[인덱스]의 평균

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'olive', 'fuchsia', 'purple', 'midnightblue', 'darkorange', 'lightpink', 'stategray', 'azure']
        #plt.figure(figsize=(6,8))
        if (plot_count == 2):
            #f, axes = plt.subplot(1, 2)
            plt.clf()
            plt.axis([-300, 300, -250, 250])
            noise_clusters = np.unique(np_labels)
            clusters = np.delete(noise_clusters, np.where(noise_clusters == -1))
            for i in clusters:
                indices = np.where(lidar_cluster_labels == i)
                X_cluster = np.array(X)[indices]; Y_cluster = np.array(Y)[indices]
                X_center = np.median(X_cluster); Y_center = np.median(Y_cluster)
                plt.scatter(X_center, Y_center, c = 'black', marker='d', s=250)
                plt.scatter(X_cluster, Y_cluster, c = colors[i%13], s = 25)

            plt.pause(0.1)
            plot_count = 1
    plt.show()