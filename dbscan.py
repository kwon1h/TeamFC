from sweeppy import Sweep
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, degrees, radians, sqrt
import itertools
import sys

plot_count = 0

#Recieve LiDAR data

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def r_u_core(dataXY, else_dataXY, eps, min_samples):

    count = 0

    for i in else_dataXY:
        if distance(dataXY, i) < eps:
            count += 1

        if count == min_samples+1:
            return True
        
    return False

def r_u_neighbor(core, dataXY, diff_list, eps, min_samples):
    
    count = 0
    
    #코어를 이웃으로 가지고 있습니까?
    for i in diff_list:
        if distance(i, core) < eps:
            count += 1
        if count >= 1:
            return True


def clustring(core, eps):
    cluster = [[] for _ in range(N)]
    # 아래 세 줄을 위에 한줄로 만들었다. 그렇다!!! chatPKH의 성능은 대단했다!!
    # clusters = []
    # for i in range(N):
    #     clusters.append([])

    for index,data enumerate(core):
        if distance(data,core) < eps:
            cluster[number].append(index)

        diff_set = set(core)-set(cluster[number])
        diff_list123 = list(cluster[number])

        diff_list = diff_list_calc(core,cluster[number])

        number += 1

        if distance(i, diff_list) < eps:
            cluster[number].append(i)


def diff_list_calc(list1,list2):
    result = list(set(list1)-set(list2))

    return result


with Sweep('/dev/ttyUSB0') as sweep:
    sweep.set_motor_speed(3)            #모터속도 설정
    sweep.set_sample_rate(1000)         #샘플레이트 설정
    speed = sweep.get_motor_speed()
    rate  = sweep.get_sample_rate()
    ready = sweep.get_motor_ready()
    print("Moter speed is " + str(speed))
    print("Sample rate is " + str(rate))
    print("Is device ready? --> " + str(ready))
    sweep.start_scanning()
    dataXY_temp = []
    dataXY = []
    for scan in sweep.get_scans():
    #for scan in itertools.islice(sweep.get_scans(), 1):

        dataXY = []
        dataX = []
        dataY = []

        plot_count += 1
        #print(plot_count)

        for data in scan[0]:
            X= data[1] * cos(radians(data[0])/1000)
            Y= data[1] * sin(radians(data[0])/1000)

            dataX.append(X)
            dataY.append(Y)

            #print (X)
            #print (Y)
            dataXY_temp.append(X)
            dataXY_temp.append(Y)
            dataXY.append(dataXY_temp)
            dataXY_temp = []
        #print(dataXY[0])
        
        if (plot_count == 5):
            plt.clf()
            plt.scatter(dataX,dataY, marker = '.', color = 'b', s = 200)
            plt.axis([-300, 300, -250, 250])
            plt.pause(0.1)
            plot_count = 0


    #받은 데이터들 중 코어 데이터 찾기
    core = [i for i in dataXY if r_u_core(i, dataXY, 10, 3)]

    # 이 아래 4줄의 코드를 위에 한 줄로 끝내버림. 친구가 짜준거리 이해 못함 일단 지리는 코드임.
    # core = []
    # for i in dataXY:
    #     if r_u_core(i, dataXY, 10, 3):
    #         core.append(i)

    # (전체 데이터)-(코어) = 이웃 + 노이즈
    #diff_set1 = set(dataXY)-set(core) # set은 집합, 차집합을 계산해서 코어가 아닌 데이터들을 r_u_neighbor 함수에 넣을 예정
    #diff_list1234 = list(diff_set1) # set을 list 로 변환, list는 배열,, diff_list는코어가 아닌 나머지 점들 모두를 의미함
    diff_list1 = diff_list_calc(dataXY,core)


    neighbor = [i for i in diff_list1 if r_u_core(core, i, diff_list1, 10, 3)]
    # (코어가 아닌 나머지)-(이웃) = 노이즈 // 아니 근데 노이즈를 계산할 필요가 있나? 그냥 클러스터링 할 때 코어+이웃 만 넣으면 되는거 아닌가?
    #diff_set2 = set(diff_list1)-set(neighbor)
    #diff_list2 = list(diff_set2)
    #filtered_data_set = set(dataXY)-set(diff_list2)
    #filtered_data_list = list(filtered_data_set)

    core_plus_neighbor = list(set(core)+set(neighbor))

    cluster = []
    cluster = clustring(core_plus_neighbor)


    plt.show()
