import numpy as np
from IPython import embed
import matplotlib.pyplot as plt
np.random.seed(1)
def calcuate_L2(v1,v2):
    distance = np.sqrt(np.sum(np.power(v1-v2,2)))
    return distance

def Kmeans(dataSet,k):
    data_cluster =np.zeros([dataSet.shape[0],2])
    center =np.zeros([k,2])#默认2个维度
    for i in range(k):
        random_index =  int(np.random.uniform(0,data_cluster.shape[0]))
        print("radom_index=",random_index)
        center[i] = dataSet[random_index]
    print(center)
    Flag =1
    while Flag: #计算每个样本所属于的簇
        Flag=0
        for i in range(dataSet.shape[0]):
            min_distance = 100000
            min_index = 0
            for j in range(k):
                distance = calcuate_L2(dataSet[i],center[j])
                # print("distance=",distance)
                if distance <min_distance:
                    min_distance=distance
                    min_index = j #这个样本所属于的簇
            if data_cluster[i][0]!=min_index:
                Flag=1
                data_cluster[i][0] = min_index
                data_cluster[i][1] = min_distance**2
        
        #对簇的坐标进行更新: 将每个样本更新完簇之后需要对每一个簇的坐标进行更新：取属于这个簇的所有样本的均值
        for i in range(k):
            index = np.where(data_cluster[:,0]==i)[0]
            center[i] = np.mean(dataSet[index],axis=0) #axis=0 对每一列取均值
    return data_cluster,center
def show_plot(dataSet,data_cluster,center):
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(len(dataSet)):
        color =mark[int(data_cluster[i][0])]
        plt.plot(dataSet[i][0],dataSet[i][1],color)
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb'] #默认最多10个类别
    for i in range(len(center)):
        plt.plot(center[i][0],center[i][1],mark[i],markersize = 12)
    plt.show()        

# np.random.seed(1)
dataSet =np.random.random([10,2])
print(dataSet)
k=2
data_cluster,center = Kmeans(dataSet,k)
# print(data_cluster)
print("center=",center)
show_plot(dataSet,data_cluster,center)