import numpy as np
def createDataSet():
    group = np.array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])
    labels = ['A','A','B','B']
    return group,labels
def knn(inputs,k):
    x, labels = createDataSet()
    distance = np.sqrt(np.sum(np.power((inputs-x),2),axis=1))
    index = np.argsort(distance)
    class_dicts ={}
    for i in range(k):
        # if labels[index[i]] not in class_dicts:
        #     class_dicts[labels[index[i]]] =1
        # else:
        #     class_dicts[labels[index[i]]] +=1
        label_item =labels[index[i]]
        class_dicts[label_item] = class_dicts.get(label_item,0) +1
    temp_count =0 
    for key,value in class_dicts.items():
        if value >temp_count:
            temp_count =value
            temp_cls = key
    return temp_cls
inputs = np.array([1.1,0.3])
K = 3
output = knn(inputs,K)
print("测试数据为:",inputs,"分类结果为：",output)
