import numpy as np 
w_shape=[1,4]
drop_prob = 0.8
a = np.random.random(w_shape) 
b = a<drop_prob
c = a*b
print("dropout前的:\n",a)
print("sum=",sum(a[0]))
print("dropout后的:\n",c)
print("sum=",sum(c[0]))
c/=drop_prob
print("sum=",sum(c[0]))