import math
list = []
for i in range(999):
    if math.gcd(i,28*40)==1:
        list.append(i)
print(list)
