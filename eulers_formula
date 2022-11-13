#计算欧拉函数

import sys
if len(sys.argv) == 1:
    a = int(input())
else:
    a = int(sys.argv[1])

def divide(a):
    res = a
    for i in range(2, int(a**0.5)+1):
        if a % i == 0: res = (res//i) * (i-1)
        while a % i == 0:
            a //= i
    if a > 1: res = (res//a) * (a-1)
    return res
 
print(divide(a))
