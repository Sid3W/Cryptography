# 计算最大公约数

import sys,math
if len(sys.argv) == 1:
    a = int(input()) # if no input parameter, ask for an input number
    b = int(input())
else:
    a = int(sys.argv[1])
    b = int(sys.argv[2])

print(math.gcd(a,b))