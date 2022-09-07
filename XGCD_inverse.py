#计算乘法逆元

def XGCD(a, b):
    if (b == 0):
        return a, 1, 0
    gcd, x, y = XGCD(b, a%b)
    return gcd, y, x-a//b*y

def inverse(a, n):
    gcd, x, y = XGCD(a, n)
    return x%n if gcd == 1 else None


import sys
if len(sys.argv) == 1:
    a = int(input()) # if no input parameter, ask for an input number
    b = int(input())
else:
    a = int(sys.argv[1])
    b = int(sys.argv[2])

print(inverse(a,b))