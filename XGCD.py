#使用扩展欧几里得算法 计算乘法逆元

XandY=[]
def XGCD(a, b):
    if (b == 0):
        XandY.append((1,0))
        return a, 1, 0

    gcd, x, y = XGCD(b, a%b)
    XandY.append((y, x-a//b*y))
    return gcd, y, x-a//b*y

def inverse(a, n):
    gcd, x, y = XGCD(a, n)
    
    for i in range(len(XandY)-1,-1,-1):print(XandY[i])
    
    return x%n if gcd == 1 else None


import sys
if len(sys.argv) == 1:
    a = int(input()) # if no input parameter, ask for an input number
    b = int(input())
else:
    a = int(sys.argv[1])
    b = int(sys.argv[2])
print("inverse: ", inverse(a,b))