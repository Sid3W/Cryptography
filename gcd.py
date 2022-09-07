def gcd(a,b):
    if a == 0:
        return b
    if b == 0:
        return a
    if a >= b:
        return gcd(a%b, b)
    else:
        return gcd(a, b%a)

import sys
if len(sys.argv) == 1:
    a = int(input()) # if no input parameter, ask for an input number
    b = int(input())
else:
    a = int(sys.argv[1])
    b = int(sys.argv[2])

print(gcd(a,b))