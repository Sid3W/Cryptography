import sys

def fac(n):
    result = []
    while n != 1:
        for i in range(2, n + 1):
            if n % i == 0:
                result.append(i)
                n //= i
                break
    return result


if len(sys.argv) == 1:
    n = int(input()) # if no input parameter, ask for an input number
else:
    n = int(sys.argv[1]) # input a number after the file name

print(fac(n))