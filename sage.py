F.<x>=GF(2^3,modulus=x^3+x^2+1)
for i in range(8):
    print(i,x^i)

print("inverse: ",(x^2+x)^(-1))