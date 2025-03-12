#(세 자리 수) × (세 자리 수)는 다음과 같은 과정을 통하여 이루어진다.
sum=0

a=int(input())
b=list(input())
b.reverse()

for i in range(3):
    int_b=int(b[i])
    c=a*int_b
    print(c)
    sum+=c*(10**i)

print(sum)