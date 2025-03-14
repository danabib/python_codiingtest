#첫째 줄에 테스트 케이스의 개수 T가 주어진다.
#각 테스트 케이스는 한 줄로 이루어져 있으며, 각 줄에 A와 B가 주어진다. (0 < A, B < 10)

n=int(input())
list_a=[]
result_b=[]

for i in range(n):
    a=list(map(int,input().split()))
    list_a.append(a)
    result_b.append(sum(a))

print(*result_b,sep="\n")
