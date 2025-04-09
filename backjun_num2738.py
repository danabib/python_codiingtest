# N*M크기의 두 행렬 A와 B가 주어졌을 때, 
# 두 행렬을 더하는 프로그램을 작성하시오.

n,m=map(int,input().split())

first=[]
last=[]
result=[[0]*m for _ in range(n)]

for i in range(n):
    first.append(list(input().split()))

for i in range(n):
    last.append(list(input().split()))

for i in range(n):
    for j in range(m):
        print(f' {int(first[i][j])  + int(last[i][j])}', end="")    
    print()
