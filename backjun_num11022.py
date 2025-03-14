# 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

import sys

n=int(sys.stdin.readline().rstrip())
result=[]
list_a=[]

for i in range(n):
    a=list(map(int,sys.stdin.readline().strip().split()))
    list_a.append(a)
    result.append(sum(a))

for i in range(n):
    print("Case #{}: {} + {} = {}".format(i+1,list_a[i][0],list_a[i][1],result[i]))