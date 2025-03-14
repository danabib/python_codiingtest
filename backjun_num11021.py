#두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

import sys

n=int(sys.stdin.readline().rstrip())
result=[]

for i in range(n):
    a,b=map(int,sys.stdin.readline().strip().split())
    result.append(a+b)

for i in range(n):
    print("Case #{}: {}".format(i+1,result[i]))