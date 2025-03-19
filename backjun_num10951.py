#두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

import sys
list_b=[]
while True:
    try:
        list_a = list(map(int, input().split()))
        list_b.append(sum(list_a))
    except:
        break
print(*list_b,sep='\n')