'''10진법 수 N이 주어진다. 
이 수를 B진법으로 바꿔 출력하는 프로그램을 작성하시오.

10진법을 넘어가는 진법은 숫자로 표시할 수 없는 자리가 있다. 
이런 경우에는 다음과 같이 알파벳 대문자를 사용한다.

A: 10, B: 11, ..., F: 15, ..., Y: 34, Z: 35'''

a, b = map(int, input().split())  # 10진법 수 a와 변환할 진법 b 입력받기
result = []

while a > 0:
    remainder = a % b  # 나머지 계산
    if remainder >= 10:  # 나머지가 10 이상이면 알파벳으로 변환
        result.append(chr(remainder - 10 + ord('A')))  # 'A'는 10에 해당
    else:
        result.append(str(remainder))  # 나머지가 10 미만이면 그대로 숫자로 추가
    a = a // b  # 몫 계산

# 결과를 뒤집어서 출력
print(''.join(reversed(result)))



