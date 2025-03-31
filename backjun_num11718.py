#입력 받은 대로 출력하는 프로그램을 작성하시오

import sys

data = sys.stdin.read()  # EOF까지 전체 입력 받음
print(data)  # 그대로 출력

'''
사용자가 한 줄씩 입력하는 경우 → input()

파일에서 여러 줄을 읽어야 하는 경우 → sys.stdin.readlines()

여러 줄을 입력받되, EOF까지 한 번에 받고 싶을 때 → sys.stdin.read() (이것도 가능)
'''