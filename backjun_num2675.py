'''
문자열 S를 입력받은 후에, 
각 문자를 R번 반복해 새 문자열 P를 만든 후 출력하는 프로그램을 작성하시오. 
즉, 첫 번째 문자를 R번 반복하고, 
두 번째 문자를 R번 반복하는 식으로 P를 만들면 된다.
S에는 QR Code "alphanumeric" 문자만 들어있다.

QR Code "alphanumeric" 문자는 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ$%*+-./: 이다.
'''
n = int(input())  # 문자열의 개수

for _ in range(n):
    num, text = input().split()  # num과 text를 입력받음
    num = int(num)  # num을 정수로 변환

    result = []  # 결과를 저장할 리스트 초기화
    for i in text:  # text의 각 문자에 대해 반복
        result.append(i * num)  # 문자를 num번만큼 반복하여 추가
    
    print("".join(result))  # 리스트의 내용을 공백 없이 이어서 출력



