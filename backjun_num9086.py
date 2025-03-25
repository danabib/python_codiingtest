#문자열을 입력으로 주면 문자열의 첫 글자와 마지막 글자를 출력하는 프로그램을 작성하시오.
n=int(input())
case=[]
for i in range(n):
    A=input()
    case.append(A)

for i in range(n):
    lengh=len(case[i])
    print("{}{}".format(case[i][0],case[i][lengh-1]))