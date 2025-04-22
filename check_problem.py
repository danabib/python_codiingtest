#혼공파 챕터 5

#1번 문제
'''
def f(x):
    return (2*x+1)
print(f(10))


def f(x):
    return (x**2+2*x+1)
print(f(10))
'''

#2번 문제
'''
def mul(*values):
    sum=1
    for i in values:
        sum*=i
    return sum

print(mul(5,7,9,10))
'''
'''
def flatten(data):
    output=[]
    for item in data:
        if type(item)==list:
            output+=flatten(item)
        else:
            output.extend([item])
            
    return output
example=[[1,2,3],[4,[5,6]],7,[8,9]]
print("원본:",example)
print("변환:",flatten(example))'''

'''#5-2 1번 문제
앉힐수있는최소사람수 = 2
앉힐수있는최대사람수 = 10
전체사람의수 = 100
memo = {}

def 문제(남은사람수, 최소테이블사이즈):
    key = (남은사람수, 최소테이블사이즈)
    if key in memo:
        return memo[key]

    if 남은사람수 == 0:
        return 1
    if 남은사람수 < 0:
        return 0

    count = 0
    for i in range(최소테이블사이즈, 앉힐수있는최대사람수 + 1):
        count += 문제(남은사람수 - i, i)

    memo[key] = count
    return count

print(문제(전체사람의수, 앉힐수있는최소사람수))'''


#5-3 1번 문제
'''numbers=[1,2,3,4,5,6]
print("::".join(str(i) for i in numbers))'''

#5-3 2번 문제
numbers=list(range(1,10+1))

print("#홀수만 추출하기")
print(list(filter(lambda x:x%2==1,numbers)))
print()

print("#3이상, 7미만 추출하기")
print(list(filter(lambda x: 3<=x<7,numbers)))
print()

print("#제곱해서 50미만 추출하기")  
print(list(filter(lambda x: x**2<50,numbers)))

