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


'''#5-3 1번 문제
numbers=[1,2,3,4,5,6]
print("::".join(str(i) for i in numbers))'''
'''
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
'''

'''try:
    number_input_a=int(input("정수입력?"))
    print("원의 반지름:",number_input_a)
    print("원의 둘레:",2*3.14*number_input_a)
    print("원의 넓이:",3.14*number_input_a*number_input_a)
except:
    print("정수를 입력하지 않았습니다.")
else:
    print("예외가 발생하지 않았습니다.")
finally:
    print("일단 프로그램이 어떻게든 끝났습니다.")'''

''''#6-1 1) 예제
numbers=[52,273,32,103,90,10,275]

print("#(1) 요소 내부에 있는 값 찾기")
print("- {}는 {} 위치에 있습니다.".format(52,numbers.index(52)))
print() 

print("#(2)요소 내부에 없는 값 찾기")
number=1000
try:
    print("- {}는 {} 위치에 있습니다.".format(52,numbers.index(number)))
except:
    print("-리스트 내부에 없는 값입니다.")
print()

print("-----정상적으로 종료되었습니다.-----")'''

'''# 7-1 1)
import os

def read_folder(path):
    output=os.listdir(".")
    for item in output:
        if os.path.isdir(path):
            read_folder(path+"/"+item)
        else:
            print("파일:",item)

read_folder(".")'''
'''
#8-1 예제
class Student:
    def __init__(self,name,score):
        self.name=name
        self.score=score

class StudentList:
    def __init__(self):
        self.students = []

    def append(self, student):
        self.students.append(student)

    def get_average(self):
        return sum(s.score for s in self.students) / len(self.students)

    def get_first_by_score(self):
        return max(self.students, key=lambda s: s.score)

    def get_last_by_score(self):
        return min(self.students, key=lambda s: s.score)


students=StudentList()
students.append(Student("구름",100))
students.append(Student("별",49))
students.append(Student("초코",81))
students.append(Student("아지",90))

print(f"학급의 평균 점수는 {students.get_average()}입니다.")
print(f"가장 성적이 높은 학생은 {students.get_first_by_score().name}입니다.")
print(f"가장 성적이 낮은 학생은 {students.get_last_by_score().name}입니다.")
'''

'''#8-2예제
class Stack:
    def __init__(self):
        self.list=[]

    def push(self,item):
        self.list.append(item)

    def pop(self):
        return self.list.pop()

stack=Stack() 
stack.push(10)
stack.push(20)
stack.push(30)

print(stack.pop())
print(stack.pop())
print(stack.pop())'''

#예제 8-3
class Queue:
    def __init__(self):
        self.list=[]    

    def enqueue(self,item):
        self.list.append(item)

    def dequeue(self):
        return self.list.pop()

queue=Queue()
queue.enqueue(10)
queue.enqueue(20)
queue.enqueue(30)

print(queue.dequeue())
print(queue.dequeue())
print(queue.dequeue())