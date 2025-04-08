'''그룹 단어란 단어에 존재하는 모든 문자에 대해서, 
각 문자가 연속해서 나타나는 경우만을 말한다. 
예를 들면, ccazzzzbb는 c, a, z, b가 모두 연속해서 나타나고, 
kin도 k, i, n이 연속해서 나타나기 때문에 그룹 단어이지만, 
aabbbccb는 b가 떨어져서 나타나기 때문에 그룹 단어가 아니다.

단어 N개를 입력으로 받아 그룹 단어의 개수를 출력하는 프로그램을 작성하시오.'''
#1.해당 알파벳이 여러개 있나?
#2.몇개 있는가?
#3.알파벳 개수대로 붙어 있는가?
'''
def count_alphabet(a,target):
    #a:리스트 target:찾을 알파벳
    count=0
    for i in a:
        if target==i:
            count+=1
    return count

def check_alphabet(a,target,n):
    #a:리스트 target:찾을 알파벳 n:연속 횟수
    count=1
    result=0
    index=a.find(target)
    last=len(a)
    for i in range(index,last-1):
        if a[i]==a[i+1]:
            count+=1
        else:
            break
    if count==n:
        result=0
    else:
        result=1

    return result


n=int(input())
word=[]
count=result_count=0

for i in range(n):
    word.append(input())

for i in word:
    result=0
    n=len(i)
    for j in range(n):
        if i[j] in i[j+1:]:
            count=count_alphabet(i,i[j])
            result=check_alphabet(i,i[j],count)
            if result==1:
                break
            else:
                pass
        else:
            pass
    if result==0:
        result_count+=1

print(result_count)        
'''

n=int(input())
word=[]
count=0
for i in range(n):
    word.append(input())

for i in word:
    check=[]
    result=0
    for j in range(len(i)):
        if i[j]!=i[j+1]:
            if i[j] in check:
                result=1
                break
        check.append(i[j])
    del check
    if result==0:
        count+=1
print(count)