#알파벳 대소문자로 된 단어가 주어지면, 
# 이 단어에서 가장 많이 사용된 알파벳이 무엇인지 알아내는 프로그램을 작성하시오. 
# 단, 대문자와 소문자를 구분하지 않는다.

a=list(input().upper())
result=[]
check2=0

for i in range(len(a)):
    check="false"
    for j in range(len(result)):
        if a[i] not in result[j][0]:
            pass
        else:
            result[j][1]+=1
            check="true"
            break
        
    if check=="false":
        result.append([a[i],1])

max_item=max(result,key=lambda x:x[1])
result.remove(max_item)

for i in result:
    if i[1] == max_item[1]:
        print("?")
        check2=1
        break
if check2 ==0:
    print(max_item[0])
