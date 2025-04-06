'''예제를 보고 규칙을 유추한 뒤에 별을 찍어 보세요.
    *
   ***
  *****
 *******
*********
 *******
  *****
   ***
    *
'''

n=int(input())

for i in range(1,n+1):
    for j in range(n,0,-1):
        if j<=i:
            print("*",end='')
        else:
            print(" ",end='')
    for j in range(1,n):
        if j<i:
            print("*",end='')
    print("")

for i in range(n):
    for j in range(n):
        if i<j:
            print("*",end='')
        else:
            print(" ",end='')

    for j in range(n-2,0,-1):
        if i<j:
            print("*",end='')

    print("")

