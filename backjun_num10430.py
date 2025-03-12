'''
(A+B)%C는 ((A%C) + (B%C))%C 와 같을까?

(A×B)%C는 ((A%C) × (B%C))%C 와 같을까?

세 수 A, B, C가 주어졌을 때, 위의 네 가지 값을 구하는 프로그램을 작성하시오.
'''

a,b,c=input().split()
int_a=int(a)
int_b=int(b)
int_c=int(c)

d=(int_a+int_b)%int_c
f=(((int_a%int_c)+(int_b%int_c))%int_c)
g=(int_a*int_b)%int_c
h=((int_a%int_c)*(int_b%int_c))%int_c

print(d)
print(f)
print(g)
print(h)