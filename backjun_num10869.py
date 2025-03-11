#두 자연수 A와 B가 주어진다. 이때, A+B, A-B, A*B, A/B(몫), A%B(나머지)를 출력하는 프로그램을 작성하시오. 

a,b=input().split()

ag_a=int(a)
ag_b=int(b)

print("{}\n{}\n{}\n{}\n{}".format(ag_a+ag_b,ag_a-ag_b,ag_a*ag_b,ag_a//ag_b,ag_a%ag_b))