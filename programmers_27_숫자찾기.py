def solution(num, k):
    arr=list(num)
    for i,ind in enumerate(num):
        if k==i:
            return ind+1
    return -1