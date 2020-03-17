# Variation of information
# Expects list of sets
def voi(X, Y):
    n = float(sum([len(x) for x in X]))
    assert n == float(sum([len(y) for y in Y]))
    ret = 0.0
    
    for x in X:
        for y in Y:
            p = len(x)/n
            q = len(y)/n
            r = len(set(x)&set(y))/n
            if r > 0:
                ret += r*(log(r/p, 2)+log(r/q, 2))
    return -1*ret
