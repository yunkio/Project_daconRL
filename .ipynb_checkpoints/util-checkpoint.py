def score_func(x, a):
    if a == 0:
        return 1.0
    if x < a:
        return 1 - (x / a)
    else:
        return 0.0