def compute_R2(x, y):
    return(((x - x.mean()) * (y - y.mean())).mean() / (x.std()*y.std()))**2
