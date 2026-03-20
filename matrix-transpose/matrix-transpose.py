import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A=np.array(A)
    sa = np.shape(A)
    ts = (sa[1], sa[0])
    At = np.empty(ts)
    for i in range(sa[0]):
        for j in range(sa[1]):
            At[j,i]=A[i,j]
    return(At)




