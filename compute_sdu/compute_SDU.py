import numpy as np
from numba import njit,get_num_threads,prange

from .subspace_utils import buildAffineHull, getNumpyAffineSpaces


DT = np.uint16

@njit(cache=True)
def compute_DDT(f, n):
    DDT = np.zeros( (2**n,2**n),dtype=DT )
    for x in range(2**n):
        for y in range(2**n):
            diff = x^y 
            outDiff = f[x]^f[y]
            DDT[diff, outDiff] += 1
    return DDT

@njit 
def computeSDUBound( n ):
    B = np.empty( (n,n),DT )
    for a in range(n):
        for b in range(n):
            B[a,b] = 2*np.ceil( (2**(n + a + b))/(2**(n+1) - 2) )
    return B

@njit(cache=True)
def computeRowSum(U, v, DDT):
    """
    Vår computeRowSum fikserer en kolonne og summerer opp verdier fra alle rader som svarer til U.
    """
    s = DT(0)
    for i in range(U.size):
        if U[i] == 0:
            continue
        s += DDT[ U[i],v ]
    return s

@njit(cache=True)
def computeColSum(V, u, DDT):
    s = 0
    for j in range(len(V)):
        if V[j] == 0 and u == 0:
            continue
        s += DDT[u, V[j]]
    return s

@njit(cache=True)
def computeRowSums( spaces,DDT,n ):
    """
    rowSums[U_index][v] hvor v svarer til kolonnen
    n: number of input-bits for S-box
    """
    rowSumsCurDim = np.zeros((len(spaces), 2**n), DT)
    for index_U in range(len(spaces)):
        U = spaces[index_U]
        for v in range(2**n):
            rowSum = computeRowSum(U, v, DDT)
            rowSumsCurDim[index_U][v] = rowSum
    return rowSumsCurDim

@njit(cache=True)
def computeColSums( spaces,DDT,n ):
    colSumsCurDim = np.zeros((len(spaces), 2**n), DT)
    for index_V in range(len(spaces)):
        V = spaces[index_V]
        for u in range(2**n):
            colSum = computeColSum(V, u, DDT)
            colSumsCurDim[index_V][u] = colSum
    return colSumsCurDim

@njit(parallel=True, cache=True)
def computeRowOptimals(rowSums, b, n):
    nSpaces = len(rowSums)
    rowOptimals = np.zeros(nSpaces, dtype=DT)
    sortedRowVectors = np.empty((nSpaces, 2**b), dtype=DT)

    for spaceIndex in prange(nSpaces):  # Parallelized outer loop
        rowSumsCurSpace = rowSums[spaceIndex]

        L0 = np.empty(2**n, dtype=DT)
        L1 = np.empty(2**n, dtype=DT)
        for v in range(2**n):
            L0[v] = rowSumsCurSpace[v]
            L1[v] = v

        idx = np.argsort(L0)[::-1]  # sort descending

        top_values = L0[idx[:2**b]]
        top_indices = L1[idx[:2**b]]

        rowOptimals[spaceIndex] = np.sum(top_values)
        sortedRowVectors[spaceIndex] = top_indices

    return rowOptimals, sortedRowVectors

@njit(cache=True)
def computeColOptimals( colSums,a,n ):
    return computeRowOptimals( colSums,a,n )

@njit(cache=True)
def algorithm_1(U, V, rowSumU, colSumV):
    tot = 0
    if U.size > V.size:
        for i in range(V.size):
            tot += rowSumU[V[i]]
    else:
        for i in range(U.size):
            tot += colSumV[U[i]]
    return tot

@njit(parallel=True,cache=True)
def getTMaxVal_parallel( in_spaces,out_spaces,in_indices,out_indices,row_sums_cur_dim,col_sums_cur_dim,n_threads=64 ):
    len_U = len(in_indices)
    len_V = len(out_indices)

    
    thread_max = np.zeros(n_threads, dtype=DT)

    for i in prange(len_U * len_V):
        thread_id = i % n_threads  # to differentiate threads, make parallelization safe.
        index_U = in_indices[i // len_V]
        index_V = out_indices[i % len_V]

        U = in_spaces[index_U]
        V = out_spaces[index_V]
        rowSumU = row_sums_cur_dim[index_U]
        colSumV = col_sums_cur_dim[index_V]

        TUV = algorithm_1(U, V, rowSumU, colSumV)
        if TUV > thread_max[thread_id]:
            thread_max[thread_id] = TUV

    # Pick the highest value.
    max_val = DT(0)
    for i in range(n_threads):
        if thread_max[i] > max_val:
            max_val = thread_max[i]

    return max_val

@njit(cache=True)
def Tf_given_output_space(U, V, colSumV):
    tot = 0
    for i in range(U.size):
            tot += colSumV[U[i]]
    return tot

@njit(cache=True)
def Tf_given_input_space(U, V, rowSumU):
    tot = 0
    for i in range(V.size):
            tot += rowSumU[V[i]]
    return tot

def guessInitialSolutions( inSpaces,outSpaces,a,b,sortedRowVectors,sortedColVectors,inputSpaceRowSums,outputSpaceColSums ):
    """
    Spaces is numpy array of all spaces of the given dimension a.
    """
    M = 0
    for index_U in range(len(inSpaces)):
        U = inSpaces[index_U]
        V = buildAffineHull( sortedRowVectors[ index_U ],b )
        """if len(V) != len(set(V)):
            continue"""
        M = max( M,Tf_given_input_space( U,V,inputSpaceRowSums[index_U] ) )
    for index_V in range(len(outSpaces)):
        V = outSpaces[index_V]
        U = buildAffineHull( sortedColVectors[ index_V ],a )
        M = max( M,Tf_given_output_space( U,V,outputSpaceColSums[index_V] ) )
        
    return M

def computeSDU( f,n ):
    f = np.array( f,DT )
    DDT = compute_DDT(f, n)
    
    numpyAffineSpaces = getNumpyAffineSpaces( n )
    T = np.zeros( ( n,n ),dtype=DT )

    M_guesses = np.zeros( ( n,n ),dtype=DT )
    
    for a in range( n ):
        rowSums = computeRowSums( numpyAffineSpaces[ a ],DDT,n )
        for b in range( n ):
            colSums = computeColSums( numpyAffineSpaces[ b ],DDT,n )
            rowOptimals,sortedRowVectors = computeRowOptimals( rowSums,b,n )
            colOptimals,sortedColVectors = computeColOptimals( colSums,a,n )

            A_order = np.stack( (np.arange( len(numpyAffineSpaces[a]),dtype=DT ),rowOptimals),1,dtype=DT )
            A_order = A_order[np.argsort(A_order[:,1])[::-1]]
            B_order = np.stack( (np.arange( len(numpyAffineSpaces[b]),dtype=DT ),colOptimals),1,dtype=DT )
            B_order = B_order[np.argsort(B_order[:,1])[::-1]]

            M_guess = 0
            if a > 0 and b > 0:
                M_guess = guessInitialSolutions( numpyAffineSpaces[a],numpyAffineSpaces[b],a,b,sortedRowVectors,sortedColVectors,rowSums,colSums )
            M_guesses[ a,b ] = M_guess

            A_max_index = len(A_order)-1
            for i in range(len(A_order)):
                if A_order[i][1] <= M_guess:
                    A_max_index = min(i,A_max_index)
                    break
            
            if not i == 0:
                B_max_index = len(B_order)-1
                for j in range(len(B_order)):
                    if B_order[j][1] <= M_guess:
                        B_max_index = min(j,B_max_index)
                        break
            
            
            if i == 0 or j == 0:
                T[ a,b ] = M_guess    
                continue

            M = max(M_guess, getTMaxVal_parallel( numpyAffineSpaces[a],numpyAffineSpaces[b],A_order[:A_max_index,0],B_order[:B_max_index,0],rowSums,colSums,get_num_threads() ))
            T[ a,b ] = M

    return T


if __name__ == "__main__":
    
    # Quick computation test for x^3 in F_2^5. 
    n = 5
    f = (0, 1, 8, 15, 10, 31, 23, 4, 26, 25, 3, 6, 9, 30, 5, 20, 14, 18, 22, 12, 24, 16, 21, 27, 2, 28, 11, 19, 13, 7, 17, 29) # x^3

    SDU = computeSDU( f,n )
    print(SDU)