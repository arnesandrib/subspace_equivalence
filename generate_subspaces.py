import numpy as np
from numba import njit

@njit(cache=True)
def gaussean_binomial_coeff( n,k,p ):
    num   = 1
    denom = 1
    for i in range( k ):
        num       *= (1 - (p**(n-i)))
        denom     *= (1 - p**(1+i))
    
    return num // denom

@njit(cache=True)
def comb( n,k ):
    if k < 0 or k > n: return 1

    if k > n - k:
        k = n - k
    
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1) # res * n-1 is always divisible by i+1.

    return res

@njit(cache=True)
def next_pivots( cur_pivots,n ):
    """
    in-place lexicographically gives the positions of the next pivots.
    returns True if a next configuration is possible, False if cur_pivots is the last one.
    """

    # find first non-zero entry from the right
    k = len(cur_pivots)

    for i in range( k-1,-1,-1 ):
        if cur_pivots[i] != i + (n-k):
            cur_pivots[i] += 1
            for j in range( i+1,k ):
                cur_pivots[j] = cur_pivots[j - 1] + 1
            return True
    return False

@njit(cache=True)
def countFreePositions( pivots,n ):
    """
    Counts the number of free positions in the generator matrix for the subspace, where the pivots of said generator matrix are in the positions described by "pivots".

    :param pivots: positions of the pivots for each row i. Assumed to be ascending
    :param n: ambient dimension of space
    :return: number of positions that can be freely assigned to 0 or 1
    :rtype: int
    """
    k = len(pivots)

    pivotIndicator = np.zeros( n,np.uint8 )
    for i in range(k):
        pivotIndicator[ pivots[i] ] = 1
    
    numFree = 0
    for j in range( n ):
        if pivotIndicator[ j ] == 0:
            # Now column j is not pivot, so all rows whose pivot has position < j are free
            for i in range( k ):
                if pivots[ i ] < j: numFree += 1
    return numFree

@njit(cache=True)
def get_all_bases( n,k ):
    """
    Every basis is given descendingly according to pivots (thus also in numerical value).
    
    :param n: Dimension of ambient space
    :param k: Dimension of subspaces whose bases are returned.
    :return: Array of every k-subspace basis of F_2^n as packed bits.
    :rtype: np.ndarray (2d)
    """

    if k == 0:
        return np.zeros((1,1),np.uint16)
    nSpaces = gaussean_binomial_coeff( n,k,2 )
    
    bases = np.empty( (nSpaces,k),np.uint16 )
    basisCount = 0

    pivots = np.arange( 0,k,1,np.uint16 )
    

    for pivotCombIndex in range( comb(n,k) ):
        # Now the pivots are decided. For all relevant bits, let's do the thing.
        numFree = countFreePositions( pivots,n )
        pivotIndicator = np.zeros( n,np.uint8 )
        for i in range(k):
            pivotIndicator[ pivots[i] ] = 1
        
        bases[ basisCount:basisCount + 2**numFree ] = 2**(n-pivots-1)

        for freeConfiguration in range( 2**(numFree) ):
            fbn = 0
            for j in range( n ):
                if pivotIndicator[ j ] == 0:
                    for i in range( k ):
                        if pivots[ i ] < j:
                            bases[ basisCount ][ i ] ^= (1 << (n-j-1)) * ((freeConfiguration >> fbn) & 1)
                            fbn += 1
            basisCount += 1
        next_pivots(pivots,n)
        
    return bases
