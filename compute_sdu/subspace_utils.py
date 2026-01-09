import numpy as np
from numba import njit

from .generate_subspaces import gaussean_binomial_coeff, get_all_bases


@njit(cache=True)
def msb_index(x):
    """
    Returns the bit position of the most significant bit of x
    
    :param x: uint*
    """
    # NTOE: Maps 0 to -1.
    msb = 0
    while x > 0:
        x >>= 1
        msb += 1
    return msb - 1

@njit(cache=True)
def isInBasis( vector,basis ):
    """
    Assumes the basis elements all the only ones with 1 in their pivot position.
    Basis is sorted descendingly.
    """
    if vector == 0: return True

    x = vector
    
    for i in range(len(basis)):
        b = basis[i]
        # xor b and x if x has 1 in pivot column of b. 
        if x & (1 << msb_index(b)):
            x ^= b
        if x == 0: return True
    
    if x == 0: return True
    return False

@njit(cache=True)
def allSpaceElements( basis,space_holder ):
    k = len(basis)
    assert k < 16, "big k"  # TODO : WHY IS THIS RESTRICTION ENFORCED??
    for indicator in range(2**k):
        v = 0
        for i in range(len(basis)):
            if indicator & (1 << i):
                v ^= basis[i]
        space_holder[indicator] = v

@njit(cache=True)
def getLowestTranslationPoints(basis, n):
    """
    Returns list of the lowest translation points giving rise to distinct affine subspaces.
    n: dimension of ambient space
    """
    k = len(basis)
    tPoints = np.empty(2**(n-k), dtype=np.uint16)
    n_t_points_found = 0
    for a in range(2**n):
        for b in range(n_t_points_found):
            if isInBasis(a^tPoints[b], basis):
                break
        else:
            tPoints[n_t_points_found] = a
            n_t_points_found += 1
        if n_t_points_found == 2**(n-k): break
    return tPoints

@njit(cache=True)
def allAffineSpacesGivenBasis( basis,n ):
    """
    n: size of ambient space
    """
    k = len(basis)
    affine_spaces = np.empty( ( 2**(n-k),2**k ),dtype=np.uint16 )

    t_points = getLowestTranslationPoints( basis,n )
    linspace_elements = np.empty( (2**k),dtype=np.uint16 )
    allSpaceElements( basis,linspace_elements )


    for i in range(len(t_points)):
        t_point = t_points[i]
        for j in range(2**k):
            affine_spaces[i] = linspace_elements^t_point

    return affine_spaces

def getAllAffineSpacesOfDimension( n,k ):
    """
    Get a numpy 2d array whose entries are the points of affine spaces of dimension k living in F_{2^n}
    
    :param n: Dimension of ambient space
    :param k: Dimension of the affine spaces
    """
    assert k <= n and k >= 0, "n and k must be compatible."

    if k == 0:
        spaces = np.empty( (2**n,1),np.uint16 )
        for i in range( 2**n ):
            spaces[i] = i
        return spaces

    # Below "spaces" means affine spaces.

    nBases = gaussean_binomial_coeff( n,k,2 )
    nSpaces = nBases * 2**(n-k)
    spaces = np.empty( ( nSpaces,2**k ),np.uint16 )

    bases = get_all_bases( n,k )
    for i in range( len(bases) ):
        basis = bases[i]
        res = allAffineSpacesGivenBasis( basis,n )
        spaces[ i * 2**(n-k) : (i+1)*2**(n-k) ] = res
    
    return spaces

# NOTE: generates a list, so cannot be jitted.
def getNumpyAffineSpaces( n:int ):
    """
    Docstring for getNumpyAffineSpaces
    
    :param n: dimension of ambient space
    :return: list of numpy arrays of affine subspaces, where first entry of the list is the translation point
    """
    affineSpaces = [ ]
    for dim in range(0,n+1):
        cur_dim_affine_spaces = getAllAffineSpacesOfDimension( n,dim )
        affineSpaces.append( cur_dim_affine_spaces )
    
    return affineSpaces

@njit(cache=True)
def addToBasisWithLength( vector,basis,basisLength ):
    """
    Presupposes that basis is on reduced row echelon form. 
    Presupposes there is room in basis, i.e., len(basis) - len(basisLength) > 0
    """
    x = vector
    
    #Update every vector in current basis and append vector in the end.

    for i in range(basisLength):
        if x & (1 << msb_index(basis[i])):
            x ^= basis[i]

    for i in range(basisLength):
        if x > basis[i]:
            for j in range( basisLength,i,-1 ):
                basis[j] = basis[ j-1 ]
            basis[i] = x
            break
    else:
        basis[basisLength] = x

@njit(cache=True)
def isInBasisWithLength( vector,basis,basisLength ):
    """
    Assumes the basis elements all the only ones with 1 in their pivot position.
    Basis is sorted descendingly.
    """
    if vector == 0: return True

    x = vector
    
    for i in range( basisLength ):
        b = basis[i]
        # xor b and x if x has 1 in pivot column of b. 
        if x & (1 << msb_index(b)):
            x ^= b
        if x == 0: return True
    
    if x == 0: return True
    return False

@njit(cache=True)
def buildSpaceDim_n( potentialBasisVectors,n ):
    """
    potentialBasisVectors: a prioritized list of candidates for being in the basis.
    Builds a basis of n lin ind. elements.
    n: number of vectors in resulting basis / dimension of space spanned by basis
    """
    #assert potentialBasisVectors[0] != 0, "Cannot add zero vectors to basis"
    # if len(potentialBasisVectors) == 0:
    #     return np.empty( 0,dtype=np.uint16 )
    if potentialBasisVectors[0] == 0: return np.zeros(1,np.uint16)
    basis = np.zeros( n,np.uint16 )
    basis[0] = potentialBasisVectors[0]
    curBasisLenght = 1
    for i in range( 1,len( potentialBasisVectors ) ):
        pb = potentialBasisVectors[i] # potential basis vector
        if not isInBasisWithLength( pb,basis,curBasisLenght ):
            addToBasisWithLength( pb,basis,curBasisLenght )
            curBasisLenght += 1
        if curBasisLenght == n:
            break
    
    return basis

@njit(cache=True)
def allAffineSpaceElements( basis,a ):
    """
    basis: basis elements as np.uint16's
    a: translation points
    """
    k = len( basis )
    affine_space = np.empty( 2**k,np.uint16 )
    allSpaceElements( basis,affine_space )
    return affine_space^a

@njit(cache=True)
def buildAffineHull( S,n ):
    """
    S: np.array of vectors to be included in some prioritized order
    n: final dimension of the basis affine hull

    Note: as opposed to the buildlinearspace, we may return this as a full numpy array of space points.    
    """
    
    basePoint = S[0]
    B = buildSpaceDim_n( S[1:]^basePoint,n )

    space = allAffineSpaceElements( B,basePoint )
    return space
