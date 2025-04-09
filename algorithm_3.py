import json
import numpy as np
from numba import njit, prange, get_num_threads
from time import perf_counter as pf

# Specify RAM availability in GB.
AVAILABLE_RAM = 10
_AVAILABLE_RAM_IN_BYTES = AVAILABLE_RAM*10**9

n = 8


if n == 5:
    f = (0, 1, 8, 15, 10, 31, 23, 4, 26, 25, 3, 6, 9, 30, 5, 20, 14, 18, 22, 12, 24, 16, 21, 27, 2, 28, 11, 19, 13, 7, 17, 29) # x^3
elif n == 6:
    f = (0, 1, 8, 15, 27, 14, 35, 48, 53, 39, 43, 63, 47, 41, 1, 1, 41, 15, 15, 47, 52, 6, 34, 22, 20, 33, 36, 23, 8, 41, 8, 47, 36, 52, 35, 53, 35, 39, 20, 22, 33, 34, 48, 53, 39, 48, 6, 23, 22, 33, 63, 14, 23, 52, 14, 43, 27, 63, 36, 6, 27, 43, 20, 34)
elif n == 7:
    f = (0, 1, 8, 15, 64, 85, 120, 107, 12, 69, 39, 104, 73, 20, 82, 9, 96, 119, 36, 53, 62, 61, 74, 79, 68, 27, 35, 122, 31, 84, 72, 5, 10, 51, 49, 14, 38, 11, 45, 6, 117, 4, 109, 26, 92, 57, 116, 23, 44, 3, 91, 114, 30, 37, 89, 100, 123, 28, 47, 78, 76, 63, 40, 93, 80, 113, 29, 58, 13, 56, 112, 67, 54, 95, 88, 55, 110, 19, 48, 75, 33, 22, 32, 17, 98, 65, 83, 118, 111, 16, 77, 52, 41, 66, 59, 86, 102, 127, 24, 7, 87, 90, 25, 18, 115, 34, 46, 121, 71, 2, 42, 105, 81, 94, 99, 106, 126, 101, 124, 97, 108, 43, 125, 60, 70, 21, 103, 50)
elif n == 8:
    f = (0, 1, 204, 124, 142, 244, 176, 228, 102, 76, 68, 79, 62, 224, 24, 223, 71, 167, 84, 27, 122, 186, 29, 161, 88, 192, 15, 92, 114, 216, 155, 188, 51, 110, 60, 13, 38, 139, 59, 206, 34, 80, 149, 154, 169, 207, 174, 182, 31, 74, 183, 190, 112, 208, 8, 156, 12, 171, 197, 100, 225, 21, 7, 123, 173, 157, 105, 83, 221, 152, 245, 2, 42, 75, 189, 148, 131, 160, 11, 220, 61, 170, 49, 73, 93, 150, 45, 39, 128, 85, 199, 162, 222, 144, 172, 9, 44, 138, 198, 159, 96, 86, 130, 28, 137, 72, 235, 242, 46, 111, 191, 175, 57, 81, 5, 70, 108, 237, 194, 52, 195, 164, 135, 229, 94, 64, 238, 107, 151, 14, 115, 16, 55, 65, 120, 118, 30, 43, 168, 58, 136, 48, 133, 125, 19, 193, 10, 153, 203, 99, 145, 25, 147, 69, 98, 90, 103, 22, 210, 247, 17, 217, 201, 211, 40, 209, 143, 3, 196, 185, 20, 63, 77, 23, 230, 240, 218, 121, 234, 181, 233, 251, 231, 227, 87, 36, 134, 177, 91, 202, 226, 241, 129, 26, 113, 41, 37, 97, 246, 200, 213, 248, 163, 158, 95, 54, 101, 184, 56, 35, 67, 249, 104, 140, 214, 215, 4, 166, 165, 53, 78, 146, 18, 89, 6, 187, 253, 126, 219, 119, 255, 127, 236, 32, 250, 116, 50, 47, 243, 180, 254, 252, 117, 232, 132, 205, 212, 66, 141, 82, 109, 33, 179, 239, 178, 106)

    AESbox = \
    [    0xE2, 0x4E, 0x54, 0xFC, 0x94, 0xC2, 0x4A, 0xCC, 0x62, 0x0D, 0x6A, 0x46, 0x3C, 0x4D, 0x8B, 0xD1,
        0x5E, 0xFA, 0x64, 0xCB, 0xB4, 0x97, 0xBE, 0x2B, 0xBC, 0x77, 0x2E, 0x03, 0xD3, 0x19, 0x59, 0xC1,
        0x1D, 0x06, 0x41, 0x6B, 0x55, 0xF0, 0x99, 0x69, 0xEA, 0x9C, 0x18, 0xAE, 0x63, 0xDF, 0xE7, 0xBB,
        0x00, 0x73, 0x66, 0xFB, 0x96, 0x4C, 0x85, 0xE4, 0x3A, 0x09, 0x45, 0xAA, 0x0F, 0xEE, 0x10, 0xEB,
        0x2D, 0x7F, 0xF4, 0x29, 0xAC, 0xCF, 0xAD, 0x91, 0x8D, 0x78, 0xC8, 0x95, 0xF9, 0x2F, 0xCE, 0xCD,
        0x08, 0x7A, 0x88, 0x38, 0x5C, 0x83, 0x2A, 0x28, 0x47, 0xDB, 0xB8, 0xC7, 0x93, 0xA4, 0x12, 0x53,
        0xFF, 0x87, 0x0E, 0x31, 0x36, 0x21, 0x58, 0x48, 0x01, 0x8E, 0x37, 0x74, 0x32, 0xCA, 0xE9, 0xB1,
        0xB7, 0xAB, 0x0C, 0xD7, 0xC4, 0x56, 0x42, 0x26, 0x07, 0x98, 0x60, 0xD9, 0xB6, 0xB9, 0x11, 0x40,
        0xEC, 0x20, 0x8C, 0xBD, 0xA0, 0xC9, 0x84, 0x04, 0x49, 0x23, 0xF1, 0x4F, 0x50, 0x1F, 0x13, 0xDC,
        0xD8, 0xC0, 0x9E, 0x57, 0xE3, 0xC3, 0x7B, 0x65, 0x3B, 0x02, 0x8F, 0x3E, 0xE8, 0x25, 0x92, 0xE5,
        0x15, 0xDD, 0xFD, 0x17, 0xA9, 0xBF, 0xD4, 0x9A, 0x7E, 0xC5, 0x39, 0x67, 0xFE, 0x76, 0x9D, 0x43,
        0xA7, 0xE1, 0xD0, 0xF5, 0x68, 0xF2, 0x1B, 0x34, 0x70, 0x05, 0xA3, 0x8A, 0xD5, 0x79, 0x86, 0xA8,
        0x30, 0xC6, 0x51, 0x4B, 0x1E, 0xA6, 0x27, 0xF6, 0x35, 0xD2, 0x6E, 0x24, 0x16, 0x82, 0x5F, 0xDA,
        0xE6, 0x75, 0xA2, 0xEF, 0x2C, 0xB2, 0x1C, 0x9F, 0x5D, 0x6F, 0x80, 0x0A, 0x72, 0x44, 0x9B, 0x6C,
        0x90, 0x0B, 0x5B, 0x33, 0x7D, 0x5A, 0x52, 0xF3, 0x61, 0xA1, 0xF7, 0xB0, 0xD6, 0x3F, 0x7C, 0x6D,
        0xED, 0x14, 0xE0, 0xA5, 0x3D, 0x22, 0xB3, 0xF8, 0x89, 0xDE, 0x71, 0x1A, 0xAF, 0xBA, 0xB5, 0x81 ]
    f = AESbox

print (AESbox)

# Compute DDT
DDT = np.zeros((2**n, 2**n), dtype=np.uint16)
for x in range(2**n):
    for y in range(2**n):
        diff = x^y 
        outDiff = f[x]^f[y]
        DDT[diff, outDiff] += 1

f = np.array(f, np.int16)

fname = f"./subspaces/subspaces_n={n}.dat"
with open(fname, "r") as infile:
    subspaces = json.load(infile)
print(f"Subspaces loaded.")

@njit
def computeRowSum(U, v, DDT):
    s = np.uint16(0)
    for u in U:
        if u == 0:
            continue
        s += DDT[u, v]
    return s

@njit
def computeColSum(V, u, DDT):
    s = 0
    for v in V:
        if v == 0 and u == 0:
            continue
        s += DDT[u, v]
    return s

@njit
def algorithm_1(U, V, rowSumU, colSumV):
    tot = 0
    if U.size > V.size:
        for i in range(V.size):
            tot += rowSumU[V[i]]
    else:
        for i in range(U.size):
            tot += colSumV[U[i]]
    return tot

@njit(parallel=True)
def getTMaxVal_parallel_safe(in_spaces, out_spaces, row_sums_cur_dim, col_sums_cur_dim):
    len_U = len(in_spaces)
    len_V = len(out_spaces)

    # Start with zero, then compare in serial pass
    max_val = 0

    # Thread-local storage
    assert len_U * len_V * 16 < _AVAILABLE_RAM_IN_BYTES, "Not enough space"
    local_max = np.zeros(len_U * len_V, dtype=np.uint16)

    for i in prange(len_U * len_V):
        index_U = i // len_V
        index_V = i % len_V

        U = in_spaces[index_U]
        V = out_spaces[index_V]
        rowSumU = row_sums_cur_dim[index_U]
        colSumV = col_sums_cur_dim[index_V]
        local_max[i] = algorithm_1(U, V, rowSumU, colSumV)

    for i in range(len_U * len_V):
        if local_max[i] > max_val:
            max_val = local_max[i]

    return max_val

@njit(parallel=True)
def getTMaxVal_parallel_minalloc(in_spaces, out_spaces, row_sums_cur_dim, col_sums_cur_dim):
    len_U = len(in_spaces)
    len_V = len(out_spaces)
    n_threads = get_num_threads()

    # One small value per thread
    thread_max = np.zeros(n_threads, dtype=np.uint16)

    for i in prange(len_U * len_V):
        thread_id = i % n_threads  # crude but safe mapping of tasks to thread "buckets"
        index_U = i // len_V
        index_V = i % len_V

        U = in_spaces[index_U]
        V = out_spaces[index_V]
        rowSumU = row_sums_cur_dim[index_U]
        colSumV = col_sums_cur_dim[index_V]

        TUV = algorithm_1(U, V, rowSumU, colSumV)
        if TUV > thread_max[thread_id]:
            thread_max[thread_id] = TUV

    # Final reduction over thread-local max values
    max_val = np.uint16(0)
    for i in range(n_threads):
        if thread_max[i] > max_val:
            max_val = thread_max[i]

    return max_val

@njit
def computeRowSums(dim, space, DDT):
    rowSumsCurDim = np.zeros((len(space), 2**n), np.uint16)
    for index_U in range(len(space)):
        U = space[index_U]
        for v in range(2**n):
            rowSum = computeRowSum(U, v, DDT)
            rowSumsCurDim[index_U][v] = rowSum
    return rowSumsCurDim

@njit
def computeColSums( dim,space,DDT ):
    colSumsCurDim = np.zeros((len(space), 2**n), np.uint16)
    for index_V in range(len(space)):
        V = space[index_V]
        for u in range(2**n):
            colSum = computeColSum(V, u, DDT)
            colSumsCurDim[index_V][u] = colSum
    return colSumsCurDim

if __name__ == "__main__":

    numpySpaces = []
    # load all subspaces to wanted data structure
    for dim in range(len(subspaces)):
        npSpacesCurDim = np.array(subspaces[dim], dtype=np.uint16)
        numpySpaces.append(npSpacesCurDim)
    
    rowSums = [[] ] # entry 0 is dim. Entry 1 is index of U. Entry 2 is v
    colSums = [[] ] # entry 0 is dim. Entry 1 is index of V. Entry 2 is u
    for dim in range(1,n):
        print(f"performing dim {dim} of preprocessing")
        rowSums.append(computeRowSums( dim,numpySpaces[dim],DDT ))
        colSums.append(computeColSums( dim,numpySpaces[dim],DDT ))

    outTable = np.zeros((n-1, n-1), dtype=int)

    t0 = pf()
    for in_dim in range(1, n):
        print(f"In_dim = {in_dim}")
        for out_dim in range(1, n):
            print(f"\tout_dim = {out_dim}")
            in_spaces = numpySpaces[in_dim]
            out_spaces = numpySpaces[out_dim]

            #outTable[in_dim-1, out_dim-1] = getTMaxVal(in_spaces, out_spaces, rowSums[in_dim], colSums[out_dim])
            #outTable[in_dim-1, out_dim-1] = getTMaxVal_parallel_safe(in_spaces, out_spaces, rowSums[in_dim], colSums[out_dim])
            outTable[in_dim-1, out_dim-1] = getTMaxVal_parallel_minalloc(in_spaces, out_spaces, rowSums[in_dim], colSums[out_dim])
    
    t1 = pf()
    print(f"Total time = {t1-t0}")
    print(outTable)