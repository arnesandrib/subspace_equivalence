import json
import numpy as np
from numba import njit
from time import perf_counter as pf
from random import shuffle
import sys
from math import ceil


# Specify RAM availability in GB.
AVAILABLE_RAM = 10
_AVAILABLE_RAM_IN_BYTES = AVAILABLE_RAM*10**9
n = 6

# n = 5:
# f = (0, 1, 8, 15, 10, 31, 23, 4, 26, 25, 3, 6, 9, 30, 5, 20, 14, 18, 22, 12, 24, 16, 21, 27, 2, 28, 11, 19, 13, 7, 17, 29) # x^3
# f = (0, 1, 5, 22, 17, 25, 4, 30, 31, 24, 18, 7, 20, 26, 9, 21, 12, 6, 23, 15, 16, 19, 27, 10, 14, 2, 29, 3, 8, 13, 11, 28) # x^5
#f = (0, 1, 20, 4, 29, 16, 26, 31, 24, 19, 7, 8, 27, 25, 21, 10, 22, 13, 30, 23, 3, 18, 17, 11, 15, 12, 2, 6, 9, 5, 28, 14) # x^7
#f = (0, 1, 26, 20, 3, 29, 27, 10, 11, 28, 15, 23, 25, 17, 31, 24, 5, 22, 21, 9, 2, 14, 16, 19, 8, 13, 6, 12, 30, 4, 18, 7) # x^9
#f = (0, 1, 7, 14, 21, 30, 15, 22, 4, 26, 16, 3, 8, 23, 13, 5, 28, 19, 12, 2, 31, 25, 9, 20, 29, 11, 10, 24, 6, 18, 27, 17) # x^11
#f = (0, 1, 28, 19, 23, 8, 18, 6, 13, 5, 27, 17, 14, 7, 2, 12, 24, 10, 3, 16, 9, 20, 15, 22, 25, 31, 30, 21, 29, 11, 4, 26) #x^13
#f = (0, 1, 31, 21, 18, 28, 10, 17, 29, 2, 22, 4, 24, 11, 25, 16, 9, 23, 27, 20, 14, 12, 19, 3, 5, 8, 7, 15, 26, 30, 6, 13) #x^15
#f = (0, 1, 18, 28, 9, 23, 14, 12, 22, 4, 25, 16, 7, 15, 6, 13, 11, 24, 2, 29, 30, 26, 8, 5, 17, 10, 21, 31, 3, 19, 20, 27) #x^30

n = 6
f = (0, 1, 8, 15, 27, 14, 35, 48, 53, 39, 43, 63, 47, 41, 1, 1, 41, 15, 15, 47, 52, 6, 34, 22, 20, 33, 36, 23, 8, 41, 8, 47, 36, 52, 35, 53, 35, 39, 20, 22, 33, 34, 48, 53, 39, 48, 6, 23, 22, 33, 63, 14, 23, 52, 14, 43, 27, 63, 36, 6, 27, 43, 20, 34)

# n = 7
# f = (0, 1, 8, 15, 64, 85, 120, 107, 12, 69, 39, 104, 73, 20, 82, 9, 96, 119, 36, 53, 62, 61, 74, 79, 68, 27, 35, 122, 31, 84, 72, 5, 10, 51, 49, 14, 38, 11, 45, 6, 117, 4, 109, 26, 92, 57, 116, 23, 44, 3, 91, 114, 30, 37, 89, 100, 123, 28, 47, 78, 76, 63, 40, 93, 80, 113, 29, 58, 13, 56, 112, 67, 54, 95, 88, 55, 110, 19, 48, 75, 33, 22, 32, 17, 98, 65, 83, 118, 111, 16, 77, 52, 41, 66, 59, 86, 102, 127, 24, 7, 87, 90, 25, 18, 115, 34, 46, 121, 71, 2, 42, 105, 81, 94, 99, 106, 126, 101, 124, 97, 108, 43, 125, 60, 70, 21, 103, 50)

# n = 8
# f = (0, 1, 204, 124, 142, 244, 176, 228, 102, 76, 68, 79, 62, 224, 24, 223, 71, 167, 84, 27, 122, 186, 29, 161, 88, 192, 15, 92, 114, 216, 155, 188, 51, 110, 60, 13, 38, 139, 59, 206, 34, 80, 149, 154, 169, 207, 174, 182, 31, 74, 183, 190, 112, 208, 8, 156, 12, 171, 197, 100, 225, 21, 7, 123, 173, 157, 105, 83, 221, 152, 245, 2, 42, 75, 189, 148, 131, 160, 11, 220, 61, 170, 49, 73, 93, 150, 45, 39, 128, 85, 199, 162, 222, 144, 172, 9, 44, 138, 198, 159, 96, 86, 130, 28, 137, 72, 235, 242, 46, 111, 191, 175, 57, 81, 5, 70, 108, 237, 194, 52, 195, 164, 135, 229, 94, 64, 238, 107, 151, 14, 115, 16, 55, 65, 120, 118, 30, 43, 168, 58, 136, 48, 133, 125, 19, 193, 10, 153, 203, 99, 145, 25, 147, 69, 98, 90, 103, 22, 210, 247, 17, 217, 201, 211, 40, 209, 143, 3, 196, 185, 20, 63, 77, 23, 230, 240, 218, 121, 234, 181, 233, 251, 231, 227, 87, 36, 134, 177, 91, 202, 226, 241, 129, 26, 113, 41, 37, 97, 246, 200, 213, 248, 163, 158, 95, 54, 101, 184, 56, 35, 67, 249, 104, 140, 214, 215, 4, 166, 165, 53, 78, 146, 18, 89, 6, 187, 253, 126, 219, 119, 255, 127, 236, 32, 250, 116, 50, 47, 243, 180, 254, 252, 117, 232, 132, 205, 212, 66, 141, 82, 109, 33, 179, 239, 178, 106)

# n = 4
# f = (0, 1, 2, 13, 4, 7, 15, 6, 8, 11, 12, 9, 3, 14, 10, 5) # G0
# f = (0, 1, 2, 13, 4, 7, 15, 6, 8, 12, 5, 3, 10, 14, 11, 9) # G3
# f = (0, 1, 2, 13, 4, 7, 15, 6, 8, 12, 14, 11, 10, 9, 3, 5) # G7

# For random examples:
#f = list(range(2**n)); shuffle(f); f = tuple(f)

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


# We do some preprocessing to avoid memory overload
# define max_memory_useage for T of every dimension (probably), and partiiton the tables T in a manner s.t. we do not get memory overflow.

numSubspacesInWorstDim, worstDim = max( (len(sList),i) for i,sList in enumerate(subspaces) )
numSubspaceElementsInWorstDim = len(subspaces[worstDim][0])

minNumberOfNeededPartitions = [0] #index is dimension. Thus, dim 0 gives 0.
numberOfSpacesPerPartition = [1] #index is dimension. Thus, dim 0 gives 0.

# Compute the total number of partitions of subspaces for each dimension.
for dim in range(1,n):
    # Size is 112 + 2*len(array).
    #memoryConsumptionOfT = 112 + 2*len(subspaces[dim])*len(subspaces[dim][0])*(numSubspacesInWorstDim*numSubspaceElementsInWorstDim)
    memoryConsumptionOfT = 112 + 2*len(subspaces[dim])*(numSubspacesInWorstDim)
    formattedNum = ""
    if memoryConsumptionOfT < 10**6:
        formattedNum = f"{memoryConsumptionOfT / 1000:.2f} kB"
    elif memoryConsumptionOfT < 10**9:
        formattedNum = f"{memoryConsumptionOfT / 1000000:.2f} MB"
    else:
        formattedNum = f"{memoryConsumptionOfT / 1000000000:.2f} GB"
    minNumberOfNeededPartitions.append(max(1, ceil(memoryConsumptionOfT / (_AVAILABLE_RAM_IN_BYTES))))
    numberOfSpacesPerPartition.append(max(1, ceil(len(subspaces[dim]) / minNumberOfNeededPartitions[dim]))) # ensure we get minimum the computed number of chunks.
    print(f"dim={dim} has minimal memory consumption {formattedNum}. We partition T in {minNumberOfNeededPartitions[dim]}")
print()

# Partition the subspaces into the list subspacePartitionsLists
subspacePartitionsLists = [subspaces[0] ]
for dim in range(1, n):
    nSubspacesOfCurDim = len(subspaces[dim])
    nPartitions = minNumberOfNeededPartitions[dim]
    partitionLength = numberOfSpacesPerPartition[dim]
    partitions = [ subspaces[dim][ i*partitionLength : (i+1)*partitionLength if i < nPartitions-1 else nSubspacesOfCurDim] for i in range(nSubspacesOfCurDim//partitionLength+1)]
    # TODO: sørge for at ant. subspacer jevner seg ut mot slutten heller enn at vi får 20 subspacer i alle partisjoner unntatt siste som har 4.
    nSpubspacesPlacedinPartition = sum( len(partition) for partition in partitions )
    print([len(partition) for partition in partitions])
    assert nSpubspacesPlacedinPartition == nSubspacesOfCurDim, "eeeerrror"

    subspacePartitionsLists.append(partitions)


# First coordinate is dim, then list of all ss's in 
ss_d_containing_x = []

for dim in range(n+1):
    containing_x = [[] for _ in range(2**n)]
    for x in range(2**n):
        for i,subspace in enumerate(subspaces[dim]):
            if x in subspace:
                containing_x[x].append(i)
        containing_x[x] = np.array(containing_x[x])
    ss_d_containing_x.append(containing_x)
"""
ss_d_containing_x : ss_d_containing_x[dim][x] : list of index of the dimension d subspaces containing x.
"""

# Create list of 
ss_d_partition_x = [[] ] # = ss_d_partition_x[dim][partition_index][x]
for dim in range(1, n):
    curPartition = subspacePartitionsLists[dim] # partition is list lists of subspaces of the same dimension.
    partitionOverview = []
    for subspaceList in curPartition:
        subspace_indices_in_partition = []
        for x in range(2**n):
            subspaces_containing_x = []
            for i,ss in enumerate(subspaceList):
                if x in ss:
                    subspaces_containing_x.append(i) # legger til index for subspacet innad i partisjonen i lista
            subspace_indices_in_partition.append(np.array(subspaces_containing_x, dtype=np.uint16))
            #subspace_indices_in_partition.append(subspaces_containing_x)
        partitionOverview.append(subspace_indices_in_partition)
    ss_d_partition_x.append(partitionOverview)


def getFormattedSizeString(T):
    ramBytes = sys.getsizeof(T) 
    if ramBytes < 10**6:
        s = f"T takes up {ramBytes} bytes = {ramBytes / 1000: .2f} kB"
    elif ramBytes < 10**9:
        s = f"T takes up {ramBytes} bytes = {ramBytes / 1000000: .2f} MB"
    else:
        s = f"T takes up {ramBytes} bytes = {ramBytes / 1000000000: .2f} GB"
    return s

def getTableMax(T):
    m = 0
    for i in range(len(T)):
        for j in range(len(T[i])):
            if i == 0 and j == 0:
                continue
            m = max(m, T[i,j])
    return m

@njit
def computeT(T, n, ss_in_containing_x, ss_out_containing_x):
    for in_diff in range(1,2**n):
        for out_diff in range(2**n):    
            
            for in_ss_index in ss_in_containing_x[in_diff]:
                for out_ss_index in ss_out_containing_x[out_diff]: 
                    T[in_ss_index][out_ss_index] += DDT[in_diff, out_diff]

t0 = pf()

finishedTable = np.zeros((n-1, n-1), dtype=int)
for in_dim in range(1,n):
    print(f"in_dim={in_dim}")
    for out_dim in range(1,n):
        print(f"\tout_dim={out_dim} (in_dim = {in_dim})")
        TMax = 0
        for in_part_index,inPartition in enumerate(subspacePartitionsLists[in_dim]):
            for out_part_index,outPartition in enumerate(subspacePartitionsLists[out_dim]):
                T = np.zeros( (len(inPartition), len(outPartition)), dtype=np.uint16)
                # print(f"sending in \n\t{ss_d_partition_x[in_dim][in_part_index]}")
                computeT(T, n, ss_d_partition_x[in_dim][in_part_index], ss_d_partition_x[out_dim][out_part_index]) # vil passere in_partition(_index??) og out_partition(_index??)
                TMax = max(TMax, getTableMax(T))
                inPartitionProgress = in_part_index/len(subspacePartitionsLists[in_dim])
                outPartitionProgress = out_part_index/len(subspacePartitionsLists[out_dim])
                print(f"At inPartition {in_part_index}/{len(subspacePartitionsLists[in_dim])}={inPartitionProgress*100:.1f}% and outPartition {out_part_index}/{len(subspacePartitionsLists[out_dim])} = {outPartitionProgress*100:.1f}%. Current TMax = {TMax} ----- {getFormattedSizeString(T)} ----- in_dim={in_dim} | out_dim={out_dim} ----- best in table : {getTableMax(T)}")
        finishedTable[in_dim-1, out_dim-1] = TMax

    print(getFormattedSizeString(T))
    ramBytes = sys.getsizeof(T)
    if ramBytes > _AVAILABLE_RAM_IN_BYTES:
        exit("Program terminated. Not enough RAM...")

        #print(m)

t1 = pf()

print(finishedTable)
print(f"Total computation took {int((t1-t0) // 60)} minutes and {(t1-t0) % 60:.2f} seconds.")