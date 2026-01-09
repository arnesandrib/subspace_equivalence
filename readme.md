# Subspace Equivalence
Welcome!

## How to use
Run the command
```console 
pip install git+https://github.com/arnesandrib/subspace_equivalence
```
Then use 
```python
from compute_sdu import computeSDU
```
Create any s-box with 2^n entries and compute its SDU. E.g.

```python
n = 5
f = (0, 1, 8, 15, 10, 31, 23, 4, 26, 25, 3, 6, 9, 30, 5, 20, 14, 18, 22, 12, 24, 16, 21, 27, 2, 28, 11, 19, 13, 7, 17, 29) # x^3
print(computeSDU( f,n ))
```
