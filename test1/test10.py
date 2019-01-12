def neighbors(array, radius, x, y):
    arrayRsize, arrayCsize = len(array), len(array[0])
    pos = [[(n+x, i+y) for n in range(-1*radius, radius+1)]
           for i in range(-1*radius, radius+1)
           ]

    def _getNum(rid, cid):
        print(rid)
    return [[_getNum(rid, cid) for rid, cid in row] for row in pos]


arr = [
    [11, 21, 31, 41, 51, 61, 71],
    [12, 22, 32, 42, 52, 62, 72],
    [13, 23, 33, 43, 53, 63, 73],
    [14, 24, 34, 44, 54, 64, 74],
    [15, 25, 35, 45, 55, 65, 75],
    [16, 26, 36, 46, 56, 66, 76],
    [17, 27, 37, 47, 57, 67, 77],
]
print(neighbors(arr, 2, 1, 1))
#print(neighbors(arr, 3, 5, 5))
