B = list()
A = list()

def jaccard_distance(A, B):
  nominator = A.symmetric_difference(B)

  denominator = A.union(B)

  distance = len(nominator)/len(denominator)

  return distance
