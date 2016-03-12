"""
Based on Long-Period Hash Functions For Procedural Texturing
by Ares Lagae and Philip Dutré.

@inproceedings{LD2006LHFFPT,
  author    = { Lagae, Ares and Dutr\'e, Philip },
  title     = { Long-Period Hash Functions For Procedural Texturing },
  booktitle = { Vision, Modeling, and Visualization 2006 },
  year      = { 2006 },
  editor    = { L. Kobbelt and T. Kuhlen and T. Aach and R. Westermann },
  pages     = { 225--228 },
  address   = { Berlin },
  month     = { November },
  publisher = { Akademische Verlagsgesellschaft Aka GmbH },
  isbn      = { 3-89838-081-5, 1-58603-688-2 },
  lirias    = { https://lirias.kuleuven.be/handle/123456789/132312 },
  url       = { http://www.vmv2006.rwth-aachen.de/ },
}
"""

import numpy

output_filename = 'hash.h'

# List of primes. We don't need too many:
# we want to keep the permutation tables small.
# Also ignore small primes since they have less effect per operation.
primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
          67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
          131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
          193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257]

# Choose the least set of consecutive primes such that
# the ratio between the least and greatest is less than a threshold.
def selectPrimes(count, maximumRatio):
    for primeIndex in range(len(primes) - count):
        least = primes[primeIndex]
        greatest = primes[primeIndex + count - 1]
        if greatest / least < maximumRatio: 
            return primes[primeIndex:(primeIndex+count)]
    raise ValueError("No set of primes within range.")

def generateHeader(count, maximumRatio):
    permutations = [numpy.random.permutation(prime) for prime in
                    selectPrimes(count, maximumRatio)]
    result = ''
    for permutation in permutations:
        result += '__constant__ unsigned int permutation%d[%d] = {' % (
            len(permutation),
            len(permutation))
        result += ', '.join(str(x) for x in permutation)
        result += '};\n'

    return result

print(generateHeader(5, 2.2))
