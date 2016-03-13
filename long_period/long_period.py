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

import fractions
import numpy

# List of primes. We don't need too many:
# we want to keep the permutation tables small.
# Also ignore small primes since they have less effect per operation.
_primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
          67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
          131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
          193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257]

_permutationVariableName = '_permutation'

# Choose the least set of consecutive primes such that
# the ratio between the least and greatest is less than a threshold.
# TODO: Change this to select (relative?) primes near outputRange.
def selectFactors(outputRange, factorCount, maximumRatio):
    filteredPrimes = [x for x in _primes if fractions.gcd(outputRange, x) == 1]
    for primeIndex in range(len(filteredPrimes) - factorCount):
        least = filteredPrimes[primeIndex]
        greatest = filteredPrimes[primeIndex + factorCount - 1]
        result = filteredPrimes[primeIndex:(primeIndex+factorCount)]
        if greatest / least < maximumRatio:
            result.append(outputRange)
            result.sort()
            return result
    raise ValueError("No set of primes within range.")

def generateHashTerm(offset, factor, argi):
    if offset > 0: offsetAddString = '%3d + ' % offset
    else: offsetAddString = ' ' * 6
    
    if argi == 0:
        moduloTerm = '(arg0 %% %d)' % factor
    else:
        moduloTerm = '((%s + arg%d) %% %d)' % (
            generateHashTerm(offset, factor, argi - 1),
            argi,
            factor)

    return '%s%d[%s%s]' % (
        _permutationVariableName,
        factor,
        offsetAddString,
        moduloTerm)

def generateHashFunction(outputRange, argc, factors):
    result = '__device__ unsigned int long_period_hash('
    result += ', '.join('unsigned int arg%d' % i for i in range(argc))
    result += '){\n'
    result += '    unsigned int result = 0;\n'
    result += '\n'

    offset = 0
    
    for factorIndex, factor in enumerate(factors):
        result += '    result = result + %s;\n' % generateHashTerm(offset, factor, argc - 1)
        offset += factor
    
    result += '\n'
    
    result += '    return (result %% %d);\n' % outputRange
    result += '}\n'

    result += '\n'
    return result

def generateHeader(outputRange, maxArgc, factorCount = 4, maximumRatio = 2.2):
    factors = selectFactors(outputRange, factorCount, maximumRatio)
    permutations = [numpy.random.permutation(factor) for factor in
                    factors]
    result = ''

    # constant section
    for permutation in permutations:
        result += '__constant__ unsigned int %s%d[%d] = {' % (
            _permutationVariableName,
            len(permutation),
            len(permutation))
        result += ', '.join(str(x) for x in permutation)
        result += '};\n'

    result += '\n'

    for argc in range(1, maxArgc+1):
        result += generateHashFunction(outputRange, argc, factors)

    return result
