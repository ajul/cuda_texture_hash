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
_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
           67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137,
           139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
           211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277,
           281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
           367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439,
           443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521,
           523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607,
           613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683,
           691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773,
           787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
           877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967,
           971, 977, 983, 991, 997, 1009, 1013, 1019, 1021]

_permutationVariableName = '_permutation'

def selectFactors(outputRange, factorCount):
    aboveCount = factorCount // 2
    belowCount = factorCount - aboveCount
    for primeIndex, prime in enumerate(_primes):
        if prime == outputRange:
            return (_primes[(primeIndex-belowCount):primeIndex] +
                    [outputRange] +
                    _primes[(primeIndex+1):(primeIndex+aboveCount+1)])
        elif prime > outputRange:
            return (_primes[(primeIndex-belowCount):primeIndex] +
                    [outputRange] +
                    _primes[primeIndex:(primeIndex+aboveCount)])

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

def generateHeader(outputRange, maxArgc, factorCount = 4):
    factors = selectFactors(outputRange, factorCount)
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

