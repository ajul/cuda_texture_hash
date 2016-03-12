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

# List of primes. We don't need too many: we want to keep the permutation tables
# small.
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
          67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
          131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191,
          193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257]

