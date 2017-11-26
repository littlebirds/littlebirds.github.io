### Advantages of tries over hash tables:
- Predictable O(k) lookup time where k is the size of the key.
- Early termination of lookup if it is not there.
- No need for hash function.
- Supports ordered traversal.
- You can quickly lookup prefixes of keys, enumerate all entires with a given prefix etc.
- If there are many common prefixes, the space required is shared.


### TODO List
1. Switching from std::map to boost::flat_map and bench mark memory consumption/performance
2. Implement erase method.
