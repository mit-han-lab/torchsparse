#ifndef _CUCKOO_MULTI_CPU_HPP_
#define _CUCKOO_MULTI_CPU_HPP_
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <google/dense_hash_map>




class HashTableCPU {

private:
    google::dense_hash_map<int64_t, int64_t> hashmap;
    
public:
    HashTableCPU(){
        //hashmap.set_empty_key(0);
    }
    ~HashTableCPU(){}
    int insert_vals(const int64_t * const keys, const int64_t * const vals, const int n);
    void lookup_vals(const int64_t * const keys, int64_t * const results, const int n);

};

#endif

