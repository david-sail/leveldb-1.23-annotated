// Copyright (c) 2012 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// A filter block is stored near the end of a Table file.  It contains
// filters (e.g., bloom filters) for all data blocks in the table combined
// into a single filter block.

#ifndef STORAGE_LEVELDB_TABLE_FILTER_BLOCK_H_
#define STORAGE_LEVELDB_TABLE_FILTER_BLOCK_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "leveldb/slice.h"
#include "util/hash.h"

namespace leveldb {

class FilterPolicy;

// A FilterBlockBuilder is used to construct all of the filters for a
// particular Table.  It generates a single string which is stored as
// a special block in the Table.
//
// The sequence of calls to FilterBlockBuilder must match the regexp:
//      (StartBlock AddKey*)* Finish
class FilterBlockBuilder {
 public:
  explicit FilterBlockBuilder(const FilterPolicy*);

  FilterBlockBuilder(const FilterBlockBuilder&) = delete;
  FilterBlockBuilder& operator=(const FilterBlockBuilder&) = delete;

  //根据block_offset创建filter;默认每2K block_offset就创建一个filter;如果不够2K啥都不做
  //block_offset代表data block的偏移量
  void StartBlock(uint64_t block_offset);
  //把key保存到key列表中。更新keys_,start_;不做具体序列化工作
  void AddKey(const Slice& key);
  //完成filter block构建;包含落盘剩余key;在文件尾部增加偏移量以及footer
  //filter data | filter data offset | filter offset's offset | footer
  Slice Finish();

 private:
 //用keys_和start_中保存的key列表创建一个filter data
  void GenerateFilter();

  const FilterPolicy* policy_;   //filter policy，比如BloomFilter;布隆过滤器构建器
  //keys_是一个string类型，所有key都顺序存放在keys_中
  //start_是一个数组，保存了每个key在keys_中的offset。start_[i]表示第i个key的offset。
  std::string keys_;             // Flattened key contents  ;key 列表,将数组拉平存储
  std::vector<size_t> start_;    // Starting index in keys_ of each key; key位置列表
  std::string result_;           // Filter data computed so far; filter列表的buffer;所有filter顺序存储在result_中的
  std::vector<Slice> tmp_keys_;  // policy_->CreateFilter() argument; 创建filter的时候,用来临时存储已插入的key
  std::vector<uint32_t> filter_offsets_; ////filter的offset; filter_offsets_[i]表示第i个filter在result_中的offset
};

class FilterBlockReader {
 public:
  // REQUIRES: "contents" and *policy must stay live while *this is live.
  FilterBlockReader(const FilterPolicy* policy, const Slice& contents);
  //查询布隆过滤器中是否存在key键
  //block_offset代表data block的偏移量
  bool KeyMayMatch(uint64_t block_offset, const Slice& key);

 private:
  const FilterPolicy* policy_; //布隆过滤器
  const char* data_;    // Pointer to filter data (at block-start)
  const char* offset_;  // Pointer to beginning of offset array (at block-end)  filter data offset‘s offset
  size_t num_;          // Number of entries in offset array
  size_t base_lg_;      // Encoding parameter (see kFilterBaseLg in .cc file)
};

}  // namespace leveldb

#endif  // STORAGE_LEVELDB_TABLE_FILTER_BLOCK_H_
