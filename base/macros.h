// Copyright 2019 PEACH LAB. All Rights Reserved.
// Author: goat.zhou@foxmail.com

// A macro to disallow the copy constructor and operator= functions.
// This should be used in the private: declarations for a class.

#define DISALLOW_COPY_AND_ASSIGN(type)    \
  type(const type&);                      \
  void operator=(const type&)

