// Copyright (c) 2021 PeachLab. All Rights Reserved.
// Author : goat.zhou@qq.com (Yang Zhou)

namespace goat {

class NnetComputerInterface {
 public:
  virtual ~NnetComputerInterface() {}
  virtual void Init(const std::string& model) = 0;
  virtual void FeedForward(const Matrix<BaseFloat>& features,
	                         Vector<BaseFloat>* inference) const = 0;
};

};
