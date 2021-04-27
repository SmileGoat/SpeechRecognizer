// Copyright (c) 2021 PeachLab. All Rights Reserved.
// Author : goat.zhou@qq.com (Yang Zhou)

#include <cstdio>
#include <iostream>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

#include "kaldi_utils/util/table-types.h"

#define LOG(x) std::cerr

int main(int argc, char** argv) {

  const char* filename = argv[1];
  const char* read_ark = argv[2];
  const char* write_ark = argv[3];

  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(filename);
  
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter>interpreter;
  builder(&interpreter);

  LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
  LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
  LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
  LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

  int t_size = interpreter->tensors_size();
  int input = interpreter->inputs()[0];

  std::vector<int> k{1, 40, 20};
  interpreter->ResizeInputTensor(0, k);
  interpreter->AllocateTensors();
  LOG(INFO) << "input: " << input << "\n";
  
  kaldi::SequentialBaseFloatMatrixReader reader(read_ark);
  kaldi::BaseFloatWriter writer(write_ark);

  for (; !reader.Done(); reader.Next()) {
    std::string utt = reader.Key();
    const kaldi::Matrix<BaseFloat> &mat = feature_reader.Value();
    float* data = mat.Data();    
    memcpy(interpreter->typed_input_tensor<float>(0), data, data.SizeInBytes());

  } 

  std::vector<float> vect{1.9,3.9,2.9};
  memcpy(interpreter->typed_input_tensor<float>(0), vect.data(), sizeof(vect));

  LOG(INFO) << "fill the input \n"; 
  interpreter->Invoke();

  float* output;
  output = interpreter->typed_tensor<float>(3);
  LOG(INFO) << "the inference output:" << output[0] << "\n";

  return 0;
}


