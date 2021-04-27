#include "inference/tflite_nnet_computer.h"

#include "tensorflow/lite/c/common.h"

#include <memory>

TFliteNnetComputer::Init(const std::string& nnet_model) {
  model_ =  tflite::FlatBufferModel::BuildFromFile(nnet_model.c_str());
  tflite::ops::builtin::BuiltinOpResolver resolver;
  builder_.reset(new tflite::InterpreterBuilder(*model_, resolver));
  (*builder_)(&interpreter_);
}

void XvectorExtractor::XvectorCompute(const kaldi::MatrixBase<BaseFloat>& features,
                                      Vector<BaseFloat>* xvector) {
  KALDI_LOG << " feature number rows: " << features.NumRows() << 
	             " " << features.NumCols();
  Timer time;
	xvector->reverse();
  int32 input_idx = interpreter_->inputs()[0];
  BaseFloat* input_pointer = interpreter_->typed_input_tensor<float>(0);
  TfLiteTensor* input_tensor = interpreter_->tensor(input_idx);
  // fill the input data, rewrite
  memcpy(input_pointer, features.Data(), input_tensor->bytes);

  interpreter_->Invoke();
  BaseFloat* output = interpreter_->typed_output_tensor<float>(0);
  int32 output_idx = interpreter_->outputs()[0];
  int32 output_bytes = interpreter_->tensor(output_idx)->bytes;
  int32 output_dim = output_bytes / sizeof(float);
  xvector->Resize(output_dim);
  memcpy(xvector->Data(), output, output_bytes);
  float second = time.Elapsed();
  KALDI_LOG << "nnet compute time: " << second;
}

