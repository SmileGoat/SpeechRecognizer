#ifndef TFLITE_NNET_COMPUTER_H_
#define TFLITE_NNET_COMPUTER_H_

#include "base/kaldi-common.h"
#include "interface/nnet_inference_interface.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"
#include "util/common-utils.h"

using kaldi::Matrix;
using kaldi::Vector;

namespace goat {

class TFliteNnetComputer::NnetComputerInterface {
  public:
   TFliteNnetComputer();
	 void Init(const std::string& nnet_model);
   bool FeedForward(const Matrix<BaseFloat>& features, 
	                  Vector<BaseFloat>* inference) const = 0;

  private:
   std::unique_ptr<tflite::FlatBufferModel> model_;
   std::unique_ptr<tflite::Interpreter> interpreter_;
   std::unique_ptr<tflite::InterpreterBuilder> builder_;
};

}  // namespace goat

#endif  // #define TFLITE_NNET_COMPUTER_H_
