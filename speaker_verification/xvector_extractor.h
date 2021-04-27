#ifndef XVECTROR_EXTRACTOR_H_
#define XVECTROR_EXTRACTOR_H_

#include "base/timer.h"
#include "base/kaldi-common.h"
#include "interface/nnet_computer_interface.h"
#include "util/common-utils.h"

#include <memory>

const int32 kChunkSize = 100;
const int32 kMinChunkSize = 20;
using kaldi::Matrix;
using kaldi::Vector;
using namespace kaldi;

class XvectorExtractor {
  public:
   explicit  XvectorExtractor(const std::string& nnet_model);
   bool ExtractXvector(const Matrix<BaseFloat>& features, Vector<BaseFloat>* xvector);

  private:
    void ExtractEmbedding(const kaldi::MatrixBase<BaseFloat>& features,
		                      Vector<BaseFloat>* xvector);
    std::unique_ptr<NnetComputerInterface> nnet_computer_;
};

#endif  // #define XVECTOR_EXTRACTOR_H_

