#include "speaker_verification/xvector_extractor.h"

XvectorExtractor::XvectorExtractor(const std::string& nnet_model) {
	nnet_computer_->Init(nnet_model);
}

void XvectorExtractor::ExtractEmbedding(const kaldi::MatrixBase<BaseFloat>& features,
                                      Vector<BaseFloat>* xvector) {
  Timer time;
	nnet_computer_->FeedForward(features, xvector);
  float second = time.Elapsed();
  KALDI_LOG << "nnet compute time: " << second;
}

bool XvectorExtractor::ExtractXvector(const Matrix<BaseFloat>& features, 
                                      Vector<BaseFloat>* xvector) {
  int32 num_rows = features.NumRows(); 
  int32 feat_dim = features.NumCols();
	int32 this_chunk_size = kChunkSize;
  int32 min_chunk_size = kMinChunkSize;
  bool pad_input = true;
  //xvector->Resize(xvector_dim, kSetZero);

  int32 num_chunks = floor(
      num_rows / static_cast<BaseFloat>(this_chunk_size));
  if (num_chunks == 0) return false;
  Vector<BaseFloat> xvector_avg(xvector_dim, kSetZero);
  BaseFloat tot_weight = 0.0;

  for (int32 chunk_indx = 0; chunk_indx < num_chunks; chunk_indx++) {
      // If we're nearing the end of the input, we may need to shift the
      // offset back so that we can get this_chunk_size frames of input to
      // the nnet.
    int32 offset = std::min(
        this_chunk_size, num_rows - chunk_indx * this_chunk_size);
    if (!pad_input && offset < min_chunk_size)
          continue;
    SubMatrix<BaseFloat> sub_features(
      features, chunk_indx * this_chunk_size, offset, 0, feat_dim);
    Vector<BaseFloat> xvector;
    tot_weight += offset;
    // Pad input if the offset is less than the minimum chunk size
    if (pad_input && offset < min_chunk_size) {
      Matrix<BaseFloat> padded_features(min_chunk_size, feat_dim);
      int32 left_context = (min_chunk_size - offset) / 2;
      int32 right_context = min_chunk_size - offset - left_context;
      for (int32 i = 0; i < left_context; i++) {
        padded_features.Row(i).CopyFromVec(sub_features.Row(0));
      }
      for (int32 i = 0; i < right_context; i++) {
        padded_features.Row(min_chunk_size - i - 1).CopyFromVec(sub_features.Row(offset - 1));
      }
      padded_features.Range(left_context, offset, 0, feat_dim).CopyFromMat(sub_features);
      ExtractEmbedding(padded_features, &xvector);
    } else {
      ExtractEmbedding(sub_features, &xvector);
    }
      xvector_avg.AddVec(offset, xvector);
  }
  xvector_avg.Scale(1.0 / tot_weight);
  xvector->CopyFromVec(xvector_avg);
  return true;
}
