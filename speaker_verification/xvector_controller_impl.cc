#include "speaker_verification/xvector_controller_impl.h"

using std::vector;

XvectorControllerImpl::XvectorControllerImpl(
    const Vector<BaseFloat>& mean_xvector,
    const std::string& nnet,
    const Plda& plda,
    const Matrix<BaseFloat>& transform)
     :mean_(mean_xvector), plda_(plda), transform_(transform) {
  xvector_extractor_.reset(new XvectorExtractor(nnet));
}

void XvectorControllerImpl::XvectorPostTransform(Vector<BaseFloat>* xvector) {
  xvector->AddVec(-1.0, mean_);
  Vector<BaseFloat> vec_out(transform_.NumRows());
  LDATransform(*xvector, &vec_out);
  NormalizeLength(&vec_out);
  xvector->Resize(vec_out.Dim());
  xvector->CopyFromVec(vec_out);
  return;
}

void XvectorControllerImpl::PLDATransform(Vector<BaseFloat>* ivector, int32 num_utt) {
  PldaConfig plda_config;
  Vector<BaseFloat> transform_vector;
  transform_vector.Resize(ivector->Dim());
  plda_.TransformIvector(plda_config, *ivector, num_utt, &transform_vector);
  ivector->Resize(transform_vector.Dim());
  ivector->CopyFromVec(transform_vector);
  return;
}

bool XvectorControllerImpl::Feature2Xvector(const Matrix<BaseFloat>& feature,
    Vector<BaseFloat>* xvector) {
    return xvector_extractor_->ExtractXvector(feature, xvector);
}

bool XvectorControllerImpl::FeedEnrollingSpeakerFeature(const Matrix<BaseFloat>& feature) {
  Vector<BaseFloat> xvector;
  // if (feature.NumCols() == 0) return false;
  bool result = false;
  result = xvector_extractor_->ExtractXvector(feature, &xvector);
  if (!result) return result;
  feeded_xvector.push_back(xvector);
  return true;
}

int32 XvectorControllerImpl::Feature2PldaInput(const vector<Matrix<BaseFloat>>& features, 
    Vector<BaseFloat>* xvector_transform) {
  vector<Vector<BaseFloat>> xvectors;
  xvectors.reserve(features.size());
  for (int32 i = 0; i < features.size(); ++i) {
    Vector<BaseFloat> xvector;
    bool result = xvector_extractor_->ExtractXvector(features[i], &xvector);
    if (result) xvectors.push_back(xvector);
  }
  int32 num_utt = xvectors.size();
  Xvectors2PldaInput(xvectors, xvector_transform);
  return num_utt;
}

int32 XvectorControllerImpl::Xvectors2PldaInput(
    const vector<Vector<BaseFloat>>& xvectors,
    Vector<BaseFloat>* xvector_transform) {
  Vector<BaseFloat> xvector_mean;
  int32 num_utt = xvectors.size();
  if (num_utt == 0) return num_utt;
  MeanVectors(xvectors, &xvector_mean);

  XvectorPostTransform(&xvector_mean);
  PLDATransform(&xvector_mean, num_utt);
  xvector_transform->Resize(xvector_mean.Dim());
  xvector_transform->CopyFromVec(xvector_mean);
  return num_utt;
}

bool XvectorControllerImpl::WriteEnrolledFeature(const std::string& Wxfilename,
    bool is_binary = true) {
  Output output;
  bool write_kaldi_header = true;
  if (!output.Open(Wxfilename, is_binary, write_kaldi_header)) {
    return false;
  }
  std::ostream& os = output.Stream();
  int enrolled_size = enrolled_speakers_.size();
  WriteToken(os, is_binary, "<num_enrolled_features>");
  WriteBasicType(os, is_binary, enrolled_size);
  WriteToken(os, is_binary, "</num_enrolled_features>");

  WriteToken(os, is_binary, "<enrolled_features>");
  for (int idx = 0; idx < enrolled_size; ++idx) {
    enrolled_speakers_[idx].Write(os, is_binary);
  }
  WriteToken(os, is_binary, "</enrolled_features>");

  WriteToken(os, is_binary, "<num_utts>");
  for (int idx = 0; idx < enrolled_size; ++idx) {
    WriteBasicType(os, is_binary, num_utts_[idx]); 
  }
  WriteToken(os, is_binary, "</num_utts>");
  output.Close();

  return true;
}

bool XvectorControllerImpl::ReadEnrolledFeature(const std::string& Rxfilename) {
  bool binary_in;
  Input input;
  if (! input.Open(Rxfilename, &binary_in)) {
    return false;
  }
  std::istream& is = input.Stream();
  int enrolled_size = 0;

  ExpectToken(is, binary_in, "<num_enrolled_features>");
  ReadBasicType(is, binary_in, &enrolled_size);
  ExpectToken(is, binary_in, "</num_enrolled_features>");

  ExpectToken(is, binary_in, "<enrolled_features>");
  enrolled_speakers_.reserve(enrolled_size);
  for (int idx = 0; idx < enrolled_size; ++idx) {
    Vector<double> enrolled_feature;
    enrolled_feature.Read(is, binary_in);
    enrolled_speakers_.push_back(enrolled_feature);
  }
  ExpectToken(is, binary_in, "</enrolled_features>");

  ExpectToken(is, binary_in, "<num_utts>");
  num_utts_.resize(enrolled_size);
  for (int idx = 0; idx < enrolled_size; ++idx) {
    ReadBasicType(is, binary_in, &(num_utts_[idx])); 
  } 
  ExpectToken(is, binary_in, "</num_utts>");

  input.Close();
  return true;
}

int32 XvectorControllerImpl::EnrollPldaInput(const Vector<BaseFloat>& enroll_xvector_transform,
                                            int32 num_utt,
                                            int32 speaker_id) {
  Vector<double> enroll_xvector_transform_dbl(enroll_xvector_transform);
  if (enrolled_speakers_.size() == 0 || speaker_id >= enrolled_speakers_.size()) {
    enrolled_speakers_.push_back(enroll_xvector_transform_dbl);
    num_utts_.push_back(num_utt);
    speaker_id = enrolled_speakers_.size() - 1;
  } else {
     enrolled_speakers_[speaker_id] = enroll_xvector_transform_dbl;
     num_utts_[speaker_id] = num_utt;
  }
  return speaker_id;
}

int32 XvectorControllerImpl::EnrollSpeaker(
    const vector<Matrix<BaseFloat>>& features,
    int32 speaker_id) {
  int32 num_utt = 0;
  Vector<BaseFloat> enroll_xvector_transform;
  num_utt = Feature2PldaInput(features, &enroll_xvector_transform);
  if (num_utt == 0) return -1;
  speaker_id = EnrollPldaInput(enroll_xvector_transform, num_utt, speaker_id);
  return speaker_id;
}

int32 XvectorControllerImpl::EnrollSpeakerFromFeededFeature(int32 speaker_id) {
  if (feeded_xvector.empty()) return kEmptyEnroll;
  Vector<BaseFloat> enroll_xvector_transform;
  int num_utt = Xvectors2PldaInput(feeded_xvector, &enroll_xvector_transform);
  speaker_id = EnrollPldaInput(enroll_xvector_transform, num_utt, speaker_id);
  feeded_xvector.clear();
  return speaker_id;
}

vector<BaseFloat> XvectorControllerImpl::ComputeSpeakerConfidences(
    const vector<Matrix<BaseFloat>>& features) {
  Vector<BaseFloat> test_xvector_tranform;
  vector<BaseFloat> scores;
  scores.reserve(features.size()); 
  int32 numutt = Feature2PldaInput(features, &test_xvector_tranform);
  if (numutt == 0) return scores;
  Vector<double> test_xvector_dbl(test_xvector_tranform);
  for (int32 spk_idx = 0; spk_idx < enrolled_speakers_.size(); ++spk_idx) {
    BaseFloat score = plda_.LogLikelihoodRatio(enrolled_speakers_[spk_idx],
                                               num_utts_[spk_idx],
                                               test_xvector_dbl);
    scores.push_back(score);
  }
  return scores; 
}

void XvectorControllerImpl::MeanVectors(const vector<Vector<BaseFloat>>& vectors,
    Vector<BaseFloat>* mean_vector) {
  if (vectors.empty()) return; 
  mean_vector->Resize(vectors[0].Dim());
  for (int32 i = 0; i < vectors.size(); ++i) {
    mean_vector->AddVec(1.0, vectors[i]);
  }
  mean_vector->Scale(1.0 / vectors.size());
}

void XvectorControllerImpl::NormalizeLength(Vector<BaseFloat>* vector) {
  BaseFloat norm = vector->Norm(2.0);
  BaseFloat ratio = norm / sqrt(ivector->Dim());
  if (ratio == 0.0) {
    KALDI_WARN << "Zero Vector";
  } else {
    vector->Scale(1.0 / ratio);
  }
}

void XvectorControllerImpl::LDATransform(const Vector<BaseFloat>& input_ivector,
  Vector<BaseFloat>* transform_vector) {
  transform_vector->Resize(transform_.NumRows());
  int32 vec_dim = input_ivector.Dim();
  int32 transform_cols = transform_.NumCols();
  if (transform_cols == vec_dim) {
    transform_vector->AddMatVec(1.0, transform_, kNoTrans, input_ivector, 0.0);
  } else {
  if (transform_cols != vec_dim + 1) {
    KALDI_ERR << "Dimension mismatch: input vector has dimension "
              << input_ivector.Dim() << " and transform has " << transform_cols
              << " columns.";
  }
  transform_vector->CopyColFromMat(transform_, vec_dim);
  transform_vector->AddMatVec(1.0,
                              transform_.Range(0, transform_.NumRows(), 0, vec_dim),
                              kNoTrans, input_ivector, 1.0);
  }
}

