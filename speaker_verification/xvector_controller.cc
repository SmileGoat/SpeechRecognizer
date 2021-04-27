#include "speaker_verification/xvector_controller.h"
#include "speaker_verification/xvector_controller_impl.h"

using std::vector;
using std::string;

XvectorController::XvectorController(
    const string& mean_xvector_rxfilename,
    const string& nnet_rxfilename,
    const string& plda_rxfilename,
    const string& transform_rxfilename) {
  KALDI_LOG << "the file have get: "<< nnet_rxfilename << " " << mean_xvector_rxfilename;
  Vector<BaseFloat> mean;
  ReadKaldiObject(mean_xvector_rxfilename, &mean);
  Matrix<BaseFloat> transform;
  ReadKaldiObject(transform_rxfilename, &transform);
  KALDI_LOG << "mean & transform init";
  Plda plda;
  ReadKaldiObject(plda_rxfilename, &plda);

  KALDI_LOG << "nnet & plad init";
  xvector_controller_impl_.reset(
    new XvectorControllerImpl(nnet_rxfilename, plda, transform, mean));
  xvector_frontend_.reset(
    new XvectorFrontend(vad_opts, sliding_window_cmn_opts, mfcc_opts));
}

void XvectorController::MakeFeature(
    const std::vector<Matrix<BaseFloat>>& wave_features,
    std::vector<Matrix<BaseFloat>>* frontend_features) {
  frontend_features->reserve(wave_features.size());
  for (int32 i = 0; i < wave_features.size(); ++i) {
    Matrix<BaseFloat> result_feature;
    xvector_frontend_->ComputeFeatures(wave_features[i], &result_feature);
    frontend_features->push_back(result_feature);
  }
}

int32 XvectorController::EnrollSpeaker(const vector<Matrix<BaseFloat>>& features,
    int32 speaker_id) {
  vector<Matrix<BaseFloat>> enroll_mfcc_features;
  MakeMFCC(features, &enroll_mfcc_features);
  return xvector_controller_impl_->EnrollSpeaker(enroll_mfcc_features, 
                                                 speaker_id);
}

bool XvectorController::FeedEnrollingSpeakerFeature(const Matrix<BaseFloat>& feature) {
  Matrix<BaseFloat> mfcc_feature;
  xvector_frontend_->ComputeFeatures(feature, &mfcc_feature);
  return xvector_controller_impl_->FeedEnrollingSpeakerFeature(mfcc_feature);
}

vector<BaseFloat> XvectorController::ComputeSpeakerConfidences(
    const vector<Matrix<BaseFloat>>& features) {
  vector<Matrix<BaseFloat>> test_mfcc_features;
  MakeMFCC(features, &test_mfcc_features);
  return xvector_controller_impl_->ComputeSpeakerConfidences(test_mfcc_features);
}

int32 XvectorController::EnrollSpeakerFromStoredFeature(int32 speaker_id) {
  return xvector_controller_impl_->EnrollSpeakerFromStoredFeature(speaker_id);
}

bool XvectorController::WriteEnrolledFeature(const std::string& Wxfilename) {
  return xvector_controller_impl_->WriteEnrolledFeature(Wxfilename);
}

bool XvectorController::ReadEnrolledFeature(const std::string& Rxfilename) {
  return xvector_controller_impl_->ReadEnrolledFeature(Rxfilename);
}

