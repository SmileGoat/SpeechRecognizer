// Copyright (c) 2021 PeachLab. All Rights Reserved.
// Author : goat.zhou@qq.com (Yang Zhou)

#include "speaker_verification/speaker_verification_client.h"
#include "speaker_verification/xvector_controller.h"

bool SpeakerVerificationClient::ReadEnrolledXvector(const std::string xvector_path) {
  xvector_controller_->ReadEnrolledFeature(xvector_path);
  have_enrolled_ = true;
  return true;
}

bool SpeakerVerificationClient::WriteEnrolledXvector(const std::string xvector_path) {
  return xvector_controller_->WriteEnrolledFeature(xvector_path);
}

SpeakerVerificationClient* SpeakerVerificationClient::GetInstance() {
  static SpeakerVerificationClient instance;
  return &instance;
}

bool SpeakerVerificationClient::HaveSpeakerEnrolled() {
  return have_enrolled_;
}

bool SpeakerVerificationClient::Init(
    std::string nnet_rxfilename,
    std::string plda_rxfilename,
    std::string transform_rxfilename,
    std::string mean_rxfilename,
    std::string xvector_file) {
  xvector_file_ = xvector_file;
  xvector_controller_.reset(new XvectorController(nnet_rxfilename,
                                                  plda_rxfilename,
                                                  transform_rxfilename,
                                                  mean_rxfilename));
	threshold_ = xvector_controller_.get_threshold();
  model_is_ready_ = true;
  std::ifstream fi(xvector_file);
  if (fi.good()) {
    ReadEnrolledXvector(xvector_file);
    have_enrolled_ = true;
		return true;
  }
  return false; 
}

// consider only one channel
void SpeakerVerificationClient::AcceptWaveData(short* wave_data, int len, kaldi::Matrix<BaseFloat>* data) {
  data->Resize(1, len); 
  for (int i = 0; i < len; ++i) {
      (*data)(0, i) = wave_data[i];
  }
}

// consider only one channel
void SpeakerVerificationClient::AcceptWaveData(char* wave_data, int len, kaldi::Matrix<BaseFloat>* data) {	
  len = len / 2;
  data->Resize(1, len);
  for (int i = 0; i < len; ++i) {
      (*data)(0, i) = *(((short *)wave_data) + i);
  }
}

int SpeakerVerificationClient::EnrollSpeaker(char* wave_data, int len) {
  if (!model_is_ready_) return -1;
  kaldi::Matrix<BaseFloat> data;
  AcceptWaveData(wave_data, len, &data);

  std::vector<kaldi::Matrix<BaseFloat>> enroll_speaker_data;
  enroll_speaker_data.push_back(data);
  int spk_id = xvector_controller_->EnrollSpeaker(enroll_speaker_data, 0);
  have_enrolled_ = (spk_id != -1);
  return spk_id;
}

bool SpeakerVerificationClient::FeedEnrollingSpeakerFeature(char* wave_data, int len) {
  if (!model_is_ready_) return false;
  kaldi::Matrix<BaseFloat> data;
  AcceptWaveData(wave_data, len, &data);
  return xvector_controller_->FeedEnrollingSpeakerFeature(data);
}

bool SpeakerVerificationClient::EnrollSpeakerAndUpdateTemplate(int speaker_id) {
  xvector_controller_->EnrollSpeakerFromStoredFeature(speaker_id);
  have_enrolled_ = true;
  return true;
}

bool SpeakerVerificationClient::VerifySpeaker(char* wave_data, int len) {
  float confidence = GetSpeakerConfidence(wave_data, len);
  return (confidence >= threshold_);
}

float SpeakerVerificationClient::GetSpeakerConfidence(char* wave_data, int len) {
  if (!model_is_ready_ || !have_enrolled_) return -1;
  kaldi::Matrix<BaseFloat> data;
  AcceptWaveData(wave_data, len, &data);
  std::vector<kaldi::Matrix<BaseFloat>> waves;
  waves.push_back(data);
  std::vector<BaseFloat> score = xvector_controller_->ComputeSpeakerConfidences(waves);
  if (score.size() == 0) { return threshold_ + 1; }
  return score[0];
}

bool SpeakerVerificationClient::DestoryClient() {
  xvector_controller_.reset();
  have_enrolled_ = false;
  model_is_ready_ = false;
  return true;
}

