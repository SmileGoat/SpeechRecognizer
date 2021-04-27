// Copyright (c) 2021 PeachLab. All Rights Reserved.
// Author : goat.zhou@qq.com (Yang Zhou)

#include "speaker_verification/speaker_verification.h"
#include "speaker_verification/speaker_verification_client.h"

#define kSpeakerId 0

bool SpeakerVerificationInitModel(const char* model_dir) {
  std::string model_path(model_dir);
  std::string nnet_file = model_path + "/final.raw";
  std::string plda_file = model_path + "/plda";
  std::string trans_file = model_path + "/transform.mat";
  std::string mean_file = model_path + "/mean.vec";
  std::string xvector_file = model_path + "/enroll.xvector";

  std::ifstream fin0(nnet_file);
  std::ifstream fin1(plda_file);
  std::ifstream fin2(trans_file);
  std::ifstream fin3(mean_file);

  if (!fin0.good() || !fin1.good() || !fin2.good() || !fin3.good()) {
    return false;
  }

  return SpeakerVerificationClient::GetInstance()->Init(nnet_file,
                                                        plda_file,
                                                        trans_file,
                                                        mean_file,
                                                        xvector_file);
}

bool FeedEnrollingSpeakerWave(char* wave_data, int length) {
  return SpeakerVerificationClient::GetInstance()->FeedEnrollingSpeakerWave(wave_data, length); 
}

bool ReadEnrolledXvector(const char* xvector_path) {
  std::string xvector_file(xvector_path);
  SpeakerVerificationClient::GetInstance()->ReadEnrolledXvector(xvector_file);
}

bool EnrollSpeakerFromHaveFeededFeatures() {
  return SpeakerVerificationClient::GetInstance()->EnrollSpeakerAndUpdateTemplate(kSpeakerId);
}

bool VerifySpeaker(char* wave_data, int length) {
  float score = SpeakerVerificationClient::GetInstance()->GetSpeakerConfidence(wave_data, length);
  bool is_speaker = (score >= SpeakerVerificationClient::GetInstance()->GetThreshold());
  return is_speaker;
}

float GetVersion() { //todo: replace this fake function
  std::cout << "verison check is zero" << std::endl;
  return 0;
}

bool HaveEnrolled() {
  return SpeakerVerificationClient::GetInstance()->HaveSpeakerEnrolled();
}

bool DestoryModel() {
  return SpeakerVerificationClient::GetInstance()->DestoryClient();
}

