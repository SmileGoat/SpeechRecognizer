// Copyright (c) 2021 PeachLab. All Rights Reserved.
// Author : goat.zhou@qq.com (Yang Zhou)

#ifndef SPEAKER_VERIFICATION_CLIENT_H_
#define SPEAKER_VERIFICATION_CLIENT_H_

#include "speaker_verification/xvector_controller.h"

class SpeakerVerificationClient {
 public:
  static SpeakerVerificationClient* GetInstance();
  bool Init(std::string nnet_rxfilename,
    std::string plda_rxfilename,
    std::string transform_rxfilename,
    std::string mean_rxfilename,
    std::string xvector_file);
  bool FeedEnrollingSpeakerWave(char* wave_data, int len);
  bool EnrollSpeakerAndUpdateTemplate(int speaker_id);
  int EnrollSpeaker(char* wave_data, int len);
  float GetSpeakerConfidence(char* wave_data, int len);
  bool VerifySpeaker(char* wave_data, int len);
  bool HaveSpeakerEnrolled();
  float GetThreshold() { return threshold_; }
  bool DestoryClient();
  bool ReadEnrolledXvector(const std::string xvector_path);

 private:
  void AcceptWaveData(short* wave_data, int len, kaldi::Matrix<BaseFloat>* data);
  void AcceptWaveData(char* wave_data, int len, kaldi::Matrix<BaseFloat>* data);
  bool WriteEnrolledXvector(const std::string xvector_path);

  std::unique_ptr<XvectorController> xvector_controller_;
  bool have_enrolled_  = false;
  bool model_is_ready_ = false;
  float threshold_;
  std::string xvector_file_;
};

#endif  // SPEAKER_VERIFICATION_CLIENT_H_

