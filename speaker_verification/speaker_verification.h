// Copyright (c) 2021 PeachLab. All Rights Reserved.
// Author : goat.zhou@qq.com (Yang Zhou)

#ifndef SPEAKER_VERIFICATION_H_
#define SPEAKER_VERIFICATION_H_

#ifdef __cplusplus
extern "C" {
#endif

bool SpeakerVerificationInitModel(const char* model_dir);

bool FeedEnrollingSpeakerWave(char* wave_data, int length);

bool EnrollSpeakerFromHaveFeededFeatures();

bool HaveEnrolled();

bool VerifySpeaker(char* wave_data, int length); 

bool ReadEnrolledXvector(const char* xvector_path);

bool DestoryModel();

float GetVersion();

#ifdef __cplusplus
}
#endif
#endif // SPEAKER_VERIFICATOIN_H_

