#ifndef XVECTOR_CONTROLLER_H_
#define XVECTOR_CONTROLLER_H_

#include "feat/frontend.h"
#include "speaker_verification/xvector_controller_impl.h"
#include "util/common-utils.h"
#include "speaker_verification/plda.h"

typedef kaldi::BaseFloat BaseFloat;
typedef kaldi::int32 int32;

namespace goat {

class XvectorController {
  public:
    XvectorController(
		  const std::string& mean_xvector_rxfilename,
      const std::string& nnet_rxfilename,
      const std::string& plda_rxfilename,
      const std::string& transform_rxfilename);
    int32 EnrollSpeaker(const std::vector<Matrix<BaseFloat>>& features, int32 speaker_id);
    int32 EnrollSpeakerFromFeededFeatures(int32 speaker_id);
    bool FeedEnrollingSpeakerFeature(const Matrix<BaseFloat>& feature);
    bool ReadEnrolledFeature(const std::string& Wxfilename);
    bool WriteEnrolledFeature(const std::string& Rxfilename);
    std::vector<BaseFloat> ComputeSpeakerConfidences(const std::vector<Matrix<BaseFloat>>& features);

  private:
    void MakeFeature(const std::vector<Matrix<BaseFloat>>& wave_features,
                  std::vector<Matrix<BaseFloat>>* frontend_features);
    std::unique_ptr<XvectorControllerImpl> xvector_controller_impl_; 
    std::unique_ptr<Frontend> xvector_frontend_; 
};

}  // namespace goat
#endif  // XVECTOR_CONTROLLER_H_

