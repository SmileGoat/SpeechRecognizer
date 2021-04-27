#ifndef XVECTOR_CONTROLLER_IMPL_H_
#define XVECTOR_CONTROLLER_IMPL_H_

#include "speaker_verification/xvector_extractor.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "speaker_verification/plda.h"

typedef kaldi::BaseFloat BaseFloat;
typedef kaldi::int32 int32;
#define kEmptyEnroll -1

class XvectorControllerImpl {
  public:
    XvectorControllerImpl(const Vector<BaseFloat>& mean_xvector,
		                      const std::string& nnet,
		                      const Plda& plda,
                          const Matrix<BaseFloat>& transform);
    int32 EnrollSpeaker(const std::vector<Matrix<BaseFloat>>& features, int32 speaker_id);
    int32 EnrollSpeakerFromFeededFeature(int32 speaker_id);
    std::vector<BaseFloat> ComputeSpeakerConfidences(
		    const std::vector<Matrix<BaseFloat>>& features) const;
    bool FeedEnrollingSpeakerFeature(const Matrix<BaseFloat>& feature);
		// todo extract this two api
    bool WriteEnrolledFeature(const std::string& Wxfilename, 
		                          const bool is_binary = true) const;
    bool ReadEnrolledFeature(const std::string& Rxfilename);
  private:
    void XvectorPostTransform(Vector<BaseFloat>* xvector);
    void NormalizeLength(Vector<BaseFloat>* ivector);
    void MeanVectors(const std::vector<Vector<BaseFloat>>& vectors, 
		     Vector<BaseFloat>* mean_vector);
    void LDATransform(const Vector<BaseFloat>& input_ivector,
        Vector<BaseFloat>* transform_vector);
    void PLDATransform(Vector<BaseFloat>* ivector, int32 num_utt);
    bool Feature2Xvector(const Matrix<BaseFloat>& feature,
        Vector<BaseFloat>* xvector);
    int32 Xvectors2PldaInput(const std::vector<Vector<BaseFloat>>& xvectors,
        Vector<BaseFloat>* xvector_transform);
    void EnrollPldaInput(const Vector<BaseFloat>& enroll_xvector_transform);
    int32 Feature2PldaInput(const std::vector<Matrix<BaseFloat>>& features,
        Vector<BaseFloat>* xvector_transform);
    int32 EnrollPldaInput(const Vector<BaseFloat>& xvector_trans, int32 num_utt,
                         int32 speaker_id);

    const Plda plda_;
    const Matrix<BaseFloat> transform_;
    const Vector<BaseFloat> mean_;
    std::vector<Vector<double>> enrolled_speakers_;
    std::vector<int32> num_utts_;
    std::vector<Vector<BaseFloat>> feeded_xvector;
    std::unique_ptr<XvectorExtractor> xvector_extractor_; 
};

#endif //  XVECTOR_CONTROLLER_IMPL_H_

