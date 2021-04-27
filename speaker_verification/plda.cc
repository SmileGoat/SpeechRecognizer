// ivector/plda.cc

// Copyright 2013     Daniel Povey
//           2015     David Snyder

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include "speaker_verification/plda.h"

namespace kaldi {

void Plda::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<Plda>");
  mean_.Write(os, binary);
  transform_.Write(os, binary);
  psi_.Write(os, binary);
  WriteToken(os, binary, "</Plda>");
}

void Plda::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<Plda>");
  mean_.Read(is, binary);
  transform_.Read(is, binary);
  psi_.Read(is, binary);
  ExpectToken(is, binary, "</Plda>");
  ComputeDerivedVars();
}

template<class Real>
/// This function computes a projection matrix that when applied makes the
/// covariance unit (i.e. all 1).
static void ComputeNormalizingTransform(const SpMatrix<Real> &covar,
                                        MatrixBase<Real> *proj) {
  int32 dim = covar.NumRows();
  TpMatrix<Real> C(dim);  // Cholesky of covar, covar = C C^T
  C.Cholesky(covar);
  C.Invert();  // The matrix that makes covar unit is C^{-1}, because
               // C^{-1} covar C^{-T} = C^{-1} C C^T C^{-T} = I.
  proj->CopyFromTp(C, kNoTrans);  // set "proj" to C^{-1}.
}


void Plda::ComputeDerivedVars() {
  KALDI_ASSERT(Dim() > 0);
  offset_.Resize(Dim());
  offset_.AddMatVec(-1.0, transform_, kNoTrans, mean_, 0.0);
}


/**
   This comment explains the thinking behind the function LogLikelihoodRatio.
   The reference is "Probabilistic Linear Discriminant Analysis" by
   Sergey Ioffe, ECCV 2006.

   I'm looking at the un-numbered equation between eqs. (4) and (5),
   that says
     P(u^p | u^g_{1...n}) =  N (u^p | \frac{n \Psi}{n \Psi + I} \bar{u}^g, I + \frac{\Psi}{n\Psi + I})

   Here, the superscript ^p refers to the "probe" example (e.g. the example
   to be classified), and u^g_1 is the first "gallery" example, i.e. the first
   training example of that class.  \psi is the between-class covariance
   matrix, assumed to be diagonalized, and I can be interpreted as the within-class
   covariance matrix which we have made unit.

   We want the likelihood ratio P(u^p | u^g_{1..n}) / P(u^p), where the
   numerator is the probability of u^p given that it's in that class, and the
   denominator is the probability of u^p with no class assumption at all
   (e.g. in its own class).

   The expression above even works for n = 0 (e.g. the denominator of the likelihood
   ratio), where it gives us
     P(u^p) = N(u^p | 0, I + \Psi)
   i.e. it's distributed with zero mean and covarance (within + between).
   The likelihood ratio we want is:
      N(u^p | \frac{n \Psi}{n \Psi + I} \bar{u}^g, I + \frac{\Psi}{n \Psi + I}) /
      N(u^p | 0, I + \Psi)
   where \bar{u}^g is the mean of the "gallery examples"; and we can expand the
   log likelihood ratio as
     - 0.5 [ (u^p - m) (I + \Psi/(n \Psi + I))^{-1} (u^p - m)  +  logdet(I + \Psi/(n \Psi + I)) ]
     + 0.5 [u^p (I + \Psi) u^p  +  logdet(I + \Psi) ]
   where m = (n \Psi)/(n \Psi + I) \bar{u}^g.

 */

double Plda::GetNormalizationFactor(
    const VectorBase<double> &transformed_ivector,
    int32 num_examples) const {
  KALDI_ASSERT(num_examples > 0);
  // Work out the normalization factor.  The covariance for an average over
  // "num_examples" training iVectors equals \Psi + I/num_examples.
  Vector<double> transformed_ivector_sq(transformed_ivector);
  transformed_ivector_sq.ApplyPow(2.0);
  // inv_covar will equal 1.0 / (\Psi + I/num_examples).
  Vector<double> inv_covar(psi_);
  inv_covar.Add(1.0 / num_examples);
  inv_covar.InvertElements();
  // "transformed_ivector" should have covariance (\Psi + I/num_examples), i.e.
  // within-class/num_examples plus between-class covariance.  So
  // transformed_ivector_sq . (I/num_examples + \Psi)^{-1} should be equal to
  //  the dimension.
  double dot_prod = VecVec(inv_covar, transformed_ivector_sq);
  return sqrt(Dim() / dot_prod);
}


double Plda::TransformIvector(const PldaConfig &config,
                              const VectorBase<double> &ivector,
                              int32 num_examples,
                              VectorBase<double> *transformed_ivector) const {
  KALDI_ASSERT(ivector.Dim() == Dim() && transformed_ivector->Dim() == Dim());
  double normalization_factor;
  transformed_ivector->CopyFromVec(offset_);
  transformed_ivector->AddMatVec(1.0, transform_, kNoTrans, ivector, 1.0);
  if (config.simple_length_norm)
    normalization_factor = sqrt(transformed_ivector->Dim())
      / transformed_ivector->Norm(2.0);
  else
    normalization_factor = GetNormalizationFactor(*transformed_ivector,
                                                  num_examples);
  if (config.normalize_length)
    transformed_ivector->Scale(normalization_factor);
  return normalization_factor;
}

// "float" version of TransformIvector.
float Plda::TransformIvector(const PldaConfig &config,
                             const VectorBase<float> &ivector,
                             int32 num_examples,
                             VectorBase<float> *transformed_ivector) const {
  Vector<double> tmp(ivector), tmp_out(ivector.Dim());
  float ans = TransformIvector(config, tmp, num_examples, &tmp_out);
  transformed_ivector->CopyFromVec(tmp_out);
  return ans;
}


// There is an extended comment within this file, referencing a paper by
// Ioffe, that may clarify what this function is doing.
double Plda::LogLikelihoodRatio(
    const VectorBase<double> &transformed_train_ivector,
    int32 n, // number of training utterances.
    const VectorBase<double> &transformed_test_ivector) const {
  int32 dim = Dim();
  double loglike_given_class, loglike_without_class;
  { // work out loglike_given_class.
    // "mean" will be the mean of the distribution if it comes from the
    // training example.  The mean is \frac{n \Psi}{n \Psi + I} \bar{u}^g
    // "variance" will be the variance of that distribution, equal to
    // I + \frac{\Psi}{n\Psi + I}.
    Vector<double> mean(dim, kUndefined);
    Vector<double> variance(dim, kUndefined);
    for (int32 i = 0; i < dim; i++) {
      mean(i) = n * psi_(i) / (n * psi_(i) + 1.0)
        * transformed_train_ivector(i);
      variance(i) = 1.0 + psi_(i) / (n * psi_(i) + 1.0);
    }
    double logdet = variance.SumLog();
    Vector<double> sqdiff(transformed_test_ivector);
    sqdiff.AddVec(-1.0, mean);
    sqdiff.ApplyPow(2.0);
    variance.InvertElements();
    loglike_given_class = -0.5 * (logdet + M_LOG_2PI * dim +
                                  VecVec(sqdiff, variance));
  }
  { // work out loglike_without_class.  Here the mean is zero and the variance
    // is I + \Psi.
    Vector<double> sqdiff(transformed_test_ivector); // there is no offset.
    sqdiff.ApplyPow(2.0);
    Vector<double> variance(psi_);
    variance.Add(1.0); // I + \Psi.
    double logdet = variance.SumLog();
    variance.InvertElements();
    loglike_without_class = -0.5 * (logdet + M_LOG_2PI * dim +
                                    VecVec(sqdiff, variance));
  }
  double loglike_ratio = loglike_given_class - loglike_without_class;
  return loglike_ratio;
}


void Plda::SmoothWithinClassCovariance(double smoothing_factor) {
  KALDI_ASSERT(smoothing_factor >= 0.0 && smoothing_factor <= 1.0);
  // smoothing_factor > 1.0 is possible but wouldn't really make sense.

  KALDI_LOG << "Smoothing within-class covariance by " << smoothing_factor
            << ", Psi is initially: " << psi_;
  Vector<double> within_class_covar(Dim());
  within_class_covar.Set(1.0); // It's now the current within-class covariance
                               // (a diagonal matrix) in the space transformed
                               // by transform_.
  within_class_covar.AddVec(smoothing_factor, psi_);
  /// We now revise our estimate of the within-class covariance to this
  /// larger value.  This means that the transform has to change to as
  /// to make this new, larger covariance unit.  And our between-class
  /// covariance in this space is now less.

  psi_.DivElements(within_class_covar);
  KALDI_LOG << "New value of Psi is " << psi_;

  within_class_covar.ApplyPow(-0.5);
  transform_.MulRowsVec(within_class_covar);

  ComputeDerivedVars();
}

void Plda::ApplyTransform(const Matrix<double> &in_transform) {
  KALDI_ASSERT(in_transform.NumRows() <= Dim()
    && in_transform.NumCols() == Dim());

  // Apply in_transform to mean_.
  Vector<double> mean_new(in_transform.NumRows());
  mean_new.AddMatVec(1.0, in_transform, kNoTrans, mean_, 0.0);
  mean_.Resize(in_transform.NumRows());
  mean_.CopyFromVec(mean_new);

  SpMatrix<double> between_var(in_transform.NumCols()),
                   within_var(in_transform.NumCols()),
                   psi_mat(in_transform.NumCols()),
                   between_var_new(Dim()),
                   within_var_new(Dim());
  Matrix<double> transform_invert(transform_);

  // Next, compute the between_var and within_var that existed
  // prior to diagonalization.
  psi_mat.AddDiagVec(1.0, psi_);
  transform_invert.Invert();
  within_var.AddMat2(1.0, transform_invert, kNoTrans, 0.0);
  between_var.AddMat2Sp(1.0, transform_invert, kNoTrans, psi_mat, 0.0);

  // Next, transform the variances using the input transformation.
  between_var_new.AddMat2Sp(1.0, in_transform, kNoTrans, between_var, 0.0);
  within_var_new.AddMat2Sp(1.0, in_transform, kNoTrans, within_var, 0.0);

  // Finally, we need to recompute psi_ and transform_. The remainder of
  // the code in this function  is a lightly modified copy of
  // PldaEstimator::GetOutput().
  Matrix<double> transform1(Dim(), Dim());
  ComputeNormalizingTransform(within_var_new, &transform1);
  // Now transform is a matrix that if we project with it,
  // within_var becomes unit.
  // between_var_proj is between_var after projecting with transform1.
  SpMatrix<double> between_var_proj(Dim());
  between_var_proj.AddMat2Sp(1.0, transform1, kNoTrans, between_var_new, 0.0);

  Matrix<double> U(Dim(), Dim());
  Vector<double> s(Dim());
  // Do symmetric eigenvalue decomposition between_var_proj = U diag(s) U^T,
  // where U is orthogonal.
  between_var_proj.Eig(&s, &U);

  KALDI_ASSERT(s.Min() >= 0.0);
  int32 n;
  s.ApplyFloor(0.0, &n);
  if (n > 0) {
    KALDI_WARN << "Floored " << n << " eigenvalues of between-class "
               << "variance to zero.";
  }
  // Sort from greatest to smallest eigenvalue.
  SortSvd(&s, &U);

  // The transform U^T will make between_var_proj diagonal with value s
  // (i.e. U^T U diag(s) U U^T = diag(s)).  The final transform that
  // makes within_var unit and between_var diagonal is U^T transform1,
  // i.e. first transform1 and then U^T.
  transform_.Resize(Dim(), Dim());
  transform_.AddMatMat(1.0, U, kTrans, transform1, kNoTrans, 0.0);
  psi_.Resize(Dim());
  psi_.CopyFromVec(s);
  ComputeDerivedVars();
}

} // namespace kaldi


