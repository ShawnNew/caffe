#ifndef CAFFE_ALPHA_OVERALL_LOSS_LAYER_HPP_
#define CAFFE_ALPHA_OVERALL_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * 
 * Compute alpha overall loss for alpha matting problem. 
 * Basically it is kind of euclidean loss; however, it is devided by the number of pixels.
 *  
 */

template <typename Dtype>
class AlphaOverallLossLayer : public LossLayer<Dtype> {
  public:
    explicit AlphaOverallLossLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param), diff_() {}
    
    virtual inline const char* type() const { return "AlphaOverallLoss"; }

    virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    /**
     * Backpropagate the gradient to the first bottom blob (prediction).
     * Because only the first bottom blob is used to update the weights when training.
     * So there is no need to override the AllowForceBackward function.
     */
    virtual inline bool AllowForceBackward(const int bottom_index) const {
      return true;
    }
  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    
    Blob<Dtype> diff_;
    static const double epsilon = 1e-6;
    vector<int> shape_img_;
    vector<Dtype> mask_;
    vector<Dtype> original_;
    vector<Dtype> fg_;
    vector<Dtype> bg_;
    vector<Dtype> gt_;
    int num_pixels;
};
}  // namespace caffe

#endif