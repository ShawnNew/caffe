#include <vector>

#include "caffe/layers/alpha_overall_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AlphaOverallLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    shape_img_.assign(
        bottom[0]->shape().begin()+2,
        bottom[0]->shape().end() );  // get shape of the image: width*height
}

template <typename Dtype>
void AlphaOverallLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);   // reshaped top shape here.
    CHECK_EQ(bottom[0]->count(2), bottom[1]->count(2))
      << "Inputs must have the same dimension.";
    diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void AlphaOverallLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
    // get mask and reshape here. 
    // mask is stored as class member mask_
    // the order of the second bottom should be tri-map, alpha, original, fg, bg.
    for (int i = 0; i < bottom[1]->shape()[0]; ++i) {  // over batch
        for (int j = 0; j < shape_img_[0]; ++j) {   // over width
            for (int k = 0; k < shape_img_[1]; ++k) { // over height
                if (bottom[1]->data_at(i, 0, j, k) == Dtype(0) || \
                    bottom[1]->data_at(i, 0, j, k) == Dtype(1)) {
                        mask_.push_back(Dtype(0));
                } else {
                    num_pixels++;
                    mask_.push_back(double(bottom[1]->data_at(i, 0, j, k)));
                }
                gt_.push_back(bottom[1]->data_at(i, 1, j, k));
            }
        }
    }
    // get original, fg and bg here.
    for (int i = 0; i < bottom[1]->shape()[0]; ++i) { // over batch
        for (int j = 0; j < 3; ++j) {                 // over channel
            for (int k = 0; k < shape_img_[0]; ++k) { // over width
                for (int t = 0; t < shape_img_[1]; ++t) {// over height
                    original_.push_back(bottom[1]->data_at(i, j+2, k, t))
                    fg_.push_back(bottom[1]->data_at(i, j+5, k, t));
                    bg_.push_back(bottom[1]->data_at(i, j+8, k ,t));
                }
            }
        }
    }   
    vector<Dtype> mul_(mask_.size());
    caffe_sub(
        count,
        bottom[0]->cpu_data(),
        gt_.data(),
        diff_.mutable_cpu_data()
    );
    caffe_mul(count, diff_.cpu_data(), mask_.data(), mul_.data());  // multiply by mask
    Dtype dot = caffe_cpu_dot(count, mul_.data(), mul_.data());            // square and sum
    Dtype loss = dot / num_pixels / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void AlphaOverallLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // grad = diff / length
    for (int i = 0; i < 2; ++i) {
        if (propagate_down[i]) {
            const Dtype sign = (i == 0) ? 1 : -1;
            const Dtype alpha = sign / bottom[0]->num();
            caffe_cpu_axpby(   // Y = alpha*X + beta*Y
                bottom[i]->count(),              // count
                alpha,                              // alpha
                diff_.cpu_data(),                   // a
                Dtype(0),                           // beta
                bottom[i]->mutable_cpu_diff());  // b
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(AlphaOverallLossLayer);
#endif

INSTANTIATE_CLASS(AlphaOverallLossLayer);
REGISTER_LAYER_CLASS(AlphaOverallLoss);

} // caffe namespace
