#include <vector>

#include "caffe/layers/alpha_prediction_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AlphaPredictionLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
    // get mask and reshape here.
    // mask is stored as class member mask_
    for (int i = 0; i < bottom[1]->shape()[0]; ++i) {
        for (int j = 0; j < shape_img_[0]; ++j) {
            for (int k = 0; k < shape_img_[1]; ++k) {
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
    vector<Dtype> mul_(mask_.size());
    caffe_sub(
        count,
        bottom[0]->cpu_data(),
        gt_.data(),
        diff_.mutable_cpu_data()
    );
    caffe_mul(count, diff_.cpu_data(), mask_.data(), mul_.data());
    Dtype dot = caffe_cpu_dot(count, mul_.data(), mul_.data());
    Dtype loss = dot / num_pixels / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void AlphaPredictionLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& progagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < 2; ++i) {
        if (progagate_down[i]) {
            const Dtype sign = (i == 0) ? 1 : -1;
            const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
            caffe_gpu_axpby(
                bottom[i]->count(),
                alpha,
                diff_.gpu_data(),
                Dtype(0),
                bottom[i]->mutable_gpu_diff()
            );
        }
    }

}

INSTANTIATE_LAYER_GPU_FUNCS(AlphaPredictionLossLayer);
} // namespace caffe
