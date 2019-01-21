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
    for (int i = 0; i < bottm[1]->shape()[0]; ++i) {}
}
}
