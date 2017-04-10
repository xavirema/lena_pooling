#include <algorithm>
#include <cfloat>
#include <vector>
#include <queue>

#include "caffe/filler.hpp"
#include "caffe/layers/lena_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;
using std::priority_queue;

template <typename Dtype>
void LenaPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //std::cerr << "This is the new LenaPolling.cpp" << std::endl;
  LenaPoolingParameter pool_param = this->layer_param_.lena_pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  CHECK_LT(pad_h_, kernel_h_);
  CHECK_LT(pad_w_, kernel_w_);
  // Allocate
  vector<int> weight_shape(2);
  weight_shape[0] = bottom[0]->channels();
  weight_shape[1] = 1;
  this->blobs_.resize(1);
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  this->blobs_[0]->Reshape(weight_shape);
  // Fill the weights
  shared_ptr<Filler<Dtype> > mu_filler(GetFiller<Dtype>(
      this->layer_param_.lena_pooling_param().mu_filler()));
  mu_filler->Fill(this->blobs_[0].get());
  // Make them learnable
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // Inverse of delta
  inv_delta_ =  1./(kernel_h_*kernel_w_);
}

template <typename Dtype>
void LenaPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  int myShape[] = {bottom[0]->num(), channels_, pooled_height_, pooled_width_, kernel_h_*kernel_w_};  
  const vector<int> mult_idx_shape(myShape, myShape+sizeof(myShape)/sizeof(int));
  mult_idx_.Reshape(mult_idx_shape);
  // Auxiliar to backward
  diff_aux_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
}

template <typename Dtype>
void LenaPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    int* mult_mask = mult_idx_.mutable_cpu_data();
    Dtype* mu = this->blobs_[0]->mutable_cpu_data();
    Dtype* diff_aux_data = diff_aux_.mutable_cpu_data();
    // Limit mu[c] to [0,1]
    for(int c = 0; c < channels_; c++)
    {
        if( mu[c] < 0 ) mu[c] = 0;
        if( mu[c] > 1 ) mu[c] = 1;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n)
    {
      for (int c = 0; c < channels_; ++c)
      {
        int num_average = 1+ceil(mu[c]*(kernel_w_*kernel_h_-1));
        for (int ph = 0; ph < pooled_height_; ++ph)
        {
          for (int pw = 0; pw < pooled_width_; ++pw)
          {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            // Perhaps build the priority queue somewhere else I mean declare it and stuff....
            priority_queue< pair<Dtype,int> > sorted_pixels;
            for (int h = hstart; h < hend; ++h)
            {
              for (int w = wstart; w < wend; ++w)
              {
                const int index = h * width_ + w;
                sorted_pixels.push(pair<Dtype,int>(bottom_data[index],index));
	      }
            }
	    //std::cerr << std::endl;
            // Get the top num_average
            Dtype top_n_average = 0;
            for( int ii = 0; ii < num_average-1; ii++)
            {
                pair<Dtype,int> element = sorted_pixels.top();
		//std::cerr << element.first << " ";
                top_n_average += element.first;
                mult_mask[pool_index+ii] = element.second;
                sorted_pixels.pop();
            }
	    //std::cerr << std::endl;
            Dtype fm = 0;
            Dtype fp = 0;
            Dtype df = inv_delta_;
            if(num_average==1)
            {
                // Get top
                pair<Dtype,int> element = sorted_pixels.top();
                fm = element.first;
                top_n_average = fm;
                mult_mask[pool_index] = element.second;
                sorted_pixels.pop();
                // Get next one
                element = sorted_pixels.top();
                fp = (fm+element.first)/2;
                
            }else if(num_average==kernel_h_*kernel_w_)
            {
		//std::cerr << top_n_average << " ";
                // Save previous average into fm
                fm = top_n_average/(num_average-1);
		//std::cerr << fm << " ";
                // Get top
                pair<Dtype,int> element = sorted_pixels.top();
                top_n_average += element.first;
		//std::cerr << top_n_average << " ";
                mult_mask[pool_index+num_average-1] = element.second;
                // Update fp
                fp = top_n_average/num_average;
		//std::cerr << fp << std::endl;
                
            }else
            {
                // Save previous
                fm = top_n_average/(num_average-1);
                pair<Dtype,int> element = sorted_pixels.top();
                // Update top_n_average and mask
                top_n_average += element.first;
                mult_mask[pool_index+num_average-1] = element.second;
                sorted_pixels.pop();
                // Get data for fp
                element = sorted_pixels.top();
                fp = top_n_average + element.first;
                df /= 2;
            }
            // Save forward and prepare for backward
            top_data[pool_index] = top_n_average / num_average;
            diff_aux_data[pool_index] = (fp-fm)/df;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        diff_aux_data += diff_aux_.offset(0, 1);
        mult_mask += mult_idx_.offset(0, 1);
      }
    }
}

template <typename Dtype>
void LenaPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //std::cerr << !propagate_down[0] << std::endl;
  if (!propagate_down[0] && !this->param_propagate_down_[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
    int* mult_mask = mult_idx_.mutable_cpu_data();
    const Dtype* mu = this->blobs_[0]->cpu_data();
    Dtype *mu_diff = this->blobs_[0]->mutable_cpu_diff();
    for(int c = 0; c < channels_; c++)
    {
        mu_diff[c] = 0;
    }
    const Dtype* diff_aux_data = this->diff_aux_.cpu_data();
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        int num_average = 1+ceil(mu[c]*(kernel_w_*kernel_h_-1));
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int pool_index = ph * pooled_width_ + pw;
            for (int ii=0; ii < num_average; ii++) {
                bottom_diff[mult_mask[pool_index+ii]] += top_diff[pool_index] / num_average;
            }
            // Diff for mu
            mu_diff[c] += top_diff[pool_index] * diff_aux_data[pool_index];
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        mult_mask += mult_idx_.offset(0, 1);
        diff_aux_data += this->diff_aux_.offset(0, 1);
      }
    }
}


#ifdef CPU_ONLY
STUB_GPU(LenaPoolingLayer);
#endif

INSTANTIATE_CLASS(LenaPoolingLayer);
REGISTER_LAYER_CLASS(LenaPooling);

}  // namespace caffe
