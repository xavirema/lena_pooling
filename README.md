# LENA pooling layer

The LEarned top-N Average (LENA) pooling layer is implemented in this repository within the caffe deep learning framework. The list of modifications from the original repository follow:

- Adding the files "src/caffe/layers/lena_pooling_layer.cpp" and "include/caffe/layers/lena_pooling_layer.hpp". No CUDA implementation is available at the moment.
- Modifying the protobuffer definition to accomodate a lena_parameter message.
- Allow boolean blobs with 5 dimensions.

If you are using LENA in your research, refer to the CVPR'17 article:

Xavier Alameda-Pineda, Andrea Pilzer, Dan Xu, Nicu Sebe and Elisa Ricci, Viraliency: Pooling Local Virality, IEEE CVPR 2017.
