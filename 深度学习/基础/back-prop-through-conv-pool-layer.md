参考：
https://blog.csdn.net/Candy_GL/article/details/79470804
https://www.cnblogs.com/pinard/p/6494810.html
https://www.slideshare.net/mobile/kuwajima/cnnbp

这几篇文章写的不错，但总感觉让人迷惑。前两篇文章的描述中已知了池化层的误差$\delta ^l$求池化层前面一层的误差，按照我的理解
$$\delta ^{l-1}=\frac{\partial J}{\partial z^{l-1}}=\frac{\partial J}{\partial z^l}\frac{\partial z^l}{\partial z^{l-1}}=\delta ^l\frac{\partial z^l}{\partial z^{l-1}}$$
这个求解过程已经和下采样上采样没有关系了。
但是按照上面参考文章的说法，$z^l$是对$a^{l-1}$下采样的结果，那就说明$l$层误差的计算没有经过下采样，也就是没有经过池化层，误差$\delta ^l$也是池化层之后层的误差，真正的池化层也在$l-1$层（可能是卷积层+池化层）。这样理解的话就说得通了。
此时，
$$\delta ^{l-1}=\delta ^l\frac{\partial z^l}{\partial a^{l-1}}\frac{\partial a^{l-1}}{\partial z^{l-1}}=upsample(\delta ^l) \cdot  \sigma'(z^{l-1})$$
至于$ \sigma'(z^{l-1})$具体是什么就和$l-1$层的类型有关了,$\sigma 表示激活函数$。

对于卷积层的推导也一样，实际的卷积层也在$l-1$层(这里求的就是卷积层)
$$\delta ^{l-1} = \frac{\partial J}{\partial z^{l}}\frac{\partial z^l}{\partial z^{l-1}}=\delta ^{l}\frac{\partial a^{l-1}}{\partial z^{l-1}}$$
如果是全连接层，这里的 $a^{l-1}=sigmoid(z^{l-1})$（这里假设激活函数是sigmoid）,但是这里是卷积层，$a^{l-1}$是$z^{l-1}$经过卷积又经过激活函数后的结果，就是$\sigma(z^{l-1}*w^{l-1}+b)$，所以这里既要求激活函数的导数有要求卷积操作的导数。最后结果就是
$$\delta ^{l-1}=\delta ^l *rot180(w^{l-1})\cdot \sigma '({z^{l-1}})$$
具体形式的推导参考上面第二篇文章。
由上式可以看出卷积层的反向传播由卷积操作和点乘组成，没有新的运算类型。
对比Caffe base_conv_layer.cpp中的forward和backward代码发现，反向传播时确实是一个转置的卷积操作
```
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}
```
```
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}
```
```
template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}
```