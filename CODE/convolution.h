 void Horizontalconv_CPU(
    float *c_out,
    float *a_in,
    float *b_in,
    int image_w,
    int image_h,
    int half_filter
);

 void Verticalconv_CPU(
    float *c_out,
    float *a_in,
    float *b_in,
    int image_w,
    int image_h,
    int half_filter
);




void Horizontalconv_CPU(
    float *c_out,
    float *a_in,
    float *b_in,
    int image_w,
    int image_h,
    int half_filter
){
    for(int y = 0; y < image_h; y++)
        for(int x = 0; x < image_w; x++){
            double sum = 0;
            for(int k = -half_filter; k <= half_filter; k++){
                int d = x + k;
                if(d >= 0 && d < image_w)
                    sum += a_in[y * image_w + d] * b_in[half_filter - k];
            }
            c_out[y * image_w + x] = (float)sum;
        }
}

 void Verticalconv_CPU(
    float *c_out,
    float *a_in,
    float *b_in,
    int image_w,
    int image_h,
    int half_filter
){
    for(int y = 0; y < image_h; y++)
        for(int x = 0; x < image_w; x++){
            double sum = 0;
            for(int k = -half_filter; k <= half_filter; k++){
                int d = y + k;
                if(d >= 0 && d < image_h)
                    sum += a_in[d * image_w + x] * b_in[half_filter - k];
            }
            c_out[y * image_w + x] = (float)sum;
        }
}
