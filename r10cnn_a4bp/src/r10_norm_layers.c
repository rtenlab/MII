#include "r10_cnn.h"

float batch_normalization_mean_array[16] = {
-3.31087418e+01f,-5.84998970e+01f,-1.21775608e+01f,+1.31604815e+01f,-2.88093853e+01f,
-4.77273798e+00f,+6.35634804e+00f,-1.34289491e+00f,-6.07791615e+00f,+7.40470171e+00f,
-4.30557404e+01f,-6.81962433e+01f,+1.37304199e+00f,+3.69957085e+01f,+6.91182556e+01f,
-5.75702238e+00f,}; 

float batch_normalization_stdev_array[16] = {
+3.80857430e+01f,+4.71483688e+01f,+5.24992485e+01f,+4.42468262e+01f,+2.71128540e+01f,
+8.30912933e+01f,+3.86604729e+01f,+2.95142174e+01f,+6.54709549e+01f,+7.48006134e+01f,
+6.31727104e+01f,+4.40300484e+01f,+5.68284798e+01f,+5.51842155e+01f,+3.58419876e+01f,
+2.17528343e+01f,}; 

float batch_normalization_gamma_array[16] = {
+6.82150722e-01f,+4.61984903e-01f,+7.32033193e-01f,+5.75750113e-01f,+5.39374650e-01f,
+8.66111517e-01f,+6.24758005e-01f,+7.60886729e-01f,+7.49000490e-01f,+6.60216868e-01f,
+5.36291420e-01f,+7.06771016e-01f,+7.56188989e-01f,+2.89155543e-01f,+4.21146512e-01f,
+5.77335954e-01f,}; 

float batch_normalization_beta_array[16] = {
+1.55678242e-02f,-2.51077056e-01f,+5.25635779e-01f,+4.84393328e-01f,-7.09040910e-02f,
+1.26705438e-01f,+6.20124996e-01f,+5.60988247e-01f,+1.13421902e-01f,+3.69690061e-01f,
-2.20120147e-01f,-3.74866545e-01f,+7.61094868e-01f,+2.13900264e-02f,-1.27738789e-01f,
+1.29260600e+00f,}; 

// void k2c_batch_norm(k2c_tensor* outputs, const k2c_tensor* inputs, const k2c_tensor* mean,
//                     const k2c_tensor* stdev, const k2c_tensor* gamma, const k2c_tensor* beta,
//                     const size_t axis) {
// k2c_batch_norm(&batch_normalization_output,&conv2d_output,&batch_normalization_mean,
// &batch_normalization_stdev,&batch_normalization_gamma,&batch_normalization_beta,batch_normalization_axis); 

void r10_batch_norm(struct exe_config *config, struct r10cnn_layer *layer)
{
    r10_tensor *inputs = &layer->ifm;
    inputs->dataf = (float*)pvPortMalloc(inputs->num_data*sizeof(float));
    nvm_to_vm(inputs->dataf, inputs->num_data, inputs->nvm_start);
    r10_tensor *outputs = &layer->ofm;
    outputs->dataf = (float*)pvPortMalloc(outputs->num_data*sizeof(float));

    // <RTEN> compatibility ad hoc
    r10_tensor mean = 
    (r10_tensor){
        .dataf = batch_normalization_mean_array,
		.ndim = 1,
		.num_data = 16,
		.shape = {16, 1, 1, 1, 1}
	};
    r10_tensor stdev =
    (r10_tensor){
        .dataf = batch_normalization_stdev_array,
        .ndim = 1,
        .num_data = 16,
        .shape = {16, 1, 1, 1, 1}
    };
    r10_tensor gamma =
    (r10_tensor){
        .dataf = batch_normalization_gamma_array,
        .ndim = 1,
        .num_data = 16,
        .shape = {16, 1, 1, 1, 1}
    };
    r10_tensor beta =
    (r10_tensor){
        .dataf = batch_normalization_beta_array,
        .ndim = 1,
        .num_data = 16,
        .shape = {16, 1, 1, 1, 1}
    };
    size_t axis = 2;
    // </RTEN>

    size_t offset = 1;
    for (size_t i=axis+1; i<inputs->ndim; ++i) {
        offset *= inputs->shape[i];
    }
    const size_t step = inputs->shape[axis];

    for (size_t i=0; i<inputs->num_data; ++i) {
        size_t idx = (i/offset)%step;
        outputs->dataf[i] = (inputs->dataf[i] - mean.dataf[idx]) /
                            stdev.dataf[idx] *
                            gamma.dataf[idx] +
                            beta.dataf[idx];
    }
    if(vm_to_nvm(outputs->dataf, outputs->num_data, outputs->nvm_start, &outputs->nvm_end) != 0){
        am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
    }
    vPortFree(outputs->dataf);
    vPortFree(inputs->dataf);
    return;
}