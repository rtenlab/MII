#include "r10_cnn.h"

size_t x0, x1, z0, z1, q, k;
size_t r, c, n, m, k0, k1;
size_t tr, tc, tn, tm;

extern float adc_voltage; // a4p_hardware.c
extern int32_t counter; // r10_cnn.c

#if defined(UART_PROFILE)
extern TickType_t begin, elapse;
#endif

/**
 * ReLU activation function.
 *   y = max(x,0) *
 * :param x: array of input values. Gets overwritten by output.
 * :param size: length of input array.
 */
void r10_relu_func(r10_tensor *r10tensor) {

    const size_t size = r10tensor->num_data;

    for (size_t i=0; i < size; ++i) {
        if (r10tensor->dataf[i] <= 0.0f) {
            r10tensor->dataf[i] = 0.0f;
        }
    }

    return;
    
}

void r10_relu(struct exe_config *config, struct r10cnn_layer *layer) {
    layer->ofm.dataf = (float*)pvPortMalloc(layer->ofm.num_data*sizeof(float));
    memset(layer->ofm.dataf,0, layer->ofm.num_data*sizeof(layer->ofm.dataf[0]));
    r10_relu_func(&layer->ofm);
    vPortFree(layer->ifm.dataf);
    return;
    
}

void float_relu_func(float *x, size_t size) {
    for (size_t i=0; i < size; ++i) {
        if (x[i] <= 0.0f) {
            x[i] = 0.0f;
        }
    }
}

/**
 * Adds bias vector b to tensor A.
 * assumes b is a rank 1 tensor that is added to the last dimension of A.
 *
 * @A: input tensor. Overwritten with outputs.
 * @b: bias tensor.
 */
void float_bias_add(float *A, float *b, size_t size) 
{
    for (size_t i=0; i < size; i+=10) {
        for (size_t j=0; j < 10; ++j) {
            A[i+j] += b[j];
        }
    }

    return;
}

void _conv2d_vanilla_xip (size_t layer_id, exe_config *config,
    const size_t stride[2], const size_t dilation[2],
    r10_tensor* weights, const r10_tensor* bias, 
    r10_tensor* ifm, r10_tensor* ofm)
{

    const size_t out_rows = ofm->shape[0];
    const size_t out_cols = ofm->shape[1];
    const size_t out_channels = ofm->shape[2];
    const size_t in_channels = ifm->shape[2];

    // size_t x0, x1, z0, z1, q, k;
    size_t weights_ix, ifm_ix;

    // am_util_stdio_printf("HERE! Layer: %ld\n", config->exe_status[2]);

    ofm->dataf = (float*)pvPortMalloc(ofm->num_data*sizeof(float));
    memset(ofm->dataf,0,ofm->num_data*sizeof(ofm->dataf[0]));


    for (x0 = config->exe_status[3]; x0 < out_rows; ++x0) {
        for (x1=config->exe_status[4]; x1 < out_cols; ++x1) {
            for (z0=config->exe_status[5]; z0 < weights->shape[0]; ++z0) {
                for (z1=config->exe_status[6]; z1 < weights->shape[1]; ++z1) {
                    for (q=config->exe_status[7]; q < in_channels; ++q) {
                        for (k=config->exe_status[8]; k < out_channels; ++k) {
                            // weights_ix
                            weights_ix = z0*(weights->shape[3]*weights->shape[2]*weights->shape[1])
                                        + z1*(weights->shape[3]*weights->shape[2])
                                        + q*(weights->shape[3]) + k;
                            // ifm_ix
                            ifm_ix = (x0*stride[0] + dilation[0]*z0)*(ifm->shape[2]*ifm->shape[1])
                                        + (x1*stride[1] + dilation[1]*z1)*(ifm->shape[2]) + q;

                            ofm->dataf[x0*(ofm->shape[2]*ofm->shape[1]) + x1*(ofm->shape[2]) + k] 
                            +=
                            ui32_to_fp32(*(uint32_t*)(weights->nvm_start + (weights_ix*DATA_SIZE))) // weights[ix]
                            *
                            ui32_to_fp32(*(uint32_t*)(ifm->nvm_start + (ifm_ix*DATA_SIZE))); // ifm[ix]
                            
                            // counter++; // REMOVE
                            // am_util_stdio_printf("Counter: %d\n", counter); // REMOVE
                        }
                    }
                }
            }
        }
    }

    r10_relu_func(ofm);
    r10_bias_add(ofm,bias);

    // Free the layer ifm data
    if (layer_id > 0){
        vPortFree(ifm->dataf);
    }
    // vPortFree(weights->dataf);
    // vPortFree(ifm->dataf);
    
    
    return;

}

void _conv2d_tiled_xip (size_t layer_id, exe_config *config,
    const size_t stride[2], const size_t dilation[2],
    tiled_param* t_param,
    const r10_tensor* weights, const r10_tensor* bias, 
    const r10_tensor* ifm, r10_tensor* ofm)
{
    const size_t out_rows = ofm->shape[0]; // R
    const size_t out_cols = ofm->shape[1]; // C
    const size_t out_channels = ofm->shape[2]; // M
    const size_t in_channels = ifm->shape[2]; // N

    const size_t Tr = t_param->shape[0];
    const size_t Tc = t_param->shape[1];
    const size_t Tn = t_param->shape[2];
    const size_t Tm = t_param->shape[3];

    size_t weights_ix, ifm_ix, ofm_ix;
    // size_t ofm_tile_idx = 0;

    size_t ofm_tile_size = Tr*Tc*Tm;
    float local_buffer[ofm_tile_size];

    // ofm->dataf = (float*)pvPortMalloc(ofm->num_data*sizeof(float)); // not eough memory for heap
    // memset(ofm->dataf,0,ofm->num_data*sizeof(ofm->dataf[0]));
    // am_util_stdio_printf("HERE! Layer: %ld\n", config->exe_status[2]);

    for (r = config->exe_status[3]; r < out_rows; r+=Tr) {
        for (c = config->exe_status[4]; c < out_cols; c+=Tc) {
            for (n = config->exe_status[5]; n<in_channels; n+=Tn){
                for (m = config->exe_status[6]; m<out_channels; m+=Tm){
                    // ----------------------------------------------
                    memset(local_buffer,0,ofm_tile_size*sizeof(local_buffer[0]));
                    // ofm_tile_idx = 0;
                    for (tr=r; tr < MIN(r+Tr, out_rows); ++tr) {
                        for (tc=c; tc < MIN(c+Tc, out_cols); ++tc) {
                            for (k0=0; k0 < weights->shape[0]; ++k0) {
                                for (k1=0; k1 < weights->shape[1]; ++k1) {
                                    for (tn=n; tn < MIN(n+Tn, in_channels); ++tn) {
                                        for (tm=m; tm < MIN(m+Tm, out_channels); ++tm) {

                                            ofm_ix = tr*(ofm->shape[2]*ofm->shape[1])
                                                        + tc*(ofm->shape[2]) + tm;

                                            weights_ix = k0*(weights->shape[3]*weights->shape[2]*weights->shape[1])
                                                        + k1*(weights->shape[3]*weights->shape[2])
                                                        + tn*(weights->shape[3]) + tm;

                                            ifm_ix = (tr*stride[0]
                                                    + dilation[0]*k0)*(ifm->shape[2]*ifm->shape[1])
                                                    + (tc*stride[1] + dilation[1]*k1)*(ifm->shape[2]) + tn;

                                            local_buffer[tr-r+tc-c+tm-m]
                                            // ofm->dataf[ofm_ix]
                                            +=
                                            ui32_to_fp32(*(uint32_t*)(weights->nvm_start + (weights_ix*DATA_SIZE))) // weights[ix]
                                            *
                                            ui32_to_fp32(*(uint32_t*)(ifm->nvm_start + (ifm_ix*DATA_SIZE))); // ifm[ix]

                                            // ofm->dataf[ofm_ix] = local_buffer[tr-r+tc-c+tm-m];

                                            // ofm_tile_idx++;
                    }}}}}}
                    float_relu_func(local_buffer, out_channels);
                    float_bias_add(local_buffer, bias->dataf, out_channels);

                    if(vm_to_nvm(local_buffer, ofm_tile_size, ofm->nvm_start, &ofm->nvm_end) != 0){
                        am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
                    }
                    // counter++; // REMOVE
                    // am_util_stdio_printf("Counter: %d\n", counter); // REMOVE
                    // ofm->nvm_start = ofm->nvm_end + DATA_SIZE;
//==================================<RTEN>============================================
            uint32_t curr_exe_status[7] = { 
                config->exe_status[0], // exe_status[0] = 1
                config->exe_status[1], // exe_status[1] = 0
                layer_id,
                r,
                c,
                n,
                m
            };
            if(0 != am_hal_mram_info_program(AM_HAL_MRAM_INFO_KEY,
                                        curr_exe_status,
                                        0,
                                        (sizeof(curr_exe_status) / sizeof(uint32_t))))
            {
                am_util_stdio_printf("ERROR! am_hal_mram_info_program return non-zero\n");
            }
//=================================</RTEN>============================================
    }}}}
    
    return;

}

void _conv2d_filter_xip (size_t layer_id, exe_config *config,
    const size_t stride[2], const size_t dilation[2],
    const r10_tensor* weights, const r10_tensor* bias, 
    const r10_tensor* ifm, r10_tensor* ofm)
{

    const size_t out_rows = ofm->shape[0];
    const size_t out_cols = ofm->shape[1];
    const size_t out_channels = ofm->shape[2];
    const size_t in_channels = ifm->shape[2];

    // ix is the index of the element in r10_tensor
    size_t weights_ix, ifm_ix, ofm_ix;

    // float *local_buffer; // float local buffer for working
    float local_buffer[out_channels]; // float local buffer for working

    for (x0 = config->exe_status[3]; x0 < out_rows; ++x0) {
        for (x1 = config->exe_status[4]; x1 < out_cols; ++x1) {

            // am_util_stdio_printf("x0: %ld\n", x0);
            // am_util_stdio_printf("x1: %ld\n", x1);

            // memset(ofm->dataf,0,out_channels*sizeof(ofm->dataf[0]));
            memset(local_buffer,0,out_channels*sizeof(local_buffer[0]));

            for (z0=0; z0 < weights->shape[0]; ++z0) {
                for (z1=0; z1 < weights->shape[1]; ++z1) {
                    for (q=0; q < in_channels; ++q) {
                        for (k=0; k < out_channels; ++k) {

                            // ofm_ix: map k to ofm
                            ofm_ix = x0*(ofm->shape[2]*ofm->shape[1]) + x1*(ofm->shape[2]) + k;
                            // weights_ix
                            weights_ix = z0*(weights->shape[3]*weights->shape[2]*weights->shape[1])
                                        + z1*(weights->shape[3]*weights->shape[2])
                                        + q*(weights->shape[3]) + k;
                            // ifm_ix
                            ifm_ix = (x0*stride[0] + dilation[0]*z0)*(ifm->shape[2]*ifm->shape[1])
                                        + (x1*stride[1] + dilation[1]*z1)*(ifm->shape[2]) + q;
                            local_buffer[k] 
                            +=
                            ui32_to_fp32(*(uint32_t*)(weights->nvm_start + (weights_ix*DATA_SIZE))) // weights[ix]
                            *
                            ui32_to_fp32(*(uint32_t*)(ifm->nvm_start + (ifm_ix*DATA_SIZE))); // ifm[ix]

                            // ofm->dataf[ofm_ix] = local_buffer[k]; // REMOVE

                            counter++; // REMOVE
                            // am_util_stdio_printf("Counter: %d\n", counter); // REMOVE
                        }
                    }
                }
            }

            float_relu_func(local_buffer, out_channels);
            float_bias_add(local_buffer, bias->dataf, out_channels);
            if(vm_to_nvm(local_buffer, out_channels, ofm->nvm_start, &ofm->nvm_end) != 0){
                am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
            }

            // vm_to_nvm(ofm->dataf, out_channels, ARB_ADDRESS); // TODO
//==================================<RTEN>============================================
            uint32_t curr_exe_status[5] = { 
                config->exe_status[0], // exe_status[0] = 1
                config->exe_status[1], // exe_status[1] = 0
                layer_id,
                x0,
                x1
            };
            if(0 != am_hal_mram_info_program(AM_HAL_MRAM_INFO_KEY,
                                        curr_exe_status,
                                        0,
                                        (sizeof(curr_exe_status) / sizeof(uint32_t))))
            {
                am_util_stdio_printf("ERROR! am_hal_mram_info_program return non-zero\n");
            }
//=================================</RTEN>============================================
        }
    }
    
    return;

}

void _conv2d_layer_xip (size_t layer_id, exe_config *config,
    const size_t stride[2], const size_t dilation[2],
    const r10_tensor* weights, const r10_tensor* bias, 
    const r10_tensor* ifm, r10_tensor* ofm)
{

    const size_t out_rows = ofm->shape[0];
    const size_t out_cols = ofm->shape[1];
    const size_t out_channels = ofm->shape[2];
    const size_t in_channels = ifm->shape[2];


    // ix is the index of the element in r10_tensor
    size_t weights_ix, ifm_ix;

    // am_util_stdio_printf("HERE! Layer: %ld\n", config->exe_status[2]);


    // ofm->dataf = working_array;

    ofm->dataf = (float*)pvPortMalloc(ofm->num_data*sizeof(float));
    memset(ofm->dataf,0,ofm->num_data*sizeof(ofm->dataf[0]));
    // nvm_to_vm(ofm->dataf, ofm->num_data, ofm->nvm_start);

    for (x0=0; x0 < out_rows; ++x0) {
        for (x1=0; x1 < out_cols; ++x1) {

            // am_util_stdio_printf("x0: %ld\n", x0);
            // am_util_stdio_printf("x1: %ld\n", x1);

            for (z0=0; z0 < weights->shape[0]; ++z0) {
                for (z1=0; z1 < weights->shape[1]; ++z1) {
                    for (q=0; q < in_channels; ++q) {
                        for (k=0; k < out_channels; ++k) {

                            // weights_ix
                            weights_ix = z0*(weights->shape[3]*weights->shape[2]*weights->shape[1])
                                        + z1*(weights->shape[3]*weights->shape[2])
                                        + q*(weights->shape[3]) + k;
                            // ifm_ix
                            ifm_ix = (x0*stride[0] + dilation[0]*z0)*(ifm->shape[2]*ifm->shape[1])
                                        + (x1*stride[1] + dilation[1]*z1)*(ifm->shape[2]) + q;
                            
                            ofm->dataf[x0*(ofm->shape[2]*ofm->shape[1]) + x1*(ofm->shape[2]) + k] 
                            +=
                            ui32_to_fp32(*(uint32_t*)(weights->nvm_start + (weights_ix*DATA_SIZE))) // weights[ix]
                            *
                            ui32_to_fp32(*(uint32_t*)(ifm->nvm_start + (ifm_ix*DATA_SIZE))); // ifm[ix]
                            // ifm->dataf[ifm_ix];

                            // counter++; // REMOVE
                            // am_util_stdio_printf("Counter: %d\n", counter); // REMOVE

                        }
                    }
                }
            }
        }
    }

    r10_relu_func(ofm);
    r10_bias_add(ofm,bias);

    if(vm_to_nvm(ofm->dataf, ofm->num_data, ofm->nvm_start, &ofm->nvm_end) != 0){
        am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
    }

    vPortFree(ofm->dataf);
    
    return;

}

void _conv2d_vanilla (size_t layer_id, exe_config *config,
    const size_t stride[2], const size_t dilation[2],
    r10_tensor* weights, const r10_tensor* bias, 
    r10_tensor* ifm, r10_tensor* ofm)
{

    const size_t out_rows = ofm->shape[0];
    const size_t out_cols = ofm->shape[1];
    const size_t out_channels = ofm->shape[2];
    const size_t in_channels = ifm->shape[2];
        
        
    //<RTENLab> malloc correct bu not enough heap - change all dnn together

    //<RTENLab> malloc correct bu not enough heap - change all dnn together
    ofm->dataf = (float*)pvPortMalloc(ofm->num_data*sizeof(float));
    memset(ofm->dataf,0,ofm->num_data*sizeof(ofm->dataf[0]));

    weights->dataf = (float*)pvPortMalloc(weights->num_data*sizeof(float));
    nvm_to_vm(weights->dataf, weights->num_data, weights->nvm_start);

    ifm->dataf = (float*)pvPortMalloc(ifm->num_data*sizeof(float));
    nvm_to_vm(ifm->dataf, ifm->num_data, ifm->nvm_start);

    for (x0 = config->exe_status[3]; x0 < out_rows; ++x0) {
        for (x1=config->exe_status[4]; x1 < out_cols; ++x1) {
            for (z0=config->exe_status[5]; z0 < weights->shape[0]; ++z0) {
                for (z1=config->exe_status[6]; z1 < weights->shape[1]; ++z1) {
                    for (q=config->exe_status[7]; q < in_channels; ++q) {
                        for (k=config->exe_status[8]; k < out_channels; ++k) {
                            ofm->dataf[x0*(ofm->shape[2]*ofm->shape[1]) + x1*(ofm->shape[2]) + k] 
                            +=
                            weights->dataf[z0*(weights->shape[3]*weights->shape[2]*weights->shape[1])
                                        + z1*(weights->shape[3]*weights->shape[2])
                                        + q*(weights->shape[3]) + k]
                            *
                            ifm->dataf[(x0*stride[0]
                                            + dilation[0]*z0)*(ifm->shape[2]*ifm->shape[1])
                                        + (x1*stride[1] + dilation[1]*z1)*(ifm->shape[2]) + q];
                            
                            // counter++; // REMOVE
                            // am_util_stdio_printf("Counter: %d\n", counter); // REMOVE
                        }
                    }
                }
            }
        }
    }

    r10_relu_func(ofm);
    r10_bias_add(ofm,bias);

    // <RTEN> mimic the JIT checkpointing data for one layer 
/*
    // checkpoint
    if(layer_id == 3){
        vm_to_nvm(ifm->dataf, ifm->num_data, ifm->nvm_start, &ifm->nvm_end);
        vm_to_nvm(ofm->dataf, ofm->num_data, ofm->nvm_start, &ofm->nvm_end);

        // readback
        nvm_to_vm(weights->dataf, weights->num_data, weights->nvm_start);
        nvm_to_vm(ifm->dataf, ifm->num_data, ifm->nvm_start);
    }
*/
    // </RTEN>

    uint32_t curr_exe_status[7] = { 
                config->exe_status[0], // exe_status[0] = 1
                config->exe_status[1], // exe_status[1] = 0
                layer_id+1
            };

    vPortFree(weights->dataf);
    vPortFree(ifm->dataf);
    
    
    return;

}

void _conv2d_tiled (size_t layer_id, exe_config *config,
    const size_t stride[2], const size_t dilation[2],
    tiled_param* t_param,
    r10_tensor* weights, const r10_tensor* bias, 
    r10_tensor* ifm, r10_tensor* ofm)
{
    const size_t out_rows = ofm->shape[0]; // R
    const size_t out_cols = ofm->shape[1]; // C
    const size_t out_channels = ofm->shape[2]; // M
    const size_t in_channels = ifm->shape[2]; // N

    const size_t Tr = t_param->shape[0];
    const size_t Tc = t_param->shape[1];
    const size_t Tn = t_param->shape[2];
    const size_t Tm = t_param->shape[3];


    size_t ofm_tile_size = Tr*Tc*Tm;
    float local_buffer[ofm_tile_size];

    ifm->dataf = (float*)pvPortMalloc(ifm->num_data*sizeof(float));
    nvm_to_vm(ifm->dataf, ifm->num_data, ifm->nvm_start);
    weights->dataf = (float*)pvPortMalloc(weights->num_data*sizeof(float));
    nvm_to_vm(weights->dataf, weights->num_data, weights->nvm_start);

    for (r = config->exe_status[3]; r < out_rows; r+=Tr) {
        for (c = config->exe_status[4]; c < out_cols; c+=Tc) {
            for (n = config->exe_status[5]; n<in_channels; n+=Tn){
                for (m = config->exe_status[6]; m<out_channels; m+=Tm){
                    // ----------------------------------------------
                    memset(local_buffer,0,ofm_tile_size*sizeof(local_buffer[0]));
                    // ofm_tile_idx = 0;
                    for (tr=r; tr < MIN(r+Tr, out_rows); ++tr) {
                        for (tc=c; tc < MIN(c+Tc, out_cols); ++tc) {
                            for (k0=0; k0 < weights->shape[0]; ++k0) {
                                for (k1=0; k1 < weights->shape[1]; ++k1) {
                                    for (tn=n; tn < MIN(n+Tn, in_channels); ++tn) {
                                        for (tm=m; tm < MIN(m+Tm, out_channels); ++tm) {

                                            local_buffer[tr-r+tc-c+tm-m]
                                            +=
                                            weights->dataf[k0*(weights->shape[3]*weights->shape[2]*weights->shape[1])
                                                            + k1*(weights->shape[3]*weights->shape[2])
                                                            + tn*(weights->shape[3]) + tm]
                                            *
                                            ifm->dataf[(tr*stride[0]
                                                            + dilation[0]*k0)*(ifm->shape[2]*ifm->shape[1])
                                                        + (tc*stride[1] + dilation[1]*k1)*(ifm->shape[2]) + tn];

                    }}}}}}
                    float_relu_func(local_buffer, out_channels);
                    float_bias_add(local_buffer, bias->dataf, out_channels);
                    if(vm_to_nvm(local_buffer, ofm_tile_size, ofm->nvm_start, &ofm->nvm_end) != 0){
                        am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
                    }
                    // ofm->nvm_start = ofm->nvm_end + DATA_SIZE;
//==================================<RTEN>============================================
            uint32_t curr_exe_status[7] = { 
                config->exe_status[0], // exe_status[0] = 1
                config->exe_status[1], // exe_status[1] = 0
                layer_id,
                r,
                c,
                n,
                m
            };
            if(0 != am_hal_mram_info_program(AM_HAL_MRAM_INFO_KEY,
                                        curr_exe_status,
                                        0,
                                        (sizeof(curr_exe_status) / sizeof(uint32_t))))
            {
                am_util_stdio_printf("ERROR! am_hal_mram_info_program return non-zero\n");
            }
//=================================</RTEN>============================================
    }}}}
    

    vPortFree(ifm->dataf);
    vPortFree(weights->dataf);
    
    return;

}

void _conv2d_filter (size_t layer_id, exe_config *config,
    const size_t stride[2], const size_t dilation[2],
    r10_tensor* weights, const r10_tensor* bias, 
    r10_tensor* ifm, r10_tensor* ofm)
{

    const size_t out_rows = ofm->shape[0];
    const size_t out_cols = ofm->shape[1];
    const size_t out_channels = ofm->shape[2];
    const size_t in_channels = ifm->shape[2];


    float local_buffer[out_channels]; // float local buffer for working

    ifm->dataf = (float*)pvPortMalloc(ifm->num_data*sizeof(float));
    nvm_to_vm(ifm->dataf, ifm->num_data, ifm->nvm_start);
    weights->dataf = (float*)pvPortMalloc(weights->num_data*sizeof(float));
    nvm_to_vm(weights->dataf, weights->num_data, weights->nvm_start);


    // local_buffer = (float*)pvPortMalloc(out_channels*sizeof(float)); // Only malloc the filter

    for (x0 = config->exe_status[3]; x0 < out_rows; ++x0) {
        for (x1 = config->exe_status[4]; x1 < out_cols; ++x1) {

            // am_util_stdio_printf("x0: %ld\n", x0);
            // am_util_stdio_printf("x1: %ld\n", x1);

            memset(local_buffer,0,out_channels*sizeof(local_buffer[0]));

            for (z0=0; z0 < weights->shape[0]; ++z0) {
                for (z1=0; z1 < weights->shape[1]; ++z1) {
                    for (q=0; q < in_channels; ++q) {
                        for (k=0; k < out_channels; ++k) {
                            local_buffer[k] 
                            +=
                            weights->dataf[z0*(weights->shape[3]*weights->shape[2]*weights->shape[1])
                                        + z1*(weights->shape[3]*weights->shape[2])
                                        + q*(weights->shape[3]) + k]
                            *
                            ifm->dataf[(x0*stride[0]
                                        + dilation[0]*z0)*(ifm->shape[2]*ifm->shape[1])
                                        + (x1*stride[1] + dilation[1]*z1)*(ifm->shape[2]) + q];
                            // counter++; // REMOVE
                            // am_util_stdio_printf("Counter: %d\n", counter); // REMOVE
                        }
                    }
                }
            }

            float_relu_func(local_buffer, out_channels);
            float_bias_add(local_buffer, bias->dataf, out_channels);
            if(vm_to_nvm(local_buffer, out_channels, ofm->nvm_start, &ofm->nvm_end) != 0){
                am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
            }

            // vm_to_nvm(ofm->dataf, out_channels, ARB_ADDRESS); // TODO
//==================================<RTEN>============================================
            uint32_t curr_exe_status[5] = { 
                config->exe_status[0], // exe_status[0] = 1
                config->exe_status[1], // exe_status[1] = 0
                layer_id,
                x0,
                x1
            };
            if(0 != am_hal_mram_info_program(AM_HAL_MRAM_INFO_KEY,
                                        curr_exe_status,
                                        0,
                                        (sizeof(curr_exe_status) / sizeof(uint32_t))))
            {
                am_util_stdio_printf("ERROR! am_hal_mram_info_program return non-zero\n");
            }
//=================================</RTEN>============================================
        }
    }

    vPortFree(ifm->dataf);
    vPortFree(weights->dataf);
    
    return;

}

void _conv2d_layer (size_t layer_id, exe_config *config,
    const size_t stride[2], const size_t dilation[2],
    r10_tensor* weights, const r10_tensor* bias, 
    r10_tensor* ifm, r10_tensor* ofm)
{

    const size_t out_rows = ofm->shape[0];
    const size_t out_cols = ofm->shape[1];
    const size_t out_channels = ofm->shape[2];
    const size_t in_channels = ifm->shape[2];

    // ofm->dataf = working_array;

    ifm->dataf = (float*)pvPortMalloc(ifm->num_data*sizeof(float));
    nvm_to_vm(ifm->dataf, ifm->num_data, ifm->nvm_start);
    weights->dataf = (float*)pvPortMalloc(weights->num_data*sizeof(float));
    nvm_to_vm(weights->dataf, weights->num_data, weights->nvm_start);

    ofm->dataf = (float*)pvPortMalloc(ofm->num_data*sizeof(float));
    memset(ofm->dataf,0,ofm->num_data*sizeof(ofm->dataf[0]));
    // nvm_to_vm(ofm->dataf, ofm->num_data, ofm->nvm_start);

    for (x0=0; x0 < out_rows; ++x0) {
        for (x1=0; x1 < out_cols; ++x1) {
            for (z0=0; z0 < weights->shape[0]; ++z0) {
                for (z1=0; z1 < weights->shape[1]; ++z1) {
                    for (q=0; q < in_channels; ++q) {
                        for (k=0; k < out_channels; ++k) {

                            ofm->dataf[x0*(ofm->shape[2]*ofm->shape[1])
                                        + x1*(ofm->shape[2]) + k] 
                            +=
                            weights->dataf[z0*(weights->shape[3]*weights->shape[2]*weights->shape[1])
                                        + z1*(weights->shape[3]*weights->shape[2])
                                        + q*(weights->shape[3]) + k]
                            *
                            ifm->dataf[(x0*stride[0]
                                        + dilation[0]*z0)*(ifm->shape[2]*ifm->shape[1])
                                        + (x1*stride[1] + dilation[1]*z1)*(ifm->shape[2]) + q];

                        }
                    }
                }
            }
        }
    }

    r10_relu_func(ofm);
    r10_bias_add(ofm,bias);

    if(vm_to_nvm(ofm->dataf, ofm->num_data, ofm->nvm_start, &ofm->nvm_end) != 0){
        am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
    }

    vPortFree(ofm->dataf);
    vPortFree(ifm->dataf);
    vPortFree(weights->dataf);
    
    return;

}

void r10_conv2d (struct exe_config *config, struct r10cnn_layer *layer)
{
    // <MII>
    enum exe_mode mode = layer->exe;
    enum mem_mode mem = layer->mem;
    // </MII>

#if defined(UART_PROFILE)
    begin = xTaskGetTickCount();
    am_util_stdio_printf("CONV%ld (V|L|T|F|S): %ld\n", layer->layer_id, mode);
#endif

    if(mem==XIP)
    {
        switch (mode)
        {
        case VANILLA:
            _conv2d_vanilla_xip(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        break;
        case TILED:
            _conv2d_tiled_xip(layer->layer_id, config, layer->stride, layer->dilation, &layer->t_param, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        break;
        case FILTER:
            _conv2d_filter_xip(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        break;
        case LAYER:
            _conv2d_layer_xip(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        break;
        default:
            am_util_stdio_printf("ERROR: Invalid mode for conv2d\n");
        break;
        }
    }
    else if (mem==NORMAL)
    {
        switch (mode)
        {
        case VANILLA:
            _conv2d_vanilla(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        break;
        case TILED:
            _conv2d_tiled(layer->layer_id, config, layer->stride, layer->dilation, &layer->t_param, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        break;
        case FILTER:
            _conv2d_filter(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        break;
        case LAYER:
            _conv2d_layer(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        break;
        default:
            am_util_stdio_printf("ERROR: Invalid mode for conv2d\n");
        break;
        }
    }
    else
    {
        am_util_stdio_printf("ERROR: Invalid memory mode for conv2d\n");
    }

    // am_util_stdio_printf("Counter: %d\n", counter); // REMOVE

#if defined(UART_PROFILE)
    elapse = xTaskGetTickCount() - begin;
    am_util_stdio_printf("CONV Layer %ld: %ld\n", layer->layer_id, elapse);
#endif 
    return;

}

/**
 * 2D (spatial) Padding.
 * Within CONV layers
 */
void r10_pad2d(struct exe_config *config, struct r10cnn_layer *layer)
{
    r10_tensor *input = &layer->ifm;
    input->dataf = (float*)pvPortMalloc(input->num_data*sizeof(float));
    nvm_to_vm(input->dataf, input->num_data, input->nvm_start);
    r10_tensor *output = &layer->ofm;
    output->dataf = (float*)pvPortMalloc(layer->ofm.num_data*sizeof(float));
    memset(output->dataf,0,output->num_data*sizeof(output->dataf[0])); // clear the corrupted

    // <RTEN> compatibility ad hoc
    float fill = 0.0f; 
    size_t pad[4] = {1,1,1,1}; 
    // </RTEN>

    const size_t in_height = input->shape[0];
    const size_t in_width = input->shape[1];
    const size_t in_channels = input->shape[2];
    const size_t pad_top = pad[0];
    const size_t pad_left = pad[2];
    const size_t pad_right = pad[3];

    // set output array to fill value
    if (fabs(fill) < 1e-6) {
        // fill is ~zero, use memset
        memset(output->dataf,0,output->num_data*sizeof(output->dataf[0]));
    }
    else {
        for(size_t i=0; i<output->num_data; ++i) {
            output->dataf[i] = fill;
        }
    }
    // memcpy the old array in the middle
    size_t offset = in_channels*(pad_left+pad_right+in_width)*pad_top +
                    in_channels*pad_left;
    const size_t num = in_channels*in_width;
    const size_t step = num+in_channels*(pad_left+pad_right);
    for (size_t i=0; i<in_height; ++i) {
        memcpy(&output->dataf[offset],
               &input->dataf[i*num],
               num*sizeof(input->dataf[0]));
        offset += step;
    }

    if(vm_to_nvm(output->dataf, output->num_data, output->nvm_start, &output->nvm_end) != 0){
        am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
    }
    vPortFree(output->dataf);
    vPortFree(input->dataf);
    return;
}



void _conv1d_layer (size_t layer_id, exe_config *config,
    const size_t stride[2], const size_t dilation[2],
    r10_tensor* weights, const r10_tensor* bias, 
    r10_tensor* input, r10_tensor* output){

    input->dataf = (float*)pvPortMalloc(input->num_data*sizeof(float));
    nvm_to_vm(input->dataf, input->num_data, input->nvm_start);
    weights->dataf = (float*)pvPortMalloc(weights->num_data*sizeof(float));
    nvm_to_vm(weights->dataf, weights->num_data, weights->nvm_start);

    output->dataf = (float*)pvPortMalloc(output->num_data*sizeof(float));
    memset(output->dataf,0,output->num_data*sizeof(output->dataf[0]));

    memset(output->dataf,0,output->num_data*sizeof(output->dataf[0]));

    const size_t out_times = output->shape[0];
    const size_t out_channels = output->shape[1];
    const size_t in_channels = input->shape[1];

    for (size_t x0=0; x0 < out_times; ++x0) {
        for (size_t z=0; z < weights->shape[0]; ++z) {
            for (size_t q=0; q < in_channels; ++q) {
                for (size_t k=0; k < out_channels; ++k) {
                    output->dataf[x0*out_channels + k] +=
                        weights->dataf[z*(weights->shape[2]*weights->shape[1]) +
                                q*(weights->shape[2]) + k]
                        *
                        input->dataf[(x0*stride[0] + dilation[0]*z)*in_channels + q];
                }
            }
        }
    }
    r10_bias_add(output,bias);
    r10_relu_func(output);

    if(vm_to_nvm(output->dataf, output->num_data, output->nvm_start, &output->nvm_end) != 0){
        am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
    }

    vPortFree(output->dataf);
    vPortFree(input->dataf);
    vPortFree(weights->dataf);

    return;

}

/**
 * 1D (temporal) Convolution.
 * Assumes a "channels last" structure.
 *
 * :param output: output tensor.
 * :param input: input tensor.
 * :param kernel: kernel tensor.
 * :param bias: bias tensor.
 * :param stride: stride length of the convolution.
 * :param dilation: dilation rate to use for dilated convolution.
 * :param activation: activation function to apply to output.
 */
void r10_conv1d(struct exe_config *config, struct r10cnn_layer *layer) {

    // <MII>
    enum exe_mode mode = layer->exe;
    enum mem_mode mem = layer->mem;
    // </MII>

#if defined(UART_PROFILE)
    begin = xTaskGetTickCount();
    am_util_stdio_printf("CONV%ld (V|L|T|F|S): %ld\n", layer->layer_id, mode);
#endif

if(mem==XIP)
    {
        switch (mode)
        {
        // case VANILLA:
        //     _conv1d_vanilla_xip(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        // break;
        // case TILED:
        //     _conv1d_tiled_xip(layer->layer_id, config, layer->stride, layer->dilation, &layer->t_param, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        // break;
        // case FILTER:
        //     _conv1d_filter_xip(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        // break;
        // case LAYER:
        //     _conv1d_layer_xip(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        // break;
        default:
            am_util_stdio_printf("ERROR: Invalid mode for conv1d\n");
        break;
        }
    }
    else if (mem==NORMAL)
    {
        switch (mode)
        {
        // case VANILLA:
        //     _conv1d_vanilla(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        // break;
        // case TILED:
        //     _conv1d_tiled(layer->layer_id, config, layer->stride, layer->dilation, &layer->t_param, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        // break;
        // case FILTER:
        //     _conv1d_filter(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        // break;
        case LAYER:
            _conv1d_layer(layer->layer_id, config, layer->stride, layer->dilation, &layer->weights, &layer->bias, &layer->ifm, &layer->ofm);
        break;
        default:
            am_util_stdio_printf("ERROR: Invalid mode for conv1d\n");
        break;
        }
    }
    else
    {
        am_util_stdio_printf("ERROR: Invalid memory mode for conv1d\n");
    }

    // am_util_stdio_printf("Counter: %d\n", counter); // REMOVE

#if defined(UART_PROFILE)
    elapse = xTaskGetTickCount() - begin;
    am_util_stdio_printf("CONV Layer %ld: %ld\n", layer->layer_id, elapse);
#endif 
    return;
}
