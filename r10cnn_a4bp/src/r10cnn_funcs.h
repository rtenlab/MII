#ifndef R10CNN_FUNCS_H
#define R10CNN_FUNCS_H

#include "r10cnn_types.h"
#include "r10cnn_config.h"
/*******************************************************************
 * r10_cnn MACROS
 *******************************************************************/
#define SWAP(x, y, T) do { T SWAP = x; x = y; y = SWAP; } while (0)
#define PERCENTAGE(V, T) (100 - (((T - V) * 100) / T))
#define MIN(x,y) ((x<y)?x:y)
#define MAX(x,y) ((x>y)?x:y)

/*******************************************************************
 * r10_cnn.c
 *******************************************************************/
int r10cnn_driver(struct exe_config *config, 
    struct r10cnn_model *r10cnn, float out_array[10]);

/*******************************************************************
 * r10cnn_conv_layers.c - CONV
 *******************************************************************/
void r10_conv2d (struct exe_config *config, struct r10cnn_layer *layer);

/*******************************************************************
 * r10cnn_pool_layers.c - POOLING
 *******************************************************************/
void r10_global_avg_pool2d(size_t layer_id, 
    exe_config *config, r10_tensor* ifm, r10_tensor* ofm);

/*******************************************************************
 * r10cnn_core_layers.c - CORE
 *******************************************************************/
void r10_dense(size_t layer_id, exe_config *config, r10_tensor* kernel, r10_tensor* bias, 
    r10_tensor* ifm, r10_tensor* ofm, r10_tensor* workspace);

/*******************************************************************
 * r10cnn_rtos.c
 *******************************************************************/

/*******************************************************************
 * r10cnn_utils.c
 *******************************************************************/
void a4p_show_r10cnn(struct r10cnn_model* r10cnn);
void a4p_show_r10cnn_layer(enum precision prec, struct r10cnn_layer* layer);
size_t get_max_size_tensor(struct r10cnn_model r10cnn);
size_t max_in_float_array(float* array, size_t size);
size_t max_in_bin16_array(bin16* array, size_t size);
size_t max_in_r10_tensor(r10_tensor* r10tensor, size_t size);
void a4p_print_config(struct exe_config* config);
void r10_bias_add(r10_tensor* A, const r10_tensor* b);
void a4p_print_float_array(float *pBuf, size_t size);
void a4p_print_uint32t_array(uint32_t *pBuf, size_t size);
int tile_valid(r10cnn_layer *layer);
int check_r10cnn(struct exe_config *config, struct r10cnn_model *r10cnn);
uint32_t fp32_to_ui32(float n);
float ui32_to_fp32(uint32_t n);
int vm_to_nvm(float* data, size_t num_data, uint32_t begin_address, uint32_t *end_address);
int validate_nvm(float* data, size_t num_data, uint32_t begin_address);
int nvm_to_vm(float* data, size_t num_data, uint32_t begin_address);
int check_info0(uint32_t* exe_status, size_t num_data);
void a4p_show_exe_status(uint32_t* exe_status);
int clear_exe_status(/*Need to aloocate for different r10cnn*/);
// void dummy();

#endif //R10CNN_FUNCS_H