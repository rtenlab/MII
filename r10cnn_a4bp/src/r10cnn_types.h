#ifndef R10CNN_TYPES_H
#define R10CNN_TYPES_H

#include "r10cnn_config.h"

/*******************************************************************
 * Common Const
 *******************************************************************/
#define MAX_MODEL_NAME 100
enum layer_func_flag
{
    CONV = 1,
    POOLING = 2,
    CORE = 3
};
typedef enum layer_func_flag layer_func_flag;


/*******************************************************************
 * Data Type Const
 *******************************************************************/
#ifndef bin16_MIN
#define bin16_MIN 0x8000
#endif //bin16_MIN

#ifndef bin16_MAX
#define bin16_MAX 0x7fff
#endif //bin16_MAX

#ifndef bin32_MIN
#define bin32_MIN  0x80000000
#endif //bin32_MIN

#ifndef bin32_MAX
#define bin32_MAX  0x7fffffff
#endif //bin32_MAX

#ifndef R10_MAX_NDIM
#define R10_MAX_NDIM 5
#endif //R10_MAX_NDIM

/*******************************************************************
 * Primitive Data Type
 *******************************************************************/
typedef signed short bin16;  // binary16 in format 0x0000 (2 bytes)
typedef signed int bin32;  // binary32 in format 0x00000000 (4 bytes)

typedef unsigned short ubin16;  // unsigned binary16 in format 0x0000 (2 bytes)
typedef unsigned int ubin32;  // unsigned binary32 in format 0x00000000 (4 bytes)

/** unsigned binary64 in format 0x000000000000 (8 bytes) - size of all */
// typedef unsigned long ub64;  


/*******************************************************************
 * Complex Data Type
 *******************************************************************/

/**
 * r10_tensor -  basic tensor unit in r10cnn
 * @data: type bin16 pointer to array of tensor values 
 *        flattened in row major order - half precision 
 * @data_f: type float Pointer to array of tensor values 
 *        flattened in row major order - full precision
 * @ndim: Rank of the tensor (number of dimensions)
 * @num_data: Number of elements in the tensor
 * @shape: Array, size of the tensor in each dimension
 *
 * Return: binary 16 value needed assign
 */
struct r10_tensor
{
    // non-volatile memory (NVM) field
    uint32_t nvm_start; // start address of tensor in NVM
    uint32_t nvm_end; // end address of tensor in NVM

    // volatile memory (VM) field - 
    // TODO: should be deleted because only need nvm address
    bin16* data_bin16; // raw data array - half precision
    float* dataf; // raw data array - full precision 

    size_t ndim; // or called channel

    size_t num_data;

    size_t shape[R10_MAX_NDIM]; // {row, col, chn, num, 5th}
};
typedef struct r10_tensor r10_tensor;

struct tiled_param
{

    size_t num_data; // number of data in tile

    size_t str;   //stride
    size_t pad;   //padding

    /*
     * {Tr, Tc, Tn, Tm, T5}
     * Tr: Tile OFM row
     * Tc: Tile OFM col
     * Tn: Tile IFM channel
     * Tm: Tile OFM channel
     * T5: fifth dimension 
     */
    size_t shape[R10_MAX_NDIM];
};
typedef struct tiled_param tiled_param;

struct r10cnn_layer
{
    size_t layer_id;
    enum layer_func_flag layer_f;

    void (*conv_func)
    (struct exe_config *config, struct r10cnn_layer *layer);

    void (*pooling_func)
    (size_t layer_id, exe_config *config,
    r10_tensor* ifm, r10_tensor* ofm);

    void (*core_func)
    (size_t layer_id, exe_config *config,
    r10_tensor* weights, r10_tensor* bias, 
    r10_tensor* ifm, r10_tensor* ofm,
    r10_tensor* workspace);

    r10_tensor ifm;
    r10_tensor ofm;
    r10_tensor weights;
    r10_tensor bias;

    // Layer specific parameters
    // CORE layers only
    r10_tensor workspace;
    
    // CONV layers only
    size_t stride[R10_MAX_NDIM];
    size_t dilation[R10_MAX_NDIM];

    // Execution specific parameters
    // Tiled parameters
    tiled_param t_param;
};
typedef struct r10cnn_layer r10cnn_layer;

struct r10cnn_model
{
    size_t num_layers; // number of layer

    r10cnn_layer *layers;

    char model_name[MAX_MODEL_NAME];
};
typedef struct r10cnn_model r10cnn_model;


#endif //R10CNN_TYPES_H