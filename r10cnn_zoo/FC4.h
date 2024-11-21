#ifndef FC4_H
#define FC4_H
#if defined AM_PART_APOLLO4B || AM_PART_APOLLO4P
#include "../../r10cnn_a4bp/src/r10_cnn.h"
#else
#include "../../libr10cnn/r10_cnn.h"
#endif

#pragma PERSISTENT(fc4)

float fc4_input0[490] =
{0,};

float fc4_output0[12] =
{0,1,0,0,0,0,0,0,0,0,0,0,};

float fc4_output_256[256] =
{0,};

float fc_fwork_0[491] =
{0,};

// float fc4_weights_1[65792] =
// {0,};

float fc4_bias[256] =
{0,};

r10cnn_layer fc4[4] = {
{
    .mem = NORMAL,
	.exe = LAYER,
	
	.layer_id = 0,
	.layer_f = CORE,
	.core_func = r10_dense, // Exclusive to r10_dense
	.ifm = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
		.dataf = fc4_input0,
		.ndim = 1,
		.num_data = 490,
		.shape = {490,1,1,1,1}
	},
	.ofm = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
        .dataf = fc4_output_256,
		.ndim = 1,
		.num_data = 256,
		.shape = {256,1,1,1,1}
	},
	.weights = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
		.ndim = 2,
		.num_data = 125696,
		.shape = {256,491,1,1,1}
	},
	.bias = (r10_tensor){
		.data_bin16 = 0,
		.dataf = fc4_bias,
		.ndim = 1,
		.num_data = 6,
		.shape = {6,1,1,1,1}
	},
	.workspace = (r10_tensor){
		.data_bin16 = 0,
		.dataf = fc_fwork_0,
		.ndim = 1,
		.num_data = 491,
		.shape = {491,1,1,1,1}
	},
    .t_param = (tiled_param){
		.num_data = 256,
		.str = 1,
		.pad = 0,
		.shape = {256,1,1,1,1} // {Tr, Tc, Tn, Tm, T5}
    }
}, // end of FC0
{
    .mem = NORMAL,
	.exe = LAYER,
	
	.layer_id = 1,
	.layer_f = CORE,
	.core_func = r10_dense, // Exclusive to r10_dense
	.ifm = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
        .dataf = fc4_output_256,
		.ndim = 1,
		.num_data = 256,
		.shape = {256,1,1,1,1}
	},
	.ofm = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
        .dataf = fc4_output_256,
		.ndim = 1,
		.num_data = 256,
		.shape = {256,1,1,1,1}
	},
	.weights = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
		.ndim = 2,
		.num_data = 65792,
		.shape = {256,257,1,1,1}
	},
	.bias = (r10_tensor){
		.data_bin16 = 0,
		.dataf = fc4_bias,
		.ndim = 1,
		.num_data = 256,
		.shape = {256,1,1,1,1}
	},
	.workspace = (r10_tensor){
		.data_bin16 = 0,
		.dataf = fc_fwork_0,
		.ndim = 1,
		.num_data = 257,
		.shape = {257,1,1,1,1}
	},
    .t_param = (tiled_param){
		.num_data = 256,
		.str = 1,
		.pad = 0,
		.shape = {256,1,1,1,1} // {Tr, Tc, Tn, Tm, T5}
    }
}, // end of FC1
{
    .mem = NORMAL,
	.exe = LAYER,
	
	.layer_id = 1,
	.layer_f = CORE,
	.core_func = r10_dense, // Exclusive to r10_dense
	.ifm = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
        .dataf = fc4_output_256,
		.ndim = 1,
		.num_data = 256,
		.shape = {256,1,1,1,1}
	},
	.ofm = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
        .dataf = fc4_output_256,
		.ndim = 1,
		.num_data = 256,
		.shape = {256,1,1,1,1}
	},
	.weights = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
		.ndim = 2,
		.num_data = 65792,
		.shape = {256,257,1,1,1}
	},
	.bias = (r10_tensor){
		.data_bin16 = 0,
		.dataf = fc4_bias,
		.ndim = 1,
		.num_data = 256,
		.shape = {256,1,1,1,1}
	},
	.workspace = (r10_tensor){
		.data_bin16 = 0,
		.dataf = fc_fwork_0,
		.ndim = 1,
		.num_data = 257,
		.shape = {257,1,1,1,1}
	},
    .t_param = (tiled_param){
		.num_data = 256,
		.str = 1,
		.pad = 0,
		.shape = {256,1,1,1,1} // {Tr, Tc, Tn, Tm, T5}
    }
}, // end of FC2
{
    .mem = NORMAL,
	.exe = LAYER,
	
	.layer_id = 1,
	.layer_f = CORE,
	.core_func = r10_dense, // Exclusive to r10_dense
	.ifm = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
        .dataf = fc4_output_256,
		.ndim = 1,
		.num_data = 256,
		.shape = {256,1,1,1,1}
	},
	.ofm = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
        .dataf = fc4_output_256,
		.ndim = 1,
		.num_data = 12,
		.shape = {12,1,1,1,1}
	},
	.weights = (r10_tensor){
		.nvm_start = 0x00120000, // fake address
		.nvm_end = 0x00000000,
		.ndim = 2,
		.num_data = 3084,
		.shape = {12,257,1,1,1}
	},
	.bias = (r10_tensor){
		.data_bin16 = 0,
		.dataf = fc4_bias,
		.ndim = 1,
		.num_data = 12,
		.shape = {12,1,1,1,1}
	},
	.workspace = (r10_tensor){
		.data_bin16 = 0,
		.dataf = fc_fwork_0,
		.ndim = 1,
		.num_data = 257,
		.shape = {257,1,1,1,1}
	},
    .t_param = (tiled_param){
		.num_data = 12,
		.str = 1,
		.pad = 0,
		.shape = {12,1,1,1,1} // {Tr, Tc, Tn, Tm, T5}
    }
} // end of FC3
};

#pragma PERSISTENT(r10cnn_fc4)
r10cnn_model r10cnn_fc4={
	.num_layers = 4,
	.layers = fc4,
	.model_name = "fc4",
	.dnn = R10CNN
};
#endif /* FC4_H */