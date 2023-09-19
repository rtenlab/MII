#include "r10_cnn.h" 

void a4p_show_r10cnn(struct r10cnn_model* r10cnn){
    am_util_stdio_printf("r10cnn Name = %s layer# = %ld\n", r10cnn->model_name, r10cnn->num_layers);
    
    for(size_t i=0;i<r10cnn->num_layers;i++){
        am_util_stdio_printf("layer id = %ld, ifm data# = %ld, ofm data# = %ld\n", 
        r10cnn->layers[i].layer_id, r10cnn->layers[i].ifm.num_data, r10cnn->layers[i].ofm.num_data);
    }

    return;
}

void a4p_show_r10cnn_layer(enum precision prec, struct r10cnn_layer* layer){
    
    am_util_stdio_printf("layer id = %ld, layer flag = %d\n", layer->layer_id, layer->layer_f);
    switch (prec)
    {
        case BINARY16:
            am_util_stdio_printf("\n=====================IFM=====================\n");
            for(size_t i=0;i<layer->ifm.num_data;i++){
                am_util_stdio_printf("%d, ", layer->ifm.data_bin16[i]);
                if((i+1)%4==0){
                    am_util_stdio_printf("\n");
                }
            }
            am_util_stdio_printf("\n===================weights===================\n");
            for(size_t i=0;i<layer->weights.num_data;i++){
                am_util_stdio_printf("%d, ", layer->weights.data_bin16[i]);
                if((i+1)%4==0){
                    am_util_stdio_printf("\n");
                }
            }
            am_util_stdio_printf("\n=====================OFM=====================\n");
            for(size_t i=0;i<layer->ofm.num_data;i++){
                am_util_stdio_printf("%d, ", layer->ofm.data_bin16[i]);
                if((i+1)%4==0){
                    am_util_stdio_printf("\n");
                }
            }
        am_util_stdio_printf("\n");
        break;
        
        case BINARY32:
            am_util_stdio_printf("\n=====================IFM=====================\n");
            for(size_t i=0;i<layer->ifm.num_data;i++){
                am_util_stdio_printf("%.8f, ", layer->ifm.dataf[i]);
                if((i+1)%4==0){
                    am_util_stdio_printf("\n");
                }
            }
            am_util_stdio_printf("\n===================weights===================\n");
            for(size_t i=0;i<layer->weights.num_data;i++){
                am_util_stdio_printf("%.8f, ", layer->weights.dataf[i]);
                if((i+1)%4==0){
                    am_util_stdio_printf("\n");
                }
            }
            am_util_stdio_printf("\n=====================OFM=====================\n");
            for(size_t i=0;i<layer->ofm.num_data;i++){
                am_util_stdio_printf("%.8f, ", layer->ofm.dataf[i]);
                if((i+1)%4==0){
                    am_util_stdio_printf("\n");
                }
            }
        am_util_stdio_printf("\n");
        break;
        
        default:
        break;
    }

    return;
}

size_t get_max_size_tensor(struct r10cnn_model r10cnn){
    size_t res = 0;

    for(size_t i=0;i<r10cnn.num_layers;i++){
        if(r10cnn.layers[i].ifm.num_data > res){
            res=r10cnn.layers[i].ifm.num_data;
        }
        if(r10cnn.layers[i].ofm.num_data > res){
            res=r10cnn.layers[i].ofm.num_data;
        }
        if(r10cnn.layers[i].weights.num_data > res){
            res=r10cnn.layers[i].weights.num_data;
        }
        if(r10cnn.layers[i].bias.num_data > res){
            res=r10cnn.layers[i].bias.num_data;
        }
    }
    
    return res;
}

size_t max_in_float_array(float* array, size_t size){
    size_t res = 0;
    float val = 0;
    for(size_t i=0;i<size;i++){
        if(array[i]>val){
            res = i;
            val = array[i];
        }
    }

    return res;
}

size_t max_in_bin16_array(bin16* array, size_t size){
    size_t res = 0;
    bin16 val = 0;
    for(size_t i=0;i<size;i++){
        if(array[i]>val){
            res = i;
            val = array[i];
        }
    }

    return res;
}

size_t max_in_r10_tensor(r10_tensor* r10tensor, size_t size){

    size_t res = 0;

    res = max_in_float_array(r10tensor->dataf, r10tensor->num_data);

    return res;
}

void a4p_print_config(struct exe_config* config){

    am_util_stdio_printf("EXE_MODE 1-LAYER | 2-TILED..................%d\n", 
        config->EXE_MODE);
    am_util_stdio_printf("Period......................................%d\n", 
        config->T_ms);

    return;
}

/**
 * Adds bias vector b to tensor A.
 * assumes b is a rank 1 tensor that is added to the last dimension of A.
 *
 * @A: input tensor. Overwritten with outputs.
 * @b: bias tensor.
 */
void r10_bias_add(r10_tensor* A, const r10_tensor* b) 
{


        for (size_t i=0; i<A->num_data; i+=b->num_data) {
            for (size_t j=0; j<b->num_data; ++j) {
                A->dataf[i+j] += b->dataf[j];
            }
        }

    return;
}

void a4p_print_float_array(float *pBuf, size_t size) {
    if(size<16){
        am_util_stdio_printf("0x%08X: ", pBuf);
        for ( size_t i = 0; i < size; i++ ){
            am_util_stdio_printf("%08f ", pBuf[i]);
        }
        am_util_stdio_printf("\n");
        return;
    }
    for ( size_t i = 0; i < size/16; i++ )
    {
        am_util_stdio_printf("0x%08X: ", pBuf + i);
        for (uint32_t j = 0; j < 4; j++)
        {
            // am_util_stdio_printf("0x%08X ", pBuf[i * 4 + j]);
            am_util_stdio_printf("%08f ", pBuf[i * 4 + j]);
        }
        am_util_stdio_printf("\n");
    }
    return;
}

void a4p_print_uint32t_array(uint32_t *pBuf, size_t size) {
    if(size<16){
        am_util_stdio_printf("0x%08X: ", pBuf);
        for ( size_t i = 0; i < size; i++ ){
            am_util_stdio_printf("%08X ", pBuf[i]);
        }
        am_util_stdio_printf("\n");
        return;
    }
    for ( size_t i = 0; i < size/16; i++ )
    {
        am_util_stdio_printf("0x%08X: ", pBuf + i);
        for (uint32_t j = 0; j < 4; j++)
        {
            // am_util_stdio_printf("0x%08X ", pBuf[i * 4 + j]);
            am_util_stdio_printf("%08X ", pBuf[i * 4 + j]);
        }
        am_util_stdio_printf("\n");
    }
    return;
}

/**
 * tile_valid - determine if t_param is valid in the given layer
 * @filename: file to read from. Assumed comma separated ascii text.
 * @array_size: how many values to read from the file.
 * 
 * Return: pointer to allocated array.
 */
int tile_valid(r10cnn_layer *layer)
{
    if
     ( ((layer->ofm.shape[0]     % layer->t_param.shape[0]) !=0) || 
      ((layer->ofm.shape[1]     % layer->t_param.shape[1]) !=0) || 
      ((layer->weights.shape[3] % layer->t_param.shape[3]) !=0) || 
      ((layer->ifm.shape[2]     % layer->t_param.shape[2]) !=0) ) { 

      am_util_stdio_printf("_tiled_forward_propagation ERROR: tile params doesn't fit %ld:\
        %ld|%ld, %ld|%ld, %ld|%ld, %ld|%ld\n",
        layer->layer_id,
        layer->ofm.shape[0], layer->t_param.shape[0],
        layer->ofm.shape[1], layer->t_param.shape[1],
        layer->weights.shape[3], layer->t_param.shape[3],
        layer->ifm.shape[2], layer->t_param.shape[2]
      );
      return -1;
    }
    return 0;
}

int check_r10cnn(struct exe_config *config, struct r10cnn_model *r10cnn){
    struct r10cnn_layer *layer;
    if(config->EXE_MODE == LAYER){
        am_util_stdio_printf("LAYER: nothing in LAYER r10cnn to be check\n");
    }
    if(config->EXE_MODE == VANILLA){
        am_util_stdio_printf("VANILLA: nothing in VANILLA r10cnn to be check\n");
    }
    if (config->EXE_MODE == SINGLE){
        am_util_stdio_printf("SINGLE: nothing in SINGLE r10cnn to be check\n");
    }
    if (config->EXE_MODE == FILTER){
        am_util_stdio_printf("FILTER: nothing in FILTER r10cnn to be check\n");
    }
    if(config->EXE_MODE == TILED){
        for(size_t i=0;i<r10cnn->num_layers;i++){
            layer = &r10cnn->layers[i];
            if (tile_valid(layer) != 0){
                am_util_stdio_printf("TILED: Tile size not matched to layer\n");
                return 1;
            }
        }
        am_util_stdio_printf("TILED: r10cnn Valid\n");
    }
    return 0;
}

uint32_t fp32_to_ui32(float n)
{
    return (uint32_t)(*(uint32_t*)&n);
}
 
float ui32_to_fp32(uint32_t n)
{
    return (float)(*(float*)&n);
}

int vm_to_nvm(float* data, size_t num_data, uint32_t begin_address, uint32_t *end_address){
	int error_flag = 0;
	uint32_t *nvm_address = (uint32_t *)begin_address;
	uint32_t temp_uint32[num_data];
	
	
	for(int i=0;i<num_data;i++){
		temp_uint32[i] = fp32_to_ui32(data[i]);
	}
    
	error_flag = 
    am_hal_mram_main_program(AM_HAL_MRAM_PROGRAM_KEY,
                                temp_uint32,
                                nvm_address,
                                num_data * DATA_SIZE);
    
    if (error_flag != 0){
        am_util_stdio_printf("vm_to_nvm error_flag: %d\n", error_flag);
    }
	
	*end_address = begin_address+((num_data-1)*DATA_SIZE);
    
    return error_flag;
}

int validate_nvm(float* data, size_t num_data, uint32_t begin_address){
    int error_flag = 0;

    for ( size_t ix = 0; ix < num_data; ix++ )
    {
        // am_util_stdio_printf("Checking nvm_address 0x%08x data checkpointed!\n", (begin_address + (ix * DATA_SIZE)));
        if ( ui32_to_fp32(*(uint32_t*)(begin_address + (ix*DATA_SIZE))) != data[ix] )
        {
            am_util_stdio_printf("ERROR: MRAM address 0x%08x did not program properly:\n"
                                 "  Expected value = 0x%08x, programmed value = 0x%08x.\n",
                                 begin_address + (ix * DATA_SIZE),
                                 data[ix],
                                 *(uint32_t*)(begin_address + (ix * DATA_SIZE)) );
            error_flag++;
        }
    }

    return error_flag;
}

int nvm_to_vm(float* data, size_t num_data, uint32_t begin_address){
	int error_flag = 0;

	for ( size_t ix = 0; ix < num_data; ix++ )
    {
        data[ix] = ui32_to_fp32(*(uint32_t*)(begin_address + (ix*DATA_SIZE)));
    }
	
	return error_flag;
}

int check_info0(uint32_t* exe_status, size_t num_data){
    int error_flag = 0;
    uint32_t ui32Info0ReadBack[5];
    //
    // Check INFO0 just programmed.
    //
    am_util_stdio_printf("  ... verifying the INFO0 just programmed.\n");
#if defined(AM_PART_APOLLO4B) || defined(AM_PART_APOLLO4L)
    if (0 != am_hal_mram_info_read(0, 0, num_data, &ui32Info0ReadBack[0]))
    {
      am_util_stdio_printf("ERROR: INFO0 Read Back failed\n");
      error_flag++;
    }
#endif
    for ( size_t ix = 0; ix < (sizeof(*exe_status) / sizeof(uint32_t)); ix++ )
    {
#if defined(AM_PART_APOLLO4)
        if ( *(uint32_t*)(ui32Info0Addr + (ix * DATA_SIZE)) != ui32Info0[ix] )
        {
            am_util_stdio_printf("ERROR: INFO0 address 0x%08x did not program properly:\n"
                                 "  Expected value = 0x%08x, programmed value = 0x%08x.\n",
                                 ui32Info0Addr + (ix * DATA_SIZE),
                                 ui32Info0[ix],
                                 *(uint32_t*)(ui32Info0Addr + (ix * DATA_SIZE)) );
            error_flag++;
            break;
        }
#elif defined (AM_PART_APOLLO4B) || defined(AM_PART_APOLLO4L)
        if ( ui32Info0ReadBack[ix] != ui32Info0[ix] )
        {
            am_util_stdio_printf("ERROR: INFO0 address 0x%08x did not program properly:\n"
                                 "  Expected value = 0x%08x, programmed value = 0x%08x.\n",
                                 ui32Info0Addr + (ix * DATA_SIZE),
                                 ui32Info0[ix],
                                 *(uint32_t*)(ui32Info0Addr + (ix * DATA_SIZE)) );
            error_flag++;
            break;
        }
#endif
    }
    return error_flag;
}

void a4p_show_exe_status(uint32_t* exe_status){
    am_util_stdio_printf("exe_status={0x%08X, 0x%08X, 0x%08X, 0x%08X, 0x%08X}\n", 
                        exe_status[0], exe_status[1], exe_status[2], exe_status[3],
                        exe_status[4]);
    return;
}

int clear_exe_status(/*Need to aloocate for different r10cnn*/){
    int error_flag = 0;
    uint32_t exe_status[10] = 
            {0x00000000, 0x00000000, 
            0x00000000, 0x00000000,  
            0x00000000, 0x00000000,
            0x00000000, 0x00000000,  
            0x00000000, 0x00000000};
    error_flag = am_hal_mram_info_program(AM_HAL_MRAM_INFO_KEY,
                            exe_status,
                            0,
                            (sizeof(exe_status) / sizeof(uint32_t)));
    return error_flag;
}