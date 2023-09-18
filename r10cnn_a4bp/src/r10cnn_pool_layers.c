#include "r10_cnn.h"

extern TickType_t begin, elapse;

void r10_global_avg_pool2d(size_t layer_id, exe_config *config, r10_tensor* ifm, r10_tensor* ofm)
{
    const size_t in_chan = ifm->shape[ifm->ndim-1];
    const float num_inv = 1.0f/(ifm->num_data/in_chan);

    enum precision prec = config->EXE_PRECISION;

    begin = xTaskGetTickCount();

    switch (prec)
    {
    case BINARY32:
        ifm->dataf = (float*)pvPortMalloc(ifm->num_data*sizeof(float));
        nvm_to_vm(ifm->dataf, ifm->num_data, ifm->nvm_start);

        // for (size_t i=0; i<ifm->num_data; i+=in_chan) {
        //     for (size_t j=0; j<in_chan; ++j) {
        //         ofm->dataf[j] 
        //         += ui32_to_fp32(*(uint32_t*)(ifm->nvm_start + ((i+j)*DATA_SIZE))) // ifm[i+j]
        //         *num_inv;
        //     }
        // }

        for (size_t i=0; i<ifm->num_data; i+=in_chan) {
            for (size_t j=0; j<in_chan; ++j) {
                ofm->dataf[j] 
                += ifm->dataf[i+j] // ifm[i+j]
                *num_inv;
            }
        }

        if(vm_to_nvm(ofm->dataf, ofm->num_data, ofm->nvm_start, &ofm->nvm_end) != 0){
            am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
        }

        if (layer_id > 0 && 
            (config->EXE_MODE == VANILLA)){
            vPortFree(ifm->dataf);
        }
    break;
        
    case BINARY16:
    // TODO
    break;
        
    default:
    break;
    }

    elapse = xTaskGetTickCount() - begin;
    am_util_stdio_printf("POOL Layer %ld: %ld\n", layer_id, elapse);

    return;
}
