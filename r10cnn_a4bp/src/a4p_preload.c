#include "../../r10cnn_zoo/inas_original.h" // for weights as static
#include "../../r10cnn_zoo/inas_12.h"
#include "../../r10cnn_zoo/mnist_inas.h" 
// #include "../../../r10cnn_zoo/inas_original_nvm.h" // for weights in nvm, must be run after preload

#include "../../r10cnn_zoo/exp1_configs.h"

uint32_t
write_weights(struct r10cnn_model *r10cnn, uint32_t begin_address){
    uint32_t end_address = 0x0;
    int32_t error_code = 0;

    struct r10cnn_layer *layer;

    am_util_stdio_printf("r10cnn: %s\n", r10cnn->model_name);

    // process all layers weights
    for(size_t i=0;i<r10cnn->num_layers;i++){
        layer = &r10cnn->layers[i];

        // write sequence
        switch (layer->layer_f)
        {
        case CONV:
        error_code = 
        vm_to_nvm(layer->weights.dataf, layer->weights.num_data, begin_address, &end_address);
        am_util_stdio_printf("layer %ld: .nvm_start = 0x%08X, .nvm_end = 0x%08X,\n", 
                            layer->layer_id, begin_address, end_address);
        break;

        case POOLING:
        // No weights for POOLING
        break;

        case CORE:
        // TODO: write weights
        break;
        
        default:
        break;
        }

        // update begin_address of next layer to the end_address of the this layer + DATA_SIZE
        begin_address = end_address + DATA_SIZE;

    }

    return end_address;
}

uint32_t
write_input(struct r10cnn_model *r10cnn, uint32_t begin_address){
    uint32_t end_address = 0x0;
    int32_t error_code = 0;

    // get first layer with ifm
    struct r10cnn_layer *layer = &r10cnn->layers[0];

    am_util_stdio_printf("r10cnn: %s input\n", r10cnn->model_name);

    vm_to_nvm(layer->ifm.dataf, layer->ifm.num_data, begin_address, &end_address);
    am_util_stdio_printf("input: .nvm_start = 0x%08X, .nvm_end = 0x%08X,\n", 
                        begin_address, end_address);

    return end_address;
}

//*****************************************************************************
//
// Main function.
//
//*****************************************************************************
int
main(void)
{
    uint32_t ui32PrgmAddr, ui32EndAddr;
    
    //
    // Set the default cache configuration
    //
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();

    //
    // Initialize the peripherals for this board.
    //
    am_bsp_low_power_init();

    //
    // Enable printing through the UART interface.
    //
    am_bsp_uart_printf_enable();

    //
    // Clear the terminal and print the banner.
    //
    am_util_stdio_terminal_clear();
    
    // print one model config
    // a4p_print_config(&a4p_exp1_configs.configs[0]);

    ui32PrgmAddr = ARB_ADDRESS; // 0x00100000 = 1024 * 1024 * 1 = 1MB

    // write weights: NEED TO CHANGE IF r10cnn_model is changed ---------
    ui32EndAddr = write_weights(&r10cnn_inas_original, ui32PrgmAddr);
    ui32PrgmAddr = ui32EndAddr + DATA_SIZE;

    ui32EndAddr = write_weights(&r10cnn_inas_12, ui32PrgmAddr);
    ui32PrgmAddr = ui32EndAddr + DATA_SIZE;

    ui32EndAddr = write_weights(&r10cnn_mnist_inas, ui32PrgmAddr);
    ui32PrgmAddr = ui32EndAddr + DATA_SIZE;
    // ------------------------------------------------------------------

    // am_util_stdio_printf("data value read weights[0]: %f\n", 
    // ui32_to_fp32(*(uint32_t*)(ui32PrgmAddr + (0*DATA_SIZE)))
    // );

    am_util_stdio_printf("----------------------------------------\n");

    // write input data (ifm)
    ui32EndAddr = write_input(&r10cnn_inas_original, ui32PrgmAddr);
    ui32PrgmAddr = ui32EndAddr + DATA_SIZE;

    ui32EndAddr = write_input(&r10cnn_inas_12, ui32PrgmAddr);
    ui32PrgmAddr = ui32EndAddr + DATA_SIZE;

    ui32EndAddr = write_input(&r10cnn_mnist_inas, ui32PrgmAddr);
    ui32PrgmAddr = ui32EndAddr + DATA_SIZE;

    // write all init states (all 0s) to INFO0
    am_hal_mram_info_program(AM_HAL_MRAM_INFO_KEY,
                            a4p_exp1_configs.configs[0].exe_status,
                            0,
                            (sizeof(a4p_exp1_configs.configs[0].exe_status) / sizeof(uint32_t)));
    
    // check if the exe_status is written correctly
    check_info0(a4p_exp1_configs.configs[0].exe_status, 5);

    am_util_stdio_printf("infernece features start address: 0x%08X\n", 
                        ui32PrgmAddr);

    while(1);
}