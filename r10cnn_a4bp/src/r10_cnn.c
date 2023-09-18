#include "r10_cnn.h"

/*******************************************************************
 * Global Variables for r10cnn infernece
 *******************************************************************/
struct r10cnn_layer *layer;
int32_t counter;
// uint32_t *curr_exe_status;
TickType_t begin, elapse; // time keeping global variables

/**
 * _layer_forward_propagation - r10cnn LAYER Mode
 * forward propagation for the r10cnn inference tasks
 * @input_tensor: r10_tensor that contains first layer IFM
 * @r10cnn: r10cnn_model obtained from r10cnn_model header file
 * 
 * Return: 0 on success. An error code otherwise
 */
int _forward_propagation(struct exe_config *config, struct r10_tensor input_tensor, struct r10cnn_model *r10cnn){
  // struct r10cnn_layer *layer;
  size_t i;
//==================================<RTEN>============================================
  i = config->exe_status[2]; // readback layer_id
  counter = 0; // REMOVE
//=================================</RTEN>============================================
  int res_label = -1;

  for(;i<r10cnn->num_layers;i++){
  // for(size_t i=0;i<5;i++){
    layer = &r10cnn->layers[i];
    
    // layer function
    switch (layer->layer_f)
    {
    case CONV:
      layer ->conv_func(config, layer);
    break;

    case POOLING:
      layer ->pooling_func(layer->layer_id, config,
      &layer->ifm, &layer->ofm);
    break;

    case CORE:
      layer ->core_func(layer->layer_id, config,
      &layer->weights, &layer->bias, &layer->ifm, &layer->ofm,
      &layer->workspace);
    break;
    
    default:
      break;
    }

    // link the completed ofm to next ifm
    if(layer->layer_id != r10cnn->num_layers-1){
      r10cnn->layers[i+1].ifm = layer->ofm; // Volatile, lose after reboot
    }
    // DEBUG: print out the last layer ofm
    // else{
    //   a4p_print_float_array(layer->ofm.dataf, 10);
    // }

    config->exe_status[2] = (uint32_t)i+1; // store the correct label into exe_status[1]
    config->exe_status[3] = 0; // reset exe_status[3] to 0
    config->exe_status[4] = 0; // reset exe_status[4] to 0
    config->exe_status[5] = 0; // reset exe_status[3] to 0
    config->exe_status[6] = 0; // reset exe_status[4] to 0
    if(0 != am_hal_mram_info_program(AM_HAL_MRAM_INFO_KEY,
      config->exe_status,
      0,
      (sizeof(config->exe_status) / sizeof(uint32_t))))
    {
        am_util_stdio_printf("ERROR! am_hal_mram_info_program return non-zero\n");
    }

    // am_util_stdio_printf("ADC Read: %.5fmV\n", adc_read());
  }

  res_label = max_in_r10_tensor(config->EXE_PRECISION, &layer->ofm, layer->ofm.num_data);

  return res_label;
}

/**
 * r10cnn_drive - main r10cnn driver
 * @config: exe_config obtained from r10cnn_model header file
 * @r10cnn: r10cnn_model obtained from r10cnn_model header file
 * 
 * Return: 0 on success. An error code otherwise
 */
int r10cnn_driver(struct exe_config *config, struct r10cnn_model *r10cnn, float out_array[10]){

  // show_r10cnn(r10cnn);
  // check_r10cnn(config, r10cnn);

  size_t curr_label=0;
  struct r10_tensor input_tensor;

  int res_label=-1;

//==================================<RTEN>============================================
  // read back INFO0 for the r10cnn
  if (0 != am_hal_mram_info_read(0, 0, 10, &config->exe_status[0]))
  {
      while(1); // ERROR: info0 read unsuccessful
  }

  // a4p_show_exe_status(config->exe_status);
  if (config->exe_status[0] == 0x00000000){
    curr_label = max_in_float_array(out_array, 10);
    // am_util_stdio_terminal_clear();
    am_util_stdio_printf("Correct Label: %ld\n", curr_label);
    am_util_stdio_printf("MEM_MODE(Norm|XIP): %ld\n", config->MEM_MODE);
    am_util_stdio_printf("EXE_MODE(V|L|T|F|S): %ld\n", config->EXE_MODE);
    config->exe_status[0] = 0x00000001; // indicate the inference commence by exe_status[0]=1
    config->exe_status[1] = curr_label; // store the correct label into exe_status[1]
    // config->exe_status[2] = 0; // reset layer_id to 0
    // config->exe_status[3] = 0; // reset x0 to 0
    // config->exe_status[4] = 0; // reset x1 to 0
    am_hal_mram_info_program(AM_HAL_MRAM_INFO_KEY,
                            config->exe_status,
                            0,
                            (sizeof(config->exe_status) / sizeof(uint32_t)));
  }
  else{
    curr_label = config->exe_status[1];
  }
  
  // check_info0(config->exe_status, 5);
//=================================</RTEN>============================================

  input_tensor = r10cnn->layers[0].ifm;

  res_label = _forward_propagation(config, input_tensor, r10cnn);
  if(res_label == -1){
    am_util_stdio_printf("_layer_driver ERROR: _layer_forward_propagation return -1");
    return -1;
  }
  am_util_stdio_printf("Result Label: %d\n", res_label);

  return 0;
}