#ifndef R10_CNN_H
#define R10_CNN_H

/*******************************************************************
 * C Standard Library includes
 *******************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*******************************************************************
 * a4p Standard AmbiqSuite includes
 *******************************************************************/
#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"

/*******************************************************************
 * a4p hardware
 *******************************************************************/
// ADC input pin constant GPIO
#define ADC_INPUT_PIN           18
#define ADC_CONVERT             5.12f // from 0-1000mV to 2.87-4.04V
void a4p_setup(void *pvParameters);
float adc_read(void);
// void a4p_setup();

/*******************************************************************
 * a4p memory map
 *******************************************************************/
#include "am_memory_map.h" // memory reference

/*******************************************************************
 * FreeRTOS include
 *******************************************************************/
#include "FreeRTOS.h" // include FreeRTOSConfig.h
#include "task.h"
#include "portmacro.h"
#include "portable.h"
#include "semphr.h"
#include "event_groups.h"

/*******************************************************************
 * r10 cnn all data types
 *******************************************************************/
#include "r10cnn_types.h"

/*******************************************************************
 * r10 cnn all functions
 *******************************************************************/
#include "r10cnn_funcs.h"

/*******************************************************************
 * All configs settings
 *******************************************************************/
#include "r10cnn_config.h"

#endif //R10_CNN_H