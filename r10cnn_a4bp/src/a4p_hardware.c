#include "r10_cnn.h"

//
// Pin configuration
//
#if ADC_INPUT_PIN == 18
const am_hal_gpio_pincfg_t g_ADC_PIN_CFG =
{
    .GP.cfg_b.uFuncSel       = AM_HAL_PIN_18_ADCSE1,
};
#else
#error Must set up a pin config for the pin that is to be used for voltage input
#endif
// Global ADC Device Handle.
static void                  *g_ADCHandle;
// Global Sample Count semaphore from ADC ISR to base level.
uint32_t                     g_ui32SampleCount = 0;
// Set up an array to use for retrieving the correction trim values.
float g_fTrims[4];
// Global Variable for ADC voltage
// init above JIT thd to avoid startup shutdown
float adc_voltage = 10000.0f;

#define JIT_THD 3300 // RTEN: need to remove, in mV

extern struct r10cnn_layer *layer;
extern size_t x0, x1, z0, z1, q, k;
// extern uint32_t *curr_exe_status;

//*****************************************************************************
//
// ADC Interrupt Service Routine (ISR)
//
//*****************************************************************************
void
am_adc_isr(void)
{
    uint32_t ui32IntStatus;

    //
    // Clear the ADC interrupt.
    //
    am_hal_adc_interrupt_status(g_ADCHandle, &ui32IntStatus, true);
    am_hal_adc_interrupt_clear(g_ADCHandle, ui32IntStatus);

    //
    // Keep grabbing samples from the ADC FIFO until it goes empty.
    //
    uint32_t ui32NumSamples = 1;
    am_hal_adc_sample_t sSample;

    //
    // Get samples until the FIFO is emptied.
    //
    while ( AM_HAL_ADC_FIFO_COUNT(ADC->FIFO) )
    {
        ui32NumSamples = 1;

        //
        // Invalidate DAXI to make sure CPU sees the new data when loaded.
        //
        am_hal_daxi_control(AM_HAL_DAXI_CONTROL_INVALIDATE, NULL);
        am_hal_adc_samples_read(g_ADCHandle, false, NULL, &ui32NumSamples, &sSample);
        
        adc_voltage = (float)(sSample.ui32Sample  * AM_HAL_ADC_VREFMV / 0x1000);
			  
    }

    adc_voltage *= ADC_CONVERT;

    // am_util_stdio_printf("ADC READING: %f!\n", adc_voltage);

/*
    // To JIT <RTEN>
    if(adc_voltage < JIT_THD){
        am_util_stdio_printf("%f Below JIT! JIT kicks in Layer: %ld\n", adc_voltage, layer->layer_id);
        // curr_exe_status[2] = (uint32_t)layer->layer_id+1; // store the correct label into exe_status[1]
        // am_hal_mram_info_program(AM_HAL_MRAM_INFO_KEY,
        //                             curr_exe_status,
        //                             0,
        //                             (sizeof(*curr_exe_status) / sizeof(uint32_t)));

        // if (layer->layer_id > 0){
        //     if(vm_to_nvm(layer->ifm.dataf, layer->ifm.num_data, layer->ifm.nvm_start, &layer->ifm.nvm_end) != 0){
        //         am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
        //     }
        // }

        if(vm_to_nvm(layer->ifm.dataf, layer->ifm.num_data, layer->ifm.nvm_start, &layer->ifm.nvm_end) != 0){
            am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
        }
        
        if(vm_to_nvm(layer->ofm.dataf, layer->ofm.num_data, layer->ofm.nvm_start, &layer->ofm.nvm_end) != 0){
            am_util_stdio_printf("ERROR! vm_to_nvm return non-zero\n");
        }

        am_util_stdio_printf("x0: %ld, x1: %ld, z0: %ld, z1: %ld, q: %ld, k: %ld\n", x0, x1, z0, z1, q, k);

        uint32_t curr_exe_status[9] = { 
                    1, // exe_status[0] = 1
                    0, // exe_status[1] = 0
                    layer->layer_id,
                    x0,
                    x1,
                    z0,
                    z1,
                    q,
                    k,
                };
        if(0 != am_hal_mram_info_program(AM_HAL_MRAM_INFO_KEY,
                                            curr_exe_status,
                                            0,
                                           (sizeof(curr_exe_status) / sizeof(uint32_t))))
        {
            am_util_stdio_printf("ERROR! am_hal_mram_info_program return non-zero\n");
        }


        am_util_stdio_printf("JIT Completed!\n");
        
        // am_hal_sysctrl_fpu_disable();
        // am_bsp_uart_printf_disable();
        // am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
        
        while(1);
    }
    // END TO JIT </RTEN>
*/


    //
    // Signal interrupt arrival to base level.
    //
    g_ui32SampleCount++;
}

//*****************************************************************************
//
// ADC Force Read Function
//
//*****************************************************************************
float
adc_read(void){

    //
    // Keep grabbing samples from the ADC FIFO until it goes empty.
    //
    uint32_t ui32NumSamples = 1;
    am_hal_adc_sample_t sSample;
    float ret = 0.0;

    //
    // Get samples until the FIFO is emptied.
    //
    while ( AM_HAL_ADC_FIFO_COUNT(ADC->FIFO) )
    {
        ui32NumSamples = 1;

        //
        // Invalidate DAXI to make sure CPU sees the new data when loaded.
        //
        am_hal_daxi_control(AM_HAL_DAXI_CONTROL_INVALIDATE, NULL);
        am_hal_adc_samples_read(g_ADCHandle, false, NULL, &ui32NumSamples, &sSample);
			  
        ret = (float)(sSample.ui32Sample  * AM_HAL_ADC_VREFMV / 0x1000);
    }

    ret *= ADC_CONVERT;

    // am_util_stdio_printf("ADC READING: %f!\n", ret);

    //
    // Signal interrupt arrival to base level.
    //
    g_ui32SampleCount++;
	
    return ret;
}

//*****************************************************************************
//
// ADC INIT Function
//
//*****************************************************************************
void
adc_init(void)
{
    am_hal_adc_config_t           ADCConfig;
    am_hal_adc_slot_config_t      ADCSlotConfig;

    //
    // Initialize the ADC and get the handle.
    //
    if ( am_hal_adc_initialize(0, &g_ADCHandle) != AM_HAL_STATUS_SUCCESS )
    {
        am_util_stdio_printf("Error - reservation of the ADC instance failed.\n");
    }

    //
    // Get the ADC correction offset and gain for this DUT.
    // Note that g_fTrims[3] must contain the value -123.456f before calling
    // the function.
    // On return g_fTrims[0] contains the offset, g_fTrims[1] the gain.
    //
    g_fTrims[0] = g_fTrims[1] = g_fTrims[2] = 0.0F;
    g_fTrims[3] = -123.456f;
    am_hal_adc_control(g_ADCHandle, AM_HAL_ADC_REQ_CORRECTION_TRIMS_GET, g_fTrims);
    

    //
    // Print ADC Correction Details - uncomment to enable
    //
    // am_util_stdio_printf(" ADC correction offset = %.6f\n", g_fTrims[0]);
    // am_util_stdio_printf(" ADC correction gain   = %.6f\n", g_fTrims[1]);

    //
    // Power on the ADC.
    //
    if (AM_HAL_STATUS_SUCCESS != am_hal_adc_power_control(g_ADCHandle,
                                                          AM_HAL_SYSCTRL_WAKE,
                                                          false) )
    {
        am_util_stdio_printf("Error - ADC power on failed.\n");
    }

    //
    // Set up internal repeat trigger timer
    //
    am_hal_adc_irtt_config_t      ADCIrttConfig =
    {
        .bIrttEnable        = true,
        .eClkDiv            = AM_HAL_ADC_RPTT_CLK_DIV16,
        .ui32IrttCountMax   = 30,
    };

    am_hal_adc_configure_irtt(g_ADCHandle, &ADCIrttConfig);

    //
    // Set up the ADC configuration parameters. These settings are reasonable
    // for accurate measurements at a low sample rate.
    //
    ADCConfig.eClock             = AM_HAL_ADC_CLKSEL_HFRC_24MHZ;
    ADCConfig.ePolarity          = AM_HAL_ADC_TRIGPOL_RISING;
    ADCConfig.eTrigger           = AM_HAL_ADC_TRIGSEL_SOFTWARE;
    ADCConfig.eClockMode         = AM_HAL_ADC_CLKMODE_LOW_LATENCY;
    ADCConfig.ePowerMode         = AM_HAL_ADC_LPMODE0;
    ADCConfig.eRepeat            = AM_HAL_ADC_REPEATING_SCAN;
    ADCConfig.eRepeatTrigger     = AM_HAL_ADC_RPTTRIGSEL_INT;
    if ( am_hal_adc_configure(g_ADCHandle, &ADCConfig) != AM_HAL_STATUS_SUCCESS )
    {
        am_util_stdio_printf("Error - configuring ADC failed.\n");
    }

    //
    // Set up an ADC slot
    //

    //! Set additional input sampling ADC clock cycles
    ADCSlotConfig.eMeasToAvg      = AM_HAL_ADC_SLOT_AVG_128;
    ADCSlotConfig.ui32TrkCyc      = AM_HAL_ADC_MIN_TRKCYC;
    ADCSlotConfig.ePrecisionMode  = AM_HAL_ADC_SLOT_12BIT;
    ADCSlotConfig.eChannel        = AM_HAL_ADC_SLOT_CHSEL_SE1;
    ADCSlotConfig.bWindowCompare  = false;
    ADCSlotConfig.bEnabled        = true;
    if (AM_HAL_STATUS_SUCCESS != am_hal_adc_configure_slot(g_ADCHandle, 0, &ADCSlotConfig))
    {
        am_util_stdio_printf("Error - configuring ADC Slot 0 failed.\n");
    }

    //
    // For this example, the samples will be coming in slowly. This means we
    // can afford to wake up for every conversion.
    //
    am_hal_adc_interrupt_enable(g_ADCHandle, AM_HAL_ADC_INT_FIFOOVR1 | AM_HAL_ADC_INT_DERR | AM_HAL_ADC_INT_DCMP | AM_HAL_ADC_INT_CNVCMP | AM_HAL_ADC_INT_SCNCMP );

    //
    // Enable the ADC.
    //
    if ( am_hal_adc_enable(g_ADCHandle) != AM_HAL_STATUS_SUCCESS )
    {
        am_util_stdio_printf("Error - enabling ADC failed.\n");
    }

    //
    // Enable internal repeat trigger timer
    //
    am_hal_adc_irtt_enable(g_ADCHandle);

}


//*****************************************************************************
//
// High priority task to run immediately after the scheduler starts.
//
// This task is used for any global initialization that must occur after the
// scheduler starts, but before any functional tasks are running. This can be
// useful for enabling events, semaphores, and other global, RTOS-specific
// features.
//
//*****************************************************************************
void
a4p_setup(void *pvParameters)
// a4p_setup()
{	
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
    // Enable printing through the ITM or UART interface.
    //
    // am_bsp_itm_printf_enable();
    am_bsp_uart_printf_enable();
        
    am_util_stdio_terminal_clear();

    // a4p_print_config(&config_inas_original);
    // am_util_stdio_printf("\rSetup Complete - begin infernece\n\r");
    // am_util_stdio_printf(" Use pin %d for input.\n", ADC_INPUT_PIN);
    // am_util_stdio_printf(" The applied voltage should be between 0 and 1.1v.\n");

    //
    // Enable floating point.
    //
    am_hal_sysctrl_fpu_enable();
    am_hal_sysctrl_fpu_stacking_enable(true);


    // <RTEN> Init ADC and Enable interrupts.
    //
    // Set a pin to act as our ADC input
    //
    am_hal_gpio_pinconfig(ADC_INPUT_PIN, g_ADC_PIN_CFG);

	//
    // Initialize the ADC.
    //
    adc_init();

/*
    //
    // Enable interrupts for the ADC.
    //
    NVIC_SetPriority(ADC_IRQn, AM_IRQ_PRIORITY_DEFAULT);
    NVIC_EnableIRQ(ADC_IRQn);
    am_hal_interrupt_master_enable();
    
    //
    // Kick Start Repeat with an ADC software trigger in REPEAT mode.
    //
    am_hal_adc_sw_trigger(g_ADCHandle);
    // </RTEN>
*/

    //
    // Reset the sample count which will be incremented by the ISR.
    //
    g_ui32SampleCount = 0;

    // am_util_stdio_printf("\rSetup Completed\n");
        
    //
    // The setup operations are complete, so suspend the setup task now.
    //
    vTaskSuspend(NULL);

    while (1);
}