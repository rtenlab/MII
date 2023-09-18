#include "../../r10cnn_zoo/inas_original_nvm.h"
#include "../../r10cnn_zoo/inas_12_nvm.h"
#include "../../r10cnn_zoo/mnist_inas_nvm.h"

#include "../../r10cnn_zoo/exp1_configs.h"

#include "r10_cnn.h"

//*****************************************************************************
//
// Task handles
//
//*****************************************************************************
TaskHandle_t xSetupTask;
TaskHandle_t xFooTask;

/* Structure that will hold the TCB of the task being created. */
StaticTask_t xFooTCB;

/* Buffer that the task being created will use as its stack.  Note this is
an array of StackType_t variables.  The size of StackType_t is dependent on
the RTOS port. */
StackType_t xFooStack[ STACK_SIZE ];

void FooTask(){
    TickType_t start, C_ticks;
    // for(;;){
        start = xTaskGetTickCount();

        r10cnn_driver(&a4p_exp1_configs.configs[0], &r10cnn_inas_original, output0); // inas_original
        // r10cnn_driver(&a4p_exp1_configs.configs[1], &r10cnn_inas_12, output1); // inas_12
        // r10cnn_driver(&a4p_exp1_configs.configs[2], &r10cnn_mnist_inas, mnist_output0);

        C_ticks = xTaskGetTickCount() - start;

        am_util_stdio_printf("FooTask Ticks Difference: %ldms\n", C_ticks);
//==================================<RTEN>============================================
        if (0 != clear_exe_status()){
            am_util_stdio_printf("FooTask: Error in clearing execution status\n");
        }
        
        // am_hal_sysctrl_fpu_disable();
        // am_bsp_uart_printf_disable();

        while(1);
        /*
        am_util_stdio_printf("Enterring Deep Sleep...\n");
        while (1)
        {
            //
            // Go to Deep Sleep until wakeup.
            //
            am_hal_sysctrl_sleep(AM_HAL_SYSCTRL_SLEEP_DEEP);
        }
        */
        
//=================================</RTEN>============================================
        // vTaskDelay(a4p_exp1_configs.configs[0].T_ms / portTICK_PERIOD_MS);
    // }
}

/**
 * Main Function - Tester Program for layer DNN
 * 
 * Return: 0 on success. An error code otherwise
 */
int
main(void){

    //
    // Create setup tasks.
    //
    xTaskCreate(
                a4p_setup, 
                "a4p_setup", 
                512, 
                0, 
                99, // make sure the setup task is highest priority
                // 0, // preemptable - 1, nonpreemptable - 0
                &xSetupTask);
    
    xFooTask = xTaskCreateStatic(
                    FooTask,
                    "FooTask",
                    STACK_SIZE,
                    0,
                    a4p_exp1_configs.configs[0].priority,
                    xFooStack,
                    &xFooTCB);
    
    //
    // Start the scheduler.
    //
    vTaskStartScheduler();

    
    return 0;
}