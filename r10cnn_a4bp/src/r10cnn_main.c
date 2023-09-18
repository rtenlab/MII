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
TaskHandle_t xCIFAR10L7Task;
TaskHandle_t xCIFAR10L12Task;
TaskHandle_t xMNISTL7Task;

/* Structure that will hold the TCB of the task being created. */
StaticTask_t xCIFAR10L7TCB;
StaticTask_t xCIFAR10L12TCB;
StaticTask_t xMNISTL7TCB;

/* Buffer that the task being created will use as its stack.  Note this is
an array of StackType_t variables.  The size of StackType_t is dependent on
the RTOS port. */
StackType_t xCIFAR10L7Stack[ STACK_SIZE ];
StackType_t xCIFAR10L12Stack[ STACK_SIZE ];
StackType_t xMNISTL7Stack[ STACK_SIZE ];

void CIFAR10L7Task(){
    TickType_t start, C_ticks;
    for(;;){
        start = xTaskGetTickCount();

        r10cnn_driver(&a4p_exp1_configs.configs[0], &r10cnn_inas_original, output0);

        C_ticks = xTaskGetTickCount() - start;
//==================================<RTEN>============================================
        am_util_stdio_printf("CIFAR10L7Task Ticks Difference: %ldms\n", C_ticks);

        if (0 != clear_exe_status()){
            am_util_stdio_printf("CIFAR10L7Task: Error in clearing execution status\n");
        }
//=================================</RTEN>============================================
        vTaskDelay(a4p_exp1_configs.configs[0].T_ms / portTICK_PERIOD_MS);
    }
}

void CIFAR10L12Task(){
    TickType_t start, C_ticks;
    for(;;){
        start = xTaskGetTickCount();

        r10cnn_driver(&a4p_exp1_configs.configs[1], &r10cnn_inas_original, output1);

        C_ticks = xTaskGetTickCount() - start;
//==================================<RTEN>============================================
        am_util_stdio_printf("CIFAR10L12Task Ticks Difference: %ldms\n", C_ticks);

        if (0 != clear_exe_status()){
            am_util_stdio_printf("CIFAR10L12Task: Error in clearing execution status\n");
        }
//=================================</RTEN>============================================
        vTaskDelay(a4p_exp1_configs.configs[1].T_ms / portTICK_PERIOD_MS);
    }
}

void MNISTL7Task(){
    TickType_t start, C_ticks;
    for(;;){
        start = xTaskGetTickCount();

        r10cnn_driver(&a4p_exp1_configs.configs[2], &r10cnn_inas_original, mnist_output0);

        C_ticks = xTaskGetTickCount() - start;
//==================================<RTEN>============================================
        am_util_stdio_printf("MNISTL7Task Ticks Difference: %ldms\n", C_ticks);

        if (0 != clear_exe_status()){
            am_util_stdio_printf("MNISTL7Task: Error in clearing execution status\n");
        }
//=================================</RTEN>============================================
        vTaskDelay(a4p_exp1_configs.configs[2].T_ms / portTICK_PERIOD_MS);
    }
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
    
    xCIFAR10L7Task = xTaskCreateStatic(
                    CIFAR10L7Task,
                    "CIFAR10L7Task",
                    STACK_SIZE,
                    0,
                    a4p_exp1_configs.configs[0].priority,
                    xCIFAR10L7Stack,
                    &xCIFAR10L7TCB);
    
    xCIFAR10L12Task = xTaskCreateStatic(
                    CIFAR10L12Task,
                    "CIFAR10L12Task",
                    STACK_SIZE,
                    0,
                    a4p_exp1_configs.configs[1].priority,
                    xCIFAR10L12Stack,
                    &xCIFAR10L12TCB);
    
    xMNISTL7Task = xTaskCreateStatic(
                    MNISTL7Task,
                    "MNISTL7Task",
                    STACK_SIZE,
                    0,
                    a4p_exp1_configs.configs[2].priority,
                    xMNISTL7Stack,
                    &xMNISTL7TCB);
    
    //
    // Start the scheduler.
    //
    vTaskStartScheduler();

    
    return 0;
}