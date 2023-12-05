#include "../../r10cnn_zoo/offline1/CIFAR10_7.h"
#include "../../r10cnn_zoo/offline1/CIFAR10_12.h"
#include "../../r10cnn_zoo/offline1/MNIST_7.h"

// #include "../../r10cnn_zoo/MLPerf/resnetv1.h"

#include "../../r10cnn_zoo/offline1/HAR_5.h"

// #include "../../r10cnn_zoo/FC4.h"

#include "../../r10cnn_zoo/offline1/exp1_configs.h"

#include "r10_cnn.h"

//*****************************************************************************
//
// Task handles
//
//*****************************************************************************
TaskHandle_t xSetupTask;
TaskHandle_t xInferenceTask;

/* Structure that will hold the TCB of the task being created. */
StaticTask_t xInferenceTCB;

/* Buffer that the task being created will use as its stack.  Note this is
an array of StackType_t variables.  The size of StackType_t is dependent on
the RTOS port. */
StackType_t xInferenceStack[ STACK_SIZE ];

void InferenceTask(){

    for(;;){
// #if defined(UART_PROFILE)
        TickType_t start, C_ticks;
        start = xTaskGetTickCount();
// #endif

        // <RTEN> The blelow 3 are mutually exclusive
        // r10cnn_driver(&a4p_exp1_configs.configs[0], &r10cnn_cifar10_7, output0); // CIFAR10_7
        r10cnn_driver(&a4p_exp1_configs.configs[1], &r10cnn_cifar10_12, output1); // CIFAR10_12
        // r10cnn_driver(&a4p_exp1_configs.configs[2], &r10cnn_mnist_7, mnist_output0); // MNIST_7
        
        // r10cnn_driver(&a4p_exp1_configs.configs[1], &r10_resnetv1, resnetv1_output); // resnetv1
        
        // r10cnn_driver(&a4p_exp1_configs.configs[0], &r10cnn_har_5, har_5_output0); // HAR_5
        
        // r10cnn_driver(&a4p_exp1_configs.configs[0], &r10cnn_fc4, fc4_output0); // FC4
        // </RTEN>
    
        if (0 != clear_exe_status()){
            am_util_stdio_printf("InferenceTask: Error in clearing execution status\n");
        }

// #if defined(UART_PROFILE)
        C_ticks = xTaskGetTickCount() - start;
        am_util_stdio_printf("InferenceTask Ticks Difference: %ldms\n", C_ticks);
// #endif
    }

    // while(1);
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
    
    xInferenceTask = xTaskCreateStatic(
                    InferenceTask,
                    "InferenceTask",
                    STACK_SIZE,
                    0,
                    a4p_exp1_configs.configs[0].priority,
                    xInferenceStack,
                    &xInferenceTCB);
    
    //
    // Start the scheduler.
    //
    vTaskStartScheduler();

    
    return 0;
}