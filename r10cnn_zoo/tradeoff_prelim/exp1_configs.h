#ifndef EXP1_CONFIGS_H
#define EXP1_CONFIGS_H

#if defined AM_PART_APOLLO4B || AM_PART_APOLLO4P
#include "../../r10cnn_a4bp/src/r10_cnn.h"
#else
#include "../../libr10cnn/r10_cnn.h"
#endif

/*******************************************************************
 * Execution Taskset
 * Unique Each Task
 *******************************************************************/
#pragma PERSISTENT(exp1_configs)
struct exe_config exp1_configs[3] = {
{ // cifar10_7
    .MEM_MODE = NORMAL,
    // .MEM_MODE = XIP,

    // .EXE_MODE = TILED,
    // .EXE_MODE = FILTER,
    // .EXE_MODE = LAYER,
    .EXE_MODE = VANILLA,

    .T_ms = 8000,
    .D_ms = 8000,
    .priority = 2,
    .exe_status = 
        {0x00000000, 
        0x00000000,  
        0x00000000,  
        0x00000000,  
        0x00000000} 
},
{ // cifar10_12
    .MEM_MODE = NORMAL,
    // .MEM_MODE = XIP,

    // .EXE_MODE = TILED,
    .EXE_MODE = FILTER,
    // .EXE_MODE = LAYER,
    // .EXE_MODE = VANILLA,
    .T_ms = 15000,
    .D_ms = 15000,
    .priority = 1,
    .exe_status = 
        {0x00000000, 
        0x00000000,  
        0x00000000,  
        0x00000000,  
        0x00000000} 
},
{ // mnist_7
    .MEM_MODE = NORMAL,
    // .MEM_MODE = XIP,

    // .EXE_MODE = TILED,
    // .EXE_MODE = FILTER,
    // .EXE_MODE = LAYER,
    .EXE_MODE = VANILLA,
    .T_ms = 6000,
    .D_ms = 6000,
    .priority = 3,
    .exe_status = 
        {0x00000000, 
        0x00000000,  
        0x00000000,  
        0x00000000,  
        0x00000000} 
}
};

#pragma PERSISTENT(a4p_exp1_configs)
struct a4p_config a4p_exp1_configs={
	.num_tasks = 3,
    .num_r10cnns = 3,
	.configs = exp1_configs,
    .EXE_TRAILS = 1
};
#endif // EXP1_CONFIGS_H