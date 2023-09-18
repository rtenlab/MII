#ifndef R10CNN_CONFIG_H
#define R10CNN_CONFIG_H

enum mem_mode
{
    NORMAL = 0, // normal mode
    XIP = 1 // XIP mode
};
typedef enum mem_mode mem_mode;

enum exe_mode
{
    NONVALID = -1,
    VANILLA = 0,
    LAYER = 1,
    TILED = 2,
    FILTER = 3,
    SINGLE = 4
};
typedef enum exe_mode exe_mode;

enum precision
{
    BINARY32 = 0, // using float - dataf
    BINARY16 = 1 // using bin16 - data_bin16
};
typedef enum precision precision;

enum dataset
{
    CIFAR10 = 0, // cifar10 dataset
    SOME_OTHER = 1 // placeholder
};
typedef enum dataset dataset;

struct exe_config
{
    enum mem_mode MEM_MODE; // memory mode

    enum exe_mode EXE_MODE;

    // enum precision EXE_PRECISION;

    size_t T_ms; // period of the current task, in ms
    size_t D_ms; // deadline of the current task, in ms

    size_t priority; // priority of the current task

    /*
     * { INFO0_SIGNATURE0-0 : Init Status Flag (0 = first start | 1 = in-inference)
     * INFO0_SIGNATURE0-1 : correct label
     * INFO0_SIGNATURE0-2 : layer_id
     * INFO0_SIGNATURE0-3 : x0 | r
     * INFO0_SIGNATURE0-4 : x1 | c
     * INFO0_SIGNATURE0-5 : z0 | n
     * INFO0_SIGNATURE0-6 : z1 | m
     * INFO0_SIGNATURE0-7 : q
     * INFO0_SIGNATURE0-8 : k
     * INFO0_SECURITY } 
    */
    uint32_t exe_status[10]; // to be stored in mram

    // Below just for compatibility from PC
    // When writting NVM with a4p_preload.c
    enum dataset DATASET;
    char* data_fname;
    char* label_fname;
    size_t EXE_TRAILS; // experiment trials number
};
typedef struct exe_config exe_config;

struct a4p_config
{
    size_t num_tasks; // number of tasks 
    size_t num_r10cnns; // number of r10cnn instances

    exe_config *configs;

    size_t EXE_TRAILS; // experiment trials number


};
typedef struct a4p_config a4p_config;

#endif //R10CNN_CONFIG_H