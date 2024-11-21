# MII
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MII uses the r10cnn project, Run Everywhere CNN. r10cnn is developed by Ziliang(Johnson) Zhang from RTEN (Real-Time Embedded and Networked systems) Lab in University of California Riverside. It includes the following components:
* ```r10cnn_a4bp```: main inference code for Apollo4 Blue Plus board
* ```r10cnn_zoo```: offline1: models for running inference on Apollo4 Blue Plus board, MLPerf: benchmark models

## Apollo4 Blue Plus Setup and Inference
To begin with, We recommend using the following development environment from your host machine:
* Windows 10/11 OS - Windows is recommended for install the KEIL uVision and we need linux to run the gcc
* Multipass or Cygwin is also recommended for compiling gcc for PC

First, download the following tools for development:
* SEGGER J-Link Software: https://www.segger.com/downloads/jlink
* KEIL uVision 5 (ARM Compiler 5): https://www.keil.com/demo/eval/arm.htm
  * (To be imported to KEIL uVision) Latest Keil Pack (CMSIS Ambiq Pack): http://www.keil.com/dd2/pack/#/third-party-download-dialog
* IAR IDE/Compiler: https://www.iar.com/iar-embedded-workbench/tools-for-arm/arm-cortex-m-edition/
    * <strong>[Compiler Version]</strong> Make sure you use the V6.19 for the ARM compiler. It is very IMPORTANT to match the version or else your code cannot be compiled
* GCC (GNU Arm Embedded Toolchain): https://gcc.gnu.org

The apollo4 need to download AmbiqSuite SDK 4.3.0 for library. We have included the necessary files for this project in ```r10cnn_a4bp/AmbiqSuite_R4.3.0```


<strong>Keil Project Import:</strong> Import the uVision project ```r10cnn_a4bp/keil/r10cnn_apollo4p_evb.uvprojx``` into your KEIL uVision 5. Again, make sure you use ARM compiler V6.19. You can check this in Project- Options for Target - Target

<strong>Weights Loading:</strong> Then, load the ```a4p_preload.c``` to the src (remove ```r10cnn_offline1.c``` if it is in the src folder in KEIL uVision 5 because they are mutually exclusive) clean and build and upload to the board. This will preload the header file with weights and input into the non-volatile memory of the board.

<strong>RUN:</strong> Finally, load the ```r10cnn_offline1.c``` to the src folder and clean and build and upload to the board. This will run the inference given the preloaded data and UART the result to your terminal. For Windows, you can use Termite or TerraTerm to read the UART output. For linux or mac, you can use screen command to read the UART output. Make sure you are using <strong>115200</strong> baud rate. 

### Toggle JIT Checkpointing in Apollo4 Blue Plus
By default, JIT is enabled and can be executed with VANILLA. To disable JIT, one may follow the following steps:
* r10cnn_zoo/exp1_configs.h: change EXE_MODE to LAYER, TILED or FILTER to avoid JIT (VANIILA)
* r10cnn_apollo4p_evb/src/a4p_hardware.c: search "RTEN" Tag and toggle off the ADC
* r10cnn_apollo4p_evb/src/r10cnn_oneshot.c or r10cnn_apollo4p_evb/src/r10cnn_main.c: make sure your r10cnn_driver is using the correct config, r10cnn_model and output
* <strong>IMPORTANT</strong>: a4p_hardware.c JIT_THD change to match the config's JIT_THD
* Yehn delete the flag ```JIT_ENABLED``` in Keil tab Project- Options for Target - C/C++ - Define

### Switching between execution modes and Checkpointing mechanisms
In each of the header file in ```r10cnn_zoo/run_model```, change each layer's ```mem``` filed:
``` c
.mem = NORMAL, // NORAML: normal execution, XIP: Execution in Place execution
.exe = TILED, // VANILLA: JIT (by default), ST-L: LAYER, ST-F: FILTER, ST-T: TILED
```
By default in the ```r10cnn_zoo/run_model/CIFAR10_7.h``` ST-T (Static Checkpointing Tiled) is enabled

## Reference Codes
* iNAS: https://github.com/EMCLab-Sinica/Intermittent-aware-NAS
* Keras2C: https://github.com/f0uriest/keras2c/
