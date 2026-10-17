set script_dir [file dirname [file normalize [info script]]]
set project_dir [file join $script_dir build]
set mem_dir [file normalize [file join $script_dir .. tb data]]
set cifar_bin [file normalize [file join $script_dir .. model test_batch.bin]]

set part "xck26-sfvc784-2LV-c"
if {[info exists ::env(HLS_PART)]} { set part $::env(HLS_PART) }

set num_images 1
if {[info exists ::env(HLS_NUM_IMAGES)]} { set num_images $::env(HLS_NUM_IMAGES) }

set skip_csim 0
if {[info exists ::env(HLS_SKIP_CSIM)]} { set skip_csim $::env(HLS_SKIP_CSIM) }

open_project -reset $project_dir
set_top cnn_accelerator
add_files [file join $script_dir cnn_accelerator.cpp]
add_files [file join $script_dir cnn_accelerator.h]
add_files -tb [file join $script_dir .. tb hls cnn_accelerator_tb.cpp] -cflags "-I$script_dir"

open_solution -reset solution1 -flow_target vivado
set_part $part
create_clock -period 5 -name default

if {!$skip_csim} {
    csim_design -argv "$mem_dir $cifar_bin $num_images"
}
csynth_design
export_design -format ip_catalog

exit
