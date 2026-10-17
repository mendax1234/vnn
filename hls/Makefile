VITIS_HLS ?= vitis-run --mode hls --tcl
HLS_PART ?= xck26-sfvc784-2LV-c
HLS_NUM_IMAGES ?= 1
HLS_SKIP_CSIM ?= 0

.PHONY: all clean

all:
	HLS_PART=$(HLS_PART) HLS_NUM_IMAGES=$(HLS_NUM_IMAGES) HLS_SKIP_CSIM=$(HLS_SKIP_CSIM) $(VITIS_HLS) run_hls.tcl

clean:
	$(RM) -r build vitis_hls.log vitis_hls.app
