nvprof ./lab1 > gpu_saxpy_V15.log 2>&1 -2
./run_nvprof_plot.sh --log gpu_saxpy_log.log --prefix gpu_saxpy
