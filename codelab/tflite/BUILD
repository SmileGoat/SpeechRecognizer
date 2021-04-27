# Copyright (c) 2020 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = "hello_tflite",
    srcs = [
        "hello_tflite.cc",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite/c:common",
        "@org_tensorflow//tensorflow/lite:framework",
        #"@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        #"@org_tensorflow//tensorflow/lite/profiling:profiler",
        #"@org_tensorflow//tensorflow/lite/tools/evaluation:utils",
    ]
)

