# Copyright (c) 2020 PeachLab. All Rights Reserved.
# Author  : goat.zhou@qq.com (Yang Zhou)

package(default_visibility = ["//visibility:public"])

cc_binary(
    name = 'mfcc-test',
    srcs = [
        'feature-mfcc-test.cc',
    ],
    deps = [
        ":wave-reader",
        ":mfcc",
    ],
)

cc_library(
    name = 'mfcc',
    srcs = [
        'feature-functions.cc',
        'feature-mfcc.cc',
    ],
    hdrs = ['feature-mfcc.h'],
    deps = [
        ":mel-computations",
    ],
)

cc_binary(
    name = 'fbank-test',
    srcs = [
        'feature-fbank-test.cc',
    ],
    deps = [
        ":fbank",
        ":wave-reader",
    ],
)

cc_library(
    name = 'fbank',
    srcs = [
        'feature-fbank.cc',
    ],
    hdrs = ['feature-fbank.h'],
    deps = [
        ":mel-computations",],
)

cc_library(
    name = 'mel-computations',
    srcs = [
        'mel-computations.cc',
    ],
    hdrs = [
        "mel-computations.h",
        "feature-window.h",
        'feature-common.h',
    ],
    deps = [
        ":feature-functions",
        ":feature-window",
        ":feature-common",
        "//base:kaldi-base",
        "//matrix:kaldi-matrix",
        "//util:kaldi-util",],
)

cc_library(
    name = 'feature-common',
    srcs = [
        'feature-common-inl.h',
        'resample.cc',
    ],
    hdrs = [
        "feature-common.h",
        "resample.h", 
    ],
    deps = [
        ":feature-functions",
        "//base:kaldi-base",
        "//matrix:kaldi-matrix",
        "//util:kaldi-util",
		],
)

cc_library(
    name = 'feature-window',
    srcs = [
        'feature-window.cc',
    ],
    hdrs = ["feature-window.h"],
    deps = [
        "//base:kaldi-base",
        "//matrix:kaldi-matrix",
        "//util:kaldi-util",
		],
)

cc_library(
    name = 'feature-functions',
    srcs = [
        'feature-functions.cc',
    ],
    hdrs = ["feature-functions.h"],
    deps = [
        "//base:kaldi-base",
        "//matrix:kaldi-matrix",
        "//util:kaldi-util",
		],
)

cc_library(
    name = 'signal',
    srcs = [
        'signal.cc',
    ],
    hdrs = ["signal.h"],
    deps = [
        "//base:kaldi-base",
        "//matrix:kaldi-matrix",
        "//util:kaldi-util",
		],
)


cc_library(
    name = 'wave-reader',
    srcs = [
        'wave-reader.cc',
    ],
    hdrs = ["wave-reader.h"],
    deps = [
        "//base:kaldi-base",
        "//matrix:kaldi-matrix",
        "//util:kaldi-util",
		],
)
