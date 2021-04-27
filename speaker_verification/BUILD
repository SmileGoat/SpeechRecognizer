# Copyright (c) 2021 Peachlab. All Rights Reserved.
# Author  : goat.zhou@qq.com (Yang Zhou)

package(default_visibility = ["//visibility:public"])

cc_library(
    name = 'speaker_verification',
    srcs = [
        'speaker_verification.cc',
    ],
    hdrs = ['speaker_verification.h'],
    deps = [
       ':speaker_verification_client',
    ],
)

cc_library(
    name = 'speaker_verification_client',
    srcs = [
        'speaker_verification_client.cc',
    ],
    hdrs = ['speaker_verification_client.h'],
    deps = [
       ':xvector_controller',
    ],
)

cc_library(
    name = 'xvector_controller',
    srcs = [
        'xvector_controller.cc',
    ],
    hdrs = ['xvector_controller.h'],
    deps = [
       ':xvector_controller_impl',
       ':xvector_frontend',
    ],
)

cc_library(
    name = 'xvector_controller_impl',
    srcs = [
        'xvector_controller_impl.cc',
    ],
    hdrs = ['xvector_controller_impl.h'],
    deps = [
       ':xvector_extractor',
       ':plda',
    ],
)

cc_library(
    name = 'plda',
    srcs = [
        'plda.cc',
    ],
    hdrs = ['plda.h'],
    deps = [
        "//base:kaldi-base",
        "//matrix:kaldi-matrix",
        "//util:kaldi-util",
    ],
)

cc_library(
    name = 'xvector_extractor',
    srcs = [
        'xvector_extractor.cc',
    ],
    hdrs = ['xvector_extractor.h'],
    deps = [
        "//base:kaldi-base",
        "//matrix:kaldi-matrix",
        "//util:kaldi-util",
				"//interface:nnet_computer_interface",
    ],
)


