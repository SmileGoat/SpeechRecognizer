# Copyright 2019 PEACH LAB. All Rights Reserved.
# Author : goat.zhou@foxmail.com (Yang Zhou)

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "compat", 
		hdrs = [
	      "basic_types.h",	
				"compat.h",
				"macros.h",
		],

)

cc_library(
    name = 'kaldi-base',
    srcs = [
        'io-funcs.cc',
				'kaldi-error.cc',
       	'kaldi-math.cc',
       	'kaldi-utils.cc',
      	'timer.cc',
	  ],
    hdrs = glob(["*.h"]),
    deps = [
	      "//common/third_party/openfst:fst",
    ],
)

cc_binary(
    name = 'kaldi-math-test',
    srcs = [
		    'kaldi-math-test.cc'
	  ],
    deps = [
       ':kaldi-base',
    ],
)
