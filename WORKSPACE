"""speech_recognizer Workspace"""

workspace(name = "org_speech_recognizer")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# START: Upstream TensorFlow dependencies
# TensorFlow build depends on these dependencies.
# Needs to be in-sync with TensorFlow sources.
http_archive(
    name = "io_bazel_rules_closure",
		sha256 = "5b00383d08dd71f28503736db0500b6fb4dda47489ff5fc6bed42557c07c6ba9",
	  strip_prefix = "rules_closure-308b05b2419edb5c8ee0471b67a40403df940149",
	  urls = [
	      "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",
	      "https://github.com/bazelbuild/rules_closure/archive/308b05b2419edb5c8ee0471b67a40403df940149.tar.gz",  # 2019-06-13
    ],
)


# https://github.com/bazelbuild/bazel-skylib/releases
http_archive(
    name = "bazel_skylib",
	  sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
	  urls = [
	      "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
	      "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
    ],
)
# END: Upstream TensorFlow dependencies

http_archive(
    name = "org_tensorflow",
    sha256 = "1f313adfbc52d4810a784d27d6b3887fec9d66e1ad67c0142e938fa4cf46fd0f",
    strip_prefix = "tensorflow-fabcd8f89cd5975331994049705e15cb75f32e0c",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/fabcd8f89cd5975331994049705e15cb75f32e0c.tar.gz",   # 2020-06-18
    ],
)

http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-3f0cf6b62ad1eb50d8736538363d3580dd640c3e",
    urls = ["https://github.com/google/googletest/archive/3f0cf6b62ad1eb50d8736538363d3580dd640c3e.zip"],
)

http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-master",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/master.zip"],
)

load("@org_tensorflow//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least("1.0.0")
load("@org_tensorflow//tensorflow:workspace.bzl", "tf_repositories")
tf_repositories(path_prefix="", tf_repo_name="org_tensorflow")

register_toolchains("@local_config_python//:py_toolchain")

load("@io_bazel_rules_closure//closure:defs.bzl", "closure_repositories")
closure_repositories()

load("@org_tensorflow//third_party/toolchains/preconfig/generate:archives.bzl",
     "bazel_toolchains_archive")
bazel_toolchains_archive()

load(
    "@bazel_toolchains//repositories:repositories.bzl",
    bazel_toolchains_repositories = "repositories",
)
bazel_toolchains_repositories()

load("@org_tensorflow//third_party/android:android_configure.bzl", "android_configure")
android_configure(name="local_config_android")
load("@local_config_android//:android.bzl", "android_workspace")
android_workspace()

# Need to export environment variable ANDROID_HOME.
android_sdk_repository(
    name = "androidsdk",
)

# Need to export environment variable ANDROID_NDK_HOME.
android_ndk_repository(
    name = "androidndk",
)
