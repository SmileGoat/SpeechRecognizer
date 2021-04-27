#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "codelab/tflite/hello_tflite.h"

namespace hello_tflite {

TEST(TFliteTest, TestHelloWorld) {
  EXPECT_EQ(add(0, 1), 1);
}

}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
