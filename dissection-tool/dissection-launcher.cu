#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdint>

#include "error-macro.h"

#include "vector-add.kernel.cu"
#include "vector-add.host.cu"

auto main(int argc, char **argv) -> int {
  if (argc < 2) {
    std::cerr << argv[0] << " <kernel_name> <...>\n";
    return -1;
  } else if (std::string("vector_add") == argv[1]) {
    return vector_add_host(argc - 1, argv + 1);
  } else if (std::string("pchase-citadel") == argv[1]) {
    std::cerr << "pchase-citadel kernel is not yet implemented\n";
  }

  std::cerr << argv[0] << " <kernel_name> <...>\n";
}
