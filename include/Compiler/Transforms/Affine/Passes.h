#ifndef LIB_TRANSFORM_AFFINE_PASSES_H_
#define LIB_TRANSFORM_AFFINE_PASSES_H_

#include "lib/Transform/Affine/AffineFullUnroll.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_REGISTRATION
#include "Compiler/Transforms/Affine/Passes.h.inc"

}  // namespace tutorial
}  // namespace mlir

#endif