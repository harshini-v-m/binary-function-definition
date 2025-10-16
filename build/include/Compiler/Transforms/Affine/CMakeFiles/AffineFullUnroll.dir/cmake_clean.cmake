file(REMOVE_RECURSE
  "AffineFullUnrollPass.md"
  "Passes.h.inc"
  "libAffineFullUnroll.a"
  "libAffineFullUnroll.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/AffineFullUnroll.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
