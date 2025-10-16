file(REMOVE_RECURSE
  "libArithcanonicalizer.a"
  "libArithcanonicalizer.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/Arithcanonicalizer.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
