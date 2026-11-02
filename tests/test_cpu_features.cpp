#include <gtest/gtest.h>

#include "tinyinfer/cpu_features.hpp"

using tinyinfer::CpuFeatures;

TEST(CpuFeatures, CachedInstanceIsStable) {
    const CpuFeatures& a = tinyinfer::cpu_features();
    const CpuFeatures& b = tinyinfer::cpu_features();
    EXPECT_EQ(&a, &b);
}

TEST(CpuFeatures, ImpliedFeaturesAreConsistent) {
    const CpuFeatures& f = tinyinfer::cpu_features();
    // every AVX-512 extension implies the foundation set, and AVX2 implies AVX
    // (EXPECT_* expands to an if/else, hence the braces)
    if (f.avx512bw || f.avx512vl || f.avx512vnni) {
        EXPECT_TRUE(f.avx512f);
    }
    if (f.avx512f) {
        EXPECT_TRUE(f.avx2);
    }
    if (f.avx2 || f.fma) {
        EXPECT_TRUE(f.avx);
    }
}

TEST(CpuFeatures, ToStringEmpty) { EXPECT_EQ(tinyinfer::to_string(CpuFeatures{}), "none"); }

TEST(CpuFeatures, ToStringOrder) {
    CpuFeatures f;
    f.avx = true;
    f.avx2 = true;
    f.avx512f = true;
    EXPECT_EQ(tinyinfer::to_string(f), "avx avx2 avx512f");
}
