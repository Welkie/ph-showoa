#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#ifdef __CUDACC__
#define PH_HD __host__ __device__
#else
#define PH_HD
#endif

// A host/device implementation of the standard mt19937 engine. The engine
// transition and tempering are specified by the C++ standard; distribution
// helpers below reproduce the libstdc++ paths used by the GCC CPU baseline.
class LegacyMt19937 {
public:
    using result_type = std::uint32_t;
    static constexpr std::size_t state_size = 624;

    PH_HD explicit LegacyMt19937(result_type seed_value = 5489u) {
        seed(seed_value);
    }

    PH_HD void seed(result_type seed_value) {
        state_[0] = seed_value;
        for (std::size_t i = 1; i < state_size; ++i) {
            const result_type previous = state_[i - 1];
            state_[i] = 1812433253u * (previous ^ (previous >> 30u)) +
                static_cast<result_type>(i);
        }
        index_ = state_size;
    }

    PH_HD result_type operator()() {
        if (index_ >= state_size) twist();
        result_type value = state_[index_++];
        value ^= value >> 11u;
        value ^= (value << 7u) & 0x9d2c5680u;
        value ^= (value << 15u) & 0xefc60000u;
        value ^= value >> 18u;
        return value;
    }

    PH_HD static constexpr result_type min() { return 0u; }
    PH_HD static constexpr result_type max() { return 0xffffffffu; }
    PH_HD std::size_t index() const { return index_; }

private:
    result_type state_[state_size]{};
    std::size_t index_ = state_size;

    PH_HD void twist() {
        constexpr result_type upper_mask = 0x80000000u;
        constexpr result_type lower_mask = 0x7fffffffu;
        constexpr result_type matrix_a = 0x9908b0dfu;
        for (std::size_t i = 0; i < state_size; ++i) {
            const result_type joined =
                (state_[i] & upper_mask) |
                (state_[(i + 1) % state_size] & lower_mask);
            result_type next = state_[(i + 397) % state_size] ^ (joined >> 1u);
            if ((joined & 1u) != 0u) next ^= matrix_a;
            state_[i] = next;
        }
        index_ = 0;
    }
};

PH_HD inline std::uint32_t legacy_uniform_u32(
    LegacyMt19937& rng,
    std::uint32_t range
) {
    if (range == 0u) return rng();
    std::uint64_t product = static_cast<std::uint64_t>(rng()) * range;
    std::uint32_t low = static_cast<std::uint32_t>(product);
    if (low < range) {
        const std::uint32_t threshold = static_cast<std::uint32_t>(-range) % range;
        while (low < threshold) {
            product = static_cast<std::uint64_t>(rng()) * range;
            low = static_cast<std::uint32_t>(product);
        }
    }
    return static_cast<std::uint32_t>(product >> 32u);
}

PH_HD inline int legacy_randint(int low, int high, LegacyMt19937& rng) {
    const std::uint64_t width =
        static_cast<std::uint64_t>(static_cast<std::int64_t>(high) - low) + 1u;
    if (width >= (std::uint64_t{1} << 32u)) {
        return static_cast<int>(rng());
    }
    return low + static_cast<int>(legacy_uniform_u32(
        rng, static_cast<std::uint32_t>(width)
    ));
}

PH_HD inline double legacy_unit_real(LegacyMt19937& rng) {
    constexpr double radix = 4294967296.0;
    const double first = static_cast<double>(rng());
    const double second = static_cast<double>(rng());
    return (first + second * radix) / (radix * radix);
}

PH_HD inline double legacy_randdouble(
    double low,
    double high,
    LegacyMt19937& rng
) {
    return low + (high - low) * legacy_unit_real(rng);
}

template <typename RandomAccessIterator>
inline void legacy_shuffle(
    RandomAccessIterator first,
    RandomAccessIterator last,
    LegacyMt19937& rng
) {
    using difference_type =
        typename std::iterator_traits<RandomAccessIterator>::difference_type;
    const difference_type count = last - first;
    if (count <= 1) return;

    // libstdc++ combines two small draws into one URBG invocation range when
    // the generator range is sufficiently larger than the sequence range.
    RandomAccessIterator current = first + 1;
    if ((count % 2) == 0) {
        const int position = legacy_randint(0, 1, rng);
        std::iter_swap(current, first + position);
        ++current;
    }
    while (current != last) {
        const std::uint64_t swap_range =
            static_cast<std::uint64_t>(current - first) + 1u;
        const std::uint64_t combined_range = swap_range * (swap_range + 1u);
        const std::uint32_t combined = legacy_uniform_u32(
            rng, static_cast<std::uint32_t>(combined_range)
        );
        const difference_type first_position = static_cast<difference_type>(
            combined / (swap_range + 1u)
        );
        const difference_type second_position = static_cast<difference_type>(
            combined % (swap_range + 1u)
        );
        std::iter_swap(current, first + first_position);
        ++current;
        std::iter_swap(current, first + second_position);
        ++current;
    }
}

#undef PH_HD
