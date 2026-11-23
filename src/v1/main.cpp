#include "baseline.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using U64 = std::uint64_t;
constexpr U64 input_header_bytes = 44;
constexpr U64 int_max = std::numeric_limits<int>::max();

/** Multiply element counts or byte sizes, rejecting integer overflow. */
U64 multiply(U64 a, U64 b) {
    if (b && a > std::numeric_limits<U64>::max() / b)
        throw std::runtime_error("payload size overflow");
    return a * b;
}

/** Add payload sizes, rejecting integer overflow. */
U64 add(U64 a, U64 b) {
    if (a > std::numeric_limits<U64>::max() - b)
        throw std::runtime_error("payload size overflow");
    return a + b;
}

/** Read exactly bytes bytes, or report a truncated or oversized payload. */
void read_bytes(std::istream& in, void* data, U64 bytes) {
    if (bytes > static_cast<U64>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("payload exceeds stream size limit");
    if (bytes && !in.read(static_cast<char*>(data), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("truncated input");
}

/** Read one scalar in the file's little-endian representation. */
template <typename T> T read_value(std::istream& in) {
    T value{};
    read_bytes(in, &value, sizeof(T));
    return value;
}

/** Write a byte range, checking stream limits and write failures. */
void write_bytes(std::ostream& out, const void* data, U64 bytes) {
    if (bytes > static_cast<U64>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("output exceeds stream size limit");
    if (bytes && !out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes)))
        throw std::runtime_error("cannot write output");
}

/** Write one scalar; main verifies that the host is little-endian. */
template <typename T> void write_value(std::ostream& out, T value) {
    write_bytes(out, &value, sizeof(T));
}

// Host-side packed inputs, using the same dimensions and layout as Python.
struct Input {
    U64 t, e, k, n;
    std::vector<std::int64_t> offsets;
    std::vector<float> x, w;
};

/** Load and validate the input header, expert boundaries, and FP32 payload. */
Input load(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open input");
    std::array<char, 8> magic{};
    read_bytes(in, magic.data(), magic.size());
    const auto version = read_value<std::uint32_t>(in);
    if (std::memcmp(magic.data(), "MOEIN001", 8) || version != 1)
        throw std::runtime_error("unsupported magic or version");
    Input a{};
    a.t = read_value<U64>(in); a.e = read_value<U64>(in);
    a.k = read_value<U64>(in); a.n = read_value<U64>(in);
    if (a.t > int_max || !a.e || a.e > int_max || !a.k || a.k > int_max || !a.n || a.n > int_max)
        throw std::runtime_error("dimensions must fit int32; only T may be zero");
    // Compute sizes safely before trusting dimensions from the file.
    const U64 x_count = multiply(a.t, a.k);
    const U64 w_count = multiply(multiply(a.e, a.k), a.n);
    const U64 offset_bytes = multiply(a.e + 1, sizeof(std::int64_t));
    const U64 expected = add(add(input_header_bytes, offset_bytes), multiply(add(x_count, w_count), sizeof(float)));
    // Check output too, before allocating any payload or device buffers.
    multiply(multiply(a.t, a.n), sizeof(float));
    if (expected > std::numeric_limits<std::size_t>::max() ||
        expected > static_cast<U64>(std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("payload exceeds host size limit");
    if (std::filesystem::file_size(path) != expected)
        throw std::runtime_error("input payload length mismatch (truncated or surplus bytes)");
    a.offsets.resize(a.e + 1);
    read_bytes(in, a.offsets.data(), offset_bytes);
    if (a.offsets.front() != 0 || a.offsets.back() != static_cast<std::int64_t>(a.t))
        throw std::runtime_error("offsets must start at 0 and end at T");
    for (std::size_t e = 0; e + 1 < a.offsets.size(); ++e)
        if (a.offsets[e + 1] < a.offsets[e])
            throw std::runtime_error("offsets must be nondecreasing");
    a.x.resize(x_count); a.w.resize(w_count);
    read_bytes(in, a.x.data(), multiply(x_count, sizeof(float)));
    read_bytes(in, a.w.data(), multiply(w_count, sizeof(float)));
    for (const auto* values : {&a.x, &a.w})
        for (float value : *values)
            if (!std::isfinite(value)) throw std::runtime_error("input contains non-finite values");
    return a;
}

/** Save finite packed outputs with the versioned T-by-N output header. */
void save(const char* path, const Input& a, const std::vector<float>& y) {
    for (float value : y)
        if (!std::isfinite(value)) throw std::runtime_error("output contains non-finite values");
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open output");
    write_bytes(out, "MOEOUT01", 8);
    write_value<std::uint32_t>(out, 1);
    write_value<U64>(out, a.t); write_value<U64>(out, a.n);
    write_bytes(out, y.data(), multiply(y.size(), sizeof(float)));
    out.close();
    if (!out) throw std::runtime_error("cannot finalize output");
}

// Destructors never throw; checked close() is used on normal completion.
struct Stream {
    cudaStream_t value = nullptr;
    /** Create the stream used for transfers and expert computation. */
    Stream() { check_cuda(cudaStreamCreate(&value), "cudaStreamCreate"); }
    // Resource owners cannot be copied, which prevents double destruction.
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    /** Release the stream on scope exit, reporting errors without throwing. */
    ~Stream() {
        if (value) {
            auto status = cudaStreamDestroy(value);
            if (status != cudaSuccess) std::cerr << "cleanup cudaStreamDestroy: " << cudaGetErrorString(status) << '\n';
        }
    }
    /** Release explicitly and throw on failure; leave no resource for the destructor. */
    void close() { check_cuda(cudaStreamDestroy(std::exchange(value, nullptr)), "cudaStreamDestroy"); }
};

struct Handle {
    cublasHandle_t value = nullptr;
    /** Create the cuBLAS context shared by all expert GEMMs. */
    Handle() { check_cublas(cublasCreate(&value), "cublasCreate"); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    /** Release the handle on scope exit, reporting errors without throwing. */
    ~Handle() {
        if (value) {
            auto status = cublasDestroy(value);
            if (status != CUBLAS_STATUS_SUCCESS) std::cerr << "cleanup cublasDestroy: " << status << '\n';
        }
    }
    /** Release the handle explicitly and report any failure as an exception. */
    void close() { check_cublas(cublasDestroy(std::exchange(value, nullptr)), "cublasDestroy"); }
};

struct DeviceBuffer {
    float* value = nullptr;
    /** Allocate device storage for count FP32 elements. */
    explicit DeviceBuffer(U64 count) {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&value), multiply(count, sizeof(float))), "cudaMalloc");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    /** Free device storage on scope exit, reporting errors without throwing. */
    ~DeviceBuffer() {
        if (value) {
            auto status = cudaFree(value);
            if (status != cudaSuccess) std::cerr << "cleanup cudaFree: " << cudaGetErrorString(status) << '\n';
        }
    }
    /** Free storage explicitly and report any failure as an exception. */
    void close() { check_cuda(cudaFree(std::exchange(value, nullptr)), "cudaFree"); }
};
} // namespace

/** Run input loading, GPU computation, and output writing; return nonzero on errors. */
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: moe_v1 INPUT.bin OUTPUT.bin");
        const std::uint16_t endian = 1;
        if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
            throw std::runtime_error("only little-endian hosts are supported");
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
        const Input a = load(argv[1]);
        std::vector<float> y(multiply(a.t, a.n));
        // Empty inputs can be written without creating any CUDA resources.
        if (a.t) {
            Stream stream;
            Handle handle;
            DeviceBuffer x(a.x.size()), w(a.w.size()), output(y.size());
            // One stream preserves upload -> GEMM -> download ordering.
            check_cuda(cudaMemcpyAsync(x.value, a.x.data(), a.x.size() * sizeof(float), cudaMemcpyHostToDevice, stream.value), "upload X");
            check_cuda(cudaMemcpyAsync(w.value, a.w.data(), a.w.size() * sizeof(float), cudaMemcpyHostToDevice, stream.value), "upload W");
            run_experts(x.value, w.value, output.value, a.offsets, static_cast<int>(a.k), static_cast<int>(a.n), handle.value, stream.value);
            check_cuda(cudaStreamSynchronize(stream.value), "GEMM synchronization");
            check_cuda(cudaMemcpyAsync(y.data(), output.value, y.size() * sizeof(float), cudaMemcpyDeviceToHost, stream.value), "download Y");
            // CPU output must not be used until the download finishes.
            check_cuda(cudaStreamSynchronize(stream.value), "download synchronization");
            output.close(); w.close(); x.close(); handle.close(); stream.close();
        }
        save(argv[2], a, y);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "moe_v1: " << error.what() << '\n';
        return 1;
    }
}
