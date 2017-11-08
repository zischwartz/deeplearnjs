"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var benchmark_1 = require("./benchmark");
var conv_benchmarks_1 = require("./conv_benchmarks");
var conv_transposed_benchmarks_1 = require("./conv_transposed_benchmarks");
var matmul_benchmarks_1 = require("./matmul_benchmarks");
var pool_benchmarks_1 = require("./pool_benchmarks");
var reduction_ops_benchmark_1 = require("./reduction_ops_benchmark");
var unary_ops_benchmark_1 = require("./unary_ops_benchmark");
function getRunGroups() {
    var groups = [];
    groups.push({
        name: 'Matrix Multiplication: ' +
            'matmul([size, size], [size, size])',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('mulmat_gpu', new matmul_benchmarks_1.MatmulGPUBenchmark()),
            new benchmark_1.BenchmarkRun('mulmat_cpu', new matmul_benchmarks_1.MatmulCPUBenchmark())
        ],
        params: {}
    });
    var convParams = { inDepth: 8, outDepth: 3, filterSize: 7, stride: 1 };
    groups.push({
        name: 'Convolution: image [size, size]',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [new benchmark_1.BenchmarkRun('conv_gpu', new conv_benchmarks_1.ConvGPUBenchmark(convParams))],
        params: convParams
    });
    var convTransposedParams = { inDepth: 8, outDepth: 3, filterSize: 7, stride: 1 };
    groups.push({
        name: 'Convolution Transposed: deconv over image [size, size]',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [new benchmark_1.BenchmarkRun('conv_transpose_gpu', new conv_transposed_benchmarks_1.ConvTransposedGPUBenchmark(convTransposedParams))],
        params: convTransposedParams
    });
    var poolParams = { depth: 8, fieldSize: 4, stride: 4, type: 'max' };
    groups.push({
        name: 'Pool Op Benchmark: input [size, size]',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(4, step); },
        options: ['max', 'min', 'avg'],
        selectedOption: 'max',
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('pool_gpu', new pool_benchmarks_1.PoolGPUBenchmark(poolParams)),
            new benchmark_1.BenchmarkRun('pool_cpu', new pool_benchmarks_1.PoolCPUBenchmark(poolParams))
        ],
        params: poolParams
    });
    groups.push({
        name: 'Unary Op Benchmark (CPU vs GPU): input [size, size]',
        min: 0,
        max: 1024,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        options: [
            'log', 'exp', 'neg', 'sqrt', 'abs', 'relu', 'sigmoid', 'sin', 'cos',
            'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh'
        ],
        selectedOption: 'log',
        stepSize: 64,
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('unary ops CPU', new unary_ops_benchmark_1.UnaryOpsCPUBenchmark()),
            new benchmark_1.BenchmarkRun('unary ops GPU', new unary_ops_benchmark_1.UnaryOpsGPUBenchmark())
        ],
        params: {}
    });
    groups.push({
        name: 'Reduction Op Benchmark (CPU vs GPU): input [size, size]',
        min: 0,
        max: 1024,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        options: ['max', 'min', 'sum', 'logSumExp'],
        selectedOption: 'max',
        stepSize: 64,
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('reduction ops CPU', new reduction_ops_benchmark_1.ReductionOpsCPUBenchmark()),
            new benchmark_1.BenchmarkRun('reduction ops GPU', new reduction_ops_benchmark_1.ReductionOpsGPUBenchmark())
        ],
        params: {}
    });
    return groups;
}
exports.getRunGroups = getRunGroups;
//# sourceMappingURL=math-benchmark-run-groups.js.map