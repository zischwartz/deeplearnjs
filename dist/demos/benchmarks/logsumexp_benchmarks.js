"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("../../src/math/ndarray");
var gpgpu_math = require("../../src/math/webgl/gpgpu_math");
var logsumexp_gpu_1 = require("../../src/math/webgl/logsumexp_gpu");
var texture_manager_1 = require("../../src/math/webgl/texture_manager");
var deeplearn_1 = require("../deeplearn");
var benchmark_1 = require("./benchmark");
var LogSumExpCPUBenchmark = (function (_super) {
    __extends(LogSumExpCPUBenchmark, _super);
    function LogSumExpCPUBenchmark() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    LogSumExpCPUBenchmark.prototype.run = function (size) {
        var math = new deeplearn_1.NDArrayMathCPU();
        var a = deeplearn_1.Array2D.randUniform([size, size], -1, 1);
        var start = performance.now();
        math.logSumExp(a);
        var end = performance.now();
        return new Promise(function (resolve, reject) {
            resolve(end - start);
        });
    };
    return LogSumExpCPUBenchmark;
}(benchmark_1.BenchmarkTest));
exports.LogSumExpCPUBenchmark = LogSumExpCPUBenchmark;
var LogSumExpGPUBenchmark = (function (_super) {
    __extends(LogSumExpGPUBenchmark, _super);
    function LogSumExpGPUBenchmark() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    LogSumExpGPUBenchmark.prototype.run = function (size) {
        return new Promise(function (resolve, reject) {
            var gpgpu = new deeplearn_1.GPGPUContext();
            var texManager = new texture_manager_1.TextureManager(gpgpu);
            ndarray_1.initializeGPU(gpgpu, texManager);
            var out = deeplearn_1.Scalar.make([], { texture: texManager.acquireTexture([1, 1]) });
            var a = deeplearn_1.Array2D.randUniform([size, size], -1, 1);
            var program = new logsumexp_gpu_1.LogSumExpProgram(a.shape, [0, 1]);
            var binary = gpgpu_math.compileProgram(gpgpu, program, [a], out);
            var benchmark = function () {
                gpgpu_math.runProgram(binary, [a], out);
            };
            var immediateCleanup = function () {
                a.dispose();
                out.dispose();
                texManager.dispose();
                gpgpu.deleteProgram(binary.webGLProgram);
                gpgpu.deleteProgram(binary.webGLProgram);
            };
            var delayedCleanup = function () {
                gpgpu.dispose();
            };
            if (deeplearn_1.ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) {
                gpgpu.runQuery(benchmark).then(function (timeElapsed) {
                    delayedCleanup();
                    resolve(timeElapsed);
                });
                immediateCleanup();
            }
            else {
                var start = performance.now();
                benchmark();
                out.getValues();
                var totalTime = performance.now() - start;
                immediateCleanup();
                delayedCleanup();
                resolve(totalTime);
            }
        });
    };
    return LogSumExpGPUBenchmark;
}(benchmark_1.BenchmarkTest));
exports.LogSumExpGPUBenchmark = LogSumExpGPUBenchmark;
//# sourceMappingURL=logsumexp_benchmarks.js.map