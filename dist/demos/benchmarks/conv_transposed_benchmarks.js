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
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("../../src/math/ndarray");
var conv_backprop_gpu_1 = require("../../src/math/webgl/conv_backprop_gpu");
var gpgpu_math = require("../../src/math/webgl/gpgpu_math");
var texture_manager_1 = require("../../src/math/webgl/texture_manager");
var deeplearn_1 = require("../deeplearn");
var benchmark_1 = require("./benchmark");
var ConvTransposedBenchmark = (function (_super) {
    __extends(ConvTransposedBenchmark, _super);
    function ConvTransposedBenchmark(params) {
        var _this = _super.call(this, params) || this;
        _this.params = params;
        return _this;
    }
    return ConvTransposedBenchmark;
}(benchmark_1.BenchmarkTest));
exports.ConvTransposedBenchmark = ConvTransposedBenchmark;
var ConvTransposedGPUBenchmark = (function (_super) {
    __extends(ConvTransposedGPUBenchmark, _super);
    function ConvTransposedGPUBenchmark() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    ConvTransposedGPUBenchmark.prototype.run = function (size) {
        return __awaiter(this, void 0, void 0, function () {
            var origInputDepth, origOutputDepth, xShape, fieldSize, origStride, origPad, gpgpu, texManager, convInfo, program, outputShape, out, x, wShape, W, inputs, binary, benchmark, cleanup, totalTime, start;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        origInputDepth = 1;
                        origOutputDepth = 1;
                        xShape = [size, size, origOutputDepth];
                        fieldSize = 11;
                        origStride = 1;
                        origPad = 1;
                        gpgpu = new deeplearn_1.GPGPUContext();
                        texManager = new texture_manager_1.TextureManager(gpgpu);
                        ndarray_1.initializeGPU(gpgpu, texManager);
                        gpgpu.enableAutomaticDebugValidation(true);
                        convInfo = deeplearn_1.conv_util.computeConvInfo(xShape, fieldSize, fieldSize, origOutputDepth, origStride, origStride, origPad);
                        program = new conv_backprop_gpu_1.Conv2DDerInputProgram(convInfo);
                        outputShape = program.outputShape;
                        out = deeplearn_1.Array3D.zeros(outputShape);
                        x = deeplearn_1.Array3D.randUniform(xShape, -1, 1);
                        wShape = deeplearn_1.conv_util.computeWeightsShape4D(origInputDepth, origOutputDepth, fieldSize, fieldSize);
                        W = deeplearn_1.Array4D.randUniform(wShape, -1, 1);
                        inputs = [x, W];
                        binary = gpgpu_math.compileProgram(gpgpu, program, inputs, out);
                        benchmark = function () {
                            gpgpu_math.runProgram(binary, inputs, out);
                        };
                        cleanup = function () {
                            out.dispose();
                            x.dispose();
                            W.dispose();
                            texManager.dispose();
                            gpgpu.deleteProgram(binary.webGLProgram);
                            gpgpu.dispose();
                        };
                        return [4, gpgpu.runQuery(benchmark)];
                    case 1:
                        _a.sent();
                        if (!deeplearn_1.ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) return [3, 3];
                        return [4, gpgpu.runQuery(benchmark)];
                    case 2:
                        totalTime = _a.sent();
                        return [3, 4];
                    case 3:
                        start = performance.now();
                        benchmark();
                        out.dataSync();
                        totalTime = performance.now() - start;
                        _a.label = 4;
                    case 4:
                        cleanup();
                        return [2, totalTime];
                }
            });
        });
    };
    return ConvTransposedGPUBenchmark;
}(ConvTransposedBenchmark));
exports.ConvTransposedGPUBenchmark = ConvTransposedGPUBenchmark;
//# sourceMappingURL=conv_transposed_benchmarks.js.map