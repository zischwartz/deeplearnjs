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
var deeplearn_1 = require("../deeplearn");
var benchmark_1 = require("./benchmark");
var UnaryOpsBenchmark = (function (_super) {
    __extends(UnaryOpsBenchmark, _super);
    function UnaryOpsBenchmark() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    UnaryOpsBenchmark.prototype.getUnaryOp = function (option, math) {
        switch (option) {
            case 'log':
                return function (input) { return math.log(input); };
            case 'exp':
                return function (input) { return math.exp(input); };
            case 'neg':
                return function (input) { return math.neg(input); };
            case 'sqrt':
                return function (input) { return math.sqrt(input); };
            case 'abs':
                return function (input) { return math.abs(input); };
            case 'relu':
                return function (input) { return math.relu(input); };
            case 'sigmoid':
                return function (input) { return math.sigmoid(input); };
            case 'sin':
                return function (input) { return math.sin(input); };
            case 'cos':
                return function (input) { return math.cos(input); };
            case 'tan':
                return function (input) { return math.tan(input); };
            case 'asin':
                return function (input) { return math.asin(input); };
            case 'acos':
                return function (input) { return math.acos(input); };
            case 'atan':
                return function (input) { return math.atan(input); };
            case 'sinh':
                return function (input) { return math.sinh(input); };
            case 'cosh':
                return function (input) { return math.cosh(input); };
            case 'tanh':
                return function (input) { return math.tanh(input); };
            default:
                throw new Error("Not found such ops: " + option);
        }
    };
    return UnaryOpsBenchmark;
}(benchmark_1.BenchmarkTest));
exports.UnaryOpsBenchmark = UnaryOpsBenchmark;
var UnaryOpsCPUBenchmark = (function (_super) {
    __extends(UnaryOpsCPUBenchmark, _super);
    function UnaryOpsCPUBenchmark() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    UnaryOpsCPUBenchmark.prototype.run = function (size, option) {
        return __awaiter(this, void 0, void 0, function () {
            var math, input, op, start, end;
            return __generator(this, function (_a) {
                math = new deeplearn_1.NDArrayMathCPU();
                input = deeplearn_1.Array2D.randUniform([size, size], -1, 1);
                op = this.getUnaryOp(option, math);
                start = performance.now();
                math.scope(function () {
                    op(input).get();
                });
                end = performance.now();
                return [2, end - start];
            });
        });
    };
    return UnaryOpsCPUBenchmark;
}(UnaryOpsBenchmark));
exports.UnaryOpsCPUBenchmark = UnaryOpsCPUBenchmark;
var UnaryOpsGPUBenchmark = (function (_super) {
    __extends(UnaryOpsGPUBenchmark, _super);
    function UnaryOpsGPUBenchmark() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    UnaryOpsGPUBenchmark.prototype.run = function (size, option) {
        return __awaiter(this, void 0, void 0, function () {
            var math, input, op, output, benchmark, cleanup, totalTime, start;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        math = new deeplearn_1.NDArrayMathGPU();
                        input = deeplearn_1.Array2D.randUniform([size, size], -1, 1);
                        op = this.getUnaryOp(option, math);
                        benchmark = function () {
                            math.scope(function () {
                                output = op(input);
                            });
                        };
                        cleanup = function () {
                            input.dispose();
                            math.dispose();
                        };
                        return [4, math.getGPGPUContext().runQuery(benchmark)];
                    case 1:
                        _a.sent();
                        if (!deeplearn_1.ENV.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')) return [3, 3];
                        return [4, math.getGPGPUContext().runQuery(benchmark)];
                    case 2:
                        totalTime = _a.sent();
                        return [3, 4];
                    case 3:
                        start = performance.now();
                        benchmark();
                        output.dataSync();
                        totalTime = performance.now() - start;
                        _a.label = 4;
                    case 4:
                        cleanup();
                        return [2, totalTime];
                }
            });
        });
    };
    return UnaryOpsGPUBenchmark;
}(UnaryOpsBenchmark));
exports.UnaryOpsGPUBenchmark = UnaryOpsGPUBenchmark;
//# sourceMappingURL=unary_ops_benchmark.js.map