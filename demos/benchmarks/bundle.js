(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var BenchmarkRun = (function () {
    function BenchmarkRun(name, benchmarkTest) {
        this.name = name;
        this.benchmarkTest = benchmarkTest;
        this.chartData = [];
    }
    return BenchmarkRun;
}());
exports.BenchmarkRun = BenchmarkRun;

},{}],2:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../../src/math/conv_util");
var conv_gpu = require("../../src/math/webgl/conv_gpu");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var test_util = require("../../src/test_util");
var OP_RUNS = 40;
exports.BENCHMARK_TEST = function (size) {
    var inputShapeRCD = [size, size, 1];
    var outputDepth = 1;
    var fieldSize = 11;
    var stride = 1;
    var zeroPad = conv_util.computeDefaultPad(inputShapeRCD, fieldSize, stride);
    var outputShapeRCD = conv_util.computeOutputShape3D(inputShapeRCD, fieldSize, outputDepth, stride, zeroPad);
    var inputTexShapeRC = conv_util.computeTexShapeFrom3D(inputShapeRCD);
    var outputTexShapeRC = conv_util.computeTexShapeFrom3D(outputShapeRCD);
    var weightsTexShapeRC = conv_util.computeWeightsTexShape(inputShapeRCD[2], outputDepth, fieldSize);
    var biasesTexShapeRC = conv_util.computeBiasesTexShape(outputDepth);
    var hasBias = true;
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(conv_gpu.getFragmentShaderSource(inputShapeRCD, outputDepth, fieldSize, stride, zeroPad, hasBias));
    var inputTexture = gpgpu.createMatrixTexture(inputTexShapeRC[0], inputTexShapeRC[1]);
    var weightsTexture = gpgpu.createMatrixTexture(weightsTexShapeRC[0], weightsTexShapeRC[1]);
    var biasesTexture = gpgpu.createMatrixTexture(biasesTexShapeRC[0], biasesTexShapeRC[1]);
    var outputTexture = gpgpu.createMatrixTexture(outputTexShapeRC[0], outputTexShapeRC[1]);
    var inputData = test_util.randomArrayInRange(inputTexShapeRC[0] * inputTexShapeRC[1], -1, 1);
    var weightsData = test_util.randomArrayInRange(weightsTexShapeRC[0] * weightsTexShapeRC[1], -1, 1);
    var biasesData = test_util.randomArrayInRange(biasesTexShapeRC[0] * biasesTexShapeRC[1], -1, 1);
    gpgpu.uploadMatrixToTexture(inputTexture, inputTexShapeRC[0], inputTexShapeRC[1], inputData);
    gpgpu.uploadMatrixToTexture(weightsTexture, weightsTexShapeRC[0], weightsTexShapeRC[1], weightsData);
    gpgpu.uploadMatrixToTexture(biasesTexture, biasesTexShapeRC[0], biasesTexShapeRC[1], biasesData);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        conv_gpu.convolve(gpgpu, program, inputTexture, weightsTexture, biasesTexture, outputTexture, outputTexShapeRC);
    }
    gpgpu.downloadMatrixFromTexture(outputTexture, outputTexShapeRC[0], outputTexShapeRC[1]);
    var end = performance.now();
    var avgTime = (end - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(inputTexture);
    gpgpu.deleteMatrixTexture(weightsTexture);
    gpgpu.deleteMatrixTexture(biasesTexture);
    gpgpu.deleteMatrixTexture(outputTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/conv_util":15,"../../src/math/webgl/conv_gpu":21,"../../src/math/webgl/gpgpu_context":22,"../../src/test_util":32}],3:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../../src/math/conv_util");
var conv_backprop_gpu = require("../../src/math/webgl/conv_backprop_gpu");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var test_util = require("../../src/test_util");
var OP_RUNS = 100;
exports.BENCHMARK_TEST = function (size) {
    var xShapeRCD = [size, size, 1];
    var origOutputDepth = 2;
    var fieldSize = 11;
    var origStride = 1;
    var origPad = 1;
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    var origInputDepth = xShapeRCD[2];
    var src = conv_backprop_gpu.getFragmentShaderConvTransposeSource(xShapeRCD, fieldSize, origInputDepth, origStride, origPad, false);
    var program = gpgpu.createProgram(src);
    var xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);
    var xTex = gpgpu.createMatrixTexture(xTexShapeRC[0], xTexShapeRC[1]);
    var xData = test_util.randomArrayInRange(xTexShapeRC[0] * xTexShapeRC[1], -1, 1);
    gpgpu.uploadMatrixToTexture(xTex, xTexShapeRC[0], xTexShapeRC[1], xData);
    var wTexShapeRC = conv_util.computeWeightsTexShape(origInputDepth, origOutputDepth, fieldSize);
    var wData = test_util.randomArrayInRange(wTexShapeRC[0] * wTexShapeRC[1], -1, 1);
    var wTex = gpgpu.createMatrixTexture(wTexShapeRC[0], wTexShapeRC[1]);
    gpgpu.uploadMatrixToTexture(wTex, wTexShapeRC[0], wTexShapeRC[1], wData);
    var dilatedRC = conv_util.computeDilatedRC([xShapeRCD[0], xShapeRCD[1]], origStride);
    var pad = fieldSize - 1 - origPad;
    var resultShapeRCD = conv_util.computeOutputShape3D([dilatedRC[0], dilatedRC[1], origOutputDepth], fieldSize, origInputDepth, 1, pad);
    var resultTexRC = conv_util.computeTexShapeFrom3D(resultShapeRCD);
    var resultTex = gpgpu.createMatrixTexture(resultTexRC[0], resultTexRC[1]);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        conv_backprop_gpu.convTranspose(gpgpu, program, xTex, wTex, null, resultTex, resultTexRC);
    }
    var y = gpgpu.downloadMatrixFromTexture(resultTex, resultTexRC[0], resultTexRC[1]);
    var end = performance.now();
    var avgTime = (end - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(resultTex);
    gpgpu.deleteMatrixTexture(xTex);
    gpgpu.deleteMatrixTexture(wTex);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/conv_util":15,"../../src/math/webgl/conv_backprop_gpu":20,"../../src/math/webgl/gpgpu_context":22,"../../src/test_util":32}],4:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_cpu_1 = require("../../src/math/math_cpu");
var ndarray_1 = require("../../src/math/ndarray");
var OPS_PER_RUN = 10;
exports.BENCHMARK_TEST = function (size) {
    var math = new math_cpu_1.NDArrayMathCPU();
    var a = ndarray_1.NDArray.randUniform([size, size], -1, 1);
    var start = performance.now();
    for (var i = 0; i < OPS_PER_RUN; i++) {
        math.logSumExp(a);
    }
    var end = performance.now();
    return (end - start) / OPS_PER_RUN;
};

},{"../../src/math/math_cpu":18,"../../src/math/ndarray":19}],5:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var logsumexp_gpu = require("../../src/math/webgl/logsumexp_gpu");
var test_util = require("../../src/test_util");
var OP_RUNS = 100;
exports.BENCHMARK_TEST = function (size) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(logsumexp_gpu.getFragmentShaderSource(size, size));
    var aTexture = gpgpu.createMatrixTexture(size, size);
    var resultTexture = gpgpu.createMatrixTexture(size, size);
    var a = test_util.randomArrayInRange(size * size, -1, 1);
    gpgpu.uploadMatrixToTexture(aTexture, size, size, a);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        logsumexp_gpu.logSumExp(gpgpu, program, aTexture, size, size, resultTexture);
    }
    gpgpu.downloadMatrixFromTexture(resultTexture, size, size);
    var avgTime = (performance.now() - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/webgl/gpgpu_context":22,"../../src/math/webgl/logsumexp_gpu":24,"../../src/test_util":32}],6:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var benchmark_1 = require("./benchmark");
var conv_gpu_benchmark = require("./conv_gpu_benchmark");
var conv_transpose_gpu_benchmark = require("./conv_transpose_gpu_benchmark");
var logsumexp_cpu_benchmark = require("./logsumexp_cpu_benchmark");
var logsumexp_gpu_benchmark = require("./logsumexp_gpu_benchmark");
var max_pool_gpu_benchmark = require("./max_pool_gpu_benchmark");
var mulmat_cpu_benchmark = require("./mulmat_cpu_benchmark");
var mulmat_gpu_benchmark = require("./mulmat_gpu_benchmark");
exports.BENCHMARK_RUN_GROUPS = [
    {
        name: 'Matrix Multiplication (CPU vs GPU): matmul([size, size], [size, size])',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('mulmat_gpu', mulmat_gpu_benchmark.BENCHMARK_TEST),
            new benchmark_1.BenchmarkRun('mulmat_cpu', mulmat_cpu_benchmark.BENCHMARK_TEST)
        ],
    },
    {
        name: 'Convolution (GPU): conv over image [size, size, 1]',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [new benchmark_1.BenchmarkRun('d1=1, d2=1, f=11, s=1', conv_gpu_benchmark.BENCHMARK_TEST)],
    },
    {
        name: 'Convolution Transposed (GPU): deconv over image [size, size, 1]',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [new benchmark_1.BenchmarkRun('d1=1, d2=1, f=11, s=1', conv_transpose_gpu_benchmark.BENCHMARK_TEST)],
    },
    {
        name: 'Max pool (GPU)',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [new benchmark_1.BenchmarkRun('d1=1, d2=1, f=11, s=1', max_pool_gpu_benchmark.MAX_POOL_BENCHMARK_TEST)],
    },
    {
        name: 'LogSumExp (CPU vs GPU): input [size, size]',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('logsumexp_gpu', logsumexp_gpu_benchmark.BENCHMARK_TEST),
            new benchmark_1.BenchmarkRun('logsumexp_cpu', logsumexp_cpu_benchmark.BENCHMARK_TEST)
        ],
    }
];

},{"./benchmark":1,"./conv_gpu_benchmark":2,"./conv_transpose_gpu_benchmark":3,"./logsumexp_cpu_benchmark":4,"./logsumexp_gpu_benchmark":5,"./max_pool_gpu_benchmark":8,"./mulmat_cpu_benchmark":9,"./mulmat_gpu_benchmark":10}],7:[function(require,module,exports){
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
require("../demo-header");
require("../demo-footer");
var polymer_spec_1 = require("../polymer-spec");
var math_benchmark_run_groups_1 = require("./math-benchmark-run-groups");
exports.MathBenchmarkPolymer = polymer_spec_1.PolymerElement({ is: 'math-benchmark', properties: { benchmarkRunGroupNames: Array } });
var MathBenchmark = (function (_super) {
    __extends(MathBenchmark, _super);
    function MathBenchmark() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    MathBenchmark.prototype.ready = function () {
        var _this = this;
        var benchmarkRunGroupNames = [];
        this.stopMessages = [];
        for (var i = 0; i < math_benchmark_run_groups_1.BENCHMARK_RUN_GROUPS.length; i++) {
            benchmarkRunGroupNames.push(math_benchmark_run_groups_1.BENCHMARK_RUN_GROUPS[i].name);
            this.stopMessages.push(false);
        }
        this.benchmarkRunGroupNames = benchmarkRunGroupNames;
        setTimeout(function () {
            var runButtons = _this.querySelectorAll('.run-test');
            var stopButtons = _this.querySelectorAll('.run-stop');
            var _loop_1 = function (i) {
                runButtons[i].addEventListener('click', function () {
                    _this.runBenchmarkGroup(i);
                });
                stopButtons[i].addEventListener('click', function () {
                    _this.stopMessages[i] = true;
                });
            };
            for (var i = 0; i < runButtons.length; i++) {
                _loop_1(i);
            }
        }, 0);
    };
    MathBenchmark.prototype.runBenchmarkGroup = function (benchmarkRunGroupIndex) {
        var benchmarkRunGroup = math_benchmark_run_groups_1.BENCHMARK_RUN_GROUPS[benchmarkRunGroupIndex];
        var canvas = this.querySelectorAll('.run-plot')[benchmarkRunGroupIndex];
        var context = canvas.getContext('2d');
        var datasets = [];
        for (var i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
            var hue = Math.floor(360 * i / benchmarkRunGroup.benchmarkRuns.length);
            datasets.push({
                data: benchmarkRunGroup.benchmarkRuns[i].chartData,
                fill: false,
                label: benchmarkRunGroup.benchmarkRuns[i].name,
                borderColor: 'hsl(' + hue + ', 100%, 40%)',
                backgroundColor: 'hsl(' + hue + ', 100%, 70%)',
                pointRadius: 0,
                pointHitRadius: 5,
                borderWidth: 1,
                lineTension: 0
            });
        }
        var chart = new Chart(context, {
            type: 'line',
            data: { datasets: datasets },
            options: {
                animation: { duration: 0 },
                responsive: false,
                scales: {
                    xAxes: [{
                            type: 'linear',
                            position: 'bottom',
                            ticks: {
                                min: benchmarkRunGroup.min,
                                max: benchmarkRunGroup.max,
                                stepSize: benchmarkRunGroup.stepSize,
                                callback: function (label) {
                                    return benchmarkRunGroup.stepToSizeTransformation != null ?
                                        benchmarkRunGroup.stepToSizeTransformation(+label) :
                                        +label;
                                }
                            }
                        }],
                    yAxes: [{
                            ticks: {
                                callback: function (label, index, labels) {
                                    return label + 'ms';
                                }
                            },
                        }]
                },
                tooltips: { mode: 'label' },
                title: { text: benchmarkRunGroup.name }
            }
        });
        canvas.style.display = 'none';
        var runMessage = this.querySelectorAll('.run-message')[benchmarkRunGroupIndex];
        runMessage.style.display = 'block';
        var runNumbersTable = this.querySelectorAll('.run-numbers-table')[benchmarkRunGroupIndex];
        runNumbersTable.innerHTML = '';
        runNumbersTable.style.display = 'none';
        var headers = ['size'];
        for (var i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
            headers.push(benchmarkRunGroup.benchmarkRuns[i].name);
        }
        runNumbersTable.appendChild(this.buildRunNumbersRow(headers));
        this.runBenchmarkSteps(chart, benchmarkRunGroup, benchmarkRunGroupIndex, benchmarkRunGroup.min);
    };
    MathBenchmark.prototype.buildRunNumbersRow = function (values) {
        var runNumberRowElement = document.createElement('div');
        runNumberRowElement.className = 'run-numbers-row math-benchmark';
        for (var i = 0; i < values.length; i++) {
            var runNumberCellElement = document.createElement('div');
            runNumberCellElement.className = 'run-numbers-cell math-benchmark';
            runNumberCellElement.innerText = values[i];
            runNumberRowElement.appendChild(runNumberCellElement);
        }
        return runNumberRowElement;
    };
    MathBenchmark.prototype.runBenchmarkSteps = function (chart, benchmarkRunGroup, benchmarkRunGroupIndex, step) {
        var _this = this;
        var runNumbersTable = this.querySelectorAll('.run-numbers-table')[benchmarkRunGroupIndex];
        if (step > benchmarkRunGroup.max ||
            this.stopMessages[benchmarkRunGroupIndex]) {
            this.stopMessages[benchmarkRunGroupIndex] = false;
            runNumbersTable.style.display = '';
            var canvas = this.querySelectorAll('.run-plot')[benchmarkRunGroupIndex];
            canvas.style.display = 'block';
            chart.update();
            var runMessage = this.querySelectorAll('.run-message')[benchmarkRunGroupIndex];
            runMessage.style.display = 'none';
            return;
        }
        var runNumberRowElement = document.createElement('div');
        runNumberRowElement.className = 'run-numbers-row math-benchmark';
        var rowValues = ['' + step];
        for (var i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
            var benchmarkRun = benchmarkRunGroup.benchmarkRuns[i];
            var benchmarkTest = benchmarkRun.benchmarkTest;
            var size = benchmarkRunGroup.stepToSizeTransformation != null ?
                benchmarkRunGroup.stepToSizeTransformation(step) :
                step;
            var resultString = void 0;
            var logString = void 0;
            var time = 0;
            var success = true;
            try {
                time = benchmarkTest(size);
                resultString = time.toFixed(3) + 'ms';
                logString = resultString;
            }
            catch (e) {
                success = false;
                resultString = 'Error';
                logString = e.message;
            }
            if (time >= 0) {
                if (success) {
                    benchmarkRun.chartData.push({ x: step, y: time });
                }
                rowValues.push(resultString);
            }
            console.log(benchmarkRun.name + '[' + step + ']: ' + logString);
        }
        runNumbersTable.appendChild(this.buildRunNumbersRow(rowValues));
        step += benchmarkRunGroup.stepSize;
        setTimeout(function () { return _this.runBenchmarkSteps(chart, benchmarkRunGroup, benchmarkRunGroupIndex, step); }, 100);
    };
    return MathBenchmark;
}(exports.MathBenchmarkPolymer));
exports.MathBenchmark = MathBenchmark;
document.registerElement(MathBenchmark.prototype.is, MathBenchmark);

},{"../demo-footer":11,"../demo-header":12,"../polymer-spec":13,"./math-benchmark-run-groups":6}],8:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../../src/math/conv_util");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var max_pool_gpu = require("../../src/math/webgl/max_pool_gpu");
var test_util = require("../../src/test_util");
var OP_RUNS = 40;
exports.MAX_POOL_BENCHMARK_TEST = function (size) {
    var inputShapeRCD = [size, size, 1];
    var outputDepth = 1;
    var fieldSize = 11;
    var stride = 1;
    var zeroPad = conv_util.computeDefaultPad(inputShapeRCD, fieldSize, stride);
    var outputShapeRCD = conv_util.computeOutputShape3D(inputShapeRCD, fieldSize, outputDepth, stride, zeroPad);
    var inputTexShapeRC = conv_util.computeTexShapeFrom3D(inputShapeRCD);
    var outputTexShapeRC = conv_util.computeTexShapeFrom3D(outputShapeRCD);
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(max_pool_gpu.getFragmentShaderMaxPoolSource(inputShapeRCD, fieldSize, stride, zeroPad));
    var inputTexture = gpgpu.createMatrixTexture(inputTexShapeRC[0], inputTexShapeRC[1]);
    var outputTexture = gpgpu.createMatrixTexture(outputTexShapeRC[0], outputTexShapeRC[1]);
    var inputData = test_util.randomArrayInRange(inputTexShapeRC[0] * inputTexShapeRC[1], -1, 1);
    gpgpu.uploadMatrixToTexture(inputTexture, inputTexShapeRC[0], inputTexShapeRC[1], inputData);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        max_pool_gpu.maxPoolCommon(gpgpu, program, inputTexture, outputTexture, outputTexShapeRC);
    }
    gpgpu.downloadMatrixFromTexture(outputTexture, outputTexShapeRC[0], outputTexShapeRC[1]);
    var end = performance.now();
    var avgTime = (end - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(inputTexture);
    gpgpu.deleteMatrixTexture(outputTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};
exports.MAX_POOL_POSNS_BENCHMARK_TEST = function (size) {
    var inputShapeRCD = [size, size, 1];
    var outputDepth = 1;
    var fieldSize = 11;
    var stride = 1;
    var zeroPad = conv_util.computeDefaultPad(inputShapeRCD, fieldSize, stride);
    var outputShapeRCD = conv_util.computeOutputShape3D(inputShapeRCD, fieldSize, outputDepth, stride, zeroPad);
    var inputTexShapeRC = conv_util.computeTexShapeFrom3D(inputShapeRCD);
    var outputTexShapeRC = conv_util.computeTexShapeFrom3D(outputShapeRCD);
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(max_pool_gpu.getFragmentShaderMaxPoolPositionsSource(inputShapeRCD, fieldSize, stride, zeroPad));
    var inputTexture = gpgpu.createMatrixTexture(inputTexShapeRC[0], inputTexShapeRC[1]);
    var outputTexture = gpgpu.createMatrixTexture(outputTexShapeRC[0], outputTexShapeRC[1]);
    var inputData = test_util.randomArrayInRange(inputTexShapeRC[0] * inputTexShapeRC[1], -1, 1);
    gpgpu.uploadMatrixToTexture(inputTexture, inputTexShapeRC[0], inputTexShapeRC[1], inputData);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        max_pool_gpu.maxPoolCommon(gpgpu, program, inputTexture, outputTexture, outputTexShapeRC);
    }
    gpgpu.downloadMatrixFromTexture(outputTexture, outputTexShapeRC[0], outputTexShapeRC[1]);
    var end = performance.now();
    var avgTime = (end - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(inputTexture);
    gpgpu.deleteMatrixTexture(outputTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/conv_util":15,"../../src/math/webgl/gpgpu_context":22,"../../src/math/webgl/max_pool_gpu":25,"../../src/test_util":32}],9:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_cpu_1 = require("../../src/math/math_cpu");
var ndarray_1 = require("../../src/math/ndarray");
var OPS_PER_SMALL_RUN = 1;
exports.BENCHMARK_TEST = function (size) {
    if (size > 512) {
        return -1;
    }
    var math = new math_cpu_1.NDArrayMathCPU();
    var a = ndarray_1.NDArray.randUniform([size, size], -1, 1);
    var b = ndarray_1.NDArray.randUniform([size, size], -1, 1);
    var runs = (size < 192) ? OPS_PER_SMALL_RUN : 1;
    var start = performance.now();
    for (var i = 0; i < runs; i++) {
        math.matMul(a, b);
    }
    var end = performance.now();
    return (end - start) / runs;
};

},{"../../src/math/math_cpu":18,"../../src/math/ndarray":19}],10:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_1 = require("../../src/math/math");
var ndarray_1 = require("../../src/math/ndarray");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var mulmat_gpu = require("../../src/math/webgl/mulmat_gpu");
var mulmat_packed_gpu = require("../../src/math/webgl/mulmat_packed_gpu");
var test_util = require("../../src/test_util");
var OP_RUNS = 40;
exports.BENCHMARK_TEST = function (size) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var aTexture = gpgpu.createMatrixTexture(size, size);
    var bTexture = gpgpu.createMatrixTexture(size, size);
    var resultTexture = gpgpu.createMatrixTexture(size, size);
    var aArr = new ndarray_1.Array2D([size, size], { texture: aTexture, textureShapeRC: [size, size] });
    var bArr = new ndarray_1.Array2D([size, size], { texture: bTexture, textureShapeRC: [size, size] });
    var resArr = new ndarray_1.Array2D([size, size], { texture: resultTexture, textureShapeRC: [size, size] });
    var program = gpgpu.createProgram(mulmat_gpu.getFragmentShader(aArr, bArr, resArr, math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.REGULAR));
    var a = test_util.randomArrayInRange(size * size, -1, 1);
    var b = test_util.randomArrayInRange(size * size, -1, 1);
    gpgpu.uploadMatrixToTexture(aTexture, size, size, a);
    gpgpu.uploadMatrixToTexture(bTexture, size, size, b);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        mulmat_gpu.multiplyMatrix(gpgpu, program, aTexture, bTexture, resultTexture, [size, size]);
    }
    var actual = gpgpu.downloadMatrixFromTexture(resultTexture, size, size);
    var avgTime = (performance.now() - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(bTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};
exports.BENCHMARK_TEST_PACKED = function (size) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(mulmat_packed_gpu.getFragmentShaderSource(size, math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.REGULAR));
    var aTexture = gpgpu.createPackedMatrixTexture(size, size);
    var bTexture = gpgpu.createPackedMatrixTexture(size, size);
    var resultTexture = gpgpu.createPackedMatrixTexture(size, size);
    var a = test_util.randomArrayInRange(size * size, -1, 1);
    var b = test_util.randomArrayInRange(size * size, -1, 1);
    gpgpu.uploadMatrixToPackedTexture(aTexture, size, size, a);
    gpgpu.uploadMatrixToPackedTexture(bTexture, size, size, b);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        mulmat_packed_gpu.multiplyMatrixPacked(gpgpu, program, aTexture, bTexture, resultTexture, [size, size]);
    }
    var actual = gpgpu.downloadMatrixFromPackedTexture(resultTexture, size, size);
    var avgTime = (performance.now() - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(bTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/math":17,"../../src/math/ndarray":19,"../../src/math/webgl/gpgpu_context":22,"../../src/math/webgl/mulmat_gpu":26,"../../src/math/webgl/mulmat_packed_gpu":27,"../../src/test_util":32}],11:[function(require,module,exports){
Polymer({ is: 'demo-footer' });

},{}],12:[function(require,module,exports){
Polymer({ is: 'demo-header' });

},{}],13:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function PolymerElement(spec) {
    return Polymer.Class(spec);
}
exports.PolymerElement = PolymerElement;

},{}],14:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
function assertConcat3DShapesMatch(x1Shape, x2Shape, axis, errorMessagePrefix) {
    if (errorMessagePrefix === void 0) { errorMessagePrefix = ''; }
    util.assert(x1Shape.length === 3, errorMessagePrefix + 'Concat3D x1 shape should be of rank 3.');
    util.assert(x2Shape.length === 3, errorMessagePrefix + 'Concat3D x2 shape should be of rank 3.');
    util.assert(axis >= 0 && axis < 3, 'Axis for concat3D must be between 0 and 2.');
    for (var i = 0; i < 3; i++) {
        util.assert((i === axis) || (x1Shape[i] === x2Shape[i]), errorMessagePrefix +
            ("Shape (" + x1Shape + ") does not match (" + x2Shape + ") along ") +
            "non-concatenated axis.");
    }
}
exports.assertConcat3DShapesMatch = assertConcat3DShapesMatch;
function computeConcat3DOutputShape(x1Shape, x2Shape, axis) {
    util.assert(x1Shape.length === 3, 'Concat3D x1 shape should be of rank 3.');
    util.assert(x2Shape.length === 3, 'Concat3D x2shape should be of rank 3.');
    var outputShape = x1Shape.slice();
    outputShape[axis] += x2Shape[axis];
    return outputShape;
}
exports.computeConcat3DOutputShape = computeConcat3DOutputShape;

},{"../util":33}],15:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
function computeOutputShape3D(inputShapeRowColDepth, fieldSize, depth, stride, zeroPad) {
    if (zeroPad == null) {
        zeroPad = computeDefaultPad(inputShapeRowColDepth, fieldSize, stride);
    }
    var inputRows = inputShapeRowColDepth[0];
    var inputCols = inputShapeRowColDepth[1];
    var outputRows = (inputRows - fieldSize + 2 * zeroPad) / stride + 1;
    util.assert(util.isInt(outputRows), "The output # of rows (" + outputRows + ") must be an integer. Change the " +
        "stride and/or zero pad parameters");
    var outputCols = (inputCols - fieldSize + 2 * zeroPad) / stride + 1;
    util.assert(util.isInt(outputCols), "The output # of columns (" + outputCols + ") must be an integer. Change " +
        "the stride and/or zero pad parameters");
    return [outputRows, outputCols, depth];
}
exports.computeOutputShape3D = computeOutputShape3D;
function computeDefaultPad(inputShape, fieldSize, stride) {
    return Math.floor((inputShape[0] * (stride - 1) - stride + fieldSize) / 2);
}
exports.computeDefaultPad = computeDefaultPad;
function computeTexShapeFrom3D(shapeRowColDepth) {
    return [shapeRowColDepth[0], shapeRowColDepth[1] * shapeRowColDepth[2]];
}
exports.computeTexShapeFrom3D = computeTexShapeFrom3D;
function computeWeightsShape4D(inputDepth, outputDepth, fSize) {
    return [fSize, fSize, inputDepth, outputDepth];
}
exports.computeWeightsShape4D = computeWeightsShape4D;
function computeWeightsTexShape(inputDepth, outputDepth, fieldSize) {
    return [fieldSize * fieldSize * inputDepth, outputDepth];
}
exports.computeWeightsTexShape = computeWeightsTexShape;
function computeBiasesTexShape(outputDepth) {
    return [1, outputDepth];
}
exports.computeBiasesTexShape = computeBiasesTexShape;
function computeDilatedRC(rc, origStride) {
    var rowsDilated = (rc[0] - 1) * origStride + 1;
    var colsDilated = (rc[1] - 1) * origStride + 1;
    return [rowsDilated, colsDilated];
}
exports.computeDilatedRC = computeDilatedRC;

},{"../util":33}],16:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function validateShapes(sourceSize, destSize) {
    var srcArea = sourceSize[0] * sourceSize[1];
    var dstArea = destSize[0] * destSize[1];
    if (srcArea !== dstArea) {
        var srcStr = '[' + sourceSize[0] + ', ' + sourceSize[1] + ']';
        var dstStr = '[' + destSize[0] + ', ' + destSize[1] + ']';
        throw new Error('copy2D shapes have different areas:\n  sourceSize ' + srcStr +
            ', area ' + srcArea + '\n  destSize ' + dstStr + ', area ' + dstArea);
    }
}
exports.validateShapes = validateShapes;

},{}],17:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
var concat3d_util = require("./concat3d_util");
var copy2d_util = require("./copy2d_util");
var ndarray_1 = require("./ndarray");
var NDArrayMath = (function () {
    function NDArrayMath(safeMode) {
        this.safeMode = safeMode;
        this.ndarrayScopes = [];
        this.ndarraysToKeep = [];
        this.activeScopeNDArraysToKeep = [];
    }
    NDArrayMath.prototype.scope = function (scopeFn) {
        var _this = this;
        this.startScope();
        var keepFn = function (ndarray) { return _this.keep(ndarray); };
        var trackFn = function (ndarray) { return _this.track(ndarray); };
        var result = scopeFn(keepFn, trackFn);
        this.endScope(result);
        return result;
    };
    NDArrayMath.prototype.startScope = function () {
        var newScope = [];
        this.ndarrayScopes.push(newScope);
        this.activeScope = newScope;
        var newNDArraysToKeep = [];
        this.ndarraysToKeep.push(newNDArraysToKeep);
        this.activeScopeNDArraysToKeep = newNDArraysToKeep;
    };
    NDArrayMath.prototype.endScope = function (result) {
        var _this = this;
        for (var i = 0; i < this.activeScope.length; i++) {
            var ndarray = this.activeScope[i];
            if (this.isNDArrayDataInList(ndarray, this.activeScopeNDArraysToKeep) ||
                (result != null && result instanceof ndarray_1.NDArray &&
                    ndarray.getData() === result.getData())) {
                continue;
            }
            ndarray.dispose();
        }
        this.ndarrayScopes.pop();
        this.activeScope = this.ndarrayScopes.length === 0 ?
            null :
            this.ndarrayScopes[this.ndarrayScopes.length - 1];
        if (result instanceof ndarray_1.NDArray &&
            !this.isNDArrayDataInList(result, this.activeScopeNDArraysToKeep)) {
            this.track(result);
        }
        else if (Array.isArray(result)) {
            result.forEach(function (r) {
                if (r instanceof ndarray_1.NDArray &&
                    !_this.isNDArrayDataInList(r, _this.activeScopeNDArraysToKeep)) {
                    _this.track(r);
                }
            });
        }
        this.ndarraysToKeep.pop();
        this.activeScopeNDArraysToKeep = this.ndarraysToKeep.length === 0 ?
            null :
            this.ndarraysToKeep[this.ndarraysToKeep.length - 1];
    };
    NDArrayMath.prototype.isNDArrayDataInList = function (ndarray, ndarrayList) {
        for (var i = 0; i < ndarrayList.length; i++) {
            if (ndarrayList[i].getData() === ndarray.getData()) {
                return true;
            }
        }
        return false;
    };
    NDArrayMath.prototype.keep = function (result) {
        if (this.activeScope == null) {
            if (this.safeMode) {
                throw new Error('You are using math in safe mode. Enclose all ' +
                    'math.method() calls inside a scope: ' +
                    'math.scope(() => {math.method();...}) to avoid memory ' +
                    'leaks.');
            }
            return result;
        }
        this.activeScopeNDArraysToKeep.push(result);
        return result;
    };
    NDArrayMath.prototype.track = function (result) {
        if (this.activeScope == null) {
            if (this.safeMode) {
                throw new Error('You are using math in safe mode. Enclose all ' +
                    'math.method() calls inside a scope: ' +
                    'math.scope(() => {math.method();...}) to avoid memory ' +
                    'leaks.');
            }
            return result;
        }
        this.activeScope.push(result);
        return result;
    };
    NDArrayMath.prototype.matMul = function (a, b, aOrientation, bOrientation) {
        if (aOrientation === void 0) { aOrientation = MatrixOrientation.REGULAR; }
        if (bOrientation === void 0) { bOrientation = MatrixOrientation.REGULAR; }
        var innerShapeA = (aOrientation === MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
        var innerShapeB = (bOrientation === MatrixOrientation.REGULAR) ? b.shape[0] : b.shape[1];
        util.assert(a.rank === 2 && b.rank === 2, "Error in matMul: inputs must be rank 2, got ranks " + a.rank +
            ("and " + b.rank + "."));
        util.assert(innerShapeA === innerShapeB, "Error in matMul: inner shapes (" + innerShapeA + ") and (" +
            (innerShapeB + ") of NDArrays with shapes " + a.shape + " and ") +
            (b.shape + " and orientations " + MatrixOrientation[aOrientation]) +
            (" and " + MatrixOrientation[bOrientation] + " must match."));
        return this.track(this.matMulInternal(a, b, aOrientation, bOrientation));
    };
    NDArrayMath.prototype.vectorTimesMatrix = function (v, matrix) {
        util.assert(v.rank === 1, "Error in vectorTimesMatrix: first input must be rank 1, but got " +
            ("rank " + v.rank + "."));
        util.assert(matrix.rank === 2, "Error in vectorTimesMatrix: second input must be rank 2, but got " +
            ("rank " + matrix.rank + "."));
        util.assert(v.size === matrix.shape[0], "Error in vectorTimesMatrix: size of first rank 1 input (" + v.size + ") " +
            "must match inner dimension of second rank 2 input, but got " +
            ("rank " + matrix.rank + "."));
        return this.matMul(v.as2D(1, v.size), matrix).as1D();
    };
    NDArrayMath.prototype.matrixTimesVector = function (matrix, v) {
        util.assert(v.rank === 1, "Error in vectorTimesMatrix: second input must rank 1, but got " +
            ("rank " + v.rank + "."));
        util.assert(matrix.rank === 2, "Error in vectorTimesMatrix: first input must be a rank 2, but got " +
            ("rank " + matrix.rank + "."));
        util.assert(v.size === matrix.shape[1], "Error in vectorTimesMatrix: size of first rank 1 input " + v.size + " " +
            "must match inner dimension of second rank 2 input, but got " +
            ("shape " + matrix.shape + "."));
        return this.matMul(matrix, v.as2D(v.size, 1)).as1D();
    };
    NDArrayMath.prototype.dotProduct = function (v1, v2) {
        util.assert(v1.rank === 1 && v2.rank === 1, "Error in dotProduct: inputs must be rank 1, but got ranks " +
            (v1.rank + " and " + v2.rank + "."));
        util.assert(v1.size === v2.size, "Error in dotProduct: size of inputs (" + v1.size + ") and (" +
            (v2.size + ") must match."));
        return this.matMul(v1.as2D(1, v1.size), v2.as2D(v2.size, 1)).asScalar();
    };
    NDArrayMath.prototype.outerProduct = function (v1, v2) {
        util.assert(v1.rank === 1 && v2.rank === 1, "Error in outerProduct: inputs must be rank 1, but got ranks " +
            (v1.rank + " and " + v2.rank + "."));
        return this.matMul(v1.as2D(v1.size, 1), v2.as2D(1, v2.size));
    };
    NDArrayMath.prototype.clone = function (ndarray) {
        return this.track(this.cloneInternal(ndarray));
    };
    NDArrayMath.prototype.reshape = function (ndarray, newShape) {
        util.assert(ndarray.size === util.sizeFromShape(newShape), "Error in reshape: old size " + ndarray.size + " must match new size " +
            (util.sizeFromShape(newShape) + "."));
        return this.track(this.reshapeInternal(ndarray, newShape));
    };
    NDArrayMath.prototype.slice2D = function (input, begin, size) {
        util.assert(begin[0] + size[0] <= input.shape[0] &&
            begin[1] + size[1] <= input.shape[1], "Error in slice2D: requested start position " + begin + " and size " +
            (size + " would overflow input of shape " + input.shape + "."));
        return this.track(this.slice2DInternal(input, begin, size));
    };
    NDArrayMath.prototype.copy2D = function (source, sourceBegin, sourceSize, dest, destBegin, destSize) {
        util.assert(sourceBegin[0] + sourceSize[0] <= source.shape[0] &&
            sourceBegin[1] + sourceSize[1] <= source.shape[1], "Error in copy2D: requested source start position " + sourceBegin + " " +
            ("and source size " + sourceSize + " would overflow source NDArray") +
            ("of shape " + source.shape + "."));
        util.assert(destBegin[0] + destSize[0] <= dest.shape[0] &&
            destBegin[1] + destSize[1] <= dest.shape[1], "Error in copy2D: requested dest start position " + destBegin + " " +
            ("and source size " + destSize + " would overflow dest NDArray of") +
            ("shape " + dest.shape + "."));
        copy2d_util.validateShapes(sourceSize, destSize);
        return this.copy2DInternal(source, sourceBegin, sourceSize, dest, destBegin, destSize);
    };
    NDArrayMath.prototype.concat3D = function (ndarray1, ndarray2, axis) {
        concat3d_util.assertConcat3DShapesMatch(ndarray1.shape, ndarray2.shape, axis, 'Error in concat3d: ');
        return this.track(this.concat3DInternal(ndarray1, ndarray2, axis));
    };
    NDArrayMath.prototype.logSumExp = function (ndarray) {
        return this.track(this.logSumExpInternal(ndarray));
    };
    NDArrayMath.prototype.sum = function (ndarray) {
        return this.track(this.sumInternal(ndarray));
    };
    NDArrayMath.prototype.argMin = function (ndarray) {
        return this.track(this.argMinInternal(ndarray));
    };
    NDArrayMath.prototype.argMax = function (ndarray) {
        return this.track(this.argMaxInternal(ndarray));
    };
    NDArrayMath.prototype.argMaxEquals = function (x1, x2) {
        util.assertShapesMatch(x1.shape, x2.shape, 'Error in argMaxEquals: ');
        return this.track(this.argMaxEqualsInternal(x1, x2));
    };
    NDArrayMath.prototype.topK = function (ndarray, k) {
        util.assert(k <= ndarray.size, "Error in topK: k value (" + k + ") must be less than size of input " +
            ("ndarray, got shape " + ndarray.shape + "."));
        var result = this.topKInternal(ndarray, k);
        this.track(result.values);
        this.track(result.indices);
        return result;
    };
    NDArrayMath.prototype.min = function (ndarray) {
        return this.track(this.minInternal(ndarray));
    };
    NDArrayMath.prototype.max = function (ndarray) {
        return this.track(this.maxInternal(ndarray));
    };
    NDArrayMath.prototype.softmax = function (x) {
        var _this = this;
        return this.scope(function () {
            var lse = _this.logSumExp(x);
            var logResult = _this.arrayMinusScalar(x, lse);
            return _this.exp(logResult);
        });
    };
    NDArrayMath.prototype.switchDim = function (a, newDim) {
        util.assert(a.rank === newDim.length, "Error in switchDim: length of input shape " + a.shape + " " +
            ("must match size of newDim array " + newDim + "."));
        return this.track(this.switchDimInternal(a, newDim));
    };
    NDArrayMath.prototype.scalarPlusArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarPlusArray: first argument must be rank 0, but got " +
            ("rank " + c.rank + "."));
        return this.track(this.scalarPlusArrayInternal(c, a));
    };
    NDArrayMath.prototype.scalarMinusArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarMinusArray: first argument must be rank 0, but got " +
            ("rank " + c.rank + "."));
        return this.track(this.scalarMinusArrayInternal(c, a));
    };
    NDArrayMath.prototype.arrayMinusScalar = function (a, c) {
        util.assert(c.size === 1, "Error in arrayMinusScalar: second argument must be rank 0, but " +
            ("got rank " + c.rank + "."));
        return this.track(this.arrayMinusScalarInternal(a, c));
    };
    NDArrayMath.prototype.neg = function (a) {
        return this.track(this.negInternal(a));
    };
    NDArrayMath.prototype.add = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in add: ');
        return this.track(this.addInternal(a, b));
    };
    NDArrayMath.prototype.sub = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in sub: ');
        return this.track(this.subInternal(a, b));
    };
    NDArrayMath.prototype.elementWiseMul = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in elementWiseMul: ');
        return this.track(this.elementWiseMulInternal(a, b));
    };
    NDArrayMath.prototype.divide = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in divide: ');
        return this.track(this.divideInternal(a, b));
    };
    NDArrayMath.prototype.scalarDividedByArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarDividedByArray: first argument must be rank 0, but " +
            ("got NDArray of rank " + c.rank + "."));
        return this.track(this.scalarDividedByArrayInternal(c, a));
    };
    NDArrayMath.prototype.arrayDividedByScalar = function (a, c) {
        util.assert(c.size === 1, "Error in arrayDividedByScalar: second argument must be rank 0, " +
            ("but got NDArray of rank " + c.rank + "."));
        return this.track(this.arrayDividedByScalarInternal(a, c));
    };
    NDArrayMath.prototype.exp = function (ndarray) {
        return this.track(this.expInternal(ndarray));
    };
    NDArrayMath.prototype.log = function (ndarray) {
        return this.track(this.logInternal(ndarray));
    };
    NDArrayMath.prototype.relu = function (ndarray) {
        return this.track(this.reluInternal(ndarray));
    };
    NDArrayMath.prototype.sigmoid = function (ndarray) {
        return this.track(this.sigmoidInternal(ndarray));
    };
    NDArrayMath.prototype.tanh = function (ndarray) {
        return this.track(this.tanhInternal(ndarray));
    };
    NDArrayMath.prototype.sin = function (ndarray) {
        return this.track(this.sinInternal(ndarray));
    };
    NDArrayMath.prototype.step = function (ndarray) {
        return this.track(this.stepInternal(ndarray));
    };
    NDArrayMath.prototype.scaledArrayAdd = function (c1, a, c2, b) {
        util.assert(c1.size === 1, "Error in scaledArrayAdd: first argument must rank 0, but got " +
            (" rank " + c1.rank + "."));
        util.assert(c2.size === 1, "Error in scaledArrayAdd: third argument must be rank 0, but got " +
            ("NDArray of rank " + c2.rank + "."));
        util.assertShapesMatch(a.shape, b.shape, 'Error in scaledArrayAdd: ');
        return this.track(this.scaledArrayAddInternal(c1, a, c2, b));
    };
    NDArrayMath.prototype.scalarTimesArray = function (c, a) {
        util.assert(c.size === 1, "Error in arrayDividedByScalar: first argument must be rank 0, but " +
            ("got rank " + c.rank + "."));
        return this.track(this.scalarTimesArrayInternal(c, a));
    };
    NDArrayMath.prototype.elementWiseMulBroadcast = function (a, b) {
        util.assert(a.rank === 2, "Error in elementWiseMulBroadcast: first argument must be " +
            ("rank 2, but got rank " + a.rank + "."));
        util.assert(b.rank === 2, "Error in elementWiseMulBroadcast: second argument must be " +
            ("rank 2, but got rank " + b.rank + "."));
        return this.track(this.elementWiseMulBroadcastInternal(a, b));
    };
    NDArrayMath.prototype.conv2d = function (x, weights, biases, stride, zeroPad) {
        util.assert(x.rank === 3, "Error in conv2d: x must be rank 3, but got rank " + x.rank + ".");
        util.assert(weights.rank === 4, "Error in conv2d: weights must be rank 4, but got rank " +
            (weights.rank + "."));
        if (biases != null) {
            util.assert(biases.rank === 1, "Error in conv2d: biases must be rank 1, but got rank " +
                (biases.rank + "."));
        }
        util.assert(x.shape[2] === weights.shape[2], "Error in conv2d: depth of input (" + x.shape[2] + ") must match  " +
            ("input depth for weights " + weights.shape[2] + "."));
        return this.track(this.conv2dInternal(x, weights, biases, stride, zeroPad));
    };
    NDArrayMath.prototype.conv2dBackProp = function (x, dy, weights, stride, pad) {
        util.assert(x.rank === 3, "Error in conv2dBackProp: x must be rank 3, but got shape " +
            (x.shape + "."));
        util.assert(dy.rank === 3, "Error in conv2dBackProp: dy must be rank 3, but got shape " +
            (dy.shape + "."));
        util.assert(weights.rank === 4, "Error in conv2dBackProp: weights must be rank 4, but got shape " +
            (weights.shape + "."));
        util.assert(x.shape[2] === weights.shape[2], "Error in conv2dBackProp: depth of x " + x.shape[2] + ") must " +
            ("match input depth for weights (" + weights.shape[2] + "."));
        util.assert(dy.shape[2] === weights.shape[3], "Error in conv2dBackProp: depth of dy (" + dy.shape[2] + ") must " +
            ("match output depth for weights (" + weights.shape[3] + ")."));
        var backpropResult = this.conv2dBackPropInternal(x, dy, weights, stride, pad);
        this.track(backpropResult.db);
        this.track(backpropResult.dw);
        this.track(backpropResult.dx);
        return backpropResult;
    };
    NDArrayMath.prototype.conv2dTranspose = function (x, weights, biases, stride, pad) {
        util.assert(x.rank === 3, "Error in conv2dTranspose: x must be rank 3, but got rank " +
            (x.rank + "."));
        util.assert(weights.rank === 4, "Error in conv2dTranspose: weights must be rank 4, but got " +
            ("rank " + weights.rank));
        if (biases != null) {
            util.assert(biases.rank === 1, "Error in conv2dTranspose: biases must be rank 1, but got ' +\n              'rank " + biases.rank + ".");
        }
        util.assert(x.shape[2] === weights.shape[3], "Error in conv2dTranspose: depth of input (" + x.shape[2] + ") must " +
            ("match input depth for weights " + weights.shape[3] + "."));
        return this.track(this.conv2dTransposeInternal(x, weights, biases, stride, pad));
    };
    NDArrayMath.prototype.maxPool = function (x, fSize, stride, pad) {
        util.assert(x.rank === 3, 'Error in maxPool: x must be rank 3 but got rank ' + x.rank + '.');
        return this.track(this.maxPoolInternal(x, fSize, stride, pad));
    };
    NDArrayMath.prototype.maxPoolBackprop = function (dy, x, fSize, stride, pad) {
        util.assert(dy.rank === 3, "Error in maxPoolBackprop: dy must be rank 3 but got rank " +
            (dy.rank + "."));
        util.assert(x.rank === 3, "Error in maxPoolBackprop: x must be rank 3 but got rank " +
            (x.rank + "."));
        return this.track(this.maxPoolBackpropInternal(dy, x, fSize, stride, pad));
    };
    NDArrayMath.prototype.minPool = function (x, fSize, stride, pad) {
        util.assert(x.rank === 3, "Error in minPool: x must be rank 3 but got rank " + x.rank + ".");
        return this.track(this.minPoolInternal(x, fSize, stride, pad));
    };
    NDArrayMath.prototype.avgPool = function (x, fSize, stride, pad) {
        util.assert(x.rank === 3, "Error in avgPool: x must be rank 3 but got rank " + x.rank + ".");
        return this.track(this.avgPoolInternal(x, fSize, stride, pad));
    };
    NDArrayMath.prototype.resizeBilinear3D = function (x, newShape2D, alignCorners) {
        if (alignCorners === void 0) { alignCorners = false; }
        util.assert(x.rank === 3, "Error in resizeBilinear3D: x must be rank 3 but got rank " + x.rank + ".");
        util.assert(newShape2D.length === 2, "Error in resizeBilinear3D: new shape must 2D, but got shape " +
            (newShape2D + "."));
        return this.track(this.resizeBilinear3DInternal(x, newShape2D, alignCorners));
    };
    NDArrayMath.prototype.batchNormalization3D = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        util.assert(x.rank === 3, "Error in batchNormalization3D: x must be rank 3 but got rank " +
            (x.rank + "."));
        util.assert(mean.rank === 3 || mean.rank === 1, "Error in batchNormalization3D: mean must be rank 3 or rank 1 but " +
            ("got rank " + mean.rank + "."));
        util.assert(variance.rank === 3 || variance.rank === 1, "Error in batchNormalization3D: variance must be rank 3 or rank 1 " +
            ("but got rank " + variance.rank + "."));
        if (scale != null) {
            util.assert(scale.rank === 3 || scale.rank === 1, "Error in batchNormalization3D: scale must be rank 3 or rank 1 " +
                ("but got rank " + scale.rank + "."));
        }
        if (offset != null) {
            util.assert(offset.rank === 3 || offset.rank === 1, "Error in batchNormalization3D: offset must be rank 3 or rank 1 " +
                ("but got rank " + offset.rank + "."));
        }
        return this.track(this.batchNormalization3DInternal(x, mean, variance, varianceEpsilon, scale, offset));
    };
    return NDArrayMath;
}());
exports.NDArrayMath = NDArrayMath;
var MatrixOrientation;
(function (MatrixOrientation) {
    MatrixOrientation[MatrixOrientation["REGULAR"] = 0] = "REGULAR";
    MatrixOrientation[MatrixOrientation["TRANSPOSED"] = 1] = "TRANSPOSED";
})(MatrixOrientation = exports.MatrixOrientation || (exports.MatrixOrientation = {}));

},{"../util":33,"./concat3d_util":14,"./copy2d_util":16,"./ndarray":19}],18:[function(require,module,exports){
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
var conv_util = require("../math/conv_util");
var util = require("../util");
var concat3d_util = require("./concat3d_util");
var copy2D_util = require("./copy2d_util");
var math_1 = require("./math");
var ndarray_1 = require("./ndarray");
var NDArrayMathCPU = (function (_super) {
    __extends(NDArrayMathCPU, _super);
    function NDArrayMathCPU(safeMode) {
        if (safeMode === void 0) { safeMode = false; }
        return _super.call(this, safeMode) || this;
    }
    NDArrayMathCPU.prototype.cloneInternal = function (ndarray) {
        return ndarray_1.NDArray.make(ndarray.shape, { values: new Float32Array(ndarray.getValues()) });
    };
    NDArrayMathCPU.prototype.reshapeInternal = function (ndarray, newShape) {
        return this.cloneInternal(ndarray).reshape(newShape);
    };
    NDArrayMathCPU.prototype.slice2DInternal = function (input, beginRowCol, sizeRowCol) {
        var result = ndarray_1.Array2D.zeros(sizeRowCol);
        this.copy2DInternal(input, beginRowCol, sizeRowCol, result, [0, 0], sizeRowCol);
        return result;
    };
    NDArrayMathCPU.prototype.copy2DInternal = function (source, sourceBeginRowCol, sourceSizeRowCol, dest, destBeginRowCol, destSizeRowCol) {
        copy2D_util.validateShapes(sourceSizeRowCol, destSizeRowCol);
        var srcValues = source.getValues();
        var dstValues = dest.getValues();
        var n = sourceSizeRowCol[0] * sourceSizeRowCol[1];
        for (var i = 0; i < n; ++i) {
            var srcRow = sourceBeginRowCol[0] + Math.floor(i / sourceSizeRowCol[1]);
            var srcCol = sourceBeginRowCol[1] + (i % sourceSizeRowCol[1]);
            var srcOff = srcRow * source.shape[1] + srcCol;
            var dstRow = destBeginRowCol[0] + Math.floor(i / destSizeRowCol[1]);
            var dstCol = destBeginRowCol[1] + (i % destSizeRowCol[1]);
            var dstOff = dstRow * dest.shape[1] + dstCol;
            dstValues[dstOff] = srcValues[srcOff];
        }
    };
    NDArrayMathCPU.prototype.concat3DInternal = function (x1, x2, axis) {
        var outputShape = concat3d_util.computeConcat3DOutputShape(x1.shape, x2.shape, axis);
        var values = ndarray_1.NDArray.zeros(outputShape);
        for (var i = 0; i < outputShape[0]; i++) {
            for (var j = 0; j < outputShape[1]; j++) {
                for (var k = 0; k < outputShape[2]; k++) {
                    var index = [i, j, k];
                    var value = void 0;
                    if (index[axis] < x1.shape[axis]) {
                        value = x1.get(i, j, k);
                    }
                    else {
                        index[axis] -= x1.shape[axis];
                        var i2 = index[0], j2 = index[1], k2 = index[2];
                        value = x2.get(i2, j2, k2);
                    }
                    values.set(value, i, j, k);
                }
            }
        }
        return values;
    };
    NDArrayMathCPU.prototype.scalarPlusArrayInternal = function (c, a) {
        var resultValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var cVal = c.get();
        for (var i = 0; i < resultValues.length; ++i) {
            resultValues[i] = cVal + aValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.scaledArrayAddInternal = function (c1, a, c2, b) {
        var cValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var bValues = b.getValues();
        var c1Val = c1.get();
        var c2Val = c2.get();
        for (var i = 0; i < cValues.length; ++i) {
            cValues[i] = c1Val * aValues[i] + c2Val * bValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: cValues });
    };
    NDArrayMathCPU.prototype.scalarTimesArrayInternal = function (c, a) {
        var newValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var cVal = c.get();
        for (var i = 0; i < aValues.length; ++i) {
            newValues[i] = cVal * aValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.scalarMinusArrayInternal = function (c, a) {
        var negA = this.negInternal(a);
        var result = this.scalarPlusArrayInternal(c, negA);
        negA.dispose();
        return result;
    };
    NDArrayMathCPU.prototype.arrayMinusScalarInternal = function (a, c) {
        var negC = this.negInternal(c);
        var result = this.scalarPlusArrayInternal(negC, a);
        negC.dispose();
        return result;
    };
    NDArrayMathCPU.prototype.negInternal = function (a) {
        return this.scalarTimesArrayInternal(ndarray_1.Scalar.NEG_ONE, a);
    };
    NDArrayMathCPU.prototype.addInternal = function (a, b) {
        return this.scaledArrayAddInternal(ndarray_1.Scalar.ONE, a, ndarray_1.Scalar.ONE, b);
    };
    NDArrayMathCPU.prototype.subInternal = function (a, b) {
        return this.scaledArrayAddInternal(ndarray_1.Scalar.ONE, a, ndarray_1.Scalar.NEG_ONE, b);
    };
    NDArrayMathCPU.prototype.matMulInternal = function (a, b, aOrientation, bOrientation) {
        if (aOrientation === void 0) { aOrientation = math_1.MatrixOrientation.REGULAR; }
        if (bOrientation === void 0) { bOrientation = math_1.MatrixOrientation.REGULAR; }
        var sharedDim = (aOrientation === math_1.MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
        var leftDim = (aOrientation === math_1.MatrixOrientation.REGULAR) ? a.shape[0] : a.shape[1];
        var rightDim = (bOrientation === math_1.MatrixOrientation.REGULAR) ? b.shape[1] : b.shape[0];
        var normalGetter = function (matrix, i, j) {
            return matrix.get(i, j);
        };
        var transposedGetter = function (matrix, i, j) {
            return matrix.get(j, i);
        };
        var aGetter = (aOrientation === math_1.MatrixOrientation.REGULAR) ?
            normalGetter :
            transposedGetter;
        var bGetter = (bOrientation === math_1.MatrixOrientation.REGULAR) ?
            normalGetter :
            transposedGetter;
        var values = new Float32Array(leftDim * rightDim);
        var index = 0;
        for (var i = 0; i < leftDim; ++i) {
            for (var j = 0; j < rightDim; ++j) {
                var sum = 0;
                for (var k = 0; k < sharedDim; ++k) {
                    sum += aGetter(a, i, k) * bGetter(b, k, j);
                }
                values[index++] = sum;
            }
        }
        return ndarray_1.Array2D.new([leftDim, rightDim], values);
    };
    NDArrayMathCPU.prototype.elementWiseMulInternal = function (a, b) {
        var newValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var bValues = b.getValues();
        for (var i = 0; i < aValues.length; ++i) {
            newValues[i] = aValues[i] * bValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.elementWiseMulBroadcastInternal = function (a, b) {
        var maxRow = Math.max(a.shape[0], b.shape[0]);
        var maxCol = Math.max(a.shape[1], b.shape[1]);
        var values = new Float32Array(maxRow * maxCol);
        var index = 0;
        for (var row = 0; row < maxRow; row++) {
            for (var col = 0; col < maxCol; col++) {
                values[index++] = a.get(row % a.shape[0], col % a.shape[1]) *
                    b.get(row % b.shape[0], col % b.shape[1]);
            }
        }
        return ndarray_1.Array2D.new([maxRow, maxCol], values);
    };
    NDArrayMathCPU.prototype.divideInternal = function (a, b) {
        var newValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var bValues = b.getValues();
        for (var i = 0; i < aValues.length; ++i) {
            newValues[i] = aValues[i] / bValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.scalarDividedByArrayInternal = function (c, a) {
        var newValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var cValue = c.get();
        for (var i = 0; i < aValues.length; ++i) {
            newValues[i] = cValue / aValues[i];
        }
        return ndarray_1.NDArray.make(a.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.arrayDividedByScalarInternal = function (a, c) {
        var newValues = new Float32Array(a.size);
        var aValues = a.getValues();
        var cValue = c.get();
        for (var i = 0; i < aValues.length; ++i) {
            newValues[i] = aValues[i] / cValue;
        }
        return ndarray_1.NDArray.make(a.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.sumInternal = function (ndarray) {
        var sum = 0;
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            sum += values[i];
        }
        return ndarray_1.Scalar.new(sum);
    };
    NDArrayMathCPU.prototype.argMinInternal = function (ndarray) {
        var min = Number.MAX_VALUE;
        var minIndex = -1;
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value < min) {
                min = value;
                minIndex = i;
            }
        }
        return ndarray_1.Scalar.new(minIndex);
    };
    NDArrayMathCPU.prototype.argMaxInternal = function (ndarray) {
        var max = Number.NEGATIVE_INFINITY;
        var maxIndex = -1;
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value > max) {
                max = value;
                maxIndex = i;
            }
        }
        return ndarray_1.Scalar.new(maxIndex);
    };
    NDArrayMathCPU.prototype.argMaxEqualsInternal = function (x1, x2) {
        var argMax1 = this.argMaxInternal(x1).get();
        var argMax2 = this.argMaxInternal(x2).get();
        if (isNaN(argMax1) || isNaN(argMax2)) {
            return ndarray_1.Scalar.new(NaN);
        }
        return ndarray_1.Scalar.new(+(argMax1 === argMax2));
    };
    NDArrayMathCPU.prototype.topKInternal = function (ndarray, k) {
        var values = ndarray.getValues();
        var valuesAndIndices = [];
        for (var i = 0; i < values.length; i++) {
            valuesAndIndices.push({ value: values[i], index: i });
        }
        valuesAndIndices.sort(function (a, b) {
            return b.value - a.value;
        });
        var topkValues = new Float32Array(k);
        var topkIndices = new Float32Array(k);
        for (var i = 0; i < k; i++) {
            topkValues[i] = valuesAndIndices[i].value;
            topkIndices[i] = valuesAndIndices[i].index;
        }
        return { values: ndarray_1.Array1D.new(topkValues), indices: ndarray_1.Array1D.new(topkIndices) };
    };
    NDArrayMathCPU.prototype.minInternal = function (ndarray) {
        var values = ndarray.getValues();
        var min = values[0];
        for (var i = 1; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value < min) {
                min = value;
            }
        }
        return ndarray_1.Scalar.new(min);
    };
    NDArrayMathCPU.prototype.maxInternal = function (ndarray) {
        var values = ndarray.getValues();
        var max = values[0];
        for (var i = 1; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value > max) {
                max = value;
            }
        }
        return ndarray_1.Scalar.new(max);
    };
    NDArrayMathCPU.prototype.expInternal = function (ndarray) {
        var values = ndarray.getValues();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            newValues[i] = Math.exp(values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.logInternal = function (ndarray) {
        var values = ndarray.getValues();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            newValues[i] = Math.log(value);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.logSumExpInternal = function (ndarray) {
        var xMax = this.max(ndarray);
        var a = this.arrayMinusScalar(ndarray, xMax);
        var b = this.exp(a);
        var c = this.sum(b);
        var d = this.log(c);
        var result = this.add(xMax, d);
        xMax.dispose();
        a.dispose();
        b.dispose();
        c.dispose();
        d.dispose();
        return result;
    };
    NDArrayMathCPU.prototype.reluInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.max(0, values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.sigmoidInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = 1 / (1 + Math.exp(-values[i]));
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.tanhInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = util.tanh(values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.sinInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.sin(values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.stepInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            resultValues[i] = value > 0 ? 1 : (value < 0 ? 0 : value);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.conv2dInternal = function (x, weights, biases, stride, pad) {
        var _a = x.shape, xRows = _a[0], xCols = _a[1], inputDepth = _a[2];
        var fieldSize = weights.shape[0];
        var outputDepth = weights.shape[3];
        var outputShape = conv_util.computeOutputShape3D([xRows, xCols, inputDepth], fieldSize, outputDepth, stride, pad);
        var y = ndarray_1.Array3D.zeros(outputShape);
        for (var d2 = 0; d2 < outputDepth; ++d2) {
            for (var yR = 0; yR < y.shape[0]; ++yR) {
                var xRCorner = yR * stride - pad;
                var xRMin = Math.max(0, xRCorner);
                var xRMax = Math.min(xRows, fieldSize + xRCorner);
                for (var yC = 0; yC < y.shape[1]; ++yC) {
                    var xCCorner = yC * stride - pad;
                    var xCMin = Math.max(0, xCCorner);
                    var xCMax = Math.min(xCols, fieldSize + xCCorner);
                    var dotProd = 0;
                    for (var xR = xRMin; xR < xRMax; ++xR) {
                        var wR = xR - xRCorner;
                        for (var xC = xCMin; xC < xCMax; ++xC) {
                            var wC = xC - xCCorner;
                            for (var d1 = 0; d1 < inputDepth; ++d1) {
                                var pixel = x.get(xR, xC, d1);
                                var weight = weights.get(wR, wC, d1, d2);
                                dotProd += pixel * weight;
                            }
                        }
                    }
                    var bias = (biases != null) ? biases.get(d2) : 0;
                    y.set(dotProd + bias, yR, yC, d2);
                }
            }
        }
        return y;
    };
    NDArrayMathCPU.prototype.conv2dBackPropInternal = function (x, dy, weights, stride, pad) {
        var fSize = weights.shape[0];
        var dw = this.conv2dDerWeights(x, dy, fSize, stride, pad);
        var db = this.conv2dDerBias(dy);
        var dx = this.conv2dTransposeInternal(dy, weights, null, stride, pad);
        return { dx: dx, db: db, dw: dw };
    };
    NDArrayMathCPU.prototype.conv2dTransposeInternal = function (x, weights, biases, origStride, origPad) {
        var fSize = weights.shape[0];
        var pad = fSize - 1 - origPad;
        var origInputDepth = weights.shape[2];
        var origOutputDepth = weights.shape[3];
        var _a = x.shape, xRows = _a[0], xCols = _a[1], xDepth = _a[2];
        var xRowsDilated = (xRows - 1) * origStride + 1;
        var xColsDilated = (xCols - 1) * origStride + 1;
        var outputShape = conv_util.computeOutputShape3D([xRowsDilated, xColsDilated, origOutputDepth], fSize, origInputDepth, 1, pad);
        var y = ndarray_1.Array3D.zeros(outputShape);
        for (var d2 = 0; d2 < origInputDepth; ++d2) {
            for (var yR = 0; yR < y.shape[0]; ++yR) {
                var xRCorner = yR - pad;
                var xRMin = Math.max(0, Math.ceil(xRCorner / origStride));
                var xRMax = Math.min(xRows, (fSize + xRCorner) / origStride);
                for (var yC = 0; yC < y.shape[1]; ++yC) {
                    var xCCorner = yC - pad;
                    var xCMin = Math.max(0, Math.ceil(xCCorner / origStride));
                    var xCMax = Math.min(xCols, (fSize + xCCorner) / origStride);
                    var dotProd = 0;
                    for (var xR = xRMin; xR < xRMax; ++xR) {
                        var wR = xR * origStride - xRCorner;
                        for (var xC = xCMin; xC < xCMax; ++xC) {
                            var wC = xC * origStride - xCCorner;
                            for (var d1 = 0; d1 < origOutputDepth; ++d1) {
                                var pixel = x.get(xR, xC, d1);
                                var weight = weights.get(fSize - 1 - wR, fSize - 1 - wC, d2, d1);
                                dotProd += pixel * weight;
                            }
                        }
                    }
                    var bias = biases != null ? biases.get(d2) : 0;
                    y.set(dotProd + bias, yR, yC, d2);
                }
            }
        }
        return y;
    };
    NDArrayMathCPU.prototype.conv2dTransposeShaderLike = function (x, origWeights, origStride, origPad) {
        var fSize = origWeights.shape[0];
        var pad = fSize - 1 - origPad;
        var origInputDepth = origWeights.shape[2];
        var origOutputDepth = origWeights.shape[3];
        var _a = x.shape, xRows = _a[0], xCols = _a[1], xDepth = _a[2];
        var xRowsDilated = (xRows - 1) * origStride + 1;
        var xColsDilated = (xCols - 1) * origStride + 1;
        var outputShape = conv_util.computeOutputShape3D([xRowsDilated, xColsDilated, origOutputDepth], fSize, origInputDepth, 1, pad);
        var y = ndarray_1.Array3D.zeros(outputShape);
        for (var d2 = 0; d2 < origInputDepth; ++d2) {
            for (var yR = 0; yR < y.shape[0]; ++yR) {
                for (var yC = 0; yC < y.shape[1]; ++yC) {
                    var xRCorner = yR - pad;
                    var xCCorner = yC - pad;
                    var dotProd = 0;
                    for (var wR = 0; wR < fSize; ++wR) {
                        var xR = (xRCorner + wR) / origStride;
                        if (xR < 0 || xR >= xRows || Math.floor(xR) !== xR) {
                            continue;
                        }
                        for (var wC = 0; wC < fSize; ++wC) {
                            var xC = (xCCorner + wC) / origStride;
                            if (xC < 0 || xC >= xCols || Math.floor(xC) !== xC) {
                                continue;
                            }
                            for (var d1 = 0; d1 < origOutputDepth; ++d1) {
                                var pixel = x.get(xR, xC, d1);
                                var weight = origWeights.get(fSize - 1 - wR, fSize - 1 - wC, d2, d1);
                                dotProd += pixel * weight;
                            }
                        }
                    }
                    y.set(dotProd, yR, yC, d2);
                }
            }
        }
        return y;
    };
    NDArrayMathCPU.prototype.conv2dDerWeights = function (x, dY, fSize, stride, zeroPad) {
        var inputDepth = x.shape[2];
        var outputDepth = dY.shape[2];
        var weightsShape = conv_util.computeWeightsShape4D(inputDepth, outputDepth, fSize);
        var dW = ndarray_1.Array4D.zeros(weightsShape);
        var yNumRows = dY.shape[0];
        var yNumCols = dY.shape[1];
        var xNumRows = x.shape[0];
        var xNumCols = x.shape[1];
        for (var wR = 0; wR < fSize; ++wR) {
            var yRMin = Math.max(0, Math.ceil((zeroPad - wR) / stride));
            var yRMax = Math.min(yNumRows, (xNumRows + zeroPad - wR) / stride);
            for (var wC = 0; wC < fSize; ++wC) {
                var yCMin = Math.max(0, Math.ceil((zeroPad - wC) / stride));
                var yCMax = Math.min(yNumCols, (xNumCols + zeroPad - wC) / stride);
                for (var d1 = 0; d1 < inputDepth; ++d1) {
                    for (var d2 = 0; d2 < outputDepth; ++d2) {
                        var dotProd = 0;
                        for (var yR = yRMin; yR < yRMax; ++yR) {
                            var xR = wR + yR * stride - zeroPad;
                            for (var yC = yCMin; yC < yCMax; ++yC) {
                                var xC = wC + yC * stride - zeroPad;
                                dotProd += x.get(xR, xC, d1) * dY.get(yR, yC, d2);
                            }
                        }
                        dW.set(dotProd, wR, wC, d1, d2);
                    }
                }
            }
        }
        return dW;
    };
    NDArrayMathCPU.prototype.conv2dDerBias = function (dY) {
        var outputDepth = dY.shape[2];
        var numRows = dY.shape[0];
        var numCols = dY.shape[1];
        var values = new Float32Array(outputDepth);
        for (var d2 = 0; d2 < outputDepth; ++d2) {
            var sum = 0;
            for (var r = 0; r < numRows; ++r) {
                for (var c = 0; c < numCols; ++c) {
                    sum += dY.get(r, c, d2);
                }
            }
            values[d2] = sum;
        }
        return ndarray_1.Array1D.new(values);
    };
    NDArrayMathCPU.prototype.switchDimInternal = function (t, newDim) {
        var newShape = new Array(t.rank);
        for (var i = 0; i < newShape.length; i++) {
            newShape[i] = t.shape[newDim[i]];
        }
        var resultValues = new Float32Array(t.size);
        var values = t.getValues();
        var result = ndarray_1.NDArray.make(newShape, { values: resultValues });
        for (var i = 0; i < t.size; ++i) {
            var loc = t.indexToLoc(i);
            var newLoc = new Array(loc.length);
            for (var i_1 = 0; i_1 < newLoc.length; i_1++) {
                newLoc[i_1] = loc[newDim[i_1]];
            }
            var newIndex = result.locToIndex(newLoc);
            resultValues[newIndex] = values[i];
        }
        return result;
    };
    NDArrayMathCPU.prototype.pool = function (x, fSize, stride, pad, poolType) {
        var _a = x.shape, xRows = _a[0], xCols = _a[1], depth = _a[2];
        var outputShape = conv_util.computeOutputShape3D([xRows, xCols, depth], fSize, depth, stride, pad);
        var y = ndarray_1.Array3D.zeros(outputShape);
        for (var d = 0; d < depth; ++d) {
            for (var yR = 0; yR < y.shape[0]; ++yR) {
                var xRCorner = yR * stride - pad;
                var xRMin = Math.max(0, xRCorner);
                var xRMax = Math.min(xRows, fSize + xRCorner);
                for (var yC = 0; yC < y.shape[1]; ++yC) {
                    var xCCorner = yC * stride - pad;
                    var xCMin = Math.max(0, xCCorner);
                    var xCMax = Math.min(xCols, fSize + xCCorner);
                    var minMaxValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
                        Number.POSITIVE_INFINITY);
                    var avgValue = 0;
                    for (var xR = xRMin; xR < xRMax; ++xR) {
                        var wR = xR - xRCorner;
                        for (var xC = xCMin; xC < xCMax; ++xC) {
                            var wC = xC - xCCorner;
                            var pixel = x.get(xR, xC, d);
                            if (isNaN(pixel)) {
                                minMaxValue = NaN;
                                avgValue = NaN;
                                break;
                            }
                            if ((poolType === 'max' && pixel > minMaxValue) ||
                                (poolType === 'min' && pixel < minMaxValue)) {
                                minMaxValue = pixel;
                            }
                            else if (poolType === 'avg') {
                                avgValue += pixel / (fSize * fSize);
                            }
                        }
                        if (isNaN(minMaxValue)) {
                            break;
                        }
                    }
                    y.set(poolType === 'avg' ? avgValue : minMaxValue, yR, yC, d);
                }
            }
        }
        return y;
    };
    NDArrayMathCPU.prototype.maxPoolInternal = function (x, fSize, stride, pad) {
        return this.pool(x, fSize, stride, pad, 'max');
    };
    NDArrayMathCPU.prototype.maxPoolPositions = function (x, fSize, stride, pad) {
        var _a = x.shape, xRows = _a[0], xCols = _a[1], depth = _a[2];
        var outputShape = conv_util.computeOutputShape3D(x.shape, fSize, depth, stride, pad);
        var maxPositions = ndarray_1.Array3D.zeros(outputShape);
        for (var d = 0; d < depth; ++d) {
            for (var yR = 0; yR < outputShape[0]; ++yR) {
                var xRCorner = yR * stride - pad;
                var xRMin = Math.max(0, xRCorner);
                var xRMax = Math.min(xRows, fSize + xRCorner);
                for (var yC = 0; yC < outputShape[1]; ++yC) {
                    var xCCorner = yC * stride - pad;
                    var xCMin = Math.max(0, xCCorner);
                    var xCMax = Math.min(xCols, fSize + xCCorner);
                    var maxValue = Number.NEGATIVE_INFINITY;
                    var maxPosition = -1;
                    for (var xR = xRMin; xR < xRMax; ++xR) {
                        var wR = xR - xRCorner;
                        for (var xC = xCMin; xC < xCMax; ++xC) {
                            var wC = xC - xCCorner;
                            var pixel = x.get(xR, xC, d);
                            if (pixel > maxValue) {
                                maxValue = pixel;
                                maxPosition = wR * fSize + wC;
                            }
                        }
                    }
                    maxPositions.set(maxPosition, yR, yC, d);
                }
            }
        }
        return maxPositions;
    };
    NDArrayMathCPU.prototype.maxPoolBackpropInternal = function (dy, x, fSize, origStride, origPad) {
        var maxPositions = this.maxPoolPositions(x, fSize, origStride, origPad);
        var pad = fSize - 1 - origPad;
        var _a = dy.shape, dyRows = _a[0], dyCols = _a[1], depth = _a[2];
        var dyRowsDilated = (dyRows - 1) * origStride + 1;
        var dxColsDilated = (dyCols - 1) * origStride + 1;
        var outputShape = conv_util.computeOutputShape3D([dyRowsDilated, dxColsDilated, depth], fSize, depth, 1, pad);
        var dx = ndarray_1.Array3D.zeros(outputShape);
        for (var d = 0; d < depth; ++d) {
            for (var dxR = 0; dxR < dx.shape[0]; ++dxR) {
                for (var dxC = 0; dxC < dx.shape[1]; ++dxC) {
                    var dyRCorner = dxR - pad;
                    var dyCCorner = dxC - pad;
                    var dotProd = 0;
                    for (var wR = 0; wR < fSize; ++wR) {
                        var dyR = (dyRCorner + wR) / origStride;
                        if (dyR < 0 || dyR >= dyRows || Math.floor(dyR) !== dyR) {
                            continue;
                        }
                        for (var wC = 0; wC < fSize; ++wC) {
                            var dyC = (dyCCorner + wC) / origStride;
                            if (dyC < 0 || dyC >= dyCols || Math.floor(dyC) !== dyC) {
                                continue;
                            }
                            var maxPos = fSize * fSize - 1 - maxPositions.get(dyR, dyC, d);
                            var curPos = wR * fSize + wC;
                            var mask = maxPos === curPos ? 1 : 0;
                            if (mask === 0) {
                                continue;
                            }
                            var pixel = dy.get(dyR, dyC, d);
                            dotProd += pixel * mask;
                        }
                    }
                    dx.set(dotProd, dxR, dxC, d);
                }
            }
        }
        return dx;
    };
    NDArrayMathCPU.prototype.minPoolInternal = function (x, fSize, stride, pad) {
        return this.pool(x, fSize, stride, pad, 'min');
    };
    NDArrayMathCPU.prototype.avgPoolInternal = function (x, fSize, stride, pad) {
        return this.pool(x, fSize, stride, pad, 'avg');
    };
    NDArrayMathCPU.prototype.resizeBilinear3DInternal = function (x, newShape2D, alignCorners) {
        var output = ndarray_1.Array3D.zeros([newShape2D[0], newShape2D[1], x.shape[2]]);
        var effectiveInputSize = alignCorners ? [x.shape[0] - 1, x.shape[1] - 1, x.shape[2]] : x.shape;
        var effectiveOutputSize = alignCorners ?
            [output.shape[0] - 1, output.shape[1] - 1, output.shape[2]] :
            output.shape;
        for (var r = 0; r < output.shape[0]; r++) {
            for (var c = 0; c < output.shape[1]; c++) {
                for (var d = 0; d < output.shape[2]; d++) {
                    var sourceFracRow = (effectiveInputSize[0]) * r / (effectiveOutputSize[0]);
                    var sourceFracCol = (effectiveInputSize[1]) * c / (effectiveOutputSize[1]);
                    var sourceRowFloor = Math.floor(sourceFracRow);
                    var sourceRowCeil = Math.min(x.shape[0] - 1, Math.ceil(sourceFracRow));
                    var sourceColFloor = Math.floor(sourceFracCol);
                    var sourceColCeil = Math.min(x.shape[1] - 1, Math.ceil(sourceFracCol));
                    var topLeft = x.get(sourceRowFloor, sourceColFloor, d);
                    var bottomLeft = x.get(sourceRowCeil, sourceColFloor, d);
                    var topRight = x.get(sourceRowFloor, sourceColCeil, d);
                    var bottomRight = x.get(sourceRowCeil, sourceColCeil, d);
                    var rowFrac = sourceFracRow - sourceRowFloor;
                    var colFrac = sourceFracCol - sourceColFloor;
                    var top_1 = topLeft + (topRight - topLeft) * colFrac;
                    var bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
                    var newValue = top_1 + (bottom - top_1) * rowFrac;
                    output.set(newValue, r, c, d);
                }
            }
        }
        return output;
    };
    NDArrayMathCPU.prototype.batchNormalization3DInternal = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        var xValues = x.getValues();
        var meanValues = mean.getValues();
        var varianceValues = variance.getValues();
        var scaleValues = scale ? scale.getValues() : new Float32Array([1]);
        var offsetValues = offset ? offset.getValues() : new Float32Array([0]);
        var outValues = new Float32Array(xValues.length);
        for (var i = 0; i < xValues.length; i++) {
            outValues[i] = offsetValues[i % offsetValues.length] +
                (xValues[i] - meanValues[i % meanValues.length]) *
                    scaleValues[i % scaleValues.length] /
                    Math.sqrt(varianceValues[i % varianceValues.length] + varianceEpsilon);
        }
        return ndarray_1.NDArray.make(x.shape, { values: outValues });
    };
    return NDArrayMathCPU;
}(math_1.NDArrayMath));
exports.NDArrayMathCPU = NDArrayMathCPU;

},{"../math/conv_util":15,"../util":33,"./concat3d_util":14,"./copy2d_util":16,"./math":17,"./ndarray":19}],19:[function(require,module,exports){
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
var util = require("../util");
var webgl_util = require("./webgl/webgl_util");
exports.GPGPU = null;
exports.TEXTURE_MANAGER = null;
function initializeGPU(gpgpu, textureManager) {
    exports.GPGPU = gpgpu;
    exports.TEXTURE_MANAGER = textureManager;
}
exports.initializeGPU = initializeGPU;
function throwIfGPUNotInitialized() {
    if (exports.GPGPU == null || exports.TEXTURE_MANAGER == null) {
        throw new Error('GPU not intialized.');
    }
}
var NDArray = (function () {
    function NDArray(shape, data) {
        util.assert(data.values != null || data.texture != null, 'Either `values` or `texture` must be defined');
        util.assert(data.texture == null || (data.textureShapeRC != null), '`textureShape` must be defined when `texture` is defined');
        this.size = util.sizeFromShape(shape);
        if (data.values != null) {
            util.assert(this.size === data.values.length, 'Constructing ndarray of shape (' + this.size + ') should match the' +
                ' length of values (' + data.values.length + ')');
        }
        this.shape = shape;
        this.data = data;
        var dim = this.shape.length;
        if (dim < 2) {
            this.strides = [];
        }
        else {
            this.strides = new Array(dim - 1);
            this.strides[dim - 2] = this.shape[dim - 1];
            for (var i = dim - 3; i >= 0; --i) {
                this.strides[i] = this.strides[i + 1] * this.shape[i + 1];
            }
        }
    }
    NDArray.zeros = function (shape) {
        var values = new Float32Array(util.sizeFromShape(shape));
        return NDArray.make(shape, { values: values });
    };
    NDArray.zerosLike = function (another) {
        return NDArray.zeros(another.shape);
    };
    NDArray.like = function (another) {
        var values = another.getValues();
        return NDArray.make(another.shape, { values: new Float32Array(values) });
    };
    NDArray.make = function (shape, data) {
        switch (shape.length) {
            case 0:
                return new Scalar(data);
            case 1:
                return new Array1D(data);
            case 2:
                return new Array2D(shape, data);
            case 3:
                return new Array3D(shape, data);
            case 4:
                return new Array4D(shape, data);
            default:
                return new NDArray(shape, data);
        }
    };
    NDArray.prototype.reshape = function (newShape) {
        if (util.arraysEqual(this.shape, newShape)) {
            return this;
        }
        util.assert(this.size === util.sizeFromShape(newShape), 'new shape and old shape must have the same number of elements.');
        return NDArray.make(newShape, this.data);
    };
    NDArray.prototype.asScalar = function () {
        util.assert(this.size === 1, 'The array must have only 1 element.');
        return this.reshape([]);
    };
    NDArray.prototype.as1D = function () {
        return this.reshape([this.size]);
    };
    NDArray.prototype.as2D = function (rows, columns) {
        return this.reshape([rows, columns]);
    };
    NDArray.prototype.as3D = function (rows, columns, depth) {
        return this.reshape([rows, columns, depth]);
    };
    NDArray.prototype.as4D = function (rows, columns, depth, depth2) {
        return this.reshape([rows, columns, depth, depth2]);
    };
    Object.defineProperty(NDArray.prototype, "rank", {
        get: function () {
            return this.shape.length;
        },
        enumerable: true,
        configurable: true
    });
    NDArray.prototype.get = function () {
        var locs = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            locs[_i] = arguments[_i];
        }
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return this.getValues()[index];
    };
    NDArray.prototype.add = function (value) {
        var locs = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            locs[_i - 1] = arguments[_i];
        }
        this.set.apply(this, [this.get.apply(this, locs) + value].concat(locs));
    };
    NDArray.prototype.set = function (value) {
        var locs = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            locs[_i - 1] = arguments[_i];
        }
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        this.getValues()[index] = value;
    };
    NDArray.prototype.locToIndex = function (locs) {
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return index;
    };
    NDArray.prototype.indexToLoc = function (index) {
        var locs = new Array(this.shape.length);
        for (var i = 0; i < locs.length - 1; ++i) {
            locs[i] = Math.floor(index / this.strides[i]);
            index -= locs[i] * this.strides[i];
        }
        locs[locs.length - 1] = index;
        return locs;
    };
    NDArray.prototype.fill = function (value) {
        this.getValues().fill(value);
    };
    NDArray.prototype.getData = function () {
        return this.data;
    };
    NDArray.prototype.getValues = function () {
        if (this.data.values == null) {
            throwIfGPUNotInitialized();
            this.data.values = exports.GPGPU.downloadMatrixFromTexture(this.data.texture, this.data.textureShapeRC[0], this.data.textureShapeRC[1]);
            this.disposeTexture();
        }
        return this.data.values;
    };
    NDArray.prototype.uploadToGPU = function (preferredTexShape) {
        throwIfGPUNotInitialized();
        this.data.textureShapeRC = webgl_util.getTextureShapeFromLogicalShape(exports.GPGPU.gl, this.shape, preferredTexShape);
        this.data.texture =
            exports.TEXTURE_MANAGER.acquireTexture(this.data.textureShapeRC);
        exports.GPGPU.uploadMatrixToTexture(this.data.texture, this.data.textureShapeRC[0], this.data.textureShapeRC[1], this.data.values);
        this.data.values = null;
    };
    NDArray.prototype.getTexture = function (preferredShapeRC) {
        if (this.data.texture == null) {
            this.uploadToGPU(preferredShapeRC);
        }
        return this.data.texture;
    };
    NDArray.prototype.getTextureShapeRC = function (preferredShapeRC) {
        if (this.data.textureShapeRC == null) {
            this.uploadToGPU(preferredShapeRC);
        }
        return this.data.textureShapeRC;
    };
    NDArray.prototype.dispose = function () {
        this.data.values = null;
        this.shape = null;
        if (this.data.texture != null) {
            this.disposeTexture();
        }
    };
    NDArray.prototype.disposeTexture = function () {
        throwIfGPUNotInitialized();
        exports.TEXTURE_MANAGER.releaseTexture(this.data.texture, this.data.textureShapeRC);
        this.data.texture = null;
        this.data.textureShapeRC = null;
    };
    NDArray.prototype.inGPU = function () {
        return this.data.texture != null;
    };
    NDArray.prototype.equals = function (t) {
        return util.arraysEqual(this.shape, t.shape) &&
            util.arraysEqual(this.getValues(), t.getValues());
    };
    NDArray.rand = function (shape, randFunction) {
        var size = util.sizeFromShape(shape);
        var values = new Float32Array(size);
        for (var i = 0; i < size; i++) {
            values[i] = randFunction();
        }
        return NDArray.make(shape, { values: values });
    };
    NDArray.randNormal = function (shape, mean, stdDev) {
        if (mean === void 0) { mean = 0; }
        if (stdDev === void 0) { stdDev = 1; }
        return NDArray.rand(shape, function () { return util.randGauss(mean, stdDev); });
    };
    NDArray.randTruncatedNormal = function (shape, mean, stdDev) {
        if (mean === void 0) { mean = 0; }
        if (stdDev === void 0) { stdDev = 1; }
        return NDArray.rand(shape, function () { return util.randGauss(mean, stdDev, true); });
    };
    NDArray.randUniform = function (shape, a, b) {
        return NDArray.rand(shape, function () { return util.randUniform(a, b); });
    };
    return NDArray;
}());
exports.NDArray = NDArray;
var Scalar = (function (_super) {
    __extends(Scalar, _super);
    function Scalar(data) {
        var _this = this;
        if (data.texture != null) {
            data.textureShapeRC = [1, 1];
        }
        _this = _super.call(this, [], data) || this;
        return _this;
    }
    Scalar.new = function (value) {
        return new Scalar({ values: new Float32Array([value]) });
    };
    Scalar.prototype.get = function () {
        return this.getValues()[0];
    };
    Scalar.prototype.set = function (value) {
        this.getValues()[0] = value;
    };
    Scalar.prototype.add = function (value) {
        this.getValues()[0] += value;
    };
    return Scalar;
}(NDArray));
Scalar.ZERO = Scalar.new(0);
Scalar.ONE = Scalar.new(1);
Scalar.TWO = Scalar.new(2);
Scalar.NEG_ONE = Scalar.new(-1);
exports.Scalar = Scalar;
var Array1D = (function (_super) {
    __extends(Array1D, _super);
    function Array1D(data) {
        var _this = this;
        var shape = (data.values != null) ?
            [data.values.length] :
            [util.sizeFromShape(data.textureShapeRC)];
        _this = _super.call(this, shape, data) || this;
        return _this;
    }
    Array1D.new = function (values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            util.assert(inferredShape.length === 1, "Error constructing Array1D. Shape of values " + inferredShape + " is " +
                "not 1 dimensional.");
        }
        return new Array1D({ values: toTypedArray(values) });
    };
    Array1D.prototype.get = function (i) {
        return this.getValues()[i];
    };
    Array1D.prototype.set = function (value, i) {
        this.getValues()[i] = value;
    };
    Array1D.prototype.add = function (value, i) {
        this.getValues()[i] += value;
    };
    Array1D.prototype.locToIndex = function (loc) {
        return loc[0];
    };
    Array1D.prototype.indexToLoc = function (index) {
        return [index];
    };
    Array1D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array1D;
}(NDArray));
exports.Array1D = Array1D;
var Array2D = (function (_super) {
    __extends(Array2D, _super);
    function Array2D(shape, data) {
        var _this = this;
        util.assert(shape.length === 2, 'Shape should be of length 2');
        _this = _super.call(this, shape, data) || this;
        _this.stride0 = _this.strides[0];
        return _this;
    }
    Array2D.new = function (shape, values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array2D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array2D(shape, { values: toTypedArray(values) });
    };
    Array2D.prototype.get = function (i, j) {
        return this.getValues()[this.stride0 * i + j];
    };
    Array2D.prototype.set = function (value, i, j) {
        this.getValues()[this.stride0 * i + j] = value;
    };
    Array2D.prototype.add = function (value, i, j) {
        this.getValues()[this.stride0 * i + j] += value;
    };
    Array2D.prototype.locToIndex = function (locs) {
        return this.stride0 * locs[0] + locs[1];
    };
    Array2D.prototype.indexToLoc = function (index) {
        return [Math.floor(index / this.stride0), index % this.stride0];
    };
    Array2D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array2D;
}(NDArray));
exports.Array2D = Array2D;
var Array3D = (function (_super) {
    __extends(Array3D, _super);
    function Array3D(shape, data) {
        var _this = this;
        util.assert(shape.length === 3, 'Shape should be of length 3');
        _this = _super.call(this, shape, data) || this;
        _this.stride0 = _this.strides[0];
        _this.stride1 = _this.strides[1];
        return _this;
    }
    Array3D.new = function (shape, values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array3D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array3D(shape, { values: toTypedArray(values) });
    };
    Array3D.prototype.get = function (i, j, k) {
        return this.getValues()[this.stride0 * i + this.stride1 * j + k];
    };
    Array3D.prototype.set = function (value, i, j, k) {
        this.getValues()[this.stride0 * i + this.stride1 * j + k] = value;
    };
    Array3D.prototype.add = function (value, i, j, k) {
        this.getValues()[this.stride0 * i + this.stride1 * j + k] += value;
    };
    Array3D.prototype.locToIndex = function (locs) {
        return this.stride0 * locs[0] + this.stride1 * locs[1] + locs[2];
    };
    Array3D.prototype.indexToLoc = function (index) {
        var i = Math.floor(index / this.stride0);
        index -= i * this.stride0;
        return [i, Math.floor(index / this.stride1), index % this.stride1];
    };
    Array3D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array3D;
}(NDArray));
exports.Array3D = Array3D;
var Array4D = (function (_super) {
    __extends(Array4D, _super);
    function Array4D(shape, data) {
        var _this = this;
        util.assert(shape.length === 4, 'Shape should be of length 4');
        _this = _super.call(this, shape, data) || this;
        _this.stride0 = _this.strides[0];
        _this.stride1 = _this.strides[1];
        _this.stride2 = _this.strides[2];
        return _this;
    }
    Array4D.new = function (shape, values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array4D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array4D(shape, { values: toTypedArray(values) });
    };
    Array4D.prototype.get = function (i, j, k, l) {
        return this.getValues()[this.stride0 * i + this.stride1 * j + this.stride2 * k + l];
    };
    Array4D.prototype.set = function (value, i, j, k, l) {
        this.getValues()[this.stride0 * i + this.stride1 * j + this.stride2 * k + l] = value;
    };
    Array4D.prototype.add = function (value, i, j, k, l) {
        this.getValues()[this.stride0 * i + this.stride1 * j + this.stride2 * k + l] += value;
    };
    Array4D.prototype.locToIndex = function (locs) {
        return this.stride0 * locs[0] + this.stride1 * locs[1] +
            this.stride2 * locs[2] + locs[3];
    };
    Array4D.prototype.indexToLoc = function (index) {
        var i = Math.floor(index / this.stride0);
        index -= i * this.stride0;
        var j = Math.floor(index / this.stride1);
        index -= j * this.stride1;
        return [i, j, Math.floor(index / this.stride2), index % this.stride2];
    };
    Array4D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array4D;
}(NDArray));
exports.Array4D = Array4D;
function toTypedArray(a) {
    return (a instanceof Float32Array) ? a : new Float32Array(util.flatten(a));
}

},{"../util":33,"./webgl/webgl_util":31}],20:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
var conv_gpu = require("./conv_gpu");
function getFragmentShaderDerWeightsSource(xShapeRowColDepth, fSize, outputDepth, stride, zeroPad) {
    var getMatrixValueOrZeroPad = conv_gpu.getFragmentShaderGetMatrixValueOrZeroPadSource();
    var inputDepth = xShapeRowColDepth[2];
    var xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRowColDepth);
    var yShape = conv_util.computeOutputShape3D(xShapeRowColDepth, fSize, outputDepth, stride, zeroPad);
    var yNumRows = yShape[0];
    var yNumCols = yShape[1];
    var yTexShapeRC = conv_util.computeTexShapeFrom3D(yShape);
    var fSizeTimesInputDepth = fSize * inputDepth;
    var prologue = "\n    precision highp float;\n    uniform sampler2D x;\n    uniform sampler2D dy;\n  ";
    return prologue + '\n' + getMatrixValueOrZeroPad + '\n' +
        ("\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const vec2 xShapeCR = vec2(" + xTexShapeRC[1] + ", " + xTexShapeRC[0] + ");\n    const vec2 dyShapeCR = vec2(" + yTexShapeRC[1] + ", " + yTexShapeRC[0] + ");\n\n    void main() {\n      vec2 wTexCR = floor(gl_FragCoord.xy);\n\n      // Map from 2D (wTexR, wTexC) to 4D (wR, wC, d1, d2).\n      float wR = floor(wTexCR.y / " + fSizeTimesInputDepth + ".0);\n      float wTexRLeftover = wTexCR.y - wR * " + fSizeTimesInputDepth + ".0;\n      float wC = floor(wTexRLeftover / " + inputDepth + ".0);\n      float d1 = mod(wTexRLeftover, " + inputDepth + ".0);\n      float d2 = wTexCR.x;\n\n      // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).\n      // ? = to be determined. : = across all values in that axis.\n      float dotProd = 0.0;\n      for (float yR = 0.0; yR < " + yNumRows + ".0; yR += 1.0) {\n        float xR = wR + yR * " + stride + ".0 - " + zeroPad + ".0;\n        float xTexR = xR;\n        float yTexR = yR;\n        for (float yC = 0.0; yC < " + yNumCols + ".0; yC += 1.0) {\n          float xC = wC + yC * " + stride + ".0 - " + zeroPad + ".0;\n\n          // Map from 3D (xR, xC, d1) to 2D (xTexR, xTexC).\n          // Map from 3D (yR, yC, d2) to 2D (yTexR, yTexC).\n          vec2 xyTexC = vec2(xC, yC) * vec2(" + inputDepth + ".0, " + outputDepth + ".0) +\n                        vec2(d1, d2);\n          float xTexC = xyTexC.x;\n          float yTexC = xyTexC.y;\n\n          // Read dy(yR, yC, d2).\n          vec2 dyUV = (vec2(yTexC, yTexR) + halfCR) / dyShapeCR;\n          float dyValue = texture2D(dy, dyUV).r;\n\n          // Read x(xR, xC, d1) (potentially zero-padded).\n          float xValue =\n            getMatrixValueOrZeroPad(x, xShapeCR, vec2(xTexC, xTexR));\n\n          dotProd += (xValue * dyValue);\n        }\n      }\n      gl_FragColor = vec4(dotProd, 0, 0, 0);\n    }");
}
exports.getFragmentShaderDerWeightsSource = getFragmentShaderDerWeightsSource;
function getFragmentShaderConvTransposeSource(xShapeRCD, fSize, origInputDepth, origStride, origPad, hasBias) {
    var pad = fSize - 1 - origPad;
    var xRows = xShapeRCD[0], xCols = xShapeRCD[1], origOutputDepth = xShapeRCD[2];
    var xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);
    var wTexShapeRC = conv_util.computeWeightsTexShape(origInputDepth, origOutputDepth, fSize);
    var getBiasValue = hasBias ?
        conv_gpu.getFragmentShaderGetBiasValueSource(origInputDepth) :
        '';
    var biasPrologue = hasBias ? 'uniform sampler2D biases;' : '';
    var biasOperation = hasBias ? 'dotProd += getBiasValue(biases, d2);' : '';
    var prologue = "\n    precision highp float;\n    uniform sampler2D x;\n    uniform sampler2D weights;\n    " + biasPrologue + "\n    ";
    return prologue + '\n' + getBiasValue + '\n' +
        ("\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const vec2 xShapeCR = vec2(" + xTexShapeRC[1] + ", " + xTexShapeRC[0] + ");\n    const vec2 wShapeCR = vec2(" + wTexShapeRC[1] + ", " + wTexShapeRC[0] + ");\n\n    void main() {\n      vec2 yTexCR = floor(gl_FragCoord.xy);\n\n      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, d2).\n      float yR = yTexCR.y;\n      float yC = floor(yTexCR.x / " + origInputDepth + ".0);\n      float d2 = mod(yTexCR.x, " + origInputDepth + ".0);\n\n      vec2 xRCCorner = vec2(yR, yC) - vec2(" + pad + ".0, " + pad + ".0);\n      float xRCorner = xRCCorner.x;\n      float xCCorner = xRCCorner.y;\n\n      // Convolve x(?, ?, d1) with w(:, :, d2, d1) to get y(yR, yC, d2).\n      // ? = to be determined. : = across all values in that axis.\n      float dotProd = 0.0;\n      for (float wR = 0.0; wR < " + fSize + ".0; wR += 1.0) {\n\n        float xR = (xRCorner + wR) / " + origStride + ".0;\n        // TODO(smilkov): Splice this with another version where you call\n        // getMatrixValueOrZeroPad(). Here and below.\n        if (xR < 0.0 || xR >= " + xRows + ".0 || fract(xR) > 0.0) {\n          continue;\n        }\n\n        float wRPerm = " + fSize + ".0 - 1.0 - wR;\n        float xTexR = xR;\n\n        for (float wC = 0.0; wC < " + fSize + ".0; wC += 1.0) {\n\n          float xC = (xCCorner + wC) / " + origStride + ".0;\n          if (xC < 0.0 || xC >= " + xCols + ".0 || fract(xC) > 0.0) {\n            continue;\n          }\n\n          float wCPerm = " + fSize + ".0 - 1.0 - wC;\n          float wTexR = wRPerm * " + fSize + ".0 * " + origInputDepth + ".0 +\n                        wCPerm * " + origInputDepth + ".0 + d2;\n\n          for (float d1 = 0.0; d1 < " + origOutputDepth + ".0; d1 += 1.0) {\n            float xTexC = xC * " + origOutputDepth + ".0 + d1;\n            float wTexC = d1;\n\n            // Read x(xR, xC, d1).\n            vec2 xUV = (vec2(xTexC, xTexR) + halfCR) / xShapeCR;\n            float xValue = texture2D(x, xUV).r;\n\n            // Read w(wRPerm, wCPerm, d2, d1).\n            vec2 wUV = (vec2(wTexC, wTexR) + halfCR) / wShapeCR;\n            float wValue = texture2D(weights, wUV).r;\n\n            dotProd += xValue * wValue;\n          }\n        }\n      }\n      " + biasOperation + "\n      gl_FragColor = vec4(dotProd, 0, 0, 0);\n    }");
}
exports.getFragmentShaderConvTransposeSource = getFragmentShaderConvTransposeSource;
function getFragmentShaderDerBiasSource(dyShapeRCD) {
    var dyTexShapeRC = conv_util.computeTexShapeFrom3D(dyShapeRCD);
    var yNumRows = dyShapeRCD[0], yNumCols = dyShapeRCD[1], outputDepth = dyShapeRCD[2];
    return "\n    precision highp float;\n    uniform sampler2D dy;\n\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const vec2 dyShapeCR = vec2(" + dyTexShapeRC[1] + ", " + dyTexShapeRC[0] + ");\n\n    void main() {\n      vec2 biasTexCR = floor(gl_FragCoord.xy);\n\n      // The bias texture RC shape is [1, d2].\n      float d2 = biasTexCR.x;\n\n      float derBias = 0.0;\n      for (float yR = 0.0; yR < " + yNumRows + ".0; yR += 1.0) {\n        float yTexR = yR;\n\n        for (float yC = 0.0; yC < " + yNumCols + ".0; yC += 1.0) {\n          // Map from 3D (yR, yC, d2) to 2D (yTexR, yTexC).\n          float yTexC = yC * " + outputDepth + ".0 + d2;\n\n          // Read dy(yR, yC, d2).\n          vec2 dyUV = (vec2(yTexC, yTexR) + halfCR) / dyShapeCR;\n          float dyValue = texture2D(dy, dyUV).r;\n\n          derBias += dyValue;\n        }\n      }\n      gl_FragColor = vec4(derBias, 0, 0, 0);\n    }";
}
exports.getFragmentShaderDerBiasSource = getFragmentShaderDerBiasSource;
function derBias(gpgpu, program, dyTex, result, resultTexShapeRC) {
    gpgpu.setOutputMatrixTexture(result, resultTexShapeRC[0], resultTexShapeRC[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(dyTex, 'dy', 0);
    gpgpu.executeProgram();
}
exports.derBias = derBias;
function derWeights(gpgpu, program, xTex, dyTex, result, resultTexShapeRC) {
    gpgpu.setOutputMatrixTexture(result, resultTexShapeRC[0], resultTexShapeRC[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(xTex, 'x', 0);
    gpgpu.setInputMatrixTexture(dyTex, 'dy', 1);
    gpgpu.executeProgram();
}
exports.derWeights = derWeights;
function convTranspose(gpgpu, program, xTex, weightsTex, biasesTex, resultTex, resultTexShapeRC) {
    gpgpu.setOutputMatrixTexture(resultTex, resultTexShapeRC[0], resultTexShapeRC[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(xTex, 'x', 0);
    gpgpu.setInputMatrixTexture(weightsTex, 'weights', 1);
    if (biasesTex != null) {
        gpgpu.setInputMatrixTexture(biasesTex, 'biases', 2);
    }
    gpgpu.executeProgram();
}
exports.convTranspose = convTranspose;

},{"../conv_util":15,"./conv_gpu":21}],21:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
function getFragmentShaderPrologueSource() {
    return "\n    precision highp float;\n    uniform sampler2D x;\n    uniform sampler2D weights;\n    uniform sampler2D biases;\n    varying vec2 resultUV;";
}
exports.getFragmentShaderPrologueSource = getFragmentShaderPrologueSource;
function getFragmentShaderGetMatrixValueOrZeroPadSource() {
    return "\n    float getMatrixValueOrZeroPad(in sampler2D matrix, vec2 matrixShapeCR,\n        vec2 requestedCR) {\n      vec2 uv = (requestedCR + vec2(0.5, 0.5)) / matrixShapeCR;\n      float value = texture2D(matrix, uv).r;\n      bool lessThanZero = any(lessThan(uv, vec2(0, 0)));\n      bool greaterThanOne = any(greaterThan(uv, vec2(1, 1)));\n      bool outside = lessThanZero || greaterThanOne;\n      return mix(value, 0.0, float(outside));\n    }";
}
exports.getFragmentShaderGetMatrixValueOrZeroPadSource = getFragmentShaderGetMatrixValueOrZeroPadSource;
function getFragmentShaderConvolveSource(xShapeRCD, fSize, outputDepth, stride, pad, hasBias) {
    var xRows = xShapeRCD[0], xCols = xShapeRCD[1], inputDepth = xShapeRCD[2];
    var xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);
    var wTexShapeRC = conv_util.computeWeightsTexShape(inputDepth, outputDepth, fSize);
    return "\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const vec2 xShapeCR = vec2(" + xTexShapeRC[1] + ", " + xTexShapeRC[0] + ");\n    const vec2 wShapeCR = vec2(" + wTexShapeRC[1] + ", " + wTexShapeRC[0] + ");\n\n    void main() {\n      vec2 yTexCR = floor(gl_FragCoord.xy);\n\n      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, d2).\n      float yR = yTexCR.y;\n      float yC = floor(yTexCR.x / " + outputDepth + ".0);\n      float d2 = mod(yTexCR.x, " + outputDepth + ".0);\n      float wTexC = d2;\n\n      vec2 xRCCorner = vec2(yR, yC) * vec2(" + stride + ", " + stride + ") -\n          vec2(" + pad + ".0, " + pad + ".0);\n      float xRCorner = xRCCorner.x;\n      float xCCorner = xRCCorner.y;\n\n      // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).\n      // ? = to be determined. : = across all values in that axis.\n      float dotProd = 0.0;\n      for (float wR = 0.0; wR < " + fSize + ".0; wR += 1.0) {\n        float xR = xRCorner + wR;\n        float xTexR = xR;\n\n        for (float wC = 0.0; wC < " + fSize + ".0; wC += 1.0) {\n          float xC = xCCorner + wC;\n\n          for (float d1 = 0.0; d1 < " + inputDepth + ".0; d1 += 1.0) {\n            float xTexC = xC * " + inputDepth + ".0 + d1;\n            float wTexR = wR * " + fSize * inputDepth + ".0 +\n                wC * " + inputDepth + ".0 + d1;\n\n            float xValue =\n                getMatrixValueOrZeroPad(x, xShapeCR, vec2(xTexC, xTexR));\n\n            // Read w(wR, wC, d1, d2).\n            vec2 wUV = (vec2(wTexC, wTexR) + halfCR) / wShapeCR;\n            float wValue = texture2D(weights, wUV).r;\n\n            dotProd += xValue * wValue;\n          }\n        }\n      }\n      if (" + hasBias + ") {\n        dotProd += getBiasValue(biases, d2);\n      }\n      gl_FragColor = vec4(dotProd, 0, 0, 0);\n    }";
}
exports.getFragmentShaderConvolveSource = getFragmentShaderConvolveSource;
function getFragmentShaderGetBiasValueSource(outputDepth) {
    return "\n    float getBiasValue(in sampler2D bias, float biasC) {\n      const vec2 biasShapeCR = vec2(" + outputDepth + ", 1);\n      vec2 biasCR = vec2(mod(biasC, " + outputDepth + ".0), 0);\n      vec2 biasUV = (biasCR + vec2(0.5, 0.5)) / biasShapeCR;\n      return texture2D(bias, biasUV).r;\n    }";
}
exports.getFragmentShaderGetBiasValueSource = getFragmentShaderGetBiasValueSource;
function getFragmentShaderSource(aShapeRowColDepth, resultDepth, fieldSize, stride, zeroPad, hasBias) {
    var aShapeRC = conv_util.computeTexShapeFrom3D(aShapeRowColDepth);
    var weightShapeRC = conv_util.computeWeightsTexShape(aShapeRowColDepth[2], resultDepth, fieldSize);
    var prologue = getFragmentShaderPrologueSource();
    var getMatrixValueOrZeroPad = getFragmentShaderGetMatrixValueOrZeroPadSource();
    var convolve = getFragmentShaderConvolveSource(aShapeRowColDepth, fieldSize, resultDepth, stride, zeroPad, hasBias);
    var getBiasValue = getFragmentShaderGetBiasValueSource(resultDepth);
    return [
        prologue,
        getMatrixValueOrZeroPad,
        getBiasValue,
        convolve,
    ].join('\n');
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function convolve(gpgpu, program, a, weights, biases, result, resultShapeRowCol) {
    gpgpu.setOutputMatrixTexture(result, resultShapeRowCol[0], resultShapeRowCol[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(a, 'x', 0);
    gpgpu.setInputMatrixTexture(weights, 'weights', 1);
    if (biases != null) {
        gpgpu.setInputMatrixTexture(biases, 'biases', 2);
    }
    gpgpu.executeProgram();
}
exports.convolve = convolve;

},{"../conv_util":15}],22:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_util = require("./gpgpu_util");
var tex_util = require("./tex_util");
var webgl_util = require("./webgl_util");
var GPGPUContext = (function () {
    function GPGPUContext(gl) {
        this.outputTexture = null;
        this.program = null;
        this.disposed = false;
        this.autoDebugValidate = false;
        if (gl != null) {
            this.gl = gl;
        }
        else {
            this.gl = gpgpu_util.createWebGLContext();
        }
        if (!webgl_util.isWebGL2Enabled()) {
            this.textureFloatExtension =
                webgl_util.getExtensionOrThrow(this.gl, 'OES_texture_float');
        }
        else {
            this.colorBufferFloatExtension =
                webgl_util.getExtensionOrThrow(this.gl, 'EXT_color_buffer_float');
        }
        this.loseContextExtension =
            webgl_util.getExtensionOrThrow(this.gl, 'WEBGL_lose_context');
        this.vertexBuffer = gpgpu_util.createVertexBuffer(this.gl);
        this.indexBuffer = gpgpu_util.createIndexBuffer(this.gl);
        this.framebuffer = webgl_util.createFramebuffer(this.gl);
    }
    GPGPUContext.prototype.dispose = function () {
        var _this = this;
        this.throwIfDisposed();
        if (this.program != null) {
            console.warn('Disposing a GPGPUContext that still has a bound WebGLProgram.' +
                ' This is probably a resource leak, delete the program with ' +
                'GPGPUContext.deleteProgram before disposing.');
        }
        if (this.outputTexture != null) {
            console.warn('Disposing a GPGPUContext that still has a bound output matrix ' +
                'texture.  This is probably a resource leak, delete the output ' +
                'matrix texture with GPGPUContext.deleteMatrixTexture before ' +
                'disposing.');
        }
        var gl = this.gl;
        webgl_util.callAndCheck(gl, function () { return gl.finish(); });
        webgl_util.callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, null); });
        webgl_util.callAndCheck(gl, function () { return gl.deleteFramebuffer(_this.framebuffer); });
        webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, null); });
        webgl_util.callAndCheck(gl, function () { return gl.deleteBuffer(_this.vertexBuffer); });
        webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null); });
        webgl_util.callAndCheck(gl, function () { return gl.deleteBuffer(_this.indexBuffer); });
        this.loseContextExtension.loseContext();
        this.disposed = true;
    };
    GPGPUContext.prototype.enableAutomaticDebugValidation = function (enabled) {
        this.autoDebugValidate = enabled;
        webgl_util.enableDebugWebGLErrorChecking(enabled);
    };
    GPGPUContext.prototype.createMatrixTexture = function (rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createMatrixTexture(this.gl, rows, columns);
    };
    GPGPUContext.prototype.uploadPixelDataToTexture = function (texture, pixels) {
        this.throwIfDisposed();
        gpgpu_util.uploadPixelDataToTexture(this.gl, texture, pixels);
    };
    GPGPUContext.prototype.createPackedMatrixTexture = function (rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createPackedMatrixTexture(this.gl, rows, columns);
    };
    GPGPUContext.prototype.deleteMatrixTexture = function (texture) {
        var _this = this;
        this.throwIfDisposed();
        if (this.outputTexture === texture) {
            webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
            this.outputTexture = null;
        }
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.deleteTexture(texture); });
    };
    GPGPUContext.prototype.uploadMatrixToTexture = function (texture, rows, columns, matrix) {
        this.throwIfDisposed();
        var numChannels = 1;
        return gpgpu_util.uploadMatrixToTexture(this.gl, texture, rows, columns, matrix, numChannels);
    };
    GPGPUContext.prototype.uploadMatrixToPackedTexture = function (texture, rows, columns, matrix) {
        this.throwIfDisposed();
        return gpgpu_util.uploadMatrixToPackedTexture(this.gl, texture, rows, columns, matrix);
    };
    GPGPUContext.prototype.downloadMatrixFromTexture = function (texture, rows, columns) {
        var _this = this;
        return this.downloadMatrixDriver(texture, function () {
            return gpgpu_util.downloadMatrixFromOutputTexture(_this.gl, rows, columns);
        });
    };
    GPGPUContext.prototype.downloadMatrixFromPackedTexture = function (texture, rows, columns) {
        var _this = this;
        return this.downloadMatrixDriver(texture, function () { return gpgpu_util.downloadMatrixFromPackedOutputTexture(_this.gl, rows, columns); });
    };
    GPGPUContext.prototype.createProgram = function (fragmentShaderSource) {
        this.throwIfDisposed();
        var gl = this.gl;
        var fragmentShader = webgl_util.createFragmentShader(gl, fragmentShaderSource);
        var vertexShader = gpgpu_util.createVertexShader(gl);
        var program = webgl_util.createProgram(gl);
        webgl_util.callAndCheck(gl, function () { return gl.attachShader(program, vertexShader); });
        webgl_util.callAndCheck(gl, function () { return gl.attachShader(program, fragmentShader); });
        webgl_util.linkProgram(gl, program);
        if (this.autoDebugValidate) {
            webgl_util.validateProgram(gl, program);
        }
        return program;
    };
    GPGPUContext.prototype.deleteProgram = function (program) {
        var _this = this;
        this.throwIfDisposed();
        if (program === this.program) {
            this.program = null;
        }
        if (program != null) {
            webgl_util.callAndCheck(this.gl, function () { return _this.gl.deleteProgram(program); });
        }
    };
    GPGPUContext.prototype.setProgram = function (program) {
        var _this = this;
        this.throwIfDisposed();
        this.program = program;
        if ((this.program != null) && this.autoDebugValidate) {
            webgl_util.validateProgram(this.gl, this.program);
        }
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.useProgram(program); });
    };
    GPGPUContext.prototype.getUniformLocation = function (uniformName) {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        return webgl_util.getProgramUniformLocationOrThrow(this.gl, this.program, uniformName);
    };
    GPGPUContext.prototype.setInputMatrixTexture = function (inputMatrixTexture, uniformName, textureUnit) {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        webgl_util.bindTextureToProgramUniformSampler(this.gl, this.program, inputMatrixTexture, uniformName, textureUnit);
    };
    GPGPUContext.prototype.setOutputMatrixTexture = function (outputMatrixTexture, rows, columns) {
        this.setOutputMatrixTextureDriver(outputMatrixTexture, columns, rows);
    };
    GPGPUContext.prototype.setOutputPackedMatrixTexture = function (outputPackedMatrixTexture, rows, columns) {
        this.throwIfDisposed();
        var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
        this.setOutputMatrixTextureDriver(outputPackedMatrixTexture, width, height);
    };
    GPGPUContext.prototype.setOutputMatrixWriteRegion = function (startRow, numRows, startColumn, numColumns) {
        this.setOutputMatrixWriteRegionDriver(startColumn, startRow, numColumns, numRows);
    };
    GPGPUContext.prototype.setOutputPackedMatrixWriteRegion = function (startRow, numRows, startColumn, numColumns) {
        throw new Error('setOutputPackedMatrixWriteRegion not implemented.');
    };
    GPGPUContext.prototype.debugValidate = function () {
        if (this.program != null) {
            webgl_util.validateProgram(this.gl, this.program);
        }
        webgl_util.validateFramebuffer(this.gl);
    };
    GPGPUContext.prototype.executeProgram = function () {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        var gl = this.gl;
        gpgpu_util.bindVertexProgramAttributeStreams(gl, this.program, this.vertexBuffer);
        if (this.autoDebugValidate) {
            this.debugValidate();
        }
        webgl_util.callAndCheck(gl, function () { return gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0); });
    };
    GPGPUContext.prototype.blockUntilAllProgramsCompleted = function () {
        var _this = this;
        this.throwIfDisposed();
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.finish(); });
    };
    GPGPUContext.prototype.downloadMatrixDriver = function (texture, downloadAndDecode) {
        this.throwIfDisposed();
        webgl_util.bindColorTextureToFramebuffer(this.gl, texture, this.framebuffer);
        var result = downloadAndDecode();
        if (this.outputTexture != null) {
            webgl_util.bindColorTextureToFramebuffer(this.gl, this.outputTexture, this.framebuffer);
            if (this.autoDebugValidate) {
                webgl_util.validateFramebuffer(this.gl);
            }
        }
        else {
            webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
        }
        return result;
    };
    GPGPUContext.prototype.setOutputMatrixTextureDriver = function (outputMatrixTextureMaybePacked, width, height) {
        this.throwIfDisposed();
        var gl = this.gl;
        webgl_util.bindColorTextureToFramebuffer(gl, outputMatrixTextureMaybePacked, this.framebuffer);
        if (this.autoDebugValidate) {
            webgl_util.validateFramebuffer(gl);
        }
        this.outputTexture = outputMatrixTextureMaybePacked;
        webgl_util.callAndCheck(gl, function () { return gl.viewport(0, 0, width, height); });
        webgl_util.callAndCheck(gl, function () { return gl.scissor(0, 0, width, height); });
    };
    GPGPUContext.prototype.setOutputMatrixWriteRegionDriver = function (x, y, width, height) {
        var _this = this;
        this.throwIfDisposed();
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.scissor(x, y, width, height); });
    };
    GPGPUContext.prototype.throwIfDisposed = function () {
        if (this.disposed) {
            throw new Error('Attempted to use disposed GPGPUContext.');
        }
    };
    GPGPUContext.prototype.throwIfNoProgram = function () {
        if (this.program == null) {
            throw new Error('No GPU program is currently set.');
        }
    };
    return GPGPUContext;
}());
exports.GPGPUContext = GPGPUContext;

},{"./gpgpu_util":23,"./tex_util":30,"./webgl_util":31}],23:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tex_util = require("./tex_util");
var webgl_util = require("./webgl_util");
function getWebGLContextAttributes() {
    return {
        alpha: false,
        antialias: false,
        premultipliedAlpha: false,
        preserveDrawingBuffer: false,
        depth: false,
        stencil: false,
        failIfMajorPerformanceCaveat: true
    };
}
exports.getWebGLContextAttributes = getWebGLContextAttributes;
function createWebGLContext(canvas) {
    var attributes = getWebGLContextAttributes();
    var gl;
    if (canvas != null) {
        gl = webgl_util.createWebGLRenderingContextFromCanvas(canvas, attributes);
    }
    else {
        gl = webgl_util.createWebGLRenderingContext(attributes);
    }
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.DEPTH_TEST); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.STENCIL_TEST); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.BLEND); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.DITHER); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.POLYGON_OFFSET_FILL); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.SAMPLE_COVERAGE); });
    webgl_util.callAndCheck(gl, function () { return gl.enable(gl.SCISSOR_TEST); });
    webgl_util.callAndCheck(gl, function () { return gl.enable(gl.CULL_FACE); });
    webgl_util.callAndCheck(gl, function () { return gl.cullFace(gl.BACK); });
    return gl;
}
exports.createWebGLContext = createWebGLContext;
function createVertexShader(gl) {
    var vertexShaderSource = "\n    precision highp float;\n    attribute vec3 clipSpacePos;\n    attribute vec2 uv;\n    varying vec2 resultUV;\n\n    void main() {\n      gl_Position = vec4(clipSpacePos, 1);\n      resultUV = uv;\n    }";
    return webgl_util.createVertexShader(gl, vertexShaderSource);
}
exports.createVertexShader = createVertexShader;
function createVertexBuffer(gl) {
    var vertexArray = new Float32Array([-1, 1, 0, 0, 1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, 0]);
    return webgl_util.createStaticVertexBuffer(gl, vertexArray);
}
exports.createVertexBuffer = createVertexBuffer;
function createIndexBuffer(gl) {
    var triangleVertexIndices = new Uint16Array([0, 1, 2, 2, 1, 3]);
    return webgl_util.createStaticIndexBuffer(gl, triangleVertexIndices);
}
exports.createIndexBuffer = createIndexBuffer;
function getTextureInternalFormat(gl, numChannels) {
    if (webgl_util.isWebGL2Enabled()) {
        if (numChannels === 4) {
            return gl.RGBA32F;
        }
        return gl.R32F;
    }
    return gl.RGBA;
}
function getTextureFormat(gl, numChannels) {
    if (webgl_util.isWebGL2Enabled() && numChannels === 1) {
        return gl.RED;
    }
    return gl.RGBA;
}
function createAndConfigureTexture(gl, width, height, numChannels) {
    webgl_util.validateTextureSize(gl, width, height);
    var texture = webgl_util.createTexture(gl);
    var tex2d = gl.TEXTURE_2D;
    var internalFormat = getTextureInternalFormat(gl, numChannels);
    var format = getTextureFormat(gl, numChannels);
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(tex2d, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST); });
    webgl_util.callAndCheck(gl, function () { return gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, format, gl.FLOAT, null); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
    return texture;
}
function createMatrixTexture(gl, rows, columns) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    var numChannels = 1;
    return createAndConfigureTexture(gl, width, height, numChannels);
}
exports.createMatrixTexture = createMatrixTexture;
function createColorMatrixTexture(gl, rows, columns) {
    var _a = tex_util.getColorMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    var numChannels = 4;
    return createAndConfigureTexture(gl, width, height, numChannels);
}
exports.createColorMatrixTexture = createColorMatrixTexture;
function createPackedMatrixTexture(gl, rows, columns) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    var numChannels = 4;
    return createAndConfigureTexture(gl, width, height, numChannels);
}
exports.createPackedMatrixTexture = createPackedMatrixTexture;
function bindVertexProgramAttributeStreams(gl, program, vertexBuffer) {
    var posOffset = 0;
    var uvOffset = 3 * 4;
    var stride = (3 * 4) + (2 * 4);
    webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer); });
    webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'clipSpacePos', vertexBuffer, 3, stride, posOffset);
    try {
        webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'uv', vertexBuffer, 2, stride, uvOffset);
    }
    catch (e) {
        if (!e.hasOwnProperty('namedVertexAttributeNotFound')) {
            throw e;
        }
    }
}
exports.bindVertexProgramAttributeStreams = bindVertexProgramAttributeStreams;
function uploadPixelDataToTexture(gl, texture, pixels) {
    var numChannels = 4;
    var internalFormat = getTextureInternalFormat(gl, numChannels);
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, gl.RGBA, gl.FLOAT, pixels); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
}
exports.uploadPixelDataToTexture = uploadPixelDataToTexture;
function uploadDataToTexture(gl, texture, width, height, data, numChannels) {
    var textureFormat = getTextureFormat(gl, numChannels);
    webgl_util.validateTextureSize(gl, width, height);
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, textureFormat, gl.FLOAT, data); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
}
function uploadMatrixToTexture(gl, texture, rows, columns, matrix, numChannels) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var channelsPerTexture = numChannels === 1 ? webgl_util.getChannelsPerTexture() : numChannels;
    var unpackedArray = new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(matrix.length, channelsPerTexture));
    tex_util.encodeMatrixToUnpackedArray(matrix, unpackedArray, channelsPerTexture);
    uploadDataToTexture(gl, texture, w, h, unpackedArray, numChannels);
}
exports.uploadMatrixToTexture = uploadMatrixToTexture;
function uploadMatrixToPackedTexture(gl, texture, rows, columns, matrix) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var packedRGBA = new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(rows, columns));
    tex_util.encodeMatrixToPackedRGBA(matrix, rows, columns, packedRGBA);
    var numChannels = 4;
    uploadDataToTexture(gl, texture, w, h, packedRGBA, numChannels);
}
exports.uploadMatrixToPackedTexture = uploadMatrixToPackedTexture;
function downloadMatrixFromOutputTexture(gl, rows, columns) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var channelsPerTexture = 4;
    var unpackedArray = new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(rows * columns, channelsPerTexture));
    var textureFormat = getTextureFormat(gl, channelsPerTexture);
    webgl_util.callAndCheck(gl, function () { return gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, unpackedArray); });
    var matrix = new Float32Array(rows * columns);
    tex_util.decodeMatrixFromUnpackedArray(unpackedArray, matrix, channelsPerTexture);
    return matrix;
}
exports.downloadMatrixFromOutputTexture = downloadMatrixFromOutputTexture;
function downloadMatrixFromPackedOutputTexture(gl, rows, columns) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var packedRGBA = new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(rows, columns));
    webgl_util.callAndCheck(gl, function () { return gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, packedRGBA); });
    var matrix = new Float32Array(rows * columns);
    return tex_util.decodeMatrixFromPackedRGBA(packedRGBA, rows, columns, matrix);
}
exports.downloadMatrixFromPackedOutputTexture = downloadMatrixFromPackedOutputTexture;

},{"./tex_util":30,"./webgl_util":31}],24:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_context_1 = require("./gpgpu_context");
function getFragmentShaderSource(rows, columns) {
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    varying vec2 resultUV;\n\n    const vec2 aDimCR = vec2(" + columns + ".0, " + rows + ".0);\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    void main() {\n      float aMax = texture2D(matrixA, halfCR / aDimCR).r;\n      for (float r = 0.0; r < aDimCR.y; r += 1.0) {\n        for (float c = 0.0; c < aDimCR.x; c += 1.0) {\n          vec2 uv = (vec2(c, r) + halfCR) / aDimCR;\n          float aCur = texture2D(matrixA, uv).r;\n          aMax = max(aMax, aCur);\n        }\n      }\n\n      float expSum = 0.0;\n      for (float r = 0.0; r < aDimCR.y; r += 1.0) {\n        for (float c = 0.0; c < aDimCR.x; c += 1.0) {\n          vec2 uv = (vec2(c, r) + halfCR) / aDimCR;\n          float aCur = texture2D(matrixA, uv).r;\n          expSum += exp(aCur - aMax);\n        }\n      }\n\n      gl_FragColor = vec4(aMax + log(expSum), 0, 0, 0);\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function logSumExp(gpgpu, logSumExpProgram, a, rows, columns, result) {
    gpgpu.setOutputMatrixTexture(result, 1, 1);
    gpgpu.setProgram(logSumExpProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.executeProgram();
}
exports.logSumExp = logSumExp;
function uploadLogSumExpDownload(a, rows, columns) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(getFragmentShaderSource(rows, columns));
    var aTexture = gpgpu.createMatrixTexture(rows, columns);
    var resultTexture = gpgpu.createMatrixTexture(1, 1);
    gpgpu.uploadMatrixToTexture(aTexture, rows, columns, a);
    logSumExp(gpgpu, program, aTexture, rows, columns, resultTexture);
    var result = gpgpu.downloadMatrixFromTexture(resultTexture, 1, 1);
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result[0];
}
exports.uploadLogSumExpDownload = uploadLogSumExpDownload;

},{"./gpgpu_context":22}],25:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var pool_gpu = require("./pool_gpu");
function getFragmentShaderMaxPoolPositionsSource(xShapeRCD, fSize, stride, pad) {
    return getFragmentShaderMaxPoolCommonSource(xShapeRCD, fSize, stride, pad, true);
}
exports.getFragmentShaderMaxPoolPositionsSource = getFragmentShaderMaxPoolPositionsSource;
function getFragmentShaderMaxPoolSource(xShapeRCD, fSize, stride, pad) {
    return getFragmentShaderMaxPoolCommonSource(xShapeRCD, fSize, stride, pad, false);
}
exports.getFragmentShaderMaxPoolSource = getFragmentShaderMaxPoolSource;
function getFragmentShaderMaxPoolCommonSource(xShapeRCD, fSize, stride, pad, computeMaxPositions) {
    return pool_gpu.getFragmentShaderPoolCommonSource(xShapeRCD, fSize, stride, pad, 'max', computeMaxPositions);
}
function maxPoolCommon(gpgpu, program, x, result, resultShapeRowCol) {
    pool_gpu.poolCommon(gpgpu, program, x, result, resultShapeRowCol);
}
exports.maxPoolCommon = maxPoolCommon;

},{"./pool_gpu":28}],26:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_1 = require("../math");
var shader_compiler = require("./shader_compiler");
function getFragmentShader(a, b, out, aOrientation, bOrientation) {
    var sharedDim = (aOrientation === math_1.MatrixOrientation.REGULAR ? a.shape[1] : a.shape[0]);
    var aSnippet = (aOrientation === math_1.MatrixOrientation.REGULAR) ? 'aRow, i' : 'i, aRow';
    var bSnippet = (bOrientation === math_1.MatrixOrientation.REGULAR) ? 'i, bCol' : 'bCol, i';
    var inputs = [{ name: 'matrixA', array: a }, { name: 'matrixB', array: b }];
    var userCode = "\n    const float sharedDim = " + sharedDim + ".0;\n\n    float dotARowBCol(float aRow, float bCol) {\n      float result = 0.0;\n      for (float i = 0.0; i < sharedDim; i += 1.0) {\n        float a = getMatrixA(" + aSnippet + ");\n        float b = getMatrixB(" + bSnippet + ");\n        result += (a * b);\n      }\n      return result;\n    }\n\n    void main() {\n      vec2 resRC = getOutputCoords();\n      setOutput(dotARowBCol(resRC.x, resRC.y));\n    }\n  ";
    return shader_compiler.makeShader(inputs, out, userCode);
}
exports.getFragmentShader = getFragmentShader;
function multiplyMatrix(gpgpu, multiplyProgram, a, b, result, outTexShape) {
    gpgpu.setOutputMatrixTexture(result, outTexShape[0], outTexShape[1]);
    gpgpu.setProgram(multiplyProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
    gpgpu.executeProgram();
}
exports.multiplyMatrix = multiplyMatrix;

},{"../math":17,"./shader_compiler":29}],27:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_1 = require("../math");
var gpgpu_context_1 = require("./gpgpu_context");
function getFragmentShaderSource(sharedDimension, aOrientation, bOrientation) {
    var sharedDimensionPacked = Math.ceil(sharedDimension / 2);
    var aSample = (aOrientation === math_1.MatrixOrientation.REGULAR) ?
        'center, resultUV.t' :
        'resultUV.t, center';
    var bSample = (bOrientation === math_1.MatrixOrientation.REGULAR) ?
        'resultUV.s, center' :
        'center, resultUV.s';
    var aSwizzle = (aOrientation === math_1.MatrixOrientation.REGULAR) ? ['a.xxzz', 'a.yyww'] :
        ['a.xxyy', 'a.zzww'];
    var bSwizzle = (bOrientation === math_1.MatrixOrientation.REGULAR) ? ['b.xyxy', 'b.zwzw'] :
        ['b.xzxz', 'b.ywyw'];
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    uniform sampler2D matrixB;\n    varying vec2 resultUV;\n\n    const float sharedDimension = " + sharedDimensionPacked + ".0;\n\n    vec4 dot2x2ARowBCol() {\n      vec4 result = vec4(0, 0, 0, 0);\n      for (float i = 0.0; i < sharedDimension; i += 1.0) {\n        float center = (i + 0.5) / sharedDimension;\n        vec4 a = texture2D(matrixA, vec2(" + aSample + "));\n        vec4 b = texture2D(matrixB, vec2(" + bSample + "));\n        result +=\n          (" + aSwizzle[0] + " * " + bSwizzle[0] + ") + (" + aSwizzle[1] + " * " + bSwizzle[1] + ");\n      }\n      return result;\n    }\n\n    void main() {\n      gl_FragColor = dot2x2ARowBCol();\n    }";
}
exports.getFragmentShaderSource = getFragmentShaderSource;
function multiplyMatrixPacked(gpgpu, multiplyProgram, a, b, result, resultShapeRowCol) {
    gpgpu.setOutputPackedMatrixTexture(result, resultShapeRowCol[0], resultShapeRowCol[1]);
    gpgpu.setProgram(multiplyProgram);
    gpgpu.setInputMatrixTexture(a, 'matrixA', 0);
    gpgpu.setInputMatrixTexture(b, 'matrixB', 1);
    gpgpu.executeProgram();
}
exports.multiplyMatrixPacked = multiplyMatrixPacked;
function uploadMultiplyMatrixPackedDownload(a, aShapeRowCol, b, bShapeRowCol, aOrientation, bOrientation) {
    if (aOrientation === void 0) { aOrientation = math_1.MatrixOrientation.REGULAR; }
    if (bOrientation === void 0) { bOrientation = math_1.MatrixOrientation.REGULAR; }
    var resultNumRows = (aOrientation === math_1.MatrixOrientation.REGULAR) ?
        aShapeRowCol[0] :
        aShapeRowCol[1];
    var resultNumCols = (bOrientation === math_1.MatrixOrientation.REGULAR) ?
        bShapeRowCol[1] :
        bShapeRowCol[0];
    var sharedDimension = (aOrientation === math_1.MatrixOrientation.REGULAR) ?
        aShapeRowCol[1] :
        aShapeRowCol[0];
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var program = gpgpu.createProgram(getFragmentShaderSource(sharedDimension, aOrientation, bOrientation));
    var aTexture = gpgpu.createPackedMatrixTexture(aShapeRowCol[0], aShapeRowCol[1]);
    var bTexture = gpgpu.createPackedMatrixTexture(bShapeRowCol[0], bShapeRowCol[1]);
    var resultTexture = gpgpu.createPackedMatrixTexture(resultNumRows, resultNumCols);
    gpgpu.uploadMatrixToPackedTexture(aTexture, aShapeRowCol[0], aShapeRowCol[1], a);
    gpgpu.uploadMatrixToPackedTexture(bTexture, bShapeRowCol[0], bShapeRowCol[1], b);
    multiplyMatrixPacked(gpgpu, program, aTexture, bTexture, resultTexture, [resultNumRows, resultNumCols]);
    var result = gpgpu.downloadMatrixFromPackedTexture(resultTexture, resultNumRows, resultNumCols);
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(bTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return result;
}
exports.uploadMultiplyMatrixPackedDownload = uploadMultiplyMatrixPackedDownload;

},{"../math":17,"./gpgpu_context":22}],28:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
var webgl_util_1 = require("./webgl_util");
function getFragmentShaderPoolCommonSource(xShapeRCD, fSize, stride, pad, poolType, computePositions) {
    if (poolType === 'avg' && computePositions) {
        throw new Error('Cannot compute positions for average pool.');
    }
    var depth = xShapeRCD[2];
    var xTexShapeRC = conv_util.computeTexShapeFrom3D(xShapeRCD);
    var returnValue = 'minMaxValue';
    if (computePositions) {
        returnValue = 'minMaxPosition';
    }
    else if (poolType === 'avg') {
        returnValue = 'avgValue';
    }
    return "\n    precision highp float;\n    uniform sampler2D x;\n    varying vec2 resultUV;\n\n    const vec2 halfCR = vec2(0.5, 0.5);\n    const vec2 xShapeCR = vec2(" + xTexShapeRC[1] + ", " + xTexShapeRC[0] + ");\n\n    " + webgl_util_1.IS_NAN_SHADER_FUNC + "\n\n    void main() {\n      vec2 yTexCR = floor(gl_FragCoord.xy);\n\n      // Map from 2D (yTexR, yTexC) to 3D (yR, yC, d2).\n      float yR = yTexCR.y;\n      float yC = floor(yTexCR.x / " + depth + ".0);\n      float d = mod(yTexCR.x, " + depth + ".0);\n\n      vec2 xRCCorner = vec2(yR, yC) * vec2(" + stride + ", " + stride + ") -\n          vec2(" + pad + ".0, " + pad + ".0);\n      float xRCorner = xRCCorner.x;\n      float xCCorner = xRCCorner.y;\n\n      // max/min x(?, ?, d) to get y(yR, yC, d).\n      // ? = to be determined\n      float minMaxValue = 0.0;\n      float minMaxValueFound = 0.0;\n      float minMaxPosition = 0.0;\n      float avgValue = 0.0;\n\n      for (float wR = 0.0; wR < " + fSize + ".0; wR += 1.0) {\n        float xR = xRCorner + wR;\n        float xTexR = xR;\n\n        for (float wC = 0.0; wC < " + fSize + ".0; wC += 1.0) {\n          float xC = xCCorner + wC;\n          float xTexC = xC * " + depth + ".0 + d;\n\n          vec2 texCR = vec2(xTexC, xTexR);\n\n          // Check if the requested UV is invalid.\n          vec2 uv = (texCR + halfCR) / xShapeCR;\n          bool lessThanZero = any(lessThan(uv, vec2(0, 0)));\n          bool greaterThanOne = any(greaterThan(uv, vec2(1, 1)));\n          bool outside = lessThanZero || greaterThanOne;\n          if (outside) {\n            continue;\n          }\n\n          float value = texture2D(x, uv).r;\n          if (isNaN(value)) {\n            gl_FragColor = vec4(value, 0, 0, 0);\n            return;\n          }\n          if (" + (poolType === 'avg') + ") {\n            avgValue += value / " + fSize * fSize + ".0;\n          } else {\n            // If a min / max value has already been found, use it. If not, use\n            // the current value.\n            float currentMinMaxValue = mix(\n                value, minMaxValue, minMaxValueFound);\n            if (value " + (poolType === 'min' ? '<=' : '>=') + " currentMinMaxValue) {\n              minMaxValue = value;\n              minMaxValueFound = 1.0;\n              if (" + computePositions + ") {\n                minMaxPosition = wR * " + fSize + ".0 + wC;\n              }\n            }\n          }\n        }\n      }\n      gl_FragColor = vec4(" + returnValue + ", 0, 0, 0);\n    }";
}
exports.getFragmentShaderPoolCommonSource = getFragmentShaderPoolCommonSource;
function poolCommon(gpgpu, program, x, result, resultShapeRowCol) {
    gpgpu.setOutputMatrixTexture(result, resultShapeRowCol[0], resultShapeRowCol[1]);
    gpgpu.setProgram(program);
    gpgpu.setInputMatrixTexture(x, 'x', 0);
    gpgpu.executeProgram();
}
exports.poolCommon = poolCommon;

},{"../conv_util":15,"./webgl_util":31}],29:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
function makeShaderKey(inputs, output) {
    var ins = inputs.map(function (x) { return x.shape + '_' + x.getTextureShapeRC(); });
    return ins.join('_') + '_' + output.shape + '_' + output.getTextureShapeRC();
}
exports.makeShaderKey = makeShaderKey;
function makeShader(inputs, output, userCode) {
    var inputPrefixSnippet = inputs.map(function (x) { return "uniform sampler2D " + x.name + ";"; }).join('\n');
    var inputSamplingSnippet = inputs.map(function (x) { return getInputSamplingSnippet(x); }).join('\n');
    var outTexShape = output.getTextureShapeRC();
    var outputSamplingSnippet = getOutputSamplingSnippet(output.shape, outTexShape);
    var source = [
        SHADER_PREFIX, inputPrefixSnippet, SAMPLE_2D_SNIPPET, inputSamplingSnippet,
        outputSamplingSnippet, userCode
    ].join('\n');
    return source;
}
exports.makeShader = makeShader;
function getInputSamplingSnippet(input) {
    var arr = input.array;
    var shape = arr.shape;
    var texShape = arr.getTextureShapeRC(shape);
    switch (shape.length) {
        case 2:
            return getSampler2D(input.name, shape, texShape);
        default:
            throw new Error(arr.rank + "-D input sampling is not yet supported");
    }
}
function getOutputSamplingSnippet(outShape, outTexShape) {
    switch (outShape.length) {
        case 2:
            return getOutput2DCoords(outShape, outTexShape);
        default:
            throw new Error(outShape.length + "-D output sampling is not yet supported");
    }
}
var SHADER_PREFIX = "\n  precision highp float;\n  varying vec2 resultUV;\n  const vec2 halfCR = vec2(0.5, 0.5);\n\n  void setOutput(float val) {\n    gl_FragColor = vec4(val, 0, 0, 0);\n  }\n";
var SAMPLE_2D_SNIPPET = "\n  float sample2D(sampler2D texture, float texNumR, float texNumC, float numC,\n      float row, float col) {\n    float index = dot(vec2(row, col), vec2(numC, 1.0));\n    float texR = floor(index / texNumC);\n    float texC = mod(index, texNumC);\n    vec2 uv = (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n    return texture2D(texture, uv).r;\n  }\n";
function getOutput2DCoords(shape, texShape) {
    if (util.arraysEqual(shape, texShape)) {
        return "\n      vec2 getOutputCoords() {\n        return floor(gl_FragCoord.yx);\n      }\n    ";
    }
    return "\n    vec2 getOutputCoords() {\n      vec2 resTexRC = floor(gl_FragCoord.yx);\n      float index = dot(resTexRC, vec2(" + texShape[1] + ".0, 1.0));\n      float r = floor(index / " + shape[1] + ".0);\n      float c = mod(index, " + shape[1] + ".0);\n      return vec2(r, c);\n    }\n  ";
}
function getSampler2D(texName, shape, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var tR = texShape[0];
    var tC = texShape[1];
    if (util.arraysEqual(shape, texShape)) {
        return "\n      float " + funcName + "(float row, float col) {\n        vec2 uv = (vec2(col, row) + halfCR) / vec2(" + tC + ".0, " + tR + ".0);\n        return texture2D(" + texName + ", uv).r;\n      }\n    ";
    }
    return "\n    float " + funcName + "(float row, float col) {\n      return sample2D(" + texName + ", " + tR + ".0, " + tC + ".0, " + shape[1] + ".0, row, col);\n    }\n  ";
}

},{"../../util":33}],30:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function getUnpackedMatrixTextureShapeWidthHeight(rows, columns) {
    return [columns, rows];
}
exports.getUnpackedMatrixTextureShapeWidthHeight = getUnpackedMatrixTextureShapeWidthHeight;
function getUnpackedArraySizeFromMatrixSize(matrixSize, channelsPerTexture) {
    return matrixSize * channelsPerTexture;
}
exports.getUnpackedArraySizeFromMatrixSize = getUnpackedArraySizeFromMatrixSize;
function getColorMatrixTextureShapeWidthHeight(rows, columns) {
    return [columns * 4, rows];
}
exports.getColorMatrixTextureShapeWidthHeight = getColorMatrixTextureShapeWidthHeight;
function getMatrixSizeFromUnpackedArraySize(unpackedSize, channelsPerTexture) {
    if (unpackedSize % channelsPerTexture !== 0) {
        throw new Error('unpackedSize (' + unpackedSize + ') must be a multiple of ' +
            channelsPerTexture);
    }
    return unpackedSize / channelsPerTexture;
}
exports.getMatrixSizeFromUnpackedArraySize = getMatrixSizeFromUnpackedArraySize;
function encodeMatrixToUnpackedArray(matrix, unpackedArray, channelsPerTexture) {
    var requiredSize = getUnpackedArraySizeFromMatrixSize(matrix.length, channelsPerTexture);
    if (unpackedArray.length < requiredSize) {
        throw new Error('unpackedArray length (' + unpackedArray.length +
            ') must be >= ' + requiredSize);
    }
    var dst = 0;
    for (var src = 0; src < matrix.length; ++src) {
        unpackedArray[dst] = matrix[src];
        dst += channelsPerTexture;
    }
}
exports.encodeMatrixToUnpackedArray = encodeMatrixToUnpackedArray;
function decodeMatrixFromUnpackedArray(unpackedArray, matrix, channelsPerTexture) {
    var requiredSize = getMatrixSizeFromUnpackedArraySize(unpackedArray.length, channelsPerTexture);
    if (matrix.length < requiredSize) {
        throw new Error('matrix length (' + matrix.length + ') must be >= ' + requiredSize);
    }
    var dst = 0;
    for (var src = 0; src < unpackedArray.length; src += channelsPerTexture) {
        matrix[dst++] = unpackedArray[src];
    }
}
exports.decodeMatrixFromUnpackedArray = decodeMatrixFromUnpackedArray;
function getPackedMatrixTextureShapeWidthHeight(rows, columns) {
    return [Math.ceil(columns / 2), Math.ceil(rows / 2)];
}
exports.getPackedMatrixTextureShapeWidthHeight = getPackedMatrixTextureShapeWidthHeight;
function getPackedRGBAArraySizeFromMatrixShape(rows, columns) {
    var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    return w * h * 4;
}
exports.getPackedRGBAArraySizeFromMatrixShape = getPackedRGBAArraySizeFromMatrixShape;
function encodeMatrixToPackedRGBA(matrix, rows, columns, packedRGBA) {
    var requiredSize = getPackedRGBAArraySizeFromMatrixShape(rows, columns);
    if (packedRGBA.length < requiredSize) {
        throw new Error('packedRGBA length (' + packedRGBA.length +
            ') must be >= ' + requiredSize);
    }
    var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), textureWidth = _a[0], textureHeight = _a[1];
    var oddWidth = (columns % 2) === 1;
    var oddHeight = (rows % 2) === 1;
    var widthInFullBlocks = Math.floor(columns / 2);
    var heightInFullBlocks = Math.floor(rows / 2);
    {
        var dstStride = (oddWidth ? 4 : 0);
        var oneRow = columns;
        var dst = 0;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            var matrixSrcRow = (blockY * 2 * columns);
            for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
                var matrixSrcCol = blockX * 2;
                var src = matrixSrcRow + matrixSrcCol;
                packedRGBA[dst] = matrix[src];
                packedRGBA[dst + 1] = matrix[src + 1];
                packedRGBA[dst + 2] = matrix[src + oneRow];
                packedRGBA[dst + 3] = matrix[src + oneRow + 1];
                dst += 4;
            }
            dst += dstStride;
        }
    }
    if (oddWidth) {
        var src = columns - 1;
        var dst = (textureWidth - 1) * 4;
        var srcStride = 2 * columns;
        var dstStride = textureWidth * 4;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            packedRGBA[dst] = matrix[src];
            packedRGBA[dst + 2] = matrix[src + columns];
            src += srcStride;
            dst += dstStride;
        }
    }
    if (oddHeight) {
        var src = (rows - 1) * columns;
        var dst = (textureHeight - 1) * textureWidth * 4;
        for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
            packedRGBA[dst++] = matrix[src++];
            packedRGBA[dst++] = matrix[src++];
            dst += 2;
        }
    }
    if (oddWidth && oddHeight) {
        packedRGBA[packedRGBA.length - 4] = matrix[matrix.length - 1];
    }
    return packedRGBA;
}
exports.encodeMatrixToPackedRGBA = encodeMatrixToPackedRGBA;
function decodeMatrixFromPackedRGBA(packedRGBA, rows, columns, matrix) {
    var requiredSize = rows * columns;
    if (requiredSize < matrix.length) {
        throw new Error('matrix length (' + matrix.length + ') must be >= ' + requiredSize);
    }
    var oddWidth = (columns % 2) === 1;
    var oddHeight = (rows % 2) === 1;
    var widthInFullBlocks = Math.floor(columns / 2);
    var heightInFullBlocks = Math.floor(rows / 2);
    var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), textureWidth = _a[0], textureHeight = _a[1];
    {
        var srcStride = oddWidth ? 4 : 0;
        var dstStride = columns + (oddWidth ? 1 : 0);
        var src = 0;
        var dstRow1 = 0;
        var dstRow2 = columns;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
                matrix[dstRow1++] = packedRGBA[src++];
                matrix[dstRow1++] = packedRGBA[src++];
                matrix[dstRow2++] = packedRGBA[src++];
                matrix[dstRow2++] = packedRGBA[src++];
            }
            src += srcStride;
            dstRow1 += dstStride;
            dstRow2 += dstStride;
        }
    }
    if (oddWidth) {
        var src = (textureWidth - 1) * 4;
        var dst = columns - 1;
        var srcStride = textureWidth * 4;
        var dstStride = 2 * columns;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            matrix[dst] = packedRGBA[src];
            matrix[dst + columns] = packedRGBA[src + 2];
            src += srcStride;
            dst += dstStride;
        }
    }
    if (oddHeight) {
        var src = (textureHeight - 1) * textureWidth * 4;
        var dst = (rows - 1) * columns;
        for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
            matrix[dst++] = packedRGBA[src++];
            matrix[dst++] = packedRGBA[src++];
            src += 2;
        }
    }
    if (oddWidth && oddHeight) {
        matrix[matrix.length - 1] = packedRGBA[packedRGBA.length - 4];
    }
    return matrix;
}
exports.decodeMatrixFromPackedRGBA = decodeMatrixFromPackedRGBA;

},{}],31:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var USE_WEBGL2_WHEN_AVAILABLE = true;
var WEBGL2_ENABLED = null;
var MAX_TEXTURE_SIZE = null;
var util = require("../../util");
exports.IS_NAN_SHADER_FUNC = "\nbool isNaN(float val) {\n  return val == val ? false : true;\n}\n";
function createWebGLRenderingContext(attributes) {
    var canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    return createWebGLRenderingContextFromCanvas(canvas, attributes);
}
exports.createWebGLRenderingContext = createWebGLRenderingContext;
function preferWebGL1() {
    USE_WEBGL2_WHEN_AVAILABLE = false;
    WEBGL2_ENABLED = null;
}
exports.preferWebGL1 = preferWebGL1;
function preferWebGL2() {
    USE_WEBGL2_WHEN_AVAILABLE = true;
    WEBGL2_ENABLED = null;
}
exports.preferWebGL2 = preferWebGL2;
function isWebGL2Enabled() {
    if (!USE_WEBGL2_WHEN_AVAILABLE) {
        return false;
    }
    if (WEBGL2_ENABLED == null) {
        var tempCanvas = document.createElement('canvas');
        var gl = tempCanvas.getContext('webgl2');
        if (gl != null) {
            WEBGL2_ENABLED = true;
            var loseContextExtension = getExtensionOrThrow(gl, 'WEBGL_lose_context');
            loseContextExtension.loseContext();
        }
        else {
            WEBGL2_ENABLED = false;
        }
    }
    return WEBGL2_ENABLED;
}
exports.isWebGL2Enabled = isWebGL2Enabled;
function createWebGLRenderingContextFromCanvas(canvas, attributes) {
    var gl;
    if (isWebGL2Enabled()) {
        gl = canvas.getContext('webgl2', attributes);
    }
    else {
        gl = (canvas.getContext('webgl', attributes) ||
            canvas.getContext('experimental-webgl', attributes));
    }
    if (gl == null) {
        throw new Error('This browser does not support WebGL.');
    }
    return gl;
}
exports.createWebGLRenderingContextFromCanvas = createWebGLRenderingContextFromCanvas;
function callAndCheck(gl, func) {
    var returnValue = func();
    checkWebGLError(gl);
    return returnValue;
}
exports.callAndCheck = callAndCheck;
var webGLDebugErrorCheckingEnabled = false;
function enableDebugWebGLErrorChecking(enabled) {
    webGLDebugErrorCheckingEnabled = enabled;
}
exports.enableDebugWebGLErrorChecking = enableDebugWebGLErrorChecking;
function checkWebGLError(gl) {
    if (webGLDebugErrorCheckingEnabled) {
        var error = gl.getError();
        if (error !== gl.NO_ERROR) {
            throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
        }
    }
}
exports.checkWebGLError = checkWebGLError;
function getWebGLErrorMessage(gl, status) {
    switch (status) {
        case gl.NO_ERROR:
            return 'NO_ERROR';
        case gl.INVALID_ENUM:
            return 'INVALID_ENUM';
        case gl.INVALID_VALUE:
            return 'INVALID_VALUE';
        case gl.INVALID_OPERATION:
            return 'INVALID_OPERATION';
        case gl.INVALID_FRAMEBUFFER_OPERATION:
            return 'INVALID_FRAMEBUFFER_OPERATION';
        case gl.OUT_OF_MEMORY:
            return 'OUT_OF_MEMORY';
        case gl.CONTEXT_LOST_WEBGL:
            return 'CONTEXT_LOST_WEBGL';
        default:
            return 'Unknown error code ' + status;
    }
}
exports.getWebGLErrorMessage = getWebGLErrorMessage;
function getExtensionOrThrow(gl, extensionName) {
    return throwIfNull(gl, function () { return gl.getExtension(extensionName); }, 'Extension "' + extensionName + '" not supported on this browser.');
}
exports.getExtensionOrThrow = getExtensionOrThrow;
function createVertexShader(gl, vertexShaderSource) {
    var vertexShader = throwIfNull(gl, function () { return gl.createShader(gl.VERTEX_SHADER); }, 'Unable to create vertex WebGLShader.');
    callAndCheck(gl, function () { return gl.shaderSource(vertexShader, vertexShaderSource); });
    callAndCheck(gl, function () { return gl.compileShader(vertexShader); });
    if (gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS) === false) {
        console.log(gl.getShaderInfoLog(vertexShader));
        throw new Error('Failed to compile vertex shader.');
    }
    return vertexShader;
}
exports.createVertexShader = createVertexShader;
function createFragmentShader(gl, fragmentShaderSource) {
    var fragmentShader = throwIfNull(gl, function () { return gl.createShader(gl.FRAGMENT_SHADER); }, 'Unable to create fragment WebGLShader.');
    callAndCheck(gl, function () { return gl.shaderSource(fragmentShader, fragmentShaderSource); });
    callAndCheck(gl, function () { return gl.compileShader(fragmentShader); });
    if (gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS) === false) {
        console.log(gl.getShaderInfoLog(fragmentShader));
        throw new Error('Failed to compile fragment shader.');
    }
    return fragmentShader;
}
exports.createFragmentShader = createFragmentShader;
function createProgram(gl) {
    return throwIfNull(gl, function () { return gl.createProgram(); }, 'Unable to create WebGLProgram.');
}
exports.createProgram = createProgram;
function linkProgram(gl, program) {
    callAndCheck(gl, function () { return gl.linkProgram(program); });
    if (gl.getProgramParameter(program, gl.LINK_STATUS) === false) {
        console.log(gl.getProgramInfoLog(program));
        throw new Error('Failed to link vertex and fragment shaders.');
    }
}
exports.linkProgram = linkProgram;
function validateProgram(gl, program) {
    callAndCheck(gl, function () { return gl.validateProgram(program); });
    if (gl.getProgramParameter(program, gl.VALIDATE_STATUS) === false) {
        console.log(gl.getProgramInfoLog(program));
        throw new Error('Shader program validation failed.');
    }
}
exports.validateProgram = validateProgram;
function createStaticVertexBuffer(gl, data) {
    var buffer = throwIfNull(gl, function () { return gl.createBuffer(); }, 'Unable to create WebGLBuffer');
    callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, buffer); });
    callAndCheck(gl, function () { return gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); });
    return buffer;
}
exports.createStaticVertexBuffer = createStaticVertexBuffer;
function createStaticIndexBuffer(gl, data) {
    var buffer = throwIfNull(gl, function () { return gl.createBuffer(); }, 'Unable to create WebGLBuffer');
    callAndCheck(gl, function () { return gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer); });
    callAndCheck(gl, function () { return gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW); });
    return buffer;
}
exports.createStaticIndexBuffer = createStaticIndexBuffer;
function queryMaxTextureSize(gl) {
    if (MAX_TEXTURE_SIZE != null) {
        return MAX_TEXTURE_SIZE;
    }
    MAX_TEXTURE_SIZE =
        callAndCheck(gl, function () { return gl.getParameter(gl.MAX_TEXTURE_SIZE); });
    return MAX_TEXTURE_SIZE;
}
exports.queryMaxTextureSize = queryMaxTextureSize;
function getChannelsPerTexture() {
    if (isWebGL2Enabled()) {
        return 1;
    }
    return 4;
}
exports.getChannelsPerTexture = getChannelsPerTexture;
function createTexture(gl) {
    return throwIfNull(gl, function () { return gl.createTexture(); }, 'Unable to create WebGLTexture.');
}
exports.createTexture = createTexture;
function validateTextureSize(gl, width, height) {
    var maxTextureSize = queryMaxTextureSize(gl);
    if ((width <= 0) || (height <= 0)) {
        var requested = '[' + width + 'x' + height + ']';
        throw new Error('Requested texture size ' + requested + ' is invalid.');
    }
    if ((width > maxTextureSize) || (height > maxTextureSize)) {
        var requested = '[' + width + 'x' + height + ']';
        var max = '[' + maxTextureSize + 'x' + maxTextureSize + ']';
        throw new Error('Requested texture size ' + requested +
            ' greater than WebGL maximum on this browser / GPU ' + max + '.');
    }
}
exports.validateTextureSize = validateTextureSize;
function createFramebuffer(gl) {
    return throwIfNull(gl, function () { return gl.createFramebuffer(); }, 'Unable to create WebGLFramebuffer.');
}
exports.createFramebuffer = createFramebuffer;
function bindVertexBufferToProgramAttribute(gl, program, attribute, buffer, arrayEntriesPerItem, itemStrideInBytes, itemOffsetInBytes) {
    var loc = gl.getAttribLocation(program, attribute);
    if (loc === -1) {
        var error = new Error('Unable to get attribute "' + attribute + '" on WebGLProgram.');
        error.namedVertexAttributeNotFound = attribute;
        throw error;
    }
    callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, buffer); });
    callAndCheck(gl, function () { return gl.vertexAttribPointer(loc, arrayEntriesPerItem, gl.FLOAT, false, itemStrideInBytes, itemOffsetInBytes); });
    callAndCheck(gl, function () { return gl.enableVertexAttribArray(loc); });
}
exports.bindVertexBufferToProgramAttribute = bindVertexBufferToProgramAttribute;
function bindTextureUnit(gl, texture, textureUnit) {
    validateTextureUnit(gl, textureUnit);
    callAndCheck(gl, function () { return gl.activeTexture(gl.TEXTURE0 + textureUnit); });
    callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
}
exports.bindTextureUnit = bindTextureUnit;
function unbindTextureUnit(gl, textureUnit) {
    validateTextureUnit(gl, textureUnit);
    callAndCheck(gl, function () { return gl.activeTexture(gl.TEXTURE0 + textureUnit); });
    callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
}
exports.unbindTextureUnit = unbindTextureUnit;
function getProgramUniformLocationOrThrow(gl, program, uniformName) {
    return throwIfNull(gl, function () { return gl.getUniformLocation(program, uniformName); }, 'uniform "' + uniformName + '" not present in program.');
}
exports.getProgramUniformLocationOrThrow = getProgramUniformLocationOrThrow;
function bindTextureToProgramUniformSampler(gl, program, texture, uniformSamplerName, textureUnit) {
    callAndCheck(gl, function () { return bindTextureUnit(gl, texture, textureUnit); });
    var samplerLocation = getProgramUniformLocationOrThrow(gl, program, uniformSamplerName);
    callAndCheck(gl, function () { return gl.uniform1i(samplerLocation, textureUnit); });
}
exports.bindTextureToProgramUniformSampler = bindTextureToProgramUniformSampler;
function bindCanvasToFramebuffer(gl) {
    callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, null); });
    callAndCheck(gl, function () { return gl.viewport(0, 0, gl.canvas.width, gl.canvas.height); });
    callAndCheck(gl, function () { return gl.scissor(0, 0, gl.canvas.width, gl.canvas.height); });
}
exports.bindCanvasToFramebuffer = bindCanvasToFramebuffer;
function bindColorTextureToFramebuffer(gl, texture, framebuffer) {
    callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer); });
    callAndCheck(gl, function () { return gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0); });
}
exports.bindColorTextureToFramebuffer = bindColorTextureToFramebuffer;
function unbindColorTextureFromFramebuffer(gl, framebuffer) {
    callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer); });
    callAndCheck(gl, function () { return gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0); });
}
exports.unbindColorTextureFromFramebuffer = unbindColorTextureFromFramebuffer;
function validateFramebuffer(gl) {
    var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Error binding framebuffer: ' + getFramebufferErrorMessage(gl, status));
    }
}
exports.validateFramebuffer = validateFramebuffer;
function getFramebufferErrorMessage(gl, status) {
    switch (status) {
        case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            return 'FRAMEBUFFER_INCOMPLETE_ATTACHMENT';
        case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            return 'FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT';
        case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
            return 'FRAMEBUFFER_INCOMPLETE_DIMENSIONS';
        case gl.FRAMEBUFFER_UNSUPPORTED:
            return 'FRAMEBUFFER_UNSUPPORTED';
        default:
            return 'unknown error ' + status;
    }
}
exports.getFramebufferErrorMessage = getFramebufferErrorMessage;
function throwIfNull(gl, returnTOrNull, failureMessage) {
    var tOrNull = callAndCheck(gl, function () { return returnTOrNull(); });
    if (tOrNull == null) {
        throw new Error(failureMessage);
    }
    return tOrNull;
}
function validateTextureUnit(gl, textureUnit) {
    var maxTextureUnit = gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS - 1;
    var glTextureUnit = textureUnit + gl.TEXTURE0;
    if (glTextureUnit < gl.TEXTURE0 || glTextureUnit > maxTextureUnit) {
        var textureUnitRange = '[gl.TEXTURE0, gl.TEXTURE' + maxTextureUnit + ']';
        throw new Error('textureUnit must be in ' + textureUnitRange + '.');
    }
}
function getTextureShapeFromLogicalShape(gl, logicalShape, preferredTexShape) {
    var maxTexSize = queryMaxTextureSize(gl);
    var size = util.sizeFromShape(logicalShape);
    if (preferredTexShape != null) {
        var sizePreferred = util.sizeFromShape(preferredTexShape);
        util.assert(size === sizePreferred, "Size of shape (" + size + ") must match size of " +
            ("preferredShape (" + sizePreferred + ")"));
        if (preferredTexShape[0] <= maxTexSize &&
            preferredTexShape[1] <= maxTexSize) {
            return preferredTexShape;
        }
    }
    if (logicalShape.length <= 1 && size <= maxTexSize) {
        return [size, 1];
    }
    else if (logicalShape.length === 2 && logicalShape[0] <= maxTexSize &&
        logicalShape[1] <= maxTexSize) {
        return logicalShape;
    }
    else if (logicalShape.length === 3 && logicalShape[0] <= maxTexSize &&
        logicalShape[1] * logicalShape[2] <= maxTexSize) {
        return [logicalShape[0], logicalShape[1] * logicalShape[2]];
    }
    else {
        return util.sizeToSquarishShape(size);
    }
}
exports.getTextureShapeFromLogicalShape = getTextureShapeFromLogicalShape;

},{"../../util":33}],32:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function expectArraysClose(actual, expected, epsilon) {
    if (actual.length !== expected.length) {
        throw new Error('Matrices have different lengths (' + actual.length + ' vs ' +
            expected.length + ').');
    }
    for (var i = 0; i < expected.length; ++i) {
        var a = actual[i];
        var e = expected[i];
        if (isNaN(a) && isNaN(e)) {
            continue;
        }
        if (isNaN(a) || isNaN(e) || Math.abs(a - e) > epsilon) {
            var actualStr = 'actual[' + i + '] === ' + a;
            var expectedStr = 'expected[' + i + '] === ' + e;
            throw new Error('Arrays differ: ' + actualStr + ', ' + expectedStr);
        }
    }
}
exports.expectArraysClose = expectArraysClose;
function randomArrayInRange(n, minValue, maxValue) {
    var v = new Float32Array(n);
    var range = maxValue - minValue;
    for (var i = 0; i < n; ++i) {
        v[i] = (Math.random() * range) + minValue;
    }
    return v;
}
exports.randomArrayInRange = randomArrayInRange;
function makeIdentity(n) {
    var i = new Float32Array(n * n);
    for (var j = 0; j < n; ++j) {
        i[(j * n) + j] = 1;
    }
    return i;
}
exports.makeIdentity = makeIdentity;
function setValue(m, mNumRows, mNumCols, v, row, column) {
    if (row >= mNumRows) {
        throw new Error('row (' + row + ') must be in [0 ' + mNumRows + '].');
    }
    if (column >= mNumCols) {
        throw new Error('column (' + column + ') must be in [0 ' + mNumCols + '].');
    }
    m[(row * mNumCols) + column] = v;
}
exports.setValue = setValue;
function cpuMultiplyMatrix(a, aRow, aCol, b, bRow, bCol) {
    var result = new Float32Array(aRow * bCol);
    for (var r = 0; r < aRow; ++r) {
        for (var c = 0; c < bCol; ++c) {
            var d = 0;
            for (var k = 0; k < aCol; ++k) {
                d += a[(r * aCol) + k] * b[(k * bCol) + c];
            }
            result[(r * bCol) + c] = d;
        }
    }
    return result;
}
exports.cpuMultiplyMatrix = cpuMultiplyMatrix;
function cpuDotProduct(a, b) {
    if (a.length !== b.length) {
        throw new Error('cpuDotProduct: incompatible vectors.');
    }
    var d = 0;
    for (var i = 0; i < a.length; ++i) {
        d += a[i] * b[i];
    }
    return d;
}
exports.cpuDotProduct = cpuDotProduct;

},{}],33:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function shuffle(array) {
    var counter = array.length;
    var temp = 0;
    var index = 0;
    while (counter > 0) {
        index = (Math.random() * counter) | 0;
        counter--;
        temp = array[counter];
        array[counter] = array[index];
        array[index] = temp;
    }
}
exports.shuffle = shuffle;
function clamp(min, x, max) {
    return Math.max(min, Math.min(x, max));
}
exports.clamp = clamp;
function randUniform(a, b) {
    return Math.random() * (b - a) + a;
}
exports.randUniform = randUniform;
function randGauss(mean, stdDev, truncated) {
    if (mean === void 0) { mean = 0; }
    if (stdDev === void 0) { stdDev = 1; }
    if (truncated === void 0) { truncated = false; }
    var v1, v2, s;
    do {
        v1 = 2 * Math.random() - 1;
        v2 = 2 * Math.random() - 1;
        s = v1 * v1 + v2 * v2;
    } while (s > 1);
    var result = Math.sqrt(-2 * Math.log(s) / s) * v1;
    if (truncated && result > 2) {
        return randGauss(mean, stdDev, true);
    }
    return mean + stdDev * result;
}
exports.randGauss = randGauss;
function distSquared(a, b) {
    var result = 0;
    for (var i = 0; i < a.length; i++) {
        var diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
}
exports.distSquared = distSquared;
function assert(expr, msg) {
    if (!expr) {
        throw new Error(msg);
    }
}
exports.assert = assert;
function assertShapesMatch(shapeA, shapeB, errorMessagePrefix) {
    if (errorMessagePrefix === void 0) { errorMessagePrefix = ''; }
    assert(arraysEqual(shapeA, shapeB), errorMessagePrefix + ("Shapes " + shapeA + " and " + shapeB + " must match"));
}
exports.assertShapesMatch = assertShapesMatch;
function flatten(arr, ret) {
    ret = (ret === undefined ? [] : ret);
    for (var i = 0; i < arr.length; ++i) {
        if (Array.isArray(arr[i])) {
            flatten(arr[i], ret);
        }
        else {
            ret.push(arr[i]);
        }
    }
    return ret;
}
exports.flatten = flatten;
function inferShape(arr) {
    var shape = [];
    while (arr instanceof Array) {
        shape.push(arr.length);
        arr = arr[0];
    }
    return shape;
}
exports.inferShape = inferShape;
function sizeFromShape(shape) {
    if (shape.length === 0) {
        return 1;
    }
    var size = shape[0];
    for (var i = 1; i < shape.length; i++) {
        size *= shape[i];
    }
    return size;
}
exports.sizeFromShape = sizeFromShape;
function isScalarShape(shape) {
    return shape.length === 0;
}
exports.isScalarShape = isScalarShape;
function arraysEqual(n1, n2) {
    if (n1.length !== n2.length) {
        return false;
    }
    for (var i = 0; i < n1.length; i++) {
        if (n1[i] !== n2[i]) {
            return false;
        }
    }
    return true;
}
exports.arraysEqual = arraysEqual;
function isInt(a) {
    return a % 1 === 0;
}
exports.isInt = isInt;
function tanh(x) {
    if (Math.tanh != null) {
        return Math.tanh(x);
    }
    if (x === Infinity) {
        return 1;
    }
    else if (x === -Infinity) {
        return -1;
    }
    else {
        var e2x = Math.exp(2 * x);
        return (e2x - 1) / (e2x + 1);
    }
}
exports.tanh = tanh;
function sizeToSquarishShape(size) {
    for (var a = Math.floor(Math.sqrt(size)); a > 1; --a) {
        if (size % a === 0) {
            return [a, size / a];
        }
    }
    return [1, size];
}
exports.sizeToSquarishShape = sizeToSquarishShape;
function createShuffledIndices(n) {
    var shuffledIndices = new Uint32Array(n);
    for (var i = 0; i < n; ++i) {
        shuffledIndices[i] = i;
    }
    shuffle(shuffledIndices);
    return shuffledIndices;
}
exports.createShuffledIndices = createShuffledIndices;

},{}]},{},[7])
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJkZW1vcy9iZW5jaG1hcmtzL2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvY29udl9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9jb252X3RyYW5zcG9zZV9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9sb2dzdW1leHBfY3B1X2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvbG9nc3VtZXhwX2dwdV9iZW5jaG1hcmsudHMiLCJkZW1vcy9iZW5jaG1hcmtzL21hdGgtYmVuY2htYXJrLXJ1bi1ncm91cHMudHMiLCJkZW1vcy9iZW5jaG1hcmtzL21hdGgtYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9tYXhfcG9vbF9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9tdWxtYXRfY3B1X2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvbXVsbWF0X2dwdV9iZW5jaG1hcmsudHMiLCJkZW1vcy9kZW1vLWZvb3Rlci50cyIsImRlbW9zL2RlbW8taGVhZGVyLnRzIiwiZGVtb3MvcG9seW1lci1zcGVjLnRzIiwic3JjL21hdGgvY29uY2F0M2RfdXRpbC50cyIsInNyYy9tYXRoL2NvbnZfdXRpbC50cyIsInNyYy9tYXRoL2NvcHkyZF91dGlsLnRzIiwic3JjL21hdGgvbWF0aC50cyIsInNyYy9tYXRoL21hdGhfY3B1LnRzIiwic3JjL21hdGgvbmRhcnJheS50cyIsInNyYy9tYXRoL3dlYmdsL2NvbnZfYmFja3Byb3BfZ3B1LnRzIiwic3JjL21hdGgvd2ViZ2wvY29udl9ncHUudHMiLCJzcmMvbWF0aC93ZWJnbC9ncGdwdV9jb250ZXh0LnRzIiwic3JjL21hdGgvd2ViZ2wvZ3BncHVfdXRpbC50cyIsInNyYy9tYXRoL3dlYmdsL2xvZ3N1bWV4cF9ncHUudHMiLCJzcmMvbWF0aC93ZWJnbC9tYXhfcG9vbF9ncHUudHMiLCJzcmMvbWF0aC93ZWJnbC9tdWxtYXRfZ3B1LnRzIiwic3JjL21hdGgvd2ViZ2wvbXVsbWF0X3BhY2tlZF9ncHUudHMiLCJzcmMvbWF0aC93ZWJnbC9wb29sX2dwdS50cyIsInNyYy9tYXRoL3dlYmdsL3NoYWRlcl9jb21waWxlci50cyIsInNyYy9tYXRoL3dlYmdsL3RleF91dGlsLnRzIiwic3JjL21hdGgvd2ViZ2wvd2ViZ2xfdXRpbC50cyIsInNyYy90ZXN0X3V0aWwudHMiLCJzcmMvdXRpbC50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7O0FDMkJBO0lBS0Usc0JBQVksSUFBWSxFQUFFLGFBQTRCO1FBQ3BELElBQUksQ0FBQyxJQUFJLEdBQUcsSUFBSSxDQUFDO1FBQ2pCLElBQUksQ0FBQyxhQUFhLEdBQUcsYUFBYSxDQUFDO1FBQ25DLElBQUksQ0FBQyxTQUFTLEdBQUcsRUFBRSxDQUFDO0lBQ3RCLENBQUM7SUFDSCxtQkFBQztBQUFELENBVkEsQUFVQyxJQUFBO0FBVlksb0NBQVk7Ozs7O0FDWnpCLG9EQUFzRDtBQUN0RCx3REFBMEQ7QUFDMUQsb0VBQWdFO0FBQ2hFLCtDQUFpRDtBQUlqRCxJQUFNLE9BQU8sR0FBRyxFQUFFLENBQUM7QUFFTixRQUFBLGNBQWMsR0FBa0IsVUFBQyxJQUFZO0lBQ3hELElBQU0sYUFBYSxHQUE2QixDQUFDLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDaEUsSUFBTSxXQUFXLEdBQUcsQ0FBQyxDQUFDO0lBQ3RCLElBQU0sU0FBUyxHQUFHLEVBQUUsQ0FBQztJQUNyQixJQUFNLE1BQU0sR0FBRyxDQUFDLENBQUM7SUFDakIsSUFBTSxPQUFPLEdBQUcsU0FBUyxDQUFDLGlCQUFpQixDQUFDLGFBQWEsRUFBRSxTQUFTLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFDOUUsSUFBTSxjQUFjLEdBQ2hCLFNBQVMsQ0FBQyxvQkFBb0IsQ0FDMUIsYUFBYSxFQUFFLFNBQVMsRUFBRSxXQUFXLEVBQUUsTUFBTSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBRWhFLElBQU0sZUFBZSxHQUFHLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQyxhQUFhLENBQUMsQ0FBQztJQUN2RSxJQUFNLGdCQUFnQixHQUFHLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQyxjQUFjLENBQUMsQ0FBQztJQUN6RSxJQUFNLGlCQUFpQixHQUFHLFNBQVMsQ0FBQyxzQkFBc0IsQ0FDdEQsYUFBYSxDQUFDLENBQUMsQ0FBQyxFQUFFLFdBQVcsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUM5QyxJQUFNLGdCQUFnQixHQUFHLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQyxXQUFXLENBQUMsQ0FBQztJQUV0RSxJQUFNLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDckIsSUFBTSxLQUFLLEdBQUcsSUFBSSw0QkFBWSxFQUFFLENBQUM7SUFDakMsSUFBTSxPQUFPLEdBQUcsS0FBSyxDQUFDLGFBQWEsQ0FBQyxRQUFRLENBQUMsdUJBQXVCLENBQ2hFLGFBQWEsRUFBRSxXQUFXLEVBQUUsU0FBUyxFQUFFLE1BQU0sRUFBRSxPQUFPLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUV0RSxJQUFNLFlBQVksR0FDZCxLQUFLLENBQUMsbUJBQW1CLENBQUMsZUFBZSxDQUFDLENBQUMsQ0FBQyxFQUFFLGVBQWUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3RFLElBQU0sY0FBYyxHQUNoQixLQUFLLENBQUMsbUJBQW1CLENBQUMsaUJBQWlCLENBQUMsQ0FBQyxDQUFDLEVBQUUsaUJBQWlCLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUMxRSxJQUFNLGFBQWEsR0FDZixLQUFLLENBQUMsbUJBQW1CLENBQUMsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN4RSxJQUFNLGFBQWEsR0FDZixLQUFLLENBQUMsbUJBQW1CLENBQUMsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUV4RSxJQUFNLFNBQVMsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQzFDLGVBQWUsQ0FBQyxDQUFDLENBQUMsR0FBRyxlQUFlLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDcEQsSUFBTSxXQUFXLEdBQUcsU0FBUyxDQUFDLGtCQUFrQixDQUM1QyxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsR0FBRyxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUN4RCxJQUFNLFVBQVUsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQzNDLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxHQUFHLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBRXRELEtBQUssQ0FBQyxxQkFBcUIsQ0FDdkIsWUFBWSxFQUFFLGVBQWUsQ0FBQyxDQUFDLENBQUMsRUFBRSxlQUFlLENBQUMsQ0FBQyxDQUFDLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDckUsS0FBSyxDQUFDLHFCQUFxQixDQUN2QixjQUFjLEVBQUUsaUJBQWlCLENBQUMsQ0FBQyxDQUFDLEVBQUUsaUJBQWlCLENBQUMsQ0FBQyxDQUFDLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFDN0UsS0FBSyxDQUFDLHFCQUFxQixDQUN2QixhQUFhLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEVBQUUsVUFBVSxDQUFDLENBQUM7SUFFekUsSUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7UUFDakMsUUFBUSxDQUFDLFFBQVEsQ0FDYixLQUFLLEVBQUUsT0FBTyxFQUFFLFlBQVksRUFBRSxjQUFjLEVBQUUsYUFBYSxFQUMzRCxhQUFhLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQztJQUN2QyxDQUFDO0lBRUQsS0FBSyxDQUFDLHlCQUF5QixDQUMzQixhQUFhLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM3RCxJQUFNLEdBQUcsR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFFOUIsSUFBTSxPQUFPLEdBQUcsQ0FBQyxHQUFHLEdBQUcsS0FBSyxDQUFDLEdBQUcsT0FBTyxDQUFDO0lBRXhDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxZQUFZLENBQUMsQ0FBQztJQUN4QyxLQUFLLENBQUMsbUJBQW1CLENBQUMsY0FBYyxDQUFDLENBQUM7SUFDMUMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLGFBQWEsQ0FBQyxDQUFDO0lBQ3pDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxhQUFhLENBQUMsQ0FBQztJQUN6QyxLQUFLLENBQUMsYUFBYSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBQzdCLEtBQUssQ0FBQyxPQUFPLEVBQUUsQ0FBQztJQUVoQixNQUFNLENBQUMsT0FBTyxDQUFDO0FBQ2pCLENBQUMsQ0FBQzs7Ozs7QUMxRUYsb0RBQXNEO0FBQ3RELDBFQUE0RTtBQUM1RSxvRUFBZ0U7QUFDaEUsK0NBQWlEO0FBSWpELElBQU0sT0FBTyxHQUFHLEdBQUcsQ0FBQztBQUVQLFFBQUEsY0FBYyxHQUFrQixVQUFDLElBQVk7SUFDeEQsSUFBTSxTQUFTLEdBQTZCLENBQUMsSUFBSSxFQUFFLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQztJQUM1RCxJQUFNLGVBQWUsR0FBRyxDQUFDLENBQUM7SUFDMUIsSUFBTSxTQUFTLEdBQUcsRUFBRSxDQUFDO0lBQ3JCLElBQU0sVUFBVSxHQUFHLENBQUMsQ0FBQztJQUNyQixJQUFNLE9BQU8sR0FBRyxDQUFDLENBQUM7SUFFbEIsSUFBTSxLQUFLLEdBQUcsSUFBSSw0QkFBWSxFQUFFLENBQUM7SUFDakMsS0FBSyxDQUFDLDhCQUE4QixDQUFDLElBQUksQ0FBQyxDQUFDO0lBQzNDLElBQU0sY0FBYyxHQUFHLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNwQyxJQUFNLEdBQUcsR0FBRyxpQkFBaUIsQ0FBQyxvQ0FBb0MsQ0FDOUQsU0FBUyxFQUFFLFNBQVMsRUFBRSxjQUFjLEVBQUUsVUFBVSxFQUFFLE9BQU8sRUFBRSxLQUFLLENBQUMsQ0FBQztJQUN0RSxJQUFNLE9BQU8sR0FBRyxLQUFLLENBQUMsYUFBYSxDQUFDLEdBQUcsQ0FBQyxDQUFDO0lBR3pDLElBQU0sV0FBVyxHQUFHLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQyxTQUFTLENBQUMsQ0FBQztJQUMvRCxJQUFNLElBQUksR0FBRyxLQUFLLENBQUMsbUJBQW1CLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3ZFLElBQU0sS0FBSyxHQUNQLFNBQVMsQ0FBQyxrQkFBa0IsQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLEdBQUcsV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ3pFLEtBQUssQ0FBQyxxQkFBcUIsQ0FBQyxJQUFJLEVBQUUsV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUMsRUFBRSxLQUFLLENBQUMsQ0FBQztJQUd6RSxJQUFNLFdBQVcsR0FBRyxTQUFTLENBQUMsc0JBQXNCLENBQ2hELGNBQWMsRUFBRSxlQUFlLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFDaEQsSUFBTSxLQUFLLEdBQ1AsU0FBUyxDQUFDLGtCQUFrQixDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsR0FBRyxXQUFXLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDekUsSUFBTSxJQUFJLEdBQUcsS0FBSyxDQUFDLG1CQUFtQixDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsRUFBRSxXQUFXLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN2RSxLQUFLLENBQUMscUJBQXFCLENBQUMsSUFBSSxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUMsRUFBRSxXQUFXLENBQUMsQ0FBQyxDQUFDLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFHekUsSUFBTSxTQUFTLEdBQ1gsU0FBUyxDQUFDLGdCQUFnQixDQUFDLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQyxFQUFFLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLFVBQVUsQ0FBQyxDQUFDO0lBQ3pFLElBQU0sR0FBRyxHQUFHLFNBQVMsR0FBRyxDQUFDLEdBQUcsT0FBTyxDQUFDO0lBQ3BDLElBQU0sY0FBYyxHQUFHLFNBQVMsQ0FBQyxvQkFBb0IsQ0FDakQsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLEVBQUUsU0FBUyxDQUFDLENBQUMsQ0FBQyxFQUFFLGVBQWUsQ0FBQyxFQUFFLFNBQVMsRUFBRSxjQUFjLEVBQ3hFLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQztJQUVaLElBQU0sV0FBVyxHQUFHLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQyxjQUFjLENBQUMsQ0FBQztJQUNwRSxJQUFNLFNBQVMsR0FBRyxLQUFLLENBQUMsbUJBQW1CLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBRTVFLElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQ2pDLGlCQUFpQixDQUFDLGFBQWEsQ0FDM0IsS0FBSyxFQUFFLE9BQU8sRUFBRSxJQUFJLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxTQUFTLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFDaEUsQ0FBQztJQUVELElBQU0sQ0FBQyxHQUFHLEtBQUssQ0FBQyx5QkFBeUIsQ0FDckMsU0FBUyxFQUFFLFdBQVcsQ0FBQyxDQUFDLENBQUMsRUFBRSxXQUFXLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUUvQyxJQUFNLEdBQUcsR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFFOUIsSUFBTSxPQUFPLEdBQUcsQ0FBQyxHQUFHLEdBQUcsS0FBSyxDQUFDLEdBQUcsT0FBTyxDQUFDO0lBRXhDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxTQUFTLENBQUMsQ0FBQztJQUNyQyxLQUFLLENBQUMsbUJBQW1CLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDaEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLElBQUksQ0FBQyxDQUFDO0lBQ2hDLEtBQUssQ0FBQyxhQUFhLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDN0IsS0FBSyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBRWhCLE1BQU0sQ0FBQyxPQUFPLENBQUM7QUFDakIsQ0FBQyxDQUFDOzs7OztBQ3JFRixvREFBdUQ7QUFDdkQsa0RBQXdEO0FBSXhELElBQU0sV0FBVyxHQUFHLEVBQUUsQ0FBQztBQUVWLFFBQUEsY0FBYyxHQUFrQixVQUFDLElBQVk7SUFDeEQsSUFBTSxJQUFJLEdBQUcsSUFBSSx5QkFBYyxFQUFFLENBQUM7SUFDbEMsSUFBTSxDQUFDLEdBQUcsaUJBQU8sQ0FBQyxXQUFXLENBQVUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDNUQsSUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsV0FBVyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7UUFDckMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNwQixDQUFDO0lBQ0QsSUFBTSxHQUFHLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQzlCLE1BQU0sQ0FBQyxDQUFDLEdBQUcsR0FBRyxLQUFLLENBQUMsR0FBRyxXQUFXLENBQUM7QUFDckMsQ0FBQyxDQUFDOzs7OztBQ2hCRixvRUFBZ0U7QUFDaEUsa0VBQW9FO0FBQ3BFLCtDQUFpRDtBQUlqRCxJQUFNLE9BQU8sR0FBRyxHQUFHLENBQUM7QUFFUCxRQUFBLGNBQWMsR0FBa0IsVUFBQyxJQUFZO0lBQ3hELElBQU0sS0FBSyxHQUFHLElBQUksNEJBQVksRUFBRSxDQUFDO0lBRWpDLElBQU0sT0FBTyxHQUNULEtBQUssQ0FBQyxhQUFhLENBQUMsYUFBYSxDQUFDLHVCQUF1QixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDO0lBRTNFLElBQU0sUUFBUSxHQUFHLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDdkQsSUFBTSxhQUFhLEdBQUcsS0FBSyxDQUFDLG1CQUFtQixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUU1RCxJQUFNLENBQUMsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQUMsSUFBSSxHQUFHLElBQUksRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUMzRCxLQUFLLENBQUMscUJBQXFCLENBQUMsUUFBUSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFFckQsSUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7UUFDakMsYUFBYSxDQUFDLFNBQVMsQ0FDbkIsS0FBSyxFQUFFLE9BQU8sRUFBRSxRQUFRLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxhQUFhLENBQUMsQ0FBQztJQUMzRCxDQUFDO0lBRUQsS0FBSyxDQUFDLHlCQUF5QixDQUFDLGFBQWEsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDM0QsSUFBTSxPQUFPLEdBQUcsQ0FBQyxXQUFXLENBQUMsR0FBRyxFQUFFLEdBQUcsS0FBSyxDQUFDLEdBQUcsT0FBTyxDQUFDO0lBRXRELEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxRQUFRLENBQUMsQ0FBQztJQUNwQyxLQUFLLENBQUMsbUJBQW1CLENBQUMsYUFBYSxDQUFDLENBQUM7SUFDekMsS0FBSyxDQUFDLGFBQWEsQ0FBQyxPQUFPLENBQUMsQ0FBQztJQUM3QixLQUFLLENBQUMsT0FBTyxFQUFFLENBQUM7SUFFaEIsTUFBTSxDQUFDLE9BQU8sQ0FBQztBQUNqQixDQUFDLENBQUM7Ozs7O0FDbkNGLHlDQUE0RDtBQUM1RCx5REFBMkQ7QUFDM0QsNkVBQStFO0FBQy9FLG1FQUFxRTtBQUNyRSxtRUFBcUU7QUFFckUsaUVBQW1FO0FBQ25FLDZEQUErRDtBQUMvRCw2REFBK0Q7QUFHbEQsUUFBQSxvQkFBb0IsR0FBd0I7SUFDdkQ7UUFDRSxJQUFJLEVBQ0Esd0VBQXdFO1FBQzVFLEdBQUcsRUFBRSxDQUFDO1FBQ04sR0FBRyxFQUFFLElBQUk7UUFDVCxRQUFRLEVBQUUsRUFBRTtRQUNaLHdCQUF3QixFQUFFLFVBQUMsSUFBWSxJQUFLLE9BQUEsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLEVBQWpCLENBQWlCO1FBQzdELGFBQWEsRUFBRTtZQUNiLElBQUksd0JBQVksQ0FBQyxZQUFZLEVBQUUsb0JBQW9CLENBQUMsY0FBYyxDQUFDO1lBQ25FLElBQUksd0JBQVksQ0FBQyxZQUFZLEVBQUUsb0JBQW9CLENBQUMsY0FBYyxDQUFDO1NBQ3BFO0tBQ0Y7SUFDRDtRQUNFLElBQUksRUFBRSxvREFBb0Q7UUFDMUQsR0FBRyxFQUFFLENBQUM7UUFDTixHQUFHLEVBQUUsSUFBSTtRQUNULFFBQVEsRUFBRSxFQUFFO1FBQ1osd0JBQXdCLEVBQUUsVUFBQyxJQUFZLElBQUssT0FBQSxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsRUFBakIsQ0FBaUI7UUFDN0QsYUFBYSxFQUFFLENBQUMsSUFBSSx3QkFBWSxDQUM1Qix1QkFBdUIsRUFBRSxrQkFBa0IsQ0FBQyxjQUFjLENBQUMsQ0FBQztLQUNqRTtJQUNEO1FBQ0UsSUFBSSxFQUFFLGlFQUFpRTtRQUN2RSxHQUFHLEVBQUUsQ0FBQztRQUNOLEdBQUcsRUFBRSxJQUFJO1FBQ1QsUUFBUSxFQUFFLEVBQUU7UUFDWix3QkFBd0IsRUFBRSxVQUFDLElBQVksSUFBSyxPQUFBLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxFQUFqQixDQUFpQjtRQUM3RCxhQUFhLEVBQUUsQ0FBQyxJQUFJLHdCQUFZLENBQzVCLHVCQUF1QixFQUFFLDRCQUE0QixDQUFDLGNBQWMsQ0FBQyxDQUFDO0tBQzNFO0lBQ0Q7UUFDRSxJQUFJLEVBQUUsZ0JBQWdCO1FBQ3RCLEdBQUcsRUFBRSxDQUFDO1FBQ04sR0FBRyxFQUFFLElBQUk7UUFDVCxRQUFRLEVBQUUsRUFBRTtRQUNaLHdCQUF3QixFQUFFLFVBQUMsSUFBWSxJQUFLLE9BQUEsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLEVBQWpCLENBQWlCO1FBQzdELGFBQWEsRUFBRSxDQUFDLElBQUksd0JBQVksQ0FDNUIsdUJBQXVCLEVBQ3ZCLHNCQUFzQixDQUFDLHVCQUF1QixDQUFDLENBQUM7S0FDckQ7SUFDRDtRQUNFLElBQUksRUFBRSw0Q0FBNEM7UUFDbEQsR0FBRyxFQUFFLENBQUM7UUFDTixHQUFHLEVBQUUsSUFBSTtRQUNULFFBQVEsRUFBRSxFQUFFO1FBQ1osd0JBQXdCLEVBQUUsVUFBQyxJQUFZLElBQUssT0FBQSxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsRUFBakIsQ0FBaUI7UUFDN0QsYUFBYSxFQUFFO1lBQ2IsSUFBSSx3QkFBWSxDQUFDLGVBQWUsRUFBRSx1QkFBdUIsQ0FBQyxjQUFjLENBQUM7WUFDekUsSUFBSSx3QkFBWSxDQUFDLGVBQWUsRUFBRSx1QkFBdUIsQ0FBQyxjQUFjLENBQUM7U0FDMUU7S0FDRjtDQUNGLENBQUM7Ozs7Ozs7Ozs7Ozs7OztBQy9ERiwwQkFBd0I7QUFDeEIsMEJBQXdCO0FBR3hCLGdEQUFtRTtBQUduRSx5RUFBaUU7QUFHdEQsUUFBQSxvQkFBb0IsR0FBRyw2QkFBYyxDQUM1QyxFQUFDLEVBQUUsRUFBRSxnQkFBZ0IsRUFBRSxVQUFVLEVBQUUsRUFBQyxzQkFBc0IsRUFBRSxLQUFLLEVBQUMsRUFBQyxDQUFDLENBQUM7QUFFekU7SUFBbUMsaUNBQW9CO0lBQXZEOztJQW1NQSxDQUFDO0lBOUxDLDZCQUFLLEdBQUw7UUFBQSxpQkF1QkM7UUFyQkMsSUFBTSxzQkFBc0IsR0FBYSxFQUFFLENBQUM7UUFDNUMsSUFBSSxDQUFDLFlBQVksR0FBRyxFQUFFLENBQUM7UUFDdkIsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxnREFBb0IsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUNyRCxzQkFBc0IsQ0FBQyxJQUFJLENBQUMsZ0RBQW9CLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7WUFDMUQsSUFBSSxDQUFDLFlBQVksQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDaEMsQ0FBQztRQUNELElBQUksQ0FBQyxzQkFBc0IsR0FBRyxzQkFBc0IsQ0FBQztRQUdyRCxVQUFVLENBQUM7WUFDVCxJQUFNLFVBQVUsR0FBRyxLQUFJLENBQUMsZ0JBQWdCLENBQUMsV0FBVyxDQUFDLENBQUM7WUFDdEQsSUFBTSxXQUFXLEdBQUcsS0FBSSxDQUFDLGdCQUFnQixDQUFDLFdBQVcsQ0FBQyxDQUFDO29DQUM5QyxDQUFDO2dCQUNSLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxnQkFBZ0IsQ0FBQyxPQUFPLEVBQUU7b0JBQ3RDLEtBQUksQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDNUIsQ0FBQyxDQUFDLENBQUM7Z0JBQ0gsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDLGdCQUFnQixDQUFDLE9BQU8sRUFBRTtvQkFDdkMsS0FBSSxDQUFDLFlBQVksQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUM7Z0JBQzlCLENBQUMsQ0FBQyxDQUFDO1lBQ0wsQ0FBQztZQVBELEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsVUFBVSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUU7d0JBQWpDLENBQUM7YUFPVDtRQUNILENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNSLENBQUM7SUFFTyx5Q0FBaUIsR0FBekIsVUFBMEIsc0JBQThCO1FBQ3RELElBQU0saUJBQWlCLEdBQUcsZ0RBQW9CLENBQUMsc0JBQXNCLENBQUMsQ0FBQztRQUV2RSxJQUFNLE1BQU0sR0FBRyxJQUFJLENBQUMsZ0JBQWdCLENBQUMsV0FBVyxDQUFDLENBQUMsc0JBQXNCLENBQ25ELENBQUM7UUFDdEIsSUFBTSxPQUFPLEdBQUcsTUFBTSxDQUFDLFVBQVUsQ0FBQyxJQUFJLENBQTZCLENBQUM7UUFFcEUsSUFBTSxRQUFRLEdBQW9CLEVBQUUsQ0FBQztRQUNyQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUNoRSxJQUFNLEdBQUcsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLEdBQUcsR0FBRyxDQUFDLEdBQUcsaUJBQWlCLENBQUMsYUFBYSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1lBQ3pFLFFBQVEsQ0FBQyxJQUFJLENBQUM7Z0JBQ1osSUFBSSxFQUFFLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxTQUFTO2dCQUNsRCxJQUFJLEVBQUUsS0FBSztnQkFDWCxLQUFLLEVBQUUsaUJBQWlCLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUk7Z0JBQzlDLFdBQVcsRUFBRSxNQUFNLEdBQUcsR0FBRyxHQUFHLGNBQWM7Z0JBQzFDLGVBQWUsRUFBRSxNQUFNLEdBQUcsR0FBRyxHQUFHLGNBQWM7Z0JBQzlDLFdBQVcsRUFBRSxDQUFDO2dCQUNkLGNBQWMsRUFBRSxDQUFDO2dCQUNqQixXQUFXLEVBQUUsQ0FBQztnQkFDZCxXQUFXLEVBQUUsQ0FBQzthQUNmLENBQUMsQ0FBQztRQUNMLENBQUM7UUFFRCxJQUFNLEtBQUssR0FBRyxJQUFJLEtBQUssQ0FBQyxPQUFPLEVBQUU7WUFDL0IsSUFBSSxFQUFFLE1BQU07WUFDWixJQUFJLEVBQUUsRUFBQyxRQUFRLFVBQUEsRUFBQztZQUNoQixPQUFPLEVBQUU7Z0JBQ1AsU0FBUyxFQUFFLEVBQUMsUUFBUSxFQUFFLENBQUMsRUFBQztnQkFDeEIsVUFBVSxFQUFFLEtBQUs7Z0JBQ2pCLE1BQU0sRUFBRTtvQkFDTixLQUFLLEVBQUUsQ0FBQzs0QkFDTixJQUFJLEVBQUUsUUFBUTs0QkFDZCxRQUFRLEVBQUUsUUFBUTs0QkFDbEIsS0FBSyxFQUFFO2dDQUNMLEdBQUcsRUFBRSxpQkFBaUIsQ0FBQyxHQUFHO2dDQUMxQixHQUFHLEVBQUUsaUJBQWlCLENBQUMsR0FBRztnQ0FDMUIsUUFBUSxFQUFFLGlCQUFpQixDQUFDLFFBQVE7Z0NBQ3BDLFFBQVEsRUFBRSxVQUFDLEtBQWE7b0NBQ3RCLE1BQU0sQ0FBQyxpQkFBaUIsQ0FBQyx3QkFBd0IsSUFBSSxJQUFJO3dDQUNyRCxpQkFBaUIsQ0FBQyx3QkFBd0IsQ0FBQyxDQUFDLEtBQUssQ0FBQzt3Q0FDbEQsQ0FBQyxLQUFLLENBQUM7Z0NBQ2IsQ0FBQzs2QkFFSzt5QkFDVCxDQUFDO29CQUNGLEtBQUssRUFBRSxDQUFDOzRCQUNOLEtBQUssRUFBRTtnQ0FDTCxRQUFRLEVBQUUsVUFBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLE1BQU07b0NBQzdCLE1BQU0sQ0FBQyxLQUFLLEdBQUcsSUFBSSxDQUFDO2dDQUN0QixDQUFDOzZCQUNGO3lCQUNGLENBQUM7aUJBQ0g7Z0JBQ0QsUUFBUSxFQUFFLEVBQUMsSUFBSSxFQUFFLE9BQU8sRUFBQztnQkFDekIsS0FBSyxFQUFFLEVBQUMsSUFBSSxFQUFFLGlCQUFpQixDQUFDLElBQUksRUFBQzthQUN0QztTQUNGLENBQUMsQ0FBQztRQUNILE1BQU0sQ0FBQyxLQUFLLENBQUMsT0FBTyxHQUFHLE1BQU0sQ0FBQztRQUU5QixJQUFNLFVBQVUsR0FDWixJQUFJLENBQUMsZ0JBQWdCLENBQUMsY0FBYyxDQUFDLENBQUMsc0JBQXNCLENBQ2pELENBQUM7UUFDaEIsVUFBVSxDQUFDLEtBQUssQ0FBQyxPQUFPLEdBQUcsT0FBTyxDQUFDO1FBRW5DLElBQU0sZUFBZSxHQUNqQixJQUFJLENBQUMsZ0JBQWdCLENBQUMsb0JBQW9CLENBQUMsQ0FBQyxzQkFBc0IsQ0FDdkQsQ0FBQztRQUNoQixlQUFlLENBQUMsU0FBUyxHQUFHLEVBQUUsQ0FBQztRQUMvQixlQUFlLENBQUMsS0FBSyxDQUFDLE9BQU8sR0FBRyxNQUFNLENBQUM7UUFHdkMsSUFBTSxPQUFPLEdBQUcsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUN6QixHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUNoRSxPQUFPLENBQUMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUN4RCxDQUFDO1FBQ0QsZUFBZSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsa0JBQWtCLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztRQUU5RCxJQUFJLENBQUMsaUJBQWlCLENBQ2xCLEtBQUssRUFBRSxpQkFBaUIsRUFBRSxzQkFBc0IsRUFDaEQsaUJBQWlCLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDN0IsQ0FBQztJQUVPLDBDQUFrQixHQUExQixVQUEyQixNQUFnQjtRQUN6QyxJQUFNLG1CQUFtQixHQUFHLFFBQVEsQ0FBQyxhQUFhLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDMUQsbUJBQW1CLENBQUMsU0FBUyxHQUFHLGdDQUFnQyxDQUFDO1FBRWpFLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsTUFBTSxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1lBQ3ZDLElBQU0sb0JBQW9CLEdBQUcsUUFBUSxDQUFDLGFBQWEsQ0FBQyxLQUFLLENBQUMsQ0FBQztZQUMzRCxvQkFBb0IsQ0FBQyxTQUFTLEdBQUcsaUNBQWlDLENBQUM7WUFDbkUsb0JBQW9CLENBQUMsU0FBUyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUMzQyxtQkFBbUIsQ0FBQyxXQUFXLENBQUMsb0JBQW9CLENBQUMsQ0FBQztRQUN4RCxDQUFDO1FBQ0QsTUFBTSxDQUFDLG1CQUFtQixDQUFDO0lBQzdCLENBQUM7SUFFTyx5Q0FBaUIsR0FBekIsVUFDSSxLQUFZLEVBQUUsaUJBQW9DLEVBQ2xELHNCQUE4QixFQUFFLElBQVk7UUFGaEQsaUJBcUVDO1FBbEVDLElBQU0sZUFBZSxHQUNqQixJQUFJLENBQUMsZ0JBQWdCLENBQUMsb0JBQW9CLENBQUMsQ0FBQyxzQkFBc0IsQ0FDdkQsQ0FBQztRQUNoQixFQUFFLENBQUMsQ0FBQyxJQUFJLEdBQUcsaUJBQWlCLENBQUMsR0FBRztZQUM1QixJQUFJLENBQUMsWUFBWSxDQUFDLHNCQUFzQixDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQzlDLElBQUksQ0FBQyxZQUFZLENBQUMsc0JBQXNCLENBQUMsR0FBRyxLQUFLLENBQUM7WUFFbEQsZUFBZSxDQUFDLEtBQUssQ0FBQyxPQUFPLEdBQUcsRUFBRSxDQUFDO1lBRW5DLElBQU0sTUFBTSxHQUNSLElBQUksQ0FBQyxnQkFBZ0IsQ0FBQyxXQUFXLENBQUMsQ0FBQyxzQkFBc0IsQ0FDeEMsQ0FBQztZQUN0QixNQUFNLENBQUMsS0FBSyxDQUFDLE9BQU8sR0FBRyxPQUFPLENBQUM7WUFDL0IsS0FBSyxDQUFDLE1BQU0sRUFBRSxDQUFDO1lBRWYsSUFBTSxVQUFVLEdBQ1osSUFBSSxDQUFDLGdCQUFnQixDQUFDLGNBQWMsQ0FBQyxDQUFDLHNCQUFzQixDQUNqRCxDQUFDO1lBQ2hCLFVBQVUsQ0FBQyxLQUFLLENBQUMsT0FBTyxHQUFHLE1BQU0sQ0FBQztZQUVsQyxNQUFNLENBQUM7UUFDVCxDQUFDO1FBRUQsSUFBTSxtQkFBbUIsR0FBRyxRQUFRLENBQUMsYUFBYSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQzFELG1CQUFtQixDQUFDLFNBQVMsR0FBRyxnQ0FBZ0MsQ0FBQztRQUVqRSxJQUFNLFNBQVMsR0FBYSxDQUFDLEVBQUUsR0FBRyxJQUFJLENBQUMsQ0FBQztRQUN4QyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUNoRSxJQUFNLFlBQVksR0FBRyxpQkFBaUIsQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDeEQsSUFBTSxhQUFhLEdBQUcsWUFBWSxDQUFDLGFBQWEsQ0FBQztZQUVqRCxJQUFNLElBQUksR0FBRyxpQkFBaUIsQ0FBQyx3QkFBd0IsSUFBSSxJQUFJO2dCQUMzRCxpQkFBaUIsQ0FBQyx3QkFBd0IsQ0FBQyxJQUFJLENBQUM7Z0JBQ2hELElBQUksQ0FBQztZQUVULElBQUksWUFBWSxTQUFRLENBQUM7WUFDekIsSUFBSSxTQUFTLFNBQVEsQ0FBQztZQUN0QixJQUFJLElBQUksR0FBRyxDQUFDLENBQUM7WUFDYixJQUFJLE9BQU8sR0FBRyxJQUFJLENBQUM7WUFFbkIsSUFBSSxDQUFDO2dCQUNILElBQUksR0FBRyxhQUFhLENBQUMsSUFBSSxDQUFDLENBQUM7Z0JBQzNCLFlBQVksR0FBRyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQztnQkFDdEMsU0FBUyxHQUFHLFlBQVksQ0FBQztZQUMzQixDQUFDO1lBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDWCxPQUFPLEdBQUcsS0FBSyxDQUFDO2dCQUNoQixZQUFZLEdBQUcsT0FBTyxDQUFDO2dCQUN2QixTQUFTLEdBQUcsQ0FBQyxDQUFDLE9BQU8sQ0FBQztZQUN4QixDQUFDO1lBRUQsRUFBRSxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ2QsRUFBRSxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztvQkFDWixZQUFZLENBQUMsU0FBUyxDQUFDLElBQUksQ0FBQyxFQUFDLENBQUMsRUFBRSxJQUFJLEVBQUUsQ0FBQyxFQUFFLElBQUksRUFBQyxDQUFDLENBQUM7Z0JBQ2xELENBQUM7Z0JBQ0QsU0FBUyxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUMsQ0FBQztZQUMvQixDQUFDO1lBQ0QsT0FBTyxDQUFDLEdBQUcsQ0FBQyxZQUFZLENBQUMsSUFBSSxHQUFHLEdBQUcsR0FBRyxJQUFJLEdBQUcsS0FBSyxHQUFHLFNBQVMsQ0FBQyxDQUFDO1FBQ2xFLENBQUM7UUFDRCxlQUFlLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxrQkFBa0IsQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDO1FBRWhFLElBQUksSUFBSSxpQkFBaUIsQ0FBQyxRQUFRLENBQUM7UUFFbkMsVUFBVSxDQUNOLGNBQU0sT0FBQSxLQUFJLENBQUMsaUJBQWlCLENBQ3hCLEtBQUssRUFBRSxpQkFBaUIsRUFBRSxzQkFBc0IsRUFBRSxJQUFJLENBQUMsRUFEckQsQ0FDcUQsRUFDM0QsR0FBRyxDQUFDLENBQUM7SUFDWCxDQUFDO0lBQ0gsb0JBQUM7QUFBRCxDQW5NQSxBQW1NQyxDQW5Na0MsNEJBQW9CLEdBbU10RDtBQW5NWSxzQ0FBYTtBQW9NMUIsUUFBUSxDQUFDLGVBQWUsQ0FBQyxhQUFhLENBQUMsU0FBUyxDQUFDLEVBQUUsRUFBRSxhQUFhLENBQUMsQ0FBQzs7Ozs7QUNqTnBFLG9EQUFzRDtBQUN0RCxvRUFBZ0U7QUFDaEUsZ0VBQWtFO0FBQ2xFLCtDQUFpRDtBQUlqRCxJQUFNLE9BQU8sR0FBRyxFQUFFLENBQUM7QUFFTixRQUFBLHVCQUF1QixHQUFrQixVQUFDLElBQVk7SUFDakUsSUFBTSxhQUFhLEdBQTZCLENBQUMsSUFBSSxFQUFFLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNoRSxJQUFNLFdBQVcsR0FBRyxDQUFDLENBQUM7SUFDdEIsSUFBTSxTQUFTLEdBQUcsRUFBRSxDQUFDO0lBQ3JCLElBQU0sTUFBTSxHQUFHLENBQUMsQ0FBQztJQUNqQixJQUFNLE9BQU8sR0FBRyxTQUFTLENBQUMsaUJBQWlCLENBQUMsYUFBYSxFQUFFLFNBQVMsRUFBRSxNQUFNLENBQUMsQ0FBQztJQUM5RSxJQUFNLGNBQWMsR0FDaEIsU0FBUyxDQUFDLG9CQUFvQixDQUMxQixhQUFhLEVBQUUsU0FBUyxFQUFFLFdBQVcsRUFBRSxNQUFNLEVBQUUsT0FBTyxDQUFDLENBQUM7SUFFaEUsSUFBTSxlQUFlLEdBQUcsU0FBUyxDQUFDLHFCQUFxQixDQUFDLGFBQWEsQ0FBQyxDQUFDO0lBQ3ZFLElBQU0sZ0JBQWdCLEdBQUcsU0FBUyxDQUFDLHFCQUFxQixDQUFDLGNBQWMsQ0FBQyxDQUFDO0lBRXpFLElBQU0sS0FBSyxHQUFHLElBQUksNEJBQVksRUFBRSxDQUFDO0lBQ2pDLElBQU0sT0FBTyxHQUNULEtBQUssQ0FBQyxhQUFhLENBQUMsWUFBWSxDQUFDLDhCQUE4QixDQUMzRCxhQUFhLEVBQUUsU0FBUyxFQUFFLE1BQU0sRUFBRSxPQUFPLENBQUMsQ0FBQyxDQUFDO0lBRXBELElBQU0sWUFBWSxHQUNkLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxlQUFlLENBQUMsQ0FBQyxDQUFDLEVBQUUsZUFBZSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDdEUsSUFBTSxhQUFhLEdBQ2YsS0FBSyxDQUFDLG1CQUFtQixDQUFDLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxFQUFFLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFFeEUsSUFBTSxTQUFTLEdBQUcsU0FBUyxDQUFDLGtCQUFrQixDQUMxQyxlQUFlLENBQUMsQ0FBQyxDQUFDLEdBQUcsZUFBZSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBRXBELEtBQUssQ0FBQyxxQkFBcUIsQ0FDdkIsWUFBWSxFQUFFLGVBQWUsQ0FBQyxDQUFDLENBQUMsRUFBRSxlQUFlLENBQUMsQ0FBQyxDQUFDLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFFckUsSUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7UUFDakMsWUFBWSxDQUFDLGFBQWEsQ0FDdEIsS0FBSyxFQUFFLE9BQU8sRUFBRSxZQUFZLEVBQUUsYUFBYSxFQUFFLGdCQUFnQixDQUFDLENBQUM7SUFDckUsQ0FBQztJQUVELEtBQUssQ0FBQyx5QkFBeUIsQ0FDM0IsYUFBYSxFQUFFLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxFQUFFLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDN0QsSUFBTSxHQUFHLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBRTlCLElBQU0sT0FBTyxHQUFHLENBQUMsR0FBRyxHQUFHLEtBQUssQ0FBQyxHQUFHLE9BQU8sQ0FBQztJQUV4QyxLQUFLLENBQUMsbUJBQW1CLENBQUMsWUFBWSxDQUFDLENBQUM7SUFDeEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLGFBQWEsQ0FBQyxDQUFDO0lBQ3pDLEtBQUssQ0FBQyxhQUFhLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDN0IsS0FBSyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBRWhCLE1BQU0sQ0FBQyxPQUFPLENBQUM7QUFDakIsQ0FBQyxDQUFDO0FBRVcsUUFBQSw2QkFBNkIsR0FBa0IsVUFBQyxJQUFZO0lBQ3ZFLElBQU0sYUFBYSxHQUE2QixDQUFDLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDaEUsSUFBTSxXQUFXLEdBQUcsQ0FBQyxDQUFDO0lBQ3RCLElBQU0sU0FBUyxHQUFHLEVBQUUsQ0FBQztJQUNyQixJQUFNLE1BQU0sR0FBRyxDQUFDLENBQUM7SUFDakIsSUFBTSxPQUFPLEdBQUcsU0FBUyxDQUFDLGlCQUFpQixDQUFDLGFBQWEsRUFBRSxTQUFTLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFDOUUsSUFBTSxjQUFjLEdBQ2hCLFNBQVMsQ0FBQyxvQkFBb0IsQ0FDMUIsYUFBYSxFQUFFLFNBQVMsRUFBRSxXQUFXLEVBQUUsTUFBTSxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBRWhFLElBQU0sZUFBZSxHQUFHLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQyxhQUFhLENBQUMsQ0FBQztJQUN2RSxJQUFNLGdCQUFnQixHQUFHLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQyxjQUFjLENBQUMsQ0FBQztJQUV6RSxJQUFNLEtBQUssR0FBRyxJQUFJLDRCQUFZLEVBQUUsQ0FBQztJQUNqQyxJQUFNLE9BQU8sR0FDVCxLQUFLLENBQUMsYUFBYSxDQUFDLFlBQVksQ0FBQyx1Q0FBdUMsQ0FDcEUsYUFBYSxFQUFFLFNBQVMsRUFBRSxNQUFNLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUVwRCxJQUFNLFlBQVksR0FDZCxLQUFLLENBQUMsbUJBQW1CLENBQUMsZUFBZSxDQUFDLENBQUMsQ0FBQyxFQUFFLGVBQWUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3RFLElBQU0sYUFBYSxHQUNmLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBRXhFLElBQU0sU0FBUyxHQUFHLFNBQVMsQ0FBQyxrQkFBa0IsQ0FDMUMsZUFBZSxDQUFDLENBQUMsQ0FBQyxHQUFHLGVBQWUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUVwRCxLQUFLLENBQUMscUJBQXFCLENBQ3ZCLFlBQVksRUFBRSxlQUFlLENBQUMsQ0FBQyxDQUFDLEVBQUUsZUFBZSxDQUFDLENBQUMsQ0FBQyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBRXJFLElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQ2pDLFlBQVksQ0FBQyxhQUFhLENBQ3RCLEtBQUssRUFBRSxPQUFPLEVBQUUsWUFBWSxFQUFFLGFBQWEsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDO0lBQ3JFLENBQUM7SUFFRCxLQUFLLENBQUMseUJBQXlCLENBQzNCLGFBQWEsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzdELElBQU0sR0FBRyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUU5QixJQUFNLE9BQU8sR0FBRyxDQUFDLEdBQUcsR0FBRyxLQUFLLENBQUMsR0FBRyxPQUFPLENBQUM7SUFFeEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLFlBQVksQ0FBQyxDQUFDO0lBQ3hDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxhQUFhLENBQUMsQ0FBQztJQUN6QyxLQUFLLENBQUMsYUFBYSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBQzdCLEtBQUssQ0FBQyxPQUFPLEVBQUUsQ0FBQztJQUVoQixNQUFNLENBQUMsT0FBTyxDQUFDO0FBQ2pCLENBQUMsQ0FBQzs7Ozs7QUN6R0Ysb0RBQXVEO0FBQ3ZELGtEQUF3RDtBQUl4RCxJQUFNLGlCQUFpQixHQUFHLENBQUMsQ0FBQztBQUVmLFFBQUEsY0FBYyxHQUFrQixVQUFDLElBQVk7SUFDeEQsRUFBRSxDQUFDLENBQUMsSUFBSSxHQUFHLEdBQUcsQ0FBQyxDQUFDLENBQUM7UUFDZixNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDWixDQUFDO0lBQ0QsSUFBTSxJQUFJLEdBQUcsSUFBSSx5QkFBYyxFQUFFLENBQUM7SUFDbEMsSUFBTSxDQUFDLEdBQUcsaUJBQU8sQ0FBQyxXQUFXLENBQVUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDNUQsSUFBTSxDQUFDLEdBQUcsaUJBQU8sQ0FBQyxXQUFXLENBQVUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDNUQsSUFBTSxJQUFJLEdBQUcsQ0FBQyxJQUFJLEdBQUcsR0FBRyxDQUFDLEdBQUcsaUJBQWlCLEdBQUcsQ0FBQyxDQUFDO0lBQ2xELElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQzlCLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ3BCLENBQUM7SUFDRCxJQUFNLEdBQUcsR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDOUIsTUFBTSxDQUFDLENBQUMsR0FBRyxHQUFHLEtBQUssQ0FBQyxHQUFHLElBQUksQ0FBQztBQUM5QixDQUFDLENBQUM7Ozs7O0FDckJGLDRDQUFzRDtBQUN0RCxrREFBK0M7QUFDL0Msb0VBQWdFO0FBQ2hFLDREQUE4RDtBQUM5RCwwRUFBNEU7QUFDNUUsK0NBQWlEO0FBSWpELElBQU0sT0FBTyxHQUFHLEVBQUUsQ0FBQztBQUVOLFFBQUEsY0FBYyxHQUFrQixVQUFDLElBQVk7SUFDeEQsSUFBTSxLQUFLLEdBQUcsSUFBSSw0QkFBWSxFQUFFLENBQUM7SUFDakMsSUFBTSxRQUFRLEdBQUcsS0FBSyxDQUFDLG1CQUFtQixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUN2RCxJQUFNLFFBQVEsR0FBRyxLQUFLLENBQUMsbUJBQW1CLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3ZELElBQU0sYUFBYSxHQUFHLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFFNUQsSUFBTSxJQUFJLEdBQUcsSUFBSSxpQkFBTyxDQUNwQixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsRUFBRSxFQUFDLE9BQU8sRUFBRSxRQUFRLEVBQUUsY0FBYyxFQUFFLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxFQUFDLENBQUMsQ0FBQztJQUNyRSxJQUFNLElBQUksR0FBRyxJQUFJLGlCQUFPLENBQ3BCLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxFQUFFLEVBQUMsT0FBTyxFQUFFLFFBQVEsRUFBRSxjQUFjLEVBQUUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUMsQ0FBQyxDQUFDO0lBQ3JFLElBQU0sTUFBTSxHQUFHLElBQUksaUJBQU8sQ0FDdEIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUUsRUFBQyxPQUFPLEVBQUUsYUFBYSxFQUFFLGNBQWMsRUFBRSxDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsRUFBQyxDQUFDLENBQUM7SUFDMUUsSUFBTSxPQUFPLEdBQUcsS0FBSyxDQUFDLGFBQWEsQ0FBQyxVQUFVLENBQUMsaUJBQWlCLENBQzVELElBQUksRUFBRSxJQUFJLEVBQUUsTUFBTSxFQUFFLHdCQUFpQixDQUFDLE9BQU8sRUFDN0Msd0JBQWlCLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUVoQyxJQUFNLENBQUMsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQUMsSUFBSSxHQUFHLElBQUksRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUMzRCxJQUFNLENBQUMsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQUMsSUFBSSxHQUFHLElBQUksRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUMzRCxLQUFLLENBQUMscUJBQXFCLENBQUMsUUFBUSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDckQsS0FBSyxDQUFDLHFCQUFxQixDQUFDLFFBQVEsRUFBRSxJQUFJLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBRXJELElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQ2pDLFVBQVUsQ0FBQyxjQUFjLENBQ3JCLEtBQUssRUFBRSxPQUFPLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxhQUFhLEVBQUUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQztJQUN2RSxDQUFDO0lBRUQsSUFBTSxNQUFNLEdBQUcsS0FBSyxDQUFDLHlCQUF5QixDQUFDLGFBQWEsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDMUUsSUFBTSxPQUFPLEdBQUcsQ0FBQyxXQUFXLENBQUMsR0FBRyxFQUFFLEdBQUcsS0FBSyxDQUFDLEdBQUcsT0FBTyxDQUFDO0lBRXRELEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxRQUFRLENBQUMsQ0FBQztJQUNwQyxLQUFLLENBQUMsbUJBQW1CLENBQUMsUUFBUSxDQUFDLENBQUM7SUFDcEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLGFBQWEsQ0FBQyxDQUFDO0lBQ3pDLEtBQUssQ0FBQyxhQUFhLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDN0IsS0FBSyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBRWhCLE1BQU0sQ0FBQyxPQUFPLENBQUM7QUFDakIsQ0FBQyxDQUFDO0FBRVcsUUFBQSxxQkFBcUIsR0FBa0IsVUFBQyxJQUFZO0lBQy9ELElBQU0sS0FBSyxHQUFHLElBQUksNEJBQVksRUFBRSxDQUFDO0lBQ2pDLElBQU0sT0FBTyxHQUNULEtBQUssQ0FBQyxhQUFhLENBQUMsaUJBQWlCLENBQUMsdUJBQXVCLENBQ3pELElBQUksRUFBRSx3QkFBaUIsQ0FBQyxPQUFPLEVBQUUsd0JBQWlCLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUVyRSxJQUFNLFFBQVEsR0FBRyxLQUFLLENBQUMseUJBQXlCLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQzdELElBQU0sUUFBUSxHQUFHLEtBQUssQ0FBQyx5QkFBeUIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDN0QsSUFBTSxhQUFhLEdBQUcsS0FBSyxDQUFDLHlCQUF5QixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUVsRSxJQUFNLENBQUMsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQUMsSUFBSSxHQUFHLElBQUksRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUMzRCxJQUFNLENBQUMsR0FBRyxTQUFTLENBQUMsa0JBQWtCLENBQUMsSUFBSSxHQUFHLElBQUksRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUMzRCxLQUFLLENBQUMsMkJBQTJCLENBQUMsUUFBUSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDM0QsS0FBSyxDQUFDLDJCQUEyQixDQUFDLFFBQVEsRUFBRSxJQUFJLEVBQUUsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBRTNELElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQ2pDLGlCQUFpQixDQUFDLG9CQUFvQixDQUNsQyxLQUFLLEVBQUUsT0FBTyxFQUFFLFFBQVEsRUFBRSxRQUFRLEVBQUUsYUFBYSxFQUFFLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUM7SUFDdkUsQ0FBQztJQUVELElBQU0sTUFBTSxHQUNSLEtBQUssQ0FBQywrQkFBK0IsQ0FBQyxhQUFhLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3JFLElBQU0sT0FBTyxHQUFHLENBQUMsV0FBVyxDQUFDLEdBQUcsRUFBRSxHQUFHLEtBQUssQ0FBQyxHQUFHLE9BQU8sQ0FBQztJQUV0RCxLQUFLLENBQUMsbUJBQW1CLENBQUMsUUFBUSxDQUFDLENBQUM7SUFDcEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLFFBQVEsQ0FBQyxDQUFDO0lBQ3BDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxhQUFhLENBQUMsQ0FBQztJQUN6QyxLQUFLLENBQUMsYUFBYSxDQUFDLE9BQU8sQ0FBQyxDQUFDO0lBQzdCLEtBQUssQ0FBQyxPQUFPLEVBQUUsQ0FBQztJQUVoQixNQUFNLENBQUMsT0FBTyxDQUFDO0FBQ2pCLENBQUMsQ0FBQzs7O0FDbkZGLE9BQU8sQ0FBQyxFQUFDLEVBQUUsRUFBRSxhQUFhLEVBQUMsQ0FBQyxDQUFDOzs7QUNBN0IsT0FBTyxDQUFDLEVBQUMsRUFBRSxFQUFFLGFBQWEsRUFBQyxDQUFDLENBQUM7Ozs7O0FDNEM3Qix3QkFBK0IsSUFBVTtJQUV2QyxNQUFNLENBQUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxJQUFXLENBQWlDLENBQUM7QUFDcEUsQ0FBQztBQUhELHdDQUdDOzs7OztBQzlDRCw4QkFBZ0M7QUFFaEMsbUNBQ0ksT0FBaUIsRUFBRSxPQUFpQixFQUFFLElBQVksRUFDbEQsa0JBQXVCO0lBQXZCLG1DQUFBLEVBQUEsdUJBQXVCO0lBQ3pCLElBQUksQ0FBQyxNQUFNLENBQ1AsT0FBTyxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQ3BCLGtCQUFrQixHQUFHLHdDQUF3QyxDQUFDLENBQUM7SUFDbkUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsTUFBTSxLQUFLLENBQUMsRUFDcEIsa0JBQWtCLEdBQUcsd0NBQXdDLENBQUMsQ0FBQztJQUVuRSxJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksSUFBSSxDQUFDLElBQUksSUFBSSxHQUFHLENBQUMsRUFBRSw0Q0FBNEMsQ0FBQyxDQUFDO0lBRXpFLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7UUFDM0IsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsS0FBSyxJQUFJLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsS0FBSyxPQUFPLENBQUMsQ0FBQyxDQUFDLENBQUMsRUFDM0Msa0JBQWtCO2FBQ2QsWUFBVSxPQUFPLDBCQUFxQixPQUFPLGFBQVUsQ0FBQTtZQUN2RCx3QkFBd0IsQ0FBQyxDQUFDO0lBQ3BDLENBQUM7QUFDSCxDQUFDO0FBcEJELDhEQW9CQztBQUVELG9DQUNJLE9BQWlCLEVBQUUsT0FBaUIsRUFDcEMsSUFBWTtJQUNkLElBQUksQ0FBQyxNQUFNLENBQUMsT0FBTyxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQUUsd0NBQXdDLENBQUMsQ0FBQztJQUM1RSxJQUFJLENBQUMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFLHVDQUF1QyxDQUFDLENBQUM7SUFFM0UsSUFBTSxXQUFXLEdBQUcsT0FBTyxDQUFDLEtBQUssRUFBRSxDQUFDO0lBQ3BDLFdBQVcsQ0FBQyxJQUFJLENBQUMsSUFBSSxPQUFPLENBQUMsSUFBSSxDQUFDLENBQUM7SUFDbkMsTUFBTSxDQUFDLFdBQXVDLENBQUM7QUFDakQsQ0FBQztBQVRELGdFQVNDOzs7OztBQ2pDRCw4QkFBZ0M7QUFFaEMsOEJBQ0kscUJBQStDLEVBQUUsU0FBaUIsRUFDbEUsS0FBYSxFQUFFLE1BQWMsRUFBRSxPQUFnQjtJQUNqRCxFQUFFLENBQUMsQ0FBQyxPQUFPLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQztRQUNwQixPQUFPLEdBQUcsaUJBQWlCLENBQUMscUJBQXFCLEVBQUUsU0FBUyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBQ3hFLENBQUM7SUFDRCxJQUFNLFNBQVMsR0FBRyxxQkFBcUIsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUMzQyxJQUFNLFNBQVMsR0FBRyxxQkFBcUIsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUMzQyxJQUFNLFVBQVUsR0FBRyxDQUFDLFNBQVMsR0FBRyxTQUFTLEdBQUcsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxHQUFHLE1BQU0sR0FBRyxDQUFDLENBQUM7SUFDdEUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsS0FBSyxDQUFDLFVBQVUsQ0FBQyxFQUN0QiwyQkFBeUIsVUFBVSxzQ0FBbUM7UUFDbEUsbUNBQW1DLENBQUMsQ0FBQztJQUU3QyxJQUFNLFVBQVUsR0FBRyxDQUFDLFNBQVMsR0FBRyxTQUFTLEdBQUcsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxHQUFHLE1BQU0sR0FBRyxDQUFDLENBQUM7SUFDdEUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsS0FBSyxDQUFDLFVBQVUsQ0FBQyxFQUN0Qiw4QkFBNEIsVUFBVSxrQ0FBK0I7UUFDakUsdUNBQXVDLENBQUMsQ0FBQztJQUVqRCxNQUFNLENBQUMsQ0FBQyxVQUFVLEVBQUUsVUFBVSxFQUFFLEtBQUssQ0FBQyxDQUFDO0FBQ3pDLENBQUM7QUFyQkQsb0RBcUJDO0FBRUQsMkJBQ0ksVUFBb0MsRUFBRSxTQUFpQixFQUN2RCxNQUFjO0lBQ2hCLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsTUFBTSxHQUFHLENBQUMsQ0FBQyxHQUFHLE1BQU0sR0FBRyxTQUFTLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQztBQUM3RSxDQUFDO0FBSkQsOENBSUM7QUFFRCwrQkFDSSxnQkFBMEM7SUFDNUMsTUFBTSxDQUFDLENBQUMsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEVBQUUsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEdBQUcsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztBQUMxRSxDQUFDO0FBSEQsc0RBR0M7QUFFRCwrQkFDSSxVQUFrQixFQUFFLFdBQW1CLEVBQ3ZDLEtBQWE7SUFDZixNQUFNLENBQUMsQ0FBQyxLQUFLLEVBQUUsS0FBSyxFQUFFLFVBQVUsRUFBRSxXQUFXLENBQUMsQ0FBQztBQUNqRCxDQUFDO0FBSkQsc0RBSUM7QUFFRCxnQ0FDSSxVQUFrQixFQUFFLFdBQW1CLEVBQ3ZDLFNBQWlCO0lBQ25CLE1BQU0sQ0FBQyxDQUFDLFNBQVMsR0FBRyxTQUFTLEdBQUcsVUFBVSxFQUFFLFdBQVcsQ0FBQyxDQUFDO0FBQzNELENBQUM7QUFKRCx3REFJQztBQUVELCtCQUFzQyxXQUFtQjtJQUN2RCxNQUFNLENBQUMsQ0FBQyxDQUFDLEVBQUUsV0FBVyxDQUFDLENBQUM7QUFDMUIsQ0FBQztBQUZELHNEQUVDO0FBRUQsMEJBQ0ksRUFBb0IsRUFBRSxVQUFrQjtJQUMxQyxJQUFNLFdBQVcsR0FBRyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxVQUFVLEdBQUcsQ0FBQyxDQUFDO0lBQ2pELElBQU0sV0FBVyxHQUFHLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxHQUFHLFVBQVUsR0FBRyxDQUFDLENBQUM7SUFDakQsTUFBTSxDQUFDLENBQUMsV0FBVyxFQUFFLFdBQVcsQ0FBQyxDQUFDO0FBQ3BDLENBQUM7QUFMRCw0Q0FLQzs7Ozs7QUN6REQsd0JBQ0ksVUFBNEIsRUFBRSxRQUEwQjtJQUMxRCxJQUFNLE9BQU8sR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzlDLElBQU0sT0FBTyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDMUMsRUFBRSxDQUFDLENBQUMsT0FBTyxLQUFLLE9BQU8sQ0FBQyxDQUFDLENBQUM7UUFDeEIsSUFBTSxNQUFNLEdBQUcsR0FBRyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBQyxHQUFHLEdBQUcsQ0FBQztRQUNoRSxJQUFNLE1BQU0sR0FBRyxHQUFHLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsR0FBRyxDQUFDO1FBQzVELE1BQU0sSUFBSSxLQUFLLENBQ1gsb0RBQW9ELEdBQUcsTUFBTTtZQUM3RCxTQUFTLEdBQUcsT0FBTyxHQUFHLGVBQWUsR0FBRyxNQUFNLEdBQUcsU0FBUyxHQUFHLE9BQU8sQ0FBQyxDQUFDO0lBQzVFLENBQUM7QUFDSCxDQUFDO0FBWEQsd0NBV0M7Ozs7O0FDWEQsOEJBQWdDO0FBQ2hDLCtDQUFpRDtBQUNqRCwyQ0FBNkM7QUFFN0MscUNBQThFO0FBSTlFO0lBV0UscUJBQW9CLFFBQWlCO1FBQWpCLGFBQVEsR0FBUixRQUFRLENBQVM7UUFWN0Isa0JBQWEsR0FBZ0IsRUFBRSxDQUFDO1FBR2hDLG1CQUFjLEdBQWdCLEVBQUUsQ0FBQztRQUNqQyw4QkFBeUIsR0FBYyxFQUFFLENBQUM7SUFNVixDQUFDO0lBVXpDLDJCQUFLLEdBQUwsVUFDSSxPQUV5RDtRQUg3RCxpQkFhQztRQVRDLElBQUksQ0FBQyxVQUFVLEVBQUUsQ0FBQztRQUVsQixJQUFNLE1BQU0sR0FBRyxVQUFvQixPQUFVLElBQVEsT0FBQSxLQUFJLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxFQUFsQixDQUFrQixDQUFDO1FBQ3hFLElBQU0sT0FBTyxHQUFHLFVBQW9CLE9BQVUsSUFBUSxPQUFBLEtBQUksQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUFDLEVBQW5CLENBQW1CLENBQUM7UUFDMUUsSUFBTSxNQUFNLEdBQUcsT0FBTyxDQUFDLE1BQU0sRUFBRSxPQUFPLENBQUMsQ0FBQztRQUV4QyxJQUFJLENBQUMsUUFBUSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBRXRCLE1BQU0sQ0FBQyxNQUFNLENBQUM7SUFDaEIsQ0FBQztJQU1ELGdDQUFVLEdBQVY7UUFDRSxJQUFNLFFBQVEsR0FBYyxFQUFFLENBQUM7UUFDL0IsSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUM7UUFDbEMsSUFBSSxDQUFDLFdBQVcsR0FBRyxRQUFRLENBQUM7UUFFNUIsSUFBTSxpQkFBaUIsR0FBYyxFQUFFLENBQUM7UUFDeEMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxJQUFJLENBQUMsaUJBQWlCLENBQUMsQ0FBQztRQUM1QyxJQUFJLENBQUMseUJBQXlCLEdBQUcsaUJBQWlCLENBQUM7SUFDckQsQ0FBQztJQU1ELDhCQUFRLEdBQVIsVUFBUyxNQUFtQjtRQUE1QixpQkFvQ0M7UUFsQ0MsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1lBQ2pELElBQU0sT0FBTyxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFFcEMsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLG1CQUFtQixDQUFDLE9BQU8sRUFBRSxJQUFJLENBQUMseUJBQXlCLENBQUM7Z0JBQ2pFLENBQUMsTUFBTSxJQUFJLElBQUksSUFBSSxNQUFNLFlBQVksaUJBQU87b0JBQzNDLE9BQU8sQ0FBQyxPQUFPLEVBQUUsS0FBTSxNQUFrQixDQUFDLE9BQU8sRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUMxRCxRQUFRLENBQUM7WUFDWCxDQUFDO1lBQ0QsT0FBTyxDQUFDLE9BQU8sRUFBRSxDQUFDO1FBQ3BCLENBQUM7UUFHRCxJQUFJLENBQUMsYUFBYSxDQUFDLEdBQUcsRUFBRSxDQUFDO1FBQ3pCLElBQUksQ0FBQyxXQUFXLEdBQUcsSUFBSSxDQUFDLGFBQWEsQ0FBQyxNQUFNLEtBQUssQ0FBQztZQUM5QyxJQUFLO1lBQ0wsSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsQ0FBQztRQUd0RCxFQUFFLENBQUMsQ0FBQyxNQUFNLFlBQVksaUJBQU87WUFDekIsQ0FBQyxJQUFJLENBQUMsbUJBQW1CLENBQUMsTUFBTSxFQUFFLElBQUksQ0FBQyx5QkFBeUIsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN0RSxJQUFJLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQ3JCLENBQUM7UUFBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLE9BQU8sQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDakMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxVQUFBLENBQUM7Z0JBQ2QsRUFBRSxDQUFDLENBQUMsQ0FBQyxZQUFZLGlCQUFPO29CQUNwQixDQUFDLEtBQUksQ0FBQyxtQkFBbUIsQ0FBQyxDQUFDLEVBQUUsS0FBSSxDQUFDLHlCQUF5QixDQUFDLENBQUMsQ0FBQyxDQUFDO29CQUNqRSxLQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNoQixDQUFDO1lBQ0gsQ0FBQyxDQUFDLENBQUM7UUFDTCxDQUFDO1FBRUQsSUFBSSxDQUFDLGNBQWMsQ0FBQyxHQUFHLEVBQUUsQ0FBQztRQUMxQixJQUFJLENBQUMseUJBQXlCLEdBQUcsSUFBSSxDQUFDLGNBQWMsQ0FBQyxNQUFNLEtBQUssQ0FBQztZQUM3RCxJQUFLO1lBQ0wsSUFBSSxDQUFDLGNBQWMsQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUMsQ0FBQztJQUMxRCxDQUFDO0lBRU8seUNBQW1CLEdBQTNCLFVBQTRCLE9BQWdCLEVBQUUsV0FBc0I7UUFDbEUsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxXQUFXLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7WUFDNUMsRUFBRSxDQUFDLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDLE9BQU8sRUFBRSxLQUFLLE9BQU8sQ0FBQyxPQUFPLEVBQUUsQ0FBQyxDQUFDLENBQUM7Z0JBQ25ELE1BQU0sQ0FBQyxJQUFJLENBQUM7WUFDZCxDQUFDO1FBQ0gsQ0FBQztRQUNELE1BQU0sQ0FBQyxLQUFLLENBQUM7SUFDZixDQUFDO0lBTUQsMEJBQUksR0FBSixVQUF3QixNQUFTO1FBQy9CLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxXQUFXLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQztZQUM3QixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQztnQkFDbEIsTUFBTSxJQUFJLEtBQUssQ0FDWCwrQ0FBK0M7b0JBQy9DLHNDQUFzQztvQkFDdEMsd0RBQXdEO29CQUN4RCxRQUFRLENBQUMsQ0FBQztZQUNoQixDQUFDO1lBQ0QsTUFBTSxDQUFDLE1BQU0sQ0FBQztRQUNoQixDQUFDO1FBQ0QsSUFBSSxDQUFDLHlCQUF5QixDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUM1QyxNQUFNLENBQUMsTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFPRCwyQkFBSyxHQUFMLFVBQXlCLE1BQVM7UUFDaEMsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFdBQVcsSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDO1lBQzdCLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO2dCQUNsQixNQUFNLElBQUksS0FBSyxDQUNYLCtDQUErQztvQkFDL0Msc0NBQXNDO29CQUN0Qyx3REFBd0Q7b0JBQ3hELFFBQVEsQ0FBQyxDQUFDO1lBQ2hCLENBQUM7WUFDRCxNQUFNLENBQUMsTUFBTSxDQUFDO1FBQ2hCLENBQUM7UUFDRCxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUM5QixNQUFNLENBQUMsTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFhRCw0QkFBTSxHQUFOLFVBQ0ksQ0FBVSxFQUFFLENBQVUsRUFBRSxZQUF3QyxFQUNoRSxZQUF3QztRQURoQiw2QkFBQSxFQUFBLGVBQWUsaUJBQWlCLENBQUMsT0FBTztRQUNoRSw2QkFBQSxFQUFBLGVBQWUsaUJBQWlCLENBQUMsT0FBTztRQUMxQyxJQUFNLFdBQVcsR0FDYixDQUFDLFlBQVksS0FBSyxpQkFBaUIsQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDM0UsSUFBTSxXQUFXLEdBQ2IsQ0FBQyxZQUFZLEtBQUssaUJBQWlCLENBQUMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRTNFLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQzVCLHVEQUFxRCxDQUFDLENBQUMsSUFBTTthQUN6RCxTQUFPLENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFFMUIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxXQUFXLEtBQUssV0FBVyxFQUMzQixvQ0FBa0MsV0FBVyxZQUFTO2FBQy9DLFdBQVcsa0NBQTZCLENBQUMsQ0FBQyxLQUFLLFVBQU8sQ0FBQTthQUN0RCxDQUFDLENBQUMsS0FBSywwQkFBcUIsaUJBQWlCLENBQUMsWUFBWSxDQUFHLENBQUE7YUFDaEUsVUFBUSxpQkFBaUIsQ0FBQyxZQUFZLENBQUMsaUJBQWMsQ0FBQSxDQUFDLENBQUM7UUFFL0QsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLFlBQVksRUFBRSxZQUFZLENBQUMsQ0FBQyxDQUFDO0lBQzNFLENBQUM7SUFVRCx1Q0FBaUIsR0FBakIsVUFBa0IsQ0FBVSxFQUFFLE1BQWU7UUFDM0MsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixrRUFBa0U7YUFDOUQsVUFBUSxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzNCLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2pCLG1FQUFtRTthQUMvRCxVQUFRLE1BQU0sQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDaEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQzFCLDZEQUEyRCxDQUFDLENBQUMsSUFBSSxPQUFJO1lBQ2pFLDZEQUE2RDthQUM3RCxVQUFRLE1BQU0sQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFFaEMsTUFBTSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDO0lBQ3ZELENBQUM7SUFPRCx1Q0FBaUIsR0FBakIsVUFBa0IsTUFBZSxFQUFFLENBQVU7UUFDM0MsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixnRUFBZ0U7YUFDNUQsVUFBUSxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzNCLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2pCLG9FQUFvRTthQUNoRSxVQUFRLE1BQU0sQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDaEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQzFCLDREQUEwRCxDQUFDLENBQUMsSUFBSSxNQUFHO1lBQy9ELDZEQUE2RDthQUM3RCxXQUFTLE1BQU0sQ0FBQyxLQUFLLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFFbEMsTUFBTSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDO0lBQ3ZELENBQUM7SUFPRCxnQ0FBVSxHQUFWLFVBQVcsRUFBVyxFQUFFLEVBQVc7UUFDakMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDOUIsNERBQTREO2FBQ3JELEVBQUUsQ0FBQyxJQUFJLGFBQVEsRUFBRSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUN0QyxJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxJQUFJLEtBQUssRUFBRSxDQUFDLElBQUksRUFDbkIsMENBQXdDLEVBQUUsQ0FBQyxJQUFJLFlBQVM7YUFDakQsRUFBRSxDQUFDLElBQUksa0JBQWUsQ0FBQSxDQUFDLENBQUM7UUFDbkMsTUFBTSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLElBQUksQ0FBQyxFQUFFLEVBQUUsQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLFFBQVEsRUFBRSxDQUFDO0lBQzFFLENBQUM7SUFPRCxrQ0FBWSxHQUFaLFVBQWEsRUFBVyxFQUFFLEVBQVc7UUFDbkMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDOUIsOERBQThEO2FBQ3ZELEVBQUUsQ0FBQyxJQUFJLGFBQVEsRUFBRSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUV0QyxNQUFNLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7SUFDL0QsQ0FBQztJQVVELDJCQUFLLEdBQUwsVUFBeUIsT0FBVTtRQUNqQyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDakQsQ0FBQztJQVVELDZCQUFPLEdBQVAsVUFDSSxPQUFXLEVBQUUsUUFBa0I7UUFDakMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsSUFBSSxLQUFLLElBQUksQ0FBQyxhQUFhLENBQUMsUUFBUSxDQUFDLEVBQzdDLGdDQUE4QixPQUFPLENBQUMsSUFBSSwwQkFBdUI7YUFDMUQsSUFBSSxDQUFDLGFBQWEsQ0FBQyxRQUFRLENBQUMsTUFBRyxDQUFBLENBQUMsQ0FBQztRQUM1QyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFTLE9BQU8sRUFBRSxRQUFRLENBQUMsQ0FBQyxDQUFDO0lBQ3JFLENBQUM7SUFZRCw2QkFBTyxHQUFQLFVBQVEsS0FBYyxFQUFFLEtBQXVCLEVBQUUsSUFBc0I7UUFFckUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDO1lBQ2hDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFDeEMsZ0RBQThDLEtBQUssZUFBWTthQUN4RCxJQUFJLHVDQUFrQyxLQUFLLENBQUMsS0FBSyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ2pFLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDO0lBQzlELENBQUM7SUFlRCw0QkFBTSxHQUFOLFVBQ0ksTUFBZSxFQUFFLFdBQTZCLEVBQzlDLFVBQTRCLEVBQUUsSUFBYSxFQUFFLFNBQTJCLEVBQ3hFLFFBQTBCO1FBQzVCLElBQUksQ0FBQyxNQUFNLENBQ1AsV0FBVyxDQUFDLENBQUMsQ0FBQyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsSUFBSSxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQztZQUM3QyxXQUFXLENBQUMsQ0FBQyxDQUFDLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBQyxJQUFJLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQ3JELHNEQUFvRCxXQUFXLE1BQUc7YUFDOUQscUJBQW1CLFVBQVUsbUNBQWdDLENBQUE7YUFDN0QsY0FBWSxNQUFNLENBQUMsS0FBSyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ3JDLElBQUksQ0FBQyxNQUFNLENBQ1AsU0FBUyxDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQztZQUN2QyxTQUFTLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxJQUFJLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQy9DLG9EQUFrRCxTQUFTLE1BQUc7YUFDMUQscUJBQW1CLFFBQVEsb0NBQWlDLENBQUE7YUFDNUQsV0FBUyxJQUFJLENBQUMsS0FBSyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ2hDLFdBQVcsQ0FBQyxjQUFjLENBQUMsVUFBVSxFQUFFLFFBQVEsQ0FBQyxDQUFDO1FBRWpELE1BQU0sQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUN0QixNQUFNLEVBQUUsV0FBVyxFQUFFLFVBQVUsRUFBRSxJQUFJLEVBQUUsU0FBUyxFQUFFLFFBQVEsQ0FBQyxDQUFDO0lBQ2xFLENBQUM7SUFvQ0QsOEJBQVEsR0FBUixVQUFTLFFBQWlCLEVBQUUsUUFBaUIsRUFBRSxJQUFZO1FBQ3pELGFBQWEsQ0FBQyx5QkFBeUIsQ0FDbkMsUUFBUSxDQUFDLEtBQUssRUFBRSxRQUFRLENBQUMsS0FBSyxFQUFFLElBQUksRUFBRSxxQkFBcUIsQ0FBQyxDQUFDO1FBQ2pFLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxnQkFBZ0IsQ0FBQyxRQUFRLEVBQUUsUUFBUSxFQUFFLElBQUksQ0FBQyxDQUFDLENBQUM7SUFDckUsQ0FBQztJQVlELCtCQUFTLEdBQVQsVUFBVSxPQUFnQjtRQUN4QixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsaUJBQWlCLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNyRCxDQUFDO0lBT0QseUJBQUcsR0FBSCxVQUFJLE9BQWdCO1FBQ2xCLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUMvQyxDQUFDO0lBT0QsNEJBQU0sR0FBTixVQUFPLE9BQWdCO1FBQ3JCLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNsRCxDQUFDO0lBT0QsNEJBQU0sR0FBTixVQUFPLE9BQWdCO1FBQ3JCLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNsRCxDQUFDO0lBUUQsa0NBQVksR0FBWixVQUFhLEVBQVcsRUFBRSxFQUFXO1FBQ25DLElBQUksQ0FBQyxpQkFBaUIsQ0FBQyxFQUFFLENBQUMsS0FBSyxFQUFFLEVBQUUsQ0FBQyxLQUFLLEVBQUUseUJBQXlCLENBQUMsQ0FBQztRQUN0RSxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsb0JBQW9CLENBQUMsRUFBRSxFQUFFLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDdkQsQ0FBQztJQVFELDBCQUFJLEdBQUosVUFBSyxPQUFnQixFQUFFLENBQVM7UUFDOUIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLElBQUksT0FBTyxDQUFDLElBQUksRUFDakIsNkJBQTJCLENBQUMsdUNBQW9DO2FBQzVELHdCQUFzQixPQUFPLENBQUMsS0FBSyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ2hELElBQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxZQUFZLENBQUMsT0FBTyxFQUFFLENBQUMsQ0FBQyxDQUFDO1FBQzdDLElBQUksQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBQzFCLElBQUksQ0FBQyxLQUFLLENBQUMsTUFBTSxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQzNCLE1BQU0sQ0FBQyxNQUFNLENBQUM7SUFDaEIsQ0FBQztJQVFELHlCQUFHLEdBQUgsVUFBSSxPQUFnQjtRQUNsQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQU9ELHlCQUFHLEdBQUgsVUFBSSxPQUFnQjtRQUNsQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQU9ELDZCQUFPLEdBQVAsVUFBUSxDQUFVO1FBQWxCLGlCQVFDO1FBUEMsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUM7WUFHaEIsSUFBTSxHQUFHLEdBQUcsS0FBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUM5QixJQUFNLFNBQVMsR0FBRyxLQUFJLENBQUMsZ0JBQWdCLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDO1lBQ2hELE1BQU0sQ0FBQyxLQUFJLENBQUMsR0FBRyxDQUFDLFNBQVMsQ0FBQyxDQUFDO1FBQzdCLENBQUMsQ0FBQyxDQUFDO0lBQ0wsQ0FBQztJQVdELCtCQUFTLEdBQVQsVUFBNkIsQ0FBSSxFQUFFLE1BQWdCO1FBQ2pELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxNQUFNLENBQUMsTUFBTSxFQUN4QiwrQ0FBNkMsQ0FBQyxDQUFDLEtBQUssTUFBRzthQUNuRCxxQ0FBbUMsTUFBTSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ3RELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUMsQ0FBQztJQUN2RCxDQUFDO0lBU0QscUNBQWUsR0FBZixVQUFtQyxDQUFTLEVBQUUsQ0FBSTtRQUNoRCxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLG1FQUFtRTthQUMvRCxVQUFRLENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDM0IsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLHVCQUF1QixDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3hELENBQUM7SUFTRCxzQ0FBZ0IsR0FBaEIsVUFBb0MsQ0FBUyxFQUFFLENBQUk7UUFDakQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixvRUFBb0U7YUFDaEUsVUFBUSxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzNCLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyx3QkFBd0IsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN6RCxDQUFDO0lBU0Qsc0NBQWdCLEdBQWhCLFVBQW9DLENBQUksRUFBRSxDQUFTO1FBQ2pELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osaUVBQWlFO2FBQzdELGNBQVksQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMvQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsd0JBQXdCLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDekQsQ0FBQztJQVFELHlCQUFHLEdBQUgsVUFBdUIsQ0FBSTtRQUN6QixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDekMsQ0FBQztJQVFELHlCQUFHLEdBQUgsVUFBdUIsQ0FBSSxFQUFFLENBQUk7UUFDL0IsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDO1FBQzNELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDNUMsQ0FBQztJQVFELHlCQUFHLEdBQUgsVUFBdUIsQ0FBSSxFQUFFLENBQUk7UUFDL0IsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDO1FBQzNELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDNUMsQ0FBQztJQVNELG9DQUFjLEdBQWQsVUFBa0MsQ0FBSSxFQUFFLENBQUk7UUFDMUMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSwyQkFBMkIsQ0FBQyxDQUFDO1FBQ3RFLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxzQkFBc0IsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN2RCxDQUFDO0lBU0QsNEJBQU0sR0FBTixVQUEwQixDQUFJLEVBQUUsQ0FBSTtRQUNsQyxJQUFJLENBQUMsaUJBQWlCLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsS0FBSyxFQUFFLG1CQUFtQixDQUFDLENBQUM7UUFDOUQsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUMvQyxDQUFDO0lBU0QsMENBQW9CLEdBQXBCLFVBQXdDLENBQVMsRUFBRSxDQUFJO1FBQ3JELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osb0VBQW9FO2FBQ2hFLHlCQUF1QixDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyw0QkFBNEIsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM3RCxDQUFDO0lBVUQsMENBQW9CLEdBQXBCLFVBQXdDLENBQUksRUFBRSxDQUFTO1FBQ3JELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osaUVBQWlFO2FBQzdELDZCQUEyQixDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzlDLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyw0QkFBNEIsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM3RCxDQUFDO0lBUUQseUJBQUcsR0FBSCxVQUF1QixPQUFVO1FBQy9CLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUMvQyxDQUFDO0lBT0QseUJBQUcsR0FBSCxVQUF1QixPQUFVO1FBQy9CLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUMvQyxDQUFDO0lBT0QsMEJBQUksR0FBSixVQUF3QixPQUFVO1FBQ2hDLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNoRCxDQUFDO0lBT0QsNkJBQU8sR0FBUCxVQUEyQixPQUFVO1FBQ25DLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNuRCxDQUFDO0lBT0QsMEJBQUksR0FBSixVQUF3QixPQUFVO1FBQ2hDLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNoRCxDQUFDO0lBT0QseUJBQUcsR0FBSCxVQUF1QixPQUFVO1FBQy9CLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUMvQyxDQUFDO0lBUUQsMEJBQUksR0FBSixVQUF3QixPQUFVO1FBQ2hDLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNoRCxDQUFDO0lBVUQsb0NBQWMsR0FBZCxVQUFrQyxFQUFVLEVBQUUsQ0FBSSxFQUFFLEVBQVUsRUFBRSxDQUFJO1FBQ2xFLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2IsK0RBQStEO2FBQzNELFdBQVMsRUFBRSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUM3QixJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNiLGtFQUFrRTthQUM5RCxxQkFBbUIsRUFBRSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUN2QyxJQUFJLENBQUMsaUJBQWlCLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsS0FBSyxFQUFFLDJCQUEyQixDQUFDLENBQUM7UUFFdEUsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLHNCQUFzQixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDL0QsQ0FBQztJQVVELHNDQUFnQixHQUFoQixVQUFvQyxDQUFTLEVBQUUsQ0FBSTtRQUNqRCxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLG9FQUFvRTthQUNoRSxjQUFZLENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDL0IsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLHdCQUF3QixDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3pELENBQUM7SUFXRCw2Q0FBdUIsR0FBdkIsVUFBd0IsQ0FBVSxFQUFFLENBQVU7UUFDNUMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWiwyREFBMkQ7YUFDdkQsMEJBQXdCLENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDM0MsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWiw0REFBNEQ7YUFDeEQsMEJBQXdCLENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDM0MsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLCtCQUErQixDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ2hFLENBQUM7SUFrQkQsNEJBQU0sR0FBTixVQUNJLENBQVUsRUFBRSxPQUFnQixFQUFFLE1BQW9CLEVBQUUsTUFBYyxFQUNsRSxPQUFlO1FBQ2pCLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1oscURBQW1ELENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQyxDQUFDO1FBQ2xFLElBQUksQ0FBQyxNQUFNLENBQ1AsT0FBTyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2xCLHdEQUF3RDthQUNqRCxPQUFPLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzVCLEVBQUUsQ0FBQyxDQUFDLE1BQU0sSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDO1lBQ25CLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2pCLHVEQUF1RDtpQkFDaEQsTUFBTSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUM3QixDQUFDO1FBRUQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQy9CLHNDQUFvQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxtQkFBZ0I7YUFDMUQsNkJBQTJCLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFHeEQsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDLEVBQUUsT0FBTyxFQUFFLE1BQU0sRUFBRSxNQUFNLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUM5RSxDQUFDO0lBY0Qsb0NBQWMsR0FBZCxVQUNJLENBQVUsRUFBRSxFQUFXLEVBQUUsT0FBZ0IsRUFBRSxNQUFjLEVBQ3pELEdBQVc7UUFDYixJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLDJEQUEyRDthQUNwRCxDQUFDLENBQUMsS0FBSyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ3ZCLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2IsNERBQTREO2FBQ3JELEVBQUUsQ0FBQyxLQUFLLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDeEIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDbEIsaUVBQWlFO2FBQzFELE9BQU8sQ0FBQyxLQUFLLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDN0IsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQy9CLHlDQUF1QyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxZQUFTO2FBQ3RELG9DQUFrQyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQy9ELElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsS0FBSyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUNoQywyQ0FBeUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsWUFBUzthQUN6RCxxQ0FBbUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsT0FBSSxDQUFBLENBQUMsQ0FBQztRQUVqRSxJQUFNLGNBQWMsR0FDaEIsSUFBSSxDQUFDLHNCQUFzQixDQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsT0FBTyxFQUFFLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQztRQUU3RCxJQUFJLENBQUMsS0FBSyxDQUFDLGNBQWMsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUM5QixJQUFJLENBQUMsS0FBSyxDQUFDLGNBQWMsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUM5QixJQUFJLENBQUMsS0FBSyxDQUFDLGNBQWMsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUU5QixNQUFNLENBQUMsY0FBYyxDQUFDO0lBQ3hCLENBQUM7SUFnQkQscUNBQWUsR0FBZixVQUNJLENBQVUsRUFBRSxPQUFnQixFQUFFLE1BQW9CLEVBQUUsTUFBYyxFQUNsRSxHQUFXO1FBQ2IsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWiwyREFBMkQ7YUFDcEQsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUN0QixJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNsQiw0REFBNEQ7YUFDeEQsVUFBUSxPQUFPLENBQUMsSUFBTSxDQUFBLENBQUMsQ0FBQztRQUNoQyxFQUFFLENBQUMsQ0FBQyxNQUFNLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQztZQUNuQixJQUFJLENBQUMsTUFBTSxDQUNQLE1BQU0sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNqQix1RkFDWSxNQUFNLENBQUMsSUFBSSxNQUFHLENBQUMsQ0FBQztRQUNsQyxDQUFDO1FBQ0QsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQy9CLCtDQUE2QyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxZQUFTO2FBQzVELG1DQUFpQyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBRTlELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUNiLElBQUksQ0FBQyx1QkFBdUIsQ0FBQyxDQUFDLEVBQUUsT0FBTyxFQUFFLE1BQU0sRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUNyRSxDQUFDO0lBYUQsNkJBQU8sR0FBUCxVQUFRLENBQVUsRUFBRSxLQUFhLEVBQUUsTUFBYyxFQUFFLEdBQVc7UUFDNUQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixrREFBa0QsR0FBRyxDQUFDLENBQUMsSUFBSSxHQUFHLEdBQUcsQ0FBQyxDQUFDO1FBQ3ZFLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLEtBQUssRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUNqRSxDQUFDO0lBYUQscUNBQWUsR0FBZixVQUNJLEVBQVcsRUFBRSxDQUFVLEVBQUUsS0FBYSxFQUFFLE1BQWMsRUFDdEQsR0FBVztRQUNiLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2IsMkRBQTJEO2FBQ3BELEVBQUUsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDdkIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWiwwREFBMEQ7YUFDbkQsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUV0QixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsdUJBQXVCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxLQUFLLEVBQUUsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUM7SUFDN0UsQ0FBQztJQWFELDZCQUFPLEdBQVAsVUFBUSxDQUFVLEVBQUUsS0FBYSxFQUFFLE1BQWMsRUFBRSxHQUFXO1FBQzVELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1oscURBQW1ELENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQyxDQUFDO1FBQ2xFLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLEtBQUssRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUNqRSxDQUFDO0lBWUQsNkJBQU8sR0FBUCxVQUFRLENBQVUsRUFBRSxLQUFhLEVBQUUsTUFBYyxFQUFFLEdBQVc7UUFDNUQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixxREFBbUQsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFDLENBQUM7UUFDbEUsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxDQUFDLEVBQUUsS0FBSyxFQUFFLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQyxDQUFDO0lBQ2pFLENBQUM7SUFjRCxzQ0FBZ0IsR0FBaEIsVUFDSSxDQUFVLEVBQUUsVUFBNEIsRUFBRSxZQUFvQjtRQUFwQiw2QkFBQSxFQUFBLG9CQUFvQjtRQUNoRSxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLDhEQUE0RCxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUMsQ0FBQztRQUMzRSxJQUFJLENBQUMsTUFBTSxDQUNQLFVBQVUsQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUN2Qiw4REFBOEQ7YUFDdkQsVUFBVSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzFCLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUNiLElBQUksQ0FBQyx3QkFBd0IsQ0FBQyxDQUFDLEVBQUUsVUFBVSxFQUFFLFlBQVksQ0FBQyxDQUFDLENBQUM7SUFDbEUsQ0FBQztJQWdCRCwwQ0FBb0IsR0FBcEIsVUFDSSxDQUFVLEVBQUUsSUFBcUIsRUFBRSxRQUF5QixFQUM1RCxlQUFzQixFQUFFLEtBQXVCLEVBQy9DLE1BQXdCO1FBRHhCLGdDQUFBLEVBQUEsc0JBQXNCO1FBRXhCLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osK0RBQStEO2FBQ3hELENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDdEIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxJQUFJLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDbEMsbUVBQW1FO2FBQy9ELGNBQVksSUFBSSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUNsQyxJQUFJLENBQUMsTUFBTSxDQUNQLFFBQVEsQ0FBQyxJQUFJLEtBQUssQ0FBQyxJQUFJLFFBQVEsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUMxQyxtRUFBbUU7YUFDL0Qsa0JBQWdCLFFBQVEsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDMUMsRUFBRSxDQUFDLENBQUMsS0FBSyxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDbEIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxLQUFLLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDcEMsZ0VBQWdFO2lCQUM1RCxrQkFBZ0IsS0FBTSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMxQyxDQUFDO1FBQ0QsRUFBRSxDQUFDLENBQUMsTUFBTSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDbkIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDdEMsaUVBQWlFO2lCQUM3RCxrQkFBZ0IsTUFBTyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMzQyxDQUFDO1FBRUQsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLDRCQUE0QixDQUMvQyxDQUFDLEVBQUUsSUFBSSxFQUFFLFFBQVEsRUFBRSxlQUFlLEVBQUUsS0FBSyxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUM7SUFDMUQsQ0FBQztJQUtILGtCQUFDO0FBQUQsQ0E1Z0NBLEFBNGdDQyxJQUFBO0FBNWdDcUIsa0NBQVc7QUE4Z0NqQyxJQUFZLGlCQUdYO0FBSEQsV0FBWSxpQkFBaUI7SUFDM0IsK0RBQU8sQ0FBQTtJQUNQLHFFQUFVLENBQUE7QUFDWixDQUFDLEVBSFcsaUJBQWlCLEdBQWpCLHlCQUFpQixLQUFqQix5QkFBaUIsUUFHNUI7Ozs7Ozs7Ozs7Ozs7OztBQ3poQ0QsNkNBQStDO0FBQy9DLDhCQUFnQztBQUVoQywrQ0FBaUQ7QUFDakQsMkNBQTZDO0FBQzdDLCtCQUFzRDtBQUN0RCxxQ0FBOEU7QUFFOUU7SUFBb0Msa0NBQVc7SUFDN0Msd0JBQVksUUFBZ0I7UUFBaEIseUJBQUEsRUFBQSxnQkFBZ0I7ZUFDMUIsa0JBQU0sUUFBUSxDQUFDO0lBQ2pCLENBQUM7SUFFUyxzQ0FBYSxHQUF2QixVQUEyQyxPQUFVO1FBQ25ELE1BQU0sQ0FBQyxpQkFBTyxDQUFDLElBQUksQ0FDZixPQUFPLENBQUMsS0FBSyxFQUFFLEVBQUMsTUFBTSxFQUFFLElBQUksWUFBWSxDQUFDLE9BQU8sQ0FBQyxTQUFTLEVBQUUsQ0FBQyxFQUFDLENBQUMsQ0FBQztJQUN0RSxDQUFDO0lBRVMsd0NBQWUsR0FBekIsVUFDSSxPQUFXLEVBQUUsUUFBa0I7UUFDakMsTUFBTSxDQUFDLElBQUksQ0FBQyxhQUFhLENBQUMsT0FBTyxDQUFDLENBQUMsT0FBTyxDQUFLLFFBQVEsQ0FBQyxDQUFDO0lBQzNELENBQUM7SUFFUyx3Q0FBZSxHQUF6QixVQUNJLEtBQWMsRUFBRSxXQUE2QixFQUM3QyxVQUE0QjtRQUM5QixJQUFNLE1BQU0sR0FBRyxpQkFBTyxDQUFDLEtBQUssQ0FBQyxVQUFVLENBQUMsQ0FBQztRQUN6QyxJQUFJLENBQUMsY0FBYyxDQUNmLEtBQUssRUFBRSxXQUFXLEVBQUUsVUFBVSxFQUFFLE1BQU0sRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxVQUFVLENBQUMsQ0FBQztRQUNoRSxNQUFNLENBQUMsTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFFUyx1Q0FBYyxHQUF4QixVQUNJLE1BQWUsRUFBRSxpQkFBbUMsRUFDcEQsZ0JBQWtDLEVBQUUsSUFBYSxFQUNqRCxlQUFpQyxFQUNqQyxjQUFnQztRQUNsQyxXQUFXLENBQUMsY0FBYyxDQUFDLGdCQUFnQixFQUFFLGNBQWMsQ0FBQyxDQUFDO1FBQzdELElBQU0sU0FBUyxHQUFHLE1BQU0sQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUNyQyxJQUFNLFNBQVMsR0FBRyxJQUFJLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDbkMsSUFBTSxDQUFDLEdBQUcsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLEdBQUcsZ0JBQWdCLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDcEQsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUMzQixJQUFNLE1BQU0sR0FBRyxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsR0FBRyxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQzFFLElBQU0sTUFBTSxHQUFHLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxHQUFHLGdCQUFnQixDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDaEUsSUFBTSxNQUFNLEdBQUcsTUFBTSxHQUFHLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDO1lBQ2pELElBQU0sTUFBTSxHQUFHLGVBQWUsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsR0FBRyxjQUFjLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUN0RSxJQUFNLE1BQU0sR0FBRyxlQUFlLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsY0FBYyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDNUQsSUFBTSxNQUFNLEdBQUcsTUFBTSxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsTUFBTSxDQUFDO1lBQy9DLFNBQVMsQ0FBQyxNQUFNLENBQUMsR0FBRyxTQUFTLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDeEMsQ0FBQztJQUNILENBQUM7SUFFUyx5Q0FBZ0IsR0FBMUIsVUFBMkIsRUFBVyxFQUFFLEVBQVcsRUFBRSxJQUFZO1FBQy9ELElBQU0sV0FBVyxHQUNiLGFBQWEsQ0FBQywwQkFBMEIsQ0FBQyxFQUFFLENBQUMsS0FBSyxFQUFFLEVBQUUsQ0FBQyxLQUFLLEVBQUUsSUFBSSxDQUFDLENBQUM7UUFFdkUsSUFBTSxNQUFNLEdBQUcsaUJBQU8sQ0FBQyxLQUFLLENBQVUsV0FBVyxDQUFDLENBQUM7UUFFbkQsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxXQUFXLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUN4QyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFdBQVcsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO2dCQUN4QyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFdBQVcsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO29CQUV4QyxJQUFNLEtBQUssR0FBNkIsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO29CQUNsRCxJQUFJLEtBQUssU0FBUSxDQUFDO29CQUNsQixFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLENBQUM7d0JBQ2pDLEtBQUssR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7b0JBQzFCLENBQUM7b0JBQUMsSUFBSSxDQUFDLENBQUM7d0JBQ04sS0FBSyxDQUFDLElBQUksQ0FBQyxJQUFJLEVBQUUsQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLENBQUM7d0JBQ3ZCLElBQUEsYUFBRSxFQUFFLGFBQUUsRUFBRSxhQUFFLENBQVU7d0JBQzNCLEtBQUssR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsRUFBRSxDQUFDLENBQUM7b0JBQzdCLENBQUM7b0JBRUQsTUFBTSxDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztnQkFDN0IsQ0FBQztZQUNILENBQUM7UUFDSCxDQUFDO1FBRUQsTUFBTSxDQUFDLE1BQU0sQ0FBQztJQUNoQixDQUFDO0lBRVMsZ0RBQXVCLEdBQWpDLFVBQXFELENBQVMsRUFBRSxDQUFJO1FBQ2xFLElBQU0sWUFBWSxHQUFHLElBQUksWUFBWSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUM5QyxJQUFNLE9BQU8sR0FBRyxDQUFDLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDOUIsSUFBTSxJQUFJLEdBQUcsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDO1FBQ3JCLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsWUFBWSxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQzdDLFlBQVksQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ3RDLENBQUM7UUFDRCxNQUFNLENBQUMsaUJBQU8sQ0FBQyxJQUFJLENBQUksQ0FBQyxDQUFDLEtBQUssRUFBRSxFQUFDLE1BQU0sRUFBRSxZQUFZLEVBQUMsQ0FBQyxDQUFDO0lBQzFELENBQUM7SUFFUywrQ0FBc0IsR0FBaEMsVUFDSSxFQUFVLEVBQUUsQ0FBSSxFQUFFLEVBQVUsRUFBRSxDQUFJO1FBQ3BDLElBQU0sT0FBTyxHQUFHLElBQUksWUFBWSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUN6QyxJQUFNLE9BQU8sR0FBRyxDQUFDLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDOUIsSUFBTSxPQUFPLEdBQUcsQ0FBQyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQzlCLElBQU0sS0FBSyxHQUFHLEVBQUUsQ0FBQyxHQUFHLEVBQUUsQ0FBQztRQUN2QixJQUFNLEtBQUssR0FBRyxFQUFFLENBQUMsR0FBRyxFQUFFLENBQUM7UUFDdkIsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxPQUFPLENBQUMsTUFBTSxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7WUFDeEMsT0FBTyxDQUFDLENBQUMsQ0FBQyxHQUFHLEtBQUssR0FBRyxPQUFPLENBQUMsQ0FBQyxDQUFDLEdBQUcsS0FBSyxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN2RCxDQUFDO1FBQ0QsTUFBTSxDQUFDLGlCQUFPLENBQUMsSUFBSSxDQUFJLENBQUMsQ0FBQyxLQUFLLEVBQUUsRUFBQyxNQUFNLEVBQUUsT0FBTyxFQUFDLENBQUMsQ0FBQztJQUNyRCxDQUFDO0lBRVMsaURBQXdCLEdBQWxDLFVBQXNELENBQVMsRUFBRSxDQUFJO1FBQ25FLElBQU0sU0FBUyxHQUFHLElBQUksWUFBWSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQztRQUMzQyxJQUFNLE9BQU8sR0FBRyxDQUFDLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDOUIsSUFBTSxJQUFJLEdBQUcsQ0FBQyxDQUFDLEdBQUcsRUFBRSxDQUFDO1FBQ3JCLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxDQUFDLE1BQU0sRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1lBQ3hDLFNBQVMsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ25DLENBQUM7UUFDRCxNQUFNLENBQUMsaUJBQU8sQ0FBQyxJQUFJLENBQUksQ0FBQyxDQUFDLEtBQUssRUFBRSxFQUFDLE1BQU0sRUFBRSxTQUFTLEVBQUMsQ0FBQyxDQUFDO0lBQ3ZELENBQUM7SUFFUyxpREFBd0IsR0FBbEMsVUFBc0QsQ0FBUyxFQUFFLENBQUk7UUFDbkUsSUFBTSxJQUFJLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUNqQyxJQUFNLE1BQU0sR0FBRyxJQUFJLENBQUMsdUJBQXVCLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxDQUFDO1FBRXJELElBQUksQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUVmLE1BQU0sQ0FBQyxNQUFNLENBQUM7SUFDaEIsQ0FBQztJQUVTLGlEQUF3QixHQUFsQyxVQUFzRCxDQUFJLEVBQUUsQ0FBUztRQUNuRSxJQUFNLElBQUksR0FBRyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQ2pDLElBQU0sTUFBTSxHQUFHLElBQUksQ0FBQyx1QkFBdUIsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUM7UUFFckQsSUFBSSxDQUFDLE9BQU8sRUFBRSxDQUFDO1FBRWYsTUFBTSxDQUFDLE1BQU0sQ0FBQztJQUNoQixDQUFDO0lBRVMsb0NBQVcsR0FBckIsVUFBeUMsQ0FBSTtRQUMzQyxNQUFNLENBQUMsSUFBSSxDQUFDLHdCQUF3QixDQUFDLGdCQUFNLENBQUMsT0FBTyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzFELENBQUM7SUFFUyxvQ0FBVyxHQUFyQixVQUF5QyxDQUFJLEVBQUUsQ0FBSTtRQUNqRCxNQUFNLENBQUMsSUFBSSxDQUFDLHNCQUFzQixDQUFJLGdCQUFNLENBQUMsR0FBRyxFQUFFLENBQUMsRUFBRSxnQkFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUN0RSxDQUFDO0lBRVMsb0NBQVcsR0FBckIsVUFBeUMsQ0FBSSxFQUFFLENBQUk7UUFDakQsTUFBTSxDQUFDLElBQUksQ0FBQyxzQkFBc0IsQ0FBSSxnQkFBTSxDQUFDLEdBQUcsRUFBRSxDQUFDLEVBQUUsZ0JBQU0sQ0FBQyxPQUFPLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDMUUsQ0FBQztJQUVTLHVDQUFjLEdBQXhCLFVBQ0ksQ0FBVSxFQUFFLENBQVUsRUFBRSxZQUF3QyxFQUNoRSxZQUF3QztRQURoQiw2QkFBQSxFQUFBLGVBQWUsd0JBQWlCLENBQUMsT0FBTztRQUNoRSw2QkFBQSxFQUFBLGVBQWUsd0JBQWlCLENBQUMsT0FBTztRQUMxQyxJQUFNLFNBQVMsR0FDWCxDQUFDLFlBQVksS0FBSyx3QkFBaUIsQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFM0UsSUFBTSxPQUFPLEdBQ1QsQ0FBQyxZQUFZLEtBQUssd0JBQWlCLENBQUMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzNFLElBQU0sUUFBUSxHQUNWLENBQUMsWUFBWSxLQUFLLHdCQUFpQixDQUFDLE9BQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUUzRSxJQUFNLFlBQVksR0FBRyxVQUFDLE1BQWUsRUFBRSxDQUFTLEVBQUUsQ0FBUztZQUN2RCxPQUFBLE1BQU0sQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUFoQixDQUFnQixDQUFDO1FBQ3JCLElBQU0sZ0JBQWdCLEdBQUcsVUFBQyxNQUFlLEVBQUUsQ0FBUyxFQUFFLENBQVM7WUFDM0QsT0FBQSxNQUFNLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUM7UUFBaEIsQ0FBZ0IsQ0FBQztRQUVyQixJQUFNLE9BQU8sR0FBRyxDQUFDLFlBQVksS0FBSyx3QkFBaUIsQ0FBQyxPQUFPLENBQUM7WUFDeEQsWUFBWTtZQUNaLGdCQUFnQixDQUFDO1FBQ3JCLElBQU0sT0FBTyxHQUFHLENBQUMsWUFBWSxLQUFLLHdCQUFpQixDQUFDLE9BQU8sQ0FBQztZQUN4RCxZQUFZO1lBQ1osZ0JBQWdCLENBQUM7UUFDckIsSUFBTSxNQUFNLEdBQUcsSUFBSSxZQUFZLENBQUMsT0FBTyxHQUFHLFFBQVEsQ0FBQyxDQUFDO1FBQ3BELElBQUksS0FBSyxHQUFHLENBQUMsQ0FBQztRQUVkLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7WUFDakMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxRQUFRLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztnQkFDbEMsSUFBSSxHQUFHLEdBQUcsQ0FBQyxDQUFDO2dCQUNaLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsU0FBUyxFQUFFLEVBQUUsQ0FBQyxFQUFFLENBQUM7b0JBRW5DLEdBQUcsSUFBSSxPQUFPLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsR0FBRyxPQUFPLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztnQkFDN0MsQ0FBQztnQkFDRCxNQUFNLENBQUMsS0FBSyxFQUFFLENBQUMsR0FBRyxHQUFHLENBQUM7WUFDeEIsQ0FBQztRQUNILENBQUM7UUFDRCxNQUFNLENBQUMsaUJBQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxPQUFPLEVBQUUsUUFBUSxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFDbEQsQ0FBQztJQUVTLCtDQUFzQixHQUFoQyxVQUFvRCxDQUFJLEVBQUUsQ0FBSTtRQUM1RCxJQUFNLFNBQVMsR0FBRyxJQUFJLFlBQVksQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDM0MsSUFBTSxPQUFPLEdBQUcsQ0FBQyxDQUFDLFNBQVMsRUFBRSxDQUFDO1FBQzlCLElBQU0sT0FBTyxHQUFHLENBQUMsQ0FBQyxTQUFTLEVBQUUsQ0FBQztRQUM5QixHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxNQUFNLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztZQUN4QyxTQUFTLENBQUMsQ0FBQyxDQUFDLEdBQUcsT0FBTyxDQUFDLENBQUMsQ0FBQyxHQUFHLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUN6QyxDQUFDO1FBQ0QsTUFBTSxDQUFDLGlCQUFPLENBQUMsSUFBSSxDQUFJLENBQUMsQ0FBQyxLQUFLLEVBQUUsRUFBQyxNQUFNLEVBQUUsU0FBUyxFQUFDLENBQUMsQ0FBQztJQUN2RCxDQUFDO0lBRVMsd0RBQStCLEdBQXpDLFVBQTBDLENBQVUsRUFBRSxDQUFVO1FBQzlELElBQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDaEQsSUFBTSxNQUFNLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUVoRCxJQUFNLE1BQU0sR0FBRyxJQUFJLFlBQVksQ0FBQyxNQUFNLEdBQUcsTUFBTSxDQUFDLENBQUM7UUFDakQsSUFBSSxLQUFLLEdBQUcsQ0FBQyxDQUFDO1FBQ2QsR0FBRyxDQUFDLENBQUMsSUFBSSxHQUFHLEdBQUcsQ0FBQyxFQUFFLEdBQUcsR0FBRyxNQUFNLEVBQUUsR0FBRyxFQUFFLEVBQUUsQ0FBQztZQUN0QyxHQUFHLENBQUMsQ0FBQyxJQUFJLEdBQUcsR0FBRyxDQUFDLEVBQUUsR0FBRyxHQUFHLE1BQU0sRUFBRSxHQUFHLEVBQUUsRUFBRSxDQUFDO2dCQUN0QyxNQUFNLENBQUMsS0FBSyxFQUFFLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO29CQUN2RCxDQUFDLENBQUMsR0FBRyxDQUFDLEdBQUcsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDaEQsQ0FBQztRQUNILENBQUM7UUFDRCxNQUFNLENBQUMsaUJBQU8sQ0FBQyxHQUFHLENBQUMsQ0FBQyxNQUFNLEVBQUUsTUFBTSxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQUVTLHVD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QU0sWUFBWSxHQUNkLFNBQVMsQ0FBQyxxQkFBcUIsQ0FBQyxVQUFVLEVBQUUsV0FBVyxFQUFFLEtBQUssQ0FBQyxDQUFDO1FBQ3BFLElBQU0sRUFBRSxHQUFHLGlCQUFPLENBQUMsS0FBSyxDQUFDLFlBQVksQ0FBQyxDQUFDO1FBRXZDLElBQU0sUUFBUSxHQUFHLEVBQUUsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDN0IsSUFBTSxRQUFRLEdBQUcsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztRQUM3QixJQUFNLFFBQVEsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBQzVCLElBQU0sUUFBUSxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFFNUIsR0FBRyxDQUFDLENBQUMsSUFBSSxFQUFFLEdBQUcsQ0FBQyxFQUFFLEVBQUUsR0FBRyxLQUFLLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQztZQUNsQyxJQUFNLEtBQUssR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsSUFBSSxDQUFDLENBQUMsT0FBTyxHQUFHLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxDQUFDLENBQUM7WUFDOUQsSUFBTSxLQUFLLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxRQUFRLEVBQUUsQ0FBQyxRQUFRLEdBQUcsT0FBTyxHQUFHLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxDQUFDO1lBRXJFLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxHQUFHLENBQUMsRUFBRSxFQUFFLEdBQUcsS0FBSyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUM7Z0JBQ2xDLElBQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsQ0FBQyxPQUFPLEdBQUcsRUFBRSxDQUFDLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQztnQkFDOUQsSUFBTSxLQUFLLEdBQUcsSUFBSSxD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QUFHLElBQUksQ0FBQyxPQUFPLEdBQUcsQ0FBQyxHQUFHLENBQUMsQ0FBQyxHQUFHLEtBQUssQ0FBQztJQUMzRSxDQUFDO0lBRUQscUJBQUcsR0FBSCxVQUFJLEtBQWEsRUFBRSxDQUFTLEVBQUUsQ0FBUyxFQUFFLENBQVMsRUFBRSxDQUFTO1FBQzNELElBQUksQ0FBQyxTQUFTLEVBQUUsQ0FDWCxJQUFJLENBQUMsT0FBTyxHQUFHLENBQUMsR0FBRyxJQUFJLENBQUMsT0FBTyxHQUFHLENBQUMsR0FBRyxJQUFJLENBQUMsT0FBTyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUM7SUFDNUUsQ0FBQztJQUVELDRCQUFVLEdBQVYsVUFBVyxJQUFzQztRQUMvQyxNQUFNLENBQUMsSUFBSSxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLE9BQU8sR0FBRyxJQUFJLENBQUMsQ0FBQyxDQUFDO1lBQ2xELElBQUksQ0FBQyxPQUFPLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUN2QyxDQUFDO0lBRUQsNEJBQVUsR0FBVixVQUFXLEtBQWE7UUFDdEIsSUFBTSxDQUFDLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxLQUFLLEdBQUcsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDO1FBQzNDLEtBQUssSUFBSSxDQUFDLEdBQUcsSUFBSSxDQUFDLE9BQU8sQ0FBQztRQUMxQixJQUFNLENBQUMsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDLEtBQUssR0FBRyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUM7UUFDM0MsS0FBSyxJQUFJLENBQUMsR0FBRyxJQUFJLENBQUMsT0FBTyxDQUFDO1FBQzFCLE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsSUFBSSxDQUFDLEtBQUssQ0FBQyxLQUFLLEdBQUcsSUFBSSxDQUFDLE9BQU8sQ0FBQyxFQUFFLEtBQUssR0FBRyxJQUFJLENBQUMsT0FBTyxD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QUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsQ0FBQztRQUNyRCxFQUFFLENBQUMsQ0FBQyxJQUFJLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDbkIsTUFBTSxDQUFDLENBQUMsQ0FBQyxFQUFFLElBQUksR0FBRyxDQUFDLENBQUMsQ0FBQztRQUN2QixDQUFDO0lBQ0gsQ0FBQztJQUNELE1BQU0sQ0FBQyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQztBQUNuQixDQUFDO0FBUEQsa0RBT0M7QUFFRCwrQkFBc0MsQ0FBUztJQUM3QyxJQUFNLGVBQWUsR0FBRyxJQUFJLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUMzQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxDQUFDO1FBQzNCLGVBQWUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUM7SUFDekIsQ0FBQztJQUNELE9BQU8sQ0FBQyxlQUFlLENBQUMsQ0FBQztJQUN6QixNQUFNLENBQUMsZUFBZSxDQUFDO0FBQ3pCLENBQUM7QUFQRCxzREFPQyIsImZpbGUiOiJnZW5lcmF0ZWQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlc0NvbnRlbnQiOlsiKGZ1bmN0aW9uIGUodCxuLHIpe2Z1bmN0aW9uIHMobyx1KXtpZighbltvXSl7aWYoIXRbb10pe3ZhciBhPXR5cGVvZiByZXF1aXJlPT1cImZ1bmN0aW9uXCImJnJlcXVpcmU7aWYoIXUmJmEpcmV0dXJuIGEobywhMCk7aWYoaSlyZXR1cm4gaShvLCEwKTt2YXIgZj1uZXcgRXJyb3IoXCJDYW5ub3QgZmluZCBtb2R1bGUgJ1wiK28rXCInXCIpO3Rocm93IGYuY29kZT1cIk1PRFVMRV9OT1RfRk9VTkRcIixmfXZhciBsPW5bb109e2V4cG9ydHM6e319O3Rbb11bMF0uY2FsbChsLmV4cG9ydHMsZnVuY3Rpb24oZSl7dmFyIG49dFtvXVsxXVtlXTtyZXR1cm4gcyhuP246ZSl9LGwsbC5leHBvcnRzLGUsdCxuLHIpfXJldHVybiBuW29dLmV4cG9ydHN9dmFyIGk9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtmb3IodmFyIG89MDtvPHIubGVuZ3RoO28rKylzKHJbb10pO3JldHVybiBzfSkiLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmV4cG9ydCBpbnRlcmZhY2UgQmVuY2htYXJrUnVuR3JvdXAge1xuICBuYW1lOiBzdHJpbmc7XG4gIC8vIE1pbiBhbmQgbWF4IHN0ZXBzIHRvIHJ1biB0aGUgYmVuY2htYXJrIHRlc3Qgb3Zlci5cbiAgbWluOiBudW1iZXI7XG4gIG1heDogbnVtYmVyO1xuICAvLyBUaGUgc2l6ZSBvZiB0aGUgc3RlcCB0byB0YWtlIGJldHdlZW4gYmVuY2htYXJrIHJ1bnMuXG4gIHN0ZXBTaXplOiBudW1iZXI7XG4gIC8vIEEgdHJhbnNmb3JtYXRpb24gb2Ygc3RlcCB0byB0aGUgc2l6ZSBwYXNzZWQgdG8gdGhlIGJlbmNobWFyayB0ZXN0LlxuICBzdGVwVG9TaXplVHJhbnNmb3JtYXRpb24/OiAoc3RlcDogbnVtYmVyKSA9PiBudW1iZXI7XG4gIGJlbmNobWFya1J1bnM6IEJlbmNobWFya1J1bltdO1xufVxuXG5leHBvcnQgY2xhc3MgQmVuY2htYXJrUnVuIHtcbiAgbmFtZTogc3RyaW5nO1xuICBiZW5jaG1hcmtUZXN0OiBCZW5jaG1hcmtUZXN0O1xuXG4gIGNoYXJ0RGF0YTogQ2hhcnREYXRhW107XG4gIGNvbnN0cnVjdG9yKG5hbWU6IHN0cmluZywgYmVuY2htYXJrVGVzdDogQmVuY2htYXJrVGVzdCkge1xuICAgIHRoaXMubmFtZSA9IG5hbWU7XG4gICAgdGhpcy5iZW5jaG1hcmtUZXN0ID0gYmVuY2htYXJrVGVzdDtcbiAgICB0aGlzLmNoYXJ0RGF0YSA9IFtdO1xuICB9XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgQmVuY2htYXJrVGVzdCB7IChzaXplOiBudW1iZXIpOiBudW1iZXI7IH1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uLy4uL3NyYy9tYXRoL2NvbnZfdXRpbCc7XG5pbXBvcnQgKiBhcyBjb252X2dwdSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9jb252X2dwdSc7XG5pbXBvcnQge0dQR1BVQ29udGV4dH0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvZ3BncHVfY29udGV4dCc7XG5pbXBvcnQgKiBhcyB0ZXN0X3V0aWwgZnJvbSAnLi4vLi4vc3JjL3Rlc3RfdXRpbCc7XG5cbmltcG9ydCB7QmVuY2htYXJrVGVzdH0gZnJvbSAnLi9iZW5jaG1hcmsnO1xuXG5jb25zdCBPUF9SVU5TID0gNDA7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgaW5wdXRTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gW3NpemUsIHNpemUsIDFdO1xuICBjb25zdCBvdXRwdXREZXB0aCA9IDE7XG4gIGNvbnN0IGZpZWxkU2l6ZSA9IDExO1xuICBjb25zdCBzdHJpZGUgPSAxO1xuICBjb25zdCB6ZXJvUGFkID0gY29udl91dGlsLmNvbXB1dGVEZWZhdWx0UGFkKGlucHV0U2hhcGVSQ0QsIGZpZWxkU2l6ZSwgc3RyaWRlKTtcbiAgY29uc3Qgb3V0cHV0U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgICAgaW5wdXRTaGFwZVJDRCwgZmllbGRTaXplLCBvdXRwdXREZXB0aCwgc3RyaWRlLCB6ZXJvUGFkKTtcblxuICBjb25zdCBpbnB1dFRleFNoYXBlUkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKGlucHV0U2hhcGVSQ0QpO1xuICBjb25zdCBvdXRwdXRUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVUZXhTaGFwZUZyb20zRChvdXRwdXRTaGFwZVJDRCk7XG4gIGNvbnN0IHdlaWdodHNUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVXZWlnaHRzVGV4U2hhcGUoXG4gICAgICBpbnB1dFNoYXBlUkNEWzJdLCBvdXRwdXREZXB0aCwgZmllbGRTaXplKTtcbiAgY29uc3QgYmlhc2VzVGV4U2hhcGVSQyA9IGNvbnZfdXRpbC5jb21wdXRlQmlhc2VzVGV4U2hhcGUob3V0cHV0RGVwdGgpO1xuXG4gIGNvbnN0IGhhc0JpYXMgPSB0cnVlO1xuICBjb25zdCBncGdwdSA9IG5ldyBHUEdQVUNvbnRleHQoKTtcbiAgY29uc3QgcHJvZ3JhbSA9IGdwZ3B1LmNyZWF0ZVByb2dyYW0oY29udl9ncHUuZ2V0RnJhZ21lbnRTaGFkZXJTb3VyY2UoXG4gICAgICBpbnB1dFNoYXBlUkNELCBvdXRwdXREZXB0aCwgZmllbGRTaXplLCBzdHJpZGUsIHplcm9QYWQsIGhhc0JpYXMpKTtcblxuICBjb25zdCBpbnB1dFRleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShpbnB1dFRleFNoYXBlUkNbMF0sIGlucHV0VGV4U2hhcGVSQ1sxXSk7XG4gIGNvbnN0IHdlaWdodHNUZXh0dXJlID1cbiAgICAgIGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUod2VpZ2h0c1RleFNoYXBlUkNbMF0sIHdlaWdodHNUZXhTaGFwZVJDWzFdKTtcbiAgY29uc3QgYmlhc2VzVGV4dHVyZSA9XG4gICAgICBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKGJpYXNlc1RleFNoYXBlUkNbMF0sIGJpYXNlc1RleFNoYXBlUkNbMV0pO1xuICBjb25zdCBvdXRwdXRUZXh0dXJlID1cbiAgICAgIGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUob3V0cHV0VGV4U2hhcGVSQ1swXSwgb3V0cHV0VGV4U2hhcGVSQ1sxXSk7XG5cbiAgY29uc3QgaW5wdXREYXRhID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShcbiAgICAgIGlucHV0VGV4U2hhcGVSQ1swXSAqIGlucHV0VGV4U2hhcGVSQ1sxXSwgLTEsIDEpO1xuICBjb25zdCB3ZWlnaHRzRGF0YSA9IHRlc3RfdXRpbC5yYW5kb21BcnJheUluUmFuZ2UoXG4gICAgICB3ZWlnaHRzVGV4U2hhcGVSQ1swXSAqIHdlaWdodHNUZXhTaGFwZVJDWzFdLCAtMSwgMSk7XG4gIGNvbnN0IGJpYXNlc0RhdGEgPSB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKFxuICAgICAgYmlhc2VzVGV4U2hhcGVSQ1swXSAqIGJpYXNlc1RleFNoYXBlUkNbMV0sIC0xLCAxKTtcblxuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICBpbnB1dFRleHR1cmUsIGlucHV0VGV4U2hhcGVSQ1swXSwgaW5wdXRUZXhTaGFwZVJDWzFdLCBpbnB1dERhdGEpO1xuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICB3ZWlnaHRzVGV4dHVyZSwgd2VpZ2h0c1RleFNoYXBlUkNbMF0sIHdlaWdodHNUZXhTaGFwZVJDWzFdLCB3ZWlnaHRzRGF0YSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvVGV4dHVyZShcbiAgICAgIGJpYXNlc1RleHR1cmUsIGJpYXNlc1RleFNoYXBlUkNbMF0sIGJpYXNlc1RleFNoYXBlUkNbMV0sIGJpYXNlc0RhdGEpO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgY29udl9ncHUuY29udm9sdmUoXG4gICAgICAgIGdwZ3B1LCBwcm9ncmFtLCBpbnB1dFRleHR1cmUsIHdlaWdodHNUZXh0dXJlLCBiaWFzZXNUZXh0dXJlLFxuICAgICAgICBvdXRwdXRUZXh0dXJlLCBvdXRwdXRUZXhTaGFwZVJDKTtcbiAgfVxuXG4gIGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUoXG4gICAgICBvdXRwdXRUZXh0dXJlLCBvdXRwdXRUZXhTaGFwZVJDWzBdLCBvdXRwdXRUZXhTaGFwZVJDWzFdKTtcbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG5cbiAgY29uc3QgYXZnVGltZSA9IChlbmQgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoaW5wdXRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZSh3ZWlnaHRzVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoYmlhc2VzVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUob3V0cHV0VGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZVByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LmRpc3Bvc2UoKTtcblxuICByZXR1cm4gYXZnVGltZTtcbn07XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi8uLi9zcmMvbWF0aC9jb252X3V0aWwnO1xuaW1wb3J0ICogYXMgY29udl9iYWNrcHJvcF9ncHUgZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvY29udl9iYWNrcHJvcF9ncHUnO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgdGVzdF91dGlsIGZyb20gJy4uLy4uL3NyYy90ZXN0X3V0aWwnO1xuXG5pbXBvcnQge0JlbmNobWFya1Rlc3R9IGZyb20gJy4vYmVuY2htYXJrJztcblxuY29uc3QgT1BfUlVOUyA9IDEwMDtcblxuZXhwb3J0IGNvbnN0IEJFTkNITUFSS19URVNUOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBjb25zdCB4U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFtzaXplLCBzaXplLCAxXTtcbiAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gMjtcbiAgY29uc3QgZmllbGRTaXplID0gMTE7XG4gIGNvbnN0IG9yaWdTdHJpZGUgPSAxO1xuICBjb25zdCBvcmlnUGFkID0gMTtcblxuICBjb25zdCBncGdwdSA9IG5ldyBHUEdQVUNvbnRleHQoKTtcbiAgZ3BncHUuZW5hYmxlQXV0b21hdGljRGVidWdWYWxpZGF0aW9uKHRydWUpO1xuICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IHhTaGFwZVJDRFsyXTtcbiAgY29uc3Qgc3JjID0gY29udl9iYWNrcHJvcF9ncHUuZ2V0RnJhZ21lbnRTaGFkZXJDb252VHJhbnNwb3NlU291cmNlKFxuICAgICAgeFNoYXBlUkNELCBmaWVsZFNpemUsIG9yaWdJbnB1dERlcHRoLCBvcmlnU3RyaWRlLCBvcmlnUGFkLCBmYWxzZSk7XG4gIGNvbnN0IHByb2dyYW0gPSBncGdwdS5jcmVhdGVQcm9ncmFtKHNyYyk7XG5cbiAgLy8gVXBsb2FkIHguXG4gIGNvbnN0IHhUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVUZXhTaGFwZUZyb20zRCh4U2hhcGVSQ0QpO1xuICBjb25zdCB4VGV4ID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZSh4VGV4U2hhcGVSQ1swXSwgeFRleFNoYXBlUkNbMV0pO1xuICBjb25zdCB4RGF0YSA9XG4gICAgICB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKHhUZXhTaGFwZVJDWzBdICogeFRleFNoYXBlUkNbMV0sIC0xLCAxKTtcbiAgZ3BncHUudXBsb2FkTWF0cml4VG9UZXh0dXJlKHhUZXgsIHhUZXhTaGFwZVJDWzBdLCB4VGV4U2hhcGVSQ1sxXSwgeERhdGEpO1xuXG4gIC8vIFVwbG9hZCB3ZWlnaHRzLlxuICBjb25zdCB3VGV4U2hhcGVSQyA9IGNvbnZfdXRpbC5jb21wdXRlV2VpZ2h0c1RleFNoYXBlKFxuICAgICAgb3JpZ0lucHV0RGVwdGgsIG9yaWdPdXRwdXREZXB0aCwgZmllbGRTaXplKTtcbiAgY29uc3Qgd0RhdGEgPVxuICAgICAgdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZSh3VGV4U2hhcGVSQ1swXSAqIHdUZXhTaGFwZVJDWzFdLCAtMSwgMSk7XG4gIGNvbnN0IHdUZXggPSBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKHdUZXhTaGFwZVJDWzBdLCB3VGV4U2hhcGVSQ1sxXSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvVGV4dHVyZSh3VGV4LCB3VGV4U2hhcGVSQ1swXSwgd1RleFNoYXBlUkNbMV0sIHdEYXRhKTtcblxuICAvLyBGaWd1cmUgb3V0IHRoZSBvdXRwdXQgc2hhcGUgYnkgZGlsYXRpbmcgdGhlIGlucHV0LlxuICBjb25zdCBkaWxhdGVkUkMgPVxuICAgICAgY29udl91dGlsLmNvbXB1dGVEaWxhdGVkUkMoW3hTaGFwZVJDRFswXSwgeFNoYXBlUkNEWzFdXSwgb3JpZ1N0cmlkZSk7XG4gIGNvbnN0IHBhZCA9IGZpZWxkU2l6ZSAtIDEgLSBvcmlnUGFkO1xuICBjb25zdCByZXN1bHRTaGFwZVJDRCA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgIFtkaWxhdGVkUkNbMF0sIGRpbGF0ZWRSQ1sxXSwgb3JpZ091dHB1dERlcHRoXSwgZmllbGRTaXplLCBvcmlnSW5wdXREZXB0aCxcbiAgICAgIDEsIHBhZCk7XG5cbiAgY29uc3QgcmVzdWx0VGV4UkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKHJlc3VsdFNoYXBlUkNEKTtcbiAgY29uc3QgcmVzdWx0VGV4ID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShyZXN1bHRUZXhSQ1swXSwgcmVzdWx0VGV4UkNbMV0pO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgY29udl9iYWNrcHJvcF9ncHUuY29udlRyYW5zcG9zZShcbiAgICAgICAgZ3BncHUsIHByb2dyYW0sIHhUZXgsIHdUZXgsIG51bGwsIHJlc3VsdFRleCwgcmVzdWx0VGV4UkMpO1xuICB9XG5cbiAgY29uc3QgeSA9IGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUoXG4gICAgICByZXN1bHRUZXgsIHJlc3VsdFRleFJDWzBdLCByZXN1bHRUZXhSQ1sxXSk7XG5cbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG5cbiAgY29uc3QgYXZnVGltZSA9IChlbmQgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUocmVzdWx0VGV4KTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZSh4VGV4KTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZSh3VGV4KTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtOREFycmF5TWF0aENQVX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvbWF0aF9jcHUnO1xuaW1wb3J0IHtBcnJheTJELCBOREFycmF5fSBmcm9tICcuLi8uLi9zcmMvbWF0aC9uZGFycmF5JztcblxuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QU19QRVJfUlVOID0gMTA7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgbWF0aCA9IG5ldyBOREFycmF5TWF0aENQVSgpO1xuICBjb25zdCBhID0gTkRBcnJheS5yYW5kVW5pZm9ybTxBcnJheTJEPihbc2l6ZSwgc2l6ZV0sIC0xLCAxKTtcbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUFNfUEVSX1JVTjsgaSsrKSB7XG4gICAgbWF0aC5sb2dTdW1FeHAoYSk7XG4gIH1cbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIHJldHVybiAoZW5kIC0gc3RhcnQpIC8gT1BTX1BFUl9SVU47XG59O1xuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge0dQR1BVQ29udGV4dH0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvZ3BncHVfY29udGV4dCc7XG5pbXBvcnQgKiBhcyBsb2dzdW1leHBfZ3B1IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2xvZ3N1bWV4cF9ncHUnO1xuaW1wb3J0ICogYXMgdGVzdF91dGlsIGZyb20gJy4uLy4uL3NyYy90ZXN0X3V0aWwnO1xuXG5pbXBvcnQge0JlbmNobWFya1Rlc3R9IGZyb20gJy4vYmVuY2htYXJrJztcblxuY29uc3QgT1BfUlVOUyA9IDEwMDtcblxuZXhwb3J0IGNvbnN0IEJFTkNITUFSS19URVNUOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBjb25zdCBncGdwdSA9IG5ldyBHUEdQVUNvbnRleHQoKTtcblxuICBjb25zdCBwcm9ncmFtID1cbiAgICAgIGdwZ3B1LmNyZWF0ZVByb2dyYW0obG9nc3VtZXhwX2dwdS5nZXRGcmFnbWVudFNoYWRlclNvdXJjZShzaXplLCBzaXplKSk7XG5cbiAgY29uc3QgYVRleHR1cmUgPSBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKHNpemUsIHNpemUpO1xuICBjb25zdCByZXN1bHRUZXh0dXJlID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShzaXplLCBzaXplKTtcblxuICBjb25zdCBhID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShzaXplICogc2l6ZSwgLTEsIDEpO1xuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoYVRleHR1cmUsIHNpemUsIHNpemUsIGEpO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgbG9nc3VtZXhwX2dwdS5sb2dTdW1FeHAoXG4gICAgICAgIGdwZ3B1LCBwcm9ncmFtLCBhVGV4dHVyZSwgc2l6ZSwgc2l6ZSwgcmVzdWx0VGV4dHVyZSk7XG4gIH1cblxuICBncGdwdS5kb3dubG9hZE1hdHJpeEZyb21UZXh0dXJlKHJlc3VsdFRleHR1cmUsIHNpemUsIHNpemUpO1xuICBjb25zdCBhdmdUaW1lID0gKHBlcmZvcm1hbmNlLm5vdygpIC0gc3RhcnQpIC8gT1BfUlVOUztcblxuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGFUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShyZXN1bHRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtCZW5jaG1hcmtSdW4sIEJlbmNobWFya1J1bkdyb3VwfSBmcm9tICcuL2JlbmNobWFyayc7XG5pbXBvcnQgKiBhcyBjb252X2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9jb252X2dwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgY29udl90cmFuc3Bvc2VfZ3B1X2JlbmNobWFyayBmcm9tICcuL2NvbnZfdHJhbnNwb3NlX2dwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgbG9nc3VtZXhwX2NwdV9iZW5jaG1hcmsgZnJvbSAnLi9sb2dzdW1leHBfY3B1X2JlbmNobWFyayc7XG5pbXBvcnQgKiBhcyBsb2dzdW1leHBfZ3B1X2JlbmNobWFyayBmcm9tICcuL2xvZ3N1bWV4cF9ncHVfYmVuY2htYXJrJztcbmltcG9ydCAqIGFzIG1heF9wb29sX2JhY2twcm9wX2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9tYXhfcG9vbF9iYWNrcHJvcF9ncHVfYmVuY2htYXJrJztcbmltcG9ydCAqIGFzIG1heF9wb29sX2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9tYXhfcG9vbF9ncHVfYmVuY2htYXJrJztcbmltcG9ydCAqIGFzIG11bG1hdF9jcHVfYmVuY2htYXJrIGZyb20gJy4vbXVsbWF0X2NwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgbXVsbWF0X2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9tdWxtYXRfZ3B1X2JlbmNobWFyayc7XG5pbXBvcnQgKiBhcyB0ZXhfdXRpbF9iZW5jaG1hcmsgZnJvbSAnLi90ZXhfdXRpbF9iZW5jaG1hcmsnO1xuXG5leHBvcnQgY29uc3QgQkVOQ0hNQVJLX1JVTl9HUk9VUFM6IEJlbmNobWFya1J1bkdyb3VwW10gPSBbXG4gIHtcbiAgICBuYW1lOlxuICAgICAgICAnTWF0cml4IE11bHRpcGxpY2F0aW9uIChDUFUgdnMgR1BVKTogbWF0bXVsKFtzaXplLCBzaXplXSwgW3NpemUsIHNpemVdKScsXG4gICAgbWluOiAwLFxuICAgIG1heDogMTAyNCxcbiAgICBzdGVwU2l6ZTogNjQsXG4gICAgc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uOiAoc3RlcDogbnVtYmVyKSA9PiBNYXRoLm1heCgxLCBzdGVwKSxcbiAgICBiZW5jaG1hcmtSdW5zOiBbXG4gICAgICBuZXcgQmVuY2htYXJrUnVuKCdtdWxtYXRfZ3B1JywgbXVsbWF0X2dwdV9iZW5jaG1hcmsuQkVOQ0hNQVJLX1RFU1QpLFxuICAgICAgbmV3IEJlbmNobWFya1J1bignbXVsbWF0X2NwdScsIG11bG1hdF9jcHVfYmVuY2htYXJrLkJFTkNITUFSS19URVNUKVxuICAgIF0sXG4gIH0sXG4gIHtcbiAgICBuYW1lOiAnQ29udm9sdXRpb24gKEdQVSk6IGNvbnYgb3ZlciBpbWFnZSBbc2l6ZSwgc2l6ZSwgMV0nLFxuICAgIG1pbjogMCxcbiAgICBtYXg6IDEwMjQsXG4gICAgc3RlcFNpemU6IDY0LFxuICAgIHN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbjogKHN0ZXA6IG51bWJlcikgPT4gTWF0aC5tYXgoMSwgc3RlcCksXG4gICAgYmVuY2htYXJrUnVuczogW25ldyBCZW5jaG1hcmtSdW4oXG4gICAgICAgICdkMT0xLCBkMj0xLCBmPTExLCBzPTEnLCBjb252X2dwdV9iZW5jaG1hcmsuQkVOQ0hNQVJLX1RFU1QpXSxcbiAgfSxcbiAge1xuICAgIG5hbWU6ICdDb252b2x1dGlvbiBUcmFuc3Bvc2VkIChHUFUpOiBkZWNvbnYgb3ZlciBpbWFnZSBbc2l6ZSwgc2l6ZSwgMV0nLFxuICAgIG1pbjogMCxcbiAgICBtYXg6IDEwMjQsXG4gICAgc3RlcFNpemU6IDY0LFxuICAgIHN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbjogKHN0ZXA6IG51bWJlcikgPT4gTWF0aC5tYXgoMSwgc3RlcCksXG4gICAgYmVuY2htYXJrUnVuczogW25ldyBCZW5jaG1hcmtSdW4oXG4gICAgICAgICdkMT0xLCBkMj0xLCBmPTExLCBzPTEnLCBjb252X3RyYW5zcG9zZV9ncHVfYmVuY2htYXJrLkJFTkNITUFSS19URVNUKV0sXG4gIH0sXG4gIHtcbiAgICBuYW1lOiAnTWF4IHBvb2wgKEdQVSknLFxuICAgIG1pbjogMCxcbiAgICBtYXg6IDEwMjQsXG4gICAgc3RlcFNpemU6IDY0LFxuICAgIHN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbjogKHN0ZXA6IG51bWJlcikgPT4gTWF0aC5tYXgoMSwgc3RlcCksXG4gICAgYmVuY2htYXJrUnVuczogW25ldyBCZW5jaG1hcmtSdW4oXG4gICAgICAgICdkMT0xLCBkMj0xLCBmPTExLCBzPTEnLFxuICAgICAgICBtYXhfcG9vbF9ncHVfYmVuY2htYXJrLk1BWF9QT09MX0JFTkNITUFSS19URVNUKV0sXG4gIH0sXG4gIHtcbiAgICBuYW1lOiAnTG9nU3VtRXhwIChDUFUgdnMgR1BVKTogaW5wdXQgW3NpemUsIHNpemVdJyxcbiAgICBtaW46IDAsXG4gICAgbWF4OiAxMDI0LFxuICAgIHN0ZXBTaXplOiA2NCxcbiAgICBzdGVwVG9TaXplVHJhbnNmb3JtYXRpb246IChzdGVwOiBudW1iZXIpID0+IE1hdGgubWF4KDEsIHN0ZXApLFxuICAgIGJlbmNobWFya1J1bnM6IFtcbiAgICAgIG5ldyBCZW5jaG1hcmtSdW4oJ2xvZ3N1bWV4cF9ncHUnLCBsb2dzdW1leHBfZ3B1X2JlbmNobWFyay5CRU5DSE1BUktfVEVTVCksXG4gICAgICBuZXcgQmVuY2htYXJrUnVuKCdsb2dzdW1leHBfY3B1JywgbG9nc3VtZXhwX2NwdV9iZW5jaG1hcmsuQkVOQ0hNQVJLX1RFU1QpXG4gICAgXSxcbiAgfVxuXTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICcuLi9kZW1vLWhlYWRlcic7XG5pbXBvcnQgJy4uL2RlbW8tZm9vdGVyJztcblxuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLXVudXNlZC12YXJpYWJsZVxuaW1wb3J0IHtQb2x5bWVyRWxlbWVudCwgUG9seW1lckhUTUxFbGVtZW50fSBmcm9tICcuLi9wb2x5bWVyLXNwZWMnO1xuaW1wb3J0IHtCZW5jaG1hcmtSdW5Hcm91cH0gZnJvbSAnLi9iZW5jaG1hcmsnO1xuXG5pbXBvcnQge0JFTkNITUFSS19SVU5fR1JPVVBTfSBmcm9tICcuL21hdGgtYmVuY2htYXJrLXJ1bi1ncm91cHMnO1xuXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6dmFyaWFibGUtbmFtZVxuZXhwb3J0IGxldCBNYXRoQmVuY2htYXJrUG9seW1lciA9IFBvbHltZXJFbGVtZW50KFxuICAgIHtpczogJ21hdGgtYmVuY2htYXJrJywgcHJvcGVydGllczoge2JlbmNobWFya1J1bkdyb3VwTmFtZXM6IEFycmF5fX0pO1xuXG5leHBvcnQgY2xhc3MgTWF0aEJlbmNobWFyayBleHRlbmRzIE1hdGhCZW5jaG1hcmtQb2x5bWVyIHtcbiAgLy8gUG9seW1lciBwcm9wZXJ0aWVzLlxuICBwcml2YXRlIGJlbmNobWFya1J1bkdyb3VwTmFtZXM6IHN0cmluZ1tdO1xuICBwcml2YXRlIHN0b3BNZXNzYWdlczogYm9vbGVhbltdO1xuXG4gIHJlYWR5KCkge1xuICAgIC8vIFNldCB1cCB0aGUgYmVuY2htYXJrcyBVSS5cbiAgICBjb25zdCBiZW5jaG1hcmtSdW5Hcm91cE5hbWVzOiBzdHJpbmdbXSA9IFtdO1xuICAgIHRoaXMuc3RvcE1lc3NhZ2VzID0gW107XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBCRU5DSE1BUktfUlVOX0dST1VQUy5sZW5ndGg7IGkrKykge1xuICAgICAgYmVuY2htYXJrUnVuR3JvdXBOYW1lcy5wdXNoKEJFTkNITUFSS19SVU5fR1JPVVBTW2ldLm5hbWUpO1xuICAgICAgdGhpcy5zdG9wTWVzc2FnZXMucHVzaChmYWxzZSk7XG4gICAgfVxuICAgIHRoaXMuYmVuY2htYXJrUnVuR3JvdXBOYW1lcyA9IGJlbmNobWFya1J1bkdyb3VwTmFtZXM7XG5cbiAgICAvLyBJbiBhIHNldFRpbWVvdXQgdG8gbGV0IHRoZSBVSSB1cGRhdGUgYmVmb3JlIHdlIGFkZCBldmVudCBsaXN0ZW5lcnMuXG4gICAgc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICBjb25zdCBydW5CdXR0b25zID0gdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLXRlc3QnKTtcbiAgICAgIGNvbnN0IHN0b3BCdXR0b25zID0gdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLXN0b3AnKTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcnVuQnV0dG9ucy5sZW5ndGg7IGkrKykge1xuICAgICAgICBydW5CdXR0b25zW2ldLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4ge1xuICAgICAgICAgIHRoaXMucnVuQmVuY2htYXJrR3JvdXAoaSk7XG4gICAgICAgIH0pO1xuICAgICAgICBzdG9wQnV0dG9uc1tpXS5hZGRFdmVudExpc3RlbmVyKCdjbGljaycsICgpID0+IHtcbiAgICAgICAgICB0aGlzLnN0b3BNZXNzYWdlc1tpXSA9IHRydWU7XG4gICAgICAgIH0pO1xuICAgICAgfVxuICAgIH0sIDApO1xuICB9XG5cbiAgcHJpdmF0ZSBydW5CZW5jaG1hcmtHcm91cChiZW5jaG1hcmtSdW5Hcm91cEluZGV4OiBudW1iZXIpIHtcbiAgICBjb25zdCBiZW5jaG1hcmtSdW5Hcm91cCA9IEJFTkNITUFSS19SVU5fR1JPVVBTW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdO1xuXG4gICAgY29uc3QgY2FudmFzID0gdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLXBsb3QnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICBIVE1MQ2FudmFzRWxlbWVudDtcbiAgICBjb25zdCBjb250ZXh0ID0gY2FudmFzLmdldENvbnRleHQoJzJkJykgYXMgQ2FudmFzUmVuZGVyaW5nQ29udGV4dDJEO1xuXG4gICAgY29uc3QgZGF0YXNldHM6IENoYXJ0RGF0YVNldHNbXSA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVucy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgaHVlID0gTWF0aC5mbG9vcigzNjAgKiBpIC8gYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVucy5sZW5ndGgpO1xuICAgICAgZGF0YXNldHMucHVzaCh7XG4gICAgICAgIGRhdGE6IGJlbmNobWFya1J1bkdyb3VwLmJlbmNobWFya1J1bnNbaV0uY2hhcnREYXRhLFxuICAgICAgICBmaWxsOiBmYWxzZSxcbiAgICAgICAgbGFiZWw6IGJlbmNobWFya1J1bkdyb3VwLmJlbmNobWFya1J1bnNbaV0ubmFtZSxcbiAgICAgICAgYm9yZGVyQ29sb3I6ICdoc2woJyArIGh1ZSArICcsIDEwMCUsIDQwJSknLFxuICAgICAgICBiYWNrZ3JvdW5kQ29sb3I6ICdoc2woJyArIGh1ZSArICcsIDEwMCUsIDcwJSknLFxuICAgICAgICBwb2ludFJhZGl1czogMCxcbiAgICAgICAgcG9pbnRIaXRSYWRpdXM6IDUsXG4gICAgICAgIGJvcmRlcldpZHRoOiAxLFxuICAgICAgICBsaW5lVGVuc2lvbjogMFxuICAgICAgfSk7XG4gICAgfVxuXG4gICAgY29uc3QgY2hhcnQgPSBuZXcgQ2hhcnQoY29udGV4dCwge1xuICAgICAgdHlwZTogJ2xpbmUnLFxuICAgICAgZGF0YToge2RhdGFzZXRzfSxcbiAgICAgIG9wdGlvbnM6IHtcbiAgICAgICAgYW5pbWF0aW9uOiB7ZHVyYXRpb246IDB9LFxuICAgICAgICByZXNwb25zaXZlOiBmYWxzZSxcbiAgICAgICAgc2NhbGVzOiB7XG4gICAgICAgICAgeEF4ZXM6IFt7XG4gICAgICAgICAgICB0eXBlOiAnbGluZWFyJyxcbiAgICAgICAgICAgIHBvc2l0aW9uOiAnYm90dG9tJyxcbiAgICAgICAgICAgIHRpY2tzOiB7XG4gICAgICAgICAgICAgIG1pbjogYmVuY2htYXJrUnVuR3JvdXAubWluLFxuICAgICAgICAgICAgICBtYXg6IGJlbmNobWFya1J1bkdyb3VwLm1heCxcbiAgICAgICAgICAgICAgc3RlcFNpemU6IGJlbmNobWFya1J1bkdyb3VwLnN0ZXBTaXplLFxuICAgICAgICAgICAgICBjYWxsYmFjazogKGxhYmVsOiBzdHJpbmcpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gYmVuY2htYXJrUnVuR3JvdXAuc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uICE9IG51bGwgP1xuICAgICAgICAgICAgICAgICAgICBiZW5jaG1hcmtSdW5Hcm91cC5zdGVwVG9TaXplVHJhbnNmb3JtYXRpb24oK2xhYmVsKSA6XG4gICAgICAgICAgICAgICAgICAgICtsYWJlbDtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICAgICAgICB9IGFzIGFueSAgLy8gTm90ZTogdGhlIHR5cGluZ3MgZm9yIHRoaXMgYXJlIGluY29ycmVjdCwgY2FzdCBhcyBhbnkuXG4gICAgICAgICAgfV0sXG4gICAgICAgICAgeUF4ZXM6IFt7XG4gICAgICAgICAgICB0aWNrczoge1xuICAgICAgICAgICAgICBjYWxsYmFjazogKGxhYmVsLCBpbmRleCwgbGFiZWxzKSA9PiB7XG4gICAgICAgICAgICAgICAgcmV0dXJuIGxhYmVsICsgJ21zJztcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfSxcbiAgICAgICAgICB9XVxuICAgICAgICB9LFxuICAgICAgICB0b29sdGlwczoge21vZGU6ICdsYWJlbCd9LFxuICAgICAgICB0aXRsZToge3RleHQ6IGJlbmNobWFya1J1bkdyb3VwLm5hbWV9XG4gICAgICB9XG4gICAgfSk7XG4gICAgY2FudmFzLnN0eWxlLmRpc3BsYXkgPSAnbm9uZSc7XG5cbiAgICBjb25zdCBydW5NZXNzYWdlID1cbiAgICAgICAgdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLW1lc3NhZ2UnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICBIVE1MRWxlbWVudDtcbiAgICBydW5NZXNzYWdlLnN0eWxlLmRpc3BsYXkgPSAnYmxvY2snO1xuXG4gICAgY29uc3QgcnVuTnVtYmVyc1RhYmxlID1cbiAgICAgICAgdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLW51bWJlcnMtdGFibGUnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICBIVE1MRWxlbWVudDtcbiAgICBydW5OdW1iZXJzVGFibGUuaW5uZXJIVE1MID0gJyc7XG4gICAgcnVuTnVtYmVyc1RhYmxlLnN0eWxlLmRpc3BsYXkgPSAnbm9uZSc7XG5cbiAgICAvLyBTZXQgdXAgdGhlIGhlYWRlciBmb3IgdGhlIHRhYmxlLlxuICAgIGNvbnN0IGhlYWRlcnMgPSBbJ3NpemUnXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGJlbmNobWFya1J1bkdyb3VwLmJlbmNobWFya1J1bnMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGhlYWRlcnMucHVzaChiZW5jaG1hcmtSdW5Hcm91cC5iZW5jaG1hcmtSdW5zW2ldLm5hbWUpO1xuICAgIH1cbiAgICBydW5OdW1iZXJzVGFibGUuYXBwZW5kQ2hpbGQodGhpcy5idWlsZFJ1bk51bWJlcnNSb3coaGVhZGVycykpO1xuXG4gICAgdGhpcy5ydW5CZW5jaG1hcmtTdGVwcyhcbiAgICAgICAgY2hhcnQsIGJlbmNobWFya1J1bkdyb3VwLCBiZW5jaG1hcmtSdW5Hcm91cEluZGV4LFxuICAgICAgICBiZW5jaG1hcmtSdW5Hcm91cC5taW4pO1xuICB9XG5cbiAgcHJpdmF0ZSBidWlsZFJ1bk51bWJlcnNSb3codmFsdWVzOiBzdHJpbmdbXSkge1xuICAgIGNvbnN0IHJ1bk51bWJlclJvd0VsZW1lbnQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTtcbiAgICBydW5OdW1iZXJSb3dFbGVtZW50LmNsYXNzTmFtZSA9ICdydW4tbnVtYmVycy1yb3cgbWF0aC1iZW5jaG1hcmsnO1xuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGNvbnN0IHJ1bk51bWJlckNlbGxFbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnZGl2Jyk7XG4gICAgICBydW5OdW1iZXJDZWxsRWxlbWVudC5jbGFzc05hbWUgPSAncnVuLW51bWJlcnMtY2VsbCBtYXRoLWJlbmNobWFyayc7XG4gICAgICBydW5OdW1iZXJDZWxsRWxlbWVudC5pbm5lclRleHQgPSB2YWx1ZXNbaV07XG4gICAgICBydW5OdW1iZXJSb3dFbGVtZW50LmFwcGVuZENoaWxkKHJ1bk51bWJlckNlbGxFbGVtZW50KTtcbiAgICB9XG4gICAgcmV0dXJuIHJ1bk51bWJlclJvd0VsZW1lbnQ7XG4gIH1cblxuICBwcml2YXRlIHJ1bkJlbmNobWFya1N0ZXBzKFxuICAgICAgY2hhcnQ6IENoYXJ0LCBiZW5jaG1hcmtSdW5Hcm91cDogQmVuY2htYXJrUnVuR3JvdXAsXG4gICAgICBiZW5jaG1hcmtSdW5Hcm91cEluZGV4OiBudW1iZXIsIHN0ZXA6IG51bWJlcikge1xuICAgIGNvbnN0IHJ1bk51bWJlcnNUYWJsZSA9XG4gICAgICAgIHRoaXMucXVlcnlTZWxlY3RvckFsbCgnLnJ1bi1udW1iZXJzLXRhYmxlJylbYmVuY2htYXJrUnVuR3JvdXBJbmRleF0gYXNcbiAgICAgICAgSFRNTEVsZW1lbnQ7XG4gICAgaWYgKHN0ZXAgPiBiZW5jaG1hcmtSdW5Hcm91cC5tYXggfHxcbiAgICAgICAgdGhpcy5zdG9wTWVzc2FnZXNbYmVuY2htYXJrUnVuR3JvdXBJbmRleF0pIHtcbiAgICAgIHRoaXMuc3RvcE1lc3NhZ2VzW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdID0gZmFsc2U7XG5cbiAgICAgIHJ1bk51bWJlcnNUYWJsZS5zdHlsZS5kaXNwbGF5ID0gJyc7XG5cbiAgICAgIGNvbnN0IGNhbnZhcyA9XG4gICAgICAgICAgdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLXBsb3QnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICAgIEhUTUxDYW52YXNFbGVtZW50O1xuICAgICAgY2FudmFzLnN0eWxlLmRpc3BsYXkgPSAnYmxvY2snO1xuICAgICAgY2hhcnQudXBkYXRlKCk7XG5cbiAgICAgIGNvbnN0IHJ1bk1lc3NhZ2UgPVxuICAgICAgICAgIHRoaXMucXVlcnlTZWxlY3RvckFsbCgnLnJ1bi1tZXNzYWdlJylbYmVuY2htYXJrUnVuR3JvdXBJbmRleF0gYXNcbiAgICAgICAgICBIVE1MRWxlbWVudDtcbiAgICAgIHJ1bk1lc3NhZ2Uuc3R5bGUuZGlzcGxheSA9ICdub25lJztcblxuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IHJ1bk51bWJlclJvd0VsZW1lbnQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTtcbiAgICBydW5OdW1iZXJSb3dFbGVtZW50LmNsYXNzTmFtZSA9ICdydW4tbnVtYmVycy1yb3cgbWF0aC1iZW5jaG1hcmsnO1xuXG4gICAgY29uc3Qgcm93VmFsdWVzOiBzdHJpbmdbXSA9IFsnJyArIHN0ZXBdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVucy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgYmVuY2htYXJrUnVuID0gYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVuc1tpXTtcbiAgICAgIGNvbnN0IGJlbmNobWFya1Rlc3QgPSBiZW5jaG1hcmtSdW4uYmVuY2htYXJrVGVzdDtcblxuICAgICAgY29uc3Qgc2l6ZSA9IGJlbmNobWFya1J1bkdyb3VwLnN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbiAhPSBudWxsID9cbiAgICAgICAgICBiZW5jaG1hcmtSdW5Hcm91cC5zdGVwVG9TaXplVHJhbnNmb3JtYXRpb24oc3RlcCkgOlxuICAgICAgICAgIHN0ZXA7XG5cbiAgICAgIGxldCByZXN1bHRTdHJpbmc6IHN0cmluZztcbiAgICAgIGxldCBsb2dTdHJpbmc6IHN0cmluZztcbiAgICAgIGxldCB0aW1lID0gMDtcbiAgICAgIGxldCBzdWNjZXNzID0gdHJ1ZTtcblxuICAgICAgdHJ5IHtcbiAgICAgICAgdGltZSA9IGJlbmNobWFya1Rlc3Qoc2l6ZSk7XG4gICAgICAgIHJlc3VsdFN0cmluZyA9IHRpbWUudG9GaXhlZCgzKSArICdtcyc7XG4gICAgICAgIGxvZ1N0cmluZyA9IHJlc3VsdFN0cmluZztcbiAgICAgIH0gY2F0Y2ggKGUpIHtcbiAgICAgICAgc3VjY2VzcyA9IGZhbHNlO1xuICAgICAgICByZXN1bHRTdHJpbmcgPSAnRXJyb3InO1xuICAgICAgICBsb2dTdHJpbmcgPSBlLm1lc3NhZ2U7XG4gICAgICB9XG5cbiAgICAgIGlmICh0aW1lID49IDApIHtcbiAgICAgICAgaWYgKHN1Y2Nlc3MpIHtcbiAgICAgICAgICBiZW5jaG1hcmtSdW4uY2hhcnREYXRhLnB1c2goe3g6IHN0ZXAsIHk6IHRpbWV9KTtcbiAgICAgICAgfVxuICAgICAgICByb3dWYWx1ZXMucHVzaChyZXN1bHRTdHJpbmcpO1xuICAgICAgfVxuICAgICAgY29uc29sZS5sb2coYmVuY2htYXJrUnVuLm5hbWUgKyAnWycgKyBzdGVwICsgJ106ICcgKyBsb2dTdHJpbmcpO1xuICAgIH1cbiAgICBydW5OdW1iZXJzVGFibGUuYXBwZW5kQ2hpbGQodGhpcy5idWlsZFJ1bk51bWJlcnNSb3cocm93VmFsdWVzKSk7XG5cbiAgICBzdGVwICs9IGJlbmNobWFya1J1bkdyb3VwLnN0ZXBTaXplO1xuICAgIC8vIEFsbG93IHRoZSBVSSB0byB1cGRhdGUuXG4gICAgc2V0VGltZW91dChcbiAgICAgICAgKCkgPT4gdGhpcy5ydW5CZW5jaG1hcmtTdGVwcyhcbiAgICAgICAgICAgIGNoYXJ0LCBiZW5jaG1hcmtSdW5Hcm91cCwgYmVuY2htYXJrUnVuR3JvdXBJbmRleCwgc3RlcCksXG4gICAgICAgIDEwMCk7XG4gIH1cbn1cbmRvY3VtZW50LnJlZ2lzdGVyRWxlbWVudChNYXRoQmVuY2htYXJrLnByb3RvdHlwZS5pcywgTWF0aEJlbmNobWFyayk7XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi8uLi9zcmMvbWF0aC9jb252X3V0aWwnO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgbWF4X3Bvb2xfZ3B1IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL21heF9wb29sX2dwdSc7XG5pbXBvcnQgKiBhcyB0ZXN0X3V0aWwgZnJvbSAnLi4vLi4vc3JjL3Rlc3RfdXRpbCc7XG5cbmltcG9ydCB7QmVuY2htYXJrVGVzdH0gZnJvbSAnLi9iZW5jaG1hcmsnO1xuXG5jb25zdCBPUF9SVU5TID0gNDA7XG5cbmV4cG9ydCBjb25zdCBNQVhfUE9PTF9CRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgaW5wdXRTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gW3NpemUsIHNpemUsIDFdO1xuICBjb25zdCBvdXRwdXREZXB0aCA9IDE7XG4gIGNvbnN0IGZpZWxkU2l6ZSA9IDExO1xuICBjb25zdCBzdHJpZGUgPSAxO1xuICBjb25zdCB6ZXJvUGFkID0gY29udl91dGlsLmNvbXB1dGVEZWZhdWx0UGFkKGlucHV0U2hhcGVSQ0QsIGZpZWxkU2l6ZSwgc3RyaWRlKTtcbiAgY29uc3Qgb3V0cHV0U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgICAgaW5wdXRTaGFwZVJDRCwgZmllbGRTaXplLCBvdXRwdXREZXB0aCwgc3RyaWRlLCB6ZXJvUGFkKTtcblxuICBjb25zdCBpbnB1dFRleFNoYXBlUkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKGlucHV0U2hhcGVSQ0QpO1xuICBjb25zdCBvdXRwdXRUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVUZXhTaGFwZUZyb20zRChvdXRwdXRTaGFwZVJDRCk7XG5cbiAgY29uc3QgZ3BncHUgPSBuZXcgR1BHUFVDb250ZXh0KCk7XG4gIGNvbnN0IHByb2dyYW0gPVxuICAgICAgZ3BncHUuY3JlYXRlUHJvZ3JhbShtYXhfcG9vbF9ncHUuZ2V0RnJhZ21lbnRTaGFkZXJNYXhQb29sU291cmNlKFxuICAgICAgICAgIGlucHV0U2hhcGVSQ0QsIGZpZWxkU2l6ZSwgc3RyaWRlLCB6ZXJvUGFkKSk7XG5cbiAgY29uc3QgaW5wdXRUZXh0dXJlID1cbiAgICAgIGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUoaW5wdXRUZXhTaGFwZVJDWzBdLCBpbnB1dFRleFNoYXBlUkNbMV0pO1xuICBjb25zdCBvdXRwdXRUZXh0dXJlID1cbiAgICAgIGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUob3V0cHV0VGV4U2hhcGVSQ1swXSwgb3V0cHV0VGV4U2hhcGVSQ1sxXSk7XG5cbiAgY29uc3QgaW5wdXREYXRhID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShcbiAgICAgIGlucHV0VGV4U2hhcGVSQ1swXSAqIGlucHV0VGV4U2hhcGVSQ1sxXSwgLTEsIDEpO1xuXG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvVGV4dHVyZShcbiAgICAgIGlucHV0VGV4dHVyZSwgaW5wdXRUZXhTaGFwZVJDWzBdLCBpbnB1dFRleFNoYXBlUkNbMV0sIGlucHV0RGF0YSk7XG5cbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUF9SVU5TOyBpKyspIHtcbiAgICBtYXhfcG9vbF9ncHUubWF4UG9vbENvbW1vbihcbiAgICAgICAgZ3BncHUsIHByb2dyYW0sIGlucHV0VGV4dHVyZSwgb3V0cHV0VGV4dHVyZSwgb3V0cHV0VGV4U2hhcGVSQyk7XG4gIH1cblxuICBncGdwdS5kb3dubG9hZE1hdHJpeEZyb21UZXh0dXJlKFxuICAgICAgb3V0cHV0VGV4dHVyZSwgb3V0cHV0VGV4U2hhcGVSQ1swXSwgb3V0cHV0VGV4U2hhcGVSQ1sxXSk7XG4gIGNvbnN0IGVuZCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuXG4gIGNvbnN0IGF2Z1RpbWUgPSAoZW5kIC0gc3RhcnQpIC8gT1BfUlVOUztcblxuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGlucHV0VGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUob3V0cHV0VGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZVByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LmRpc3Bvc2UoKTtcblxuICByZXR1cm4gYXZnVGltZTtcbn07XG5cbmV4cG9ydCBjb25zdCBNQVhfUE9PTF9QT1NOU19CRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgaW5wdXRTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gW3NpemUsIHNpemUsIDFdO1xuICBjb25zdCBvdXRwdXREZXB0aCA9IDE7XG4gIGNvbnN0IGZpZWxkU2l6ZSA9IDExO1xuICBjb25zdCBzdHJpZGUgPSAxO1xuICBjb25zdCB6ZXJvUGFkID0gY29udl91dGlsLmNvbXB1dGVEZWZhdWx0UGFkKGlucHV0U2hhcGVSQ0QsIGZpZWxkU2l6ZSwgc3RyaWRlKTtcbiAgY29uc3Qgb3V0cHV0U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9XG4gICAgICBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgICAgaW5wdXRTaGFwZVJDRCwgZmllbGRTaXplLCBvdXRwdXREZXB0aCwgc3RyaWRlLCB6ZXJvUGFkKTtcblxuICBjb25zdCBpbnB1dFRleFNoYXBlUkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKGlucHV0U2hhcGVSQ0QpO1xuICBjb25zdCBvdXRwdXRUZXhTaGFwZVJDID0gY29udl91dGlsLmNvbXB1dGVUZXhTaGFwZUZyb20zRChvdXRwdXRTaGFwZVJDRCk7XG5cbiAgY29uc3QgZ3BncHUgPSBuZXcgR1BHUFVDb250ZXh0KCk7XG4gIGNvbnN0IHByb2dyYW06IFdlYkdMUHJvZ3JhbSA9XG4gICAgICBncGdwdS5jcmVhdGVQcm9ncmFtKG1heF9wb29sX2dwdS5nZXRGcmFnbWVudFNoYWRlck1heFBvb2xQb3NpdGlvbnNTb3VyY2UoXG4gICAgICAgICAgaW5wdXRTaGFwZVJDRCwgZmllbGRTaXplLCBzdHJpZGUsIHplcm9QYWQpKTtcblxuICBjb25zdCBpbnB1dFRleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShpbnB1dFRleFNoYXBlUkNbMF0sIGlucHV0VGV4U2hhcGVSQ1sxXSk7XG4gIGNvbnN0IG91dHB1dFRleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShvdXRwdXRUZXhTaGFwZVJDWzBdLCBvdXRwdXRUZXhTaGFwZVJDWzFdKTtcblxuICBjb25zdCBpbnB1dERhdGEgPSB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKFxuICAgICAgaW5wdXRUZXhTaGFwZVJDWzBdICogaW5wdXRUZXhTaGFwZVJDWzFdLCAtMSwgMSk7XG5cbiAgZ3BncHUudXBsb2FkTWF0cml4VG9UZXh0dXJlKFxuICAgICAgaW5wdXRUZXh0dXJlLCBpbnB1dFRleFNoYXBlUkNbMF0sIGlucHV0VGV4U2hhcGVSQ1sxXSwgaW5wdXREYXRhKTtcblxuICBjb25zdCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IE9QX1JVTlM7IGkrKykge1xuICAgIG1heF9wb29sX2dwdS5tYXhQb29sQ29tbW9uKFxuICAgICAgICBncGdwdSwgcHJvZ3JhbSwgaW5wdXRUZXh0dXJlLCBvdXRwdXRUZXh0dXJlLCBvdXRwdXRUZXhTaGFwZVJDKTtcbiAgfVxuXG4gIGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUoXG4gICAgICBvdXRwdXRUZXh0dXJlLCBvdXRwdXRUZXhTaGFwZVJDWzBdLCBvdXRwdXRUZXhTaGFwZVJDWzFdKTtcbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG5cbiAgY29uc3QgYXZnVGltZSA9IChlbmQgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoaW5wdXRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShvdXRwdXRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTsiLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7TkRBcnJheU1hdGhDUFV9IGZyb20gJy4uLy4uL3NyYy9tYXRoL21hdGhfY3B1JztcbmltcG9ydCB7QXJyYXkyRCwgTkRBcnJheX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvbmRhcnJheSc7XG5cbmltcG9ydCB7QmVuY2htYXJrVGVzdH0gZnJvbSAnLi9iZW5jaG1hcmsnO1xuXG5jb25zdCBPUFNfUEVSX1NNQUxMX1JVTiA9IDE7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgaWYgKHNpemUgPiA1MTIpIHtcbiAgICByZXR1cm4gLTE7XG4gIH1cbiAgY29uc3QgbWF0aCA9IG5ldyBOREFycmF5TWF0aENQVSgpO1xuICBjb25zdCBhID0gTkRBcnJheS5yYW5kVW5pZm9ybTxBcnJheTJEPihbc2l6ZSwgc2l6ZV0sIC0xLCAxKTtcbiAgY29uc3QgYiA9IE5EQXJyYXkucmFuZFVuaWZvcm08QXJyYXkyRD4oW3NpemUsIHNpemVdLCAtMSwgMSk7XG4gIGNvbnN0IHJ1bnMgPSAoc2l6ZSA8IDE5MikgPyBPUFNfUEVSX1NNQUxMX1JVTiA6IDE7XG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcnVuczsgaSsrKSB7XG4gICAgbWF0aC5tYXRNdWwoYSwgYik7XG4gIH1cbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIHJldHVybiAoZW5kIC0gc3RhcnQpIC8gcnVucztcbn07XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7TWF0cml4T3JpZW50YXRpb259IGZyb20gJy4uLy4uL3NyYy9tYXRoL21hdGgnO1xuaW1wb3J0IHtBcnJheTJEfSBmcm9tICcuLi8uLi9zcmMvbWF0aC9uZGFycmF5JztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9ncGdwdV9jb250ZXh0JztcbmltcG9ydCAqIGFzIG11bG1hdF9ncHUgZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvbXVsbWF0X2dwdSc7XG5pbXBvcnQgKiBhcyBtdWxtYXRfcGFja2VkX2dwdSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9tdWxtYXRfcGFja2VkX2dwdSc7XG5pbXBvcnQgKiBhcyB0ZXN0X3V0aWwgZnJvbSAnLi4vLi4vc3JjL3Rlc3RfdXRpbCc7XG5cbmltcG9ydCB7QmVuY2htYXJrVGVzdH0gZnJvbSAnLi9iZW5jaG1hcmsnO1xuXG5jb25zdCBPUF9SVU5TID0gNDA7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgZ3BncHUgPSBuZXcgR1BHUFVDb250ZXh0KCk7XG4gIGNvbnN0IGFUZXh0dXJlID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShzaXplLCBzaXplKTtcbiAgY29uc3QgYlRleHR1cmUgPSBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKHNpemUsIHNpemUpO1xuICBjb25zdCByZXN1bHRUZXh0dXJlID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShzaXplLCBzaXplKTtcblxuICBjb25zdCBhQXJyID0gbmV3IEFycmF5MkQoXG4gICAgICBbc2l6ZSwgc2l6ZV0sIHt0ZXh0dXJlOiBhVGV4dHVyZSwgdGV4dHVyZVNoYXBlUkM6IFtzaXplLCBzaXplXX0pO1xuICBjb25zdCBiQXJyID0gbmV3IEFycmF5MkQoXG4gICAgICBbc2l6ZSwgc2l6ZV0sIHt0ZXh0dXJlOiBiVGV4dHVyZSwgdGV4dHVyZVNoYXBlUkM6IFtzaXplLCBzaXplXX0pO1xuICBjb25zdCByZXNBcnIgPSBuZXcgQXJyYXkyRChcbiAgICAgIFtzaXplLCBzaXplXSwge3RleHR1cmU6IHJlc3VsdFRleHR1cmUsIHRleHR1cmVTaGFwZVJDOiBbc2l6ZSwgc2l6ZV19KTtcbiAgY29uc3QgcHJvZ3JhbSA9IGdwZ3B1LmNyZWF0ZVByb2dyYW0obXVsbWF0X2dwdS5nZXRGcmFnbWVudFNoYWRlcihcbiAgICAgIGFBcnIsIGJBcnIsIHJlc0FyciwgTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUixcbiAgICAgIE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpKTtcblxuICBjb25zdCBhID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShzaXplICogc2l6ZSwgLTEsIDEpO1xuICBjb25zdCBiID0gdGVzdF91dGlsLnJhbmRvbUFycmF5SW5SYW5nZShzaXplICogc2l6ZSwgLTEsIDEpO1xuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoYVRleHR1cmUsIHNpemUsIHNpemUsIGEpO1xuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoYlRleHR1cmUsIHNpemUsIHNpemUsIGIpO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgbXVsbWF0X2dwdS5tdWx0aXBseU1hdHJpeChcbiAgICAgICAgZ3BncHUsIHByb2dyYW0sIGFUZXh0dXJlLCBiVGV4dHVyZSwgcmVzdWx0VGV4dHVyZSwgW3NpemUsIHNpemVdKTtcbiAgfVxuXG4gIGNvbnN0IGFjdHVhbCA9IGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUocmVzdWx0VGV4dHVyZSwgc2l6ZSwgc2l6ZSk7XG4gIGNvbnN0IGF2Z1RpbWUgPSAocGVyZm9ybWFuY2Uubm93KCkgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoYVRleHR1cmUpO1xuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGJUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShyZXN1bHRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTtcblxuZXhwb3J0IGNvbnN0IEJFTkNITUFSS19URVNUX1BBQ0tFRDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgZ3BncHUgPSBuZXcgR1BHUFVDb250ZXh0KCk7XG4gIGNvbnN0IHByb2dyYW06IFdlYkdMUHJvZ3JhbSA9XG4gICAgICBncGdwdS5jcmVhdGVQcm9ncmFtKG11bG1hdF9wYWNrZWRfZ3B1LmdldEZyYWdtZW50U2hhZGVyU291cmNlKFxuICAgICAgICAgIHNpemUsIE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIsIE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpKTtcblxuICBjb25zdCBhVGV4dHVyZSA9IGdwZ3B1LmNyZWF0ZVBhY2tlZE1hdHJpeFRleHR1cmUoc2l6ZSwgc2l6ZSk7XG4gIGNvbnN0IGJUZXh0dXJlID0gZ3BncHUuY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShzaXplLCBzaXplKTtcbiAgY29uc3QgcmVzdWx0VGV4dHVyZSA9IGdwZ3B1LmNyZWF0ZVBhY2tlZE1hdHJpeFRleHR1cmUoc2l6ZSwgc2l6ZSk7XG5cbiAgY29uc3QgYSA9IHRlc3RfdXRpbC5yYW5kb21BcnJheUluUmFuZ2Uoc2l6ZSAqIHNpemUsIC0xLCAxKTtcbiAgY29uc3QgYiA9IHRlc3RfdXRpbC5yYW5kb21BcnJheUluUmFuZ2Uoc2l6ZSAqIHNpemUsIC0xLCAxKTtcbiAgZ3BncHUudXBsb2FkTWF0cml4VG9QYWNrZWRUZXh0dXJlKGFUZXh0dXJlLCBzaXplLCBzaXplLCBhKTtcbiAgZ3BncHUudXBsb2FkTWF0cml4VG9QYWNrZWRUZXh0dXJlKGJUZXh0dXJlLCBzaXplLCBzaXplLCBiKTtcblxuICBjb25zdCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IE9QX1JVTlM7IGkrKykge1xuICAgIG11bG1hdF9wYWNrZWRfZ3B1Lm11bHRpcGx5TWF0cml4UGFja2VkKFxuICAgICAgICBncGdwdSwgcHJvZ3JhbSwgYVRleHR1cmUsIGJUZXh0dXJlLCByZXN1bHRUZXh0dXJlLCBbc2l6ZSwgc2l6ZV0pO1xuICB9XG5cbiAgY29uc3QgYWN0dWFsID1cbiAgICAgIGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVBhY2tlZFRleHR1cmUocmVzdWx0VGV4dHVyZSwgc2l6ZSwgc2l6ZSk7XG4gIGNvbnN0IGF2Z1RpbWUgPSAocGVyZm9ybWFuY2Uubm93KCkgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoYVRleHR1cmUpO1xuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGJUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShyZXN1bHRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblBvbHltZXIoe2lzOiAnZGVtby1mb290ZXInfSk7XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5Qb2x5bWVyKHtpczogJ2RlbW8taGVhZGVyJ30pO1xuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG4vKipcbiAqIEBmaWxlb3ZlcnZpZXdcbiAqXG4gKiBEZWZpbmVzIGFuIGludGVyZmFjZSBmb3IgY3JlYXRpbmcgUG9seW1lciBlbGVtZW50cyBpbiBUeXBlc2NyaXB0IHdpdGggdGhlXG4gKiBjb3JyZWN0IHR5cGluZ3MuIEEgUG9seW1lciBlbGVtZW50IHNob3VsZCBiZSBkZWZpbmVkIGxpa2UgdGhpczpcbiAqXG4gKiBgYGBcbiAqIGxldCBNeUVsZW1lbnRQb2x5bWVyID0gUG9seW1lckVsZW1lbnQoe1xuICogICBpczogJ215LXBvbHltZXItZWxlbWVudCcsXG4gKiAgIHByb3BlcnRpZXM6IHtcbiAqICAgICBmb286IHN0cmluZyxcbiAqICAgICBiYXI6IEFycmF5XG4gKiAgIH1cbiAqIH0pO1xuICpcbiAqIGNsYXNzIE15RWxlbWVudCBleHRlbmRzIE15RWxlbWVudFBvbHltZXIge1xuICogICBmb286IHN0cmluZztcbiAqICAgYmFyOiBudW1iZXJbXTtcbiAqXG4gKiAgIHJlYWR5KCkge1xuICogICAgIGNvbnNvbGUubG9nKCdNeUVsZW1lbnQgaW5pdGlhbGl6ZWQhJyk7XG4gKiAgIH1cbiAqIH1cbiAqXG4gKiBkb2N1bWVudC5yZWdpc3RlckVsZW1lbnQoTXlFbGVtZW50LnByb3RvdHlwZS5pcywgTXlFbGVtZW50KTtcbiAqIGBgYFxuICovXG5cbmV4cG9ydCB0eXBlIFNwZWMgPSB7XG4gIGlzOiBzdHJpbmc7IHByb3BlcnRpZXM6IHtcbiAgICBba2V5OiBzdHJpbmddOiAoRnVuY3Rpb258e1xuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgdHlwZTogRnVuY3Rpb24sIHZhbHVlPzogYW55O1xuICAgICAgcmVmbGVjdFRvQXR0cmlidXRlPzogYm9vbGVhbjtcbiAgICAgIHJlYWRvbmx5PzogYm9vbGVhbjtcbiAgICAgIG5vdGlmeT86IGJvb2xlYW47XG4gICAgICBjb21wdXRlZD86IHN0cmluZztcbiAgICAgIG9ic2VydmVyPzogc3RyaW5nO1xuICAgIH0pXG4gIH07XG4gIG9ic2VydmVycz86IHN0cmluZ1tdO1xufTtcblxuZXhwb3J0IGZ1bmN0aW9uIFBvbHltZXJFbGVtZW50KHNwZWM6IFNwZWMpIHtcbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICByZXR1cm4gUG9seW1lci5DbGFzcyhzcGVjIGFzIGFueSkgYXMge25ldyAoKTogUG9seW1lckhUTUxFbGVtZW50fTtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBQb2x5bWVySFRNTEVsZW1lbnQgZXh0ZW5kcyBIVE1MRWxlbWVudCwgcG9seW1lci5CYXNlIHt9XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBhc3NlcnRDb25jYXQzRFNoYXBlc01hdGNoKFxuICAgIHgxU2hhcGU6IG51bWJlcltdLCB4MlNoYXBlOiBudW1iZXJbXSwgYXhpczogbnVtYmVyLFxuICAgIGVycm9yTWVzc2FnZVByZWZpeCA9ICcnKSB7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgeDFTaGFwZS5sZW5ndGggPT09IDMsXG4gICAgICBlcnJvck1lc3NhZ2VQcmVmaXggKyAnQ29uY2F0M0QgeDEgc2hhcGUgc2hvdWxkIGJlIG9mIHJhbmsgMy4nKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4MlNoYXBlLmxlbmd0aCA9PT0gMyxcbiAgICAgIGVycm9yTWVzc2FnZVByZWZpeCArICdDb25jYXQzRCB4MiBzaGFwZSBzaG91bGQgYmUgb2YgcmFuayAzLicpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgYXhpcyA+PSAwICYmIGF4aXMgPCAzLCAnQXhpcyBmb3IgY29uY2F0M0QgbXVzdCBiZSBiZXR3ZWVuIDAgYW5kIDIuJyk7XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCAzOyBpKyspIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgKGkgPT09IGF4aXMpIHx8ICh4MVNoYXBlW2ldID09PSB4MlNoYXBlW2ldKSxcbiAgICAgICAgZXJyb3JNZXNzYWdlUHJlZml4ICtcbiAgICAgICAgICAgIGBTaGFwZSAoJHt4MVNoYXBlfSkgZG9lcyBub3QgbWF0Y2ggKCR7eDJTaGFwZX0pIGFsb25nIGAgK1xuICAgICAgICAgICAgYG5vbi1jb25jYXRlbmF0ZWQgYXhpcy5gKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZUNvbmNhdDNET3V0cHV0U2hhcGUoXG4gICAgeDFTaGFwZTogbnVtYmVyW10sIHgyU2hhcGU6IG51bWJlcltdLFxuICAgIGF4aXM6IG51bWJlcik6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSB7XG4gIHV0aWwuYXNzZXJ0KHgxU2hhcGUubGVuZ3RoID09PSAzLCAnQ29uY2F0M0QgeDEgc2hhcGUgc2hvdWxkIGJlIG9mIHJhbmsgMy4nKTtcbiAgdXRpbC5hc3NlcnQoeDJTaGFwZS5sZW5ndGggPT09IDMsICdDb25jYXQzRCB4MnNoYXBlIHNob3VsZCBiZSBvZiByYW5rIDMuJyk7XG5cbiAgY29uc3Qgb3V0cHV0U2hhcGUgPSB4MVNoYXBlLnNsaWNlKCk7XG4gIG91dHB1dFNoYXBlW2F4aXNdICs9IHgyU2hhcGVbYXhpc107XG4gIHJldHVybiBvdXRwdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG59IiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgaW5wdXRTaGFwZVJvd0NvbERlcHRoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZpZWxkU2l6ZTogbnVtYmVyLFxuICAgIGRlcHRoOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCB6ZXJvUGFkPzogbnVtYmVyKTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdIHtcbiAgaWYgKHplcm9QYWQgPT0gbnVsbCkge1xuICAgIHplcm9QYWQgPSBjb21wdXRlRGVmYXVsdFBhZChpbnB1dFNoYXBlUm93Q29sRGVwdGgsIGZpZWxkU2l6ZSwgc3RyaWRlKTtcbiAgfVxuICBjb25zdCBpbnB1dFJvd3MgPSBpbnB1dFNoYXBlUm93Q29sRGVwdGhbMF07XG4gIGNvbnN0IGlucHV0Q29scyA9IGlucHV0U2hhcGVSb3dDb2xEZXB0aFsxXTtcbiAgY29uc3Qgb3V0cHV0Um93cyA9IChpbnB1dFJvd3MgLSBmaWVsZFNpemUgKyAyICogemVyb1BhZCkgLyBzdHJpZGUgKyAxO1xuICB1dGlsLmFzc2VydChcbiAgICAgIHV0aWwuaXNJbnQob3V0cHV0Um93cyksXG4gICAgICBgVGhlIG91dHB1dCAjIG9mIHJvd3MgKCR7b3V0cHV0Um93c30pIG11c3QgYmUgYW4gaW50ZWdlci4gQ2hhbmdlIHRoZSBgICtcbiAgICAgICAgICBgc3RyaWRlIGFuZC9vciB6ZXJvIHBhZCBwYXJhbWV0ZXJzYCk7XG5cbiAgY29uc3Qgb3V0cHV0Q29scyA9IChpbnB1dENvbHMgLSBmaWVsZFNpemUgKyAyICogemVyb1BhZCkgLyBzdHJpZGUgKyAxO1xuICB1dGlsLmFzc2VydChcbiAgICAgIHV0aWwuaXNJbnQob3V0cHV0Q29scyksXG4gICAgICBgVGhlIG91dHB1dCAjIG9mIGNvbHVtbnMgKCR7b3V0cHV0Q29sc30pIG11c3QgYmUgYW4gaW50ZWdlci4gQ2hhbmdlIGAgK1xuICAgICAgICAgIGB0aGUgc3RyaWRlIGFuZC9vciB6ZXJvIHBhZCBwYXJhbWV0ZXJzYCk7XG5cbiAgcmV0dXJuIFtvdXRwdXRSb3dzLCBvdXRwdXRDb2xzLCBkZXB0aF07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlRGVmYXVsdFBhZChcbiAgICBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZpZWxkU2l6ZTogbnVtYmVyLFxuICAgIHN0cmlkZTogbnVtYmVyKTogbnVtYmVyIHtcbiAgcmV0dXJuIE1hdGguZmxvb3IoKGlucHV0U2hhcGVbMF0gKiAoc3RyaWRlIC0gMSkgLSBzdHJpZGUgKyBmaWVsZFNpemUpIC8gMik7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlVGV4U2hhcGVGcm9tM0QoXG4gICAgc2hhcGVSb3dDb2xEZXB0aDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdKTogW251bWJlciwgbnVtYmVyXSB7XG4gIHJldHVybiBbc2hhcGVSb3dDb2xEZXB0aFswXSwgc2hhcGVSb3dDb2xEZXB0aFsxXSAqIHNoYXBlUm93Q29sRGVwdGhbMl1dO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZVdlaWdodHNTaGFwZTREKFxuICAgIGlucHV0RGVwdGg6IG51bWJlciwgb3V0cHV0RGVwdGg6IG51bWJlcixcbiAgICBmU2l6ZTogbnVtYmVyKTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0ge1xuICByZXR1cm4gW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZVdlaWdodHNUZXhTaGFwZShcbiAgICBpbnB1dERlcHRoOiBudW1iZXIsIG91dHB1dERlcHRoOiBudW1iZXIsXG4gICAgZmllbGRTaXplOiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgcmV0dXJuIFtmaWVsZFNpemUgKiBmaWVsZFNpemUgKiBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlQmlhc2VzVGV4U2hhcGUob3V0cHV0RGVwdGg6IG51bWJlcik6IFtudW1iZXIsIG51bWJlcl0ge1xuICByZXR1cm4gWzEsIG91dHB1dERlcHRoXTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbXB1dGVEaWxhdGVkUkMoXG4gICAgcmM6IFtudW1iZXIsIG51bWJlcl0sIG9yaWdTdHJpZGU6IG51bWJlcik6IFtudW1iZXIsIG51bWJlcl0ge1xuICBjb25zdCByb3dzRGlsYXRlZCA9IChyY1swXSAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG4gIGNvbnN0IGNvbHNEaWxhdGVkID0gKHJjWzFdIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcbiAgcmV0dXJuIFtyb3dzRGlsYXRlZCwgY29sc0RpbGF0ZWRdO1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5leHBvcnQgZnVuY3Rpb24gdmFsaWRhdGVTaGFwZXMoXG4gICAgc291cmNlU2l6ZTogW251bWJlciwgbnVtYmVyXSwgZGVzdFNpemU6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgY29uc3Qgc3JjQXJlYSA9IHNvdXJjZVNpemVbMF0gKiBzb3VyY2VTaXplWzFdO1xuICBjb25zdCBkc3RBcmVhID0gZGVzdFNpemVbMF0gKiBkZXN0U2l6ZVsxXTtcbiAgaWYgKHNyY0FyZWEgIT09IGRzdEFyZWEpIHtcbiAgICBjb25zdCBzcmNTdHIgPSAnWycgKyBzb3VyY2VTaXplWzBdICsgJywgJyArIHNvdXJjZVNpemVbMV0gKyAnXSc7XG4gICAgY29uc3QgZHN0U3RyID0gJ1snICsgZGVzdFNpemVbMF0gKyAnLCAnICsgZGVzdFNpemVbMV0gKyAnXSc7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnY29weTJEIHNoYXBlcyBoYXZlIGRpZmZlcmVudCBhcmVhczpcXG4gIHNvdXJjZVNpemUgJyArIHNyY1N0ciArXG4gICAgICAgICcsIGFyZWEgJyArIHNyY0FyZWEgKyAnXFxuICBkZXN0U2l6ZSAnICsgZHN0U3RyICsgJywgYXJlYSAnICsgZHN0QXJlYSk7XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcbmltcG9ydCAqIGFzIGNvbmNhdDNkX3V0aWwgZnJvbSAnLi9jb25jYXQzZF91dGlsJztcbmltcG9ydCAqIGFzIGNvcHkyZF91dGlsIGZyb20gJy4vY29weTJkX3V0aWwnO1xuXG5pbXBvcnQge0FycmF5MUQsIEFycmF5MkQsIEFycmF5M0QsIEFycmF5NEQsIE5EQXJyYXksIFNjYWxhcn0gZnJvbSAnLi9uZGFycmF5JztcblxuZXhwb3J0IHR5cGUgU2NvcGVSZXN1bHQgPSBOREFycmF5W118TkRBcnJheXx2b2lkO1xuXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgTkRBcnJheU1hdGgge1xuICBwcml2YXRlIG5kYXJyYXlTY29wZXM6IE5EQXJyYXlbXVtdID0gW107XG4gIHByaXZhdGUgYWN0aXZlU2NvcGU6IE5EQXJyYXlbXTtcblxuICBwcml2YXRlIG5kYXJyYXlzVG9LZWVwOiBOREFycmF5W11bXSA9IFtdO1xuICBwcml2YXRlIGFjdGl2ZVNjb3BlTkRBcnJheXNUb0tlZXA6IE5EQXJyYXlbXSA9IFtdO1xuXG4gIC8qKlxuICAgKiBAcGFyYW0gc2FmZU1vZGUgSW4gc2FmZSBtb2RlLCB5b3UgbXVzdCB1c2UgbWF0aCBvcGVyYXRpb25zIGluc2lkZVxuICAgKiBhIG1hdGguc2NvcGUoKSB3aGljaCB3aWxsIGF1dG9tYXRpY2FsbHkgY2xlYW4gdXAgaW50ZXJtZWRpYXRlIE5EQXJyYXlzLlxuICAgKi9cbiAgY29uc3RydWN0b3IocHJpdmF0ZSBzYWZlTW9kZTogYm9vbGVhbikge31cblxuICAvKipcbiAgICogQ3JlYXRlIGEgbmV3IG1hdGggc2NvcGUuIFB1dCBjaGFpbmVkIG1hdGggb3BlcmF0aW9ucyBpbnNpZGUgYSBzY29wZVxuICAgKiBmdW5jdGlvbiBjbG9zdXJlIHNvIHRoYXQgdGhlIGxpYnJhcnkgYXV0b21hdGljYWxseSBjbGVhbnMgdXAgTkRBcnJheXNcbiAgICogZnJvbSBpbnRlcm1lZGlhdGUgbWF0aCBvcGVyYXRpb25zLiBZb3UgbXVzdCBjcmVhdGUgYSBzY29wZSBpbiBzYWZlIG1vZGVcbiAgICogdG8gY2FsbCBtYXRoIG9wZXJhdGlvbnMuIElmIGEgcmVzdWx0IGlzIHJldHVybmVkIGZyb20gdGhlIHNjb3BlLCBpdCB3aWxsXG4gICAqIGFsc28gYmUgdHJhY2tlZCwgd2hpY2ggbWVhbnMgdGhlcmUgbXVzdCBiZSB5ZXQgYW5vdGhlciB3cmFwcGluZyBzY29wZS5cbiAgICogQHBhcmFtIHNjb3BlRm4gVGhlIGZ1bmN0aW9uIHRvIGV4ZWN1dGUgd2l0aCBjaGFpbmVkIG1hdGggb3BlcmF0aW9ucy5cbiAgICovXG4gIHNjb3BlPFQgZXh0ZW5kcyBTY29wZVJlc3VsdD4oXG4gICAgICBzY29wZUZuOlxuICAgICAgICAgIChrZWVwOiA8VDEgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUMSkgPT4gVDEsXG4gICAgICAgICAgIHRyYWNrOiA8VDIgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUMikgPT4gVDIpID0+IFQpIHtcbiAgICB0aGlzLnN0YXJ0U2NvcGUoKTtcblxuICAgIGNvbnN0IGtlZXBGbiA9IDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQgPT4gdGhpcy5rZWVwKG5kYXJyYXkpO1xuICAgIGNvbnN0IHRyYWNrRm4gPSA8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUID0+IHRoaXMudHJhY2sobmRhcnJheSk7XG4gICAgY29uc3QgcmVzdWx0ID0gc2NvcGVGbihrZWVwRm4sIHRyYWNrRm4pO1xuXG4gICAgdGhpcy5lbmRTY29wZShyZXN1bHQpO1xuXG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIC8qKlxuICAgKiBTdGFydCBhIHNjb3BlLiBVc2UgdGhpcyB3aXRoIGVuZFNjb3BlKCkgdG8gYWNoaWV2ZSB0aGUgc2FtZSBmdW5jdGlvbmFsaXR5XG4gICAqIGFzIHNjb3BlKCkgd2l0aG91dCB0aGUgbmVlZCBmb3IgYSBmdW5jdGlvbiBjbG9zdXJlLlxuICAgKi9cbiAgc3RhcnRTY29wZSgpIHtcbiAgICBjb25zdCBuZXdTY29wZTogTkRBcnJheVtdID0gW107XG4gICAgdGhpcy5uZGFycmF5U2NvcGVzLnB1c2gobmV3U2NvcGUpO1xuICAgIHRoaXMuYWN0aXZlU2NvcGUgPSBuZXdTY29wZTtcblxuICAgIGNvbnN0IG5ld05EQXJyYXlzVG9LZWVwOiBOREFycmF5W10gPSBbXTtcbiAgICB0aGlzLm5kYXJyYXlzVG9LZWVwLnB1c2gobmV3TkRBcnJheXNUb0tlZXApO1xuICAgIHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcCA9IG5ld05EQXJyYXlzVG9LZWVwO1xuICB9XG5cbiAgLyoqXG4gICAqIEVuZCBhIHNjb3BlLiBVc2UgdGhpcyB3aXRoIHN0YXJ0U2NvcGUoKSB0byBhY2hpZXZlIHRoZSBzYW1lIGZ1bmN0aW9uYWxpdHlcbiAgICogYXMgc2NvcGUoKSB3aXRob3V0IHRoZSBuZWVkIGZvciBhIGZ1bmN0aW9uIGNsb3N1cmUuXG4gICAqL1xuICBlbmRTY29wZShyZXN1bHQ6IFNjb3BlUmVzdWx0KSB7XG4gICAgLy8gRGlzcG9zZSB0aGUgY3VycmVudCBzY29wZS5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHRoaXMuYWN0aXZlU2NvcGUubGVuZ3RoOyBpKyspIHtcbiAgICAgIGNvbnN0IG5kYXJyYXkgPSB0aGlzLmFjdGl2ZVNjb3BlW2ldO1xuXG4gICAgICBpZiAodGhpcy5pc05EQXJyYXlEYXRhSW5MaXN0KG5kYXJyYXksIHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcCkgfHxcbiAgICAgICAgICAocmVzdWx0ICE9IG51bGwgJiYgcmVzdWx0IGluc3RhbmNlb2YgTkRBcnJheSAmJlxuICAgICAgICAgICBuZGFycmF5LmdldERhdGEoKSA9PT0gKHJlc3VsdCBhcyBOREFycmF5KS5nZXREYXRhKCkpKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgbmRhcnJheS5kaXNwb3NlKCk7XG4gICAgfVxuXG4gICAgLy8gUG9wIHRoZSBjdXJyZW50IHNjb3BlLlxuICAgIHRoaXMubmRhcnJheVNjb3Blcy5wb3AoKTtcbiAgICB0aGlzLmFjdGl2ZVNjb3BlID0gdGhpcy5uZGFycmF5U2NvcGVzLmxlbmd0aCA9PT0gMCA/XG4gICAgICAgIG51bGwhIDpcbiAgICAgICAgdGhpcy5uZGFycmF5U2NvcGVzW3RoaXMubmRhcnJheVNjb3Blcy5sZW5ndGggLSAxXTtcblxuICAgIC8vIFRyYWNrIHRoZSBjdXJyZW50IHJlc3VsdCBpbiB0aGUgcGFyZW50IHNjb3BlLlxuICAgIGlmIChyZXN1bHQgaW5zdGFuY2VvZiBOREFycmF5ICYmXG4gICAgICAgICF0aGlzLmlzTkRBcnJheURhdGFJbkxpc3QocmVzdWx0LCB0aGlzLmFjdGl2ZVNjb3BlTkRBcnJheXNUb0tlZXApKSB7XG4gICAgICB0aGlzLnRyYWNrKHJlc3VsdCk7XG4gICAgfSBlbHNlIGlmIChBcnJheS5pc0FycmF5KHJlc3VsdCkpIHtcbiAgICAgIHJlc3VsdC5mb3JFYWNoKHIgPT4ge1xuICAgICAgICBpZiAociBpbnN0YW5jZW9mIE5EQXJyYXkgJiZcbiAgICAgICAgICAgICF0aGlzLmlzTkRBcnJheURhdGFJbkxpc3QociwgdGhpcy5hY3RpdmVTY29wZU5EQXJyYXlzVG9LZWVwKSkge1xuICAgICAgICAgIHRoaXMudHJhY2socik7XG4gICAgICAgIH1cbiAgICAgIH0pO1xuICAgIH1cblxuICAgIHRoaXMubmRhcnJheXNUb0tlZXAucG9wKCk7XG4gICAgdGhpcy5hY3RpdmVTY29wZU5EQXJyYXlzVG9LZWVwID0gdGhpcy5uZGFycmF5c1RvS2VlcC5sZW5ndGggPT09IDAgP1xuICAgICAgICBudWxsISA6XG4gICAgICAgIHRoaXMubmRhcnJheXNUb0tlZXBbdGhpcy5uZGFycmF5c1RvS2VlcC5sZW5ndGggLSAxXTtcbiAgfVxuXG4gIHByaXZhdGUgaXNOREFycmF5RGF0YUluTGlzdChuZGFycmF5OiBOREFycmF5LCBuZGFycmF5TGlzdDogTkRBcnJheVtdKSB7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBuZGFycmF5TGlzdC5sZW5ndGg7IGkrKykge1xuICAgICAgaWYgKG5kYXJyYXlMaXN0W2ldLmdldERhdGEoKSA9PT0gbmRhcnJheS5nZXREYXRhKCkpIHtcbiAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuXG4gIC8qKlxuICAgKiBLZWVwcyBhbiBOREFycmF5IGluIHRoZSBjdXJyZW50IHNjb3BlIGZyb20gYmVpbmcgZGlzcG9zZWQgYXV0b21hdGljYWxseS5cbiAgICogQHBhcmFtIHJlc3VsdCBUaGUgTkRBcnJheSB0byBrZWVwIGZyb20gYmVpbmcgZGlzcG9zZWQuXG4gICAqL1xuICBrZWVwPFQgZXh0ZW5kcyBOREFycmF5PihyZXN1bHQ6IFQpOiBUIHtcbiAgICBpZiAodGhpcy5hY3RpdmVTY29wZSA9PSBudWxsKSB7XG4gICAgICBpZiAodGhpcy5zYWZlTW9kZSkge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAnWW91IGFyZSB1c2luZyBtYXRoIGluIHNhZmUgbW9kZS4gRW5jbG9zZSBhbGwgJyArXG4gICAgICAgICAgICAnbWF0aC5tZXRob2QoKSBjYWxscyBpbnNpZGUgYSBzY29wZTogJyArXG4gICAgICAgICAgICAnbWF0aC5zY29wZSgoKSA9PiB7bWF0aC5tZXRob2QoKTsuLi59KSB0byBhdm9pZCBtZW1vcnkgJyArXG4gICAgICAgICAgICAnbGVha3MuJyk7XG4gICAgICB9XG4gICAgICByZXR1cm4gcmVzdWx0O1xuICAgIH1cbiAgICB0aGlzLmFjdGl2ZVNjb3BlTkRBcnJheXNUb0tlZXAucHVzaChyZXN1bHQpO1xuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICAvKipcbiAgICogVHJhY2tzIGFuIE5EQXJyYXkgaW4gdGhlIGN1cnJlbnQgc2NvcGUgdG8gYmUgYXV0b21hdGljYWxseSBjbGVhbmVkIHVwIHdoZW5cbiAgICogdGhlIGN1cnJlbnQgc2NvcGUgZW5kcywgYW5kIHJldHVybnMgdGhlIHZhbHVlLlxuICAgKiBAcGFyYW0gcmVzdWx0IFRoZSBOREFycmF5IHRvIHRyYWNrIGluIHRoZSBjdXJyZW50IHNjb3BlLlxuICAgKi9cbiAgdHJhY2s8VCBleHRlbmRzIE5EQXJyYXk+KHJlc3VsdDogVCk6IFQge1xuICAgIGlmICh0aGlzLmFjdGl2ZVNjb3BlID09IG51bGwpIHtcbiAgICAgIGlmICh0aGlzLnNhZmVNb2RlKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICdZb3UgYXJlIHVzaW5nIG1hdGggaW4gc2FmZSBtb2RlLiBFbmNsb3NlIGFsbCAnICtcbiAgICAgICAgICAgICdtYXRoLm1ldGhvZCgpIGNhbGxzIGluc2lkZSBhIHNjb3BlOiAnICtcbiAgICAgICAgICAgICdtYXRoLnNjb3BlKCgpID0+IHttYXRoLm1ldGhvZCgpOy4uLn0pIHRvIGF2b2lkIG1lbW9yeSAnICtcbiAgICAgICAgICAgICdsZWFrcy4nKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiByZXN1bHQ7XG4gICAgfVxuICAgIHRoaXMuYWN0aXZlU2NvcGUucHVzaChyZXN1bHQpO1xuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGRvdCBwcm9kdWN0IG9mIHR3byBtYXRyaWNlcywgQSAqIEIuIFRoZXNlIG11c3QgYmUgbWF0cmljZXMsXG4gICAqIHVzZSBtYXRyaXhUaW1lc1ZlY3RvciBhbmQgdmVjdG9yVGltZXNNYXRyaXgsIGRvdFByb2R1Y3QsIGFuZCBvdXRlclByb2R1Y3RcbiAgICogaW4gb3RoZXIgY2FzZXMuXG4gICAqIEBwYXJhbSBhIEZpcnN0IG1hdHJpeCBpbiBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSBiIFNlY29uZCBtYXRyaXggaW4gZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKiBAcGFyYW0gYU9yaWVudGF0aW9uIFRoZSBNYXRyaXhPcmllbnRhdGlvbiBvZiBBLiBJZiB1c2luZyBUUkFOU1BPU0VELCB3aWxsXG4gICAqIGNvbXB1dGUgQV5UICogQi5cbiAgICogQHBhcmFtIGJPcmllbnRhdGlvbiBUaGUgTWF0cml4T3JpZW50YXRpb24gb2YgQi4gSWYgdXNpbmcgVFJBTlNQT1NFRCwgd2lsbFxuICAgKiBjb21wdXRlIEEgKiBCXlQuXG4gICAqL1xuICBtYXRNdWwoXG4gICAgICBhOiBBcnJheTJELCBiOiBBcnJheTJELCBhT3JpZW50YXRpb24gPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSLFxuICAgICAgYk9yaWVudGF0aW9uID0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUik6IEFycmF5MkQge1xuICAgIGNvbnN0IGlubmVyU2hhcGVBID1cbiAgICAgICAgKGFPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyBhLnNoYXBlWzFdIDogYS5zaGFwZVswXTtcbiAgICBjb25zdCBpbm5lclNoYXBlQiA9XG4gICAgICAgIChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID8gYi5zaGFwZVswXSA6IGIuc2hhcGVbMV07XG5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYS5yYW5rID09PSAyICYmIGIucmFuayA9PT0gMixcbiAgICAgICAgYEVycm9yIGluIG1hdE11bDogaW5wdXRzIG11c3QgYmUgcmFuayAyLCBnb3QgcmFua3MgJHthLnJhbmt9YCArXG4gICAgICAgICAgICBgYW5kICR7Yi5yYW5rfS5gKTtcblxuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBpbm5lclNoYXBlQSA9PT0gaW5uZXJTaGFwZUIsXG4gICAgICAgIGBFcnJvciBpbiBtYXRNdWw6IGlubmVyIHNoYXBlcyAoJHtpbm5lclNoYXBlQX0pIGFuZCAoYCArXG4gICAgICAgICAgICBgJHtpbm5lclNoYXBlQn0pIG9mIE5EQXJyYXlzIHdpdGggc2hhcGVzICR7YS5zaGFwZX0gYW5kIGAgK1xuICAgICAgICAgICAgYCR7Yi5zaGFwZX0gYW5kIG9yaWVudGF0aW9ucyAke01hdHJpeE9yaWVudGF0aW9uW2FPcmllbnRhdGlvbl19YCArXG4gICAgICAgICAgICBgIGFuZCAke01hdHJpeE9yaWVudGF0aW9uW2JPcmllbnRhdGlvbl19IG11c3QgbWF0Y2guYCk7XG5cbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLm1hdE11bEludGVybmFsKGEsIGIsIGFPcmllbnRhdGlvbiwgYk9yaWVudGF0aW9uKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG1hdE11bEludGVybmFsKFxuICAgICAgYTogQXJyYXkyRCwgYjogQXJyYXkyRCwgYU9yaWVudGF0aW9uOiBNYXRyaXhPcmllbnRhdGlvbixcbiAgICAgIGJPcmllbnRhdGlvbjogTWF0cml4T3JpZW50YXRpb24pOiBBcnJheTJEO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgZG90IHByb2R1Y3Qgb2YgYSB2ZWN0b3IgYW5kIGEgbWF0cml4LCB2ICogQi5cbiAgICogQHBhcmFtIHYgVGhlIHZlY3RvciBpbiBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSBtYXRyaXggVGhlIG1hdHJpeCBpbiBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqL1xuICB2ZWN0b3JUaW1lc01hdHJpeCh2OiBBcnJheTFELCBtYXRyaXg6IEFycmF5MkQpOiBBcnJheTFEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdi5yYW5rID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gdmVjdG9yVGltZXNNYXRyaXg6IGZpcnN0IGlucHV0IG11c3QgYmUgcmFuayAxLCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYHJhbmsgJHt2LnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBtYXRyaXgucmFuayA9PT0gMixcbiAgICAgICAgYEVycm9yIGluIHZlY3RvclRpbWVzTWF0cml4OiBzZWNvbmQgaW5wdXQgbXVzdCBiZSByYW5rIDIsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke21hdHJpeC5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdi5zaXplID09PSBtYXRyaXguc2hhcGVbMF0sXG4gICAgICAgIGBFcnJvciBpbiB2ZWN0b3JUaW1lc01hdHJpeDogc2l6ZSBvZiBmaXJzdCByYW5rIDEgaW5wdXQgKCR7di5zaXplfSkgYCArXG4gICAgICAgICAgICBgbXVzdCBtYXRjaCBpbm5lciBkaW1lbnNpb24gb2Ygc2Vjb25kIHJhbmsgMiBpbnB1dCwgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGByYW5rICR7bWF0cml4LnJhbmt9LmApO1xuXG4gICAgcmV0dXJuIHRoaXMubWF0TXVsKHYuYXMyRCgxLCB2LnNpemUpLCBtYXRyaXgpLmFzMUQoKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgZG90IHByb2R1Y3Qgb2YgYSBtYXRyaXggYW5kIHZlY3RvciwgQSAqIHYuXG4gICAqIEBwYXJhbSBtYXRyaXggVGhlIG1hdHJpeCBpbiBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSB2IFRoZSB2ZWN0b3IgaW4gZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKi9cbiAgbWF0cml4VGltZXNWZWN0b3IobWF0cml4OiBBcnJheTJELCB2OiBBcnJheTFEKTogQXJyYXkxRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHYucmFuayA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIHZlY3RvclRpbWVzTWF0cml4OiBzZWNvbmQgaW5wdXQgbXVzdCByYW5rIDEsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke3YucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIG1hdHJpeC5yYW5rID09PSAyLFxuICAgICAgICBgRXJyb3IgaW4gdmVjdG9yVGltZXNNYXRyaXg6IGZpcnN0IGlucHV0IG11c3QgYmUgYSByYW5rIDIsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke21hdHJpeC5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdi5zaXplID09PSBtYXRyaXguc2hhcGVbMV0sXG4gICAgICAgIGBFcnJvciBpbiB2ZWN0b3JUaW1lc01hdHJpeDogc2l6ZSBvZiBmaXJzdCByYW5rIDEgaW5wdXQgJHt2LnNpemV9IGAgK1xuICAgICAgICAgICAgYG11c3QgbWF0Y2ggaW5uZXIgZGltZW5zaW9uIG9mIHNlY29uZCByYW5rIDIgaW5wdXQsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgc2hhcGUgJHttYXRyaXguc2hhcGV9LmApO1xuXG4gICAgcmV0dXJuIHRoaXMubWF0TXVsKG1hdHJpeCwgdi5hczJEKHYuc2l6ZSwgMSkpLmFzMUQoKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgZG90IHByb2R1Y3Qgb2YgdHdvIHZlY3RvcnMsIHYxICogdjIuXG4gICAqIEBwYXJhbSB2MSBUaGUgZmlyc3QgdmVjdG9yIGluIHRoZSBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSB2MiBUaGUgc2Vjb25kIHZlY3RvciBpbiB0aGUgZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKi9cbiAgZG90UHJvZHVjdCh2MTogQXJyYXkxRCwgdjI6IEFycmF5MUQpOiBTY2FsYXIge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB2MS5yYW5rID09PSAxICYmIHYyLnJhbmsgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBkb3RQcm9kdWN0OiBpbnB1dHMgbXVzdCBiZSByYW5rIDEsIGJ1dCBnb3QgcmFua3MgYCArXG4gICAgICAgICAgICBgJHt2MS5yYW5rfSBhbmQgJHt2Mi5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdjEuc2l6ZSA9PT0gdjIuc2l6ZSxcbiAgICAgICAgYEVycm9yIGluIGRvdFByb2R1Y3Q6IHNpemUgb2YgaW5wdXRzICgke3YxLnNpemV9KSBhbmQgKGAgK1xuICAgICAgICAgICAgYCR7djIuc2l6ZX0pIG11c3QgbWF0Y2guYCk7XG4gICAgcmV0dXJuIHRoaXMubWF0TXVsKHYxLmFzMkQoMSwgdjEuc2l6ZSksIHYyLmFzMkQodjIuc2l6ZSwgMSkpLmFzU2NhbGFyKCk7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIG91dGVyIHByb2R1Y3Qgb2YgdHdvIHZlY3RvcnMsIHYxIGFuZCB2Mi5cbiAgICogQHBhcmFtIHYxIFRoZSBmaXJzdCB2ZWN0b3IgaW4gdGhlIG91dGVyIHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKiBAcGFyYW0gdjIgVGhlIHNlY29uZCB2ZWN0b3IgaW4gdGhlIGRvdCBwcm9kdWN0IG9wZXJhdGlvbi5cbiAgICovXG4gIG91dGVyUHJvZHVjdCh2MTogQXJyYXkxRCwgdjI6IEFycmF5MUQpOiBBcnJheTJEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdjEucmFuayA9PT0gMSAmJiB2Mi5yYW5rID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gb3V0ZXJQcm9kdWN0OiBpbnB1dHMgbXVzdCBiZSByYW5rIDEsIGJ1dCBnb3QgcmFua3MgYCArXG4gICAgICAgICAgICBgJHt2MS5yYW5rfSBhbmQgJHt2Mi5yYW5rfS5gKTtcblxuICAgIHJldHVybiB0aGlzLm1hdE11bCh2MS5hczJEKHYxLnNpemUsIDEpLCB2Mi5hczJEKDEsIHYyLnNpemUpKTtcbiAgfVxuXG4gIC8vLy8vLy8vLy8vLy8vL1xuICAvLyBTaGFwZSBvcHMgLy9cbiAgLy8vLy8vLy8vLy8vLy8vXG5cbiAgLyoqXG4gICAqIENsb25lcyBhbiBOREFycmF5IG9mIGFueSBzaGFwZS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIE5EQXJyYXkgdG8gY2xvbmUuXG4gICAqL1xuICBjbG9uZTxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuY2xvbmVJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNsb25lSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBSZXNoYXBlcyBhbiBOREFycmF5IHRvIGEgbmV3IHNoYXBlLiBUaGUgc2l6ZSBvZiB0aGUgaW5wdXQgTkRBcnJheSBtdXN0XG4gICAqIG1hdGNoIHRoZSBzaXplIG9mIHRoZSByZXF1ZXN0ZWQgc2hhcGUuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0gbmV3U2hhcGUgVGhlIG5ldyBzaGFwZSB0byByZXNoYXBlIHRoZSBOREFycmF5IHRvLiBNdXN0IGJlIHRoZSBzYW1lXG4gICAqIHNpemUgYXMgdGhlIE5EQXJyYXkuXG4gICAqL1xuICByZXNoYXBlPFQxIGV4dGVuZHMgTkRBcnJheSwgVDIgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIG5kYXJyYXk6IFQxLCBuZXdTaGFwZTogbnVtYmVyW10pOiBUMiB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIG5kYXJyYXkuc2l6ZSA9PT0gdXRpbC5zaXplRnJvbVNoYXBlKG5ld1NoYXBlKSxcbiAgICAgICAgYEVycm9yIGluIHJlc2hhcGU6IG9sZCBzaXplICR7bmRhcnJheS5zaXplfSBtdXN0IG1hdGNoIG5ldyBzaXplIGAgK1xuICAgICAgICAgICAgYCR7dXRpbC5zaXplRnJvbVNoYXBlKG5ld1NoYXBlKX0uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5yZXNoYXBlSW50ZXJuYWw8VDEsIFQyPihuZGFycmF5LCBuZXdTaGFwZSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCByZXNoYXBlSW50ZXJuYWw8VDEgZXh0ZW5kcyBOREFycmF5LCBUMiBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgbmRhcnJheTogVDEsIG5ld1NoYXBlOiBudW1iZXJbXSk6IFQyO1xuXG4gIC8qKlxuICAgKiBFeHRyYWN0cyBhIHNsaWNlIGZyb20gYSBtYXRyaXguIFRoZSBvcGVyYXRpb24gZXh0cmFjZXMgYSBzbGljZSBmcm9tIGlucHV0XG4gICAqIHRoYXQgc3RhcnRzIGF0IGNvb3JkaW5hdGVzIGBiZWdpbmAgYW5kIGlzIG9mIHNpemUgYHNpemVgLlxuICAgKiBAcGFyYW0gaW5wdXQgVGhlIGlucHV0IG1hdHJpeCB0byBzbGljZSBmcm9tLlxuICAgKiBAcGFyYW0gYmVnaW4gVGhlIDJEIGNvb3JkaW5hdGVzIGluIHRoZSBpbnB1dCBtYXRyaXggdG8gc3RhcnQgdGhlIHNsaWNlXG4gICAqIGZyb20uXG4gICAqIEBwYXJhbSBzaXplIFRoZSBzaWNlIG9mIHRoZSAyRCB3aW5kb3cgdG8gc2xpY2UuXG4gICAqL1xuICBzbGljZTJEKGlucHV0OiBBcnJheTJELCBiZWdpbjogW251bWJlciwgbnVtYmVyXSwgc2l6ZTogW251bWJlciwgbnVtYmVyXSk6XG4gICAgICBBcnJheTJEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYmVnaW5bMF0gKyBzaXplWzBdIDw9IGlucHV0LnNoYXBlWzBdICYmXG4gICAgICAgICAgICBiZWdpblsxXSArIHNpemVbMV0gPD0gaW5wdXQuc2hhcGVbMV0sXG4gICAgICAgIGBFcnJvciBpbiBzbGljZTJEOiByZXF1ZXN0ZWQgc3RhcnQgcG9zaXRpb24gJHtiZWdpbn0gYW5kIHNpemUgYCArXG4gICAgICAgICAgICBgJHtzaXplfSB3b3VsZCBvdmVyZmxvdyBpbnB1dCBvZiBzaGFwZSAke2lucHV0LnNoYXBlfS5gKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnNsaWNlMkRJbnRlcm5hbChpbnB1dCwgYmVnaW4sIHNpemUpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc2xpY2UyREludGVybmFsKFxuICAgICAgaW5wdXQ6IEFycmF5MkQsIGJlZ2luOiBbbnVtYmVyLCBudW1iZXJdLCBzaXplOiBbbnVtYmVyLCBudW1iZXJdKTogQXJyYXkyRDtcblxuICAvKipcbiAgICogQ29waWVzIGEgd2luZG93IGZyb20gdGhlIGBzb3VyY2VgIG1hdHJpeCBzdGFydGluZyBhdCBgc291cmNlQmVnaW5gIGFuZCBpc1xuICAgKiBvZiBzaXplIGBzb3VyY2VTaXplYCB0byBhIHdpbmRvdyBpbiB0aGUgYGRlc3RgIG1hdHJpeCBzdGFydGluZyBhdFxuICAgKiBgZGVzdEJlZ2luYCBhbmQgaXMgb2Ygc2l6ZSBgZGVzdFNpemVgL1xuICAgKiBAcGFyYW0gc291cmNlIFRoZSBzb3VyY2UgbWF0cml4IHRvIGNvcHkgZnJvbS5cbiAgICogQHBhcmFtIHNvdXJjZUJlZ2luIFRoZSBjb29yZGluYXRlcyB0byBzdGFydCB0aGUgY29weSBmcm9tLlxuICAgKiBAcGFyYW0gc291cmNlU2l6ZSBUaGUgc2l6ZSBvZiB0aGUgY29weSB3aW5kb3cuXG4gICAqIEBwYXJhbSBkZXN0IFRoZSBkZXN0aW5hdGlvbiBtYXRyaXggdG8gY29weSB0by5cbiAgICogQHBhcmFtIGRlc3RCZWdpbiBUaGUgY29vcmRpbmF0ZXMgaW4gYGRlc3RgIHRvIGNvcHkgdG8uXG4gICAqIEBwYXJhbSBkZXN0U2l6ZSBUaGUgc2l6ZSBvZiB0aGUgZGVzdGluYXRpb24gd2luZG93LlxuICAgKi9cbiAgY29weTJEKFxuICAgICAgc291cmNlOiBBcnJheTJELCBzb3VyY2VCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHNvdXJjZVNpemU6IFtudW1iZXIsIG51bWJlcl0sIGRlc3Q6IEFycmF5MkQsIGRlc3RCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIGRlc3RTaXplOiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHNvdXJjZUJlZ2luWzBdICsgc291cmNlU2l6ZVswXSA8PSBzb3VyY2Uuc2hhcGVbMF0gJiZcbiAgICAgICAgICAgIHNvdXJjZUJlZ2luWzFdICsgc291cmNlU2l6ZVsxXSA8PSBzb3VyY2Uuc2hhcGVbMV0sXG4gICAgICAgIGBFcnJvciBpbiBjb3B5MkQ6IHJlcXVlc3RlZCBzb3VyY2Ugc3RhcnQgcG9zaXRpb24gJHtzb3VyY2VCZWdpbn0gYCArXG4gICAgICAgICAgICBgYW5kIHNvdXJjZSBzaXplICR7c291cmNlU2l6ZX0gd291bGQgb3ZlcmZsb3cgc291cmNlIE5EQXJyYXlgICtcbiAgICAgICAgICAgIGBvZiBzaGFwZSAke3NvdXJjZS5zaGFwZX0uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGRlc3RCZWdpblswXSArIGRlc3RTaXplWzBdIDw9IGRlc3Quc2hhcGVbMF0gJiZcbiAgICAgICAgICAgIGRlc3RCZWdpblsxXSArIGRlc3RTaXplWzFdIDw9IGRlc3Quc2hhcGVbMV0sXG4gICAgICAgIGBFcnJvciBpbiBjb3B5MkQ6IHJlcXVlc3RlZCBkZXN0IHN0YXJ0IHBvc2l0aW9uICR7ZGVzdEJlZ2lufSBgICtcbiAgICAgICAgICAgIGBhbmQgc291cmNlIHNpemUgJHtkZXN0U2l6ZX0gd291bGQgb3ZlcmZsb3cgZGVzdCBOREFycmF5IG9mYCArXG4gICAgICAgICAgICBgc2hhcGUgJHtkZXN0LnNoYXBlfS5gKTtcbiAgICBjb3B5MmRfdXRpbC52YWxpZGF0ZVNoYXBlcyhzb3VyY2VTaXplLCBkZXN0U2l6ZSk7XG5cbiAgICByZXR1cm4gdGhpcy5jb3B5MkRJbnRlcm5hbChcbiAgICAgICAgc291cmNlLCBzb3VyY2VCZWdpbiwgc291cmNlU2l6ZSwgZGVzdCwgZGVzdEJlZ2luLCBkZXN0U2l6ZSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvcHkyREludGVybmFsKFxuICAgICAgc291cmNlOiBBcnJheTJELCBzb3VyY2VCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHNvdXJjZVNpemU6IFtudW1iZXIsIG51bWJlcl0sIGRlc3Q6IEFycmF5MkQsIGRlc3RCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIGRlc3RTaXplOiBbbnVtYmVyLCBudW1iZXJdKTogdm9pZDtcblxuICAvKipcbiAgICogQ29uY2F0ZW5hdGVzIHR3byAzRCBuZGFycmF5cyBhbG9uZyBhIGdpdmVuIGF4aXMuXG4gICAqXG4gICAqIEZvciBleGFtcGxlLCBpZjpcbiAgICogQTogc2hhcGUoMiwgMSwgMykgPSB8IHIxLCBnMSwgYjEgfFxuICAgKiAgICAgICAgICAgICAgICAgICAgIHwgcjIsIGcyLCBiMiB8XG4gICAqXG4gICAqIEI6IHNoYXBlKDIsIDEsIDMpID0gfCByMywgZzMsIGIzIHxcbiAgICogICAgICAgICAgICAgICAgICAgICB8IHI0LCBnNCwgYjQgfFxuICAgKlxuICAgKiBDID0gY29uY2F0M0QoQSwgQiwgYXhpcylcbiAgICpcbiAgICogaWYgYXhpcyA9IDA6XG4gICAqIEM6IHNoYXBlKDQsIDEsIDMpID0gfCByMSwgZzEsIGIxIHxcbiAgICogICAgICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIgfFxuICAgKiAgICAgICAgICAgICAgICAgICAgIHwgcjMsIGczLCBiMyB8XG4gICAqICAgICAgICAgICAgICAgICAgICAgfCByNCwgZzQsIGI0IHxcbiAgICpcbiAgICogaWYgYXhpcyA9IDE6XG4gICAqIEM6IHNoYXBlKDIsIDIsIDMpID0gfCByMSwgZzEsIGIxLCByMywgZzMsIGIzIHxcbiAgICogICAgICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIsIHI0LCBnNCwgYjQgfFxuICAgKlxuICAgKiBpZiBheGlzID0gMjpcbiAgICogQyA9IHNoYXBlKDIsIDEsIDYpID0gfCByMSwgZzEsIGIxLCByMywgZzMsIGIzIHxcbiAgICogICAgICAgICAgICAgICAgICAgICAgfCByMiwgZzIsIGIyLCByNCwgZzQsIGI0IHxcbiAgICpcbiAgICogQHBhcmFtIG5kYXJyYXkxIFRoZSBmaXJzdCBhcnJheSB0byBjb25jYXQuXG4gICAqIEBwYXJhbSBuZGFycmF5MiBUaGUgc2Vjb25kIGFycmF5IHRvIGNvbmF0LlxuICAgKiBAcGFyYW0gYXhpcyBUaGUgYXhpcyB0byBjb25jYXRlIGFsb25nLlxuICAgKi9cbiAgY29uY2F0M0QobmRhcnJheTE6IEFycmF5M0QsIG5kYXJyYXkyOiBBcnJheTNELCBheGlzOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICBjb25jYXQzZF91dGlsLmFzc2VydENvbmNhdDNEU2hhcGVzTWF0Y2goXG4gICAgICAgIG5kYXJyYXkxLnNoYXBlLCBuZGFycmF5Mi5zaGFwZSwgYXhpcywgJ0Vycm9yIGluIGNvbmNhdDNkOiAnKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmNvbmNhdDNESW50ZXJuYWwobmRhcnJheTEsIG5kYXJyYXkyLCBheGlzKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvbmNhdDNESW50ZXJuYWwoXG4gICAgICBuZGFycmF5MTogQXJyYXkzRCwgbmRhcnJheTI6IEFycmF5M0QsIGF4aXM6IG51bWJlcik6IEFycmF5M0Q7XG5cbiAgLy8vLy8vLy8vLy8vLy8vLy8vL1xuICAvLyBSZWR1Y3Rpb24gb3BzIC8vXG4gIC8vLy8vLy8vLy8vLy8vLy8vLy9cblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIHRoZSBsb2coc3VtKGUgXiB4KSkgZm9yIGVhY2ggeCBpbiB0aGUgaW5wdXQgbmRhcnJheS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkgdG8gY29tcHV0ZSB0aGUgbG9nU3VtRXhwIG92ZXIuXG4gICAqL1xuICBsb2dTdW1FeHAobmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5sb2dTdW1FeHBJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGxvZ1N1bUV4cEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXI7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBzdW0gb2YgYWxsIHRoZSBlbnRyaWVzIGluIHRoZSBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheSB0byBjb21wdXRlIHRoZSBzdW0gb3Zlci5cbiAgICovXG4gIHN1bShuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnN1bUludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc3VtSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGZsYXR0ZW5lZCBpbmRleCBvZiB0aGUgbWluaW11bSBlbGVtZW50IGluIHRoZSBuZGFycmF5LlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIGFyZ01pbihuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmFyZ01pbkludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgYXJnTWluSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGZsYXR0ZW5lZCBpbmRleCBvZiB0aGUgbWF4aW11bSBlbGVtZW50IGluIHRoZSBuZGFycmF5LlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIGFyZ01heChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmFyZ01heEludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgYXJnTWF4SW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogUmV0dXJucyBhIDEgaWYgdGhlIGFyZ01heCBvZiB4MSBhbmQgeDIgYXJlIHRoZSBzYW1lLCBvdGhlcndpc2UgMC5cbiAgICogQHBhcmFtIHgxIFRoZSBmaXJzdCBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0geDIgVGhlIHNlY29uZCBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgYXJnTWF4RXF1YWxzKHgxOiBOREFycmF5LCB4MjogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaCh4MS5zaGFwZSwgeDIuc2hhcGUsICdFcnJvciBpbiBhcmdNYXhFcXVhbHM6ICcpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuYXJnTWF4RXF1YWxzSW50ZXJuYWwoeDEsIHgyKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGFyZ01heEVxdWFsc0ludGVybmFsKHgxOiBOREFycmF5LCB4MjogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIHRvcCBLIHZhbHVlcyBhbmQgZmxhdHRlbmVkIGluZGljZXMuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0gayBIb3cgbWFueSB0b3AgdmFsdWVzIHRvIGNvbXB1dGUuXG4gICAqL1xuICB0b3BLKG5kYXJyYXk6IE5EQXJyYXksIGs6IG51bWJlcik6IHt2YWx1ZXM6IEFycmF5MUQsIGluZGljZXM6IEFycmF5MUR9IHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgayA8PSBuZGFycmF5LnNpemUsXG4gICAgICAgIGBFcnJvciBpbiB0b3BLOiBrIHZhbHVlICgke2t9KSBtdXN0IGJlIGxlc3MgdGhhbiBzaXplIG9mIGlucHV0IGAgK1xuICAgICAgICAgICAgYG5kYXJyYXksIGdvdCBzaGFwZSAke25kYXJyYXkuc2hhcGV9LmApO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRoaXMudG9wS0ludGVybmFsKG5kYXJyYXksIGspO1xuICAgIHRoaXMudHJhY2socmVzdWx0LnZhbHVlcyk7XG4gICAgdGhpcy50cmFjayhyZXN1bHQuaW5kaWNlcyk7XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgdG9wS0ludGVybmFsKG5kYXJyYXk6IE5EQXJyYXksIGs6IG51bWJlcik6XG4gICAgICB7dmFsdWVzOiBBcnJheTFELCBpbmRpY2VzOiBBcnJheTFEfTtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIG1pbmltdW0gdmFsdWUgZnJvbSB0aGUgaW5wdXQuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgbWluKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMubWluSW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBtaW5JbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgbWF4aW11bSB2YWx1ZSBmcm9tIHRoZSBpbnB1dC5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICBtYXgobmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5tYXhJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG1heEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXI7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBzb2Z0bWF4IG5vcm1hbGl6ZWQgdmVjdG9yIGZyb20gdGhlIGlucHV0IHZlY3Rvci5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IHZlY3Rvci5cbiAgICovXG4gIHNvZnRtYXgoeDogQXJyYXkxRCk6IEFycmF5MUQge1xuICAgIHJldHVybiB0aGlzLnNjb3BlKCgpID0+IHtcbiAgICAgIC8vIERvIGl0IGluIGxvZyBzcGFjZSBmb3IgbnVtZXJpY2FsIHN0YWJpbGl0eS5cbiAgICAgIC8vIGV4cChYIC0gbG9nU3VtRXhwKFgpKVxuICAgICAgY29uc3QgbHNlID0gdGhpcy5sb2dTdW1FeHAoeCk7XG4gICAgICBjb25zdCBsb2dSZXN1bHQgPSB0aGlzLmFycmF5TWludXNTY2FsYXIoeCwgbHNlKTtcbiAgICAgIHJldHVybiB0aGlzLmV4cChsb2dSZXN1bHQpO1xuICAgIH0pO1xuICB9XG5cbiAgLy8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuICAvLyBFbGVtZW50LXdpc2Ugb3BzIC8vXG4gIC8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cblxuICAvKipcbiAgICogU3dpdGNoZXMgZGltZW5zaW9ucyBvZiB0aGUgaW5wdXQgTkRBcnJheS5cbiAgICogQHBhcmFtIGEgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqIEBwYXJhbSBuZXdEaW0gVGhlIG5ldyBpbmRpY2VzIHRoYXQgZGVmaW5lIHdoaWNoIHNoYXBlcyB2YWx1ZXMgdG8gc3dpdGNoLlxuICAgKi9cbiAgc3dpdGNoRGltPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBuZXdEaW06IG51bWJlcltdKTogVCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGEucmFuayA9PT0gbmV3RGltLmxlbmd0aCxcbiAgICAgICAgYEVycm9yIGluIHN3aXRjaERpbTogbGVuZ3RoIG9mIGlucHV0IHNoYXBlICR7YS5zaGFwZX0gYCArXG4gICAgICAgICAgICBgbXVzdCBtYXRjaCBzaXplIG9mIG5ld0RpbSBhcnJheSAke25ld0RpbX0uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zd2l0Y2hEaW1JbnRlcm5hbChhLCBuZXdEaW0pKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc3dpdGNoRGltSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgYTogVCwgbmV3RGltOiBudW1iZXJbXSk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgc2NhbGFyIHBsdXMgTkRBcnJheSwgYyArIEEuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgYyBpbiBjICsgQS5cbiAgICogQHBhcmFtIGEgVGhlIE5EQXJyYXkgQSBpbiBjICsgQS5cbiAgICovXG4gIHNjYWxhclBsdXNBcnJheTxUIGV4dGVuZHMgTkRBcnJheT4oYzogU2NhbGFyLCBhOiBUKTogVCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGMuc2l6ZSA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIHNjYWxhclBsdXNBcnJheTogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zY2FsYXJQbHVzQXJyYXlJbnRlcm5hbChjLCBhKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHNjYWxhclBsdXNBcnJheUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIGM6IFNjYWxhciwgYTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgc2NhbGFyIG1pbnVzIE5EQXJyYXksIGMgLSBBLlxuICAgKiBAcGFyYW0gYyBUaGUgc2NhbGFyIGMgaW4gYyAtIEEuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IEEgaW4gYyAtIEEuXG4gICAqL1xuICBzY2FsYXJNaW51c0FycmF5PFQgZXh0ZW5kcyBOREFycmF5PihjOiBTY2FsYXIsIGE6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYy5zaXplID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gc2NhbGFyTWludXNBcnJheTogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zY2FsYXJNaW51c0FycmF5SW50ZXJuYWwoYywgYSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzY2FsYXJNaW51c0FycmF5SW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgYzogU2NhbGFyLCBhOiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgYSBzY2FsYXIgbWludXMgTkRBcnJheSwgQSAtIGMuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IEEgaW4gQSAtIGMuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgYyBpbiBBIC0gYy5cbiAgICovXG4gIGFycmF5TWludXNTY2FsYXI8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGM6IFNjYWxhcik6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBhcnJheU1pbnVzU2NhbGFyOiBzZWNvbmQgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBgICtcbiAgICAgICAgICAgIGBnb3QgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5hcnJheU1pbnVzU2NhbGFySW50ZXJuYWwoYSwgYykpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBhcnJheU1pbnVzU2NhbGFySW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgYTogVCwgYzogU2NhbGFyKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgLTEgKiBBIGVsZW1lbnQtd2lzZS5cbiAgICogQHBhcmFtIGEgVGhlIGlucHV0IGFycmF5LlxuICAgKi9cbiAgbmVnPFQgZXh0ZW5kcyBOREFycmF5PihhOiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5uZWdJbnRlcm5hbChhKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG5lZ0ludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBUKTogVDtcblxuICAvKipcbiAgICogQWRkcyB0d28gTkRBcnJheXMgZWxlbWVudC13aXNlLCBBICsgQi4gSW5wdXRzIG11c3QgYmUgdGhlIHNhbWUgc2hhcGUuXG4gICAqIEBwYXJhbSBhIFRoZSBmaXJzdCBOREFycmF5IHRvIGFkZCBlbGVtZW50LXdpc2UuXG4gICAqIEBwYXJhbSBiIFRoZSBzZWNvbmQgTkRBcnJheSB0byBhZGQgZWxlbWVudC13aXNlLlxuICAgKi9cbiAgYWRkPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChhLnNoYXBlLCBiLnNoYXBlLCAnRXJyb3IgaW4gYWRkOiAnKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmFkZEludGVybmFsKGEsIGIpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgYWRkSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBTdWJ0cmFjdHMgdHdvIE5EQXJyYXlzIGVsZW1lbnQtd2lzZSwgQSAtIEIuIElucHV0cyBtdXN0IGJlIHRoZSBzYW1lIHNoYXBlLlxuICAgKiBAcGFyYW0gYSBUaGUgZmlyc3QgTkRBcnJheSB0byBzdWJ0cmFjdCBlbGVtZW50LXdpc2UuXG4gICAqIEBwYXJhbSBiIFRoZSBzZWNvbmQgTkRBcnJheSB0byBzdWJ0cmFjdCBlbGVtZW50LXdpc2UuXG4gICAqL1xuICBzdWI8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydFNoYXBlc01hdGNoKGEuc2hhcGUsIGIuc2hhcGUsICdFcnJvciBpbiBzdWI6ICcpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuc3ViSW50ZXJuYWwoYSwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzdWJJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIE11bHRpcGxpZXMgdHdvIE5EQXJyYXlzIGVsZW1lbnQtd2lzZSAoaGFkYW1hcmQgcHJvZHVjdCksIEEgKiBCLiBJbnB1dHMgbXVzdFxuICAgKiBiZSB0aGUgc2FtZSBzaGFwZS5cbiAgICogQHBhcmFtIGEgVGhlIGZpcnN0IE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKiBAcGFyYW0gYiBUaGUgc2Vjb25kIE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKi9cbiAgZWxlbWVudFdpc2VNdWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydFNoYXBlc01hdGNoKGEuc2hhcGUsIGIuc2hhcGUsICdFcnJvciBpbiBlbGVtZW50V2lzZU11bDogJyk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5lbGVtZW50V2lzZU11bEludGVybmFsKGEsIGIpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgZWxlbWVudFdpc2VNdWxJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIERpdmlkZXMgdHdvIE5EQXJyYXlzIGVsZW1lbnQtd2lzZSAoaGFkYW1hcmQgcHJvZHVjdCksIEEgLyBCLiBJbnB1dHMgbXVzdCBiZVxuICAgKiB0aGUgc2FtZSBzaGFwZS5cbiAgICogQHBhcmFtIGEgVGhlIGZpcnN0IE5EQXJyYXkgdG8gZGl2aWRlIGVsZW1lbnQtd2lzZS5cbiAgICogQHBhcmFtIGIgVGhlIHNlY29uZCBOREFycmF5IHRvIGRpdmlkZSBlbGVtZW50LXdpc2UuXG4gICAqL1xuICBkaXZpZGU8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydFNoYXBlc01hdGNoKGEuc2hhcGUsIGIuc2hhcGUsICdFcnJvciBpbiBkaXZpZGU6ICcpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuZGl2aWRlSW50ZXJuYWwoYSwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBkaXZpZGVJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgc2NhbGFyIGRpdmlkZWQgYnkgYW4gTkRBcnJheSwgYnJvYWRjYXN0ZWQgb3ZlciB0aGUgTkRBcnJheSwgYyAvXG4gICAqIEEuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgdmFsdWUgaW4gYyAvIEEuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IHZhbHVlIGluIGMgLyBBLlxuICAgKi9cbiAgc2NhbGFyRGl2aWRlZEJ5QXJyYXk8VCBleHRlbmRzIE5EQXJyYXk+KGM6IFNjYWxhciwgYTogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBzY2FsYXJEaXZpZGVkQnlBcnJheTogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBgICtcbiAgICAgICAgICAgIGBnb3QgTkRBcnJheSBvZiByYW5rICR7Yy5yYW5rfS5gKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnNjYWxhckRpdmlkZWRCeUFycmF5SW50ZXJuYWwoYywgYSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzY2FsYXJEaXZpZGVkQnlBcnJheUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIGM6IFNjYWxhciwgYTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGFuIE5EQXJyYXkgZGl2aWRlZCBieSBhIHNjYWxhciwgYnJvYWRjYXN0ZWQgb3ZlciB0aGUgTkRBcnJheSwgQSAvXG4gICAqIGMuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IHZhbHVlIGluIEEgLyBjLlxuICAgKiBAcGFyYW0gYyBUaGUgc2NhbGFyIHZhbHVlIGluIEEgLyBjLlxuICAgKi9cbiAgYXJyYXlEaXZpZGVkQnlTY2FsYXI8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGM6IFNjYWxhcik6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBhcnJheURpdmlkZWRCeVNjYWxhcjogc2Vjb25kIGFyZ3VtZW50IG11c3QgYmUgcmFuayAwLCBgICtcbiAgICAgICAgICAgIGBidXQgZ290IE5EQXJyYXkgb2YgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5hcnJheURpdmlkZWRCeVNjYWxhckludGVybmFsKGEsIGMpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgYXJyYXlEaXZpZGVkQnlTY2FsYXJJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oXG4gICAgICBhOiBULCBjOiBTY2FsYXIpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBleHBvbmVudGlhbCBvZiB0aGUgaW5wdXQgTkRBcnJheSBlbGVtZW50LXdpc2UuIHkgPSBlIF4geFxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIGV4cDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuZXhwSW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBleHBJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIG5hdHVyYWwgbG9nYXJpdGhtIG9mIHRoZSBpbnB1dCBOREFycmF5IGVsZW1lbnQtd2lzZS4geSA9IGxuKHgpXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgbG9nPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5sb2dJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGxvZ0ludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgcmVjdGlmaWVkIGxpbmVhciBlbGVtZW50LXdpc2UsIG1heCh4LCAwKS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICByZWx1PFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5yZWx1SW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCByZWx1SW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBzaWdtb2lkIGVsZW1lbnQtd2lzZSwgeSA9IDEgLyAoMSArIGV4cCgteCkpLlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIHNpZ21vaWQ8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnNpZ21vaWRJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHNpZ21vaWRJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGh5cGVyYm9saWMgdGFuZ2VudCBvZiB0aGUgaW5wdXQgTkRBcnJheSBlbGVtZW50LXdpc2UuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgdGFuaDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMudGFuaEludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgdGFuaEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgc2luIG9mIHRoZSBpbnB1dCBOREFycmF5IGVsZW1lbnQtd2lzZSwgeSA9IHNpbih4KS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICBzaW48VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnNpbkludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc2luSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBzdGVwIG9mIHRoZSBpbnB1dCBOREFycmF5IGVsZW1lbnQtd2lzZSwgeSA9IDEgaWYgeCA+IDAgfCAwIGlmIHggPD1cbiAgICogMFxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIHN0ZXA8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnN0ZXBJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHN0ZXBJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgc2NhbGVkIGFycmF5IGFkZCBvcGVyYXRpb24sIGMxICogQSArIGMyICogQi5cbiAgICogQHBhcmFtIGMxIFRoZSBmaXJzdCBzY2FsYXIgaW4gdGhlIHNjYWxlZCBhcnJheSBhZGQgY29tcHV0YXRpb24uXG4gICAqIEBwYXJhbSBhIFRoZSBmaXJzdCBOREFycmF5IGluIHRoZSBzY2FsZWQgYXJyYXkgYWRkIGNvbXB1dGF0aW9uLlxuICAgKiBAcGFyYW0gYzIgVGhlIHNlY29uZCBzY2FsYXIgaW4gdGhlIHNjYWxlZCBhcnJheSBhZGQgY29tcHV0YXRpb24uXG4gICAqIEBwYXJhbSBjYiBUaGUgc2Vjb25kIE5EQXJyYXkgaW4gdGhlIHNjYWxlZCBhcnJheSBhZGQgY29tcHV0YXRpb24uXG4gICAqL1xuICBzY2FsZWRBcnJheUFkZDxUIGV4dGVuZHMgTkRBcnJheT4oYzE6IFNjYWxhciwgYTogVCwgYzI6IFNjYWxhciwgYjogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjMS5zaXplID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gc2NhbGVkQXJyYXlBZGQ6IGZpcnN0IGFyZ3VtZW50IG11c3QgcmFuayAwLCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYCByYW5rICR7YzEucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGMyLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBzY2FsZWRBcnJheUFkZDogdGhpcmQgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgTkRBcnJheSBvZiByYW5rICR7YzIucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChhLnNoYXBlLCBiLnNoYXBlLCAnRXJyb3IgaW4gc2NhbGVkQXJyYXlBZGQ6ICcpO1xuXG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zY2FsZWRBcnJheUFkZEludGVybmFsKGMxLCBhLCBjMiwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzY2FsZWRBcnJheUFkZEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIGMxOiBTY2FsYXIsIGE6IFQsIGMyOiBTY2FsYXIsIGI6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBhIHNjYWxhciB0aW1lcyBhcnJheSBvcGVyYXRpb24gYnJvYWRjYXN0ZWQgb3ZlciB0aGUgTkRBcnJheSwgYyAqXG4gICAqIEEuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgaW4gdGhlIG9wZXJhdGlvbi5cbiAgICogQHBhcmFtIEEgdGhlIE5EQXJyYXkgaW4gdGhlIG9wZXJhdGlvbiB0aGF0IHdpbGwgYmUgYnJvYWRjYXN0ZWQgb3Zlci5cbiAgICovXG4gIHNjYWxhclRpbWVzQXJyYXk8VCBleHRlbmRzIE5EQXJyYXk+KGM6IFNjYWxhciwgYTogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBhcnJheURpdmlkZWRCeVNjYWxhcjogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBgICtcbiAgICAgICAgICAgIGBnb3QgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zY2FsYXJUaW1lc0FycmF5SW50ZXJuYWwoYywgYSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzY2FsYXJUaW1lc0FycmF5SW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgYzogU2NhbGFyLCBhOiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgYW4gZWxlbWVudC13aXNlIGJyb2FkY2FzdGVkIG11bHRpcGxpY2F0aW9uIG9mIHR3byBtYXRyaWNlcyBBIGFuZFxuICAgKiBCLiBXaWxsIHJldHVybiBhIG5ldyBtYXRyaXggdGhhdCBpcyB0aGUgbWF4IG9mIEEgYW5kIEIsIHdoZXJlIHRoZSBzbWFsbGVyXG4gICAqIG1hdHJpeCB3aWxsIGJyb2FkY2FzdCBvdmVyIHRoZSBsYXJnZXIgbWF0cml4LlxuICAgKiBAcGFyYW0gYyBUaGUgc2NhbGFyIGluIHRoZSBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSBBIHRoZSBOREFycmF5IGluIHRoZSBvcGVyYXRpb24gdGhhdCB3aWxsIGJlIGJyb2FkY2FzdGVkIG92ZXIuXG4gICAqL1xuICBlbGVtZW50V2lzZU11bEJyb2FkY2FzdChhOiBBcnJheTJELCBiOiBBcnJheTJEKTogQXJyYXkyRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGEucmFuayA9PT0gMixcbiAgICAgICAgYEVycm9yIGluIGVsZW1lbnRXaXNlTXVsQnJvYWRjYXN0OiBmaXJzdCBhcmd1bWVudCBtdXN0IGJlIGAgK1xuICAgICAgICAgICAgYHJhbmsgMiwgYnV0IGdvdCByYW5rICR7YS5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYi5yYW5rID09PSAyLFxuICAgICAgICBgRXJyb3IgaW4gZWxlbWVudFdpc2VNdWxCcm9hZGNhc3Q6IHNlY29uZCBhcmd1bWVudCBtdXN0IGJlIGAgK1xuICAgICAgICAgICAgYHJhbmsgMiwgYnV0IGdvdCByYW5rICR7Yi5yYW5rfS5gKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmVsZW1lbnRXaXNlTXVsQnJvYWRjYXN0SW50ZXJuYWwoYSwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBlbGVtZW50V2lzZU11bEJyb2FkY2FzdEludGVybmFsKGE6IEFycmF5MkQsIGI6IEFycmF5MkQpOlxuICAgICAgQXJyYXkyRDtcblxuICAvLy8vLy8vLy8vLy8vLy8vLy8vLy9cbiAgLy8gQ29udm9sdXRpb24gb3BzIC8vXG4gIC8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBhIDJEIGNvbnZvbHV0aW9uIG92ZXIgdGhlIGlucHV0IHguXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgbXVzdCBiZSByYW5rIDMsIG9mIHNoYXBlIFtyb3dzLCBjb2xzLCBkZXB0aDFdLlxuICAgKiBAcGFyYW0gd2VpZ2h0cyBUaGUgd2VpZ2h0cyBOREFycmF5LCBtdXN0IGJlIHJhbmsgNCwgb2Ygc2hhcGUgW2YsIGYsIGRlcHRoMSxcbiAgICogZGVwdGgyXS5cbiAgICogQHBhcmFtIGJpYXNlcyBPcHRpb25hbCBiaWFzZXMgTkRBcnJheSwgbXVzdCBiZSByYW5rIDEgb2Ygc2hhcGUgW2RlcHRoMl0uXG4gICAqIEBwYXJhbSBzdHJpZGUgVGhlIHN0cmlkZSBvZiB0aGUgY29udm9sdXRpb24uXG4gICAqIEBwYXJhbSB6ZXJvUGFkIFRoZSB6ZXJvIHBhZGRpbmcgb2YgZWFjaCBzaWRlIG9mIHRoZSBpbnB1dCBOREFycmF5LiBXaWxsIHBhZFxuICAgKiBlcXVhbGx5IG9uIGFsbCBzaWRlcy5cbiAgICovXG4gIGNvbnYyZChcbiAgICAgIHg6IEFycmF5M0QsIHdlaWdodHM6IEFycmF5NEQsIGJpYXNlczogQXJyYXkxRHxudWxsLCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHplcm9QYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnJhbmsgPT09IDMsXG4gICAgICAgIGBFcnJvciBpbiBjb252MmQ6IHggbXVzdCBiZSByYW5rIDMsIGJ1dCBnb3QgcmFuayAke3gucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHdlaWdodHMucmFuayA9PT0gNCxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZDogd2VpZ2h0cyBtdXN0IGJlIHJhbmsgNCwgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgICAgYCR7d2VpZ2h0cy5yYW5rfS5gKTtcbiAgICBpZiAoYmlhc2VzICE9IG51bGwpIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGJpYXNlcy5yYW5rID09PSAxLFxuICAgICAgICAgIGBFcnJvciBpbiBjb252MmQ6IGJpYXNlcyBtdXN0IGJlIHJhbmsgMSwgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgICAgICBgJHtiaWFzZXMucmFua30uYCk7XG4gICAgfVxuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHguc2hhcGVbMl0gPT09IHdlaWdodHMuc2hhcGVbMl0sXG4gICAgICAgIGBFcnJvciBpbiBjb252MmQ6IGRlcHRoIG9mIGlucHV0ICgke3guc2hhcGVbMl19KSBtdXN0IG1hdGNoICBgICtcbiAgICAgICAgICAgIGBpbnB1dCBkZXB0aCBmb3Igd2VpZ2h0cyAke3dlaWdodHMuc2hhcGVbMl19LmApO1xuXG5cbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmNvbnYyZEludGVybmFsKHgsIHdlaWdodHMsIGJpYXNlcywgc3RyaWRlLCB6ZXJvUGFkKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvbnYyZEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhc2VzOiBBcnJheTFEfG51bGwsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgemVyb1BhZDogbnVtYmVyKTogQXJyYXkzRDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGJhY2twcm9wIG9mIGEgMkQgY29udm9sdXRpb24uXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgbXVzdCBiZSByYW5rIDMsIG9mIHNoYXBlIFt4cm93cywgeGNvbHMsIGRlcHRoMV0uXG4gICAqIEBwYXJhbSBkeSBUaGUgZHkgaW1hZ2UsIG11c3QgYmUgcmFuayAzLCBvZiBzaGFwZSBbeXJvd3MsIHljb2xzLCBkZXB0aDJdLlxuICAgKiBAcGFyYW0gd2VpZ2h0cyBUaGUgd2VpZ2h0cyBOREFycmF5LCBtdXN0IGJlIHJhbmsgNCwgb2Ygc2hhcGUgW2YsIGYsIGRlcHRoMSxcbiAgICogZGVwdGgyXS5cbiAgICogQHBhcmFtIHN0cmlkZSBUaGUgc3RyaWRlIG9mIHRoZSBvcmlnaW5hbCBjb252b2x1dGlvbi5cbiAgICogQHBhcmFtIHBhZCBUaGUgcGFkZGluZyBvZiB0aGUgb3JpZ2luYWwgY29udm9sdXRpb24uXG4gICAqL1xuICBjb252MmRCYWNrUHJvcChcbiAgICAgIHg6IEFycmF5M0QsIGR5OiBBcnJheTNELCB3ZWlnaHRzOiBBcnJheTRELCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHBhZDogbnVtYmVyKToge2R4OiBBcnJheTNELCBkdzogQXJyYXk0RCwgZGI6IEFycmF5MUR9IHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkQmFja1Byb3A6IHggbXVzdCBiZSByYW5rIDMsIGJ1dCBnb3Qgc2hhcGUgYCArXG4gICAgICAgICAgICBgJHt4LnNoYXBlfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgZHkucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZEJhY2tQcm9wOiBkeSBtdXN0IGJlIHJhbmsgMywgYnV0IGdvdCBzaGFwZSBgICtcbiAgICAgICAgICAgIGAke2R5LnNoYXBlfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgd2VpZ2h0cy5yYW5rID09PSA0LFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkQmFja1Byb3A6IHdlaWdodHMgbXVzdCBiZSByYW5rIDQsIGJ1dCBnb3Qgc2hhcGUgYCArXG4gICAgICAgICAgICBgJHt3ZWlnaHRzLnNoYXBlfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5zaGFwZVsyXSA9PT0gd2VpZ2h0cy5zaGFwZVsyXSxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZEJhY2tQcm9wOiBkZXB0aCBvZiB4ICR7eC5zaGFwZVsyXX0pIG11c3QgYCArXG4gICAgICAgICAgICBgbWF0Y2ggaW5wdXQgZGVwdGggZm9yIHdlaWdodHMgKCR7d2VpZ2h0cy5zaGFwZVsyXX0uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGR5LnNoYXBlWzJdID09PSB3ZWlnaHRzLnNoYXBlWzNdLFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkQmFja1Byb3A6IGRlcHRoIG9mIGR5ICgke2R5LnNoYXBlWzJdfSkgbXVzdCBgICtcbiAgICAgICAgICAgIGBtYXRjaCBvdXRwdXQgZGVwdGggZm9yIHdlaWdodHMgKCR7d2VpZ2h0cy5zaGFwZVszXX0pLmApO1xuXG4gICAgY29uc3QgYmFja3Byb3BSZXN1bHQgPVxuICAgICAgICB0aGlzLmNvbnYyZEJhY2tQcm9wSW50ZXJuYWwoeCwgZHksIHdlaWdodHMsIHN0cmlkZSwgcGFkKTtcblxuICAgIHRoaXMudHJhY2soYmFja3Byb3BSZXN1bHQuZGIpO1xuICAgIHRoaXMudHJhY2soYmFja3Byb3BSZXN1bHQuZHcpO1xuICAgIHRoaXMudHJhY2soYmFja3Byb3BSZXN1bHQuZHgpO1xuXG4gICAgcmV0dXJuIGJhY2twcm9wUmVzdWx0O1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBjb252MmRCYWNrUHJvcEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZHk6IEFycmF5M0QsIHdlaWdodHM6IEFycmF5NEQsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiB7ZHg6IEFycmF5M0QsIGR3OiBBcnJheTRELCBkYjogQXJyYXkxRH07XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSB0cmFuc3Bvc2VkIDJEIGNvbnZvbHV0aW9uIG9mIGFuIGltYWdlLCBhbHNvIGtub3duIGFzIGFcbiAgICogZGVjb252b2x1dGlvbi5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IGltYWdlLCBtdXN0IGJlIHJhbmsgMywgb2Ygc2hhcGUgW3hyb3dzLCB4Y29scywgZGVwdGgxXS5cbiAgICogQHBhcmFtIHdlaWdodHMgVGhlIHdlaWdodHMgTkRBcnJheSwgbXVzdCBiZSByYW5rIDQsIG9mIHNoYXBlIFtmLCBmLCBkZXB0aDEsXG4gICAqIGRlcHRoMl0uXG4gICAqIEBwYXJhbSBiaWFzZXMgT3B0aW9uYWwgYmlhc2VzIE5EQXJyYXksIG11c3QgYmUgcmFuayAxIG9mIHNoYXBlIFtkZXB0aDJdLlxuICAgKiBAcGFyYW0gc3RyaWRlIFRoZSBzdHJpZGUgb2YgdGhlIGNvbnZvbHV0aW9uLlxuICAgKiBAcGFyYW0gcGFkIFRoZSBwYWRkaW5nIG9mIGVhY2ggc2lkZSBvZiB0aGUgaW5wdXQgTkRBcnJheS4gV2lsbCBwYWQgZXF1YWxseVxuICAgKiBvbiBhbGwgc2lkZXMuXG4gICAqL1xuICBjb252MmRUcmFuc3Bvc2UoXG4gICAgICB4OiBBcnJheTNELCB3ZWlnaHRzOiBBcnJheTRELCBiaWFzZXM6IEFycmF5MUR8bnVsbCwgc3RyaWRlOiBudW1iZXIsXG4gICAgICBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnJhbmsgPT09IDMsXG4gICAgICAgIGBFcnJvciBpbiBjb252MmRUcmFuc3Bvc2U6IHggbXVzdCBiZSByYW5rIDMsIGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICAgIGAke3gucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHdlaWdodHMucmFuayA9PT0gNCxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZFRyYW5zcG9zZTogd2VpZ2h0cyBtdXN0IGJlIHJhbmsgNCwgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGByYW5rICR7d2VpZ2h0cy5yYW5rfWApO1xuICAgIGlmIChiaWFzZXMgIT0gbnVsbCkge1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgYmlhc2VzLnJhbmsgPT09IDEsXG4gICAgICAgICAgYEVycm9yIGluIGNvbnYyZFRyYW5zcG9zZTogYmlhc2VzIG11c3QgYmUgcmFuayAxLCBidXQgZ290ICcgK1xuICAgICAgICAgICAgICAncmFuayAke2JpYXNlcy5yYW5rfS5gKTtcbiAgICB9XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHguc2hhcGVbMl0gPT09IHdlaWdodHMuc2hhcGVbM10sXG4gICAgICAgIGBFcnJvciBpbiBjb252MmRUcmFuc3Bvc2U6IGRlcHRoIG9mIGlucHV0ICgke3guc2hhcGVbMl19KSBtdXN0IGAgK1xuICAgICAgICAgICAgYG1hdGNoIGlucHV0IGRlcHRoIGZvciB3ZWlnaHRzICR7d2VpZ2h0cy5zaGFwZVszXX0uYCk7XG5cbiAgICByZXR1cm4gdGhpcy50cmFjayhcbiAgICAgICAgdGhpcy5jb252MmRUcmFuc3Bvc2VJbnRlcm5hbCh4LCB3ZWlnaHRzLCBiaWFzZXMsIHN0cmlkZSwgcGFkKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvbnYyZFRyYW5zcG9zZUludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhc2VzOiBBcnJheTFEfG51bGwsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiBBcnJheTNEO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgMkQgbWF4IHBvb2xpbmcgb2YgYW4gaW1hZ2UuXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgbXVzdCBiZSByYW5rIDMuXG4gICAqIEBwYXJhbSBmU2l6ZSBUaGUgZmllbGQgc2l6ZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBzdHJpZGUgVGhlIHN0cmlkZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBwYWQgVGhlIHBhZGRpbmcgb2YgZWFjaCBzaWRlIG9mIHRoZSBpbnB1dCBOREFycmF5LiBXaWxsIHBhZCBlcXVhbGx5XG4gICAqIG9uIGFsbCBzaWRlcy5cbiAgICovXG4gIG1heFBvb2woeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgJ0Vycm9yIGluIG1heFBvb2w6IHggbXVzdCBiZSByYW5rIDMgYnV0IGdvdCByYW5rICcgKyB4LnJhbmsgKyAnLicpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMubWF4UG9vbEludGVybmFsKHgsIGZTaXplLCBzdHJpZGUsIHBhZCkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBtYXhQb29sSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgYmFja3Byb3Agb2YgYSBtYXggcG9vbC5cbiAgICogQHBhcmFtIGR5IFRoZSBkeSBlcnJvci5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IGltYWdlLCBtdXN0IGJlIHJhbmsgMy5cbiAgICogQHBhcmFtIGZTaXplIFRoZSBmaWVsZCBzaXplIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHN0cmlkZSBUaGUgc3RyaWRlIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHBhZCBUaGUgcGFkZGluZyBvZiBlYWNoIHNpZGUgb2YgdGhlIGlucHV0IE5EQXJyYXkuIFdpbGwgcGFkIGVxdWFsbHlcbiAgICogb24gYWxsIHNpZGVzLlxuICAgKi9cbiAgbWF4UG9vbEJhY2twcm9wKFxuICAgICAgZHk6IEFycmF5M0QsIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgZHkucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIG1heFBvb2xCYWNrcHJvcDogZHkgbXVzdCBiZSByYW5rIDMgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgICAgYCR7ZHkucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIG1heFBvb2xCYWNrcHJvcDogeCBtdXN0IGJlIHJhbmsgMyBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgICBgJHt4LnJhbmt9LmApO1xuXG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5tYXhQb29sQmFja3Byb3BJbnRlcm5hbChkeSwgeCwgZlNpemUsIHN0cmlkZSwgcGFkKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG1heFBvb2xCYWNrcHJvcEludGVybmFsKFxuICAgICAgZHk6IEFycmF5M0QsIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiBBcnJheTNEO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgMkQgbWluIHBvb2xpbmcgb2YgYW4gaW1hZ2UuXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgbXVzdCBiZSByYW5rIDMuXG4gICAqIEBwYXJhbSBmU2l6ZSBUaGUgZmllbGQgc2l6ZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBzdHJpZGUgVGhlIHN0cmlkZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBwYWQgVGhlIHBhZGRpbmcgb2YgZWFjaCBzaWRlIG9mIHRoZSBpbnB1dCBOREFycmF5LiBXaWxsIHBhZCBlcXVhbGx5XG4gICAqIG9uIGFsbCBzaWRlcy5cbiAgICovXG4gIG1pblBvb2woeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIG1pblBvb2w6IHggbXVzdCBiZSByYW5rIDMgYnV0IGdvdCByYW5rICR7eC5yYW5rfS5gKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLm1pblBvb2xJbnRlcm5hbCh4LCBmU2l6ZSwgc3RyaWRlLCBwYWQpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgbWluUG9vbEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIDJEIGF2ZXJhZ2UgcG9vbGluZyBvZiBhbiBpbWFnZS5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IGltYWdlLCBtdXN0IGJlIHJhbmsgMy5cbiAgICogQHBhcmFtIGZTaXplIFRoZSBmaWVsZCBzaXplIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHN0cmlkZSBUaGUgc3RyaWRlIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHBhZCBUaGUgcGFkZGluZyBvZiBlYWNoIHNpZGUgb2YgdGhlIGlucHV0IE5EQXJyYXkuIFdpbGwgcGFkIGVxdWFsbHlcbiAgICogb24gYWxsIHNpZGVzLlxuICAgKi9cbiAgYXZnUG9vbCh4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gYXZnUG9vbDogeCBtdXN0IGJlIHJhbmsgMyBidXQgZ290IHJhbmsgJHt4LnJhbmt9LmApO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuYXZnUG9vbEludGVybmFsKHgsIGZTaXplLCBzdHJpZGUsIHBhZCkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBhdmdQb29sSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEO1xuXG4gIC8qXG4gICAqIEJpbGluZWFyIHJlc2l6ZSBhIDNEIGFycmF5IHBlciBlYWNoIGNoYW5uZWwgdG8gYSBuZXcgMkQgc2hhcGUuXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBBcnJheTNELlxuICAgKiBAcGFyYW0gbmV3U2hhcGUyRCBUaGUgbmV3IHNoYXBlIHRvIHJlc2l6ZSB0aGUgQXJyYXkzRCB0by4gRWFjaCBjaGFubmVsIGlzXG4gICAqIHJlc2l6ZWQgaW5kaXZpZHVhbGx5LlxuICAgKiBAcGFyYW0gYWxpZ25Db3JuZXJzIEFuIG9wdGlvbmFsIGJvb2wuIERlZmF1bHRzIHRvIEZhbHNlLiBJZiB0cnVlLCByZXNjYWxlXG4gICAqIGlucHV0IGJ5IChuZXdfaGVpZ2h0IC0gMSkgLyAoaGVpZ2h0IC0gMSksIHdoaWNoIGV4YWN0bHkgYWxpZ25zIHRoZSA0XG4gICAqIGNvcm5lcnMgb2YgaW1hZ2VzIGFuZCByZXNpemVkIGltYWdlcy4gSWYgZmFsc2UsIHJlc2NhbGUgYnkgbmV3X2hlaWdodCAvXG4gICAqIGhlaWdodC4gVHJlYXQgc2ltaWxhcmx5IHRoZSB3aWR0aCBkaW1lbnNpb24uXG4gICAqL1xuICByZXNpemVCaWxpbmVhcjNEKFxuICAgICAgeDogQXJyYXkzRCwgbmV3U2hhcGUyRDogW251bWJlciwgbnVtYmVyXSwgYWxpZ25Db3JuZXJzID0gZmFsc2UpOiBBcnJheTNEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gcmVzaXplQmlsaW5lYXIzRDogeCBtdXN0IGJlIHJhbmsgMyBidXQgZ290IHJhbmsgJHt4LnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBuZXdTaGFwZTJELmxlbmd0aCA9PT0gMixcbiAgICAgICAgYEVycm9yIGluIHJlc2l6ZUJpbGluZWFyM0Q6IG5ldyBzaGFwZSBtdXN0IDJELCBidXQgZ290IHNoYXBlIGAgK1xuICAgICAgICAgICAgYCR7bmV3U2hhcGUyRH0uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2soXG4gICAgICAgIHRoaXMucmVzaXplQmlsaW5lYXIzREludGVybmFsKHgsIG5ld1NoYXBlMkQsIGFsaWduQ29ybmVycykpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCByZXNpemVCaWxpbmVhcjNESW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBuZXdTaGFwZTJEOiBbbnVtYmVyLCBudW1iZXJdLCBhbGlnbkNvcm5lcnM6IGJvb2xlYW4pOiBBcnJheTNEO1xuXG4gIC8qKlxuICAgKiBCYXRjaCBub3JtYWxpemF0aW9uIDNELiBNZWFuLCB2YXJpYW5jZSwgc2NhbGUsIGFuZCBvZmZzZXQgY2FuIGJlIG9mIHR3b1xuICAgKiBzaGFwZXM6IDEpIFRoZSBzYW1lIHNoYXBlIGFzIHRoZSBpbnB1dDogYW4gQXJyYXkzRC4gMikgSW4gdGhlIGNvbW1vbiBjYXNlLFxuICAgKiB0aGUgZGVwdGggZGltZW5zaW9uIGlzIHRoZSBsYXN0IGRpbWVuc2lvbiBvZiB4LCBzbyB0aGUgdmFsdWVzIHdvdWxkIGJlIGFuXG4gICAqIEFycmF5MUQgb2Ygc2hhcGUgW2RlcHRoXS5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqIEBwYXJhbSBtZWFuIEEgbWVhbiBOREFycmF5LlxuICAgKiBAcGFyYW0gdmFyaWFuY2UgQSB2YXJpYW5jZSBOREFycmF5LlxuICAgKiBAcGFyYW0gdmFyaWFuY2VFcHNpbG9uIEEgc21hbGwgZmxvYXQgbnVtYmVyIHRvIGF2b2lkIGRpdmlkaW5nIGJ5IDAuXG4gICAqIEBwYXJhbSBzY2FsZSBBIHNjYWxlIE5EQXJyYXkuXG4gICAqIEBwYXJhbSBvZmZzZXQgQW4gb2Zmc2V0IE5EQXJyYXkuXG4gICAqL1xuICBiYXRjaE5vcm1hbGl6YXRpb24zRChcbiAgICAgIHg6IEFycmF5M0QsIG1lYW46IEFycmF5M0R8QXJyYXkxRCwgdmFyaWFuY2U6IEFycmF5M0R8QXJyYXkxRCxcbiAgICAgIHZhcmlhbmNlRXBzaWxvbiA9IC4wMDEsIHNjYWxlPzogQXJyYXkzRHxBcnJheTFELFxuICAgICAgb2Zmc2V0PzogQXJyYXkzRHxBcnJheTFEKTogQXJyYXkzRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIGJhdGNoTm9ybWFsaXphdGlvbjNEOiB4IG11c3QgYmUgcmFuayAzIGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICAgIGAke3gucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIG1lYW4ucmFuayA9PT0gMyB8fCBtZWFuLnJhbmsgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBiYXRjaE5vcm1hbGl6YXRpb24zRDogbWVhbiBtdXN0IGJlIHJhbmsgMyBvciByYW5rIDEgYnV0IGAgK1xuICAgICAgICAgICAgYGdvdCByYW5rICR7bWVhbi5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdmFyaWFuY2UucmFuayA9PT0gMyB8fCB2YXJpYW5jZS5yYW5rID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gYmF0Y2hOb3JtYWxpemF0aW9uM0Q6IHZhcmlhbmNlIG11c3QgYmUgcmFuayAzIG9yIHJhbmsgMSBgICtcbiAgICAgICAgICAgIGBidXQgZ290IHJhbmsgJHt2YXJpYW5jZS5yYW5rfS5gKTtcbiAgICBpZiAoc2NhbGUgIT0gbnVsbCkge1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgc2NhbGUucmFuayA9PT0gMyB8fCBzY2FsZS5yYW5rID09PSAxLFxuICAgICAgICAgIGBFcnJvciBpbiBiYXRjaE5vcm1hbGl6YXRpb24zRDogc2NhbGUgbXVzdCBiZSByYW5rIDMgb3IgcmFuayAxIGAgK1xuICAgICAgICAgICAgICBgYnV0IGdvdCByYW5rICR7c2NhbGUhLnJhbmt9LmApO1xuICAgIH1cbiAgICBpZiAob2Zmc2V0ICE9IG51bGwpIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIG9mZnNldC5yYW5rID09PSAzIHx8IG9mZnNldC5yYW5rID09PSAxLFxuICAgICAgICAgIGBFcnJvciBpbiBiYXRjaE5vcm1hbGl6YXRpb24zRDogb2Zmc2V0IG11c3QgYmUgcmFuayAzIG9yIHJhbmsgMSBgICtcbiAgICAgICAgICAgICAgYGJ1dCBnb3QgcmFuayAke29mZnNldCEucmFua30uYCk7XG4gICAgfVxuXG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5iYXRjaE5vcm1hbGl6YXRpb24zREludGVybmFsKFxuICAgICAgICB4LCBtZWFuLCB2YXJpYW5jZSwgdmFyaWFuY2VFcHNpbG9uLCBzY2FsZSwgb2Zmc2V0KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGJhdGNoTm9ybWFsaXphdGlvbjNESW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBtZWFuOiBBcnJheTNEfEFycmF5MUQsIHZhcmlhbmNlOiBBcnJheTNEfEFycmF5MUQsXG4gICAgICB2YXJpYW5jZUVwc2lsb246IG51bWJlciwgc2NhbGU/OiBBcnJheTNEfEFycmF5MUQsXG4gICAgICBvZmZzZXQ/OiBBcnJheTNEfEFycmF5MUQpOiBBcnJheTNEO1xufVxuXG5leHBvcnQgZW51bSBNYXRyaXhPcmllbnRhdGlvbiB7XG4gIFJFR1VMQVIsXG4gIFRSQU5TUE9TRURcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uL21hdGgvY29udl91dGlsJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCAqIGFzIGNvbmNhdDNkX3V0aWwgZnJvbSAnLi9jb25jYXQzZF91dGlsJztcbmltcG9ydCAqIGFzIGNvcHkyRF91dGlsIGZyb20gJy4vY29weTJkX3V0aWwnO1xuaW1wb3J0IHtNYXRyaXhPcmllbnRhdGlvbiwgTkRBcnJheU1hdGh9IGZyb20gJy4vbWF0aCc7XG5pbXBvcnQge0FycmF5MUQsIEFycmF5MkQsIEFycmF5M0QsIEFycmF5NEQsIE5EQXJyYXksIFNjYWxhcn0gZnJvbSAnLi9uZGFycmF5JztcblxuZXhwb3J0IGNsYXNzIE5EQXJyYXlNYXRoQ1BVIGV4dGVuZHMgTkRBcnJheU1hdGgge1xuICBjb25zdHJ1Y3RvcihzYWZlTW9kZSA9IGZhbHNlKSB7XG4gICAgc3VwZXIoc2FmZU1vZGUpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGNsb25lSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KFxuICAgICAgICBuZGFycmF5LnNoYXBlLCB7dmFsdWVzOiBuZXcgRmxvYXQzMkFycmF5KG5kYXJyYXkuZ2V0VmFsdWVzKCkpfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgcmVzaGFwZUludGVybmFsPFQxIGV4dGVuZHMgTkRBcnJheSwgVDIgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIG5kYXJyYXk6IFQxLCBuZXdTaGFwZTogbnVtYmVyW10pOiBUMiB7XG4gICAgcmV0dXJuIHRoaXMuY2xvbmVJbnRlcm5hbChuZGFycmF5KS5yZXNoYXBlPFQyPihuZXdTaGFwZSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgc2xpY2UyREludGVybmFsKFxuICAgICAgaW5wdXQ6IEFycmF5MkQsIGJlZ2luUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgc2l6ZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSk6IEFycmF5MkQge1xuICAgIGNvbnN0IHJlc3VsdCA9IEFycmF5MkQuemVyb3Moc2l6ZVJvd0NvbCk7XG4gICAgdGhpcy5jb3B5MkRJbnRlcm5hbChcbiAgICAgICAgaW5wdXQsIGJlZ2luUm93Q29sLCBzaXplUm93Q29sLCByZXN1bHQsIFswLCAwXSwgc2l6ZVJvd0NvbCk7XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHByb3RlY3RlZCBjb3B5MkRJbnRlcm5hbChcbiAgICAgIHNvdXJjZTogQXJyYXkyRCwgc291cmNlQmVnaW5Sb3dDb2w6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBzb3VyY2VTaXplUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdLCBkZXN0OiBBcnJheTJELFxuICAgICAgZGVzdEJlZ2luUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgZGVzdFNpemVSb3dDb2w6IFtudW1iZXIsIG51bWJlcl0pOiB2b2lkIHtcbiAgICBjb3B5MkRfdXRpbC52YWxpZGF0ZVNoYXBlcyhzb3VyY2VTaXplUm93Q29sLCBkZXN0U2l6ZVJvd0NvbCk7XG4gICAgY29uc3Qgc3JjVmFsdWVzID0gc291cmNlLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IGRzdFZhbHVlcyA9IGRlc3QuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgbiA9IHNvdXJjZVNpemVSb3dDb2xbMF0gKiBzb3VyY2VTaXplUm93Q29sWzFdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbjsgKytpKSB7XG4gICAgICBjb25zdCBzcmNSb3cgPSBzb3VyY2VCZWdpblJvd0NvbFswXSArIE1hdGguZmxvb3IoaSAvIHNvdXJjZVNpemVSb3dDb2xbMV0pO1xuICAgICAgY29uc3Qgc3JjQ29sID0gc291cmNlQmVnaW5Sb3dDb2xbMV0gKyAoaSAlIHNvdXJjZVNpemVSb3dDb2xbMV0pO1xuICAgICAgY29uc3Qgc3JjT2ZmID0gc3JjUm93ICogc291cmNlLnNoYXBlWzFdICsgc3JjQ29sO1xuICAgICAgY29uc3QgZHN0Um93ID0gZGVzdEJlZ2luUm93Q29sWzBdICsgTWF0aC5mbG9vcihpIC8gZGVzdFNpemVSb3dDb2xbMV0pO1xuICAgICAgY29uc3QgZHN0Q29sID0gZGVzdEJlZ2luUm93Q29sWzFdICsgKGkgJSBkZXN0U2l6ZVJvd0NvbFsxXSk7XG4gICAgICBjb25zdCBkc3RPZmYgPSBkc3RSb3cgKiBkZXN0LnNoYXBlWzFdICsgZHN0Q29sO1xuICAgICAgZHN0VmFsdWVzW2RzdE9mZl0gPSBzcmNWYWx1ZXNbc3JjT2ZmXTtcbiAgICB9XG4gIH1cblxuICBwcm90ZWN0ZWQgY29uY2F0M0RJbnRlcm5hbCh4MTogQXJyYXkzRCwgeDI6IEFycmF5M0QsIGF4aXM6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIGNvbnN0IG91dHB1dFNoYXBlID1cbiAgICAgICAgY29uY2F0M2RfdXRpbC5jb21wdXRlQ29uY2F0M0RPdXRwdXRTaGFwZSh4MS5zaGFwZSwgeDIuc2hhcGUsIGF4aXMpO1xuXG4gICAgY29uc3QgdmFsdWVzID0gTkRBcnJheS56ZXJvczxBcnJheTNEPihvdXRwdXRTaGFwZSk7XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG91dHB1dFNoYXBlWzBdOyBpKyspIHtcbiAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgb3V0cHV0U2hhcGVbMV07IGorKykge1xuICAgICAgICBmb3IgKGxldCBrID0gMDsgayA8IG91dHB1dFNoYXBlWzJdOyBrKyspIHtcbiAgICAgICAgICAvLyBTaGFkZXIgYmVnaW5zLlxuICAgICAgICAgIGNvbnN0IGluZGV4OiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbaSwgaiwga107XG4gICAgICAgICAgbGV0IHZhbHVlOiBudW1iZXI7XG4gICAgICAgICAgaWYgKGluZGV4W2F4aXNdIDwgeDEuc2hhcGVbYXhpc10pIHtcbiAgICAgICAgICAgIHZhbHVlID0geDEuZ2V0KGksIGosIGspO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICBpbmRleFtheGlzXSAtPSB4MS5zaGFwZVtheGlzXTtcbiAgICAgICAgICAgIGNvbnN0IFtpMiwgajIsIGsyXSA9IGluZGV4O1xuICAgICAgICAgICAgdmFsdWUgPSB4Mi5nZXQoaTIsIGoyLCBrMik7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgdmFsdWVzLnNldCh2YWx1ZSwgaSwgaiwgayk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gdmFsdWVzO1xuICB9XG5cbiAgcHJvdGVjdGVkIHNjYWxhclBsdXNBcnJheUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihjOiBTY2FsYXIsIGE6IFQpOiBUIHtcbiAgICBjb25zdCByZXN1bHRWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGEuc2l6ZSk7XG4gICAgY29uc3QgYVZhbHVlcyA9IGEuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgY1ZhbCA9IGMuZ2V0KCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCByZXN1bHRWYWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIHJlc3VsdFZhbHVlc1tpXSA9IGNWYWwgKyBhVmFsdWVzW2ldO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KGEuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHNjYWxlZEFycmF5QWRkSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgYzE6IFNjYWxhciwgYTogVCwgYzI6IFNjYWxhciwgYjogVCkge1xuICAgIGNvbnN0IGNWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGEuc2l6ZSk7XG4gICAgY29uc3QgYVZhbHVlcyA9IGEuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgYlZhbHVlcyA9IGIuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgYzFWYWwgPSBjMS5nZXQoKTtcbiAgICBjb25zdCBjMlZhbCA9IGMyLmdldCgpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgY1ZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY1ZhbHVlc1tpXSA9IGMxVmFsICogYVZhbHVlc1tpXSArIGMyVmFsICogYlZhbHVlc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihhLnNoYXBlLCB7dmFsdWVzOiBjVmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgc2NhbGFyVGltZXNBcnJheUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihjOiBTY2FsYXIsIGE6IFQpOiBUIHtcbiAgICBjb25zdCBuZXdWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGEuc2l6ZSk7XG4gICAgY29uc3QgYVZhbHVlcyA9IGEuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgY1ZhbCA9IGMuZ2V0KCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBhVmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBuZXdWYWx1ZXNbaV0gPSBjVmFsICogYVZhbHVlc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihhLnNoYXBlLCB7dmFsdWVzOiBuZXdWYWx1ZXN9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBzY2FsYXJNaW51c0FycmF5SW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KGM6IFNjYWxhciwgYTogVCk6IFQge1xuICAgIGNvbnN0IG5lZ0EgPSB0aGlzLm5lZ0ludGVybmFsKGEpO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRoaXMuc2NhbGFyUGx1c0FycmF5SW50ZXJuYWwoYywgbmVnQSk7XG5cbiAgICBuZWdBLmRpc3Bvc2UoKTtcblxuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICBwcm90ZWN0ZWQgYXJyYXlNaW51c1NjYWxhckludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBjOiBTY2FsYXIpOiBUIHtcbiAgICBjb25zdCBuZWdDID0gdGhpcy5uZWdJbnRlcm5hbChjKTtcbiAgICBjb25zdCByZXN1bHQgPSB0aGlzLnNjYWxhclBsdXNBcnJheUludGVybmFsKG5lZ0MsIGEpO1xuXG4gICAgbmVnQy5kaXNwb3NlKCk7XG5cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHJvdGVjdGVkIG5lZ0ludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMuc2NhbGFyVGltZXNBcnJheUludGVybmFsKFNjYWxhci5ORUdfT05FLCBhKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBhZGRJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnNjYWxlZEFycmF5QWRkSW50ZXJuYWw8VD4oU2NhbGFyLk9ORSwgYSwgU2NhbGFyLk9ORSwgYik7XG4gIH1cblxuICBwcm90ZWN0ZWQgc3ViSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy5zY2FsZWRBcnJheUFkZEludGVybmFsPFQ+KFNjYWxhci5PTkUsIGEsIFNjYWxhci5ORUdfT05FLCBiKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBtYXRNdWxJbnRlcm5hbChcbiAgICAgIGE6IEFycmF5MkQsIGI6IEFycmF5MkQsIGFPcmllbnRhdGlvbiA9IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIsXG4gICAgICBiT3JpZW50YXRpb24gPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKTogQXJyYXkyRCB7XG4gICAgY29uc3Qgc2hhcmVkRGltID1cbiAgICAgICAgKGFPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyBhLnNoYXBlWzFdIDogYS5zaGFwZVswXTtcblxuICAgIGNvbnN0IGxlZnREaW0gPVxuICAgICAgICAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IGEuc2hhcGVbMF0gOiBhLnNoYXBlWzFdO1xuICAgIGNvbnN0IHJpZ2h0RGltID1cbiAgICAgICAgKGJPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyBiLnNoYXBlWzFdIDogYi5zaGFwZVswXTtcblxuICAgIGNvbnN0IG5vcm1hbEdldHRlciA9IChtYXRyaXg6IEFycmF5MkQsIGk6IG51bWJlciwgajogbnVtYmVyKSA9PlxuICAgICAgICBtYXRyaXguZ2V0KGksIGopO1xuICAgIGNvbnN0IHRyYW5zcG9zZWRHZXR0ZXIgPSAobWF0cml4OiBBcnJheTJELCBpOiBudW1iZXIsIGo6IG51bWJlcikgPT5cbiAgICAgICAgbWF0cml4LmdldChqLCBpKTtcblxuICAgIGNvbnN0IGFHZXR0ZXIgPSAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICAgIG5vcm1hbEdldHRlciA6XG4gICAgICAgIHRyYW5zcG9zZWRHZXR0ZXI7XG4gICAgY29uc3QgYkdldHRlciA9IChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgICAgbm9ybWFsR2V0dGVyIDpcbiAgICAgICAgdHJhbnNwb3NlZEdldHRlcjtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGxlZnREaW0gKiByaWdodERpbSk7XG4gICAgbGV0IGluZGV4ID0gMDtcblxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbGVmdERpbTsgKytpKSB7XG4gICAgICBmb3IgKGxldCBqID0gMDsgaiA8IHJpZ2h0RGltOyArK2opIHtcbiAgICAgICAgbGV0IHN1bSA9IDA7XG4gICAgICAgIGZvciAobGV0IGsgPSAwOyBrIDwgc2hhcmVkRGltOyArK2spIHtcbiAgICAgICAgICAvLyBUT0RPOiBvcHRpbWl6ZSBDUFUgbWF0bXVsLlxuICAgICAgICAgIHN1bSArPSBhR2V0dGVyKGEsIGksIGspICogYkdldHRlcihiLCBrLCBqKTtcbiAgICAgICAgfVxuICAgICAgICB2YWx1ZXNbaW5kZXgrK10gPSBzdW07XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBBcnJheTJELm5ldyhbbGVmdERpbSwgcmlnaHREaW1dLCB2YWx1ZXMpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGVsZW1lbnRXaXNlTXVsSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICBjb25zdCBuZXdWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGEuc2l6ZSk7XG4gICAgY29uc3QgYVZhbHVlcyA9IGEuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgYlZhbHVlcyA9IGIuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBhVmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBuZXdWYWx1ZXNbaV0gPSBhVmFsdWVzW2ldICogYlZhbHVlc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihhLnNoYXBlLCB7dmFsdWVzOiBuZXdWYWx1ZXN9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBlbGVtZW50V2lzZU11bEJyb2FkY2FzdEludGVybmFsKGE6IEFycmF5MkQsIGI6IEFycmF5MkQpOiBBcnJheTJEIHtcbiAgICBjb25zdCBtYXhSb3cgPSBNYXRoLm1heChhLnNoYXBlWzBdLCBiLnNoYXBlWzBdKTtcbiAgICBjb25zdCBtYXhDb2wgPSBNYXRoLm1heChhLnNoYXBlWzFdLCBiLnNoYXBlWzFdKTtcblxuICAgIGNvbnN0IHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobWF4Um93ICogbWF4Q29sKTtcbiAgICBsZXQgaW5kZXggPSAwO1xuICAgIGZvciAobGV0IHJvdyA9IDA7IHJvdyA8IG1heFJvdzsgcm93KyspIHtcbiAgICAgIGZvciAobGV0IGNvbCA9IDA7IGNvbCA8IG1heENvbDsgY29sKyspIHtcbiAgICAgICAgdmFsdWVzW2luZGV4KytdID0gYS5nZXQocm93ICUgYS5zaGFwZVswXSwgY29sICUgYS5zaGFwZVsxXSkgKlxuICAgICAgICAgICAgYi5nZXQocm93ICUgYi5zaGFwZVswXSwgY29sICUgYi5zaGFwZVsxXSk7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBBcnJheTJELm5ldyhbbWF4Um93LCBtYXhDb2xdLCB2YWx1ZXMpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGRpdmlkZUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgY29uc3QgbmV3VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShhLnNpemUpO1xuICAgIGNvbnN0IGFWYWx1ZXMgPSBhLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IGJWYWx1ZXMgPSBiLmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYVZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgbmV3VmFsdWVzW2ldID0gYVZhbHVlc1tpXSAvIGJWYWx1ZXNbaV07XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4oYS5zaGFwZSwge3ZhbHVlczogbmV3VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgc2NhbGFyRGl2aWRlZEJ5QXJyYXlJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYzogU2NhbGFyLCBhOiBUKTpcbiAgICAgIFQge1xuICAgIGNvbnN0IG5ld1ZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkoYS5zaXplKTtcbiAgICBjb25zdCBhVmFsdWVzID0gYS5nZXRWYWx1ZXMoKTtcbiAgICBjb25zdCBjVmFsdWUgPSBjLmdldCgpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYVZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgbmV3VmFsdWVzW2ldID0gY1ZhbHVlIC8gYVZhbHVlc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihhLnNoYXBlLCB7dmFsdWVzOiBuZXdWYWx1ZXN9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBhcnJheURpdmlkZWRCeVNjYWxhckludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBjOiBTY2FsYXIpOlxuICAgICAgVCB7XG4gICAgY29uc3QgbmV3VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShhLnNpemUpO1xuICAgIGNvbnN0IGFWYWx1ZXMgPSBhLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IGNWYWx1ZSA9IGMuZ2V0KCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBhVmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBuZXdWYWx1ZXNbaV0gPSBhVmFsdWVzW2ldIC8gY1ZhbHVlO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KGEuc2hhcGUsIHt2YWx1ZXM6IG5ld1ZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHN1bUludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGxldCBzdW0gPSAwO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIHN1bSArPSB2YWx1ZXNbaV07XG4gICAgfVxuICAgIHJldHVybiBTY2FsYXIubmV3KHN1bSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgYXJnTWluSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgbGV0IG1pbiA9IE51bWJlci5NQVhfVkFMVUU7XG4gICAgbGV0IG1pbkluZGV4ID0gLTE7XG4gICAgY29uc3QgdmFsdWVzID0gbmRhcnJheS5nZXRWYWx1ZXMoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3QgdmFsdWUgPSB2YWx1ZXNbaV07XG4gICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgICB9XG4gICAgICBpZiAodmFsdWUgPCBtaW4pIHtcbiAgICAgICAgbWluID0gdmFsdWU7XG4gICAgICAgIG1pbkluZGV4ID0gaTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIFNjYWxhci5uZXcobWluSW5kZXgpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGFyZ01heEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGxldCBtYXggPSBOdW1iZXIuTkVHQVRJVkVfSU5GSU5JVFk7XG4gICAgbGV0IG1heEluZGV4ID0gLTE7XG4gICAgY29uc3QgdmFsdWVzID0gbmRhcnJheS5nZXRWYWx1ZXMoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3QgdmFsdWUgPSB2YWx1ZXNbaV07XG4gICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgICB9XG4gICAgICBpZiAodmFsdWUgPiBtYXgpIHtcbiAgICAgICAgbWF4ID0gdmFsdWU7XG4gICAgICAgIG1heEluZGV4ID0gaTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIFNjYWxhci5uZXcobWF4SW5kZXgpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGFyZ01heEVxdWFsc0ludGVybmFsKHgxOiBOREFycmF5LCB4MjogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgY29uc3QgYXJnTWF4MSA9IHRoaXMuYXJnTWF4SW50ZXJuYWwoeDEpLmdldCgpO1xuICAgIGNvbnN0IGFyZ01heDIgPSB0aGlzLmFyZ01heEludGVybmFsKHgyKS5nZXQoKTtcbiAgICBpZiAoaXNOYU4oYXJnTWF4MSkgfHwgaXNOYU4oYXJnTWF4MikpIHtcbiAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgfVxuICAgIHJldHVybiBTY2FsYXIubmV3KCsoYXJnTWF4MSA9PT0gYXJnTWF4MikpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHRvcEtJbnRlcm5hbChuZGFycmF5OiBOREFycmF5LCBrOiBudW1iZXIpOlxuICAgICAge3ZhbHVlczogQXJyYXkxRCwgaW5kaWNlczogQXJyYXkxRH0ge1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgdmFsdWVzQW5kSW5kaWNlczogQXJyYXk8e3ZhbHVlOiBudW1iZXIsIGluZGV4OiBudW1iZXJ9PiA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICB2YWx1ZXNBbmRJbmRpY2VzLnB1c2goe3ZhbHVlOiB2YWx1ZXNbaV0sIGluZGV4OiBpfSk7XG4gICAgfVxuICAgIHZhbHVlc0FuZEluZGljZXMuc29ydCgoYSwgYikgPT4ge1xuICAgICAgcmV0dXJuIGIudmFsdWUgLSBhLnZhbHVlO1xuICAgIH0pO1xuICAgIGNvbnN0IHRvcGtWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGspO1xuICAgIGNvbnN0IHRvcGtJbmRpY2VzID0gbmV3IEZsb2F0MzJBcnJheShrKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGs7IGkrKykge1xuICAgICAgdG9wa1ZhbHVlc1tpXSA9IHZhbHVlc0FuZEluZGljZXNbaV0udmFsdWU7XG4gICAgICB0b3BrSW5kaWNlc1tpXSA9IHZhbHVlc0FuZEluZGljZXNbaV0uaW5kZXg7XG4gICAgfVxuICAgIHJldHVybiB7dmFsdWVzOiBBcnJheTFELm5ldyh0b3BrVmFsdWVzKSwgaW5kaWNlczogQXJyYXkxRC5uZXcodG9wa0luZGljZXMpfTtcbiAgfVxuXG4gIHByb3RlY3RlZCBtaW5JbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGxldCBtaW4gPSB2YWx1ZXNbMF07XG4gICAgZm9yIChsZXQgaSA9IDE7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgaWYgKGlzTmFOKHZhbHVlKSkge1xuICAgICAgICByZXR1cm4gU2NhbGFyLm5ldyhOYU4pO1xuICAgICAgfVxuICAgICAgaWYgKHZhbHVlIDwgbWluKSB7XG4gICAgICAgIG1pbiA9IHZhbHVlO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gU2NhbGFyLm5ldyhtaW4pO1xuICB9XG5cbiAgcHJvdGVjdGVkIG1heEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgbGV0IG1heCA9IHZhbHVlc1swXTtcbiAgICBmb3IgKGxldCBpID0gMTsgaSA8IHZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3QgdmFsdWUgPSB2YWx1ZXNbaV07XG4gICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgICB9XG4gICAgICBpZiAodmFsdWUgPiBtYXgpIHtcbiAgICAgICAgbWF4ID0gdmFsdWU7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBTY2FsYXIubmV3KG1heCk7XG4gIH1cblxuICBwcm90ZWN0ZWQgZXhwSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IG5ld1ZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkodmFsdWVzLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIG5ld1ZhbHVlc1tpXSA9IE1hdGguZXhwKHZhbHVlc1tpXSk7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmRhcnJheS5zaGFwZSwge3ZhbHVlczogbmV3VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgbG9nSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IG5ld1ZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkodmFsdWVzLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgbmV3VmFsdWVzW2ldID0gTWF0aC5sb2codmFsdWUpO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IG5ld1ZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIGxvZ1N1bUV4cEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGNvbnN0IHhNYXggPSB0aGlzLm1heChuZGFycmF5KTtcbiAgICBjb25zdCBhID0gdGhpcy5hcnJheU1pbnVzU2NhbGFyKG5kYXJyYXksIHhNYXgpO1xuICAgIGNvbnN0IGIgPSB0aGlzLmV4cChhKTtcbiAgICBjb25zdCBjID0gdGhpcy5zdW0oYik7XG4gICAgY29uc3QgZCA9IHRoaXMubG9nKGMpO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRoaXMuYWRkKHhNYXgsIGQpO1xuXG4gICAgeE1heC5kaXNwb3NlKCk7XG4gICAgYS5kaXNwb3NlKCk7XG4gICAgYi5kaXNwb3NlKCk7XG4gICAgYy5kaXNwb3NlKCk7XG4gICAgZC5kaXNwb3NlKCk7XG5cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHJvdGVjdGVkIHJlbHVJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIGNvbnN0IHJlc3VsdFZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobmRhcnJheS5zaXplKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICByZXN1bHRWYWx1ZXNbaV0gPSBNYXRoLm1heCgwLCB2YWx1ZXNbaV0pO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHNpZ21vaWRJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIGNvbnN0IHJlc3VsdFZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobmRhcnJheS5zaXplKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICByZXN1bHRWYWx1ZXNbaV0gPSAxIC8gKDEgKyBNYXRoLmV4cCgtdmFsdWVzW2ldKSk7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmRhcnJheS5zaGFwZSwge3ZhbHVlczogcmVzdWx0VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgdGFuaEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgY29uc3QgcmVzdWx0VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShuZGFycmF5LnNpemUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIHJlc3VsdFZhbHVlc1tpXSA9IHV0aWwudGFuaCh2YWx1ZXNbaV0pO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHNpbkludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgY29uc3QgcmVzdWx0VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShuZGFycmF5LnNpemUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIHJlc3VsdFZhbHVlc1tpXSA9IE1hdGguc2luKHZhbHVlc1tpXSk7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmRhcnJheS5zaGFwZSwge3ZhbHVlczogcmVzdWx0VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgc3RlcEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgY29uc3QgcmVzdWx0VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShuZGFycmF5LnNpemUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgcmVzdWx0VmFsdWVzW2ldID0gdmFsdWUgPiAwID8gMSA6ICh2YWx1ZSA8IDAgPyAwIDogdmFsdWUpO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgLyoqXG4gICAqIGltYWdlIGlzIG9mIHNoYXBlIFtyLCBjLCBkMV0uXG4gICAqIHdlaWdodHMgaXMgb2Ygc2hhcGUgW0YsIEYsIGQxLCBkMl0uXG4gICAqL1xuICBwcm90ZWN0ZWQgY29udjJkSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCB3ZWlnaHRzOiBBcnJheTRELCBiaWFzZXM6IEFycmF5MUR8bnVsbCwgc3RyaWRlOiBudW1iZXIsXG4gICAgICBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIGNvbnN0IFt4Um93cywgeENvbHMsIGlucHV0RGVwdGhdID0geC5zaGFwZTtcbiAgICBjb25zdCBmaWVsZFNpemUgPSB3ZWlnaHRzLnNoYXBlWzBdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gd2VpZ2h0cy5zaGFwZVszXTtcbiAgICBjb25zdCBvdXRwdXRTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgICAgW3hSb3dzLCB4Q29scywgaW5wdXREZXB0aF0sIGZpZWxkU2l6ZSwgb3V0cHV0RGVwdGgsIHN0cmlkZSwgcGFkKTtcbiAgICBjb25zdCB5ID0gQXJyYXkzRC56ZXJvcyhvdXRwdXRTaGFwZSk7XG4gICAgZm9yIChsZXQgZDIgPSAwOyBkMiA8IG91dHB1dERlcHRoOyArK2QyKSB7XG4gICAgICBmb3IgKGxldCB5UiA9IDA7IHlSIDwgeS5zaGFwZVswXTsgKyt5Uikge1xuICAgICAgICBjb25zdCB4UkNvcm5lciA9IHlSICogc3RyaWRlIC0gcGFkO1xuICAgICAgICBjb25zdCB4Uk1pbiA9IE1hdGgubWF4KDAsIHhSQ29ybmVyKTtcbiAgICAgICAgY29uc3QgeFJNYXggPSBNYXRoLm1pbih4Um93cywgZmllbGRTaXplICsgeFJDb3JuZXIpO1xuICAgICAgICBmb3IgKGxldCB5QyA9IDA7IHlDIDwgeS5zaGFwZVsxXTsgKyt5Qykge1xuICAgICAgICAgIGNvbnN0IHhDQ29ybmVyID0geUMgKiBzdHJpZGUgLSBwYWQ7XG4gICAgICAgICAgY29uc3QgeENNaW4gPSBNYXRoLm1heCgwLCB4Q0Nvcm5lcik7XG4gICAgICAgICAgY29uc3QgeENNYXggPSBNYXRoLm1pbih4Q29scywgZmllbGRTaXplICsgeENDb3JuZXIpO1xuICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB4UiA9IHhSTWluOyB4UiA8IHhSTWF4OyArK3hSKSB7XG4gICAgICAgICAgICBjb25zdCB3UiA9IHhSIC0geFJDb3JuZXI7XG4gICAgICAgICAgICBmb3IgKGxldCB4QyA9IHhDTWluOyB4QyA8IHhDTWF4OyArK3hDKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHdDID0geEMgLSB4Q0Nvcm5lcjtcbiAgICAgICAgICAgICAgZm9yIChsZXQgZDEgPSAwOyBkMSA8IGlucHV0RGVwdGg7ICsrZDEpIHtcbiAgICAgICAgICAgICAgICBjb25zdCBwaXhlbCA9IHguZ2V0KHhSLCB4QywgZDEpO1xuICAgICAgICAgICAgICAgIGNvbnN0IHdlaWdodCA9IHdlaWdodHMuZ2V0KHdSLCB3QywgZDEsIGQyKTtcbiAgICAgICAgICAgICAgICBkb3RQcm9kICs9IHBpeGVsICogd2VpZ2h0O1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIGNvbnN0IGJpYXMgPSAoYmlhc2VzICE9IG51bGwpID8gYmlhc2VzLmdldChkMikgOiAwO1xuICAgICAgICAgIHkuc2V0KGRvdFByb2QgKyBiaWFzLCB5UiwgeUMsIGQyKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfVxuXG4gIHByb3RlY3RlZCBjb252MmRCYWNrUHJvcEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZHk6IEFycmF5M0QsIHdlaWdodHM6IEFycmF5NEQsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiB7ZHg6IEFycmF5M0QsIGR3OiBBcnJheTRELCBkYjogQXJyYXkxRH0ge1xuICAgIGNvbnN0IGZTaXplID0gd2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBkdyA9IHRoaXMuY29udjJkRGVyV2VpZ2h0cyh4LCBkeSwgZlNpemUsIHN0cmlkZSwgcGFkKTtcbiAgICBjb25zdCBkYiA9IHRoaXMuY29udjJkRGVyQmlhcyhkeSk7XG4gICAgY29uc3QgZHggPSB0aGlzLmNvbnYyZFRyYW5zcG9zZUludGVybmFsKGR5LCB3ZWlnaHRzLCBudWxsLCBzdHJpZGUsIHBhZCk7XG4gICAgcmV0dXJuIHtkeCwgZGIsIGR3fTtcbiAgfVxuXG4gIC8qKlxuICAgKiBpbWFnZSBpcyBvZiBzaGFwZSBbciwgYywgZDFdLlxuICAgKiB3ZWlnaHRzIGlzIG9mIHNoYXBlIFtGLCBGLCBkMSwgZDJdLlxuICAgKi9cbiAgcHJvdGVjdGVkIGNvbnYyZFRyYW5zcG9zZUludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhc2VzOiBBcnJheTFEfG51bGwsIG9yaWdTdHJpZGU6IG51bWJlcixcbiAgICAgIG9yaWdQYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIGNvbnN0IGZTaXplID0gd2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBwYWQgPSBmU2l6ZSAtIDEgLSBvcmlnUGFkO1xuICAgIGNvbnN0IG9yaWdJbnB1dERlcHRoID0gd2VpZ2h0cy5zaGFwZVsyXTtcbiAgICBjb25zdCBvcmlnT3V0cHV0RGVwdGggPSB3ZWlnaHRzLnNoYXBlWzNdO1xuICAgIGNvbnN0IFt4Um93cywgeENvbHMsIHhEZXB0aF0gPSB4LnNoYXBlO1xuXG4gICAgLy8gRGlsYXRlIHRoZSBpbnB1dC5cbiAgICBjb25zdCB4Um93c0RpbGF0ZWQgPSAoeFJvd3MgLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuICAgIGNvbnN0IHhDb2xzRGlsYXRlZCA9ICh4Q29scyAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG5cbiAgICBjb25zdCBvdXRwdXRTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgICAgW3hSb3dzRGlsYXRlZCwgeENvbHNEaWxhdGVkLCBvcmlnT3V0cHV0RGVwdGhdLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGgsIDEsXG4gICAgICAgIHBhZCk7XG4gICAgY29uc3QgeSA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICAgIGZvciAobGV0IGQyID0gMDsgZDIgPCBvcmlnSW5wdXREZXB0aDsgKytkMikge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IHkuc2hhcGVbMF07ICsreVIpIHtcbiAgICAgICAgY29uc3QgeFJDb3JuZXIgPSB5UiAtIHBhZDtcbiAgICAgICAgY29uc3QgeFJNaW4gPSBNYXRoLm1heCgwLCBNYXRoLmNlaWwoeFJDb3JuZXIgLyBvcmlnU3RyaWRlKSk7XG4gICAgICAgIGNvbnN0IHhSTWF4ID0gTWF0aC5taW4oeFJvd3MsIChmU2l6ZSArIHhSQ29ybmVyKSAvIG9yaWdTdHJpZGUpO1xuXG4gICAgICAgIGZvciAobGV0IHlDID0gMDsgeUMgPCB5LnNoYXBlWzFdOyArK3lDKSB7XG4gICAgICAgICAgY29uc3QgeENDb3JuZXIgPSB5QyAtIHBhZDtcbiAgICAgICAgICBjb25zdCB4Q01pbiA9IE1hdGgubWF4KDAsIE1hdGguY2VpbCh4Q0Nvcm5lciAvIG9yaWdTdHJpZGUpKTtcbiAgICAgICAgICBjb25zdCB4Q01heCA9IE1hdGgubWluKHhDb2xzLCAoZlNpemUgKyB4Q0Nvcm5lcikgLyBvcmlnU3RyaWRlKTtcblxuICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB4UiA9IHhSTWluOyB4UiA8IHhSTWF4OyArK3hSKSB7XG4gICAgICAgICAgICBjb25zdCB3UiA9IHhSICogb3JpZ1N0cmlkZSAtIHhSQ29ybmVyO1xuXG4gICAgICAgICAgICBmb3IgKGxldCB4QyA9IHhDTWluOyB4QyA8IHhDTWF4OyArK3hDKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHdDID0geEMgKiBvcmlnU3RyaWRlIC0geENDb3JuZXI7XG5cbiAgICAgICAgICAgICAgZm9yIChsZXQgZDEgPSAwOyBkMSA8IG9yaWdPdXRwdXREZXB0aDsgKytkMSkge1xuICAgICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geC5nZXQoeFIsIHhDLCBkMSk7XG4gICAgICAgICAgICAgICAgY29uc3Qgd2VpZ2h0ID1cbiAgICAgICAgICAgICAgICAgICAgd2VpZ2h0cy5nZXQoZlNpemUgLSAxIC0gd1IsIGZTaXplIC0gMSAtIHdDLCBkMiwgZDEpO1xuICAgICAgICAgICAgICAgIGRvdFByb2QgKz0gcGl4ZWwgKiB3ZWlnaHQ7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgY29uc3QgYmlhcyA9IGJpYXNlcyAhPSBudWxsID8gYmlhc2VzLmdldChkMikgOiAwO1xuICAgICAgICAgIHkuc2V0KGRvdFByb2QgKyBiaWFzLCB5UiwgeUMsIGQyKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfVxuXG4gIC8qKlxuICAgKiBpbWFnZSBpcyBvZiBzaGFwZSBbciwgYywgZDFdLlxuICAgKiB3ZWlnaHRzIGlzIG9mIHNoYXBlIFtGLCBGLCBkMSwgZDJdLlxuICAgKi9cbiAgcHJvdGVjdGVkIGNvbnYyZFRyYW5zcG9zZVNoYWRlckxpa2UoXG4gICAgICB4OiBBcnJheTNELCBvcmlnV2VpZ2h0czogQXJyYXk0RCwgb3JpZ1N0cmlkZTogbnVtYmVyLFxuICAgICAgb3JpZ1BhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgY29uc3QgZlNpemUgPSBvcmlnV2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBwYWQgPSBmU2l6ZSAtIDEgLSBvcmlnUGFkO1xuICAgIGNvbnN0IG9yaWdJbnB1dERlcHRoID0gb3JpZ1dlaWdodHMuc2hhcGVbMl07XG4gICAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gb3JpZ1dlaWdodHMuc2hhcGVbM107XG4gICAgY29uc3QgW3hSb3dzLCB4Q29scywgeERlcHRoXSA9IHguc2hhcGU7XG5cbiAgICAvLyBEaWxhdGUgdGhlIGlucHV0LlxuICAgIGNvbnN0IHhSb3dzRGlsYXRlZCA9ICh4Um93cyAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG4gICAgY29uc3QgeENvbHNEaWxhdGVkID0gKHhDb2xzIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcblxuICAgIGNvbnN0IG91dHB1dFNoYXBlID0gY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKFxuICAgICAgICBbeFJvd3NEaWxhdGVkLCB4Q29sc0RpbGF0ZWQsIG9yaWdPdXRwdXREZXB0aF0sIGZTaXplLCBvcmlnSW5wdXREZXB0aCwgMSxcbiAgICAgICAgcGFkKTtcbiAgICBjb25zdCB5ID0gQXJyYXkzRC56ZXJvcyhvdXRwdXRTaGFwZSk7XG5cbiAgICBmb3IgKGxldCBkMiA9IDA7IGQyIDwgb3JpZ0lucHV0RGVwdGg7ICsrZDIpIHtcbiAgICAgIGZvciAobGV0IHlSID0gMDsgeVIgPCB5LnNoYXBlWzBdOyArK3lSKSB7XG4gICAgICAgIGZvciAobGV0IHlDID0gMDsgeUMgPCB5LnNoYXBlWzFdOyArK3lDKSB7XG4gICAgICAgICAgLy8gU2hhZGVyIGNvZGUgYmVnaW5zLlxuICAgICAgICAgIGNvbnN0IHhSQ29ybmVyID0geVIgLSBwYWQ7XG4gICAgICAgICAgY29uc3QgeENDb3JuZXIgPSB5QyAtIHBhZDtcbiAgICAgICAgICBsZXQgZG90UHJvZCA9IDA7XG4gICAgICAgICAgZm9yIChsZXQgd1IgPSAwOyB3UiA8IGZTaXplOyArK3dSKSB7XG4gICAgICAgICAgICBjb25zdCB4UiA9ICh4UkNvcm5lciArIHdSKSAvIG9yaWdTdHJpZGU7XG4gICAgICAgICAgICBpZiAoeFIgPCAwIHx8IHhSID49IHhSb3dzIHx8IE1hdGguZmxvb3IoeFIpICE9PSB4Uikge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGZvciAobGV0IHdDID0gMDsgd0MgPCBmU2l6ZTsgKyt3Qykge1xuICAgICAgICAgICAgICBjb25zdCB4QyA9ICh4Q0Nvcm5lciArIHdDKSAvIG9yaWdTdHJpZGU7XG4gICAgICAgICAgICAgIGlmICh4QyA8IDAgfHwgeEMgPj0geENvbHMgfHwgTWF0aC5mbG9vcih4QykgIT09IHhDKSB7XG4gICAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgZm9yIChsZXQgZDEgPSAwOyBkMSA8IG9yaWdPdXRwdXREZXB0aDsgKytkMSkge1xuICAgICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geC5nZXQoeFIsIHhDLCBkMSk7XG4gICAgICAgICAgICAgICAgY29uc3Qgd2VpZ2h0ID1cbiAgICAgICAgICAgICAgICAgICAgb3JpZ1dlaWdodHMuZ2V0KGZTaXplIC0gMSAtIHdSLCBmU2l6ZSAtIDEgLSB3QywgZDIsIGQxKTtcbiAgICAgICAgICAgICAgICBkb3RQcm9kICs9IHBpeGVsICogd2VpZ2h0O1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIHkuc2V0KGRvdFByb2QsIHlSLCB5QywgZDIpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiB5O1xuICB9XG5cbiAgY29udjJkRGVyV2VpZ2h0cyhcbiAgICAgIHg6IEFycmF5M0QsIGRZOiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHplcm9QYWQ6IG51bWJlcik6IEFycmF5NEQge1xuICAgIGNvbnN0IGlucHV0RGVwdGggPSB4LnNoYXBlWzJdO1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gZFkuc2hhcGVbMl07XG4gICAgY29uc3Qgd2VpZ2h0c1NoYXBlID1cbiAgICAgICAgY29udl91dGlsLmNvbXB1dGVXZWlnaHRzU2hhcGU0RChpbnB1dERlcHRoLCBvdXRwdXREZXB0aCwgZlNpemUpO1xuICAgIGNvbnN0IGRXID0gQXJyYXk0RC56ZXJvcyh3ZWlnaHRzU2hhcGUpO1xuXG4gICAgY29uc3QgeU51bVJvd3MgPSBkWS5zaGFwZVswXTtcbiAgICBjb25zdCB5TnVtQ29scyA9IGRZLnNoYXBlWzFdO1xuICAgIGNvbnN0IHhOdW1Sb3dzID0geC5zaGFwZVswXTtcbiAgICBjb25zdCB4TnVtQ29scyA9IHguc2hhcGVbMV07XG5cbiAgICBmb3IgKGxldCB3UiA9IDA7IHdSIDwgZlNpemU7ICsrd1IpIHtcbiAgICAgIGNvbnN0IHlSTWluID0gTWF0aC5tYXgoMCwgTWF0aC5jZWlsKCh6ZXJvUGFkIC0gd1IpIC8gc3RyaWRlKSk7XG4gICAgICBjb25zdCB5Uk1heCA9IE1hdGgubWluKHlOdW1Sb3dzLCAoeE51bVJvd3MgKyB6ZXJvUGFkIC0gd1IpIC8gc3RyaWRlKTtcblxuICAgICAgZm9yIChsZXQgd0MgPSAwOyB3QyA8IGZTaXplOyArK3dDKSB7XG4gICAgICAgIGNvbnN0IHlDTWluID0gTWF0aC5tYXgoMCwgTWF0aC5jZWlsKCh6ZXJvUGFkIC0gd0MpIC8gc3RyaWRlKSk7XG4gICAgICAgIGNvbnN0IHlDTWF4ID0gTWF0aC5taW4oeU51bUNvbHMsICh4TnVtQ29scyArIHplcm9QYWQgLSB3QykgLyBzdHJpZGUpO1xuXG4gICAgICAgIGZvciAobGV0IGQxID0gMDsgZDEgPCBpbnB1dERlcHRoOyArK2QxKSB7XG4gICAgICAgICAgZm9yIChsZXQgZDIgPSAwOyBkMiA8IG91dHB1dERlcHRoOyArK2QyKSB7XG4gICAgICAgICAgICAvLyBOZWVkIHRvIGNvbnZvbHZlLlxuICAgICAgICAgICAgbGV0IGRvdFByb2QgPSAwO1xuICAgICAgICAgICAgZm9yIChsZXQgeVIgPSB5Uk1pbjsgeVIgPCB5Uk1heDsgKyt5Uikge1xuICAgICAgICAgICAgICBjb25zdCB4UiA9IHdSICsgeVIgKiBzdHJpZGUgLSB6ZXJvUGFkO1xuICAgICAgICAgICAgICBmb3IgKGxldCB5QyA9IHlDTWluOyB5QyA8IHlDTWF4OyArK3lDKSB7XG4gICAgICAgICAgICAgICAgY29uc3QgeEMgPSB3QyArIHlDICogc3RyaWRlIC0gemVyb1BhZDtcbiAgICAgICAgICAgICAgICBkb3RQcm9kICs9IHguZ2V0KHhSLCB4QywgZDEpICogZFkuZ2V0KHlSLCB5QywgZDIpO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBkVy5zZXQoZG90UHJvZCwgd1IsIHdDLCBkMSwgZDIpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gZFc7XG4gIH1cblxuICBjb252MmREZXJCaWFzKGRZOiBBcnJheTNEKTogQXJyYXkxRCB7XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSBkWS5zaGFwZVsyXTtcbiAgICBjb25zdCBudW1Sb3dzID0gZFkuc2hhcGVbMF07XG4gICAgY29uc3QgbnVtQ29scyA9IGRZLnNoYXBlWzFdO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkob3V0cHV0RGVwdGgpO1xuICAgIGZvciAobGV0IGQyID0gMDsgZDIgPCBvdXRwdXREZXB0aDsgKytkMikge1xuICAgICAgbGV0IHN1bSA9IDA7XG4gICAgICBmb3IgKGxldCByID0gMDsgciA8IG51bVJvd3M7ICsrcikge1xuICAgICAgICBmb3IgKGxldCBjID0gMDsgYyA8IG51bUNvbHM7ICsrYykge1xuICAgICAgICAgIHN1bSArPSBkWS5nZXQociwgYywgZDIpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICB2YWx1ZXNbZDJdID0gc3VtO1xuICAgIH1cbiAgICByZXR1cm4gQXJyYXkxRC5uZXcodmFsdWVzKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBzd2l0Y2hEaW1JbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4odDogVCwgbmV3RGltOiBudW1iZXJbXSk6IFQge1xuICAgIGNvbnN0IG5ld1NoYXBlOiBudW1iZXJbXSA9IG5ldyBBcnJheSh0LnJhbmspO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbmV3U2hhcGUubGVuZ3RoOyBpKyspIHtcbiAgICAgIG5ld1NoYXBlW2ldID0gdC5zaGFwZVtuZXdEaW1baV1dO1xuICAgIH1cbiAgICBjb25zdCByZXN1bHRWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHQuc2l6ZSk7XG4gICAgY29uc3QgdmFsdWVzID0gdC5nZXRWYWx1ZXMoKTtcbiAgICBjb25zdCByZXN1bHQgPSBOREFycmF5Lm1ha2U8VD4obmV3U2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdC5zaXplOyArK2kpIHtcbiAgICAgIGNvbnN0IGxvYyA9IHQuaW5kZXhUb0xvYyhpKTtcblxuICAgICAgLy8gUGVybXV0ZSBsb2NhdGlvbi5cbiAgICAgIGNvbnN0IG5ld0xvYzogbnVtYmVyW10gPSBuZXcgQXJyYXkobG9jLmxlbmd0aCk7XG4gICAgICBmb3IgKGxldCBpID0gMDsgaSA8IG5ld0xvYy5sZW5ndGg7IGkrKykge1xuICAgICAgICBuZXdMb2NbaV0gPSBsb2NbbmV3RGltW2ldXTtcbiAgICAgIH1cblxuICAgICAgY29uc3QgbmV3SW5kZXggPSByZXN1bHQubG9jVG9JbmRleChuZXdMb2MpO1xuICAgICAgcmVzdWx0VmFsdWVzW25ld0luZGV4XSA9IHZhbHVlc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHByaXZhdGUgcG9vbChcbiAgICAgIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcixcbiAgICAgIHBvb2xUeXBlOiAnbWF4J3wnbWluJ3wnYXZnJykge1xuICAgIGNvbnN0IFt4Um93cywgeENvbHMsIGRlcHRoXSA9IHguc2hhcGU7XG4gICAgY29uc3Qgb3V0cHV0U2hhcGUgPSBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgIFt4Um93cywgeENvbHMsIGRlcHRoXSwgZlNpemUsIGRlcHRoLCBzdHJpZGUsIHBhZCk7XG4gICAgY29uc3QgeSA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICAgIGZvciAobGV0IGQgPSAwOyBkIDwgZGVwdGg7ICsrZCkge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IHkuc2hhcGVbMF07ICsreVIpIHtcbiAgICAgICAgY29uc3QgeFJDb3JuZXIgPSB5UiAqIHN0cmlkZSAtIHBhZDtcbiAgICAgICAgY29uc3QgeFJNaW4gPSBNYXRoLm1heCgwLCB4UkNvcm5lcik7XG4gICAgICAgIGNvbnN0IHhSTWF4ID0gTWF0aC5taW4oeFJvd3MsIGZTaXplICsgeFJDb3JuZXIpO1xuICAgICAgICBmb3IgKGxldCB5QyA9IDA7IHlDIDwgeS5zaGFwZVsxXTsgKyt5Qykge1xuICAgICAgICAgIGNvbnN0IHhDQ29ybmVyID0geUMgKiBzdHJpZGUgLSBwYWQ7XG4gICAgICAgICAgY29uc3QgeENNaW4gPSBNYXRoLm1heCgwLCB4Q0Nvcm5lcik7XG4gICAgICAgICAgY29uc3QgeENNYXggPSBNYXRoLm1pbih4Q29scywgZlNpemUgKyB4Q0Nvcm5lcik7XG5cblxuICAgICAgICAgIGxldCBtaW5NYXhWYWx1ZSA9XG4gICAgICAgICAgICAgIChwb29sVHlwZSA9PT0gJ21heCcgPyBOdW1iZXIuTkVHQVRJVkVfSU5GSU5JVFkgOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgTnVtYmVyLlBPU0lUSVZFX0lORklOSVRZKTtcbiAgICAgICAgICBsZXQgYXZnVmFsdWUgPSAwO1xuXG4gICAgICAgICAgZm9yIChsZXQgeFIgPSB4Uk1pbjsgeFIgPCB4Uk1heDsgKyt4Uikge1xuICAgICAgICAgICAgY29uc3Qgd1IgPSB4UiAtIHhSQ29ybmVyO1xuICAgICAgICAgICAgZm9yIChsZXQgeEMgPSB4Q01pbjsgeEMgPCB4Q01heDsgKyt4Qykge1xuICAgICAgICAgICAgICBjb25zdCB3QyA9IHhDIC0geENDb3JuZXI7XG4gICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geC5nZXQoeFIsIHhDLCBkKTtcbiAgICAgICAgICAgICAgaWYgKGlzTmFOKHBpeGVsKSkge1xuICAgICAgICAgICAgICAgIG1pbk1heFZhbHVlID0gTmFOO1xuICAgICAgICAgICAgICAgIGF2Z1ZhbHVlID0gTmFOO1xuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIGlmICgocG9vbFR5cGUgPT09ICdtYXgnICYmIHBpeGVsID4gbWluTWF4VmFsdWUpIHx8XG4gICAgICAgICAgICAgICAgICAocG9vbFR5cGUgPT09ICdtaW4nICYmIHBpeGVsIDwgbWluTWF4VmFsdWUpKSB7XG4gICAgICAgICAgICAgICAgbWluTWF4VmFsdWUgPSBwaXhlbDtcbiAgICAgICAgICAgICAgfSBlbHNlIGlmIChwb29sVHlwZSA9PT0gJ2F2ZycpIHtcbiAgICAgICAgICAgICAgICBhdmdWYWx1ZSArPSBwaXhlbCAvIChmU2l6ZSAqIGZTaXplKTtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgaWYgKGlzTmFOKG1pbk1heFZhbHVlKSkge1xuICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgeS5zZXQocG9vbFR5cGUgPT09ICdhdmcnID8gYXZnVmFsdWUgOiBtaW5NYXhWYWx1ZSwgeVIsIHlDLCBkKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfVxuXG4gIHByb3RlY3RlZCBtYXhQb29sSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICByZXR1cm4gdGhpcy5wb29sKHgsIGZTaXplLCBzdHJpZGUsIHBhZCwgJ21heCcpO1xuICB9XG5cbiAgbWF4UG9vbFBvc2l0aW9ucyh4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpIHtcbiAgICBjb25zdCBbeFJvd3MsIHhDb2xzLCBkZXB0aF0gPSB4LnNoYXBlO1xuICAgIGNvbnN0IG91dHB1dFNoYXBlID1cbiAgICAgICAgY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKHguc2hhcGUsIGZTaXplLCBkZXB0aCwgc3RyaWRlLCBwYWQpO1xuICAgIGNvbnN0IG1heFBvc2l0aW9ucyA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICAgIGZvciAobGV0IGQgPSAwOyBkIDwgZGVwdGg7ICsrZCkge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IG91dHB1dFNoYXBlWzBdOyArK3lSKSB7XG4gICAgICAgIGNvbnN0IHhSQ29ybmVyID0geVIgKiBzdHJpZGUgLSBwYWQ7XG4gICAgICAgIGNvbnN0IHhSTWluID0gTWF0aC5tYXgoMCwgeFJDb3JuZXIpO1xuICAgICAgICBjb25zdCB4Uk1heCA9IE1hdGgubWluKHhSb3dzLCBmU2l6ZSArIHhSQ29ybmVyKTtcbiAgICAgICAgZm9yIChsZXQgeUMgPSAwOyB5QyA8IG91dHB1dFNoYXBlWzFdOyArK3lDKSB7XG4gICAgICAgICAgY29uc3QgeENDb3JuZXIgPSB5QyAqIHN0cmlkZSAtIHBhZDtcbiAgICAgICAgICBjb25zdCB4Q01pbiA9IE1hdGgubWF4KDAsIHhDQ29ybmVyKTtcbiAgICAgICAgICBjb25zdCB4Q01heCA9IE1hdGgubWluKHhDb2xzLCBmU2l6ZSArIHhDQ29ybmVyKTtcbiAgICAgICAgICBsZXQgbWF4VmFsdWUgPSBOdW1iZXIuTkVHQVRJVkVfSU5GSU5JVFk7XG4gICAgICAgICAgbGV0IG1heFBvc2l0aW9uID0gLTE7XG4gICAgICAgICAgZm9yIChsZXQgeFIgPSB4Uk1pbjsgeFIgPCB4Uk1heDsgKyt4Uikge1xuICAgICAgICAgICAgY29uc3Qgd1IgPSB4UiAtIHhSQ29ybmVyO1xuICAgICAgICAgICAgZm9yIChsZXQgeEMgPSB4Q01pbjsgeEMgPCB4Q01heDsgKyt4Qykge1xuICAgICAgICAgICAgICBjb25zdCB3QyA9IHhDIC0geENDb3JuZXI7XG4gICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geC5nZXQoeFIsIHhDLCBkKTtcbiAgICAgICAgICAgICAgaWYgKHBpeGVsID4gbWF4VmFsdWUpIHtcbiAgICAgICAgICAgICAgICBtYXhWYWx1ZSA9IHBpeGVsO1xuICAgICAgICAgICAgICAgIG1heFBvc2l0aW9uID0gd1IgKiBmU2l6ZSArIHdDO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIG1heFBvc2l0aW9ucy5zZXQobWF4UG9zaXRpb24sIHlSLCB5QywgZCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG1heFBvc2l0aW9ucztcbiAgfVxuXG4gIHByb3RlY3RlZCBtYXhQb29sQmFja3Byb3BJbnRlcm5hbChcbiAgICAgIGR5OiBBcnJheTNELCB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBvcmlnU3RyaWRlOiBudW1iZXIsXG4gICAgICBvcmlnUGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICBjb25zdCBtYXhQb3NpdGlvbnMgPSB0aGlzLm1heFBvb2xQb3NpdGlvbnMoeCwgZlNpemUsIG9yaWdTdHJpZGUsIG9yaWdQYWQpO1xuICAgIGNvbnN0IHBhZCA9IGZTaXplIC0gMSAtIG9yaWdQYWQ7XG4gICAgY29uc3QgW2R5Um93cywgZHlDb2xzLCBkZXB0aF0gPSBkeS5zaGFwZTtcblxuICAgIC8vIERpbGF0ZSB0aGUgaW5wdXQuXG4gICAgY29uc3QgZHlSb3dzRGlsYXRlZCA9IChkeVJvd3MgLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuICAgIGNvbnN0IGR4Q29sc0RpbGF0ZWQgPSAoZHlDb2xzIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcblxuICAgIGNvbnN0IG91dHB1dFNoYXBlID0gY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKFxuICAgICAgICBbZHlSb3dzRGlsYXRlZCwgZHhDb2xzRGlsYXRlZCwgZGVwdGhdLCBmU2l6ZSwgZGVwdGgsIDEsIHBhZCk7XG4gICAgY29uc3QgZHggPSBBcnJheTNELnplcm9zKG91dHB1dFNoYXBlKTtcblxuICAgIGZvciAobGV0IGQgPSAwOyBkIDwgZGVwdGg7ICsrZCkge1xuICAgICAgZm9yIChsZXQgZHhSID0gMDsgZHhSIDwgZHguc2hhcGVbMF07ICsrZHhSKSB7XG4gICAgICAgIGZvciAobGV0IGR4QyA9IDA7IGR4QyA8IGR4LnNoYXBlWzFdOyArK2R4Qykge1xuICAgICAgICAgIC8vIFNoYWRlciBjb2RlIGJlZ2lucy5cbiAgICAgICAgICBjb25zdCBkeVJDb3JuZXIgPSBkeFIgLSBwYWQ7XG4gICAgICAgICAgY29uc3QgZHlDQ29ybmVyID0gZHhDIC0gcGFkO1xuICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB3UiA9IDA7IHdSIDwgZlNpemU7ICsrd1IpIHtcbiAgICAgICAgICAgIGNvbnN0IGR5UiA9IChkeVJDb3JuZXIgKyB3UikgLyBvcmlnU3RyaWRlO1xuICAgICAgICAgICAgaWYgKGR5UiA8IDAgfHwgZHlSID49IGR5Um93cyB8fCBNYXRoLmZsb29yKGR5UikgIT09IGR5Uikge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGZvciAobGV0IHdDID0gMDsgd0MgPCBmU2l6ZTsgKyt3Qykge1xuICAgICAgICAgICAgICBjb25zdCBkeUMgPSAoZHlDQ29ybmVyICsgd0MpIC8gb3JpZ1N0cmlkZTtcbiAgICAgICAgICAgICAgaWYgKGR5QyA8IDAgfHwgZHlDID49IGR5Q29scyB8fCBNYXRoLmZsb29yKGR5QykgIT09IGR5Qykge1xuICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIGNvbnN0IG1heFBvcyA9IGZTaXplICogZlNpemUgLSAxIC0gbWF4UG9zaXRpb25zLmdldChkeVIsIGR5QywgZCk7XG4gICAgICAgICAgICAgIGNvbnN0IGN1clBvcyA9IHdSICogZlNpemUgKyB3QztcblxuICAgICAgICAgICAgICBjb25zdCBtYXNrID0gbWF4UG9zID09PSBjdXJQb3MgPyAxIDogMDtcbiAgICAgICAgICAgICAgaWYgKG1hc2sgPT09IDApIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0gZHkuZ2V0KGR5UiwgZHlDLCBkKTtcbiAgICAgICAgICAgICAgZG90UHJvZCArPSBwaXhlbCAqIG1hc2s7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIGR4LnNldChkb3RQcm9kLCBkeFIsIGR4QywgZCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGR4O1xuICB9XG5cbiAgcHJvdGVjdGVkIG1pblBvb2xJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHJldHVybiB0aGlzLnBvb2woeCwgZlNpemUsIHN0cmlkZSwgcGFkLCAnbWluJyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgYXZnUG9vbEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgcmV0dXJuIHRoaXMucG9vbCh4LCBmU2l6ZSwgc3RyaWRlLCBwYWQsICdhdmcnKTtcbiAgfVxuXG4gIHByb3RlY3RlZCByZXNpemVCaWxpbmVhcjNESW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBuZXdTaGFwZTJEOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgYWxpZ25Db3JuZXJzOiBib29sZWFuKTogQXJyYXkzRCB7XG4gICAgY29uc3Qgb3V0cHV0ID0gQXJyYXkzRC56ZXJvcyhbbmV3U2hhcGUyRFswXSwgbmV3U2hhcGUyRFsxXSwgeC5zaGFwZVsyXV0pO1xuXG4gICAgY29uc3QgZWZmZWN0aXZlSW5wdXRTaXplID1cbiAgICAgICAgYWxpZ25Db3JuZXJzID8gW3guc2hhcGVbMF0gLSAxLCB4LnNoYXBlWzFdIC0gMSwgeC5zaGFwZVsyXV0gOiB4LnNoYXBlO1xuICAgIGNvbnN0IGVmZmVjdGl2ZU91dHB1dFNpemUgPSBhbGlnbkNvcm5lcnMgP1xuICAgICAgICBbb3V0cHV0LnNoYXBlWzBdIC0gMSwgb3V0cHV0LnNoYXBlWzFdIC0gMSwgb3V0cHV0LnNoYXBlWzJdXSA6XG4gICAgICAgIG91dHB1dC5zaGFwZTtcbiAgICBmb3IgKGxldCByID0gMDsgciA8IG91dHB1dC5zaGFwZVswXTsgcisrKSB7XG4gICAgICBmb3IgKGxldCBjID0gMDsgYyA8IG91dHB1dC5zaGFwZVsxXTsgYysrKSB7XG4gICAgICAgIGZvciAobGV0IGQgPSAwOyBkIDwgb3V0cHV0LnNoYXBlWzJdOyBkKyspIHtcbiAgICAgICAgICAvLyBCZWdpbiBzaGFkZXIuXG5cbiAgICAgICAgICAvLyBDb21wdXRlIHRoZSBmcmFjdGlvbmFsIGluZGV4IG9mIHRoZSBzb3VyY2UuXG4gICAgICAgICAgY29uc3Qgc291cmNlRnJhY1JvdyA9XG4gICAgICAgICAgICAgIChlZmZlY3RpdmVJbnB1dFNpemVbMF0pICogciAvIChlZmZlY3RpdmVPdXRwdXRTaXplWzBdKTtcbiAgICAgICAgICBjb25zdCBzb3VyY2VGcmFjQ29sID1cbiAgICAgICAgICAgICAgKGVmZmVjdGl2ZUlucHV0U2l6ZVsxXSkgKiBjIC8gKGVmZmVjdGl2ZU91dHB1dFNpemVbMV0pO1xuXG4gICAgICAgICAgY29uc3Qgc291cmNlUm93Rmxvb3IgPSBNYXRoLmZsb29yKHNvdXJjZUZyYWNSb3cpO1xuICAgICAgICAgIGNvbnN0IHNvdXJjZVJvd0NlaWwgPVxuICAgICAgICAgICAgICBNYXRoLm1pbih4LnNoYXBlWzBdIC0gMSwgTWF0aC5jZWlsKHNvdXJjZUZyYWNSb3cpKTtcbiAgICAgICAgICBjb25zdCBzb3VyY2VDb2xGbG9vciA9IE1hdGguZmxvb3Ioc291cmNlRnJhY0NvbCk7XG4gICAgICAgICAgY29uc3Qgc291cmNlQ29sQ2VpbCA9XG4gICAgICAgICAgICAgIE1hdGgubWluKHguc2hhcGVbMV0gLSAxLCBNYXRoLmNlaWwoc291cmNlRnJhY0NvbCkpO1xuXG4gICAgICAgICAgY29uc3QgdG9wTGVmdCA9IHguZ2V0KHNvdXJjZVJvd0Zsb29yLCBzb3VyY2VDb2xGbG9vciwgZCk7XG4gICAgICAgICAgY29uc3QgYm90dG9tTGVmdCA9IHguZ2V0KHNvdXJjZVJvd0NlaWwsIHNvdXJjZUNvbEZsb29yLCBkKTtcbiAgICAgICAgICBjb25zdCB0b3BSaWdodCA9IHguZ2V0KHNvdXJjZVJvd0Zsb29yLCBzb3VyY2VDb2xDZWlsLCBkKTtcbiAgICAgICAgICBjb25zdCBib3R0b21SaWdodCA9IHguZ2V0KHNvdXJjZVJvd0NlaWwsIHNvdXJjZUNvbENlaWwsIGQpO1xuXG4gICAgICAgICAgY29uc3Qgcm93RnJhYyA9IHNvdXJjZUZyYWNSb3cgLSBzb3VyY2VSb3dGbG9vcjtcbiAgICAgICAgICBjb25zdCBjb2xGcmFjID0gc291cmNlRnJhY0NvbCAtIHNvdXJjZUNvbEZsb29yO1xuXG4gICAgICAgICAgY29uc3QgdG9wID0gdG9wTGVmdCArICh0b3BSaWdodCAtIHRvcExlZnQpICogY29sRnJhYztcbiAgICAgICAgICBjb25zdCBib3R0b20gPSBib3R0b21MZWZ0ICsgKGJvdHRvbVJpZ2h0IC0gYm90dG9tTGVmdCkgKiBjb2xGcmFjO1xuICAgICAgICAgIGNvbnN0IG5ld1ZhbHVlID0gdG9wICsgKGJvdHRvbSAtIHRvcCkgKiByb3dGcmFjO1xuXG4gICAgICAgICAgb3V0cHV0LnNldChuZXdWYWx1ZSwgciwgYywgZCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gb3V0cHV0O1xuICB9XG5cbiAgcHJvdGVjdGVkIGJhdGNoTm9ybWFsaXphdGlvbjNESW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBtZWFuOiBBcnJheTNEfEFycmF5MUQsIHZhcmlhbmNlOiBBcnJheTNEfEFycmF5MUQsXG4gICAgICB2YXJpYW5jZUVwc2lsb24gPSAuMDAxLCBzY2FsZT86IEFycmF5M0R8QXJyYXkxRCxcbiAgICAgIG9mZnNldD86IEFycmF5M0R8QXJyYXkxRCk6IEFycmF5M0Qge1xuICAgIGNvbnN0IHhWYWx1ZXMgPSB4LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IG1lYW5WYWx1ZXMgPSBtZWFuLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IHZhcmlhbmNlVmFsdWVzID0gdmFyaWFuY2UuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3Qgc2NhbGVWYWx1ZXMgPSBzY2FsZSA/IHNjYWxlLmdldFZhbHVlcygpIDogbmV3IEZsb2F0MzJBcnJheShbMV0pO1xuICAgIGNvbnN0IG9mZnNldFZhbHVlcyA9IG9mZnNldCA/IG9mZnNldC5nZXRWYWx1ZXMoKSA6IG5ldyBGbG9hdDMyQXJyYXkoWzBdKTtcbiAgICBjb25zdCBvdXRWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHhWYWx1ZXMubGVuZ3RoKTtcblxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgeFZhbHVlcy5sZW5ndGg7IGkrKykge1xuICAgICAgb3V0VmFsdWVzW2ldID0gb2Zmc2V0VmFsdWVzW2kgJSBvZmZzZXRWYWx1ZXMubGVuZ3RoXSArXG4gICAgICAgICAgKHhWYWx1ZXNbaV0gLSBtZWFuVmFsdWVzW2kgJSBtZWFuVmFsdWVzLmxlbmd0aF0pICpcbiAgICAgICAgICAgICAgc2NhbGVWYWx1ZXNbaSAlIHNjYWxlVmFsdWVzLmxlbmd0aF0gL1xuICAgICAgICAgICAgICBNYXRoLnNxcnQoXG4gICAgICAgICAgICAgICAgICB2YXJpYW5jZVZhbHVlc1tpICUgdmFyaWFuY2VWYWx1ZXMubGVuZ3RoXSArIHZhcmlhbmNlRXBzaWxvbik7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8QXJyYXkzRD4oeC5zaGFwZSwge3ZhbHVlczogb3V0VmFsdWVzfSk7XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vd2ViZ2wvZ3BncHVfY29udGV4dCc7XG5pbXBvcnQge1RleHR1cmVNYW5hZ2VyfSBmcm9tICcuL3dlYmdsL3RleHR1cmVfbWFuYWdlcic7XG5pbXBvcnQgKiBhcyB3ZWJnbF91dGlsIGZyb20gJy4vd2ViZ2wvd2ViZ2xfdXRpbCc7XG5cbi8vIFRoZXNlIGdsb2JhbCB2YXJpYWJsZXMgbmVlZCB0byBiZSBpbml0aWFsaXplZCB0byBudWxsIHNvIHRoYXQgY2xvc3VyZSBrbm93c1xuLy8gbm90IHRvIHNlYWwgdGhlbS5cbi8qKiBAaGlkZGVuICovXG5leHBvcnQgbGV0IEdQR1BVOiBHUEdQVUNvbnRleHQgPSBudWxsITtcbi8qKiBAaGlkZGVuICovXG5leHBvcnQgbGV0IFRFWFRVUkVfTUFOQUdFUjogVGV4dHVyZU1hbmFnZXIgPSBudWxsITtcblxuLyoqIEBoaWRkZW4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgTkRBcnJheURhdGEge1xuICB2YWx1ZXM/OiBGbG9hdDMyQXJyYXk7XG4gIHRleHR1cmU/OiBXZWJHTFRleHR1cmU7XG4gIC8qKiBbcm93cywgY29sdW1uc10gc2hhcGUgb2YgdGhlIHRleHR1cmUuICovXG4gIHRleHR1cmVTaGFwZVJDPzogW251bWJlciwgbnVtYmVyXTtcbn1cblxuLyoqIEBoaWRkZW4gKi9cbmV4cG9ydCBmdW5jdGlvbiBpbml0aWFsaXplR1BVKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHRleHR1cmVNYW5hZ2VyOiBUZXh0dXJlTWFuYWdlcikge1xuICBHUEdQVSA9IGdwZ3B1O1xuICBURVhUVVJFX01BTkFHRVIgPSB0ZXh0dXJlTWFuYWdlcjtcbn1cblxuZnVuY3Rpb24gdGhyb3dJZkdQVU5vdEluaXRpYWxpemVkKCkge1xuICBpZiAoR1BHUFUgPT0gbnVsbCB8fCBURVhUVVJFX01BTkFHRVIgPT0gbnVsbCkge1xuICAgIHRocm93IG5ldyBFcnJvcignR1BVIG5vdCBpbnRpYWxpemVkLicpO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBOREFycmF5IHtcbiAgLyoqIFRoZSBzaGFwZSBvZiB0aGUgbmRhcnJheS4gKi9cbiAgc2hhcGU6IG51bWJlcltdO1xuICAvKiogTnVtYmVyIG9mIGVsZW1lbnRzIGluIHRoZSBuZGFycmF5LiAqL1xuICBzaXplOiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIE51bWJlciBvZiBlbGVtZW50cyB0byBza2lwIGluIGVhY2ggZGltZW5zaW9uIHdoZW4gaW5kZXhpbmcuIFNlZVxuICAgKiBodHRwczovL2RvY3Muc2NpcHkub3JnL2RvYy9udW1weS9yZWZlcmVuY2UvZ2VuZXJhdGVkL251bXB5Lm5kYXJyYXkuc3RyaWRlcy5odG1sXG4gICAqL1xuICBwcm90ZWN0ZWQgc3RyaWRlczogbnVtYmVyW107XG5cbiAgcHJpdmF0ZSBkYXRhOiBOREFycmF5RGF0YTtcblxuICBwcm90ZWN0ZWQgY29uc3RydWN0b3Ioc2hhcGU6IG51bWJlcltdLCBkYXRhOiBOREFycmF5RGF0YSkge1xuICAgIC8vIFNhbml0eSBjaGVja3MuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGRhdGEudmFsdWVzICE9IG51bGwgfHwgZGF0YS50ZXh0dXJlICE9IG51bGwsXG4gICAgICAgICdFaXRoZXIgYHZhbHVlc2Agb3IgYHRleHR1cmVgIG11c3QgYmUgZGVmaW5lZCcpO1xuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGRhdGEudGV4dHVyZSA9PSBudWxsIHx8IChkYXRhLnRleHR1cmVTaGFwZVJDICE9IG51bGwpLFxuICAgICAgICAnYHRleHR1cmVTaGFwZWAgbXVzdCBiZSBkZWZpbmVkIHdoZW4gYHRleHR1cmVgIGlzIGRlZmluZWQnKTtcblxuICAgIHRoaXMuc2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZShzaGFwZSk7XG5cbiAgICBpZiAoZGF0YS52YWx1ZXMgIT0gbnVsbCkge1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgdGhpcy5zaXplID09PSBkYXRhLnZhbHVlcy5sZW5ndGgsXG4gICAgICAgICAgJ0NvbnN0cnVjdGluZyBuZGFycmF5IG9mIHNoYXBlICgnICsgdGhpcy5zaXplICsgJykgc2hvdWxkIG1hdGNoIHRoZScgK1xuICAgICAgICAgICAgICAnIGxlbmd0aCBvZiB2YWx1ZXMgKCcgKyBkYXRhLnZhbHVlcy5sZW5ndGggKyAnKScpO1xuICAgIH1cblxuICAgIHRoaXMuc2hhcGUgPSBzaGFwZTtcbiAgICB0aGlzLmRhdGEgPSBkYXRhO1xuICAgIGNvbnN0IGRpbSA9IHRoaXMuc2hhcGUubGVuZ3RoO1xuXG4gICAgaWYgKGRpbSA8IDIpIHtcbiAgICAgIHRoaXMuc3RyaWRlcyA9IFtdO1xuICAgIH0gZWxzZSB7XG4gICAgICAvLyBMYXN0IGRpbWVuc2lvbiBoYXMgaW1wbGljaXQgc3RyaWRlIG9mIDEsIHRodXMgaGF2aW5nIEQtMSAoaW5zdGVhZCBvZiBEKVxuICAgICAgLy8gc3RyaWRlcy5cbiAgICAgIHRoaXMuc3RyaWRlcyA9IG5ldyBBcnJheShkaW0gLSAxKTtcbiAgICAgIHRoaXMuc3RyaWRlc1tkaW0gLSAyXSA9IHRoaXMuc2hhcGVbZGltIC0gMV07XG4gICAgICBmb3IgKGxldCBpID0gZGltIC0gMzsgaSA+PSAwOyAtLWkpIHtcbiAgICAgICAgdGhpcy5zdHJpZGVzW2ldID0gdGhpcy5zdHJpZGVzW2kgKyAxXSAqIHRoaXMuc2hhcGVbaSArIDFdO1xuICAgICAgfVxuICAgIH1cbiAgfVxuXG4gIC8qKiBDcmVhdGVzIGEgbmRhcnJheSBvZiB6ZXJvcyB3aXRoIHRoZSBzcGVjaWZpZWQgc2hhcGUuICovXG4gIHN0YXRpYyB6ZXJvczxUIGV4dGVuZHMgTkRBcnJheT4oc2hhcGU6IG51bWJlcltdKTogVCB7XG4gICAgY29uc3QgdmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheSh1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpKTtcbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KHNoYXBlLCB7dmFsdWVzfSk7XG4gIH1cblxuICAvKiogQ3JlYXRlcyBhIG5kYXJyYXkgb2YgemVyb3Mgd2l0aCB0aGUgc2FtZSBzaGFwZSBhcyB0aGUgc3BlY2lmaWVkIG5kYXJyYXkuXG4gICAqL1xuICBzdGF0aWMgemVyb3NMaWtlPFQgZXh0ZW5kcyBOREFycmF5Pihhbm90aGVyOiBUKTogVCB7XG4gICAgcmV0dXJuIE5EQXJyYXkuemVyb3MoYW5vdGhlci5zaGFwZSkgYXMgVDtcbiAgfVxuXG4gIC8qKiBDcmVhdGVzIGEgbmRhcnJheSB3aXRoIHRoZSBzYW1lIHZhbHVlcy9zaGFwZSBhcyB0aGUgc3BlY2lmaWVkIG5kYXJyYXkuICovXG4gIHN0YXRpYyBsaWtlPFQgZXh0ZW5kcyBOREFycmF5Pihhbm90aGVyOiBUKTogVCB7XG4gICAgY29uc3QgdmFsdWVzID0gYW5vdGhlci5nZXRWYWx1ZXMoKTtcbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KGFub3RoZXIuc2hhcGUsIHt2YWx1ZXM6IG5ldyBGbG9hdDMyQXJyYXkodmFsdWVzKX0pO1xuICB9XG5cbiAgLyoqXG4gICAqIE1ha2VzIGEgbmV3IG5kYXJyYXkgd2l0aCB0aGUgcHJvdmlkZWQgc2hhcGUgYW5kIHZhbHVlcy4gVmFsdWVzIHNob3VsZCBiZSBpblxuICAgKiBhIGZsYXQgYXJyYXkuXG4gICAqL1xuICBzdGF0aWMgbWFrZTxUIGV4dGVuZHMgTkRBcnJheT4oc2hhcGU6IG51bWJlcltdLCBkYXRhOiBOREFycmF5RGF0YSk6IFQge1xuICAgIHN3aXRjaCAoc2hhcGUubGVuZ3RoKSB7XG4gICAgICBjYXNlIDA6XG4gICAgICAgIHJldHVybiBuZXcgU2NhbGFyKGRhdGEpIGFzIFQ7XG4gICAgICBjYXNlIDE6XG4gICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgcmV0dXJuIG5ldyBBcnJheTFEKGRhdGEpIGFzIGFueTtcbiAgICAgIGNhc2UgMjpcbiAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgICByZXR1cm4gbmV3IEFycmF5MkQoc2hhcGUgYXMgW251bWJlciwgbnVtYmVyXSwgZGF0YSkgYXMgYW55O1xuICAgICAgY2FzZSAzOlxuICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICAgIHJldHVybiBuZXcgQXJyYXkzRChzaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGRhdGEpIGFzIGFueTtcbiAgICAgIGNhc2UgNDpcbiAgICAgICAgcmV0dXJuIG5ldyBBcnJheTREKFxuICAgICAgICAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgICAgICAgICAgICBzaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZGF0YSkgYXMgYW55O1xuICAgICAgZGVmYXVsdDpcbiAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgICByZXR1cm4gbmV3IE5EQXJyYXkoc2hhcGUsIGRhdGEpIGFzIGFueTtcbiAgICB9XG4gIH1cblxuICAvKiogUmVzaGFwZXMgdGhlIGN1cnJlbnQgbmRhcnJheSBpbnRvIHRoZSBwcm92aWRlZCBzaGFwZS4gKi9cbiAgcmVzaGFwZTxUIGV4dGVuZHMgTkRBcnJheT4obmV3U2hhcGU6IG51bWJlcltdKTogVCB7XG4gICAgaWYgKHV0aWwuYXJyYXlzRXF1YWwodGhpcy5zaGFwZSwgbmV3U2hhcGUpKSB7XG4gICAgICAvLyBOby1vcC5cbiAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgIHJldHVybiB0aGlzIGFzIGFueTtcbiAgICB9XG5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdGhpcy5zaXplID09PSB1dGlsLnNpemVGcm9tU2hhcGUobmV3U2hhcGUpLFxuICAgICAgICAnbmV3IHNoYXBlIGFuZCBvbGQgc2hhcGUgbXVzdCBoYXZlIHRoZSBzYW1lIG51bWJlciBvZiBlbGVtZW50cy4nKTtcblxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmV3U2hhcGUsIHRoaXMuZGF0YSk7XG4gIH1cblxuICBhc1NjYWxhcigpOiBTY2FsYXIge1xuICAgIHV0aWwuYXNzZXJ0KHRoaXMuc2l6ZSA9PT0gMSwgJ1RoZSBhcnJheSBtdXN0IGhhdmUgb25seSAxIGVsZW1lbnQuJyk7XG4gICAgcmV0dXJuIHRoaXMucmVzaGFwZTxTY2FsYXI+KFtdKTtcbiAgfVxuXG4gIGFzMUQoKTogQXJyYXkxRCB7XG4gICAgcmV0dXJuIHRoaXMucmVzaGFwZTxBcnJheTFEPihbdGhpcy5zaXplXSk7XG4gIH1cblxuICBhczJEKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogQXJyYXkyRCB7XG4gICAgcmV0dXJuIHRoaXMucmVzaGFwZTxBcnJheTJEPihbcm93cywgY29sdW1uc10pO1xuICB9XG5cbiAgYXMzRChyb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlciwgZGVwdGg6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHJldHVybiB0aGlzLnJlc2hhcGU8QXJyYXkzRD4oW3Jvd3MsIGNvbHVtbnMsIGRlcHRoXSk7XG4gIH1cblxuICBhczREKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyLCBkZXB0aDogbnVtYmVyLCBkZXB0aDI6IG51bWJlcik6IEFycmF5NEQge1xuICAgIHJldHVybiB0aGlzLnJlc2hhcGU8QXJyYXk0RD4oW3Jvd3MsIGNvbHVtbnMsIGRlcHRoLCBkZXB0aDJdKTtcbiAgfVxuXG4gIGdldCByYW5rKCk6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMuc2hhcGUubGVuZ3RoO1xuICB9XG5cbiAgZ2V0KC4uLmxvY3M6IG51bWJlcltdKSB7XG4gICAgbGV0IGluZGV4ID0gbG9jc1tsb2NzLmxlbmd0aCAtIDFdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbG9jcy5sZW5ndGggLSAxOyArK2kpIHtcbiAgICAgIGluZGV4ICs9IHRoaXMuc3RyaWRlc1tpXSAqIGxvY3NbaV07XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmdldFZhbHVlcygpW2luZGV4XTtcbiAgfVxuXG4gIGFkZCh2YWx1ZTogbnVtYmVyLCAuLi5sb2NzOiBudW1iZXJbXSkge1xuICAgIHRoaXMuc2V0KHRoaXMuZ2V0KC4uLmxvY3MpICsgdmFsdWUsIC4uLmxvY3MpO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIsIC4uLmxvY3M6IG51bWJlcltdKSB7XG4gICAgbGV0IGluZGV4ID0gbG9jc1tsb2NzLmxlbmd0aCAtIDFdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbG9jcy5sZW5ndGggLSAxOyArK2kpIHtcbiAgICAgIGluZGV4ICs9IHRoaXMuc3RyaWRlc1tpXSAqIGxvY3NbaV07XG4gICAgfVxuICAgIHRoaXMuZ2V0VmFsdWVzKClbaW5kZXhdID0gdmFsdWU7XG4gIH1cblxuICBsb2NUb0luZGV4KGxvY3M6IG51bWJlcltdKTogbnVtYmVyIHtcbiAgICBsZXQgaW5kZXggPSBsb2NzW2xvY3MubGVuZ3RoIC0gMV07XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsb2NzLmxlbmd0aCAtIDE7ICsraSkge1xuICAgICAgaW5kZXggKz0gdGhpcy5zdHJpZGVzW2ldICogbG9jc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIGluZGV4O1xuICB9XG5cbiAgaW5kZXhUb0xvYyhpbmRleDogbnVtYmVyKTogbnVtYmVyW10ge1xuICAgIGNvbnN0IGxvY3M6IG51bWJlcltdID0gbmV3IEFycmF5KHRoaXMuc2hhcGUubGVuZ3RoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGxvY3MubGVuZ3RoIC0gMTsgKytpKSB7XG4gICAgICBsb2NzW2ldID0gTWF0aC5mbG9vcihpbmRleCAvIHRoaXMuc3RyaWRlc1tpXSk7XG4gICAgICBpbmRleCAtPSBsb2NzW2ldICogdGhpcy5zdHJpZGVzW2ldO1xuICAgIH1cbiAgICBsb2NzW2xvY3MubGVuZ3RoIC0gMV0gPSBpbmRleDtcbiAgICByZXR1cm4gbG9jcztcbiAgfVxuXG4gIGZpbGwodmFsdWU6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKCkuZmlsbCh2YWx1ZSk7XG4gIH1cblxuICBnZXREYXRhKCk6IE5EQXJyYXlEYXRhIHtcbiAgICByZXR1cm4gdGhpcy5kYXRhO1xuICB9XG5cbiAgZ2V0VmFsdWVzKCk6IEZsb2F0MzJBcnJheSB7XG4gICAgaWYgKHRoaXMuZGF0YS52YWx1ZXMgPT0gbnVsbCkge1xuICAgICAgdGhyb3dJZkdQVU5vdEluaXRpYWxpemVkKCk7XG4gICAgICB0aGlzLmRhdGEudmFsdWVzID0gR1BHUFUuZG93bmxvYWRNYXRyaXhGcm9tVGV4dHVyZShcbiAgICAgICAgICB0aGlzLmRhdGEudGV4dHVyZSEsIHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQyFbMF0sXG4gICAgICAgICAgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDIVsxXSk7XG4gICAgICB0aGlzLmRpc3Bvc2VUZXh0dXJlKCk7XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmRhdGEudmFsdWVzO1xuICB9XG5cbiAgcHJpdmF0ZSB1cGxvYWRUb0dQVShwcmVmZXJyZWRUZXhTaGFwZT86IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgICB0aHJvd0lmR1BVTm90SW5pdGlhbGl6ZWQoKTtcbiAgICB0aGlzLmRhdGEudGV4dHVyZVNoYXBlUkMgPSB3ZWJnbF91dGlsLmdldFRleHR1cmVTaGFwZUZyb21Mb2dpY2FsU2hhcGUoXG4gICAgICAgIEdQR1BVLmdsLCB0aGlzLnNoYXBlLCBwcmVmZXJyZWRUZXhTaGFwZSk7XG4gICAgdGhpcy5kYXRhLnRleHR1cmUgPVxuICAgICAgICBURVhUVVJFX01BTkFHRVIuYWNxdWlyZVRleHR1cmUodGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDKTtcblxuICAgIEdQR1BVLnVwbG9hZE1hdHJpeFRvVGV4dHVyZShcbiAgICAgICAgdGhpcy5kYXRhLnRleHR1cmUsIHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQ1swXSxcbiAgICAgICAgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDWzFdLCB0aGlzLmRhdGEudmFsdWVzISk7XG5cbiAgICB0aGlzLmRhdGEudmFsdWVzID0gbnVsbCE7XG4gIH1cblxuICBnZXRUZXh0dXJlKHByZWZlcnJlZFNoYXBlUkM/OiBbbnVtYmVyLCBudW1iZXJdKTogV2ViR0xUZXh0dXJlIHtcbiAgICBpZiAodGhpcy5kYXRhLnRleHR1cmUgPT0gbnVsbCkge1xuICAgICAgdGhpcy51cGxvYWRUb0dQVShwcmVmZXJyZWRTaGFwZVJDKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuZGF0YS50ZXh0dXJlITtcbiAgfVxuXG4gIGdldFRleHR1cmVTaGFwZVJDKHByZWZlcnJlZFNoYXBlUkM/OiBbbnVtYmVyLCBudW1iZXJdKTogW251bWJlciwgbnVtYmVyXSB7XG4gICAgaWYgKHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQyA9PSBudWxsKSB7XG4gICAgICB0aGlzLnVwbG9hZFRvR1BVKHByZWZlcnJlZFNoYXBlUkMpO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDITtcbiAgfVxuXG4gIGRpc3Bvc2UoKTogdm9pZCB7XG4gICAgdGhpcy5kYXRhLnZhbHVlcyA9IG51bGwhO1xuICAgIHRoaXMuc2hhcGUgPSBudWxsITtcbiAgICBpZiAodGhpcy5kYXRhLnRleHR1cmUgIT0gbnVsbCkge1xuICAgICAgdGhpcy5kaXNwb3NlVGV4dHVyZSgpO1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgZGlzcG9zZVRleHR1cmUoKSB7XG4gICAgdGhyb3dJZkdQVU5vdEluaXRpYWxpemVkKCk7XG4gICAgVEVYVFVSRV9NQU5BR0VSLnJlbGVhc2VUZXh0dXJlKFxuICAgICAgICB0aGlzLmRhdGEudGV4dHVyZSEsIHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQyEpO1xuICAgIHRoaXMuZGF0YS50ZXh0dXJlID0gbnVsbCE7XG4gICAgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDID0gbnVsbCE7XG4gIH1cblxuICBpbkdQVSgpOiBib29sZWFuIHtcbiAgICByZXR1cm4gdGhpcy5kYXRhLnRleHR1cmUgIT0gbnVsbDtcbiAgfVxuXG4gIGVxdWFscyh0OiBOREFycmF5KTogYm9vbGVhbiB7XG4gICAgcmV0dXJuIHV0aWwuYXJyYXlzRXF1YWwodGhpcy5zaGFwZSwgdC5zaGFwZSkgJiZcbiAgICAgICAgdXRpbC5hcnJheXNFcXVhbCh0aGlzLmdldFZhbHVlcygpLCB0LmdldFZhbHVlcygpKTtcbiAgfVxuXG4gIHN0YXRpYyByYW5kPFQgZXh0ZW5kcyBOREFycmF5PihzaGFwZTogbnVtYmVyW10sIHJhbmRGdW5jdGlvbjogKCkgPT4gbnVtYmVyKTpcbiAgICAgIFQge1xuICAgIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUoc2hhcGUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkoc2l6ZSk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBzaXplOyBpKyspIHtcbiAgICAgIHZhbHVlc1tpXSA9IHJhbmRGdW5jdGlvbigpO1xuICAgIH1cblxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4oc2hhcGUsIHt2YWx1ZXN9KTtcbiAgfVxuXG4gIHN0YXRpYyByYW5kTm9ybWFsPFQgZXh0ZW5kcyBOREFycmF5PihzaGFwZTogbnVtYmVyW10sIG1lYW4gPSAwLCBzdGREZXYgPSAxKSB7XG4gICAgcmV0dXJuIE5EQXJyYXkucmFuZDxUPihzaGFwZSwgKCkgPT4gdXRpbC5yYW5kR2F1c3MobWVhbiwgc3RkRGV2KSk7XG4gIH1cblxuICBzdGF0aWMgcmFuZFRydW5jYXRlZE5vcm1hbDxUIGV4dGVuZHMgTkRBcnJheT4oXG4gICAgICBzaGFwZTogbnVtYmVyW10sIG1lYW4gPSAwLCBzdGREZXYgPSAxKSB7XG4gICAgcmV0dXJuIE5EQXJyYXkucmFuZDxUPihzaGFwZSwgKCkgPT4gdXRpbC5yYW5kR2F1c3MobWVhbiwgc3RkRGV2LCB0cnVlKSk7XG4gIH1cblxuICBzdGF0aWMgcmFuZFVuaWZvcm08VCBleHRlbmRzIE5EQXJyYXk+KHNoYXBlOiBudW1iZXJbXSwgYTogbnVtYmVyLCBiOiBudW1iZXIpIHtcbiAgICByZXR1cm4gTkRBcnJheS5yYW5kPFQ+KHNoYXBlLCAoKSA9PiB1dGlsLnJhbmRVbmlmb3JtKGEsIGIpKTtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgU2NhbGFyIGV4dGVuZHMgTkRBcnJheSB7XG4gIGNvbnN0cnVjdG9yKGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgaWYgKGRhdGEudGV4dHVyZSAhPSBudWxsKSB7XG4gICAgICBkYXRhLnRleHR1cmVTaGFwZVJDID0gWzEsIDFdO1xuICAgIH1cbiAgICBzdXBlcihbXSwgZGF0YSk7XG4gIH1cblxuICBzdGF0aWMgbmV3KHZhbHVlOiBudW1iZXIpIHtcbiAgICByZXR1cm4gbmV3IFNjYWxhcih7dmFsdWVzOiBuZXcgRmxvYXQzMkFycmF5KFt2YWx1ZV0pfSk7XG4gIH1cblxuICBzdGF0aWMgWkVSTyA9IFNjYWxhci5uZXcoMCk7XG4gIHN0YXRpYyBPTkUgPSBTY2FsYXIubmV3KDEpO1xuICBzdGF0aWMgVFdPID0gU2NhbGFyLm5ldygyKTtcbiAgc3RhdGljIE5FR19PTkUgPSBTY2FsYXIubmV3KC0xKTtcblxuICBnZXQoKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5nZXRWYWx1ZXMoKVswXTtcbiAgfVxuXG4gIHNldCh2YWx1ZTogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVswXSA9IHZhbHVlO1xuICB9XG5cbiAgYWRkKHZhbHVlOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpWzBdICs9IHZhbHVlO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBBcnJheTFEIGV4dGVuZHMgTkRBcnJheSB7XG4gIHNoYXBlOiBbbnVtYmVyXTtcblxuICBjb25zdHJ1Y3RvcihkYXRhOiBOREFycmF5RGF0YSkge1xuICAgIGNvbnN0IHNoYXBlID0gKGRhdGEudmFsdWVzICE9IG51bGwpID9cbiAgICAgICAgW2RhdGEudmFsdWVzLmxlbmd0aF0gOlxuICAgICAgICBbdXRpbC5zaXplRnJvbVNoYXBlKGRhdGEudGV4dHVyZVNoYXBlUkMhKV07XG4gICAgc3VwZXIoc2hhcGUsIGRhdGEpO1xuICB9XG5cbiAgc3RhdGljIG5ldyh2YWx1ZXM6IEZsb2F0MzJBcnJheXxudW1iZXJbXSkge1xuICAgIGlmICghKHZhbHVlcyBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkpIHtcbiAgICAgIGNvbnN0IGluZmVycmVkU2hhcGUgPSB1dGlsLmluZmVyU2hhcGUodmFsdWVzKTtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGluZmVycmVkU2hhcGUubGVuZ3RoID09PSAxLFxuICAgICAgICAgIGBFcnJvciBjb25zdHJ1Y3RpbmcgQXJyYXkxRC4gU2hhcGUgb2YgdmFsdWVzICR7aW5mZXJyZWRTaGFwZX0gaXMgYCArXG4gICAgICAgICAgICAgIGBub3QgMSBkaW1lbnNpb25hbC5gKTtcbiAgICB9XG4gICAgcmV0dXJuIG5ldyBBcnJheTFEKHt2YWx1ZXM6IHRvVHlwZWRBcnJheSh2YWx1ZXMpfSk7XG4gIH1cblxuICBnZXQoaTogbnVtYmVyKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5nZXRWYWx1ZXMoKVtpXTtcbiAgfVxuXG4gIHNldCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpW2ldID0gdmFsdWU7XG4gIH1cblxuICBhZGQodmFsdWU6IG51bWJlciwgaTogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVtpXSArPSB2YWx1ZTtcbiAgfVxuXG4gIGxvY1RvSW5kZXgobG9jOiBbbnVtYmVyXSk6IG51bWJlciB7XG4gICAgcmV0dXJuIGxvY1swXTtcbiAgfVxuXG4gIGluZGV4VG9Mb2MoaW5kZXg6IG51bWJlcik6IFtudW1iZXJdIHtcbiAgICByZXR1cm4gW2luZGV4XTtcbiAgfVxuXG4gIHN0YXRpYyB6ZXJvcyhzaGFwZTogW251bWJlcl0pOiBBcnJheTFEIHtcbiAgICByZXR1cm4gTkRBcnJheS56ZXJvczxBcnJheTFEPihzaGFwZSk7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIEFycmF5MkQgZXh0ZW5kcyBOREFycmF5IHtcbiAgc2hhcGU6IFtudW1iZXIsIG51bWJlcl07XG5cbiAgcHJpdmF0ZSBzdHJpZGUwOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3Ioc2hhcGU6IFtudW1iZXIsIG51bWJlcl0sIGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgdXRpbC5hc3NlcnQoc2hhcGUubGVuZ3RoID09PSAyLCAnU2hhcGUgc2hvdWxkIGJlIG9mIGxlbmd0aCAyJyk7XG4gICAgc3VwZXIoc2hhcGUsIGRhdGEpO1xuICAgIHRoaXMuc3RyaWRlMCA9IHRoaXMuc3RyaWRlc1swXTtcbiAgfVxuXG4gIHN0YXRpYyBuZXcoXG4gICAgICBzaGFwZTogW251bWJlciwgbnVtYmVyXSwgdmFsdWVzOiBGbG9hdDMyQXJyYXl8bnVtYmVyW118bnVtYmVyW11bXSkge1xuICAgIGlmICghKHZhbHVlcyBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkpIHtcbiAgICAgIGNvbnN0IGluZmVycmVkU2hhcGUgPSB1dGlsLmluZmVyU2hhcGUodmFsdWVzKTtcbiAgICAgIGlmIChpbmZlcnJlZFNoYXBlLmxlbmd0aCA+IDEpIHtcbiAgICAgICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChcbiAgICAgICAgICAgIHNoYXBlLCBpbmZlcnJlZFNoYXBlLFxuICAgICAgICAgICAgYEVycm9yIHdoZW4gY29uc3RydWN0aW5nIEFycmF5MkQuIFNoYXBlIG9mIHZhbHVlcyBgICtcbiAgICAgICAgICAgICAgICBgJHtpbmZlcnJlZFNoYXBlfSBkb2VzIG5vdCBtYXRjaCB0aGUgcHJvdmlkZWQgc2hhcGUgYCArXG4gICAgICAgICAgICAgICAgYCR7c2hhcGV9LiBgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG5ldyBBcnJheTJEKHNoYXBlLCB7dmFsdWVzOiB0b1R5cGVkQXJyYXkodmFsdWVzKX0pO1xuICB9XG5cbiAgZ2V0KGk6IG51bWJlciwgajogbnVtYmVyKSB7XG4gICAgcmV0dXJuIHRoaXMuZ2V0VmFsdWVzKClbdGhpcy5zdHJpZGUwICogaSArIGpdO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVt0aGlzLnN0cmlkZTAgKiBpICsgal0gPSB2YWx1ZTtcbiAgfVxuXG4gIGFkZCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIsIGo6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClbdGhpcy5zdHJpZGUwICogaSArIGpdICs9IHZhbHVlO1xuICB9XG5cbiAgbG9jVG9JbmRleChsb2NzOiBbbnVtYmVyLCBudW1iZXJdKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5zdHJpZGUwICogbG9jc1swXSArIGxvY3NbMV07XG4gIH1cblxuICBpbmRleFRvTG9jKGluZGV4OiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgICByZXR1cm4gW01hdGguZmxvb3IoaW5kZXggLyB0aGlzLnN0cmlkZTApLCBpbmRleCAlIHRoaXMuc3RyaWRlMF07XG4gIH1cblxuICBzdGF0aWMgemVyb3Moc2hhcGU6IFtudW1iZXIsIG51bWJlcl0pOiBBcnJheTJEIHtcbiAgICByZXR1cm4gTkRBcnJheS56ZXJvczxBcnJheTJEPihzaGFwZSk7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIEFycmF5M0QgZXh0ZW5kcyBOREFycmF5IHtcbiAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgcHJpdmF0ZSBzdHJpZGUwOiBudW1iZXI7XG4gIHByaXZhdGUgc3RyaWRlMTogbnVtYmVyO1xuXG4gIGNvbnN0cnVjdG9yKHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgdXRpbC5hc3NlcnQoc2hhcGUubGVuZ3RoID09PSAzLCAnU2hhcGUgc2hvdWxkIGJlIG9mIGxlbmd0aCAzJyk7XG4gICAgc3VwZXIoc2hhcGUsIGRhdGEpO1xuICAgIHRoaXMuc3RyaWRlMCA9IHRoaXMuc3RyaWRlc1swXTtcbiAgICB0aGlzLnN0cmlkZTEgPSB0aGlzLnN0cmlkZXNbMV07XG4gIH1cblxuICBzdGF0aWMgbmV3KFxuICAgICAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICAgIHZhbHVlczogRmxvYXQzMkFycmF5fG51bWJlcltdfG51bWJlcltdW11bXSkge1xuICAgIGlmICghKHZhbHVlcyBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkpIHtcbiAgICAgIGNvbnN0IGluZmVycmVkU2hhcGUgPSB1dGlsLmluZmVyU2hhcGUodmFsdWVzKTtcbiAgICAgIGlmIChpbmZlcnJlZFNoYXBlLmxlbmd0aCA+IDEpIHtcbiAgICAgICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChcbiAgICAgICAgICAgIHNoYXBlLCBpbmZlcnJlZFNoYXBlLFxuICAgICAgICAgICAgYEVycm9yIHdoZW4gY29uc3RydWN0aW5nIEFycmF5M0QuIFNoYXBlIG9mIHZhbHVlcyBgICtcbiAgICAgICAgICAgICAgICBgJHtpbmZlcnJlZFNoYXBlfSBkb2VzIG5vdCBtYXRjaCB0aGUgcHJvdmlkZWQgc2hhcGUgYCArXG4gICAgICAgICAgICAgICAgYCR7c2hhcGV9LiBgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG5ldyBBcnJheTNEKHNoYXBlLCB7dmFsdWVzOiB0b1R5cGVkQXJyYXkodmFsdWVzKX0pO1xuICB9XG5cbiAgZ2V0KGk6IG51bWJlciwgajogbnVtYmVyLCBrOiBudW1iZXIpIHtcbiAgICByZXR1cm4gdGhpcy5nZXRWYWx1ZXMoKVt0aGlzLnN0cmlkZTAgKiBpICsgdGhpcy5zdHJpZGUxICogaiArIGtdO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyLCBrOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpW3RoaXMuc3RyaWRlMCAqIGkgKyB0aGlzLnN0cmlkZTEgKiBqICsga10gPSB2YWx1ZTtcbiAgfVxuXG4gIGFkZCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIsIGo6IG51bWJlciwgazogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVt0aGlzLnN0cmlkZTAgKiBpICsgdGhpcy5zdHJpZGUxICogaiArIGtdICs9IHZhbHVlO1xuICB9XG5cbiAgbG9jVG9JbmRleChsb2NzOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLnN0cmlkZTAgKiBsb2NzWzBdICsgdGhpcy5zdHJpZGUxICogbG9jc1sxXSArIGxvY3NbMl07XG4gIH1cblxuICBpbmRleFRvTG9jKGluZGV4OiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0ge1xuICAgIGNvbnN0IGkgPSBNYXRoLmZsb29yKGluZGV4IC8gdGhpcy5zdHJpZGUwKTtcbiAgICBpbmRleCAtPSBpICogdGhpcy5zdHJpZGUwO1xuICAgIHJldHVybiBbaSwgTWF0aC5mbG9vcihpbmRleCAvIHRoaXMuc3RyaWRlMSksIGluZGV4ICUgdGhpcy5zdHJpZGUxXTtcbiAgfVxuXG4gIHN0YXRpYyB6ZXJvcyhzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdKTogQXJyYXkzRCB7XG4gICAgcmV0dXJuIE5EQXJyYXkuemVyb3M8QXJyYXkzRD4oc2hhcGUpO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBBcnJheTREIGV4dGVuZHMgTkRBcnJheSB7XG4gIHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgcHJpdmF0ZSBzdHJpZGUwOiBudW1iZXI7XG4gIHByaXZhdGUgc3RyaWRlMTogbnVtYmVyO1xuICBwcml2YXRlIHN0cmlkZTI6IG51bWJlcjtcblxuICBjb25zdHJ1Y3RvcihzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgdXRpbC5hc3NlcnQoc2hhcGUubGVuZ3RoID09PSA0LCAnU2hhcGUgc2hvdWxkIGJlIG9mIGxlbmd0aCA0Jyk7XG4gICAgc3VwZXIoc2hhcGUsIGRhdGEpO1xuICAgIHRoaXMuc3RyaWRlMCA9IHRoaXMuc3RyaWRlc1swXTtcbiAgICB0aGlzLnN0cmlkZTEgPSB0aGlzLnN0cmlkZXNbMV07XG4gICAgdGhpcy5zdHJpZGUyID0gdGhpcy5zdHJpZGVzWzJdO1xuICB9XG5cbiAgc3RhdGljIG5ldyhcbiAgICAgIHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICAgIHZhbHVlczogRmxvYXQzMkFycmF5fG51bWJlcltdfG51bWJlcltdW11bXVtdKSB7XG4gICAgaWYgKCEodmFsdWVzIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KSkge1xuICAgICAgY29uc3QgaW5mZXJyZWRTaGFwZSA9IHV0aWwuaW5mZXJTaGFwZSh2YWx1ZXMpO1xuICAgICAgaWYgKGluZmVycmVkU2hhcGUubGVuZ3RoID4gMSkge1xuICAgICAgICB1dGlsLmFzc2VydFNoYXBlc01hdGNoKFxuICAgICAgICAgICAgc2hhcGUsIGluZmVycmVkU2hhcGUsXG4gICAgICAgICAgICBgRXJyb3Igd2hlbiBjb25zdHJ1Y3RpbmcgQXJyYXk0RC4gU2hhcGUgb2YgdmFsdWVzIGAgK1xuICAgICAgICAgICAgICAgIGAke2luZmVycmVkU2hhcGV9IGRvZXMgbm90IG1hdGNoIHRoZSBwcm92aWRlZCBzaGFwZSBgICtcbiAgICAgICAgICAgICAgICBgJHtzaGFwZX0uIGApO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gbmV3IEFycmF5NEQoc2hhcGUsIHt2YWx1ZXM6IHRvVHlwZWRBcnJheSh2YWx1ZXMpfSk7XG4gIH1cblxuICBnZXQoaTogbnVtYmVyLCBqOiBudW1iZXIsIGs6IG51bWJlciwgbDogbnVtYmVyKSB7XG4gICAgcmV0dXJuIHRoaXMuZ2V0VmFsdWVzKClcbiAgICAgICAgW3RoaXMuc3RyaWRlMCAqIGkgKyB0aGlzLnN0cmlkZTEgKiBqICsgdGhpcy5zdHJpZGUyICogayArIGxdO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyLCBrOiBudW1iZXIsIGw6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClcbiAgICAgICAgW3RoaXMuc3RyaWRlMCAqIGkgKyB0aGlzLnN0cmlkZTEgKiBqICsgdGhpcy5zdHJpZGUyICogayArIGxdID0gdmFsdWU7XG4gIH1cblxuICBhZGQodmFsdWU6IG51bWJlciwgaTogbnVtYmVyLCBqOiBudW1iZXIsIGs6IG51bWJlciwgbDogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVxuICAgICAgICBbdGhpcy5zdHJpZGUwICogaSArIHRoaXMuc3RyaWRlMSAqIGogKyB0aGlzLnN0cmlkZTIgKiBrICsgbF0gKz0gdmFsdWU7XG4gIH1cblxuICBsb2NUb0luZGV4KGxvY3M6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5zdHJpZGUwICogbG9jc1swXSArIHRoaXMuc3RyaWRlMSAqIGxvY3NbMV0gK1xuICAgICAgICB0aGlzLnN0cmlkZTIgKiBsb2NzWzJdICsgbG9jc1szXTtcbiAgfVxuXG4gIGluZGV4VG9Mb2MoaW5kZXg6IG51bWJlcik6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdIHtcbiAgICBjb25zdCBpID0gTWF0aC5mbG9vcihpbmRleCAvIHRoaXMuc3RyaWRlMCk7XG4gICAgaW5kZXggLT0gaSAqIHRoaXMuc3RyaWRlMDtcbiAgICBjb25zdCBqID0gTWF0aC5mbG9vcihpbmRleCAvIHRoaXMuc3RyaWRlMSk7XG4gICAgaW5kZXggLT0gaiAqIHRoaXMuc3RyaWRlMTtcbiAgICByZXR1cm4gW2ksIGosIE1hdGguZmxvb3IoaW5kZXggLyB0aGlzLnN0cmlkZTIpLCBpbmRleCAlIHRoaXMuc3RyaWRlMl07XG4gIH1cblxuICBzdGF0aWMgemVyb3Moc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdKTogQXJyYXk0RCB7XG4gICAgcmV0dXJuIE5EQXJyYXkuemVyb3M8QXJyYXk0RD4oc2hhcGUpO1xuICB9XG59XG5cbnR5cGUgQXJyYXlEYXRhID0gRmxvYXQzMkFycmF5fG51bWJlcltdfG51bWJlcltdW118bnVtYmVyW11bXVtdfG51bWJlcltdW11bXVtdO1xuXG5mdW5jdGlvbiB0b1R5cGVkQXJyYXkoYTogQXJyYXlEYXRhKTogRmxvYXQzMkFycmF5IHtcbiAgcmV0dXJuIChhIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KSA/IGEgOiBuZXcgRmxvYXQzMkFycmF5KHV0aWwuZmxhdHRlbihhKSk7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi9jb252X3V0aWwnO1xuXG5pbXBvcnQgKiBhcyBjb252X2dwdSBmcm9tICcuL2NvbnZfZ3B1JztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuL2dwZ3B1X2NvbnRleHQnO1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhZ21lbnRTaGFkZXJEZXJXZWlnaHRzU291cmNlKFxuICAgIHhTaGFwZVJvd0NvbERlcHRoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZTaXplOiBudW1iZXIsXG4gICAgb3V0cHV0RGVwdGg6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHplcm9QYWQ6IG51bWJlcikge1xuICBjb25zdCBnZXRNYXRyaXhWYWx1ZU9yWmVyb1BhZCA9XG4gICAgICBjb252X2dwdS5nZXRGcmFnbWVudFNoYWRlckdldE1hdHJpeFZhbHVlT3JaZXJvUGFkU291cmNlKCk7XG4gIGNvbnN0IGlucHV0RGVwdGggPSB4U2hhcGVSb3dDb2xEZXB0aFsyXTtcblxuICBjb25zdCB4VGV4U2hhcGVSQyA9IGNvbnZfdXRpbC5jb21wdXRlVGV4U2hhcGVGcm9tM0QoeFNoYXBlUm93Q29sRGVwdGgpO1xuXG4gIGNvbnN0IHlTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgIHhTaGFwZVJvd0NvbERlcHRoLCBmU2l6ZSwgb3V0cHV0RGVwdGgsIHN0cmlkZSwgemVyb1BhZCk7XG4gIGNvbnN0IHlOdW1Sb3dzID0geVNoYXBlWzBdO1xuICBjb25zdCB5TnVtQ29scyA9IHlTaGFwZVsxXTtcbiAgY29uc3QgeVRleFNoYXBlUkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKHlTaGFwZSk7XG5cbiAgY29uc3QgZlNpemVUaW1lc0lucHV0RGVwdGggPSBmU2l6ZSAqIGlucHV0RGVwdGg7XG5cbiAgY29uc3QgcHJvbG9ndWUgPSBgXG4gICAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICAgIHVuaWZvcm0gc2FtcGxlcjJEIHg7XG4gICAgdW5pZm9ybSBzYW1wbGVyMkQgZHk7XG4gIGA7XG5cbiAgcmV0dXJuIHByb2xvZ3VlICsgJ1xcbicgKyBnZXRNYXRyaXhWYWx1ZU9yWmVyb1BhZCArICdcXG4nICtcbiAgICAgIGBcbiAgICBjb25zdCB2ZWMyIGhhbGZDUiA9IHZlYzIoMC41LCAwLjUpO1xuICAgIGNvbnN0IHZlYzIgeFNoYXBlQ1IgPSB2ZWMyKCR7eFRleFNoYXBlUkNbMV19LCAke3hUZXhTaGFwZVJDWzBdfSk7XG4gICAgY29uc3QgdmVjMiBkeVNoYXBlQ1IgPSB2ZWMyKCR7eVRleFNoYXBlUkNbMV19LCAke3lUZXhTaGFwZVJDWzBdfSk7XG5cbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICB2ZWMyIHdUZXhDUiA9IGZsb29yKGdsX0ZyYWdDb29yZC54eSk7XG5cbiAgICAgIC8vIE1hcCBmcm9tIDJEICh3VGV4Uiwgd1RleEMpIHRvIDREICh3Uiwgd0MsIGQxLCBkMikuXG4gICAgICBmbG9hdCB3UiA9IGZsb29yKHdUZXhDUi55IC8gJHtmU2l6ZVRpbWVzSW5wdXREZXB0aH0uMCk7XG4gICAgICBmbG9hdCB3VGV4UkxlZnRvdmVyID0gd1RleENSLnkgLSB3UiAqICR7ZlNpemVUaW1lc0lucHV0RGVwdGh9LjA7XG4gICAgICBmbG9hdCB3QyA9IGZsb29yKHdUZXhSTGVmdG92ZXIgLyAke2lucHV0RGVwdGh9LjApO1xuICAgICAgZmxvYXQgZDEgPSBtb2Qod1RleFJMZWZ0b3ZlciwgJHtpbnB1dERlcHRofS4wKTtcbiAgICAgIGZsb2F0IGQyID0gd1RleENSLng7XG5cbiAgICAgIC8vIENvbnZvbHZlIHgoPywgPywgZDEpIHdpdGggZHkoOiwgOiwgZDIpIHRvIGdldCBkdyh3Uiwgd0MsIGQxLCBkMikuXG4gICAgICAvLyA/ID0gdG8gYmUgZGV0ZXJtaW5lZC4gOiA9IGFjcm9zcyBhbGwgdmFsdWVzIGluIHRoYXQgYXhpcy5cbiAgICAgIGZsb2F0IGRvdFByb2QgPSAwLjA7XG4gICAgICBmb3IgKGZsb2F0IHlSID0gMC4wOyB5UiA8ICR7eU51bVJvd3N9LjA7IHlSICs9IDEuMCkge1xuICAgICAgICBmbG9hdCB4UiA9IHdSICsgeVIgKiAke3N0cmlkZX0uMCAtICR7emVyb1BhZH0uMDtcbiAgICAgICAgZmxvYXQgeFRleFIgPSB4UjtcbiAgICAgICAgZmxvYXQgeVRleFIgPSB5UjtcbiAgICAgICAgZm9yIChmbG9hdCB5QyA9IDAuMDsgeUMgPCAke3lOdW1Db2xzfS4wOyB5QyArPSAxLjApIHtcbiAgICAgICAgICBmbG9hdCB4QyA9IHdDICsgeUMgKiAke3N0cmlkZX0uMCAtICR7emVyb1BhZH0uMDtcblxuICAgICAgICAgIC8vIE1hcCBmcm9tIDNEICh4UiwgeEMsIGQxKSB0byAyRCAoeFRleFIsIHhUZXhDKS5cbiAgICAgICAgICAvLyBNYXAgZnJvbSAzRCAoeVIsIHlDLCBkMikgdG8gMkQgKHlUZXhSLCB5VGV4QykuXG4gICAgICAgICAgdmVjMiB4eVRleEMgPSB2ZWMyKHhDLCB5QykgKiB2ZWMyKCR7aW5wdXREZXB0aH0uMCwgJHtvdXRwdXREZXB0aH0uMCkgK1xuICAgICAgICAgICAgICAgICAgICAgICAgdmVjMihkMSwgZDIpO1xuICAgICAgICAgIGZsb2F0IHhUZXhDID0geHlUZXhDLng7XG4gICAgICAgICAgZmxvYXQgeVRleEMgPSB4eVRleEMueTtcblxuICAgICAgICAgIC8vIFJlYWQgZHkoeVIsIHlDLCBkMikuXG4gICAgICAgICAgdmVjMiBkeVVWID0gKHZlYzIoeVRleEMsIHlUZXhSKSArIGhhbGZDUikgLyBkeVNoYXBlQ1I7XG4gICAgICAgICAgZmxvYXQgZHlWYWx1ZSA9IHRleHR1cmUyRChkeSwgZHlVVikucjtcblxuICAgICAgICAgIC8vIFJlYWQgeCh4UiwgeEMsIGQxKSAocG90ZW50aWFsbHkgemVyby1wYWRkZWQpLlxuICAgICAgICAgIGZsb2F0IHhWYWx1ZSA9XG4gICAgICAgICAgICBnZXRNYXRyaXhWYWx1ZU9yWmVyb1BhZCh4LCB4U2hhcGVDUiwgdmVjMih4VGV4QywgeFRleFIpKTtcblxuICAgICAgICAgIGRvdFByb2QgKz0gKHhWYWx1ZSAqIGR5VmFsdWUpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KGRvdFByb2QsIDAsIDAsIDApO1xuICAgIH1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhZ21lbnRTaGFkZXJDb252VHJhbnNwb3NlU291cmNlKFxuICAgIHhTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBmU2l6ZTogbnVtYmVyLCBvcmlnSW5wdXREZXB0aDogbnVtYmVyLFxuICAgIG9yaWdTdHJpZGU6IG51bWJlciwgb3JpZ1BhZDogbnVtYmVyLCBoYXNCaWFzOiBib29sZWFuKSB7XG4gIGNvbnN0IHBhZCA9IGZTaXplIC0gMSAtIG9yaWdQYWQ7XG4gIGNvbnN0IFt4Um93cywgeENvbHMsIG9yaWdPdXRwdXREZXB0aF0gPSB4U2hhcGVSQ0Q7XG5cbiAgY29uc3QgeFRleFNoYXBlUkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKHhTaGFwZVJDRCk7XG4gIGNvbnN0IHdUZXhTaGFwZVJDID1cbiAgICAgIGNvbnZfdXRpbC5jb21wdXRlV2VpZ2h0c1RleFNoYXBlKG9yaWdJbnB1dERlcHRoLCBvcmlnT3V0cHV0RGVwdGgsIGZTaXplKTtcblxuICBjb25zdCBnZXRCaWFzVmFsdWUgPSBoYXNCaWFzID9cbiAgICAgIGNvbnZfZ3B1LmdldEZyYWdtZW50U2hhZGVyR2V0Qmlhc1ZhbHVlU291cmNlKG9yaWdJbnB1dERlcHRoKSA6XG4gICAgICAnJztcbiAgY29uc3QgYmlhc1Byb2xvZ3VlID0gaGFzQmlhcyA/ICd1bmlmb3JtIHNhbXBsZXIyRCBiaWFzZXM7JyA6ICcnO1xuICBjb25zdCBiaWFzT3BlcmF0aW9uID0gaGFzQmlhcyA/ICdkb3RQcm9kICs9IGdldEJpYXNWYWx1ZShiaWFzZXMsIGQyKTsnIDogJyc7XG5cbiAgY29uc3QgcHJvbG9ndWUgPSBgXG4gICAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICAgIHVuaWZvcm0gc2FtcGxlcjJEIHg7XG4gICAgdW5pZm9ybSBzYW1wbGVyMkQgd2VpZ2h0cztcbiAgICAke2JpYXNQcm9sb2d1ZX1cbiAgICBgO1xuXG4gIHJldHVybiBwcm9sb2d1ZSArICdcXG4nICsgZ2V0Qmlhc1ZhbHVlICsgJ1xcbicgK1xuICAgICAgYFxuICAgIGNvbnN0IHZlYzIgaGFsZkNSID0gdmVjMigwLjUsIDAuNSk7XG4gICAgY29uc3QgdmVjMiB4U2hhcGVDUiA9IHZlYzIoJHt4VGV4U2hhcGVSQ1sxXX0sICR7eFRleFNoYXBlUkNbMF19KTtcbiAgICBjb25zdCB2ZWMyIHdTaGFwZUNSID0gdmVjMigke3dUZXhTaGFwZVJDWzFdfSwgJHt3VGV4U2hhcGVSQ1swXX0pO1xuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgdmVjMiB5VGV4Q1IgPSBmbG9vcihnbF9GcmFnQ29vcmQueHkpO1xuXG4gICAgICAvLyBNYXAgZnJvbSAyRCAoeVRleFIsIHlUZXhDKSB0byAzRCAoeVIsIHlDLCBkMikuXG4gICAgICBmbG9hdCB5UiA9IHlUZXhDUi55O1xuICAgICAgZmxvYXQgeUMgPSBmbG9vcih5VGV4Q1IueCAvICR7b3JpZ0lucHV0RGVwdGh9LjApO1xuICAgICAgZmxvYXQgZDIgPSBtb2QoeVRleENSLngsICR7b3JpZ0lucHV0RGVwdGh9LjApO1xuXG4gICAgICB2ZWMyIHhSQ0Nvcm5lciA9IHZlYzIoeVIsIHlDKSAtIHZlYzIoJHtwYWR9LjAsICR7cGFkfS4wKTtcbiAgICAgIGZsb2F0IHhSQ29ybmVyID0geFJDQ29ybmVyLng7XG4gICAgICBmbG9hdCB4Q0Nvcm5lciA9IHhSQ0Nvcm5lci55O1xuXG4gICAgICAvLyBDb252b2x2ZSB4KD8sID8sIGQxKSB3aXRoIHcoOiwgOiwgZDIsIGQxKSB0byBnZXQgeSh5UiwgeUMsIGQyKS5cbiAgICAgIC8vID8gPSB0byBiZSBkZXRlcm1pbmVkLiA6ID0gYWNyb3NzIGFsbCB2YWx1ZXMgaW4gdGhhdCBheGlzLlxuICAgICAgZmxvYXQgZG90UHJvZCA9IDAuMDtcbiAgICAgIGZvciAoZmxvYXQgd1IgPSAwLjA7IHdSIDwgJHtmU2l6ZX0uMDsgd1IgKz0gMS4wKSB7XG5cbiAgICAgICAgZmxvYXQgeFIgPSAoeFJDb3JuZXIgKyB3UikgLyAke29yaWdTdHJpZGV9LjA7XG4gICAgICAgIC8vIFRPRE8oc21pbGtvdik6IFNwbGljZSB0aGlzIHdpdGggYW5vdGhlciB2ZXJzaW9uIHdoZXJlIHlvdSBjYWxsXG4gICAgICAgIC8vIGdldE1hdHJpeFZhbHVlT3JaZXJvUGFkKCkuIEhlcmUgYW5kIGJlbG93LlxuICAgICAgICBpZiAoeFIgPCAwLjAgfHwgeFIgPj0gJHt4Um93c30uMCB8fCBmcmFjdCh4UikgPiAwLjApIHtcbiAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgfVxuXG4gICAgICAgIGZsb2F0IHdSUGVybSA9ICR7ZlNpemV9LjAgLSAxLjAgLSB3UjtcbiAgICAgICAgZmxvYXQgeFRleFIgPSB4UjtcblxuICAgICAgICBmb3IgKGZsb2F0IHdDID0gMC4wOyB3QyA8ICR7ZlNpemV9LjA7IHdDICs9IDEuMCkge1xuXG4gICAgICAgICAgZmxvYXQgeEMgPSAoeENDb3JuZXIgKyB3QykgLyAke29yaWdTdHJpZGV9LjA7XG4gICAgICAgICAgaWYgKHhDIDwgMC4wIHx8IHhDID49ICR7eENvbHN9LjAgfHwgZnJhY3QoeEMpID4gMC4wKSB7XG4gICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBmbG9hdCB3Q1Blcm0gPSAke2ZTaXplfS4wIC0gMS4wIC0gd0M7XG4gICAgICAgICAgZmxvYXQgd1RleFIgPSB3UlBlcm0gKiAke2ZTaXplfS4wICogJHtvcmlnSW5wdXREZXB0aH0uMCArXG4gICAgICAgICAgICAgICAgICAgICAgICB3Q1Blcm0gKiAke29yaWdJbnB1dERlcHRofS4wICsgZDI7XG5cbiAgICAgICAgICBmb3IgKGZsb2F0IGQxID0gMC4wOyBkMSA8ICR7b3JpZ091dHB1dERlcHRofS4wOyBkMSArPSAxLjApIHtcbiAgICAgICAgICAgIGZsb2F0IHhUZXhDID0geEMgKiAke29yaWdPdXRwdXREZXB0aH0uMCArIGQxO1xuICAgICAgICAgICAgZmxvYXQgd1RleEMgPSBkMTtcblxuICAgICAgICAgICAgLy8gUmVhZCB4KHhSLCB4QywgZDEpLlxuICAgICAgICAgICAgdmVjMiB4VVYgPSAodmVjMih4VGV4QywgeFRleFIpICsgaGFsZkNSKSAvIHhTaGFwZUNSO1xuICAgICAgICAgICAgZmxvYXQgeFZhbHVlID0gdGV4dHVyZTJEKHgsIHhVVikucjtcblxuICAgICAgICAgICAgLy8gUmVhZCB3KHdSUGVybSwgd0NQZXJtLCBkMiwgZDEpLlxuICAgICAgICAgICAgdmVjMiB3VVYgPSAodmVjMih3VGV4Qywgd1RleFIpICsgaGFsZkNSKSAvIHdTaGFwZUNSO1xuICAgICAgICAgICAgZmxvYXQgd1ZhbHVlID0gdGV4dHVyZTJEKHdlaWdodHMsIHdVVikucjtcblxuICAgICAgICAgICAgZG90UHJvZCArPSB4VmFsdWUgKiB3VmFsdWU7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgICAke2JpYXNPcGVyYXRpb259XG4gICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KGRvdFByb2QsIDAsIDAsIDApO1xuICAgIH1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhZ21lbnRTaGFkZXJEZXJCaWFzU291cmNlKFxuICAgIGR5U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSkge1xuICBjb25zdCBkeVRleFNoYXBlUkMgPSBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKGR5U2hhcGVSQ0QpO1xuICBjb25zdCBbeU51bVJvd3MsIHlOdW1Db2xzLCBvdXRwdXREZXB0aF0gPSBkeVNoYXBlUkNEO1xuXG4gIHJldHVybiBgXG4gICAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICAgIHVuaWZvcm0gc2FtcGxlcjJEIGR5O1xuXG4gICAgY29uc3QgdmVjMiBoYWxmQ1IgPSB2ZWMyKDAuNSwgMC41KTtcbiAgICBjb25zdCB2ZWMyIGR5U2hhcGVDUiA9IHZlYzIoJHtkeVRleFNoYXBlUkNbMV19LCAke2R5VGV4U2hhcGVSQ1swXX0pO1xuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgdmVjMiBiaWFzVGV4Q1IgPSBmbG9vcihnbF9GcmFnQ29vcmQueHkpO1xuXG4gICAgICAvLyBUaGUgYmlhcyB0ZXh0dXJlIFJDIHNoYXBlIGlzIFsxLCBkMl0uXG4gICAgICBmbG9hdCBkMiA9IGJpYXNUZXhDUi54O1xuXG4gICAgICBmbG9hdCBkZXJCaWFzID0gMC4wO1xuICAgICAgZm9yIChmbG9hdCB5UiA9IDAuMDsgeVIgPCAke3lOdW1Sb3dzfS4wOyB5UiArPSAxLjApIHtcbiAgICAgICAgZmxvYXQgeVRleFIgPSB5UjtcblxuICAgICAgICBmb3IgKGZsb2F0IHlDID0gMC4wOyB5QyA8ICR7eU51bUNvbHN9LjA7IHlDICs9IDEuMCkge1xuICAgICAgICAgIC8vIE1hcCBmcm9tIDNEICh5UiwgeUMsIGQyKSB0byAyRCAoeVRleFIsIHlUZXhDKS5cbiAgICAgICAgICBmbG9hdCB5VGV4QyA9IHlDICogJHtvdXRwdXREZXB0aH0uMCArIGQyO1xuXG4gICAgICAgICAgLy8gUmVhZCBkeSh5UiwgeUMsIGQyKS5cbiAgICAgICAgICB2ZWMyIGR5VVYgPSAodmVjMih5VGV4QywgeVRleFIpICsgaGFsZkNSKSAvIGR5U2hhcGVDUjtcbiAgICAgICAgICBmbG9hdCBkeVZhbHVlID0gdGV4dHVyZTJEKGR5LCBkeVVWKS5yO1xuXG4gICAgICAgICAgZGVyQmlhcyArPSBkeVZhbHVlO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBnbF9GcmFnQ29sb3IgPSB2ZWM0KGRlckJpYXMsIDAsIDAsIDApO1xuICAgIH1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZGVyQmlhcyhcbiAgICBncGdwdTogR1BHUFVDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0sIGR5VGV4OiBXZWJHTFRleHR1cmUsXG4gICAgcmVzdWx0OiBXZWJHTFRleHR1cmUsIHJlc3VsdFRleFNoYXBlUkM6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgIHJlc3VsdCwgcmVzdWx0VGV4U2hhcGVSQ1swXSwgcmVzdWx0VGV4U2hhcGVSQ1sxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZShkeVRleCwgJ2R5JywgMCk7XG4gIGdwZ3B1LmV4ZWN1dGVQcm9ncmFtKCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBkZXJXZWlnaHRzKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgeFRleDogV2ViR0xUZXh0dXJlLFxuICAgIGR5VGV4OiBXZWJHTFRleHR1cmUsIHJlc3VsdDogV2ViR0xUZXh0dXJlLFxuICAgIHJlc3VsdFRleFNoYXBlUkM6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgIHJlc3VsdCwgcmVzdWx0VGV4U2hhcGVSQ1swXSwgcmVzdWx0VGV4U2hhcGVSQ1sxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZSh4VGV4LCAneCcsIDApO1xuICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUoZHlUZXgsICdkeScsIDEpO1xuICBncGdwdS5leGVjdXRlUHJvZ3JhbSgpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29udlRyYW5zcG9zZShcbiAgICBncGdwdTogR1BHUFVDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0sIHhUZXg6IFdlYkdMVGV4dHVyZSxcbiAgICB3ZWlnaHRzVGV4OiBXZWJHTFRleHR1cmUsIGJpYXNlc1RleDogV2ViR0xUZXh0dXJlfG51bGwsXG4gICAgcmVzdWx0VGV4OiBXZWJHTFRleHR1cmUsIHJlc3VsdFRleFNoYXBlUkM6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgIHJlc3VsdFRleCwgcmVzdWx0VGV4U2hhcGVSQ1swXSwgcmVzdWx0VGV4U2hhcGVSQ1sxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZSh4VGV4LCAneCcsIDApO1xuICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUod2VpZ2h0c1RleCwgJ3dlaWdodHMnLCAxKTtcbiAgaWYgKGJpYXNlc1RleCAhPSBudWxsKSB7XG4gICAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGJpYXNlc1RleCwgJ2JpYXNlcycsIDIpO1xuICB9XG4gIGdwZ3B1LmV4ZWN1dGVQcm9ncmFtKCk7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi9jb252X3V0aWwnO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vZ3BncHVfY29udGV4dCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlclByb2xvZ3VlU291cmNlKCk6IHN0cmluZyB7XG4gIHJldHVybiBgXG4gICAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICAgIHVuaWZvcm0gc2FtcGxlcjJEIHg7XG4gICAgdW5pZm9ybSBzYW1wbGVyMkQgd2VpZ2h0cztcbiAgICB1bmlmb3JtIHNhbXBsZXIyRCBiaWFzZXM7XG4gICAgdmFyeWluZyB2ZWMyIHJlc3VsdFVWO2A7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlckdldE1hdHJpeFZhbHVlT3JaZXJvUGFkU291cmNlKCk6IHN0cmluZyB7XG4gIHJldHVybiBgXG4gICAgZmxvYXQgZ2V0TWF0cml4VmFsdWVPclplcm9QYWQoaW4gc2FtcGxlcjJEIG1hdHJpeCwgdmVjMiBtYXRyaXhTaGFwZUNSLFxuICAgICAgICB2ZWMyIHJlcXVlc3RlZENSKSB7XG4gICAgICB2ZWMyIHV2ID0gKHJlcXVlc3RlZENSICsgdmVjMigwLjUsIDAuNSkpIC8gbWF0cml4U2hhcGVDUjtcbiAgICAgIGZsb2F0IHZhbHVlID0gdGV4dHVyZTJEKG1hdHJpeCwgdXYpLnI7XG4gICAgICBib29sIGxlc3NUaGFuWmVybyA9IGFueShsZXNzVGhhbih1diwgdmVjMigwLCAwKSkpO1xuICAgICAgYm9vbCBncmVhdGVyVGhhbk9uZSA9IGFueShncmVhdGVyVGhhbih1diwgdmVjMigxLCAxKSkpO1xuICAgICAgYm9vbCBvdXRzaWRlID0gbGVzc1RoYW5aZXJvIHx8IGdyZWF0ZXJUaGFuT25lO1xuICAgICAgcmV0dXJuIG1peCh2YWx1ZSwgMC4wLCBmbG9hdChvdXRzaWRlKSk7XG4gICAgfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlckNvbnZvbHZlU291cmNlKFxuICAgIHhTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBmU2l6ZTogbnVtYmVyLCBvdXRwdXREZXB0aDogbnVtYmVyLFxuICAgIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlciwgaGFzQmlhczogYm9vbGVhbikge1xuICBjb25zdCBbeFJvd3MsIHhDb2xzLCBpbnB1dERlcHRoXSA9IHhTaGFwZVJDRDtcblxuICBjb25zdCB4VGV4U2hhcGVSQyA9IGNvbnZfdXRpbC5jb21wdXRlVGV4U2hhcGVGcm9tM0QoeFNoYXBlUkNEKTtcbiAgY29uc3Qgd1RleFNoYXBlUkMgPVxuICAgICAgY29udl91dGlsLmNvbXB1dGVXZWlnaHRzVGV4U2hhcGUoaW5wdXREZXB0aCwgb3V0cHV0RGVwdGgsIGZTaXplKTtcblxuICByZXR1cm4gYFxuICAgIGNvbnN0IHZlYzIgaGFsZkNSID0gdmVjMigwLjUsIDAuNSk7XG4gICAgY29uc3QgdmVjMiB4U2hhcGVDUiA9IHZlYzIoJHt4VGV4U2hhcGVSQ1sxXX0sICR7eFRleFNoYXBlUkNbMF19KTtcbiAgICBjb25zdCB2ZWMyIHdTaGFwZUNSID0gdmVjMigke3dUZXhTaGFwZVJDWzFdfSwgJHt3VGV4U2hhcGVSQ1swXX0pO1xuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgdmVjMiB5VGV4Q1IgPSBmbG9vcihnbF9GcmFnQ29vcmQueHkpO1xuXG4gICAgICAvLyBNYXAgZnJvbSAyRCAoeVRleFIsIHlUZXhDKSB0byAzRCAoeVIsIHlDLCBkMikuXG4gICAgICBmbG9hdCB5UiA9IHlUZXhDUi55O1xuICAgICAgZmxvYXQgeUMgPSBmbG9vcih5VGV4Q1IueCAvICR7b3V0cHV0RGVwdGh9LjApO1xuICAgICAgZmxvYXQgZDIgPSBtb2QoeVRleENSLngsICR7b3V0cHV0RGVwdGh9LjApO1xuICAgICAgZmxvYXQgd1RleEMgPSBkMjtcblxuICAgICAgdmVjMiB4UkNDb3JuZXIgPSB2ZWMyKHlSLCB5QykgKiB2ZWMyKCR7c3RyaWRlfSwgJHtzdHJpZGV9KSAtXG4gICAgICAgICAgdmVjMigke3BhZH0uMCwgJHtwYWR9LjApO1xuICAgICAgZmxvYXQgeFJDb3JuZXIgPSB4UkNDb3JuZXIueDtcbiAgICAgIGZsb2F0IHhDQ29ybmVyID0geFJDQ29ybmVyLnk7XG5cbiAgICAgIC8vIENvbnZvbHZlIHgoPywgPywgZDEpIHdpdGggdyg6LCA6LCBkMSwgZDIpIHRvIGdldCB5KHlSLCB5QywgZDIpLlxuICAgICAgLy8gPyA9IHRvIGJlIGRldGVybWluZWQuIDogPSBhY3Jvc3MgYWxsIHZhbHVlcyBpbiB0aGF0IGF4aXMuXG4gICAgICBmbG9hdCBkb3RQcm9kID0gMC4wO1xuICAgICAgZm9yIChmbG9hdCB3UiA9IDAuMDsgd1IgPCAke2ZTaXplfS4wOyB3UiArPSAxLjApIHtcbiAgICAgICAgZmxvYXQgeFIgPSB4UkNvcm5lciArIHdSO1xuICAgICAgICBmbG9hdCB4VGV4UiA9IHhSO1xuXG4gICAgICAgIGZvciAoZmxvYXQgd0MgPSAwLjA7IHdDIDwgJHtmU2l6ZX0uMDsgd0MgKz0gMS4wKSB7XG4gICAgICAgICAgZmxvYXQgeEMgPSB4Q0Nvcm5lciArIHdDO1xuXG4gICAgICAgICAgZm9yIChmbG9hdCBkMSA9IDAuMDsgZDEgPCAke2lucHV0RGVwdGh9LjA7IGQxICs9IDEuMCkge1xuICAgICAgICAgICAgZmxvYXQgeFRleEMgPSB4QyAqICR7aW5wdXREZXB0aH0uMCArIGQxO1xuICAgICAgICAgICAgZmxvYXQgd1RleFIgPSB3UiAqICR7ZlNpemUgKiBpbnB1dERlcHRofS4wICtcbiAgICAgICAgICAgICAgICB3QyAqICR7aW5wdXREZXB0aH0uMCArIGQxO1xuXG4gICAgICAgICAgICBmbG9hdCB4VmFsdWUgPVxuICAgICAgICAgICAgICAgIGdldE1hdHJpeFZhbHVlT3JaZXJvUGFkKHgsIHhTaGFwZUNSLCB2ZWMyKHhUZXhDLCB4VGV4UikpO1xuXG4gICAgICAgICAgICAvLyBSZWFkIHcod1IsIHdDLCBkMSwgZDIpLlxuICAgICAgICAgICAgdmVjMiB3VVYgPSAodmVjMih3VGV4Qywgd1RleFIpICsgaGFsZkNSKSAvIHdTaGFwZUNSO1xuICAgICAgICAgICAgZmxvYXQgd1ZhbHVlID0gdGV4dHVyZTJEKHdlaWdodHMsIHdVVikucjtcblxuICAgICAgICAgICAgZG90UHJvZCArPSB4VmFsdWUgKiB3VmFsdWU7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgICBpZiAoJHtoYXNCaWFzfSkge1xuICAgICAgICBkb3RQcm9kICs9IGdldEJpYXNWYWx1ZShiaWFzZXMsIGQyKTtcbiAgICAgIH1cbiAgICAgIGdsX0ZyYWdDb2xvciA9IHZlYzQoZG90UHJvZCwgMCwgMCwgMCk7XG4gICAgfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlckdldEJpYXNWYWx1ZVNvdXJjZShvdXRwdXREZXB0aDogbnVtYmVyKTpcbiAgICBzdHJpbmcge1xuICByZXR1cm4gYFxuICAgIGZsb2F0IGdldEJpYXNWYWx1ZShpbiBzYW1wbGVyMkQgYmlhcywgZmxvYXQgYmlhc0MpIHtcbiAgICAgIGNvbnN0IHZlYzIgYmlhc1NoYXBlQ1IgPSB2ZWMyKCR7b3V0cHV0RGVwdGh9LCAxKTtcbiAgICAgIHZlYzIgYmlhc0NSID0gdmVjMihtb2QoYmlhc0MsICR7b3V0cHV0RGVwdGh9LjApLCAwKTtcbiAgICAgIHZlYzIgYmlhc1VWID0gKGJpYXNDUiArIHZlYzIoMC41LCAwLjUpKSAvIGJpYXNTaGFwZUNSO1xuICAgICAgcmV0dXJuIHRleHR1cmUyRChiaWFzLCBiaWFzVVYpLnI7XG4gICAgfWA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlclNvdXJjZShcbiAgICBhU2hhcGVSb3dDb2xEZXB0aDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCByZXN1bHREZXB0aDogbnVtYmVyLFxuICAgIGZpZWxkU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgemVyb1BhZDogbnVtYmVyLFxuICAgIGhhc0JpYXM6IGJvb2xlYW4pOiBzdHJpbmcge1xuICBjb25zdCBhU2hhcGVSQzogW251bWJlciwgbnVtYmVyXSA9XG4gICAgICBjb252X3V0aWwuY29tcHV0ZVRleFNoYXBlRnJvbTNEKGFTaGFwZVJvd0NvbERlcHRoKTtcblxuICBjb25zdCB3ZWlnaHRTaGFwZVJDOiBbbnVtYmVyLCBudW1iZXJdID0gY29udl91dGlsLmNvbXB1dGVXZWlnaHRzVGV4U2hhcGUoXG4gICAgICBhU2hhcGVSb3dDb2xEZXB0aFsyXSwgcmVzdWx0RGVwdGgsIGZpZWxkU2l6ZSk7XG5cbiAgY29uc3QgcHJvbG9ndWUgPSBnZXRGcmFnbWVudFNoYWRlclByb2xvZ3VlU291cmNlKCk7XG4gIGNvbnN0IGdldE1hdHJpeFZhbHVlT3JaZXJvUGFkID1cbiAgICAgIGdldEZyYWdtZW50U2hhZGVyR2V0TWF0cml4VmFsdWVPclplcm9QYWRTb3VyY2UoKTtcbiAgY29uc3QgY29udm9sdmUgPSBnZXRGcmFnbWVudFNoYWRlckNvbnZvbHZlU291cmNlKFxuICAgICAgYVNoYXBlUm93Q29sRGVwdGgsIGZpZWxkU2l6ZSwgcmVzdWx0RGVwdGgsIHN0cmlkZSwgemVyb1BhZCwgaGFzQmlhcyk7XG4gIGNvbnN0IGdldEJpYXNWYWx1ZSA9IGdldEZyYWdtZW50U2hhZGVyR2V0Qmlhc1ZhbHVlU291cmNlKHJlc3VsdERlcHRoKTtcblxuICByZXR1cm4gW1xuICAgIHByb2xvZ3VlLFxuICAgIGdldE1hdHJpeFZhbHVlT3JaZXJvUGFkLFxuICAgIGdldEJpYXNWYWx1ZSxcbiAgICBjb252b2x2ZSxcbiAgXS5qb2luKCdcXG4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbnZvbHZlKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgYTogV2ViR0xUZXh0dXJlLFxuICAgIHdlaWdodHM6IFdlYkdMVGV4dHVyZSwgYmlhc2VzOiBXZWJHTFRleHR1cmV8bnVsbCwgcmVzdWx0OiBXZWJHTFRleHR1cmUsXG4gICAgcmVzdWx0U2hhcGVSb3dDb2w6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgIHJlc3VsdCwgcmVzdWx0U2hhcGVSb3dDb2xbMF0sIHJlc3VsdFNoYXBlUm93Q29sWzFdKTtcbiAgZ3BncHUuc2V0UHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGEsICd4JywgMCk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZSh3ZWlnaHRzLCAnd2VpZ2h0cycsIDEpO1xuICBpZiAoYmlhc2VzICE9IG51bGwpIHtcbiAgICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUoYmlhc2VzLCAnYmlhc2VzJywgMik7XG4gIH1cbiAgZ3BncHUuZXhlY3V0ZVByb2dyYW0oKTtcbn0iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGdwZ3B1X3V0aWwgZnJvbSAnLi9ncGdwdV91dGlsJztcbmltcG9ydCAqIGFzIHRleF91dGlsIGZyb20gJy4vdGV4X3V0aWwnO1xuaW1wb3J0ICogYXMgd2ViZ2xfdXRpbCBmcm9tICcuL3dlYmdsX3V0aWwnO1xuXG5pbXBvcnQge1dlYkdMTG9zZUNvbnRleHRFeHRlbnNpb259IGZyb20gJy4vd2ViZ2xfdXRpbCc7XG5cbmV4cG9ydCBjbGFzcyBHUEdQVUNvbnRleHQge1xuICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0O1xuICB0ZXh0dXJlRmxvYXRFeHRlbnNpb246IHt9O1xuICBjb2xvckJ1ZmZlckZsb2F0RXh0ZW5zaW9uOiB7fTtcbiAgbG9zZUNvbnRleHRFeHRlbnNpb246IFdlYkdMTG9zZUNvbnRleHRFeHRlbnNpb247XG4gIHZlcnRleEJ1ZmZlcjogV2ViR0xCdWZmZXI7XG4gIGluZGV4QnVmZmVyOiBXZWJHTEJ1ZmZlcjtcbiAgZnJhbWVidWZmZXI6IFdlYkdMRnJhbWVidWZmZXI7XG4gIG91dHB1dFRleHR1cmU6IFdlYkdMVGV4dHVyZXxudWxsID0gbnVsbDtcbiAgcHJvZ3JhbTogV2ViR0xQcm9ncmFtfG51bGwgPSBudWxsO1xuICBwcml2YXRlIGRpc3Bvc2VkID0gZmFsc2U7XG4gIHByaXZhdGUgYXV0b0RlYnVnVmFsaWRhdGUgPSBmYWxzZTtcblxuICBjb25zdHJ1Y3RvcihnbD86IFdlYkdMUmVuZGVyaW5nQ29udGV4dCkge1xuICAgIGlmIChnbCAhPSBudWxsKSB7XG4gICAgICB0aGlzLmdsID0gZ2w7XG4gICAgfSBlbHNlIHtcbiAgICAgIHRoaXMuZ2wgPSBncGdwdV91dGlsLmNyZWF0ZVdlYkdMQ29udGV4dCgpO1xuICAgIH1cblxuICAgIC8vIFdlYkdMIDIuMCBlbmFibGVzIHRleHR1cmUgZmxvYXRzIHdpdGhvdXQgYW4gZXh0ZW5zaW9uLlxuICAgIGlmICghd2ViZ2xfdXRpbC5pc1dlYkdMMkVuYWJsZWQoKSkge1xuICAgICAgdGhpcy50ZXh0dXJlRmxvYXRFeHRlbnNpb24gPVxuICAgICAgICAgIHdlYmdsX3V0aWwuZ2V0RXh0ZW5zaW9uT3JUaHJvdyh0aGlzLmdsLCAnT0VTX3RleHR1cmVfZmxvYXQnKTtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5jb2xvckJ1ZmZlckZsb2F0RXh0ZW5zaW9uID1cbiAgICAgICAgICB3ZWJnbF91dGlsLmdldEV4dGVuc2lvbk9yVGhyb3codGhpcy5nbCwgJ0VYVF9jb2xvcl9idWZmZXJfZmxvYXQnKTtcbiAgICB9XG5cbiAgICB0aGlzLmxvc2VDb250ZXh0RXh0ZW5zaW9uID1cbiAgICAgICAgd2ViZ2xfdXRpbC5nZXRFeHRlbnNpb25PclRocm93KHRoaXMuZ2wsICdXRUJHTF9sb3NlX2NvbnRleHQnKSBhc1xuICAgICAgICBXZWJHTExvc2VDb250ZXh0RXh0ZW5zaW9uO1xuICAgIHRoaXMudmVydGV4QnVmZmVyID0gZ3BncHVfdXRpbC5jcmVhdGVWZXJ0ZXhCdWZmZXIodGhpcy5nbCk7XG4gICAgdGhpcy5pbmRleEJ1ZmZlciA9IGdwZ3B1X3V0aWwuY3JlYXRlSW5kZXhCdWZmZXIodGhpcy5nbCk7XG4gICAgdGhpcy5mcmFtZWJ1ZmZlciA9IHdlYmdsX3V0aWwuY3JlYXRlRnJhbWVidWZmZXIodGhpcy5nbCk7XG4gIH1cblxuICBwdWJsaWMgZGlzcG9zZSgpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGlmICh0aGlzLnByb2dyYW0gIT0gbnVsbCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICdEaXNwb3NpbmcgYSBHUEdQVUNvbnRleHQgdGhhdCBzdGlsbCBoYXMgYSBib3VuZCBXZWJHTFByb2dyYW0uJyArXG4gICAgICAgICAgJyBUaGlzIGlzIHByb2JhYmx5IGEgcmVzb3VyY2UgbGVhaywgZGVsZXRlIHRoZSBwcm9ncmFtIHdpdGggJyArXG4gICAgICAgICAgJ0dQR1BVQ29udGV4dC5kZWxldGVQcm9ncmFtIGJlZm9yZSBkaXNwb3NpbmcuJyk7XG4gICAgfVxuICAgIGlmICh0aGlzLm91dHB1dFRleHR1cmUgIT0gbnVsbCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICdEaXNwb3NpbmcgYSBHUEdQVUNvbnRleHQgdGhhdCBzdGlsbCBoYXMgYSBib3VuZCBvdXRwdXQgbWF0cml4ICcgK1xuICAgICAgICAgICd0ZXh0dXJlLiAgVGhpcyBpcyBwcm9iYWJseSBhIHJlc291cmNlIGxlYWssIGRlbGV0ZSB0aGUgb3V0cHV0ICcgK1xuICAgICAgICAgICdtYXRyaXggdGV4dHVyZSB3aXRoIEdQR1BVQ29udGV4dC5kZWxldGVNYXRyaXhUZXh0dXJlIGJlZm9yZSAnICtcbiAgICAgICAgICAnZGlzcG9zaW5nLicpO1xuICAgIH1cbiAgICBjb25zdCBnbCA9IHRoaXMuZ2w7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmZpbmlzaCgpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBudWxsKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRlbGV0ZUZyYW1lYnVmZmVyKHRoaXMuZnJhbWVidWZmZXIpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5BUlJBWV9CVUZGRVIsIG51bGwpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGVsZXRlQnVmZmVyKHRoaXMudmVydGV4QnVmZmVyKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICAgIGdsLCAoKSA9PiBnbC5iaW5kQnVmZmVyKGdsLkVMRU1FTlRfQVJSQVlfQlVGRkVSLCBudWxsKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRlbGV0ZUJ1ZmZlcih0aGlzLmluZGV4QnVmZmVyKSk7XG4gICAgdGhpcy5sb3NlQ29udGV4dEV4dGVuc2lvbi5sb3NlQ29udGV4dCgpO1xuICAgIHRoaXMuZGlzcG9zZWQgPSB0cnVlO1xuICB9XG5cbiAgcHVibGljIGVuYWJsZUF1dG9tYXRpY0RlYnVnVmFsaWRhdGlvbihlbmFibGVkOiBib29sZWFuKSB7XG4gICAgdGhpcy5hdXRvRGVidWdWYWxpZGF0ZSA9IGVuYWJsZWQ7XG4gICAgd2ViZ2xfdXRpbC5lbmFibGVEZWJ1Z1dlYkdMRXJyb3JDaGVja2luZyhlbmFibGVkKTtcbiAgfVxuXG4gIHB1YmxpYyBjcmVhdGVNYXRyaXhUZXh0dXJlKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogV2ViR0xUZXh0dXJlIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHJldHVybiBncGdwdV91dGlsLmNyZWF0ZU1hdHJpeFRleHR1cmUodGhpcy5nbCwgcm93cywgY29sdW1ucyk7XG4gIH1cblxuICBwdWJsaWMgdXBsb2FkUGl4ZWxEYXRhVG9UZXh0dXJlKFxuICAgICAgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgICAgcGl4ZWxzOiBJbWFnZURhdGF8SFRNTEltYWdlRWxlbWVudHxIVE1MQ2FudmFzRWxlbWVudHxIVE1MVmlkZW9FbGVtZW50KSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICBncGdwdV91dGlsLnVwbG9hZFBpeGVsRGF0YVRvVGV4dHVyZSh0aGlzLmdsLCB0ZXh0dXJlLCBwaXhlbHMpO1xuICB9XG5cbiAgcHVibGljIGNyZWF0ZVBhY2tlZE1hdHJpeFRleHR1cmUocm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOlxuICAgICAgV2ViR0xUZXh0dXJlIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHJldHVybiBncGdwdV91dGlsLmNyZWF0ZVBhY2tlZE1hdHJpeFRleHR1cmUodGhpcy5nbCwgcm93cywgY29sdW1ucyk7XG4gIH1cblxuICBwdWJsaWMgZGVsZXRlTWF0cml4VGV4dHVyZSh0ZXh0dXJlOiBXZWJHTFRleHR1cmUpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGlmICh0aGlzLm91dHB1dFRleHR1cmUgPT09IHRleHR1cmUpIHtcbiAgICAgIHdlYmdsX3V0aWwudW5iaW5kQ29sb3JUZXh0dXJlRnJvbUZyYW1lYnVmZmVyKHRoaXMuZ2wsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgICAgdGhpcy5vdXRwdXRUZXh0dXJlID0gbnVsbDtcbiAgICB9XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2sodGhpcy5nbCwgKCkgPT4gdGhpcy5nbC5kZWxldGVUZXh0dXJlKHRleHR1cmUpKTtcbiAgfVxuXG4gIHB1YmxpYyB1cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyLFxuICAgICAgbWF0cml4OiBGbG9hdDMyQXJyYXkpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGNvbnN0IG51bUNoYW5uZWxzID0gMTtcbiAgICByZXR1cm4gZ3BncHVfdXRpbC51cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICAgIHRoaXMuZ2wsIHRleHR1cmUsIHJvd3MsIGNvbHVtbnMsIG1hdHJpeCwgbnVtQ2hhbm5lbHMpO1xuICB9XG5cbiAgcHVibGljIHVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIsXG4gICAgICBtYXRyaXg6IEZsb2F0MzJBcnJheSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwudXBsb2FkTWF0cml4VG9QYWNrZWRUZXh0dXJlKFxuICAgICAgICB0aGlzLmdsLCB0ZXh0dXJlLCByb3dzLCBjb2x1bW5zLCBtYXRyaXgpO1xuICB9XG5cbiAgcHVibGljIGRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUoXG4gICAgICB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogRmxvYXQzMkFycmF5IHtcbiAgICByZXR1cm4gdGhpcy5kb3dubG9hZE1hdHJpeERyaXZlcihcbiAgICAgICAgdGV4dHVyZSxcbiAgICAgICAgKCkgPT5cbiAgICAgICAgICAgIGdwZ3B1X3V0aWwuZG93bmxvYWRNYXRyaXhGcm9tT3V0cHV0VGV4dHVyZSh0aGlzLmdsLCByb3dzLCBjb2x1bW5zKSk7XG4gIH1cblxuICBwdWJsaWMgZG93bmxvYWRNYXRyaXhGcm9tUGFja2VkVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBGbG9hdDMyQXJyYXkge1xuICAgIHJldHVybiB0aGlzLmRvd25sb2FkTWF0cml4RHJpdmVyKFxuICAgICAgICB0ZXh0dXJlLFxuICAgICAgICAoKSA9PiBncGdwdV91dGlsLmRvd25sb2FkTWF0cml4RnJvbVBhY2tlZE91dHB1dFRleHR1cmUoXG4gICAgICAgICAgICB0aGlzLmdsLCByb3dzLCBjb2x1bW5zKSk7XG4gIH1cblxuICBwdWJsaWMgY3JlYXRlUHJvZ3JhbShmcmFnbWVudFNoYWRlclNvdXJjZTogc3RyaW5nKTogV2ViR0xQcm9ncmFtIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGNvbnN0IGdsID0gdGhpcy5nbDtcbiAgICBjb25zdCBmcmFnbWVudFNoYWRlcjogV2ViR0xTaGFkZXIgPVxuICAgICAgICB3ZWJnbF91dGlsLmNyZWF0ZUZyYWdtZW50U2hhZGVyKGdsLCBmcmFnbWVudFNoYWRlclNvdXJjZSk7XG4gICAgY29uc3QgdmVydGV4U2hhZGVyOiBXZWJHTFNoYWRlciA9IGdwZ3B1X3V0aWwuY3JlYXRlVmVydGV4U2hhZGVyKGdsKTtcbiAgICBjb25zdCBwcm9ncmFtOiBXZWJHTFByb2dyYW0gPSB3ZWJnbF91dGlsLmNyZWF0ZVByb2dyYW0oZ2wpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5hdHRhY2hTaGFkZXIocHJvZ3JhbSwgdmVydGV4U2hhZGVyKSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmF0dGFjaFNoYWRlcihwcm9ncmFtLCBmcmFnbWVudFNoYWRlcikpO1xuICAgIHdlYmdsX3V0aWwubGlua1Byb2dyYW0oZ2wsIHByb2dyYW0pO1xuICAgIGlmICh0aGlzLmF1dG9EZWJ1Z1ZhbGlkYXRlKSB7XG4gICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlUHJvZ3JhbShnbCwgcHJvZ3JhbSk7XG4gICAgfVxuXG4gICAgcmV0dXJuIHByb2dyYW07XG4gIH1cblxuICBwdWJsaWMgZGVsZXRlUHJvZ3JhbShwcm9ncmFtOiBXZWJHTFByb2dyYW0pIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGlmIChwcm9ncmFtID09PSB0aGlzLnByb2dyYW0pIHtcbiAgICAgIHRoaXMucHJvZ3JhbSA9IG51bGw7XG4gICAgfVxuICAgIGlmIChwcm9ncmFtICE9IG51bGwpIHtcbiAgICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKHRoaXMuZ2wsICgpID0+IHRoaXMuZ2wuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKSk7XG4gICAgfVxuICB9XG5cbiAgcHVibGljIHNldFByb2dyYW0ocHJvZ3JhbTogV2ViR0xQcm9ncmFtfG51bGwpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHRoaXMucHJvZ3JhbSA9IHByb2dyYW07XG4gICAgaWYgKCh0aGlzLnByb2dyYW0gIT0gbnVsbCkgJiYgdGhpcy5hdXRvRGVidWdWYWxpZGF0ZSkge1xuICAgICAgd2ViZ2xfdXRpbC52YWxpZGF0ZVByb2dyYW0odGhpcy5nbCwgdGhpcy5wcm9ncmFtKTtcbiAgICB9XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2sodGhpcy5nbCwgKCkgPT4gdGhpcy5nbC51c2VQcm9ncmFtKHByb2dyYW0pKTtcbiAgfVxuXG4gIHB1YmxpYyBnZXRVbmlmb3JtTG9jYXRpb24odW5pZm9ybU5hbWU6IHN0cmluZyk6IFdlYkdMVW5pZm9ybUxvY2F0aW9uIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHRoaXMudGhyb3dJZk5vUHJvZ3JhbSgpO1xuICAgIHJldHVybiB3ZWJnbF91dGlsLmdldFByb2dyYW1Vbmlmb3JtTG9jYXRpb25PclRocm93KFxuICAgICAgICB0aGlzLmdsLCB0aGlzLnByb2dyYW0hLCB1bmlmb3JtTmFtZSk7XG4gIH1cblxuICBwdWJsaWMgc2V0SW5wdXRNYXRyaXhUZXh0dXJlKFxuICAgICAgaW5wdXRNYXRyaXhUZXh0dXJlOiBXZWJHTFRleHR1cmUsIHVuaWZvcm1OYW1lOiBzdHJpbmcsXG4gICAgICB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB0aGlzLnRocm93SWZOb1Byb2dyYW0oKTtcbiAgICB3ZWJnbF91dGlsLmJpbmRUZXh0dXJlVG9Qcm9ncmFtVW5pZm9ybVNhbXBsZXIoXG4gICAgICAgIHRoaXMuZ2wsIHRoaXMucHJvZ3JhbSEsIGlucHV0TWF0cml4VGV4dHVyZSwgdW5pZm9ybU5hbWUsIHRleHR1cmVVbml0KTtcbiAgfVxuXG4gIHB1YmxpYyBzZXRPdXRwdXRNYXRyaXhUZXh0dXJlKFxuICAgICAgb3V0cHV0TWF0cml4VGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcikge1xuICAgIHRoaXMuc2V0T3V0cHV0TWF0cml4VGV4dHVyZURyaXZlcihvdXRwdXRNYXRyaXhUZXh0dXJlLCBjb2x1bW5zLCByb3dzKTtcbiAgfVxuXG4gIHB1YmxpYyBzZXRPdXRwdXRQYWNrZWRNYXRyaXhUZXh0dXJlKFxuICAgICAgb3V0cHV0UGFja2VkTWF0cml4VGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcikge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgW3dpZHRoLCBoZWlnaHRdID1cbiAgICAgICAgdGV4X3V0aWwuZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gICAgdGhpcy5zZXRPdXRwdXRNYXRyaXhUZXh0dXJlRHJpdmVyKG91dHB1dFBhY2tlZE1hdHJpeFRleHR1cmUsIHdpZHRoLCBoZWlnaHQpO1xuICB9XG5cbiAgcHVibGljIHNldE91dHB1dE1hdHJpeFdyaXRlUmVnaW9uKFxuICAgICAgc3RhcnRSb3c6IG51bWJlciwgbnVtUm93czogbnVtYmVyLCBzdGFydENvbHVtbjogbnVtYmVyLFxuICAgICAgbnVtQ29sdW1uczogbnVtYmVyKSB7XG4gICAgdGhpcy5zZXRPdXRwdXRNYXRyaXhXcml0ZVJlZ2lvbkRyaXZlcihcbiAgICAgICAgc3RhcnRDb2x1bW4sIHN0YXJ0Um93LCBudW1Db2x1bW5zLCBudW1Sb3dzKTtcbiAgfVxuXG4gIHB1YmxpYyBzZXRPdXRwdXRQYWNrZWRNYXRyaXhXcml0ZVJlZ2lvbihcbiAgICAgIHN0YXJ0Um93OiBudW1iZXIsIG51bVJvd3M6IG51bWJlciwgc3RhcnRDb2x1bW46IG51bWJlcixcbiAgICAgIG51bUNvbHVtbnM6IG51bWJlcikge1xuICAgIHRocm93IG5ldyBFcnJvcignc2V0T3V0cHV0UGFja2VkTWF0cml4V3JpdGVSZWdpb24gbm90IGltcGxlbWVudGVkLicpO1xuICB9XG5cbiAgcHVibGljIGRlYnVnVmFsaWRhdGUoKSB7XG4gICAgaWYgKHRoaXMucHJvZ3JhbSAhPSBudWxsKSB7XG4gICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlUHJvZ3JhbSh0aGlzLmdsLCB0aGlzLnByb2dyYW0pO1xuICAgIH1cbiAgICB3ZWJnbF91dGlsLnZhbGlkYXRlRnJhbWVidWZmZXIodGhpcy5nbCk7XG4gIH1cblxuICBwdWJsaWMgZXhlY3V0ZVByb2dyYW0oKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB0aGlzLnRocm93SWZOb1Byb2dyYW0oKTtcbiAgICBjb25zdCBnbCA9IHRoaXMuZ2w7XG4gICAgZ3BncHVfdXRpbC5iaW5kVmVydGV4UHJvZ3JhbUF0dHJpYnV0ZVN0cmVhbXMoXG4gICAgICAgIGdsLCB0aGlzLnByb2dyYW0hLCB0aGlzLnZlcnRleEJ1ZmZlcik7XG4gICAgaWYgKHRoaXMuYXV0b0RlYnVnVmFsaWRhdGUpIHtcbiAgICAgIHRoaXMuZGVidWdWYWxpZGF0ZSgpO1xuICAgIH1cbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgICAgZ2wsICgpID0+IGdsLmRyYXdFbGVtZW50cyhnbC5UUklBTkdMRVMsIDYsIGdsLlVOU0lHTkVEX1NIT1JULCAwKSk7XG4gIH1cblxuICBwdWJsaWMgYmxvY2tVbnRpbEFsbFByb2dyYW1zQ29tcGxldGVkKCkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2sodGhpcy5nbCwgKCkgPT4gdGhpcy5nbC5maW5pc2goKSk7XG4gIH1cblxuICBwcml2YXRlIGRvd25sb2FkTWF0cml4RHJpdmVyKFxuICAgICAgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgICAgZG93bmxvYWRBbmREZWNvZGU6ICgpID0+IEZsb2F0MzJBcnJheSk6IEZsb2F0MzJBcnJheSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB3ZWJnbF91dGlsLmJpbmRDb2xvclRleHR1cmVUb0ZyYW1lYnVmZmVyKFxuICAgICAgICB0aGlzLmdsLCB0ZXh0dXJlLCB0aGlzLmZyYW1lYnVmZmVyKTtcbiAgICBjb25zdCByZXN1bHQgPSBkb3dubG9hZEFuZERlY29kZSgpO1xuICAgIGlmICh0aGlzLm91dHB1dFRleHR1cmUgIT0gbnVsbCkge1xuICAgICAgd2ViZ2xfdXRpbC5iaW5kQ29sb3JUZXh0dXJlVG9GcmFtZWJ1ZmZlcihcbiAgICAgICAgICB0aGlzLmdsLCB0aGlzLm91dHB1dFRleHR1cmUsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgICAgaWYgKHRoaXMuYXV0b0RlYnVnVmFsaWRhdGUpIHtcbiAgICAgICAgd2ViZ2xfdXRpbC52YWxpZGF0ZUZyYW1lYnVmZmVyKHRoaXMuZ2wpO1xuICAgICAgfVxuICAgIH0gZWxzZSB7XG4gICAgICB3ZWJnbF91dGlsLnVuYmluZENvbG9yVGV4dHVyZUZyb21GcmFtZWJ1ZmZlcih0aGlzLmdsLCB0aGlzLmZyYW1lYnVmZmVyKTtcbiAgICB9XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHByaXZhdGUgc2V0T3V0cHV0TWF0cml4VGV4dHVyZURyaXZlcihcbiAgICAgIG91dHB1dE1hdHJpeFRleHR1cmVNYXliZVBhY2tlZDogV2ViR0xUZXh0dXJlLCB3aWR0aDogbnVtYmVyLFxuICAgICAgaGVpZ2h0OiBudW1iZXIpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGNvbnN0IGdsID0gdGhpcy5nbDtcbiAgICB3ZWJnbF91dGlsLmJpbmRDb2xvclRleHR1cmVUb0ZyYW1lYnVmZmVyKFxuICAgICAgICBnbCwgb3V0cHV0TWF0cml4VGV4dHVyZU1heWJlUGFja2VkLCB0aGlzLmZyYW1lYnVmZmVyKTtcbiAgICBpZiAodGhpcy5hdXRvRGVidWdWYWxpZGF0ZSkge1xuICAgICAgd2ViZ2xfdXRpbC52YWxpZGF0ZUZyYW1lYnVmZmVyKGdsKTtcbiAgICB9XG4gICAgdGhpcy5vdXRwdXRUZXh0dXJlID0gb3V0cHV0TWF0cml4VGV4dHVyZU1heWJlUGFja2VkO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC52aWV3cG9ydCgwLCAwLCB3aWR0aCwgaGVpZ2h0KSk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnNjaXNzb3IoMCwgMCwgd2lkdGgsIGhlaWdodCkpO1xuICB9XG5cbiAgcHJpdmF0ZSBzZXRPdXRwdXRNYXRyaXhXcml0ZVJlZ2lvbkRyaXZlcihcbiAgICAgIHg6IG51bWJlciwgeTogbnVtYmVyLCB3aWR0aDogbnVtYmVyLCBoZWlnaHQ6IG51bWJlcikge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICAgIHRoaXMuZ2wsICgpID0+IHRoaXMuZ2wuc2Npc3Nvcih4LCB5LCB3aWR0aCwgaGVpZ2h0KSk7XG4gIH1cblxuICBwcml2YXRlIHRocm93SWZEaXNwb3NlZCgpIHtcbiAgICBpZiAodGhpcy5kaXNwb3NlZCkge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdBdHRlbXB0ZWQgdG8gdXNlIGRpc3Bvc2VkIEdQR1BVQ29udGV4dC4nKTtcbiAgICB9XG4gIH1cblxuICBwcml2YXRlIHRocm93SWZOb1Byb2dyYW0oKSB7XG4gICAgaWYgKHRoaXMucHJvZ3JhbSA9PSBudWxsKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ05vIEdQVSBwcm9ncmFtIGlzIGN1cnJlbnRseSBzZXQuJyk7XG4gICAgfVxuICB9XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIHRleF91dGlsIGZyb20gJy4vdGV4X3V0aWwnO1xuaW1wb3J0ICogYXMgd2ViZ2xfdXRpbCBmcm9tICcuL3dlYmdsX3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0V2ViR0xDb250ZXh0QXR0cmlidXRlcygpOiBXZWJHTENvbnRleHRBdHRyaWJ1dGVzIHtcbiAgcmV0dXJuIHtcbiAgICBhbHBoYTogZmFsc2UsXG4gICAgYW50aWFsaWFzOiBmYWxzZSxcbiAgICBwcmVtdWx0aXBsaWVkQWxwaGE6IGZhbHNlLFxuICAgIHByZXNlcnZlRHJhd2luZ0J1ZmZlcjogZmFsc2UsXG4gICAgZGVwdGg6IGZhbHNlLFxuICAgIHN0ZW5jaWw6IGZhbHNlLFxuICAgIGZhaWxJZk1ham9yUGVyZm9ybWFuY2VDYXZlYXQ6IHRydWVcbiAgfTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVdlYkdMQ29udGV4dChjYW52YXM/OiBIVE1MQ2FudmFzRWxlbWVudCkge1xuICBjb25zdCBhdHRyaWJ1dGVzID0gZ2V0V2ViR0xDb250ZXh0QXR0cmlidXRlcygpO1xuICBsZXQgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dDtcbiAgaWYgKGNhbnZhcyAhPSBudWxsKSB7XG4gICAgZ2wgPSB3ZWJnbF91dGlsLmNyZWF0ZVdlYkdMUmVuZGVyaW5nQ29udGV4dEZyb21DYW52YXMoY2FudmFzLCBhdHRyaWJ1dGVzKTtcbiAgfSBlbHNlIHtcbiAgICBnbCA9IHdlYmdsX3V0aWwuY3JlYXRlV2ViR0xSZW5kZXJpbmdDb250ZXh0KGF0dHJpYnV0ZXMpO1xuICB9XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLkRFUFRIX1RFU1QpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRpc2FibGUoZ2wuU1RFTkNJTF9URVNUKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLkJMRU5EKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLkRJVEhFUikpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGlzYWJsZShnbC5QT0xZR09OX09GRlNFVF9GSUxMKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLlNBTVBMRV9DT1ZFUkFHRSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZW5hYmxlKGdsLlNDSVNTT1JfVEVTVCkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZW5hYmxlKGdsLkNVTExfRkFDRSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuY3VsbEZhY2UoZ2wuQkFDSykpO1xuICByZXR1cm4gZ2w7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVWZXJ0ZXhTaGFkZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMU2hhZGVyIHtcbiAgY29uc3QgdmVydGV4U2hhZGVyU291cmNlID0gYFxuICAgIHByZWNpc2lvbiBoaWdocCBmbG9hdDtcbiAgICBhdHRyaWJ1dGUgdmVjMyBjbGlwU3BhY2VQb3M7XG4gICAgYXR0cmlidXRlIHZlYzIgdXY7XG4gICAgdmFyeWluZyB2ZWMyIHJlc3VsdFVWO1xuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgZ2xfUG9zaXRpb24gPSB2ZWM0KGNsaXBTcGFjZVBvcywgMSk7XG4gICAgICByZXN1bHRVViA9IHV2O1xuICAgIH1gO1xuICByZXR1cm4gd2ViZ2xfdXRpbC5jcmVhdGVWZXJ0ZXhTaGFkZXIoZ2wsIHZlcnRleFNoYWRlclNvdXJjZSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVWZXJ0ZXhCdWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMQnVmZmVyIHtcbiAgLy8gW3ggeSB6IHUgdl0gKiBbdXBwZXItbGVmdCwgbG93ZXItbGVmdCwgdXBwZXItcmlnaHQsIGxvd2VyLXJpZ2h0XVxuICBjb25zdCB2ZXJ0ZXhBcnJheSA9IG5ldyBGbG9hdDMyQXJyYXkoXG4gICAgICBbLTEsIDEsIDAsIDAsIDEsIC0xLCAtMSwgMCwgMCwgMCwgMSwgMSwgMCwgMSwgMSwgMSwgLTEsIDAsIDEsIDBdKTtcbiAgcmV0dXJuIHdlYmdsX3V0aWwuY3JlYXRlU3RhdGljVmVydGV4QnVmZmVyKGdsLCB2ZXJ0ZXhBcnJheSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVJbmRleEJ1ZmZlcihnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogV2ViR0xCdWZmZXIge1xuICAvLyBPcGVuR0wgKGFuZCBXZWJHTCkgaGF2ZSBcIkNDVyA9PSBmcm9udFwiIHdpbmRpbmdcbiAgY29uc3QgdHJpYW5nbGVWZXJ0ZXhJbmRpY2VzID0gbmV3IFVpbnQxNkFycmF5KFswLCAxLCAyLCAyLCAxLCAzXSk7XG4gIHJldHVybiB3ZWJnbF91dGlsLmNyZWF0ZVN0YXRpY0luZGV4QnVmZmVyKGdsLCB0cmlhbmdsZVZlcnRleEluZGljZXMpO1xufVxuXG5mdW5jdGlvbiBnZXRUZXh0dXJlSW50ZXJuYWxGb3JtYXQoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgbnVtQ2hhbm5lbHM6IG51bWJlcik6IG51bWJlciB7XG4gIGlmICh3ZWJnbF91dGlsLmlzV2ViR0wyRW5hYmxlZCgpKSB7XG4gICAgaWYgKG51bUNoYW5uZWxzID09PSA0KSB7XG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICByZXR1cm4gKGdsIGFzIGFueSkuUkdCQTMyRjtcbiAgICB9XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIHJldHVybiAoZ2wgYXMgYW55KS5SMzJGO1xuICB9XG4gIHJldHVybiBnbC5SR0JBO1xufVxuXG5mdW5jdGlvbiBnZXRUZXh0dXJlRm9ybWF0KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIG51bUNoYW5uZWxzOiBudW1iZXIpOiBudW1iZXIge1xuICBpZiAod2ViZ2xfdXRpbC5pc1dlYkdMMkVuYWJsZWQoKSAmJiBudW1DaGFubmVscyA9PT0gMSkge1xuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICByZXR1cm4gKGdsIGFzIGFueSkuUkVEO1xuICB9XG4gIHJldHVybiBnbC5SR0JBO1xufVxuXG5mdW5jdGlvbiBjcmVhdGVBbmRDb25maWd1cmVUZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLFxuICAgIG51bUNoYW5uZWxzOiBudW1iZXIpOiBXZWJHTFRleHR1cmUge1xuICB3ZWJnbF91dGlsLnZhbGlkYXRlVGV4dHVyZVNpemUoZ2wsIHdpZHRoLCBoZWlnaHQpO1xuICBjb25zdCB0ZXh0dXJlID0gd2ViZ2xfdXRpbC5jcmVhdGVUZXh0dXJlKGdsKTtcblxuICBjb25zdCB0ZXgyZCA9IGdsLlRFWFRVUkVfMkQ7XG4gIGNvbnN0IGludGVybmFsRm9ybWF0ID0gZ2V0VGV4dHVyZUludGVybmFsRm9ybWF0KGdsLCBudW1DaGFubmVscyk7XG4gIGNvbnN0IGZvcm1hdCA9IGdldFRleHR1cmVGb3JtYXQoZ2wsIG51bUNoYW5uZWxzKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRUZXh0dXJlKHRleDJkLCB0ZXh0dXJlKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnRleFBhcmFtZXRlcmkodGV4MmQsIGdsLlRFWFRVUkVfV1JBUF9TLCBnbC5DTEFNUF9UT19FREdFKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnRleFBhcmFtZXRlcmkodGV4MmQsIGdsLlRFWFRVUkVfV1JBUF9ULCBnbC5DTEFNUF9UT19FREdFKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnRleFBhcmFtZXRlcmkodGV4MmQsIGdsLlRFWFRVUkVfTUlOX0ZJTFRFUiwgZ2wuTkVBUkVTVCkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC50ZXhQYXJhbWV0ZXJpKHRleDJkLCBnbC5URVhUVVJFX01BR19GSUxURVIsIGdsLk5FQVJFU1QpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICBnbCxcbiAgICAgICgpID0+IGdsLnRleEltYWdlMkQoXG4gICAgICAgICAgdGV4MmQsIDAsIGludGVybmFsRm9ybWF0LCB3aWR0aCwgaGVpZ2h0LCAwLCBmb3JtYXQsIGdsLkZMT0FULCBudWxsKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKSk7XG4gIHJldHVybiB0ZXh0dXJlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlTWF0cml4VGV4dHVyZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFdlYkdMVGV4dHVyZSB7XG4gIGNvbnN0IFt3aWR0aCwgaGVpZ2h0XSA9XG4gICAgICB0ZXhfdXRpbC5nZXRVbnBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KHJvd3MsIGNvbHVtbnMpO1xuICBjb25zdCBudW1DaGFubmVscyA9IDE7XG4gIHJldHVybiBjcmVhdGVBbmRDb25maWd1cmVUZXh0dXJlKGdsLCB3aWR0aCwgaGVpZ2h0LCBudW1DaGFubmVscyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVDb2xvck1hdHJpeFRleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBXZWJHTFRleHR1cmUge1xuICBjb25zdCBbd2lkdGgsIGhlaWdodF0gPVxuICAgICAgdGV4X3V0aWwuZ2V0Q29sb3JNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcbiAgY29uc3QgbnVtQ2hhbm5lbHMgPSA0O1xuICByZXR1cm4gY3JlYXRlQW5kQ29uZmlndXJlVGV4dHVyZShnbCwgd2lkdGgsIGhlaWdodCwgbnVtQ2hhbm5lbHMpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFdlYkdMVGV4dHVyZSB7XG4gIGNvbnN0IFt3aWR0aCwgaGVpZ2h0XSA9XG4gICAgICB0ZXhfdXRpbC5nZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcbiAgY29uc3QgbnVtQ2hhbm5lbHMgPSA0O1xuICByZXR1cm4gY3JlYXRlQW5kQ29uZmlndXJlVGV4dHVyZShnbCwgd2lkdGgsIGhlaWdodCwgbnVtQ2hhbm5lbHMpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZFZlcnRleFByb2dyYW1BdHRyaWJ1dGVTdHJlYW1zKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSxcbiAgICB2ZXJ0ZXhCdWZmZXI6IFdlYkdMQnVmZmVyKSB7XG4gIGNvbnN0IHBvc09mZnNldCA9IDA7ICAgICAgICAgICAgICAgLy8geCBpcyB0aGUgZmlyc3QgYnVmZmVyIGVsZW1lbnRcbiAgY29uc3QgdXZPZmZzZXQgPSAzICogNDsgICAgICAgICAgICAvLyB1diBjb21lcyBhZnRlciBbeCB5IHpdXG4gIGNvbnN0IHN0cmlkZSA9ICgzICogNCkgKyAoMiAqIDQpOyAgLy8geHl6ICsgdXYsIGVhY2ggZW50cnkgaXMgNC1ieXRlIGZsb2F0LlxuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC5iaW5kQnVmZmVyKGdsLkFSUkFZX0JVRkZFUiwgdmVydGV4QnVmZmVyKSk7XG4gIHdlYmdsX3V0aWwuYmluZFZlcnRleEJ1ZmZlclRvUHJvZ3JhbUF0dHJpYnV0ZShcbiAgICAgIGdsLCBwcm9ncmFtLCAnY2xpcFNwYWNlUG9zJywgdmVydGV4QnVmZmVyLCAzLCBzdHJpZGUsIHBvc09mZnNldCk7XG4gIHRyeSB7XG4gICAgd2ViZ2xfdXRpbC5iaW5kVmVydGV4QnVmZmVyVG9Qcm9ncmFtQXR0cmlidXRlKFxuICAgICAgICBnbCwgcHJvZ3JhbSwgJ3V2JywgdmVydGV4QnVmZmVyLCAyLCBzdHJpZGUsIHV2T2Zmc2V0KTtcbiAgfSBjYXRjaCAoZSkge1xuICAgIC8vIFByb2dyYW1zIHdpdGggMXgxIG91dHB1dCB0ZXh0dXJlcyBkb24ndCB1c2UgdGhlIHV2IGF0dHJpYnV0ZS5cbiAgICAvLyBUaGlzIGNhbiBjYXVzZSB0aGUgc2hhZGVyIGxpbmtlciB0byBkZWFkLXN0cmlwIGl0LCBzbyB3ZSBzaG91bGRuJ3RcbiAgICAvLyBjb21wbGFpbiBvciBmYWlsIGlmIGl0J3Mgbm90IHByZXNlbnQuXG4gICAgaWYgKCFlLmhhc093blByb3BlcnR5KCduYW1lZFZlcnRleEF0dHJpYnV0ZU5vdEZvdW5kJykpIHtcbiAgICAgIHRocm93IGU7XG4gICAgfVxuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1cGxvYWRQaXhlbERhdGFUb1RleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgIHBpeGVsczogSW1hZ2VEYXRhfEhUTUxJbWFnZUVsZW1lbnR8SFRNTENhbnZhc0VsZW1lbnR8SFRNTFZpZGVvRWxlbWVudCkge1xuICBjb25zdCBudW1DaGFubmVscyA9IDQ7XG4gIGNvbnN0IGludGVybmFsRm9ybWF0ID0gZ2V0VGV4dHVyZUludGVybmFsRm9ybWF0KGdsLCBudW1DaGFubmVscyk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0ZXh0dXJlKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsXG4gICAgICAoKSA9PiBnbC50ZXhJbWFnZTJEKFxuICAgICAgICAgIGdsLlRFWFRVUkVfMkQsIDAsIGludGVybmFsRm9ybWF0LCBnbC5SR0JBLCBnbC5GTE9BVCwgcGl4ZWxzKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKSk7XG59XG5cbmZ1bmN0aW9uIHVwbG9hZERhdGFUb1RleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCB3aWR0aDogbnVtYmVyLFxuICAgIGhlaWdodDogbnVtYmVyLCBkYXRhOiBGbG9hdDMyQXJyYXksIG51bUNoYW5uZWxzOiBudW1iZXIpIHtcbiAgY29uc3QgdGV4dHVyZUZvcm1hdCA9IGdldFRleHR1cmVGb3JtYXQoZ2wsIG51bUNoYW5uZWxzKTtcblxuICB3ZWJnbF91dGlsLnZhbGlkYXRlVGV4dHVyZVNpemUoZ2wsIHdpZHRoLCBoZWlnaHQpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLFxuICAgICAgKCkgPT4gZ2wudGV4U3ViSW1hZ2UyRChcbiAgICAgICAgICBnbC5URVhUVVJFXzJELCAwLCAwLCAwLCB3aWR0aCwgaGVpZ2h0LCB0ZXh0dXJlRm9ybWF0LCBnbC5GTE9BVCxcbiAgICAgICAgICBkYXRhKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsXG4gICAgY29sdW1uczogbnVtYmVyLCBtYXRyaXg6IEZsb2F0MzJBcnJheSwgbnVtQ2hhbm5lbHM6IG51bWJlcikge1xuICBjb25zdCBbdywgaF0gPVxuICAgICAgdGV4X3V0aWwuZ2V0VW5wYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcblxuICBjb25zdCBjaGFubmVsc1BlclRleHR1cmUgPVxuICAgICAgbnVtQ2hhbm5lbHMgPT09IDEgPyB3ZWJnbF91dGlsLmdldENoYW5uZWxzUGVyVGV4dHVyZSgpIDogbnVtQ2hhbm5lbHM7XG4gIGNvbnN0IHVucGFja2VkQXJyYXkgPVxuICAgICAgbmV3IEZsb2F0MzJBcnJheSh0ZXhfdXRpbC5nZXRVbnBhY2tlZEFycmF5U2l6ZUZyb21NYXRyaXhTaXplKFxuICAgICAgICAgIG1hdHJpeC5sZW5ndGgsIGNoYW5uZWxzUGVyVGV4dHVyZSkpO1xuICB0ZXhfdXRpbC5lbmNvZGVNYXRyaXhUb1VucGFja2VkQXJyYXkoXG4gICAgICBtYXRyaXgsIHVucGFja2VkQXJyYXksIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG5cbiAgdXBsb2FkRGF0YVRvVGV4dHVyZShnbCwgdGV4dHVyZSwgdywgaCwgdW5wYWNrZWRBcnJheSwgbnVtQ2hhbm5lbHMpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdXBsb2FkTWF0cml4VG9QYWNrZWRUZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLFxuICAgIGNvbHVtbnM6IG51bWJlciwgbWF0cml4OiBGbG9hdDMyQXJyYXkpIHtcbiAgY29uc3QgW3csIGhdID0gdGV4X3V0aWwuZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gIGNvbnN0IHBhY2tlZFJHQkEgPSBuZXcgRmxvYXQzMkFycmF5KFxuICAgICAgdGV4X3V0aWwuZ2V0UGFja2VkUkdCQUFycmF5U2l6ZUZyb21NYXRyaXhTaGFwZShyb3dzLCBjb2x1bW5zKSk7XG4gIHRleF91dGlsLmVuY29kZU1hdHJpeFRvUGFja2VkUkdCQShtYXRyaXgsIHJvd3MsIGNvbHVtbnMsIHBhY2tlZFJHQkEpO1xuICBjb25zdCBudW1DaGFubmVscyA9IDQ7XG4gIHVwbG9hZERhdGFUb1RleHR1cmUoZ2wsIHRleHR1cmUsIHcsIGgsIHBhY2tlZFJHQkEsIG51bUNoYW5uZWxzKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRvd25sb2FkTWF0cml4RnJvbU91dHB1dFRleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBGbG9hdDMyQXJyYXkge1xuICBjb25zdCBbdywgaF0gPVxuICAgICAgdGV4X3V0aWwuZ2V0VW5wYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcblxuICBjb25zdCBjaGFubmVsc1BlclRleHR1cmUgPSA0O1xuICBjb25zdCB1bnBhY2tlZEFycmF5ID1cbiAgICAgIG5ldyBGbG9hdDMyQXJyYXkodGV4X3V0aWwuZ2V0VW5wYWNrZWRBcnJheVNpemVGcm9tTWF0cml4U2l6ZShcbiAgICAgICAgICByb3dzICogY29sdW1ucywgY2hhbm5lbHNQZXJUZXh0dXJlKSk7XG4gIGNvbnN0IHRleHR1cmVGb3JtYXQgPSBnZXRUZXh0dXJlRm9ybWF0KGdsLCBjaGFubmVsc1BlclRleHR1cmUpO1xuXG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnJlYWRQaXhlbHMoMCwgMCwgdywgaCwgZ2wuUkdCQSwgZ2wuRkxPQVQsIHVucGFja2VkQXJyYXkpKTtcblxuICBjb25zdCBtYXRyaXggPSBuZXcgRmxvYXQzMkFycmF5KHJvd3MgKiBjb2x1bW5zKTtcbiAgdGV4X3V0aWwuZGVjb2RlTWF0cml4RnJvbVVucGFja2VkQXJyYXkoXG4gICAgICB1bnBhY2tlZEFycmF5LCBtYXRyaXgsIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG4gIHJldHVybiBtYXRyaXg7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBkb3dubG9hZE1hdHJpeEZyb21QYWNrZWRPdXRwdXRUZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogRmxvYXQzMkFycmF5IHtcbiAgY29uc3QgW3csIGhdID0gdGV4X3V0aWwuZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gIGNvbnN0IHBhY2tlZFJHQkEgPSBuZXcgRmxvYXQzMkFycmF5KFxuICAgICAgdGV4X3V0aWwuZ2V0UGFja2VkUkdCQUFycmF5U2l6ZUZyb21NYXRyaXhTaGFwZShyb3dzLCBjb2x1bW5zKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnJlYWRQaXhlbHMoMCwgMCwgdywgaCwgZ2wuUkdCQSwgZ2wuRkxPQVQsIHBhY2tlZFJHQkEpKTtcbiAgY29uc3QgbWF0cml4ID0gbmV3IEZsb2F0MzJBcnJheShyb3dzICogY29sdW1ucyk7XG4gIHJldHVybiB0ZXhfdXRpbC5kZWNvZGVNYXRyaXhGcm9tUGFja2VkUkdCQShwYWNrZWRSR0JBLCByb3dzLCBjb2x1bW5zLCBtYXRyaXgpO1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge0dQR1BVQ29udGV4dH0gZnJvbSAnLi9ncGdwdV9jb250ZXh0JztcblxuZXhwb3J0IGZ1bmN0aW9uIGdldEZyYWdtZW50U2hhZGVyU291cmNlKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogc3RyaW5nIHtcbiAgcmV0dXJuIGBcbiAgICBwcmVjaXNpb24gaGlnaHAgZmxvYXQ7XG4gICAgdW5pZm9ybSBzYW1wbGVyMkQgbWF0cml4QTtcbiAgICB2YXJ5aW5nIHZlYzIgcmVzdWx0VVY7XG5cbiAgICBjb25zdCB2ZWMyIGFEaW1DUiA9IHZlYzIoJHtjb2x1bW5zfS4wLCAke3Jvd3N9LjApO1xuICAgIGNvbnN0IHZlYzIgaGFsZkNSID0gdmVjMigwLjUsIDAuNSk7XG5cbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICBmbG9hdCBhTWF4ID0gdGV4dHVyZTJEKG1hdHJpeEEsIGhhbGZDUiAvIGFEaW1DUikucjtcbiAgICAgIGZvciAoZmxvYXQgciA9IDAuMDsgciA8IGFEaW1DUi55OyByICs9IDEuMCkge1xuICAgICAgICBmb3IgKGZsb2F0IGMgPSAwLjA7IGMgPCBhRGltQ1IueDsgYyArPSAxLjApIHtcbiAgICAgICAgICB2ZWMyIHV2ID0gKHZlYzIoYywgcikgKyBoYWxmQ1IpIC8gYURpbUNSO1xuICAgICAgICAgIGZsb2F0IGFDdXIgPSB0ZXh0dXJlMkQobWF0cml4QSwgdXYpLnI7XG4gICAgICAgICAgYU1heCA9IG1heChhTWF4LCBhQ3VyKTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICBmbG9hdCBleHBTdW0gPSAwLjA7XG4gICAgICBmb3IgKGZsb2F0IHIgPSAwLjA7IHIgPCBhRGltQ1IueTsgciArPSAxLjApIHtcbiAgICAgICAgZm9yIChmbG9hdCBjID0gMC4wOyBjIDwgYURpbUNSLng7IGMgKz0gMS4wKSB7XG4gICAgICAgICAgdmVjMiB1diA9ICh2ZWMyKGMsIHIpICsgaGFsZkNSKSAvIGFEaW1DUjtcbiAgICAgICAgICBmbG9hdCBhQ3VyID0gdGV4dHVyZTJEKG1hdHJpeEEsIHV2KS5yO1xuICAgICAgICAgIGV4cFN1bSArPSBleHAoYUN1ciAtIGFNYXgpO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGdsX0ZyYWdDb2xvciA9IHZlYzQoYU1heCArIGxvZyhleHBTdW0pLCAwLCAwLCAwKTtcbiAgICB9YDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGxvZ1N1bUV4cChcbiAgICBncGdwdTogR1BHUFVDb250ZXh0LCBsb2dTdW1FeHBQcm9ncmFtOiBXZWJHTFByb2dyYW0sIGE6IFdlYkdMVGV4dHVyZSxcbiAgICByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlciwgcmVzdWx0OiBXZWJHTFRleHR1cmUpIHtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShyZXN1bHQsIDEsIDEpO1xuICBncGdwdS5zZXRQcm9ncmFtKGxvZ1N1bUV4cFByb2dyYW0pO1xuICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUoYSwgJ21hdHJpeEEnLCAwKTtcbiAgZ3BncHUuZXhlY3V0ZVByb2dyYW0oKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHVwbG9hZExvZ1N1bUV4cERvd25sb2FkKFxuICAgIGE6IEZsb2F0MzJBcnJheSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBudW1iZXIge1xuICBjb25zdCBncGdwdSA9IG5ldyBHUEdQVUNvbnRleHQoKTtcbiAgY29uc3QgcHJvZ3JhbSA9IGdwZ3B1LmNyZWF0ZVByb2dyYW0oZ2V0RnJhZ21lbnRTaGFkZXJTb3VyY2Uocm93cywgY29sdW1ucykpO1xuICBjb25zdCBhVGV4dHVyZSA9IGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUocm93cywgY29sdW1ucyk7XG4gIGNvbnN0IHJlc3VsdFRleHR1cmUgPSBncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKDEsIDEpO1xuICBncGdwdS51cGxvYWRNYXRyaXhUb1RleHR1cmUoYVRleHR1cmUsIHJvd3MsIGNvbHVtbnMsIGEpO1xuICBsb2dTdW1FeHAoZ3BncHUsIHByb2dyYW0sIGFUZXh0dXJlLCByb3dzLCBjb2x1bW5zLCByZXN1bHRUZXh0dXJlKTtcbiAgY29uc3QgcmVzdWx0ID0gZ3BncHUuZG93bmxvYWRNYXRyaXhGcm9tVGV4dHVyZShyZXN1bHRUZXh0dXJlLCAxLCAxKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShhVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUocmVzdWx0VGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZVByb2dyYW0ocHJvZ3JhbSk7XG4gIGdwZ3B1LmRpc3Bvc2UoKTtcbiAgcmV0dXJuIHJlc3VsdFswXTtcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vZ3BncHVfY29udGV4dCc7XG5pbXBvcnQgKiBhcyBwb29sX2dwdSBmcm9tICcuL3Bvb2xfZ3B1JztcblxuZXhwb3J0IGZ1bmN0aW9uIGdldEZyYWdtZW50U2hhZGVyTWF4UG9vbFBvc2l0aW9uc1NvdXJjZShcbiAgICB4U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsXG4gICAgcGFkOiBudW1iZXIpIHtcbiAgcmV0dXJuIGdldEZyYWdtZW50U2hhZGVyTWF4UG9vbENvbW1vblNvdXJjZShcbiAgICAgIHhTaGFwZVJDRCwgZlNpemUsIHN0cmlkZSwgcGFkLCB0cnVlKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldEZyYWdtZW50U2hhZGVyTWF4UG9vbFNvdXJjZShcbiAgICB4U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsXG4gICAgcGFkOiBudW1iZXIpIHtcbiAgcmV0dXJuIGdldEZyYWdtZW50U2hhZGVyTWF4UG9vbENvbW1vblNvdXJjZShcbiAgICAgIHhTaGFwZVJDRCwgZlNpemUsIHN0cmlkZSwgcGFkLCBmYWxzZSk7XG59XG5cbmZ1bmN0aW9uIGdldEZyYWdtZW50U2hhZGVyTWF4UG9vbENvbW1vblNvdXJjZShcbiAgICB4U2hhcGVSQ0Q6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsXG4gICAgcGFkOiBudW1iZXIsIGNvbXB1dGVNYXhQb3NpdGlvbnM6IGJvb2xlYW4pIHtcbiAgcmV0dXJuIHBvb2xfZ3B1LmdldEZyYWdtZW50U2hhZGVyUG9vbENvbW1vblNvdXJjZShcbiAgICAgIHhTaGFwZVJDRCwgZlNpemUsIHN0cmlkZSwgcGFkLCAnbWF4JywgY29tcHV0ZU1heFBvc2l0aW9ucyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtYXhQb29sQ29tbW9uKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgeDogV2ViR0xUZXh0dXJlLFxuICAgIHJlc3VsdDogV2ViR0xUZXh0dXJlLCByZXN1bHRTaGFwZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSkge1xuICBwb29sX2dwdS5wb29sQ29tbW9uKGdwZ3B1LCBwcm9ncmFtLCB4LCByZXN1bHQsIHJlc3VsdFNoYXBlUm93Q29sKTtcbn0iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7TWF0cml4T3JpZW50YXRpb259IGZyb20gJy4uL21hdGgnO1xuaW1wb3J0IHtBcnJheTJEfSBmcm9tICcuLi9uZGFycmF5JztcblxuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vZ3BncHVfY29udGV4dCc7XG5pbXBvcnQgKiBhcyBzaGFkZXJfY29tcGlsZXIgZnJvbSAnLi9zaGFkZXJfY29tcGlsZXInO1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhZ21lbnRTaGFkZXIoXG4gICAgYTogQXJyYXkyRCwgYjogQXJyYXkyRCwgb3V0OiBBcnJheTJELCBhT3JpZW50YXRpb246IE1hdHJpeE9yaWVudGF0aW9uLFxuICAgIGJPcmllbnRhdGlvbjogTWF0cml4T3JpZW50YXRpb24pOiBzdHJpbmcge1xuICBjb25zdCBzaGFyZWREaW0gPVxuICAgICAgKGFPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUiA/IGEuc2hhcGVbMV0gOiBhLnNoYXBlWzBdKTtcbiAgY29uc3QgYVNuaXBwZXQgPVxuICAgICAgKGFPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyAnYVJvdywgaScgOiAnaSwgYVJvdyc7XG4gIGNvbnN0IGJTbmlwcGV0ID1cbiAgICAgIChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID8gJ2ksIGJDb2wnIDogJ2JDb2wsIGknO1xuXG4gIGNvbnN0IGlucHV0cyA9IFt7bmFtZTogJ21hdHJpeEEnLCBhcnJheTogYX0sIHtuYW1lOiAnbWF0cml4QicsIGFycmF5OiBifV07XG4gIGNvbnN0IHVzZXJDb2RlID0gYFxuICAgIGNvbnN0IGZsb2F0IHNoYXJlZERpbSA9ICR7c2hhcmVkRGltfS4wO1xuXG4gICAgZmxvYXQgZG90QVJvd0JDb2woZmxvYXQgYVJvdywgZmxvYXQgYkNvbCkge1xuICAgICAgZmxvYXQgcmVzdWx0ID0gMC4wO1xuICAgICAgZm9yIChmbG9hdCBpID0gMC4wOyBpIDwgc2hhcmVkRGltOyBpICs9IDEuMCkge1xuICAgICAgICBmbG9hdCBhID0gZ2V0TWF0cml4QSgke2FTbmlwcGV0fSk7XG4gICAgICAgIGZsb2F0IGIgPSBnZXRNYXRyaXhCKCR7YlNuaXBwZXR9KTtcbiAgICAgICAgcmVzdWx0ICs9IChhICogYik7XG4gICAgICB9XG4gICAgICByZXR1cm4gcmVzdWx0O1xuICAgIH1cblxuICAgIHZvaWQgbWFpbigpIHtcbiAgICAgIHZlYzIgcmVzUkMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgIHNldE91dHB1dChkb3RBUm93QkNvbChyZXNSQy54LCByZXNSQy55KSk7XG4gICAgfVxuICBgO1xuICByZXR1cm4gc2hhZGVyX2NvbXBpbGVyLm1ha2VTaGFkZXIoaW5wdXRzLCBvdXQsIHVzZXJDb2RlKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIG11bHRpcGx5TWF0cml4KFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIG11bHRpcGx5UHJvZ3JhbTogV2ViR0xQcm9ncmFtLCBhOiBXZWJHTFRleHR1cmUsXG4gICAgYjogV2ViR0xUZXh0dXJlLCByZXN1bHQ6IFdlYkdMVGV4dHVyZSwgb3V0VGV4U2hhcGU6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShyZXN1bHQsIG91dFRleFNoYXBlWzBdLCBvdXRUZXhTaGFwZVsxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0obXVsdGlwbHlQcm9ncmFtKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGEsICdtYXRyaXhBJywgMCk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZShiLCAnbWF0cml4QicsIDEpO1xuICBncGdwdS5leGVjdXRlUHJvZ3JhbSgpO1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge01hdHJpeE9yaWVudGF0aW9ufSBmcm9tICcuLi9tYXRoJztcblxuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vZ3BncHVfY29udGV4dCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlclNvdXJjZShcbiAgICBzaGFyZWREaW1lbnNpb246IG51bWJlciwgYU9yaWVudGF0aW9uOiBNYXRyaXhPcmllbnRhdGlvbixcbiAgICBiT3JpZW50YXRpb246IE1hdHJpeE9yaWVudGF0aW9uKTogc3RyaW5nIHtcbiAgLypcbiAgICAgIEEgPSBbMCAxICAgQiA9IFswIDEgIG91dCA9IFtBMCpCMCtBMSpCMiBBMCpCMStBMSpCM1xuICAgICAgICAgICAyIDNdICAgICAgIDIgM10gICAgICAgIEEyKkIwK0ExKkIyIEEyKkIxK0F3KkIzXVxuICAgICAgb3V0LjAgPSBBMCAqIEIwICsgQTEgKiBCMlxuICAgICAgb3V0LjEgPSBBMCAqIEIxICsgQTEgKiBCM1xuICAgICAgb3V0LjIgPSBBMiAqIEIwICsgQTMgKiBCMlxuICAgICAgb3V0LjMgPSBBMiAqIEIxICsgQTMgKiBCM1xuXG4gICAgICBBKkIgICAgID0gQS54eHp6ICogQi54eXh5ICsgQS55eXd3ICogQi56d3p3XG4gICAgICBBXnQqQiAgID0gQS54eHl5ICogQi54eXh5ICsgQS56end3ICogQi56d3p3XG4gICAgICBBKkJedCAgID0gQS54eHp6ICogQi54enh6ICsgQS55eXd3ICogQi55d3l3XG4gICAgICBBXnQqQl50ID0gQS54eHl5ICogQi54enh6ICsgQS56end3ICogQi55d3l3XG4gICAqL1xuICBjb25zdCBzaGFyZWREaW1lbnNpb25QYWNrZWQgPSBNYXRoLmNlaWwoc2hhcmVkRGltZW5zaW9uIC8gMik7XG4gIGNvbnN0IGFTYW1wbGUgPSAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICAnY2VudGVyLCByZXN1bHRVVi50JyA6XG4gICAgICAncmVzdWx0VVYudCwgY2VudGVyJztcbiAgY29uc3QgYlNhbXBsZSA9IChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgICdyZXN1bHRVVi5zLCBjZW50ZXInIDpcbiAgICAgICdjZW50ZXIsIHJlc3VsdFVWLnMnO1xuICBjb25zdCBhU3dpenpsZTogW3N0cmluZywgc3RyaW5nXSA9XG4gICAgICAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IFsnYS54eHp6JywgJ2EueXl3dyddIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgWydhLnh4eXknLCAnYS56end3J107XG4gIGNvbnN0IGJTd2l6emxlOiBbc3RyaW5nLCBzdHJpbmddID1cbiAgICAgIChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID8gWydiLnh5eHknLCAnYi56d3p3J10gOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBbJ2IueHp4eicsICdiLnl3eXcnXTtcbiAgcmV0dXJuIGBcbiAgICBwcmVjaXNpb24gaGlnaHAgZmxvYXQ7XG4gICAgdW5pZm9ybSBzYW1wbGVyMkQgbWF0cml4QTtcbiAgICB1bmlmb3JtIHNhbXBsZXIyRCBtYXRyaXhCO1xuICAgIHZhcnlpbmcgdmVjMiByZXN1bHRVVjtcblxuICAgIGNvbnN0IGZsb2F0IHNoYXJlZERpbWVuc2lvbiA9ICR7c2hhcmVkRGltZW5zaW9uUGFja2VkfS4wO1xuXG4gICAgdmVjNCBkb3QyeDJBUm93QkNvbCgpIHtcbiAgICAgIHZlYzQgcmVzdWx0ID0gdmVjNCgwLCAwLCAwLCAwKTtcbiAgICAgIGZvciAoZmxvYXQgaSA9IDAuMDsgaSA8IHNoYXJlZERpbWVuc2lvbjsgaSArPSAxLjApIHtcbiAgICAgICAgZmxvYXQgY2VudGVyID0gKGkgKyAwLjUpIC8gc2hhcmVkRGltZW5zaW9uO1xuICAgICAgICB2ZWM0IGEgPSB0ZXh0dXJlMkQobWF0cml4QSwgdmVjMigke2FTYW1wbGV9KSk7XG4gICAgICAgIHZlYzQgYiA9IHRleHR1cmUyRChtYXRyaXhCLCB2ZWMyKCR7YlNhbXBsZX0pKTtcbiAgICAgICAgcmVzdWx0ICs9XG4gICAgICAgICAgKCR7YVN3aXp6bGVbMF19ICogJHtiU3dpenpsZVswXX0pICsgKCR7YVN3aXp6bGVbMV19ICogJHtiU3dpenpsZVsxXX0pO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICB9XG5cbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICBnbF9GcmFnQ29sb3IgPSBkb3QyeDJBUm93QkNvbCgpO1xuICAgIH1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbXVsdGlwbHlNYXRyaXhQYWNrZWQoXG4gICAgZ3BncHU6IEdQR1BVQ29udGV4dCwgbXVsdGlwbHlQcm9ncmFtOiBXZWJHTFByb2dyYW0sIGE6IFdlYkdMVGV4dHVyZSxcbiAgICBiOiBXZWJHTFRleHR1cmUsIHJlc3VsdDogV2ViR0xUZXh0dXJlLFxuICAgIHJlc3VsdFNoYXBlUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gIGdwZ3B1LnNldE91dHB1dFBhY2tlZE1hdHJpeFRleHR1cmUoXG4gICAgICByZXN1bHQsIHJlc3VsdFNoYXBlUm93Q29sWzBdLCByZXN1bHRTaGFwZVJvd0NvbFsxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0obXVsdGlwbHlQcm9ncmFtKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGEsICdtYXRyaXhBJywgMCk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZShiLCAnbWF0cml4QicsIDEpO1xuICBncGdwdS5leGVjdXRlUHJvZ3JhbSgpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdXBsb2FkTXVsdGlwbHlNYXRyaXhQYWNrZWREb3dubG9hZChcbiAgICBhOiBGbG9hdDMyQXJyYXksIGFTaGFwZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSwgYjogRmxvYXQzMkFycmF5LFxuICAgIGJTaGFwZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSwgYU9yaWVudGF0aW9uID0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUixcbiAgICBiT3JpZW50YXRpb24gPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKTogRmxvYXQzMkFycmF5IHtcbiAgY29uc3QgcmVzdWx0TnVtUm93cyA9IChhT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgIGFTaGFwZVJvd0NvbFswXSA6XG4gICAgICBhU2hhcGVSb3dDb2xbMV07XG4gIGNvbnN0IHJlc3VsdE51bUNvbHMgPSAoYk9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICBiU2hhcGVSb3dDb2xbMV0gOlxuICAgICAgYlNoYXBlUm93Q29sWzBdO1xuICBjb25zdCBzaGFyZWREaW1lbnNpb24gPSAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICBhU2hhcGVSb3dDb2xbMV0gOlxuICAgICAgYVNoYXBlUm93Q29sWzBdO1xuXG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCBwcm9ncmFtOiBXZWJHTFByb2dyYW0gPSBncGdwdS5jcmVhdGVQcm9ncmFtKFxuICAgICAgZ2V0RnJhZ21lbnRTaGFkZXJTb3VyY2Uoc2hhcmVkRGltZW5zaW9uLCBhT3JpZW50YXRpb24sIGJPcmllbnRhdGlvbikpO1xuXG4gIGNvbnN0IGFUZXh0dXJlOiBXZWJHTFRleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShhU2hhcGVSb3dDb2xbMF0sIGFTaGFwZVJvd0NvbFsxXSk7XG4gIGNvbnN0IGJUZXh0dXJlOiBXZWJHTFRleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShiU2hhcGVSb3dDb2xbMF0sIGJTaGFwZVJvd0NvbFsxXSk7XG4gIGNvbnN0IHJlc3VsdFRleHR1cmU6IFdlYkdMVGV4dHVyZSA9XG4gICAgICBncGdwdS5jcmVhdGVQYWNrZWRNYXRyaXhUZXh0dXJlKHJlc3VsdE51bVJvd3MsIHJlc3VsdE51bUNvbHMpO1xuXG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShcbiAgICAgIGFUZXh0dXJlLCBhU2hhcGVSb3dDb2xbMF0sIGFTaGFwZVJvd0NvbFsxXSwgYSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShcbiAgICAgIGJUZXh0dXJlLCBiU2hhcGVSb3dDb2xbMF0sIGJTaGFwZVJvd0NvbFsxXSwgYik7XG5cbiAgbXVsdGlwbHlNYXRyaXhQYWNrZWQoXG4gICAgICBncGdwdSwgcHJvZ3JhbSwgYVRleHR1cmUsIGJUZXh0dXJlLCByZXN1bHRUZXh0dXJlLFxuICAgICAgW3Jlc3VsdE51bVJvd3MsIHJlc3VsdE51bUNvbHNdKTtcblxuICBjb25zdCByZXN1bHQgPSBncGdwdS5kb3dubG9hZE1hdHJpeEZyb21QYWNrZWRUZXh0dXJlKFxuICAgICAgcmVzdWx0VGV4dHVyZSwgcmVzdWx0TnVtUm93cywgcmVzdWx0TnVtQ29scyk7XG5cbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShhVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoYlRleHR1cmUpO1xuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKHJlc3VsdFRleHR1cmUpO1xuICBncGdwdS5kZWxldGVQcm9ncmFtKHByb2dyYW0pO1xuICBncGdwdS5kaXNwb3NlKCk7XG5cbiAgcmV0dXJuIHJlc3VsdDtcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uL2NvbnZfdXRpbCc7XG5pbXBvcnQge0dQR1BVQ29udGV4dH0gZnJvbSAnLi9ncGdwdV9jb250ZXh0JztcbmltcG9ydCB7SVNfTkFOX1NIQURFUl9GVU5DfSBmcm9tICcuL3dlYmdsX3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhZ21lbnRTaGFkZXJQb29sQ29tbW9uU291cmNlKFxuICAgIHhTaGFwZVJDRDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlcixcbiAgICBwYWQ6IG51bWJlciwgcG9vbFR5cGU6ICdtYXgnfCdtaW4nfCdhdmcnLCBjb21wdXRlUG9zaXRpb25zOiBib29sZWFuKSB7XG4gIGlmIChwb29sVHlwZSA9PT0gJ2F2ZycgJiYgY29tcHV0ZVBvc2l0aW9ucykge1xuICAgIHRocm93IG5ldyBFcnJvcignQ2Fubm90IGNvbXB1dGUgcG9zaXRpb25zIGZvciBhdmVyYWdlIHBvb2wuJyk7XG4gIH1cblxuICBjb25zdCBkZXB0aCA9IHhTaGFwZVJDRFsyXTtcblxuICBjb25zdCB4VGV4U2hhcGVSQyA9IGNvbnZfdXRpbC5jb21wdXRlVGV4U2hhcGVGcm9tM0QoeFNoYXBlUkNEKTtcblxuICBsZXQgcmV0dXJuVmFsdWUgPSAnbWluTWF4VmFsdWUnO1xuICBpZiAoY29tcHV0ZVBvc2l0aW9ucykge1xuICAgIHJldHVyblZhbHVlID0gJ21pbk1heFBvc2l0aW9uJztcbiAgfSBlbHNlIGlmIChwb29sVHlwZSA9PT0gJ2F2ZycpIHtcbiAgICByZXR1cm5WYWx1ZSA9ICdhdmdWYWx1ZSc7XG4gIH1cblxuICByZXR1cm4gYFxuICAgIHByZWNpc2lvbiBoaWdocCBmbG9hdDtcbiAgICB1bmlmb3JtIHNhbXBsZXIyRCB4O1xuICAgIHZhcnlpbmcgdmVjMiByZXN1bHRVVjtcblxuICAgIGNvbnN0IHZlYzIgaGFsZkNSID0gdmVjMigwLjUsIDAuNSk7XG4gICAgY29uc3QgdmVjMiB4U2hhcGVDUiA9IHZlYzIoJHt4VGV4U2hhcGVSQ1sxXX0sICR7eFRleFNoYXBlUkNbMF19KTtcblxuICAgICR7SVNfTkFOX1NIQURFUl9GVU5DfVxuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgdmVjMiB5VGV4Q1IgPSBmbG9vcihnbF9GcmFnQ29vcmQueHkpO1xuXG4gICAgICAvLyBNYXAgZnJvbSAyRCAoeVRleFIsIHlUZXhDKSB0byAzRCAoeVIsIHlDLCBkMikuXG4gICAgICBmbG9hdCB5UiA9IHlUZXhDUi55O1xuICAgICAgZmxvYXQgeUMgPSBmbG9vcih5VGV4Q1IueCAvICR7ZGVwdGh9LjApO1xuICAgICAgZmxvYXQgZCA9IG1vZCh5VGV4Q1IueCwgJHtkZXB0aH0uMCk7XG5cbiAgICAgIHZlYzIgeFJDQ29ybmVyID0gdmVjMih5UiwgeUMpICogdmVjMigke3N0cmlkZX0sICR7c3RyaWRlfSkgLVxuICAgICAgICAgIHZlYzIoJHtwYWR9LjAsICR7cGFkfS4wKTtcbiAgICAgIGZsb2F0IHhSQ29ybmVyID0geFJDQ29ybmVyLng7XG4gICAgICBmbG9hdCB4Q0Nvcm5lciA9IHhSQ0Nvcm5lci55O1xuXG4gICAgICAvLyBtYXgvbWluIHgoPywgPywgZCkgdG8gZ2V0IHkoeVIsIHlDLCBkKS5cbiAgICAgIC8vID8gPSB0byBiZSBkZXRlcm1pbmVkXG4gICAgICBmbG9hdCBtaW5NYXhWYWx1ZSA9IDAuMDtcbiAgICAgIGZsb2F0IG1pbk1heFZhbHVlRm91bmQgPSAwLjA7XG4gICAgICBmbG9hdCBtaW5NYXhQb3NpdGlvbiA9IDAuMDtcbiAgICAgIGZsb2F0IGF2Z1ZhbHVlID0gMC4wO1xuXG4gICAgICBmb3IgKGZsb2F0IHdSID0gMC4wOyB3UiA8ICR7ZlNpemV9LjA7IHdSICs9IDEuMCkge1xuICAgICAgICBmbG9hdCB4UiA9IHhSQ29ybmVyICsgd1I7XG4gICAgICAgIGZsb2F0IHhUZXhSID0geFI7XG5cbiAgICAgICAgZm9yIChmbG9hdCB3QyA9IDAuMDsgd0MgPCAke2ZTaXplfS4wOyB3QyArPSAxLjApIHtcbiAgICAgICAgICBmbG9hdCB4QyA9IHhDQ29ybmVyICsgd0M7XG4gICAgICAgICAgZmxvYXQgeFRleEMgPSB4QyAqICR7ZGVwdGh9LjAgKyBkO1xuXG4gICAgICAgICAgdmVjMiB0ZXhDUiA9IHZlYzIoeFRleEMsIHhUZXhSKTtcblxuICAgICAgICAgIC8vIENoZWNrIGlmIHRoZSByZXF1ZXN0ZWQgVVYgaXMgaW52YWxpZC5cbiAgICAgICAgICB2ZWMyIHV2ID0gKHRleENSICsgaGFsZkNSKSAvIHhTaGFwZUNSO1xuICAgICAgICAgIGJvb2wgbGVzc1RoYW5aZXJvID0gYW55KGxlc3NUaGFuKHV2LCB2ZWMyKDAsIDApKSk7XG4gICAgICAgICAgYm9vbCBncmVhdGVyVGhhbk9uZSA9IGFueShncmVhdGVyVGhhbih1diwgdmVjMigxLCAxKSkpO1xuICAgICAgICAgIGJvb2wgb3V0c2lkZSA9IGxlc3NUaGFuWmVybyB8fCBncmVhdGVyVGhhbk9uZTtcbiAgICAgICAgICBpZiAob3V0c2lkZSkge1xuICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgZmxvYXQgdmFsdWUgPSB0ZXh0dXJlMkQoeCwgdXYpLnI7XG4gICAgICAgICAgaWYgKGlzTmFOKHZhbHVlKSkge1xuICAgICAgICAgICAgZ2xfRnJhZ0NvbG9yID0gdmVjNCh2YWx1ZSwgMCwgMCwgMCk7XG4gICAgICAgICAgICByZXR1cm47XG4gICAgICAgICAgfVxuICAgICAgICAgIGlmICgke3Bvb2xUeXBlID09PSAnYXZnJ30pIHtcbiAgICAgICAgICAgIGF2Z1ZhbHVlICs9IHZhbHVlIC8gJHtmU2l6ZSAqIGZTaXplfS4wO1xuICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAvLyBJZiBhIG1pbiAvIG1heCB2YWx1ZSBoYXMgYWxyZWFkeSBiZWVuIGZvdW5kLCB1c2UgaXQuIElmIG5vdCwgdXNlXG4gICAgICAgICAgICAvLyB0aGUgY3VycmVudCB2YWx1ZS5cbiAgICAgICAgICAgIGZsb2F0IGN1cnJlbnRNaW5NYXhWYWx1ZSA9IG1peChcbiAgICAgICAgICAgICAgICB2YWx1ZSwgbWluTWF4VmFsdWUsIG1pbk1heFZhbHVlRm91bmQpO1xuICAgICAgICAgICAgaWYgKHZhbHVlICR7cG9vbFR5cGUgPT09ICdtaW4nID8gJzw9JyA6ICc+PSd9IGN1cnJlbnRNaW5NYXhWYWx1ZSkge1xuICAgICAgICAgICAgICBtaW5NYXhWYWx1ZSA9IHZhbHVlO1xuICAgICAgICAgICAgICBtaW5NYXhWYWx1ZUZvdW5kID0gMS4wO1xuICAgICAgICAgICAgICBpZiAoJHtjb21wdXRlUG9zaXRpb25zfSkge1xuICAgICAgICAgICAgICAgIG1pbk1heFBvc2l0aW9uID0gd1IgKiAke2ZTaXplfS4wICsgd0M7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGdsX0ZyYWdDb2xvciA9IHZlYzQoJHtyZXR1cm5WYWx1ZX0sIDAsIDAsIDApO1xuICAgIH1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gcG9vbENvbW1vbihcbiAgICBncGdwdTogR1BHUFVDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0sIHg6IFdlYkdMVGV4dHVyZSxcbiAgICByZXN1bHQ6IFdlYkdMVGV4dHVyZSwgcmVzdWx0U2hhcGVSb3dDb2w6IFtudW1iZXIsIG51bWJlcl0pIHtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShcbiAgICAgIHJlc3VsdCwgcmVzdWx0U2hhcGVSb3dDb2xbMF0sIHJlc3VsdFNoYXBlUm93Q29sWzFdKTtcbiAgZ3BncHUuc2V0UHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKHgsICd4JywgMCk7XG4gIGdwZ3B1LmV4ZWN1dGVQcm9ncmFtKCk7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vLi4vdXRpbCc7XG5pbXBvcnQge05EQXJyYXl9IGZyb20gJy4uL25kYXJyYXknO1xuXG5leHBvcnQgdHlwZSBJbnB1dCA9IHtcbiAgbmFtZTogc3RyaW5nOyBhcnJheTogTkRBcnJheTtcbn07XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlU2hhZGVyS2V5KGlucHV0czogTkRBcnJheVtdLCBvdXRwdXQ6IE5EQXJyYXkpOiBzdHJpbmcge1xuICBjb25zdCBpbnMgPSBpbnB1dHMubWFwKHggPT4geC5zaGFwZSArICdfJyArIHguZ2V0VGV4dHVyZVNoYXBlUkMoKSk7XG4gIHJldHVybiBpbnMuam9pbignXycpICsgJ18nICsgb3V0cHV0LnNoYXBlICsgJ18nICsgb3V0cHV0LmdldFRleHR1cmVTaGFwZVJDKCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlU2hhZGVyKFxuICAgIGlucHV0czogSW5wdXRbXSwgb3V0cHV0OiBOREFycmF5LCB1c2VyQ29kZTogc3RyaW5nKTogc3RyaW5nIHtcbiAgY29uc3QgaW5wdXRQcmVmaXhTbmlwcGV0ID1cbiAgICAgIGlucHV0cy5tYXAoeCA9PiBgdW5pZm9ybSBzYW1wbGVyMkQgJHt4Lm5hbWV9O2ApLmpvaW4oJ1xcbicpO1xuICBjb25zdCBpbnB1dFNhbXBsaW5nU25pcHBldCA9XG4gICAgICBpbnB1dHMubWFwKHggPT4gZ2V0SW5wdXRTYW1wbGluZ1NuaXBwZXQoeCkpLmpvaW4oJ1xcbicpO1xuICBjb25zdCBvdXRUZXhTaGFwZSA9IG91dHB1dC5nZXRUZXh0dXJlU2hhcGVSQygpO1xuICBjb25zdCBvdXRwdXRTYW1wbGluZ1NuaXBwZXQgPVxuICAgICAgZ2V0T3V0cHV0U2FtcGxpbmdTbmlwcGV0KG91dHB1dC5zaGFwZSwgb3V0VGV4U2hhcGUpO1xuICBjb25zdCBzb3VyY2UgPSBbXG4gICAgU0hBREVSX1BSRUZJWCwgaW5wdXRQcmVmaXhTbmlwcGV0LCBTQU1QTEVfMkRfU05JUFBFVCwgaW5wdXRTYW1wbGluZ1NuaXBwZXQsXG4gICAgb3V0cHV0U2FtcGxpbmdTbmlwcGV0LCB1c2VyQ29kZVxuICBdLmpvaW4oJ1xcbicpO1xuICByZXR1cm4gc291cmNlO1xufVxuXG5mdW5jdGlvbiBnZXRJbnB1dFNhbXBsaW5nU25pcHBldChpbnB1dDogSW5wdXQpIHtcbiAgY29uc3QgYXJyID0gaW5wdXQuYXJyYXk7XG4gIGNvbnN0IHNoYXBlID0gYXJyLnNoYXBlO1xuICBjb25zdCB0ZXhTaGFwZSA9IGFyci5nZXRUZXh0dXJlU2hhcGVSQyhzaGFwZSBhcyBbbnVtYmVyLCBudW1iZXJdKTtcbiAgc3dpdGNoIChzaGFwZS5sZW5ndGgpIHtcbiAgICBjYXNlIDI6XG4gICAgICByZXR1cm4gZ2V0U2FtcGxlcjJEKGlucHV0Lm5hbWUsIHNoYXBlIGFzIFtudW1iZXIsIG51bWJlcl0sIHRleFNoYXBlKTtcbiAgICBkZWZhdWx0OlxuICAgICAgdGhyb3cgbmV3IEVycm9yKGAke2Fyci5yYW5rfS1EIGlucHV0IHNhbXBsaW5nIGlzIG5vdCB5ZXQgc3VwcG9ydGVkYCk7XG4gIH1cbn1cblxuZnVuY3Rpb24gZ2V0T3V0cHV0U2FtcGxpbmdTbmlwcGV0KFxuICAgIG91dFNoYXBlOiBudW1iZXJbXSwgb3V0VGV4U2hhcGU6IFtudW1iZXIsIG51bWJlcl0pOiBzdHJpbmcge1xuICBzd2l0Y2ggKG91dFNoYXBlLmxlbmd0aCkge1xuICAgIGNhc2UgMjpcbiAgICAgIHJldHVybiBnZXRPdXRwdXQyRENvb3JkcyhvdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXJdLCBvdXRUZXhTaGFwZSk7XG4gICAgZGVmYXVsdDpcbiAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICBgJHtvdXRTaGFwZS5sZW5ndGh9LUQgb3V0cHV0IHNhbXBsaW5nIGlzIG5vdCB5ZXQgc3VwcG9ydGVkYCk7XG4gIH1cbn1cblxuY29uc3QgU0hBREVSX1BSRUZJWCA9IGBcbiAgcHJlY2lzaW9uIGhpZ2hwIGZsb2F0O1xuICB2YXJ5aW5nIHZlYzIgcmVzdWx0VVY7XG4gIGNvbnN0IHZlYzIgaGFsZkNSID0gdmVjMigwLjUsIDAuNSk7XG5cbiAgdm9pZCBzZXRPdXRwdXQoZmxvYXQgdmFsKSB7XG4gICAgZ2xfRnJhZ0NvbG9yID0gdmVjNCh2YWwsIDAsIDAsIDApO1xuICB9XG5gO1xuXG5jb25zdCBTQU1QTEVfMkRfU05JUFBFVCA9IGBcbiAgZmxvYXQgc2FtcGxlMkQoc2FtcGxlcjJEIHRleHR1cmUsIGZsb2F0IHRleE51bVIsIGZsb2F0IHRleE51bUMsIGZsb2F0IG51bUMsXG4gICAgICBmbG9hdCByb3csIGZsb2F0IGNvbCkge1xuICAgIGZsb2F0IGluZGV4ID0gZG90KHZlYzIocm93LCBjb2wpLCB2ZWMyKG51bUMsIDEuMCkpO1xuICAgIGZsb2F0IHRleFIgPSBmbG9vcihpbmRleCAvIHRleE51bUMpO1xuICAgIGZsb2F0IHRleEMgPSBtb2QoaW5kZXgsIHRleE51bUMpO1xuICAgIHZlYzIgdXYgPSAodmVjMih0ZXhDLCB0ZXhSKSArIGhhbGZDUikgLyB2ZWMyKHRleE51bUMsIHRleE51bVIpO1xuICAgIHJldHVybiB0ZXh0dXJlMkQodGV4dHVyZSwgdXYpLnI7XG4gIH1cbmA7XG5cbmZ1bmN0aW9uIGdldE91dHB1dDJEQ29vcmRzKFxuICAgIHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLCB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSkge1xuICBpZiAodXRpbC5hcnJheXNFcXVhbChzaGFwZSwgdGV4U2hhcGUpKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIHZlYzIgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgICByZXR1cm4gZmxvb3IoZ2xfRnJhZ0Nvb3JkLnl4KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG4gIHJldHVybiBgXG4gICAgdmVjMiBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICB2ZWMyIHJlc1RleFJDID0gZmxvb3IoZ2xfRnJhZ0Nvb3JkLnl4KTtcbiAgICAgIGZsb2F0IGluZGV4ID0gZG90KHJlc1RleFJDLCB2ZWMyKCR7dGV4U2hhcGVbMV19LjAsIDEuMCkpO1xuICAgICAgZmxvYXQgciA9IGZsb29yKGluZGV4IC8gJHtzaGFwZVsxXX0uMCk7XG4gICAgICBmbG9hdCBjID0gbW9kKGluZGV4LCAke3NoYXBlWzFdfS4wKTtcbiAgICAgIHJldHVybiB2ZWMyKHIsIGMpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0U2FtcGxlcjJEKFxuICAgIHRleE5hbWU6IHN0cmluZywgc2hhcGU6IFtudW1iZXIsIG51bWJlcl0sIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gIGNvbnN0IGZ1bmNOYW1lID0gJ2dldCcgKyB0ZXhOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKTtcbiAgY29uc3QgdFIgPSB0ZXhTaGFwZVswXTtcbiAgY29uc3QgdEMgPSB0ZXhTaGFwZVsxXTtcbiAgaWYgKHV0aWwuYXJyYXlzRXF1YWwoc2hhcGUsIHRleFNoYXBlKSkge1xuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShmbG9hdCByb3csIGZsb2F0IGNvbCkge1xuICAgICAgICB2ZWMyIHV2ID0gKHZlYzIoY29sLCByb3cpICsgaGFsZkNSKSAvIHZlYzIoJHt0Q30uMCwgJHt0Un0uMCk7XG4gICAgICAgIHJldHVybiB0ZXh0dXJlMkQoJHt0ZXhOYW1lfSwgdXYpLnI7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICByZXR1cm4gYFxuICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IHJvdywgZmxvYXQgY29sKSB7XG4gICAgICByZXR1cm4gc2FtcGxlMkQoJHt0ZXhOYW1lfSwgJHt0Un0uMCwgJHt0Q30uMCwgJHtzaGFwZVsxXX0uMCwgcm93LCBjb2wpO1xuICAgIH1cbiAgYDtcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFVucGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQoXG4gICAgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgcmV0dXJuIFtjb2x1bW5zLCByb3dzXTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFVucGFja2VkQXJyYXlTaXplRnJvbU1hdHJpeFNpemUoXG4gICAgbWF0cml4U2l6ZTogbnVtYmVyLCBjaGFubmVsc1BlclRleHR1cmU6IG51bWJlcik6IG51bWJlciB7XG4gIHJldHVybiBtYXRyaXhTaXplICogY2hhbm5lbHNQZXJUZXh0dXJlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0Q29sb3JNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChcbiAgICByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFtudW1iZXIsIG51bWJlcl0ge1xuICByZXR1cm4gW2NvbHVtbnMgKiA0LCByb3dzXTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldE1hdHJpeFNpemVGcm9tVW5wYWNrZWRBcnJheVNpemUoXG4gICAgdW5wYWNrZWRTaXplOiBudW1iZXIsIGNoYW5uZWxzUGVyVGV4dHVyZTogbnVtYmVyKTogbnVtYmVyIHtcbiAgaWYgKHVucGFja2VkU2l6ZSAlIGNoYW5uZWxzUGVyVGV4dHVyZSAhPT0gMCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ3VucGFja2VkU2l6ZSAoJyArIHVucGFja2VkU2l6ZSArICcpIG11c3QgYmUgYSBtdWx0aXBsZSBvZiAnICtcbiAgICAgICAgY2hhbm5lbHNQZXJUZXh0dXJlKTtcbiAgfVxuICByZXR1cm4gdW5wYWNrZWRTaXplIC8gY2hhbm5lbHNQZXJUZXh0dXJlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZW5jb2RlTWF0cml4VG9VbnBhY2tlZEFycmF5KFxuICAgIG1hdHJpeDogRmxvYXQzMkFycmF5LCB1bnBhY2tlZEFycmF5OiBGbG9hdDMyQXJyYXksXG4gICAgY2hhbm5lbHNQZXJUZXh0dXJlOiBudW1iZXIpIHtcbiAgY29uc3QgcmVxdWlyZWRTaXplID1cbiAgICAgIGdldFVucGFja2VkQXJyYXlTaXplRnJvbU1hdHJpeFNpemUobWF0cml4Lmxlbmd0aCwgY2hhbm5lbHNQZXJUZXh0dXJlKTtcbiAgaWYgKHVucGFja2VkQXJyYXkubGVuZ3RoIDwgcmVxdWlyZWRTaXplKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAndW5wYWNrZWRBcnJheSBsZW5ndGggKCcgKyB1bnBhY2tlZEFycmF5Lmxlbmd0aCArXG4gICAgICAgICcpIG11c3QgYmUgPj0gJyArIHJlcXVpcmVkU2l6ZSk7XG4gIH1cbiAgbGV0IGRzdCA9IDA7XG4gIGZvciAobGV0IHNyYyA9IDA7IHNyYyA8IG1hdHJpeC5sZW5ndGg7ICsrc3JjKSB7XG4gICAgdW5wYWNrZWRBcnJheVtkc3RdID0gbWF0cml4W3NyY107XG4gICAgZHN0ICs9IGNoYW5uZWxzUGVyVGV4dHVyZTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZGVjb2RlTWF0cml4RnJvbVVucGFja2VkQXJyYXkoXG4gICAgdW5wYWNrZWRBcnJheTogRmxvYXQzMkFycmF5LCBtYXRyaXg6IEZsb2F0MzJBcnJheSxcbiAgICBjaGFubmVsc1BlclRleHR1cmU6IG51bWJlcikge1xuICBjb25zdCByZXF1aXJlZFNpemUgPSBnZXRNYXRyaXhTaXplRnJvbVVucGFja2VkQXJyYXlTaXplKFxuICAgICAgdW5wYWNrZWRBcnJheS5sZW5ndGgsIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG4gIGlmIChtYXRyaXgubGVuZ3RoIDwgcmVxdWlyZWRTaXplKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnbWF0cml4IGxlbmd0aCAoJyArIG1hdHJpeC5sZW5ndGggKyAnKSBtdXN0IGJlID49ICcgKyByZXF1aXJlZFNpemUpO1xuICB9XG4gIGxldCBkc3QgPSAwO1xuICBmb3IgKGxldCBzcmMgPSAwOyBzcmMgPCB1bnBhY2tlZEFycmF5Lmxlbmd0aDsgc3JjICs9IGNoYW5uZWxzUGVyVGV4dHVyZSkge1xuICAgIG1hdHJpeFtkc3QrK10gPSB1bnBhY2tlZEFycmF5W3NyY107XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KFxuICAgIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogW251bWJlciwgbnVtYmVyXSB7XG4gIHJldHVybiBbTWF0aC5jZWlsKGNvbHVtbnMgLyAyKSwgTWF0aC5jZWlsKHJvd3MgLyAyKV07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRQYWNrZWRSR0JBQXJyYXlTaXplRnJvbU1hdHJpeFNoYXBlKFxuICAgIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogbnVtYmVyIHtcbiAgY29uc3QgW3csIGhdID0gZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gIHJldHVybiB3ICogaCAqIDQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBlbmNvZGVNYXRyaXhUb1BhY2tlZFJHQkEoXG4gICAgbWF0cml4OiBGbG9hdDMyQXJyYXksIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyLFxuICAgIHBhY2tlZFJHQkE6IEZsb2F0MzJBcnJheSkge1xuICBjb25zdCByZXF1aXJlZFNpemUgPSBnZXRQYWNrZWRSR0JBQXJyYXlTaXplRnJvbU1hdHJpeFNoYXBlKHJvd3MsIGNvbHVtbnMpO1xuICBpZiAocGFja2VkUkdCQS5sZW5ndGggPCByZXF1aXJlZFNpemUpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdwYWNrZWRSR0JBIGxlbmd0aCAoJyArIHBhY2tlZFJHQkEubGVuZ3RoICtcbiAgICAgICAgJykgbXVzdCBiZSA+PSAnICsgcmVxdWlyZWRTaXplKTtcbiAgfVxuICAvKlxuICAgIFVucGFja2VkIG1hdHJpeCwgcm93LW1ham9yIG9yZGVyIGluIEZsb2F0MzJBcnJheVsxNl06ICBBIEIgQyBEXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIEUgRiBHIEhcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgSSBKIEsgTFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBNIE4gTyBQXG5cbiAgICBQYWNrZWQgbWF0cml4LCAyeDIgUkdCQTMyIHRleHR1cmUgKG1lbW9yeSB2aWV3KTogICAgICAgQUJFRiBDREdIIElKTU4gS0xPUFxuXG4gICAgUGFja2VkIG1hdHJpeCwgMngyIFJHQkEzMiB0ZXh0dXJlIChtYXRyaXggdmlldyk6ICAgICAgIEFCfENEXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIEVGfEdIXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIC0tKy0tXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIElKfEtMXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIE1OfE9QXG4gICAqL1xuICBjb25zdCBbdGV4dHVyZVdpZHRoLCB0ZXh0dXJlSGVpZ2h0XSA9XG4gICAgICBnZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcbiAgY29uc3Qgb2RkV2lkdGggPSAoY29sdW1ucyAlIDIpID09PSAxO1xuICBjb25zdCBvZGRIZWlnaHQgPSAocm93cyAlIDIpID09PSAxO1xuICBjb25zdCB3aWR0aEluRnVsbEJsb2NrcyA9IE1hdGguZmxvb3IoY29sdW1ucyAvIDIpO1xuICBjb25zdCBoZWlnaHRJbkZ1bGxCbG9ja3MgPSBNYXRoLmZsb29yKHJvd3MgLyAyKTtcblxuICAvLyBsb29wIG92ZXIgZnVsbCAyeDIgYmxvY2tzXG4gIHtcbiAgICBjb25zdCBkc3RTdHJpZGUgPSAob2RkV2lkdGggPyA0IDogMCk7XG4gICAgY29uc3Qgb25lUm93ID0gY29sdW1ucztcbiAgICBsZXQgZHN0ID0gMDtcbiAgICBmb3IgKGxldCBibG9ja1kgPSAwOyBibG9ja1kgPCBoZWlnaHRJbkZ1bGxCbG9ja3M7ICsrYmxvY2tZKSB7XG4gICAgICBjb25zdCBtYXRyaXhTcmNSb3cgPSAoYmxvY2tZICogMiAqIGNvbHVtbnMpO1xuICAgICAgZm9yIChsZXQgYmxvY2tYID0gMDsgYmxvY2tYIDwgd2lkdGhJbkZ1bGxCbG9ja3M7ICsrYmxvY2tYKSB7XG4gICAgICAgIGNvbnN0IG1hdHJpeFNyY0NvbCA9IGJsb2NrWCAqIDI7XG4gICAgICAgIGNvbnN0IHNyYyA9IG1hdHJpeFNyY1JvdyArIG1hdHJpeFNyY0NvbDtcbiAgICAgICAgcGFja2VkUkdCQVtkc3RdID0gbWF0cml4W3NyY107XG4gICAgICAgIHBhY2tlZFJHQkFbZHN0ICsgMV0gPSBtYXRyaXhbc3JjICsgMV07XG4gICAgICAgIHBhY2tlZFJHQkFbZHN0ICsgMl0gPSBtYXRyaXhbc3JjICsgb25lUm93XTtcbiAgICAgICAgcGFja2VkUkdCQVtkc3QgKyAzXSA9IG1hdHJpeFtzcmMgKyBvbmVSb3cgKyAxXTtcbiAgICAgICAgZHN0ICs9IDQ7XG4gICAgICB9XG4gICAgICBkc3QgKz0gZHN0U3RyaWRlO1xuICAgIH1cbiAgfVxuXG4gIC8vIGxvb3AgZG93biBmaW5hbCBvZGQgY29sdW1uXG4gIGlmIChvZGRXaWR0aCkge1xuICAgIGxldCBzcmMgPSBjb2x1bW5zIC0gMTtcbiAgICBsZXQgZHN0ID0gKHRleHR1cmVXaWR0aCAtIDEpICogNDtcbiAgICBjb25zdCBzcmNTdHJpZGUgPSAyICogY29sdW1ucztcbiAgICBjb25zdCBkc3RTdHJpZGUgPSB0ZXh0dXJlV2lkdGggKiA0O1xuICAgIGZvciAobGV0IGJsb2NrWSA9IDA7IGJsb2NrWSA8IGhlaWdodEluRnVsbEJsb2NrczsgKytibG9ja1kpIHtcbiAgICAgIHBhY2tlZFJHQkFbZHN0XSA9IG1hdHJpeFtzcmNdO1xuICAgICAgcGFja2VkUkdCQVtkc3QgKyAyXSA9IG1hdHJpeFtzcmMgKyBjb2x1bW5zXTtcbiAgICAgIHNyYyArPSBzcmNTdHJpZGU7XG4gICAgICBkc3QgKz0gZHN0U3RyaWRlO1xuICAgIH1cbiAgfVxuXG4gIC8vIGxvb3AgYWNyb3NzIGZpbmFsIHJvd1xuICBpZiAob2RkSGVpZ2h0KSB7XG4gICAgbGV0IHNyYyA9IChyb3dzIC0gMSkgKiBjb2x1bW5zO1xuICAgIGxldCBkc3QgPSAodGV4dHVyZUhlaWdodCAtIDEpICogdGV4dHVyZVdpZHRoICogNDtcbiAgICBmb3IgKGxldCBibG9ja1ggPSAwOyBibG9ja1ggPCB3aWR0aEluRnVsbEJsb2NrczsgKytibG9ja1gpIHtcbiAgICAgIHBhY2tlZFJHQkFbZHN0KytdID0gbWF0cml4W3NyYysrXTtcbiAgICAgIHBhY2tlZFJHQkFbZHN0KytdID0gbWF0cml4W3NyYysrXTtcbiAgICAgIGRzdCArPSAyO1xuICAgIH1cbiAgfVxuXG4gIC8vIGZpbGwgaW4gYm90dG9tLXJpZ2h0IHRleGVsXG4gIGlmIChvZGRXaWR0aCAmJiBvZGRIZWlnaHQpIHtcbiAgICBwYWNrZWRSR0JBW3BhY2tlZFJHQkEubGVuZ3RoIC0gNF0gPSBtYXRyaXhbbWF0cml4Lmxlbmd0aCAtIDFdO1xuICB9XG5cbiAgcmV0dXJuIHBhY2tlZFJHQkE7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBkZWNvZGVNYXRyaXhGcm9tUGFja2VkUkdCQShcbiAgICBwYWNrZWRSR0JBOiBGbG9hdDMyQXJyYXksIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyLFxuICAgIG1hdHJpeDogRmxvYXQzMkFycmF5KTogRmxvYXQzMkFycmF5IHtcbiAgY29uc3QgcmVxdWlyZWRTaXplID0gcm93cyAqIGNvbHVtbnM7XG4gIGlmIChyZXF1aXJlZFNpemUgPCBtYXRyaXgubGVuZ3RoKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnbWF0cml4IGxlbmd0aCAoJyArIG1hdHJpeC5sZW5ndGggKyAnKSBtdXN0IGJlID49ICcgKyByZXF1aXJlZFNpemUpO1xuICB9XG4gIGNvbnN0IG9kZFdpZHRoID0gKGNvbHVtbnMgJSAyKSA9PT0gMTtcbiAgY29uc3Qgb2RkSGVpZ2h0ID0gKHJvd3MgJSAyKSA9PT0gMTtcbiAgY29uc3Qgd2lkdGhJbkZ1bGxCbG9ja3MgPSBNYXRoLmZsb29yKGNvbHVtbnMgLyAyKTtcbiAgY29uc3QgaGVpZ2h0SW5GdWxsQmxvY2tzID0gTWF0aC5mbG9vcihyb3dzIC8gMik7XG4gIGNvbnN0IFt0ZXh0dXJlV2lkdGgsIHRleHR1cmVIZWlnaHRdID1cbiAgICAgIGdldFBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KHJvd3MsIGNvbHVtbnMpO1xuXG4gIC8vIGxvb3Agb3ZlciBmdWxsIDJ4MiBibG9ja3NcbiAge1xuICAgIGNvbnN0IHNyY1N0cmlkZSA9IG9kZFdpZHRoID8gNCA6IDA7XG4gICAgY29uc3QgZHN0U3RyaWRlID0gY29sdW1ucyArIChvZGRXaWR0aCA/IDEgOiAwKTtcbiAgICBsZXQgc3JjID0gMDtcbiAgICBsZXQgZHN0Um93MSA9IDA7XG4gICAgbGV0IGRzdFJvdzIgPSBjb2x1bW5zO1xuICAgIGZvciAobGV0IGJsb2NrWSA9IDA7IGJsb2NrWSA8IGhlaWdodEluRnVsbEJsb2NrczsgKytibG9ja1kpIHtcbiAgICAgIGZvciAobGV0IGJsb2NrWCA9IDA7IGJsb2NrWCA8IHdpZHRoSW5GdWxsQmxvY2tzOyArK2Jsb2NrWCkge1xuICAgICAgICBtYXRyaXhbZHN0Um93MSsrXSA9IHBhY2tlZFJHQkFbc3JjKytdO1xuICAgICAgICBtYXRyaXhbZHN0Um93MSsrXSA9IHBhY2tlZFJHQkFbc3JjKytdO1xuICAgICAgICBtYXRyaXhbZHN0Um93MisrXSA9IHBhY2tlZFJHQkFbc3JjKytdO1xuICAgICAgICBtYXRyaXhbZHN0Um93MisrXSA9IHBhY2tlZFJHQkFbc3JjKytdO1xuICAgICAgfVxuICAgICAgc3JjICs9IHNyY1N0cmlkZTtcbiAgICAgIGRzdFJvdzEgKz0gZHN0U3RyaWRlO1xuICAgICAgZHN0Um93MiArPSBkc3RTdHJpZGU7XG4gICAgfVxuICB9XG5cbiAgLy8gbG9vcCBkb3duIGZpbmFsIGNvbHVtblxuICBpZiAob2RkV2lkdGgpIHtcbiAgICBsZXQgc3JjID0gKHRleHR1cmVXaWR0aCAtIDEpICogNDtcbiAgICBsZXQgZHN0ID0gY29sdW1ucyAtIDE7XG4gICAgY29uc3Qgc3JjU3RyaWRlID0gdGV4dHVyZVdpZHRoICogNDtcbiAgICBjb25zdCBkc3RTdHJpZGUgPSAyICogY29sdW1ucztcbiAgICBmb3IgKGxldCBibG9ja1kgPSAwOyBibG9ja1kgPCBoZWlnaHRJbkZ1bGxCbG9ja3M7ICsrYmxvY2tZKSB7XG4gICAgICBtYXRyaXhbZHN0XSA9IHBhY2tlZFJHQkFbc3JjXTtcbiAgICAgIG1hdHJpeFtkc3QgKyBjb2x1bW5zXSA9IHBhY2tlZFJHQkFbc3JjICsgMl07XG4gICAgICBzcmMgKz0gc3JjU3RyaWRlO1xuICAgICAgZHN0ICs9IGRzdFN0cmlkZTtcbiAgICB9XG4gIH1cblxuICAvLyBsb29wIGFjcm9zcyBmaW5hbCByb3dcbiAgaWYgKG9kZEhlaWdodCkge1xuICAgIGxldCBzcmMgPSAodGV4dHVyZUhlaWdodCAtIDEpICogdGV4dHVyZVdpZHRoICogNDtcbiAgICBsZXQgZHN0ID0gKHJvd3MgLSAxKSAqIGNvbHVtbnM7XG4gICAgZm9yIChsZXQgYmxvY2tYID0gMDsgYmxvY2tYIDwgd2lkdGhJbkZ1bGxCbG9ja3M7ICsrYmxvY2tYKSB7XG4gICAgICBtYXRyaXhbZHN0KytdID0gcGFja2VkUkdCQVtzcmMrK107XG4gICAgICBtYXRyaXhbZHN0KytdID0gcGFja2VkUkdCQVtzcmMrK107XG4gICAgICBzcmMgKz0gMjtcbiAgICB9XG4gIH1cblxuICAvLyBmaWxsIGluIGJvdHRvbS1yaWdodCBjZWxsXG4gIGlmIChvZGRXaWR0aCAmJiBvZGRIZWlnaHQpIHtcbiAgICBtYXRyaXhbbWF0cml4Lmxlbmd0aCAtIDFdID0gcGFja2VkUkdCQVtwYWNrZWRSR0JBLmxlbmd0aCAtIDRdO1xuICB9XG5cbiAgcmV0dXJuIG1hdHJpeDtcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxubGV0IFVTRV9XRUJHTDJfV0hFTl9BVkFJTEFCTEUgPSB0cnVlO1xubGV0IFdFQkdMMl9FTkFCTEVEOiBib29sZWFufHVuZGVmaW5lZCA9IG51bGwhO1xubGV0IE1BWF9URVhUVVJFX1NJWkU6IG51bWJlciA9IG51bGwhO1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uLy4uL3V0aWwnO1xuXG5leHBvcnQgaW50ZXJmYWNlIFdlYkdMQ29udGV4dEF0dHJpYnV0ZXMge1xuICBhbHBoYT86IGJvb2xlYW47XG4gIGFudGlhbGlhcz86IGJvb2xlYW47XG4gIHByZW11bHRpcGxpZWRBbHBoYT86IGJvb2xlYW47XG4gIHByZXNlcnZlRHJhd2luZ0J1ZmZlcj86IGJvb2xlYW47XG4gIGRlcHRoPzogYm9vbGVhbjtcbiAgc3RlbmNpbD86IGJvb2xlYW47XG4gIGZhaWxJZk1ham9yUGVyZm9ybWFuY2VDYXZlYXQ/OiBib29sZWFuO1xufVxuXG4vKiogQGhpZGRlbiAqL1xuZXhwb3J0IGNvbnN0IElTX05BTl9TSEFERVJfRlVOQyA9IGBcbmJvb2wgaXNOYU4oZmxvYXQgdmFsKSB7XG4gIHJldHVybiB2YWwgPT0gdmFsID8gZmFsc2UgOiB0cnVlO1xufVxuYDtcblxuZXhwb3J0IGludGVyZmFjZSBXZWJHTExvc2VDb250ZXh0RXh0ZW5zaW9uIHsgbG9zZUNvbnRleHQoKTogdm9pZDsgfVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlV2ViR0xSZW5kZXJpbmdDb250ZXh0KGF0dHJpYnV0ZXM6IFdlYkdMQ29udGV4dEF0dHJpYnV0ZXMpOlxuICAgIFdlYkdMUmVuZGVyaW5nQ29udGV4dCB7XG4gIGNvbnN0IGNhbnZhcyA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2NhbnZhcycpO1xuICBjYW52YXMud2lkdGggPSAxO1xuICBjYW52YXMuaGVpZ2h0ID0gMTtcbiAgcmV0dXJuIGNyZWF0ZVdlYkdMUmVuZGVyaW5nQ29udGV4dEZyb21DYW52YXMoY2FudmFzLCBhdHRyaWJ1dGVzKTtcbn1cblxuLyoqXG4gKiBGb3JjZSB0aGUgbGlicmFyeSB0byBwcmVmZXIgV2ViR0wgMS4wIGluc3RlYWQgb2YgV2ViR0wgMi4wIGV2ZW4gd2hlbiBXZWJHTFxuICogMi4wIGlzIGF2YWlsYWJsZS5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHByZWZlcldlYkdMMSgpIHtcbiAgVVNFX1dFQkdMMl9XSEVOX0FWQUlMQUJMRSA9IGZhbHNlO1xuICBXRUJHTDJfRU5BQkxFRCA9IG51bGw7XG59XG5cbi8qKlxuICogUHJlZmVyIFdlYkdMIDIuMCB0byBXZWJHTCAxLjAuIFRoaXMgaXMgdGhlIGRlZmF1bHQgY29uZmlndXJhdGlvbi5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHByZWZlcldlYkdMMigpIHtcbiAgVVNFX1dFQkdMMl9XSEVOX0FWQUlMQUJMRSA9IHRydWU7XG4gIFdFQkdMMl9FTkFCTEVEID0gbnVsbDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGlzV2ViR0wyRW5hYmxlZCgpIHtcbiAgaWYgKCFVU0VfV0VCR0wyX1dIRU5fQVZBSUxBQkxFKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG5cbiAgaWYgKFdFQkdMMl9FTkFCTEVEID09IG51bGwpIHtcbiAgICBjb25zdCB0ZW1wQ2FudmFzID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnY2FudmFzJyk7XG4gICAgY29uc3QgZ2wgPSB0ZW1wQ2FudmFzLmdldENvbnRleHQoJ3dlYmdsMicpO1xuICAgIGlmIChnbCAhPSBudWxsKSB7XG4gICAgICBXRUJHTDJfRU5BQkxFRCA9IHRydWU7XG5cbiAgICAgIGNvbnN0IGxvc2VDb250ZXh0RXh0ZW5zaW9uID1cbiAgICAgICAgICBnZXRFeHRlbnNpb25PclRocm93KFxuICAgICAgICAgICAgICBnbCBhcyBXZWJHTFJlbmRlcmluZ0NvbnRleHQsICdXRUJHTF9sb3NlX2NvbnRleHQnKSBhc1xuICAgICAgICAgIFdlYkdMTG9zZUNvbnRleHRFeHRlbnNpb247XG4gICAgICBsb3NlQ29udGV4dEV4dGVuc2lvbi5sb3NlQ29udGV4dCgpO1xuICAgIH0gZWxzZSB7XG4gICAgICBXRUJHTDJfRU5BQkxFRCA9IGZhbHNlO1xuICAgIH1cbiAgfVxuICByZXR1cm4gV0VCR0wyX0VOQUJMRUQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVXZWJHTFJlbmRlcmluZ0NvbnRleHRGcm9tQ2FudmFzKFxuICAgIGNhbnZhczogSFRNTENhbnZhc0VsZW1lbnQsXG4gICAgYXR0cmlidXRlczogV2ViR0xDb250ZXh0QXR0cmlidXRlcyk6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCB7XG4gIGxldCBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0O1xuICBpZiAoaXNXZWJHTDJFbmFibGVkKCkpIHtcbiAgICBnbCA9IGNhbnZhcy5nZXRDb250ZXh0KCd3ZWJnbDInLCBhdHRyaWJ1dGVzKSBhcyBXZWJHTFJlbmRlcmluZ0NvbnRleHQ7XG4gIH0gZWxzZSB7XG4gICAgZ2wgPSAoY2FudmFzLmdldENvbnRleHQoJ3dlYmdsJywgYXR0cmlidXRlcykgfHxcbiAgICAgICAgICBjYW52YXMuZ2V0Q29udGV4dCgnZXhwZXJpbWVudGFsLXdlYmdsJywgYXR0cmlidXRlcykpIGFzXG4gICAgICAgIFdlYkdMUmVuZGVyaW5nQ29udGV4dDtcbiAgfVxuXG4gIGlmIChnbCA9PSBudWxsKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdUaGlzIGJyb3dzZXIgZG9lcyBub3Qgc3VwcG9ydCBXZWJHTC4nKTtcbiAgfVxuICByZXR1cm4gZ2w7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjYWxsQW5kQ2hlY2s8VD4oZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZnVuYzogKCkgPT4gVCk6IFQge1xuICBjb25zdCByZXR1cm5WYWx1ZSA9IGZ1bmMoKTtcbiAgY2hlY2tXZWJHTEVycm9yKGdsKTtcbiAgcmV0dXJuIHJldHVyblZhbHVlO1xufVxuXG5sZXQgd2ViR0xEZWJ1Z0Vycm9yQ2hlY2tpbmdFbmFibGVkID0gZmFsc2U7XG5cbmV4cG9ydCBmdW5jdGlvbiBlbmFibGVEZWJ1Z1dlYkdMRXJyb3JDaGVja2luZyhlbmFibGVkOiBib29sZWFuKSB7XG4gIHdlYkdMRGVidWdFcnJvckNoZWNraW5nRW5hYmxlZCA9IGVuYWJsZWQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjaGVja1dlYkdMRXJyb3IoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCkge1xuICBpZiAod2ViR0xEZWJ1Z0Vycm9yQ2hlY2tpbmdFbmFibGVkKSB7XG4gICAgY29uc3QgZXJyb3IgPSBnbC5nZXRFcnJvcigpO1xuICAgIGlmIChlcnJvciAhPT0gZ2wuTk9fRVJST1IpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignV2ViR0wgRXJyb3I6ICcgKyBnZXRXZWJHTEVycm9yTWVzc2FnZShnbCwgZXJyb3IpKTtcbiAgICB9XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFdlYkdMRXJyb3JNZXNzYWdlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHN0YXR1czogbnVtYmVyKTogc3RyaW5nIHtcbiAgc3dpdGNoIChzdGF0dXMpIHtcbiAgICBjYXNlIGdsLk5PX0VSUk9SOlxuICAgICAgcmV0dXJuICdOT19FUlJPUic7XG4gICAgY2FzZSBnbC5JTlZBTElEX0VOVU06XG4gICAgICByZXR1cm4gJ0lOVkFMSURfRU5VTSc7XG4gICAgY2FzZSBnbC5JTlZBTElEX1ZBTFVFOlxuICAgICAgcmV0dXJuICdJTlZBTElEX1ZBTFVFJztcbiAgICBjYXNlIGdsLklOVkFMSURfT1BFUkFUSU9OOlxuICAgICAgcmV0dXJuICdJTlZBTElEX09QRVJBVElPTic7XG4gICAgY2FzZSBnbC5JTlZBTElEX0ZSQU1FQlVGRkVSX09QRVJBVElPTjpcbiAgICAgIHJldHVybiAnSU5WQUxJRF9GUkFNRUJVRkZFUl9PUEVSQVRJT04nO1xuICAgIGNhc2UgZ2wuT1VUX09GX01FTU9SWTpcbiAgICAgIHJldHVybiAnT1VUX09GX01FTU9SWSc7XG4gICAgY2FzZSBnbC5DT05URVhUX0xPU1RfV0VCR0w6XG4gICAgICByZXR1cm4gJ0NPTlRFWFRfTE9TVF9XRUJHTCc7XG4gICAgZGVmYXVsdDpcbiAgICAgIHJldHVybiAnVW5rbm93biBlcnJvciBjb2RlICcgKyBzdGF0dXM7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldEV4dGVuc2lvbk9yVGhyb3coXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZXh0ZW5zaW9uTmFtZTogc3RyaW5nKToge30ge1xuICByZXR1cm4gdGhyb3dJZk51bGw8e30+KFxuICAgICAgZ2wsICgpID0+IGdsLmdldEV4dGVuc2lvbihleHRlbnNpb25OYW1lKSxcbiAgICAgICdFeHRlbnNpb24gXCInICsgZXh0ZW5zaW9uTmFtZSArICdcIiBub3Qgc3VwcG9ydGVkIG9uIHRoaXMgYnJvd3Nlci4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVZlcnRleFNoYWRlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB2ZXJ0ZXhTaGFkZXJTb3VyY2U6IHN0cmluZyk6IFdlYkdMU2hhZGVyIHtcbiAgY29uc3QgdmVydGV4U2hhZGVyOiBXZWJHTFNoYWRlciA9IHRocm93SWZOdWxsPFdlYkdMU2hhZGVyPihcbiAgICAgIGdsLCAoKSA9PiBnbC5jcmVhdGVTaGFkZXIoZ2wuVkVSVEVYX1NIQURFUiksXG4gICAgICAnVW5hYmxlIHRvIGNyZWF0ZSB2ZXJ0ZXggV2ViR0xTaGFkZXIuJyk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuc2hhZGVyU291cmNlKHZlcnRleFNoYWRlciwgdmVydGV4U2hhZGVyU291cmNlKSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuY29tcGlsZVNoYWRlcih2ZXJ0ZXhTaGFkZXIpKTtcbiAgaWYgKGdsLmdldFNoYWRlclBhcmFtZXRlcih2ZXJ0ZXhTaGFkZXIsIGdsLkNPTVBJTEVfU1RBVFVTKSA9PT0gZmFsc2UpIHtcbiAgICBjb25zb2xlLmxvZyhnbC5nZXRTaGFkZXJJbmZvTG9nKHZlcnRleFNoYWRlcikpO1xuICAgIHRocm93IG5ldyBFcnJvcignRmFpbGVkIHRvIGNvbXBpbGUgdmVydGV4IHNoYWRlci4nKTtcbiAgfVxuICByZXR1cm4gdmVydGV4U2hhZGVyO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlRnJhZ21lbnRTaGFkZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZnJhZ21lbnRTaGFkZXJTb3VyY2U6IHN0cmluZyk6IFdlYkdMU2hhZGVyIHtcbiAgY29uc3QgZnJhZ21lbnRTaGFkZXI6IFdlYkdMU2hhZGVyID0gdGhyb3dJZk51bGw8V2ViR0xTaGFkZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZVNoYWRlcihnbC5GUkFHTUVOVF9TSEFERVIpLFxuICAgICAgJ1VuYWJsZSB0byBjcmVhdGUgZnJhZ21lbnQgV2ViR0xTaGFkZXIuJyk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuc2hhZGVyU291cmNlKGZyYWdtZW50U2hhZGVyLCBmcmFnbWVudFNoYWRlclNvdXJjZSkpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmNvbXBpbGVTaGFkZXIoZnJhZ21lbnRTaGFkZXIpKTtcbiAgaWYgKGdsLmdldFNoYWRlclBhcmFtZXRlcihmcmFnbWVudFNoYWRlciwgZ2wuQ09NUElMRV9TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGNvbnNvbGUubG9nKGdsLmdldFNoYWRlckluZm9Mb2coZnJhZ21lbnRTaGFkZXIpKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0ZhaWxlZCB0byBjb21waWxlIGZyYWdtZW50IHNoYWRlci4nKTtcbiAgfVxuICByZXR1cm4gZnJhZ21lbnRTaGFkZXI7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVQcm9ncmFtKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpOiBXZWJHTFByb2dyYW0ge1xuICByZXR1cm4gdGhyb3dJZk51bGw8V2ViR0xQcm9ncmFtPihcbiAgICAgIGdsLCAoKSA9PiBnbC5jcmVhdGVQcm9ncmFtKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMUHJvZ3JhbS4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGxpbmtQcm9ncmFtKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSkge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmxpbmtQcm9ncmFtKHByb2dyYW0pKTtcbiAgaWYgKGdsLmdldFByb2dyYW1QYXJhbWV0ZXIocHJvZ3JhbSwgZ2wuTElOS19TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGNvbnNvbGUubG9nKGdsLmdldFByb2dyYW1JbmZvTG9nKHByb2dyYW0pKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0ZhaWxlZCB0byBsaW5rIHZlcnRleCBhbmQgZnJhZ21lbnQgc2hhZGVycy4nKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gdmFsaWRhdGVQcm9ncmFtKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSkge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnZhbGlkYXRlUHJvZ3JhbShwcm9ncmFtKSk7XG4gIGlmIChnbC5nZXRQcm9ncmFtUGFyYW1ldGVyKHByb2dyYW0sIGdsLlZBTElEQVRFX1NUQVRVUykgPT09IGZhbHNlKSB7XG4gICAgY29uc29sZS5sb2coZ2wuZ2V0UHJvZ3JhbUluZm9Mb2cocHJvZ3JhbSkpO1xuICAgIHRocm93IG5ldyBFcnJvcignU2hhZGVyIHByb2dyYW0gdmFsaWRhdGlvbiBmYWlsZWQuJyk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVN0YXRpY1ZlcnRleEJ1ZmZlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBkYXRhOiBGbG9hdDMyQXJyYXkpOiBXZWJHTEJ1ZmZlciB7XG4gIGNvbnN0IGJ1ZmZlcjogV2ViR0xCdWZmZXIgPSB0aHJvd0lmTnVsbDxXZWJHTEJ1ZmZlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlQnVmZmVyKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMQnVmZmVyJyk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5BUlJBWV9CVUZGRVIsIGJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJ1ZmZlckRhdGEoZ2wuQVJSQVlfQlVGRkVSLCBkYXRhLCBnbC5TVEFUSUNfRFJBVykpO1xuICByZXR1cm4gYnVmZmVyO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlU3RhdGljSW5kZXhCdWZmZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZGF0YTogVWludDE2QXJyYXkpOiBXZWJHTEJ1ZmZlciB7XG4gIGNvbnN0IGJ1ZmZlcjogV2ViR0xCdWZmZXIgPSB0aHJvd0lmTnVsbDxXZWJHTEJ1ZmZlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlQnVmZmVyKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMQnVmZmVyJyk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5FTEVNRU5UX0FSUkFZX0JVRkZFUiwgYnVmZmVyKSk7XG4gIGNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC5idWZmZXJEYXRhKGdsLkVMRU1FTlRfQVJSQVlfQlVGRkVSLCBkYXRhLCBnbC5TVEFUSUNfRFJBVykpO1xuICByZXR1cm4gYnVmZmVyO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gcXVlcnlNYXhUZXh0dXJlU2l6ZShnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogbnVtYmVyIHtcbiAgaWYgKE1BWF9URVhUVVJFX1NJWkUgIT0gbnVsbCkge1xuICAgIHJldHVybiBNQVhfVEVYVFVSRV9TSVpFO1xuICB9XG4gIE1BWF9URVhUVVJFX1NJWkUgPVxuICAgICAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbCEuZ2V0UGFyYW1ldGVyKGdsIS5NQVhfVEVYVFVSRV9TSVpFKSk7XG4gIHJldHVybiBNQVhfVEVYVFVSRV9TSVpFO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0Q2hhbm5lbHNQZXJUZXh0dXJlKCk6IG51bWJlciB7XG4gIGlmIChpc1dlYkdMMkVuYWJsZWQoKSkge1xuICAgIHJldHVybiAxO1xuICB9XG4gIHJldHVybiA0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlVGV4dHVyZShnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogV2ViR0xUZXh0dXJlIHtcbiAgcmV0dXJuIHRocm93SWZOdWxsPFdlYkdMVGV4dHVyZT4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlVGV4dHVyZSgpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTFRleHR1cmUuJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB2YWxpZGF0ZVRleHR1cmVTaXplKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyKSB7XG4gIGNvbnN0IG1heFRleHR1cmVTaXplOiBudW1iZXIgPSBxdWVyeU1heFRleHR1cmVTaXplKGdsKTtcbiAgaWYgKCh3aWR0aCA8PSAwKSB8fCAoaGVpZ2h0IDw9IDApKSB7XG4gICAgY29uc3QgcmVxdWVzdGVkID0gJ1snICsgd2lkdGggKyAneCcgKyBoZWlnaHQgKyAnXSc7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdSZXF1ZXN0ZWQgdGV4dHVyZSBzaXplICcgKyByZXF1ZXN0ZWQgKyAnIGlzIGludmFsaWQuJyk7XG4gIH1cbiAgaWYgKCh3aWR0aCA+IG1heFRleHR1cmVTaXplKSB8fCAoaGVpZ2h0ID4gbWF4VGV4dHVyZVNpemUpKSB7XG4gICAgY29uc3QgcmVxdWVzdGVkID0gJ1snICsgd2lkdGggKyAneCcgKyBoZWlnaHQgKyAnXSc7XG4gICAgY29uc3QgbWF4ID0gJ1snICsgbWF4VGV4dHVyZVNpemUgKyAneCcgKyBtYXhUZXh0dXJlU2l6ZSArICddJztcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdSZXF1ZXN0ZWQgdGV4dHVyZSBzaXplICcgKyByZXF1ZXN0ZWQgK1xuICAgICAgICAnIGdyZWF0ZXIgdGhhbiBXZWJHTCBtYXhpbXVtIG9uIHRoaXMgYnJvd3NlciAvIEdQVSAnICsgbWF4ICsgJy4nKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlRnJhbWVidWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMRnJhbWVidWZmZXIge1xuICByZXR1cm4gdGhyb3dJZk51bGw8V2ViR0xGcmFtZWJ1ZmZlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlRnJhbWVidWZmZXIoKSwgJ1VuYWJsZSB0byBjcmVhdGUgV2ViR0xGcmFtZWJ1ZmZlci4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRWZXJ0ZXhCdWZmZXJUb1Byb2dyYW1BdHRyaWJ1dGUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcHJvZ3JhbTogV2ViR0xQcm9ncmFtLCBhdHRyaWJ1dGU6IHN0cmluZyxcbiAgICBidWZmZXI6IFdlYkdMQnVmZmVyLCBhcnJheUVudHJpZXNQZXJJdGVtOiBudW1iZXIsIGl0ZW1TdHJpZGVJbkJ5dGVzOiBudW1iZXIsXG4gICAgaXRlbU9mZnNldEluQnl0ZXM6IG51bWJlcikge1xuICBjb25zdCBsb2MgPSBnbC5nZXRBdHRyaWJMb2NhdGlvbihwcm9ncmFtLCBhdHRyaWJ1dGUpO1xuICBpZiAobG9jID09PSAtMSkge1xuICAgIGNvbnN0IGVycm9yID0gbmV3IEVycm9yKFxuICAgICAgICAnVW5hYmxlIHRvIGdldCBhdHRyaWJ1dGUgXCInICsgYXR0cmlidXRlICsgJ1wiIG9uIFdlYkdMUHJvZ3JhbS4nKTtcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgKGVycm9yIGFzIGFueSkubmFtZWRWZXJ0ZXhBdHRyaWJ1dGVOb3RGb3VuZCA9IGF0dHJpYnV0ZTtcbiAgICB0aHJvdyBlcnJvcjtcbiAgfVxuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuQVJSQVlfQlVGRkVSLCBidWZmZXIpKTtcbiAgY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsXG4gICAgICAoKSA9PiBnbC52ZXJ0ZXhBdHRyaWJQb2ludGVyKFxuICAgICAgICAgIGxvYywgYXJyYXlFbnRyaWVzUGVySXRlbSwgZ2wuRkxPQVQsIGZhbHNlLCBpdGVtU3RyaWRlSW5CeXRlcyxcbiAgICAgICAgICBpdGVtT2Zmc2V0SW5CeXRlcykpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmVuYWJsZVZlcnRleEF0dHJpYkFycmF5KGxvYykpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZFRleHR1cmVVbml0KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgdGV4dHVyZVVuaXQ6IG51bWJlcikge1xuICB2YWxpZGF0ZVRleHR1cmVVbml0KGdsLCB0ZXh0dXJlVW5pdCk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYWN0aXZlVGV4dHVyZShnbC5URVhUVVJFMCArIHRleHR1cmVVbml0KSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdW5iaW5kVGV4dHVyZVVuaXQoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZVVuaXQ6IG51bWJlcikge1xuICB2YWxpZGF0ZVRleHR1cmVVbml0KGdsLCB0ZXh0dXJlVW5pdCk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYWN0aXZlVGV4dHVyZShnbC5URVhUVVJFMCArIHRleHR1cmVVbml0KSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgbnVsbCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0UHJvZ3JhbVVuaWZvcm1Mb2NhdGlvbk9yVGhyb3coXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcHJvZ3JhbTogV2ViR0xQcm9ncmFtLFxuICAgIHVuaWZvcm1OYW1lOiBzdHJpbmcpOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbiB7XG4gIHJldHVybiB0aHJvd0lmTnVsbDxXZWJHTFVuaWZvcm1Mb2NhdGlvbj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKHByb2dyYW0sIHVuaWZvcm1OYW1lKSxcbiAgICAgICd1bmlmb3JtIFwiJyArIHVuaWZvcm1OYW1lICsgJ1wiIG5vdCBwcmVzZW50IGluIHByb2dyYW0uJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBiaW5kVGV4dHVyZVRvUHJvZ3JhbVVuaWZvcm1TYW1wbGVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgIHVuaWZvcm1TYW1wbGVyTmFtZTogc3RyaW5nLCB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gYmluZFRleHR1cmVVbml0KGdsLCB0ZXh0dXJlLCB0ZXh0dXJlVW5pdCkpO1xuICBjb25zdCBzYW1wbGVyTG9jYXRpb24gPVxuICAgICAgZ2V0UHJvZ3JhbVVuaWZvcm1Mb2NhdGlvbk9yVGhyb3coZ2wsIHByb2dyYW0sIHVuaWZvcm1TYW1wbGVyTmFtZSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wudW5pZm9ybTFpKHNhbXBsZXJMb2NhdGlvbiwgdGV4dHVyZVVuaXQpKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRDYW52YXNUb0ZyYW1lYnVmZmVyKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIG51bGwpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC52aWV3cG9ydCgwLCAwLCBnbC5jYW52YXMud2lkdGgsIGdsLmNhbnZhcy5oZWlnaHQpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5zY2lzc29yKDAsIDAsIGdsLmNhbnZhcy53aWR0aCwgZ2wuY2FudmFzLmhlaWdodCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZENvbG9yVGV4dHVyZVRvRnJhbWVidWZmZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgIGZyYW1lYnVmZmVyOiBXZWJHTEZyYW1lYnVmZmVyKSB7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBmcmFtZWJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soXG4gICAgICBnbCxcbiAgICAgICgpID0+IGdsLmZyYW1lYnVmZmVyVGV4dHVyZTJEKFxuICAgICAgICAgIGdsLkZSQU1FQlVGRkVSLCBnbC5DT0xPUl9BVFRBQ0hNRU5UMCwgZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSwgMCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdW5iaW5kQ29sb3JUZXh0dXJlRnJvbUZyYW1lYnVmZmVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGZyYW1lYnVmZmVyOiBXZWJHTEZyYW1lYnVmZmVyKSB7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBmcmFtZWJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soXG4gICAgICBnbCxcbiAgICAgICgpID0+IGdsLmZyYW1lYnVmZmVyVGV4dHVyZTJEKFxuICAgICAgICAgIGdsLkZSQU1FQlVGRkVSLCBnbC5DT0xPUl9BVFRBQ0hNRU5UMCwgZ2wuVEVYVFVSRV8yRCwgbnVsbCwgMCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdmFsaWRhdGVGcmFtZWJ1ZmZlcihnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KSB7XG4gIGNvbnN0IHN0YXR1cyA9IGdsLmNoZWNrRnJhbWVidWZmZXJTdGF0dXMoZ2wuRlJBTUVCVUZGRVIpO1xuICBpZiAoc3RhdHVzICE9PSBnbC5GUkFNRUJVRkZFUl9DT01QTEVURSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ0Vycm9yIGJpbmRpbmcgZnJhbWVidWZmZXI6ICcgKyBnZXRGcmFtZWJ1ZmZlckVycm9yTWVzc2FnZShnbCwgc3RhdHVzKSk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldEZyYW1lYnVmZmVyRXJyb3JNZXNzYWdlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHN0YXR1czogbnVtYmVyKTogc3RyaW5nIHtcbiAgc3dpdGNoIChzdGF0dXMpIHtcbiAgICBjYXNlIGdsLkZSQU1FQlVGRkVSX0lOQ09NUExFVEVfQVRUQUNITUVOVDpcbiAgICAgIHJldHVybiAnRlJBTUVCVUZGRVJfSU5DT01QTEVURV9BVFRBQ0hNRU5UJztcbiAgICBjYXNlIGdsLkZSQU1FQlVGRkVSX0lOQ09NUExFVEVfTUlTU0lOR19BVFRBQ0hNRU5UOlxuICAgICAgcmV0dXJuICdGUkFNRUJVRkZFUl9JTkNPTVBMRVRFX01JU1NJTkdfQVRUQUNITUVOVCc7XG4gICAgY2FzZSBnbC5GUkFNRUJVRkZFUl9JTkNPTVBMRVRFX0RJTUVOU0lPTlM6XG4gICAgICByZXR1cm4gJ0ZSQU1FQlVGRkVSX0lOQ09NUExFVEVfRElNRU5TSU9OUyc7XG4gICAgY2FzZSBnbC5GUkFNRUJVRkZFUl9VTlNVUFBPUlRFRDpcbiAgICAgIHJldHVybiAnRlJBTUVCVUZGRVJfVU5TVVBQT1JURUQnO1xuICAgIGRlZmF1bHQ6XG4gICAgICByZXR1cm4gJ3Vua25vd24gZXJyb3IgJyArIHN0YXR1cztcbiAgfVxufVxuXG5mdW5jdGlvbiB0aHJvd0lmTnVsbDxUPihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCByZXR1cm5UT3JOdWxsOiAoKSA9PiBUIHwgbnVsbCxcbiAgICBmYWlsdXJlTWVzc2FnZTogc3RyaW5nKTogVCB7XG4gIGNvbnN0IHRPck51bGw6IFR8bnVsbCA9IGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gcmV0dXJuVE9yTnVsbCgpKTtcbiAgaWYgKHRPck51bGwgPT0gbnVsbCkge1xuICAgIHRocm93IG5ldyBFcnJvcihmYWlsdXJlTWVzc2FnZSk7XG4gIH1cbiAgcmV0dXJuIHRPck51bGwgYXMgVDtcbn1cblxuZnVuY3Rpb24gdmFsaWRhdGVUZXh0dXJlVW5pdChnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gIGNvbnN0IG1heFRleHR1cmVVbml0ID0gZ2wuTUFYX0NPTUJJTkVEX1RFWFRVUkVfSU1BR0VfVU5JVFMgLSAxO1xuICBjb25zdCBnbFRleHR1cmVVbml0ID0gdGV4dHVyZVVuaXQgKyBnbC5URVhUVVJFMDtcbiAgaWYgKGdsVGV4dHVyZVVuaXQgPCBnbC5URVhUVVJFMCB8fCBnbFRleHR1cmVVbml0ID4gbWF4VGV4dHVyZVVuaXQpIHtcbiAgICBjb25zdCB0ZXh0dXJlVW5pdFJhbmdlID0gJ1tnbC5URVhUVVJFMCwgZ2wuVEVYVFVSRScgKyBtYXhUZXh0dXJlVW5pdCArICddJztcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ3RleHR1cmVVbml0IG11c3QgYmUgaW4gJyArIHRleHR1cmVVbml0UmFuZ2UgKyAnLicpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRUZXh0dXJlU2hhcGVGcm9tTG9naWNhbFNoYXBlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGxvZ2ljYWxTaGFwZTogbnVtYmVyW10sXG4gICAgcHJlZmVycmVkVGV4U2hhcGU/OiBbbnVtYmVyLCBudW1iZXJdKTogW251bWJlciwgbnVtYmVyXSB7XG4gIGNvbnN0IG1heFRleFNpemUgPSBxdWVyeU1heFRleHR1cmVTaXplKGdsKTtcbiAgY29uc3Qgc2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZShsb2dpY2FsU2hhcGUpO1xuICBpZiAocHJlZmVycmVkVGV4U2hhcGUgIT0gbnVsbCkge1xuICAgIGNvbnN0IHNpemVQcmVmZXJyZWQgPSB1dGlsLnNpemVGcm9tU2hhcGUocHJlZmVycmVkVGV4U2hhcGUpO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBzaXplID09PSBzaXplUHJlZmVycmVkLFxuICAgICAgICBgU2l6ZSBvZiBzaGFwZSAoJHtzaXplfSkgbXVzdCBtYXRjaCBzaXplIG9mIGAgK1xuICAgICAgICAgICAgYHByZWZlcnJlZFNoYXBlICgke3NpemVQcmVmZXJyZWR9KWApO1xuICAgIGlmIChwcmVmZXJyZWRUZXhTaGFwZVswXSA8PSBtYXhUZXhTaXplICYmXG4gICAgICAgIHByZWZlcnJlZFRleFNoYXBlWzFdIDw9IG1heFRleFNpemUpIHtcbiAgICAgIHJldHVybiBwcmVmZXJyZWRUZXhTaGFwZTtcbiAgICB9XG4gIH1cblxuICBpZiAobG9naWNhbFNoYXBlLmxlbmd0aCA8PSAxICYmIHNpemUgPD0gbWF4VGV4U2l6ZSkge1xuICAgIHJldHVybiBbc2l6ZSwgMV07XG4gIH0gZWxzZSBpZiAoXG4gICAgICBsb2dpY2FsU2hhcGUubGVuZ3RoID09PSAyICYmIGxvZ2ljYWxTaGFwZVswXSA8PSBtYXhUZXhTaXplICYmXG4gICAgICBsb2dpY2FsU2hhcGVbMV0gPD0gbWF4VGV4U2l6ZSkge1xuICAgIHJldHVybiBsb2dpY2FsU2hhcGUgYXMgW251bWJlciwgbnVtYmVyXTtcbiAgfSBlbHNlIGlmIChcbiAgICAgIGxvZ2ljYWxTaGFwZS5sZW5ndGggPT09IDMgJiYgbG9naWNhbFNoYXBlWzBdIDw9IG1heFRleFNpemUgJiZcbiAgICAgIGxvZ2ljYWxTaGFwZVsxXSAqIGxvZ2ljYWxTaGFwZVsyXSA8PSBtYXhUZXhTaXplKSB7XG4gICAgcmV0dXJuIFtsb2dpY2FsU2hhcGVbMF0sIGxvZ2ljYWxTaGFwZVsxXSAqIGxvZ2ljYWxTaGFwZVsyXV07XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIHV0aWwuc2l6ZVRvU3F1YXJpc2hTaGFwZShzaXplKTtcbiAgfVxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5leHBvcnQgZnVuY3Rpb24gZXhwZWN0QXJyYXlzQ2xvc2UoXG4gICAgYWN0dWFsOiBGbG9hdDMyQXJyYXksIGV4cGVjdGVkOiBGbG9hdDMyQXJyYXksIGVwc2lsb246IG51bWJlcikge1xuICBpZiAoYWN0dWFsLmxlbmd0aCAhPT0gZXhwZWN0ZWQubGVuZ3RoKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnTWF0cmljZXMgaGF2ZSBkaWZmZXJlbnQgbGVuZ3RocyAoJyArIGFjdHVhbC5sZW5ndGggKyAnIHZzICcgK1xuICAgICAgICBleHBlY3RlZC5sZW5ndGggKyAnKS4nKTtcbiAgfVxuICBmb3IgKGxldCBpID0gMDsgaSA8IGV4cGVjdGVkLmxlbmd0aDsgKytpKSB7XG4gICAgY29uc3QgYSA9IGFjdHVhbFtpXTtcbiAgICBjb25zdCBlID0gZXhwZWN0ZWRbaV07XG4gICAgaWYgKGlzTmFOKGEpICYmIGlzTmFOKGUpKSB7XG4gICAgICBjb250aW51ZTtcbiAgICB9XG4gICAgaWYgKGlzTmFOKGEpIHx8IGlzTmFOKGUpIHx8IE1hdGguYWJzKGEgLSBlKSA+IGVwc2lsb24pIHtcbiAgICAgIGNvbnN0IGFjdHVhbFN0ciA9ICdhY3R1YWxbJyArIGkgKyAnXSA9PT0gJyArIGE7XG4gICAgICBjb25zdCBleHBlY3RlZFN0ciA9ICdleHBlY3RlZFsnICsgaSArICddID09PSAnICsgZTtcbiAgICAgIHRocm93IG5ldyBFcnJvcignQXJyYXlzIGRpZmZlcjogJyArIGFjdHVhbFN0ciArICcsICcgKyBleHBlY3RlZFN0cik7XG4gICAgfVxuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiByYW5kb21BcnJheUluUmFuZ2UoXG4gICAgbjogbnVtYmVyLCBtaW5WYWx1ZTogbnVtYmVyLCBtYXhWYWx1ZTogbnVtYmVyKTogRmxvYXQzMkFycmF5IHtcbiAgY29uc3QgdiA9IG5ldyBGbG9hdDMyQXJyYXkobik7XG4gIGNvbnN0IHJhbmdlID0gbWF4VmFsdWUgLSBtaW5WYWx1ZTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBuOyArK2kpIHtcbiAgICB2W2ldID0gKE1hdGgucmFuZG9tKCkgKiByYW5nZSkgKyBtaW5WYWx1ZTtcbiAgfVxuICByZXR1cm4gdjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIG1ha2VJZGVudGl0eShuOiBudW1iZXIpOiBGbG9hdDMyQXJyYXkge1xuICBjb25zdCBpID0gbmV3IEZsb2F0MzJBcnJheShuICogbik7XG4gIGZvciAobGV0IGogPSAwOyBqIDwgbjsgKytqKSB7XG4gICAgaVsoaiAqIG4pICsgal0gPSAxO1xuICB9XG4gIHJldHVybiBpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gc2V0VmFsdWUoXG4gICAgbTogRmxvYXQzMkFycmF5LCBtTnVtUm93czogbnVtYmVyLCBtTnVtQ29sczogbnVtYmVyLCB2OiBudW1iZXIsIHJvdzogbnVtYmVyLFxuICAgIGNvbHVtbjogbnVtYmVyKSB7XG4gIGlmIChyb3cgPj0gbU51bVJvd3MpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ3JvdyAoJyArIHJvdyArICcpIG11c3QgYmUgaW4gWzAgJyArIG1OdW1Sb3dzICsgJ10uJyk7XG4gIH1cbiAgaWYgKGNvbHVtbiA+PSBtTnVtQ29scykge1xuICAgIHRocm93IG5ldyBFcnJvcignY29sdW1uICgnICsgY29sdW1uICsgJykgbXVzdCBiZSBpbiBbMCAnICsgbU51bUNvbHMgKyAnXS4nKTtcbiAgfVxuICBtWyhyb3cgKiBtTnVtQ29scykgKyBjb2x1bW5dID0gdjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNwdU11bHRpcGx5TWF0cml4KFxuICAgIGE6IEZsb2F0MzJBcnJheSwgYVJvdzogbnVtYmVyLCBhQ29sOiBudW1iZXIsIGI6IEZsb2F0MzJBcnJheSwgYlJvdzogbnVtYmVyLFxuICAgIGJDb2w6IG51bWJlcikge1xuICBjb25zdCByZXN1bHQgPSBuZXcgRmxvYXQzMkFycmF5KGFSb3cgKiBiQ29sKTtcbiAgZm9yIChsZXQgciA9IDA7IHIgPCBhUm93OyArK3IpIHtcbiAgICBmb3IgKGxldCBjID0gMDsgYyA8IGJDb2w7ICsrYykge1xuICAgICAgbGV0IGQgPSAwO1xuICAgICAgZm9yIChsZXQgayA9IDA7IGsgPCBhQ29sOyArK2spIHtcbiAgICAgICAgZCArPSBhWyhyICogYUNvbCkgKyBrXSAqIGJbKGsgKiBiQ29sKSArIGNdO1xuICAgICAgfVxuICAgICAgcmVzdWx0WyhyICogYkNvbCkgKyBjXSA9IGQ7XG4gICAgfVxuICB9XG4gIHJldHVybiByZXN1bHQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcHVEb3RQcm9kdWN0KGE6IEZsb2F0MzJBcnJheSwgYjogRmxvYXQzMkFycmF5KTogbnVtYmVyIHtcbiAgaWYgKGEubGVuZ3RoICE9PSBiLmxlbmd0aCkge1xuICAgIHRocm93IG5ldyBFcnJvcignY3B1RG90UHJvZHVjdDogaW5jb21wYXRpYmxlIHZlY3RvcnMuJyk7XG4gIH1cbiAgbGV0IGQgPSAwO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IGEubGVuZ3RoOyArK2kpIHtcbiAgICBkICs9IGFbaV0gKiBiW2ldO1xuICB9XG4gIHJldHVybiBkO1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5leHBvcnQgdHlwZSBWZWN0b3IgPSBudW1iZXJbXSB8IEZsb2F0NjRBcnJheSB8IEZsb2F0MzJBcnJheSB8IEludDMyQXJyYXkgfFxuICAgIEludDhBcnJheSB8IEludDE2QXJyYXk7XG5cbi8qKiBTaHVmZmxlcyB0aGUgYXJyYXkgdXNpbmcgRmlzaGVyLVlhdGVzIGFsZ29yaXRobS4gKi9cbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbmV4cG9ydCBmdW5jdGlvbiBzaHVmZmxlKGFycmF5OiBhbnlbXXxVaW50MzJBcnJheXxJbnQzMkFycmF5fFxuICAgICAgICAgICAgICAgICAgICAgICAgRmxvYXQzMkFycmF5KTogdm9pZCB7XG4gIGxldCBjb3VudGVyID0gYXJyYXkubGVuZ3RoO1xuICBsZXQgdGVtcCA9IDA7XG4gIGxldCBpbmRleCA9IDA7XG4gIC8vIFdoaWxlIHRoZXJlIGFyZSBlbGVtZW50cyBpbiB0aGUgYXJyYXlcbiAgd2hpbGUgKGNvdW50ZXIgPiAwKSB7XG4gICAgLy8gUGljayBhIHJhbmRvbSBpbmRleFxuICAgIGluZGV4ID0gKE1hdGgucmFuZG9tKCkgKiBjb3VudGVyKSB8IDA7XG4gICAgLy8gRGVjcmVhc2UgY291bnRlciBieSAxXG4gICAgY291bnRlci0tO1xuICAgIC8vIEFuZCBzd2FwIHRoZSBsYXN0IGVsZW1lbnQgd2l0aCBpdFxuICAgIHRlbXAgPSBhcnJheVtjb3VudGVyXTtcbiAgICBhcnJheVtjb3VudGVyXSA9IGFycmF5W2luZGV4XTtcbiAgICBhcnJheVtpbmRleF0gPSB0ZW1wO1xuICB9XG59XG5cbi8qKiBDbGFtcHMgYSB2YWx1ZSB0byBhIHNwZWNpZmllZCByYW5nZS4gKi9cbmV4cG9ydCBmdW5jdGlvbiBjbGFtcChtaW46IG51bWJlciwgeDogbnVtYmVyLCBtYXg6IG51bWJlcik6IG51bWJlciB7XG4gIHJldHVybiBNYXRoLm1heChtaW4sIE1hdGgubWluKHgsIG1heCkpO1xufVxuXG4vKiogUmV0dXJucyBhIHNhbXBsZSBmcm9tIGEgdW5pZm9ybSBbYSwgYl0gZGlzdHJpYnV0aW9uLiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHJhbmRVbmlmb3JtKGE6IG51bWJlciwgYjogbnVtYmVyKSB7XG4gIHJldHVybiBNYXRoLnJhbmRvbSgpICogKGIgLSBhKSArIGE7XG59XG5cbi8qKlxuICogU2FtcGxlcyBmcm9tIGEgZ2F1c3NpYW4gZGlzdHJpYnV0aW9uLlxuICpcbiAqIEBwYXJhbSBtZWFuIFRoZSBtZWFuLiBEZWZhdWx0IGlzIDAuXG4gKiBAcGFyYW0gc3RkRGV2IFRoZSBzdGFuZGFyZCBkZXZpYXRpb24uIERlZmF1bHQgaXMgMS5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIHJhbmRHYXVzcyhtZWFuID0gMCwgc3RkRGV2ID0gMSwgdHJ1bmNhdGVkID0gZmFsc2UpOiBudW1iZXIge1xuICBsZXQgdjE6IG51bWJlciwgdjI6IG51bWJlciwgczogbnVtYmVyO1xuICBkbyB7XG4gICAgdjEgPSAyICogTWF0aC5yYW5kb20oKSAtIDE7XG4gICAgdjIgPSAyICogTWF0aC5yYW5kb20oKSAtIDE7XG4gICAgcyA9IHYxICogdjEgKyB2MiAqIHYyO1xuICB9IHdoaWxlIChzID4gMSk7XG5cbiAgY29uc3QgcmVzdWx0ID0gTWF0aC5zcXJ0KC0yICogTWF0aC5sb2cocykgLyBzKSAqIHYxO1xuICBpZiAodHJ1bmNhdGVkICYmIHJlc3VsdCA+IDIpIHtcbiAgICByZXR1cm4gcmFuZEdhdXNzKG1lYW4sIHN0ZERldiwgdHJ1ZSk7XG4gIH1cbiAgcmV0dXJuIG1lYW4gKyBzdGREZXYgKiByZXN1bHQ7XG59XG5cbi8qKiBSZXR1cm5zIHNxdWFyZWQgZXVjbGVkaWFuIGRpc3RhbmNlIGJldHdlZW4gdHdvIHZlY3RvcnMuICovXG5leHBvcnQgZnVuY3Rpb24gZGlzdFNxdWFyZWQoYTogVmVjdG9yLCBiOiBWZWN0b3IpOiBudW1iZXIge1xuICBsZXQgcmVzdWx0ID0gMDtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBhLmxlbmd0aDsgaSsrKSB7XG4gICAgY29uc3QgZGlmZiA9IGFbaV0gLSBiW2ldO1xuICAgIHJlc3VsdCArPSBkaWZmICogZGlmZjtcbiAgfVxuICByZXR1cm4gcmVzdWx0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYXNzZXJ0KGV4cHI6IGJvb2xlYW4sIG1zZzogc3RyaW5nKSB7XG4gIGlmICghZXhwcikge1xuICAgIHRocm93IG5ldyBFcnJvcihtc2cpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBhc3NlcnRTaGFwZXNNYXRjaChcbiAgICBzaGFwZUE6IG51bWJlcltdLCBzaGFwZUI6IG51bWJlcltdLCBlcnJvck1lc3NhZ2VQcmVmaXggPSAnJyk6IHZvaWQge1xuICBhc3NlcnQoXG4gICAgICBhcnJheXNFcXVhbChzaGFwZUEsIHNoYXBlQiksXG4gICAgICBlcnJvck1lc3NhZ2VQcmVmaXggKyBgU2hhcGVzICR7c2hhcGVBfSBhbmQgJHtzaGFwZUJ9IG11c3QgbWF0Y2hgKTtcbn1cblxuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuZXhwb3J0IGZ1bmN0aW9uIGZsYXR0ZW4oYXJyOiBhbnlbXSwgcmV0PzogbnVtYmVyW10pOiBudW1iZXJbXSB7XG4gIHJldCA9IChyZXQgPT09IHVuZGVmaW5lZCA/IFtdIDogcmV0KTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBhcnIubGVuZ3RoOyArK2kpIHtcbiAgICBpZiAoQXJyYXkuaXNBcnJheShhcnJbaV0pKSB7XG4gICAgICBmbGF0dGVuKGFycltpXSwgcmV0KTtcbiAgICB9IGVsc2Uge1xuICAgICAgcmV0LnB1c2goYXJyW2ldKTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIHJldDtcbn1cblxuZXhwb3J0IHR5cGUgQXJyYXlEYXRhID0gbnVtYmVyfG51bWJlcltdfG51bWJlcltdW118bnVtYmVyW11bXVtdfG51bWJlcltdW11bXVtdO1xuXG5leHBvcnQgZnVuY3Rpb24gaW5mZXJTaGFwZShhcnI6IEFycmF5RGF0YSk6IG51bWJlcltdIHtcbiAgY29uc3Qgc2hhcGU6IG51bWJlcltdID0gW107XG4gIHdoaWxlIChhcnIgaW5zdGFuY2VvZiBBcnJheSkge1xuICAgIHNoYXBlLnB1c2goYXJyLmxlbmd0aCk7XG4gICAgYXJyID0gYXJyWzBdO1xuICB9XG4gIHJldHVybiBzaGFwZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHNpemVGcm9tU2hhcGUoc2hhcGU6IG51bWJlcltdKTogbnVtYmVyIHtcbiAgaWYgKHNoYXBlLmxlbmd0aCA9PT0gMCkge1xuICAgIC8vIFNjYWxhci5cbiAgICByZXR1cm4gMTtcbiAgfVxuICBsZXQgc2l6ZSA9IHNoYXBlWzBdO1xuICBmb3IgKGxldCBpID0gMTsgaSA8IHNoYXBlLmxlbmd0aDsgaSsrKSB7XG4gICAgc2l6ZSAqPSBzaGFwZVtpXTtcbiAgfVxuICByZXR1cm4gc2l6ZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGlzU2NhbGFyU2hhcGUoc2hhcGU6IG51bWJlcltdKTogYm9vbGVhbiB7XG4gIHJldHVybiBzaGFwZS5sZW5ndGggPT09IDA7XG59XG5cbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbmV4cG9ydCBmdW5jdGlvbiBhcnJheXNFcXVhbChuMTogYW55W118RmxvYXQzMkFycmF5LCBuMjogYW55W118RmxvYXQzMkFycmF5KSB7XG4gIGlmIChuMS5sZW5ndGggIT09IG4yLmxlbmd0aCkge1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuICBmb3IgKGxldCBpID0gMDsgaSA8IG4xLmxlbmd0aDsgaSsrKSB7XG4gICAgaWYgKG4xW2ldICE9PSBuMltpXSkge1xuICAgICAgcmV0dXJuIGZhbHNlO1xuICAgIH1cbiAgfVxuICByZXR1cm4gdHJ1ZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGlzSW50KGE6IG51bWJlcik6IGJvb2xlYW4ge1xuICByZXR1cm4gYSAlIDEgPT09IDA7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB0YW5oKHg6IG51bWJlcik6IG51bWJlciB7XG4gIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgaWYgKChNYXRoIGFzIGFueSkudGFuaCAhPSBudWxsKSB7XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIHJldHVybiAoTWF0aCBhcyBhbnkpLnRhbmgoeCk7XG4gIH1cbiAgaWYgKHggPT09IEluZmluaXR5KSB7XG4gICAgcmV0dXJuIDE7XG4gIH0gZWxzZSBpZiAoeCA9PT0gLUluZmluaXR5KSB7XG4gICAgcmV0dXJuIC0xO1xuICB9IGVsc2Uge1xuICAgIGNvbnN0IGUyeCA9IE1hdGguZXhwKDIgKiB4KTtcbiAgICByZXR1cm4gKGUyeCAtIDEpIC8gKGUyeCArIDEpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBzaXplVG9TcXVhcmlzaFNoYXBlKHNpemU6IG51bWJlcik6IFtudW1iZXIsIG51bWJlcl0ge1xuICBmb3IgKGxldCBhID0gTWF0aC5mbG9vcihNYXRoLnNxcnQoc2l6ZSkpOyBhID4gMTsgLS1hKSB7XG4gICAgaWYgKHNpemUgJSBhID09PSAwKSB7XG4gICAgICByZXR1cm4gW2EsIHNpemUgLyBhXTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIFsxLCBzaXplXTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVNodWZmbGVkSW5kaWNlcyhuOiBudW1iZXIpOiBVaW50MzJBcnJheSB7XG4gIGNvbnN0IHNodWZmbGVkSW5kaWNlcyA9IG5ldyBVaW50MzJBcnJheShuKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBuOyArK2kpIHtcbiAgICBzaHVmZmxlZEluZGljZXNbaV0gPSBpO1xuICB9XG4gIHNodWZmbGUoc2h1ZmZsZWRJbmRpY2VzKTtcbiAgcmV0dXJuIHNodWZmbGVkSW5kaWNlcztcbn1cbiJdfQ==
