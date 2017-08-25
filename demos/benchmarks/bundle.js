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
var ndarray_1 = require("../../src/math/ndarray");
var conv_gpu_1 = require("../../src/math/webgl/conv_gpu");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var gpgpu_math = require("../../src/math/webgl/gpgpu_math");
var texture_manager_1 = require("../../src/math/webgl/texture_manager");
var OP_RUNS = 40;
exports.BENCHMARK_TEST = function (size) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var texManager = new texture_manager_1.TextureManager(gpgpu);
    ndarray_1.initializeGPU(gpgpu, texManager);
    var inDepth = 1;
    var inShape = [size, size, inDepth];
    var outDepth = 1;
    var filterSize = 11;
    var stride = 1;
    var zeroPad = conv_util.computeDefaultPad(inShape, filterSize, stride);
    var hasBias = true;
    var outputInfo = conv_util.computeOutputInfo(inShape, filterSize, filterSize, outDepth, stride, stride, zeroPad);
    var program = new conv_gpu_1.Conv2DProgram(inShape, filterSize, filterSize, stride, stride, outputInfo, hasBias);
    var outputShape = program.outputShape;
    var out = ndarray_1.Array3D.zeros(outputShape);
    var x = ndarray_1.Array3D.randUniform(inShape, -1, 1);
    var wShape = conv_util.computeWeightsShape4D(1, outDepth, filterSize);
    var W = ndarray_1.Array4D.randUniform(wShape, -1, 1);
    var b = ndarray_1.Array1D.randUniform([outDepth], -1, 1);
    var inputs = [x, W, b];
    var binary = gpgpu_math.compileProgram(gpgpu, program, inputs, out);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        gpgpu_math.runProgram(binary, inputs, out);
    }
    out.getValues();
    var avgTime = (performance.now() - start) / OP_RUNS;
    x.dispose();
    W.dispose();
    b.dispose();
    out.dispose();
    texManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/conv_util":15,"../../src/math/ndarray":19,"../../src/math/webgl/conv_gpu":21,"../../src/math/webgl/gpgpu_context":22,"../../src/math/webgl/gpgpu_math":23,"../../src/math/webgl/texture_manager":31}],3:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../../src/math/conv_util");
var ndarray_1 = require("../../src/math/ndarray");
var conv_backprop_gpu_1 = require("../../src/math/webgl/conv_backprop_gpu");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var gpgpu_math = require("../../src/math/webgl/gpgpu_math");
var texture_manager_1 = require("../../src/math/webgl/texture_manager");
var OP_RUNS = 40;
exports.BENCHMARK_TEST = function (size) {
    var origInputDepth = 1;
    var origOutputDepth = 2;
    var xShape = [size, size, 1];
    var fieldSize = 11;
    var origStride = 1;
    var origPad = 1;
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var texManager = new texture_manager_1.TextureManager(gpgpu);
    ndarray_1.initializeGPU(gpgpu, texManager);
    gpgpu.enableAutomaticDebugValidation(true);
    var hasBias = false;
    var program = new conv_backprop_gpu_1.Conv2DTransposeProgram(xShape, fieldSize, origInputDepth, origStride, origPad, hasBias);
    var outputShape = program.outputShape;
    var out = ndarray_1.Array3D.zeros(outputShape);
    var x = ndarray_1.Array3D.randUniform(xShape, -1, 1);
    var wShape = conv_util.computeWeightsShape4D(origInputDepth, origOutputDepth, fieldSize);
    var W = ndarray_1.Array4D.randUniform(wShape, -1, 1);
    var inputs = [x, W];
    var binary = gpgpu_math.compileProgram(gpgpu, program, inputs, out);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        gpgpu_math.runProgram(binary, inputs, out);
    }
    out.getValues();
    var avgTime = (performance.now() - start) / OP_RUNS;
    texManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/conv_util":15,"../../src/math/ndarray":19,"../../src/math/webgl/conv_backprop_gpu":20,"../../src/math/webgl/gpgpu_context":22,"../../src/math/webgl/gpgpu_math":23,"../../src/math/webgl/texture_manager":31}],4:[function(require,module,exports){
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
var ndarray_1 = require("../../src/math/ndarray");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var gpgpu_math = require("../../src/math/webgl/gpgpu_math");
var logsumexp_gpu_1 = require("../../src/math/webgl/logsumexp_gpu");
var texture_manager_1 = require("../../src/math/webgl/texture_manager");
var OP_RUNS = 2;
exports.BENCHMARK_TEST = function (size) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var texManager = new texture_manager_1.TextureManager(gpgpu);
    ndarray_1.initializeGPU(gpgpu, texManager);
    var out = new ndarray_1.Scalar({ texture: texManager.acquireTexture([1, 1]) });
    var a = ndarray_1.Array2D.randUniform([size, size], -1, 1);
    var program = new logsumexp_gpu_1.LogSumExpProgram(a.size);
    var binary = gpgpu_math.compileProgram(gpgpu, program, [a], out);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        gpgpu_math.runProgram(binary, [a], out);
    }
    out.getValues();
    var avgTime = (performance.now() - start) / OP_RUNS;
    a.dispose();
    out.dispose();
    texManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/ndarray":19,"../../src/math/webgl/gpgpu_context":22,"../../src/math/webgl/gpgpu_math":23,"../../src/math/webgl/logsumexp_gpu":25,"../../src/math/webgl/texture_manager":31}],6:[function(require,module,exports){
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
        name: 'Matrix Multiplication (CPU vs GPU): ' +
            'matmul([size, size], [size, size])',
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
var ndarray_1 = require("../../src/math/ndarray");
var gpgpu_context_1 = require("../../src/math/webgl/gpgpu_context");
var gpgpu_math = require("../../src/math/webgl/gpgpu_math");
var pool_gpu_1 = require("../../src/math/webgl/pool_gpu");
var texture_manager_1 = require("../../src/math/webgl/texture_manager");
var OP_RUNS = 40;
exports.MAX_POOL_BENCHMARK_TEST = function (size) {
    var positions = false;
    return testMaxPool(size, positions);
};
exports.MAX_POOL_POSNS_BENCHMARK_TEST = function (size) {
    var positions = true;
    return testMaxPool(size, positions);
};
function testMaxPool(size, positions) {
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    var texManager = new texture_manager_1.TextureManager(gpgpu);
    ndarray_1.initializeGPU(gpgpu, texManager);
    var outputDepth = 1;
    var xShape = [size, size, outputDepth];
    var fieldSize = 11;
    var stride = 1;
    var zeroPad = conv_util.computeDefaultPad(xShape, fieldSize, stride);
    var program = new pool_gpu_1.Pool2DProgram(xShape, fieldSize, stride, zeroPad, 'max', positions);
    var res = ndarray_1.NDArray.zeros(program.outputShape);
    var x = ndarray_1.Array3D.randUniform(xShape, -1, 1);
    var binary = gpgpu_math.compileProgram(gpgpu, program, [x], res);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        gpgpu_math.runProgram(binary, [x], res);
    }
    res.getValues();
    var avgTime = (performance.now() - start) / OP_RUNS;
    x.dispose();
    res.dispose();
    texManager.dispose();
    gpgpu.deleteProgram(binary.webGLProgram);
    gpgpu.dispose();
    return avgTime;
}

},{"../../src/math/conv_util":15,"../../src/math/ndarray":19,"../../src/math/webgl/gpgpu_context":22,"../../src/math/webgl/gpgpu_math":23,"../../src/math/webgl/pool_gpu":28,"../../src/math/webgl/texture_manager":31}],9:[function(require,module,exports){
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
var mulmat_gpu_1 = require("../../src/math/webgl/mulmat_gpu");
var gpgpu_math = require("../../src/math/webgl/gpgpu_math");
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
    var program = new mulmat_gpu_1.MatMulProgram(aArr.shape, bArr.shape);
    var binary = gpgpu_math.compileProgram(gpgpu, program, [aArr, bArr], resArr);
    var a = test_util.randomArrayInRange(size * size, -1, 1);
    var b = test_util.randomArrayInRange(size * size, -1, 1);
    gpgpu.uploadMatrixToTexture(aTexture, size, size, a);
    gpgpu.uploadMatrixToTexture(bTexture, size, size, b);
    var start = performance.now();
    for (var i = 0; i < OP_RUNS; i++) {
        gpgpu_math.runProgram(binary, [aArr, bArr], resArr);
    }
    gpgpu.downloadMatrixFromTexture(resultTexture, size, size);
    var avgTime = (performance.now() - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(bTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(binary.webGLProgram);
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
    gpgpu.downloadMatrixFromPackedTexture(resultTexture, size, size);
    var avgTime = (performance.now() - start) / OP_RUNS;
    gpgpu.deleteMatrixTexture(aTexture);
    gpgpu.deleteMatrixTexture(bTexture);
    gpgpu.deleteMatrixTexture(resultTexture);
    gpgpu.deleteProgram(program);
    gpgpu.dispose();
    return avgTime;
};

},{"../../src/math/math":17,"../../src/math/ndarray":19,"../../src/math/webgl/gpgpu_context":22,"../../src/math/webgl/gpgpu_math":23,"../../src/math/webgl/mulmat_gpu":26,"../../src/math/webgl/mulmat_packed_gpu":27,"../../src/test_util":33}],11:[function(require,module,exports){
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

},{"../util":34}],15:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
function computeOutputInfo(inShape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, padding) {
    if (typeof padding === 'number') {
        var outShape_1 = computeOutputShape3D(inShape, filterHeight, outDepth, strideHeight, padding);
        return {
            shape: outShape_1,
            paddingInfo: { top: padding, bottom: padding, left: padding, right: padding }
        };
    }
    var inHeight = inShape[0];
    var inWidth = inShape[1];
    var outShape;
    var paddingInfo;
    if (padding === 'same') {
        var outHeight = Math.ceil(inHeight / strideHeight);
        var outWidth = Math.ceil(inWidth / strideWidth);
        outShape = [outHeight, outWidth, outDepth];
        var padAlongHeight = (outHeight - 1) * strideHeight + filterHeight - inHeight;
        var padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
        var top_1 = Math.floor(padAlongHeight / 2);
        var bottom = padAlongHeight - top_1;
        var left = Math.floor(padAlongWidth / 2);
        var right = padAlongWidth - left;
        paddingInfo = { top: top_1, bottom: bottom, left: left, right: right };
    }
    else if (padding === 'valid') {
        var outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
        var outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
        outShape = [outHeight, outWidth, outDepth];
        paddingInfo = { top: 0, bottom: 0, left: 0, right: 0 };
    }
    else {
        throw Error("Unknown padding parameter: " + padding);
    }
    return { shape: outShape, paddingInfo: paddingInfo };
}
exports.computeOutputInfo = computeOutputInfo;
function computeOutputShape3D(inShape, fieldSize, outDepth, stride, zeroPad) {
    if (zeroPad == null) {
        zeroPad = computeDefaultPad(inShape, fieldSize, stride);
    }
    var inputRows = inShape[0];
    var inputCols = inShape[1];
    var outputRows = (inputRows - fieldSize + 2 * zeroPad) / stride + 1;
    util.assert(util.isInt(outputRows), "The output # of rows (" + outputRows + ") must be an integer. Change the " +
        "stride and/or zero pad parameters");
    var outputCols = (inputCols - fieldSize + 2 * zeroPad) / stride + 1;
    util.assert(util.isInt(outputCols), "The output # of columns (" + outputCols + ") must be an integer. Change " +
        "the stride and/or zero pad parameters");
    return [outputRows, outputCols, outDepth];
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
function computeDilatedRC(rc, origStride) {
    var rowsDilated = (rc[0] - 1) * origStride + 1;
    var colsDilated = (rc[1] - 1) * origStride + 1;
    return [rowsDilated, colsDilated];
}
exports.computeDilatedRC = computeDilatedRC;

},{"../util":34}],16:[function(require,module,exports){
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
var conv_util = require("./conv_util");
var copy2d_util = require("./copy2d_util");
var ndarray_1 = require("./ndarray");
var NDArrayMath = (function () {
    function NDArrayMath(safeMode) {
        this.safeMode = safeMode;
        this.ndarrayScopes = [];
        this.ndarraysToKeep = [];
        this.activeScopeNDArraysToKeep = [];
        this.debugMode = false;
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
    NDArrayMath.prototype.enableDebugMode = function () {
        this.debugMode = true;
        console.warn('Debugging mode is ON. The output of every math call will ' +
            'be downloaded to CPU and checked for NaNs. ' +
            'This significantly impacts performance.');
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
        var arraysToKeep = this.activeScopeNDArraysToKeep;
        if (result != null) {
            arraysToKeep = arraysToKeep.concat(result);
        }
        for (var i = 0; i < this.activeScope.length; i++) {
            var ndarray = this.activeScope[i];
            if (this.isNDArrayDataInList(ndarray, arraysToKeep)) {
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
    NDArrayMath.prototype.checkForNaN = function (arr) {
        var vals = arr.getValues();
        for (var i = 0; i < vals.length; i++) {
            if (isNaN(vals[i])) {
                throw Error('The result NDArray of the last math call has NaNs.');
            }
        }
    };
    NDArrayMath.prototype.track = function (result) {
        if (this.debugMode) {
            this.checkForNaN(result);
        }
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
        console.warn('math.reshape() is deprecated. Please call reshape() ' +
            'directly on the ndarray object');
        return ndarray.reshape(newShape);
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
        return this.add(c, a);
    };
    NDArrayMath.prototype.scalarMinusArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarMinusArray: first argument must be rank 0, but got " +
            ("rank " + c.rank + "."));
        return this.sub(c, a);
    };
    NDArrayMath.prototype.arrayMinusScalar = function (a, c) {
        util.assert(c.size === 1, "Error in arrayMinusScalar: second argument must be rank 0, but " +
            ("got rank " + c.rank + "."));
        return this.sub(a, c);
    };
    NDArrayMath.prototype.neg = function (a) {
        return this.track(this.negInternal(a));
    };
    NDArrayMath.prototype.add = function (a, b) {
        util.assertAndGetBroadcastedShape(a.shape, b.shape);
        return this.track(this.addInternal(a, b));
    };
    NDArrayMath.prototype.addStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in addStrict: ');
        return this.add(a, b);
    };
    NDArrayMath.prototype.sub = function (a, b) {
        util.assertAndGetBroadcastedShape(a.shape, b.shape);
        return this.track(this.subInternal(a, b));
    };
    NDArrayMath.prototype.subStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in subStrict: ');
        return this.sub(a, b);
    };
    NDArrayMath.prototype.multiply = function (a, b) {
        util.assertAndGetBroadcastedShape(a.shape, b.shape);
        return this.track(this.multiplyInternal(a, b));
    };
    NDArrayMath.prototype.elementWiseMul = function (a, b) {
        return this.multiplyStrict(a, b);
    };
    NDArrayMath.prototype.multiplyStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in multiplyStrict: ');
        return this.multiply(a, b);
    };
    NDArrayMath.prototype.divide = function (a, b) {
        util.assertAndGetBroadcastedShape(a.shape, b.shape);
        return this.track(this.divideInternal(a, b));
    };
    NDArrayMath.prototype.divideStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in divideStrict: ');
        return this.divide(a, b);
    };
    NDArrayMath.prototype.scalarDividedByArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarDividedByArray: first argument must be rank 0, but " +
            ("got NDArray of rank " + c.rank + "."));
        return this.divide(c, a);
    };
    NDArrayMath.prototype.arrayDividedByScalar = function (a, c) {
        util.assert(c.size === 1, "Error in arrayDividedByScalar: second argument must be rank 0, " +
            ("but got NDArray of rank " + c.rank + "."));
        return this.divide(a, c);
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
        return this.multiply(c, a);
    };
    NDArrayMath.prototype.elementWiseMulBroadcast = function (a, b) {
        util.assert(a.rank === 2, "Error in elementWiseMulBroadcast: first argument must be " +
            ("rank 2, but got rank " + a.rank + "."));
        util.assert(b.rank === 2, "Error in elementWiseMulBroadcast: second argument must be " +
            ("rank 2, but got rank " + b.rank + "."));
        return this.multiply(a, b);
    };
    NDArrayMath.prototype.conv2d = function (x, weights, bias, strides, pad) {
        util.assert(x.rank === 3, "Error in conv2d: x must be rank 3, but got rank " + x.rank + ".");
        util.assert(weights.rank === 4, "Error in conv2d: weights must be rank 4, but got rank " +
            (weights.rank + "."));
        if (bias != null) {
            util.assert(bias.rank === 1, "Error in conv2d: bias must be rank 1, but got rank " +
                (bias.rank + "."));
        }
        util.assert(x.shape[2] === weights.shape[2], "Error in conv2d: depth of input (" + x.shape[2] + ") must match  " +
            ("input depth for weights " + weights.shape[2] + "."));
        var filterHeight = weights.shape[0];
        var filterWidth = weights.shape[1];
        var outDepth = weights.shape[3];
        var strideHeight;
        var strideWidth;
        if (typeof strides === 'number') {
            strideHeight = strides;
            strideWidth = strides;
        }
        else {
            strideHeight = strides[0];
            strideWidth = strides[1];
        }
        var outputInfo = conv_util.computeOutputInfo(x.shape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, pad);
        return this.track(this.conv2dInternal(x, weights, bias, strideHeight, strideWidth, outputInfo));
    };
    NDArrayMath.prototype.conv2dBackProp = function (x, dy, weights, strides, pad) {
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
        var filterHeight = weights.shape[0];
        var filterWidth = weights.shape[1];
        var outDepth = weights.shape[3];
        var strideHeight;
        var strideWidth;
        if (typeof strides === 'number') {
            strideHeight = strides;
            strideWidth = strides;
        }
        else {
            strideHeight = strides[0];
            strideWidth = strides[1];
        }
        var outputInfo = conv_util.computeOutputInfo(x.shape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, pad);
        var backpropResult = this.conv2dBackPropInternal(x, dy, weights, strideHeight, strideWidth, outputInfo);
        this.track(backpropResult.db);
        this.track(backpropResult.dw);
        this.track(backpropResult.dx);
        return backpropResult;
    };
    NDArrayMath.prototype.conv2dTranspose = function (x, weights, bias, strides, pad) {
        util.assert(x.rank === 3, "Error in conv2dTranspose: x must be rank 3, but got rank " +
            (x.rank + "."));
        util.assert(weights.rank === 4, "Error in conv2dTranspose: weights must be rank 4, but got " +
            ("rank " + weights.rank));
        if (bias != null) {
            util.assert(bias.rank === 1, "Error in conv2dTranspose: bias must be rank 1, but got ' +\n              'rank " + bias.rank + ".");
        }
        util.assert(x.shape[2] === weights.shape[3], "Error in conv2dTranspose: depth of input (" + x.shape[2] + ") must " +
            ("match input depth for weights " + weights.shape[3] + "."));
        var filterHeight = weights.shape[0];
        var filterWidth = weights.shape[1];
        var outDepth = weights.shape[3];
        var strideHeight;
        var strideWidth;
        if (typeof strides === 'number') {
            strideHeight = strides;
            strideWidth = strides;
        }
        else {
            strideHeight = strides[0];
            strideWidth = strides[1];
        }
        var outputInfo = conv_util.computeOutputInfo(x.shape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, pad);
        return this.track(this.conv2dTransposeInternal(x, weights, bias, strideHeight, strideWidth, outputInfo));
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
    NDArrayMath.prototype.multiRNNCell = function (lstmCells, data, c, h) {
        util.assert(data.shape[0] === 1, "Error in multiRNNCell: first dimension of data is " + data.shape[0] + ", " +
            "but batch sizes > 1 are not yet supported.");
        var res = this.scope(function () {
            var input = data;
            var newStates = [];
            for (var i = 0; i < lstmCells.length; i++) {
                var output = lstmCells[i](input, c[i], h[i]);
                newStates.push(output[0]);
                newStates.push(output[1]);
                input = output[1];
            }
            return newStates;
        });
        var newC = [];
        var newH = [];
        for (var i = 0; i < res.length; i += 2) {
            newC.push(res[i]);
            newH.push(res[i + 1]);
        }
        return [newC, newH];
    };
    NDArrayMath.prototype.basicLSTMCell = function (forgetBias, lstmKernel, lstmBias, data, c, h) {
        var _this = this;
        var res = this.scope(function () {
            util.assert(data.shape[0] === 1, "Error in multiRNNCell: first dimension of data is " +
                (data.shape[0] + ", but batch sizes > 1 are not yet supported."));
            var data3D = data.as3D(1, 1, data.shape[1]);
            var h3D = h.as3D(1, 1, h.shape[1]);
            var combined3D = _this.concat3D(data3D, h3D, 2);
            var combined2D = combined3D.as2D(1, data.shape[1] + h.shape[1]);
            var weighted = _this.matMul(combined2D, lstmKernel);
            var res = _this.add(weighted, lstmBias);
            var i = _this.slice2D(res, [0, 0], [res.shape[0], res.shape[1] / 4]);
            var j = _this.slice2D(res, [0, res.shape[1] / 4 * 1], [res.shape[0], res.shape[1] / 4]);
            var f = _this.slice2D(res, [0, res.shape[1] / 4 * 2], [res.shape[0], res.shape[1] / 4]);
            var o = _this.slice2D(res, [0, res.shape[1] / 4 * 3], [res.shape[0], res.shape[1] / 4]);
            var newC = _this.add(_this.multiplyStrict(c, _this.sigmoid(_this.scalarPlusArray(forgetBias, f))), _this.multiplyStrict(_this.sigmoid(i), _this.tanh(j)));
            var newH = _this.multiplyStrict(_this.tanh(newC), _this.sigmoid(o));
            return [newC, newH];
        });
        return [res[0], res[1]];
    };
    return NDArrayMath;
}());
exports.NDArrayMath = NDArrayMath;
var MatrixOrientation;
(function (MatrixOrientation) {
    MatrixOrientation[MatrixOrientation["REGULAR"] = 0] = "REGULAR";
    MatrixOrientation[MatrixOrientation["TRANSPOSED"] = 1] = "TRANSPOSED";
})(MatrixOrientation = exports.MatrixOrientation || (exports.MatrixOrientation = {}));

},{"../util":34,"./concat3d_util":14,"./conv_util":15,"./copy2d_util":16,"./ndarray":19}],18:[function(require,module,exports){
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
var concat3d_util = require("./concat3d_util");
var conv_util = require("./conv_util");
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
        var values = ndarray_1.Array3D.zeros(outputShape);
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
    NDArrayMathCPU.prototype.scaledArrayAddInternal = function (c1, a, c2, b) {
        var newShape = util.assertAndGetBroadcastedShape(a.shape, b.shape);
        var newValues = new Float32Array(util.sizeFromShape(newShape));
        var aValues = a.getValues();
        var bValues = b.getValues();
        var c1Val = c1.get();
        var c2Val = c2.get();
        for (var i = 0; i < newValues.length; ++i) {
            newValues[i] = c1Val * aValues[i % a.size] + c2Val * bValues[i % b.size];
        }
        return ndarray_1.NDArray.make(newShape, { values: newValues });
    };
    NDArrayMathCPU.prototype.negInternal = function (a) {
        return this.scalarTimesArray(ndarray_1.Scalar.NEG_ONE, a);
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
    NDArrayMathCPU.prototype.multiplyInternal = function (a, b) {
        var newShape = util.assertAndGetBroadcastedShape(a.shape, b.shape);
        var newValues = new Float32Array(util.sizeFromShape(newShape));
        var aValues = a.getValues();
        var bValues = b.getValues();
        for (var i = 0; i < newValues.length; ++i) {
            newValues[i] = aValues[i % a.size] * bValues[i % b.size];
        }
        return ndarray_1.NDArray.make(newShape, { values: newValues });
    };
    NDArrayMathCPU.prototype.divideInternal = function (a, b) {
        var newShape = util.assertAndGetBroadcastedShape(a.shape, b.shape);
        var newValues = new Float32Array(util.sizeFromShape(newShape));
        var aValues = a.getValues();
        var bValues = b.getValues();
        for (var i = 0; i < newValues.length; ++i) {
            newValues[i] = aValues[i % a.size] / bValues[i % b.size];
        }
        return ndarray_1.NDArray.make(newShape, { values: newValues });
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
    NDArrayMathCPU.prototype.conv2dInternal = function (x, weights, biases, strideHeight, strideWidth, outputInfo) {
        var _a = x.shape, xRows = _a[0], xCols = _a[1], inputDepth = _a[2];
        var filterHeight = weights.shape[0];
        var filterWidth = weights.shape[1];
        var outDepth = weights.shape[3];
        var padLeft = outputInfo.paddingInfo.left;
        var padTop = outputInfo.paddingInfo.top;
        var y = ndarray_1.Array3D.zeros(outputInfo.shape);
        for (var d2 = 0; d2 < outDepth; ++d2) {
            for (var yR = 0; yR < y.shape[0]; ++yR) {
                var xRCorner = yR * strideHeight - padLeft;
                var xRMin = Math.max(0, xRCorner);
                var xRMax = Math.min(xRows, filterHeight + xRCorner);
                for (var yC = 0; yC < y.shape[1]; ++yC) {
                    var xCCorner = yC * strideWidth - padTop;
                    var xCMin = Math.max(0, xCCorner);
                    var xCMax = Math.min(xCols, filterWidth + xCCorner);
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
        var xRows = x.shape[0];
        var xCols = x.shape[1];
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
        var xRows = x.shape[0];
        var xCols = x.shape[1];
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
                        for (var xC = xCMin; xC < xCMax; ++xC) {
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

},{"../util":34,"./concat3d_util":14,"./conv_util":15,"./copy2d_util":16,"./math":17,"./ndarray":19}],19:[function(require,module,exports){
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
    Scalar.ZERO = Scalar.new(0);
    Scalar.ONE = Scalar.new(1);
    Scalar.TWO = Scalar.new(2);
    Scalar.NEG_ONE = Scalar.new(-1);
    return Scalar;
}(NDArray));
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

},{"../util":34,"./webgl/webgl_util":32}],20:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
var Conv2DDerWeightsProgram = (function () {
    function Conv2DDerWeightsProgram(xShape, fSize, outputDepth, stride, zeroPad) {
        this.variableNames = ['x', 'dy'];
        var yShape = conv_util.computeOutputShape3D(xShape, fSize, outputDepth, stride, zeroPad);
        var yNumRows = yShape[0];
        var yNumCols = yShape[1];
        var xNumRows = xShape[0];
        var xNumCols = xShape[1];
        this.outputShape =
            conv_util.computeWeightsShape4D(xShape[2], outputDepth, fSize);
        this.params = [stride, zeroPad];
        this.userCode = "\n      void main() {\n        vec4 coords = getOutputCoords();\n        float wR = coords.x;\n        float wC = coords.y;\n        float d1 = coords.z;\n        float d2 = coords.w;\n\n        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int iyR = 0; iyR < " + yNumRows + "; iyR++) {\n          float yR = float(iyR);\n          float xR = wR + yR * " + stride + ".0 - " + zeroPad + ".0;\n\n          if (xR < 0.0 || xR >= " + xNumRows + ".0) {\n            continue;\n          }\n\n          for (int iyC = 0; iyC < " + yNumCols + "; iyC++) {\n            float yC = float(iyC);\n            float xC = wC + yC * " + stride + ".0 - " + zeroPad + ".0;\n\n            if (xC < 0.0 || xC >= " + xNumCols + ".0) {\n              continue;\n            }\n\n            float dyValue = getDy(yR, yC, d2);\n            float xValue = getX(xR, xC, d1);\n            dotProd += (xValue * dyValue);\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
    }
    return Conv2DDerWeightsProgram;
}());
exports.Conv2DDerWeightsProgram = Conv2DDerWeightsProgram;
var Conv2DTransposeProgram = (function () {
    function Conv2DTransposeProgram(xShape, fSize, origInputDepth, origStride, origPad, hasBias) {
        this.variableNames = ['x', 'W', 'bias'];
        var xRows = xShape[0], xCols = xShape[1], origOutputDepth = xShape[2];
        var biasSnippet = hasBias ? 'dotProd += getBias(d2);' : '';
        var xRowsDilated = (xRows - 1) * origStride + 1;
        var xColsDilated = (xCols - 1) * origStride + 1;
        var pad = fSize - 1 - origPad;
        this.outputShape = conv_util.computeOutputShape3D([xRowsDilated, xColsDilated, origOutputDepth], fSize, origInputDepth, 1, pad);
        this.params = [pad, fSize, origStride, hasBias];
        this.userCode = "\n      void main() {\n        vec3 coords = getOutputCoords();\n        float yR = coords.x;\n        float yC = coords.y;\n        float d2 = coords.z;\n\n        vec2 xRCCorner = vec2(yR, yC) - vec2(" + pad + ".0, " + pad + ".0);\n        float xRCorner = xRCCorner.x;\n        float xCCorner = xRCCorner.y;\n\n        // Convolve x(?, ?, d1) with w(:, :, d2, d1) to get y(yR, yC, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int iwR = 0; iwR < " + fSize + "; iwR++) {\n          float wR = float(iwR);\n          float xR = (xRCorner + wR) / " + origStride + ".0;\n\n          if (xR < 0.0 || xR >= " + xRows + ".0 || fract(xR) > 0.0) {\n            continue;\n          }\n\n          float wRPerm = " + fSize + ".0 - 1.0 - wR;\n\n          for (int iwC = 0; iwC < " + fSize + "; iwC++) {\n            float wC = float(iwC);\n            float xC = (xCCorner + wC) / " + origStride + ".0;\n\n            if (xC < 0.0 || xC >= " + xCols + ".0 || fract(xC) > 0.0) {\n              continue;\n            }\n\n            float wCPerm = " + fSize + ".0 - 1.0 - wC;\n\n            for (int id1 = 0; id1 < " + origOutputDepth + "; id1++) {\n              float d1 = float(id1);\n              float xValue = getX(xR, xC, d1);\n              float wValue = getW(wRPerm, wCPerm, d2, d1);\n              dotProd += xValue * wValue;\n            }\n          }\n        }\n        " + biasSnippet + "\n        setOutput(dotProd);\n      }\n    ";
    }
    return Conv2DTransposeProgram;
}());
exports.Conv2DTransposeProgram = Conv2DTransposeProgram;
var Conv2DDerBiasProgram = (function () {
    function Conv2DDerBiasProgram(yShape) {
        this.variableNames = ['dy'];
        this.params = [];
        var yNumRows = yShape[0], yNumCols = yShape[1], outputDepth = yShape[2];
        this.outputShape = [outputDepth];
        this.userCode = "\n      void main() {\n        float d2 = getOutputCoords();\n\n        float derBias = 0.0;\n        for (int iyR = 0; iyR < " + yNumRows + "; iyR++) {\n          float yR = float(iyR);\n          for (int iyC = 0; iyC < " + yNumCols + "; iyC++) {\n            float yC = float(iyC);\n            derBias += getDy(yR, yC, d2);\n          }\n        }\n        setOutput(derBias);\n      }\n    ";
    }
    return Conv2DDerBiasProgram;
}());
exports.Conv2DDerBiasProgram = Conv2DDerBiasProgram;

},{"../conv_util":15}],21:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Conv2DProgram = (function () {
    function Conv2DProgram(xShape, filterHeight, filterWidth, strideHeight, strideWidth, outputInfo, hasBias) {
        this.variableNames = ['x', 'W', 'bias'];
        this.outputShape = outputInfo.shape;
        var inputDepth = xShape[2];
        this.params =
            [filterWidth, filterHeight, strideHeight, strideWidth, hasBias];
        var biasSnippet = hasBias ? 'dotProd += getBias(d2);' : '';
        var xNumRows = xShape[0];
        var xNumCols = xShape[1];
        var padTop = outputInfo.paddingInfo.top;
        var padLeft = outputInfo.paddingInfo.left;
        this.userCode = "\n      const vec2 strides = vec2(" + strideHeight + ".0, " + strideWidth + ".0);\n      const vec2 pads = vec2(" + padTop + ".0, " + padLeft + ".0);\n\n      void main() {\n        vec3 coords = getOutputCoords();\n        float yR = coords.x;\n        float yC = coords.y;\n        float d2 = coords.z;\n\n        vec2 xRCCorner = vec2(yR, yC) * strides - pads;\n        float xRCorner = xRCCorner.x;\n        float xCCorner = xRCCorner.y;\n\n        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int iwR = 0; iwR < " + filterHeight + "; iwR++) {\n          float wR = float(iwR);\n          float xR = xRCorner + wR;\n\n          if (xR < 0.0 || xR >= " + xNumRows + ".0) {\n            continue;\n          }\n\n          for (int iwC = 0; iwC < " + filterWidth + "; iwC++) {\n            float wC = float(iwC);\n            float xC = xCCorner + wC;\n\n            if (xC < 0.0 || xC >= " + xNumCols + ".0) {\n              continue;\n            }\n\n            for (int id1 = 0; id1 < " + inputDepth + "; id1++) {\n              float d1 = float(id1);\n              float xValue = getX(xR, xC, d1);\n              float wValue = getW(wR, wC, d1, d2);\n              dotProd += xValue * wValue;\n            }\n          }\n        }\n        " + biasSnippet + "\n        setOutput(dotProd);\n      }\n    ";
    }
    return Conv2DProgram;
}());
exports.Conv2DProgram = Conv2DProgram;

},{}],22:[function(require,module,exports){
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

},{"./gpgpu_util":24,"./tex_util":30,"./webgl_util":32}],23:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
var shader_compiler = require("./shader_compiler");
function compileProgram(gpgpu, program, inputs, output) {
    var userCode = program.userCode;
    var inputInfos = inputs.map(function (input, i) {
        var shapeInfo = {
            logicalShape: input.shape,
            texShape: input.getTextureShapeRC()
        };
        return { name: program.variableNames[i], shapeInfo: shapeInfo };
    });
    var inShapeInfos = inputInfos.map(function (x) { return x.shapeInfo; });
    var outShapeInfo = {
        logicalShape: output.shape,
        texShape: output.getTextureShapeRC()
    };
    var source = shader_compiler.makeShader(inputInfos, outShapeInfo, userCode, program.supportsBroadcasting === true);
    return {
        program: program,
        source: source,
        webGLProgram: gpgpu.createProgram(source), gpgpu: gpgpu, inShapeInfos: inShapeInfos, outShapeInfo: outShapeInfo
    };
}
exports.compileProgram = compileProgram;
function validateBinaryAndProgram(shapeInfos, inputs) {
    if (shapeInfos.length !== inputs.length) {
        throw Error("Binary was compiled with " + shapeInfos.length + " inputs, but " +
            ("was executed with " + inputs.length + " inputs"));
    }
    shapeInfos.forEach(function (s, i) {
        var shapeA = s.logicalShape;
        var texShapeA = s.texShape;
        var shapeB = inputs[i].shape;
        var texShapeB = inputs[i].getTextureShapeRC();
        if (!util.arraysEqual(shapeA, shapeB)) {
            throw Error("Binary was compiled with different shapes than " +
                ("the current args. Shapes " + shapeA + " and " + shapeB + " must match"));
        }
        if (!util.arraysEqual(texShapeA, texShapeB)) {
            throw Error("Binary was compiled with different texture shapes than the" +
                (" current args. Shape " + texShapeA + " and " + texShapeB + " must match"));
        }
    });
}
function runProgram(binary, inputs, output, customSetup) {
    validateBinaryAndProgram(binary.inShapeInfos, inputs);
    validateBinaryAndProgram([binary.outShapeInfo], [output]);
    var outTex = output.getTexture();
    var outTexShape = output.getTextureShapeRC();
    var gpgpu = binary.gpgpu;
    gpgpu.setOutputMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
    gpgpu.setProgram(binary.webGLProgram);
    inputs.forEach(function (input, i) {
        var tex = input.getTexture();
        gpgpu.setInputMatrixTexture(tex, binary.program.variableNames[i], i);
    });
    if (customSetup != null) {
        customSetup(gpgpu);
    }
    gpgpu.executeProgram();
}
exports.runProgram = runProgram;
function makeShaderKey(program, inputs, output) {
    var params = program.params;
    var keyStart = inputs.concat(output).map(function (x) { return x.shape + '_' + x.getTextureShapeRC(); });
    var keyEnd = params.map(function (p) { return p.toString(); });
    var key = [program.constructor.name];
    key.push((program.supportsBroadcasting === true).toString());
    key = key.concat(keyStart, keyEnd);
    return key.join('_');
}
exports.makeShaderKey = makeShaderKey;

},{"../../util":34,"./shader_compiler":29}],24:[function(require,module,exports){
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

},{"./tex_util":30,"./webgl_util":32}],25:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var LogSumExpProgram = (function () {
    function LogSumExpProgram(aSize) {
        this.variableNames = ['A'];
        this.params = [];
        this.outputShape = [];
        this.userCode = "\n      void main() {\n        float aMax = getAFlat(0.0);\n        for (int i = 0; i < " + aSize + "; i++) {\n          aMax = max(aMax, getAFlat(float(i)));\n        }\n\n        float expSum = 0.0;\n        for (int i = 0; i < " + aSize + "; i++) {\n          expSum += exp(getAFlat(float(i)) - aMax);\n        }\n\n        setOutput(aMax + log(expSum));\n      }\n    ";
    }
    return LogSumExpProgram;
}());
exports.LogSumExpProgram = LogSumExpProgram;

},{}],26:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_1 = require("../math");
var MatMulProgram = (function () {
    function MatMulProgram(aShape, bShape, aOrient, bOrient) {
        if (aOrient === void 0) { aOrient = math_1.MatrixOrientation.REGULAR; }
        if (bOrient === void 0) { bOrient = math_1.MatrixOrientation.REGULAR; }
        this.variableNames = ['matrixA', 'matrixB'];
        this.params = [aOrient, bOrient];
        var outerShapeA = (aOrient === math_1.MatrixOrientation.REGULAR) ? aShape[0] : aShape[1];
        var outerShapeB = (bOrient === math_1.MatrixOrientation.REGULAR) ? bShape[1] : bShape[0];
        this.outputShape = [outerShapeA, outerShapeB];
        var sharedDim = (aOrient === math_1.MatrixOrientation.REGULAR ? aShape[1] : aShape[0]);
        var aSnippet = (aOrient === math_1.MatrixOrientation.REGULAR) ? 'aRow, i' : 'i, aRow';
        var bSnippet = (bOrient === math_1.MatrixOrientation.REGULAR) ? 'i, bCol' : 'bCol, i';
        this.userCode = "\n      const int sharedDim = " + sharedDim + ";\n\n      float dotARowBCol(float aRow, float bCol) {\n        float result = 0.0;\n        for (int ii = 0; ii < sharedDim; ii++) {\n          float i = float(ii);\n          float a = getMatrixA(" + aSnippet + ");\n          float b = getMatrixB(" + bSnippet + ");\n          result += (a * b);\n        }\n        return result;\n      }\n\n      void main() {\n        vec2 resRC = getOutputCoords();\n        setOutput(dotARowBCol(resRC.x, resRC.y));\n      }\n    ";
    }
    return MatMulProgram;
}());
exports.MatMulProgram = MatMulProgram;

},{"../math":17}],27:[function(require,module,exports){
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
    return "\n    precision highp float;\n    uniform sampler2D matrixA;\n    uniform sampler2D matrixB;\n    varying vec2 resultUV;\n\n    const float sharedDimension = " + sharedDimensionPacked + ".0;\n\n    vec4 dot2x2ARowBCol() {\n      vec4 result = vec4(0, 0, 0, 0);\n      for (int ii = 0; ii < " + sharedDimensionPacked + "; ii++) {\n        float i = float(ii);\n        float center = (i + 0.5) / sharedDimension;\n        vec4 a = texture2D(matrixA, vec2(" + aSample + "));\n        vec4 b = texture2D(matrixB, vec2(" + bSample + "));\n        result +=\n          (" + aSwizzle[0] + " * " + bSwizzle[0] + ") + (" + aSwizzle[1] + " * " + bSwizzle[1] + ");\n      }\n      return result;\n    }\n\n    void main() {\n      gl_FragColor = dot2x2ARowBCol();\n    }";
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
var Pool2DProgram = (function () {
    function Pool2DProgram(xShape, fSize, stride, pad, poolType, computePositions) {
        this.variableNames = ['x'];
        if (poolType === 'avg' && computePositions) {
            throw new Error('Cannot compute positions for average pool.');
        }
        var returnValue = 'minMaxValue';
        if (computePositions) {
            returnValue = 'minMaxPosition';
        }
        else if (poolType === 'avg') {
            returnValue = "avgValue / " + fSize * fSize + ".0";
        }
        var xRowsLimit = xShape[0] - 0.5;
        var xColsLimit = xShape[1] - 0.5;
        this.params = [stride, pad, fSize, computePositions];
        this.outputShape =
            conv_util.computeOutputShape3D(xShape, fSize, xShape[2], stride, pad);
        this.userCode = "\n      void main() {\n        vec3 coords = getOutputCoords();\n        float yR = coords.x;\n        float yC = coords.y;\n        float d = coords.z;\n\n        vec2 xRCCorner = vec2(yR, yC) * vec2(" + stride + ".0, " + stride + ".0) -\n            vec2(" + pad + ".0, " + pad + ".0);\n        float xRCorner = xRCCorner.x;\n        float xCCorner = xRCCorner.y;\n\n        // max/min x(?, ?, d) to get y(yR, yC, d).\n        // ? = to be determined\n        float minMaxValue = 0.0;\n        float minMaxValueFound = 0.0;\n        float minMaxPosition = 0.0;\n        float avgValue = 0.0;\n\n        for (int iwR = 0; iwR < " + fSize + "; iwR++) {\n          float wR = float(iwR);\n          float xR = xRCorner + wR;\n\n          if (xR < 0.0 || xR > " + xRowsLimit + ") {\n            continue;\n          }\n\n          for (int iwC = 0; iwC < " + fSize + "; iwC++) {\n            float wC = float(iwC);\n            float xC = xCCorner + wC;\n\n            if (xC < 0.0 || xC > " + xColsLimit + ") {\n              continue;\n            }\n\n            float value = getX(xR, xC, d);\n\n            if (isNaN(value)) {\n              setOutput(value);\n              return;\n            }\n\n            if (" + (poolType === 'avg') + ") {\n              avgValue += value;\n            } else {\n              // If a min / max value has already been found, use it. If not,\n              // use the current value.\n              float currMinMaxValue = mix(\n                  value, minMaxValue, minMaxValueFound);\n              if (value " + (poolType === 'min' ? '<=' : '>=') + " currMinMaxValue) {\n                minMaxValue = value;\n                minMaxValueFound = 1.0;\n                if (" + computePositions + ") {\n                  minMaxPosition = wR * " + fSize + ".0 + wC;\n                }\n              }\n            }\n          }\n        }\n        setOutput(" + returnValue + ");\n      }\n    ";
    }
    return Pool2DProgram;
}());
exports.Pool2DProgram = Pool2DProgram;

},{"../conv_util":15}],29:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
function makeShader(inputsInfo, outputShape, userCode, broadcast) {
    var inputPrefixSnippet = inputsInfo.map(function (x) { return "uniform sampler2D " + x.name + ";"; }).join('\n');
    var inputSamplingSnippet = inputsInfo.map(function (x) { return getInputSamplingSnippet(x, outputShape, broadcast); })
        .join('\n');
    var outTexShape = outputShape.texShape;
    var outputSamplingSnippet = getOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
    var source = [
        SHADER_PREFIX, inputPrefixSnippet, inputSamplingSnippet,
        outputSamplingSnippet, userCode
    ].join('\n');
    return source;
}
exports.makeShader = makeShader;
function getInputSamplingSnippet(inInfo, outShapeInfo, broadcast) {
    var shape = inInfo.shapeInfo.logicalShape;
    var texShape = inInfo.shapeInfo.texShape;
    var outTexShape = outShapeInfo.texShape;
    var res = '';
    switch (shape.length) {
        case 0:
            res += getSamplerScalar(inInfo.name);
            break;
        case 1:
            res += getSampler1D(inInfo.name, texShape);
            break;
        case 2:
            res += getSampler2D(inInfo.name, shape, texShape);
            break;
        case 3:
            res += getSampler3D(inInfo.name, shape, texShape);
            break;
        case 4:
            res += getSampler4D(inInfo.name, shape, texShape);
            break;
        default:
            throw new Error(shape.length + "-D input sampling" +
                " is not yet supported");
    }
    if (broadcast ||
        util.arraysEqual(inInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape)) {
        res +=
            getSamplerAtOutputCoords(inInfo.name, texShape, outTexShape, broadcast);
    }
    res += getSamplerFlat(inInfo.name, texShape);
    return res;
}
function getOutputSamplingSnippet(outShape, outTexShape) {
    switch (outShape.length) {
        case 0:
            return '';
        case 1:
            return getOutput1DCoords(outShape, outTexShape);
        case 2:
            return getOutput2DCoords(outShape, outTexShape);
        case 3:
            return getOutput3DCoords(outShape, outTexShape);
        case 4:
            return getOutput4DCoords(outShape, outTexShape);
        default:
            throw new Error(outShape.length + "-D output sampling is not yet supported");
    }
}
var SAMPLE_1D_SNIPPET = "\nvec2 UVfrom1D(float texNumR, float texNumC, float index) {\n  float texR = floor(index / texNumC);\n  float texC = mod(index, texNumC);\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SAMPLE_2D_SNIPPET = "\nvec2 UVfrom2D(float texNumR, float texNumC, float numC, float row,\n    float col) {\n  float index = dot(vec2(row, col), vec2(numC, 1.0));\n  float texR = floor(index / texNumC);\n  float texC = mod(index, texNumC);\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SAMPLE_3D_SNIPPET = "\nvec2 UVfrom3D(float texNumR, float texNumC, float stride0,\n    float stride1, float row, float col, float depth) {\n  float index = dot(vec3(row, col, depth), vec3(stride0, stride1, 1.0));\n  float texR = floor(index / texNumC);\n  float texC = mod(index, texNumC);\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SAMPLE_4D_SNIPPET = "\nvec2 UVfrom4D(float texNumR, float texNumC, float stride0,\n    float stride1, float stride2, float row, float col, float depth,\n    float depth2) {\n  float index = dot(vec4(row, col, depth, depth2),\n                    vec4(stride0, stride1, stride2, 1.0));\n  float texR = floor(index / texNumC);\n  float texC = mod(index, texNumC);\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SHADER_PREFIX = "\n  precision highp float;\n  varying vec2 resultUV;\n  const vec2 halfCR = vec2(0.5, 0.5);\n\n  float sample(sampler2D texture, vec2 uv) {\n    return texture2D(texture, uv).r;\n  }\n\n  void setOutput(float val) {\n    gl_FragColor = vec4(val, 0, 0, 0);\n  }\n\n  bool isNaN(float val) {\n    return val == val ? false : true;\n  }\n  " + SAMPLE_1D_SNIPPET + "\n  " + SAMPLE_2D_SNIPPET + "\n  " + SAMPLE_3D_SNIPPET + "\n  " + SAMPLE_4D_SNIPPET + "\n";
function getOutput1DCoords(shape, texShape) {
    if (texShape[0] === 1) {
        return "\n      float getOutputCoords() {\n        return floor(gl_FragCoord.x);\n      }\n    ";
    }
    if (texShape[1] === 1) {
        return "\n      float getOutputCoords() {\n        return floor(gl_FragCoord.y);\n      }\n    ";
    }
    return "\n    float getOutputCoords() {\n      vec2 resTexRC = floor(gl_FragCoord.yx);\n      return dot(resTexRC, vec2(" + texShape[1] + ".0, 1.0));\n    }\n  ";
}
function getOutput3DCoords(shape, texShape) {
    var stride0 = shape[1] * shape[2];
    var stride1 = shape[2];
    return "\n    vec3 getOutputCoords() {\n      vec2 resTexRC = floor(gl_FragCoord.yx);\n      float index = dot(resTexRC, vec2(" + texShape[1] + ".0, 1.0));\n      float r = floor(index / " + stride0 + ".0);\n      index -= r * " + stride0 + ".0;\n      float c = floor(index / " + stride1 + ".0);\n      float d = mod(index, " + stride1 + ".0);\n      return vec3(r, c, d);\n    }\n  ";
}
function getOutput4DCoords(shape, texShape) {
    var stride2 = shape[3];
    var stride1 = shape[2] * stride2;
    var stride0 = shape[1] * stride1;
    return "\n    vec4 getOutputCoords() {\n      vec2 resTexRC = floor(gl_FragCoord.yx);\n      float index = dot(resTexRC, vec2(" + texShape[1] + ".0, 1.0));\n\n      float r = floor(index / " + stride0 + ".0);\n      index -= r * " + stride0 + ".0;\n\n      float c = floor(index / " + stride1 + ".0);\n      index -= c * " + stride1 + ".0;\n\n      float d = floor(index / " + stride2 + ".0);\n      float d2 = mod(index, " + stride2 + ".0);\n\n      return vec4(r, c, d, d2);\n    }\n  ";
}
function getOutput2DCoords(shape, texShape) {
    if (util.arraysEqual(shape, texShape)) {
        return "\n      vec2 getOutputCoords() {\n        return floor(gl_FragCoord.yx);\n      }\n    ";
    }
    return "\n    vec2 getOutputCoords() {\n      vec2 resTexRC = floor(gl_FragCoord.yx);\n      float index = dot(resTexRC, vec2(" + texShape[1] + ".0, 1.0));\n      float r = floor(index / " + shape[1] + ".0);\n      float c = mod(index, " + shape[1] + ".0);\n      return vec2(r, c);\n    }\n  ";
}
function getSamplerScalar(texName) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    return "\n    float " + funcName + "() {\n      return sample(" + texName + ", halfCR);\n    }\n  ";
}
function getSampler1D(texName, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var tR = texShape[0];
    var tC = texShape[1];
    if (texShape[0] === 1 && texShape[1] === 1) {
        return "\n      float " + funcName + "(float index) {\n        return sample(" + texName + ", halfCR);\n      }\n    ";
    }
    if (texShape[1] === 1) {
        return "\n      float " + funcName + "(float index) {\n        vec2 uv = vec2(0.5, (index + 0.5) / " + tR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    if (texShape[0] === 1) {
        return "\n      float " + funcName + "(float index) {\n        vec2 uv = vec2((index + 0.5) / " + tC + ".0, 0.5);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(float index) {\n      vec2 uv = UVfrom1D(" + tR + ".0, " + tC + ".0, index);\n      return sample(" + texName + ", uv);\n    }\n  ";
}
function getSampler3D(texName, shape, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var tR = texShape[0];
    var tC = texShape[1];
    var stride0 = shape[1] * shape[2];
    var stride1 = shape[2];
    if (tC === stride0) {
        return "\n      float " + funcName + "(float row, float col, float depth) {\n        float texR = row;\n        float texC = dot(vec2(col, depth), vec2(" + stride1 + ", 1.0));\n        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(" + tC + ".0, " + tR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(float row, float col, float depth) {\n      vec2 uv = UVfrom3D(" + tR + ".0, " + tC + ".0, " + stride0 + ".0, " + stride1 + ".0, row,\n        col, depth);\n      return sample(" + texName + ", uv);\n    }\n  ";
}
function getSampler4D(texName, shape, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var tR = texShape[0];
    var tC = texShape[1];
    var stride2 = shape[3];
    var stride1 = shape[2] * stride2;
    var stride0 = shape[1] * stride1;
    if (tC === stride0) {
        return "\n      float " + funcName + "(float row, float col, float depth, float depth2) {\n        float texR = row;\n        float texC = dot(vec3(col, depth, depth2),\n                         vec3(" + stride1 + ".0, " + stride2 + ".0, 1.0));\n        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(" + tC + ".0, " + tR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(float row, float col, float depth, float depth2) {\n      vec2 uv = UVfrom4D(" + tR + ".0, " + tC + ".0, " + stride0 + ".0, " + stride1 + ".0,\n          " + stride2 + ".0, row, col, depth, depth2);\n      return sample(" + texName + ", uv);\n    }\n  ";
}
function getSampler2D(texName, shape, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var tR = texShape[0];
    var tC = texShape[1];
    if (util.arraysEqual(shape, texShape)) {
        return "\n      float " + funcName + "(float row, float col) {\n        vec2 uv = (vec2(col, row) + halfCR) / vec2(" + tC + ".0, " + tR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    if (tC === 1) {
        return "\n      float " + funcName + "(float row, float col) {\n        float index = dot(vec2(row, col), vec2(" + shape[1] + ".0, 1.0));\n        vec2 uv = vec2(0.5, (index + 0.5) / " + tR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    if (tR === 1) {
        return "\n      float " + funcName + "(float row, float col) {\n        float index = dot(vec2(row, col), vec2(" + shape[1] + ".0, 1.0));\n        vec2 uv = vec2((index + 0.5) / " + tC + ".0, 0.5);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(float row, float col) {\n      vec2 uv = UVfrom2D(" + tR + ".0, " + tC + ".0, " + shape[1] + ".0, row, col);\n      return sample(" + texName + ", uv);\n    }\n  ";
}
function getSamplerFlat(texName, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1) + 'Flat';
    var tNumR = texShape[0];
    var tNumC = texShape[1];
    if (tNumC === 1 && tNumR === 1) {
        return "\n      float " + funcName + "(float index) {\n        return sample(" + texName + ", halfCR);\n      }\n    ";
    }
    if (tNumC === 1) {
        return "\n      float " + funcName + "(float index) {\n        vec2 uv = vec2(0.5, (index + 0.5) / " + tNumR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    if (tNumR === 1) {
        return "\n      float " + funcName + "(float index) {\n        vec2 uv = vec2((index + 0.5) / " + tNumC + ".0, 0.5);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(float index) {\n      float texR = floor(index / " + tNumC + ".0);\n      float texC = mod(index, " + tNumC + ".0);\n      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(" + tNumC + ".0, " + tNumR + ".0);\n      return sample(" + texName + ", uv);\n    }\n  ";
}
function getSamplerAtOutputCoords(texName, inTexShape, outTexShape, broadcast) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1) +
        'AtOutCoords';
    if (util.arraysEqual(inTexShape, outTexShape)) {
        return "\n      float " + funcName + "() {\n        return sample(" + texName + ", resultUV);\n      }\n    ";
    }
    var inSize = util.sizeFromShape(inTexShape);
    var broadcastSnippet = broadcast ? "index = mod(index, " + inSize + ".0);" : '';
    return "\n    float " + funcName + "() {\n      vec2 resTexRC = floor(gl_FragCoord.yx);\n      float index = dot(resTexRC, vec2(" + outTexShape[1] + ".0, 1.0));\n      " + broadcastSnippet + "\n      float texR = floor(index / " + inTexShape[1] + ".0);\n      float texC = mod(index, " + inTexShape[1] + ".0);\n      vec2 uv = (vec2(texC, texR) + halfCR) /\n                 vec2(" + inTexShape[1] + ".0, " + inTexShape[0] + ".0);\n      return sample(" + texName + ", uv);\n    }\n  ";
}

},{"../../util":34}],30:[function(require,module,exports){
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
var TextureManager = (function () {
    function TextureManager(gpgpu) {
        this.gpgpu = gpgpu;
        this.numUsedTextures = 0;
        this.numFreeTextures = 0;
        this.freeTextures = {};
        this.logEnabled = false;
        this.usedTextureCount = {};
    }
    TextureManager.prototype.acquireTexture = function (shapeRC) {
        var shapeKey = getKeyFromTextureShape(shapeRC);
        if (!(shapeKey in this.freeTextures)) {
            this.freeTextures[shapeKey] = [];
        }
        if (!(shapeKey in this.usedTextureCount)) {
            this.usedTextureCount[shapeKey] = 0;
        }
        this.usedTextureCount[shapeKey]++;
        if (this.freeTextures[shapeKey].length > 0) {
            this.numFreeTextures--;
            this.numUsedTextures++;
            this.log();
            return this.freeTextures[shapeKey].shift();
        }
        this.numUsedTextures++;
        this.log();
        return this.gpgpu.createMatrixTexture(shapeRC[0], shapeRC[1]);
    };
    TextureManager.prototype.releaseTexture = function (texture, shape) {
        var shapeKey = getKeyFromTextureShape(shape);
        if (!(shapeKey in this.freeTextures)) {
            this.freeTextures[shapeKey] = [];
        }
        this.freeTextures[shapeKey].push(texture);
        this.numFreeTextures++;
        this.numUsedTextures--;
        this.usedTextureCount[shapeKey]--;
        this.log();
    };
    TextureManager.prototype.log = function () {
        if (!this.logEnabled) {
            return;
        }
        var total = this.numFreeTextures + this.numUsedTextures;
        console.log('Free/Used', this.numFreeTextures + ' / ' + this.numUsedTextures, "(" + total + ")");
    };
    TextureManager.prototype.getNumUsedTextures = function () {
        return this.numUsedTextures;
    };
    TextureManager.prototype.getNumFreeTextures = function () {
        return this.numFreeTextures;
    };
    TextureManager.prototype.dispose = function () {
        for (var shape in this.freeTextures) {
            if (this.freeTextures.hasOwnProperty(shape)) {
                for (var i = 0; i < this.freeTextures[shape].length; i++) {
                    this.gpgpu.deleteMatrixTexture(this.freeTextures[shape][i]);
                }
            }
        }
    };
    return TextureManager;
}());
exports.TextureManager = TextureManager;
function getKeyFromTextureShape(shapeRowsCol) {
    return shapeRowsCol[0] + '_' + shapeRowsCol[1];
}

},{}],32:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var USE_WEBGL2_WHEN_AVAILABLE = true;
var WEBGL2_ENABLED = null;
var MAX_TEXTURE_SIZE = null;
var util = require("../../util");
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
function getTextureShapeFromLogicalShape(gl, logShape, preferredTexShape) {
    var maxTexSize = queryMaxTextureSize(gl);
    var size = util.sizeFromShape(logShape);
    if (preferredTexShape != null) {
        var sizePreferred = util.sizeFromShape(preferredTexShape);
        util.assert(size === sizePreferred, "Size of shape (" + size + ") must match size of " +
            ("preferredShape (" + sizePreferred + ")"));
        if (preferredTexShape[0] <= maxTexSize &&
            preferredTexShape[1] <= maxTexSize) {
            return preferredTexShape;
        }
    }
    if (logShape.length <= 1 && size <= maxTexSize) {
        return [size, 1];
    }
    else if (logShape.length === 2 && logShape[0] <= maxTexSize &&
        logShape[1] <= maxTexSize) {
        return logShape;
    }
    else if (logShape.length === 3 && logShape[0] <= maxTexSize &&
        logShape[1] * logShape[2] <= maxTexSize) {
        return [logShape[0], logShape[1] * logShape[2]];
    }
    else if (logShape.length === 4 && logShape[0] <= maxTexSize &&
        logShape[1] * logShape[2] * logShape[3] <= maxTexSize) {
        return [logShape[0], logShape[1] * logShape[2] * logShape[3]];
    }
    else {
        return util.sizeToSquarishShape(size);
    }
}
exports.getTextureShapeFromLogicalShape = getTextureShapeFromLogicalShape;

},{"../../util":34}],33:[function(require,module,exports){
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

},{}],34:[function(require,module,exports){
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
function assertAndGetBroadcastedShape(shapeA, shapeB) {
    var result = [];
    var nextADimMustBeOne = false;
    var nextBDimMustBeOne = false;
    var errMsg = "Operands could not be broadcast together with shapes " +
        (shapeA + " and " + shapeB + ". Currently, we only support a ") +
        "stricter version of broadcasting than numpy.";
    var l = Math.max(shapeA.length, shapeB.length);
    shapeA = shapeA.slice().reverse();
    shapeB = shapeB.slice().reverse();
    for (var i = 0; i < l; i++) {
        var a = shapeA[i] || 1;
        var b = shapeB[i] || 1;
        if ((b > 1 && nextBDimMustBeOne) || (a > 1 && nextADimMustBeOne)) {
            throw Error(errMsg);
        }
        if (a > 1 && b === 1) {
            nextBDimMustBeOne = true;
        }
        if (b > 1 && a === 1) {
            nextADimMustBeOne = true;
        }
        if (a > 1 && b > 1 && a !== b) {
            throw Error(errMsg);
        }
        result.push(Math.max(a, b));
    }
    return result.reverse();
}
exports.assertAndGetBroadcastedShape = assertAndGetBroadcastedShape;

},{}]},{},[7])
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJkZW1vcy9iZW5jaG1hcmtzL2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvY29udl9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9jb252X3RyYW5zcG9zZV9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9sb2dzdW1leHBfY3B1X2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvbG9nc3VtZXhwX2dwdV9iZW5jaG1hcmsudHMiLCJkZW1vcy9iZW5jaG1hcmtzL21hdGgtYmVuY2htYXJrLXJ1bi1ncm91cHMudHMiLCJkZW1vcy9iZW5jaG1hcmtzL21hdGgtYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9tYXhfcG9vbF9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9tdWxtYXRfY3B1X2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvbXVsbWF0X2dwdV9iZW5jaG1hcmsudHMiLCJkZW1vcy9kZW1vLWZvb3Rlci50cyIsImRlbW9zL2RlbW8taGVhZGVyLnRzIiwiZGVtb3MvcG9seW1lci1zcGVjLnRzIiwic3JjL21hdGgvY29uY2F0M2RfdXRpbC50cyIsInNyYy9tYXRoL2NvbnZfdXRpbC50cyIsInNyYy9tYXRoL2NvcHkyZF91dGlsLnRzIiwic3JjL21hdGgvbWF0aC50cyIsInNyYy9tYXRoL21hdGhfY3B1LnRzIiwic3JjL21hdGgvbmRhcnJheS50cyIsInNyYy9tYXRoL3dlYmdsL2NvbnZfYmFja3Byb3BfZ3B1LnRzIiwic3JjL21hdGgvd2ViZ2wvY29udl9ncHUudHMiLCJzcmMvbWF0aC93ZWJnbC9ncGdwdV9jb250ZXh0LnRzIiwic3JjL21hdGgvd2ViZ2wvZ3BncHVfbWF0aC50cyIsInNyYy9tYXRoL3dlYmdsL2dwZ3B1X3V0aWwudHMiLCJzcmMvbWF0aC93ZWJnbC9sb2dzdW1leHBfZ3B1LnRzIiwic3JjL21hdGgvd2ViZ2wvbXVsbWF0X2dwdS50cyIsInNyYy9tYXRoL3dlYmdsL211bG1hdF9wYWNrZWRfZ3B1LnRzIiwic3JjL21hdGgvd2ViZ2wvcG9vbF9ncHUudHMiLCJzcmMvbWF0aC93ZWJnbC9zaGFkZXJfY29tcGlsZXIudHMiLCJzcmMvbWF0aC93ZWJnbC90ZXhfdXRpbC50cyIsInNyYy9tYXRoL3dlYmdsL3RleHR1cmVfbWFuYWdlci50cyIsInNyYy9tYXRoL3dlYmdsL3dlYmdsX3V0aWwudHMiLCJzcmMvdGVzdF91dGlsLnRzIiwic3JjL3V0aWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7OztBQzJCQTtJQUtFLHNCQUFZLElBQVksRUFBRSxhQUE0QjtRQUNwRCxJQUFJLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQztRQUNqQixJQUFJLENBQUMsYUFBYSxHQUFHLGFBQWEsQ0FBQztRQUNuQyxJQUFJLENBQUMsU0FBUyxHQUFHLEVBQUUsQ0FBQztJQUN0QixDQUFDO0lBQ0gsbUJBQUM7QUFBRCxDQVZBLEFBVUMsSUFBQTtBQVZZLG9DQUFZOzs7OztBQ1p6QixvREFBc0Q7QUFDdEQsa0RBQWdGO0FBQ2hGLDBEQUE0RDtBQUM1RCxvRUFBZ0U7QUFDaEUsNERBQThEO0FBQzlELHdFQUFvRTtBQUlwRSxJQUFNLE9BQU8sR0FBRyxFQUFFLENBQUM7QUFFTixRQUFBLGNBQWMsR0FBa0IsVUFBQyxJQUFZO0lBQ3hELElBQU0sS0FBSyxHQUFHLElBQUksNEJBQVksRUFBRSxDQUFDO0lBQ2pDLElBQU0sVUFBVSxHQUFHLElBQUksZ0NBQWMsQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUM3Qyx1QkFBYSxDQUFDLEtBQUssRUFBRSxVQUFVLENBQUMsQ0FBQztJQUVqQyxJQUFNLE9BQU8sR0FBRyxDQUFDLENBQUM7SUFDbEIsSUFBTSxPQUFPLEdBQTZCLENBQUMsSUFBSSxFQUFFLElBQUksRUFBRSxPQUFPLENBQUMsQ0FBQztJQUNoRSxJQUFNLFFBQVEsR0FBRyxDQUFDLENBQUM7SUFD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QUFDLENBQUMsUUFBUSxDQUFDLE1BQU0sSUFBSSxDQUFDLElBQUksSUFBSSxJQUFJLFVBQVUsQ0FBQyxDQUFDLENBQUM7UUFDL0MsTUFBTSxDQUFDLENBQUMsSUFBSSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ25CLENBQUM7SUFBQyxJQUFJLENBQUMsRUFBRSxDQUFDLENBQ04sUUFBUSxDQUFDLE1BQU0sS0FBSyxDQUFDLElBQUksUUFBUSxDQUFDLENBQUMsQ0FBQyxJQUFJLFVBQVU7UUFDbEQsUUFBUSxDQUFDLENBQUMsQ0FBQyxJQUFJLFVBQVUsQ0FBQyxDQUFDLENBQUM7UUFDOUIsTUFBTSxDQUFDLFFBQTRCLENBQUM7SUFDdEMsQ0FBQztJQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FDTixRQUFRLENBQUMsTUFBTSxLQUFLLENBQUMsSUFBSSxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksVUFBVTtRQUNsRCxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxJQUFJLFVBQVUsQ0FBQyxDQUFDLENBQUM7UUFDNUMsTUFBTSxDQUFDLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNsRCxDQUFDO0lBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUNOLFFBQVEsQ0FBQyxNQUFNLEtBQUssQ0FBQyxJQUFJLFFBQVEsQ0FBQyxDQUFDLENBQUMsSUFBSSxVQUFVO1FBQ2xELFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxJQUFJLFVBQVUsQ0FBQyxDQUFDLENBQUM7UUFDMUQsTUFBTSxDQUFDLENBQUMsUUFBUSxDQUFDLENBQUMsQ0FBQyxFQUFFLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7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WUFDckIsaUJBQWlCLEdBQUcsSUFBSSxDQUFDO1FBQzNCLENBQUM7UUFDRCxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDOUIsTUFBTSxLQUFLLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDdEIsQ0FBQztRQUNELE1BQU0sQ0FBQyxJQUFJLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM5QixDQUFDO0lBQ0QsTUFBTSxDQUFDLE1BQU0sQ0FBQyxPQUFPLEVBQUUsQ0FBQztBQUMxQixDQUFDO0FBOUJELG9FQThCQyIsImZpbGUiOiJnZW5lcmF0ZWQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlc0NvbnRlbnQiOlsiKGZ1bmN0aW9uIGUodCxuLHIpe2Z1bmN0aW9uIHMobyx1KXtpZighbltvXSl7aWYoIXRbb10pe3ZhciBhPXR5cGVvZiByZXF1aXJlPT1cImZ1bmN0aW9uXCImJnJlcXVpcmU7aWYoIXUmJmEpcmV0dXJuIGEobywhMCk7aWYoaSlyZXR1cm4gaShvLCEwKTt2YXIgZj1uZXcgRXJyb3IoXCJDYW5ub3QgZmluZCBtb2R1bGUgJ1wiK28rXCInXCIpO3Rocm93IGYuY29kZT1cIk1PRFVMRV9OT1RfRk9VTkRcIixmfXZhciBsPW5bb109e2V4cG9ydHM6e319O3Rbb11bMF0uY2FsbChsLmV4cG9ydHMsZnVuY3Rpb24oZSl7dmFyIG49dFtvXVsxXVtlXTtyZXR1cm4gcyhuP246ZSl9LGwsbC5leHBvcnRzLGUsdCxuLHIpfXJldHVybiBuW29dLmV4cG9ydHN9dmFyIGk9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtmb3IodmFyIG89MDtvPHIubGVuZ3RoO28rKylzKHJbb10pO3JldHVybiBzfSkiLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmV4cG9ydCBpbnRlcmZhY2UgQmVuY2htYXJrUnVuR3JvdXAge1xuICBuYW1lOiBzdHJpbmc7XG4gIC8vIE1pbiBhbmQgbWF4IHN0ZXBzIHRvIHJ1biB0aGUgYmVuY2htYXJrIHRlc3Qgb3Zlci5cbiAgbWluOiBudW1iZXI7XG4gIG1heDogbnVtYmVyO1xuICAvLyBUaGUgc2l6ZSBvZiB0aGUgc3RlcCB0byB0YWtlIGJldHdlZW4gYmVuY2htYXJrIHJ1bnMuXG4gIHN0ZXBTaXplOiBudW1iZXI7XG4gIC8vIEEgdHJhbnNmb3JtYXRpb24gb2Ygc3RlcCB0byB0aGUgc2l6ZSBwYXNzZWQgdG8gdGhlIGJlbmNobWFyayB0ZXN0LlxuICBzdGVwVG9TaXplVHJhbnNmb3JtYXRpb24/OiAoc3RlcDogbnVtYmVyKSA9PiBudW1iZXI7XG4gIGJlbmNobWFya1J1bnM6IEJlbmNobWFya1J1bltdO1xufVxuXG5leHBvcnQgY2xhc3MgQmVuY2htYXJrUnVuIHtcbiAgbmFtZTogc3RyaW5nO1xuICBiZW5jaG1hcmtUZXN0OiBCZW5jaG1hcmtUZXN0O1xuXG4gIGNoYXJ0RGF0YTogQ2hhcnREYXRhW107XG4gIGNvbnN0cnVjdG9yKG5hbWU6IHN0cmluZywgYmVuY2htYXJrVGVzdDogQmVuY2htYXJrVGVzdCkge1xuICAgIHRoaXMubmFtZSA9IG5hbWU7XG4gICAgdGhpcy5iZW5jaG1hcmtUZXN0ID0gYmVuY2htYXJrVGVzdDtcbiAgICB0aGlzLmNoYXJ0RGF0YSA9IFtdO1xuICB9XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgQmVuY2htYXJrVGVzdCB7IChzaXplOiBudW1iZXIpOiBudW1iZXI7IH1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uLy4uL3NyYy9tYXRoL2NvbnZfdXRpbCc7XG5pbXBvcnQge0FycmF5MUQsIEFycmF5M0QsIEFycmF5NEQsIGluaXRpYWxpemVHUFV9IGZyb20gJy4uLy4uL3NyYy9tYXRoL25kYXJyYXknO1xuaW1wb3J0IHtDb252MkRQcm9ncmFtfSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9jb252X2dwdSc7XG5pbXBvcnQge0dQR1BVQ29udGV4dH0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvZ3BncHVfY29udGV4dCc7XG5pbXBvcnQgKiBhcyBncGdwdV9tYXRoIGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X21hdGgnO1xuaW1wb3J0IHtUZXh0dXJlTWFuYWdlcn0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvdGV4dHVyZV9tYW5hZ2VyJztcblxuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QX1JVTlMgPSA0MDtcblxuZXhwb3J0IGNvbnN0IEJFTkNITUFSS19URVNUOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBjb25zdCBncGdwdSA9IG5ldyBHUEdQVUNvbnRleHQoKTtcbiAgY29uc3QgdGV4TWFuYWdlciA9IG5ldyBUZXh0dXJlTWFuYWdlcihncGdwdSk7XG4gIGluaXRpYWxpemVHUFUoZ3BncHUsIHRleE1hbmFnZXIpO1xuXG4gIGNvbnN0IGluRGVwdGggPSAxO1xuICBjb25zdCBpblNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbc2l6ZSwgc2l6ZSwgaW5EZXB0aF07XG4gIGNvbnN0IG91dERlcHRoID0gMTtcbiAgY29uc3QgZmlsdGVyU2l6ZSA9IDExO1xuICBjb25zdCBzdHJpZGUgPSAxO1xuICBjb25zdCB6ZXJvUGFkID0gY29udl91dGlsLmNvbXB1dGVEZWZhdWx0UGFkKGluU2hhcGUsIGZpbHRlclNpemUsIHN0cmlkZSk7XG5cbiAgY29uc3QgaGFzQmlhcyA9IHRydWU7XG4gIGNvbnN0IG91dHB1dEluZm8gPSBjb252X3V0aWwuY29tcHV0ZU91dHB1dEluZm8oXG4gICAgICBpblNoYXBlLCBmaWx0ZXJTaXplLCBmaWx0ZXJTaXplLCBvdXREZXB0aCwgc3RyaWRlLCBzdHJpZGUsIHplcm9QYWQpO1xuICBjb25zdCBwcm9ncmFtID0gbmV3IENvbnYyRFByb2dyYW0oXG4gICAgICBpblNoYXBlLCBmaWx0ZXJTaXplLCBmaWx0ZXJTaXplLCBzdHJpZGUsIHN0cmlkZSwgb3V0cHV0SW5mbywgaGFzQmlhcyk7XG4gIGNvbnN0IG91dHB1dFNoYXBlID0gcHJvZ3JhbS5vdXRwdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIGNvbnN0IG91dCA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICBjb25zdCB4ID0gQXJyYXkzRC5yYW5kVW5pZm9ybShpblNoYXBlLCAtMSwgMSk7XG4gIGNvbnN0IHdTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlV2VpZ2h0c1NoYXBlNEQoMSwgb3V0RGVwdGgsIGZpbHRlclNpemUpO1xuICBjb25zdCBXID0gQXJyYXk0RC5yYW5kVW5pZm9ybSh3U2hhcGUsIC0xLCAxKTtcbiAgY29uc3QgYiA9IEFycmF5MUQucmFuZFVuaWZvcm0oW291dERlcHRoXSwgLTEsIDEpO1xuICBjb25zdCBpbnB1dHMgPSBbeCwgVywgYl07XG4gIGNvbnN0IGJpbmFyeSA9IGdwZ3B1X21hdGguY29tcGlsZVByb2dyYW0oZ3BncHUsIHByb2dyYW0sIGlucHV0cywgb3V0KTtcblxuICBjb25zdCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IE9QX1JVTlM7IGkrKykge1xuICAgIGdwZ3B1X21hdGgucnVuUHJvZ3JhbShiaW5hcnksIGlucHV0cywgb3V0KTtcbiAgfVxuICBvdXQuZ2V0VmFsdWVzKCk7XG4gIGNvbnN0IGF2Z1RpbWUgPSAocGVyZm9ybWFuY2Uubm93KCkgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIHguZGlzcG9zZSgpO1xuICBXLmRpc3Bvc2UoKTtcbiAgYi5kaXNwb3NlKCk7XG4gIG91dC5kaXNwb3NlKCk7XG4gIHRleE1hbmFnZXIuZGlzcG9zZSgpO1xuICBncGdwdS5kZWxldGVQcm9ncmFtKGJpbmFyeS53ZWJHTFByb2dyYW0pO1xuICBncGdwdS5kaXNwb3NlKCk7XG5cbiAgcmV0dXJuIGF2Z1RpbWU7XG59O1xuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyBjb252X3V0aWwgZnJvbSAnLi4vLi4vc3JjL21hdGgvY29udl91dGlsJztcbmltcG9ydCB7QXJyYXkzRCwgQXJyYXk0RCwgaW5pdGlhbGl6ZUdQVX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvbmRhcnJheSc7XG5pbXBvcnQge0NvbnYyRFRyYW5zcG9zZVByb2dyYW19IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2NvbnZfYmFja3Byb3BfZ3B1JztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9ncGdwdV9jb250ZXh0JztcbmltcG9ydCAqIGFzIGdwZ3B1X21hdGggZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvZ3BncHVfbWF0aCc7XG5pbXBvcnQge1RleHR1cmVNYW5hZ2VyfSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC90ZXh0dXJlX21hbmFnZXInO1xuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QX1JVTlMgPSA0MDtcblxuZXhwb3J0IGNvbnN0IEJFTkNITUFSS19URVNUOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IDE7XG4gIGNvbnN0IG9yaWdPdXRwdXREZXB0aCA9IDI7XG4gIGNvbnN0IHhTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gW3NpemUsIHNpemUsIDFdO1xuICBjb25zdCBmaWVsZFNpemUgPSAxMTtcbiAgY29uc3Qgb3JpZ1N0cmlkZSA9IDE7XG4gIGNvbnN0IG9yaWdQYWQgPSAxO1xuXG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCB0ZXhNYW5hZ2VyID0gbmV3IFRleHR1cmVNYW5hZ2VyKGdwZ3B1KTtcbiAgaW5pdGlhbGl6ZUdQVShncGdwdSwgdGV4TWFuYWdlcik7XG4gIGdwZ3B1LmVuYWJsZUF1dG9tYXRpY0RlYnVnVmFsaWRhdGlvbih0cnVlKTtcblxuICBjb25zdCBoYXNCaWFzID0gZmFsc2U7XG4gIGNvbnN0IHByb2dyYW0gPSBuZXcgQ29udjJEVHJhbnNwb3NlUHJvZ3JhbShcbiAgICAgIHhTaGFwZSwgZmllbGRTaXplLCBvcmlnSW5wdXREZXB0aCwgb3JpZ1N0cmlkZSwgb3JpZ1BhZCwgaGFzQmlhcyk7XG4gIGNvbnN0IG91dHB1dFNoYXBlID0gcHJvZ3JhbS5vdXRwdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIGNvbnN0IG91dCA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICBjb25zdCB4ID0gQXJyYXkzRC5yYW5kVW5pZm9ybSh4U2hhcGUsIC0xLCAxKTtcbiAgY29uc3Qgd1NoYXBlID0gY29udl91dGlsLmNvbXB1dGVXZWlnaHRzU2hhcGU0RChcbiAgICAgIG9yaWdJbnB1dERlcHRoLCBvcmlnT3V0cHV0RGVwdGgsIGZpZWxkU2l6ZSk7XG4gIGNvbnN0IFcgPSBBcnJheTRELnJhbmRVbmlmb3JtKHdTaGFwZSwgLTEsIDEpO1xuICBjb25zdCBpbnB1dHMgPSBbeCwgV107XG4gIGNvbnN0IGJpbmFyeSA9IGdwZ3B1X21hdGguY29tcGlsZVByb2dyYW0oZ3BncHUsIHByb2dyYW0sIGlucHV0cywgb3V0KTtcbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUF9SVU5TOyBpKyspIHtcbiAgICBncGdwdV9tYXRoLnJ1blByb2dyYW0oYmluYXJ5LCBpbnB1dHMsIG91dCk7XG4gIH1cbiAgb3V0LmdldFZhbHVlcygpO1xuICBjb25zdCBhdmdUaW1lID0gKHBlcmZvcm1hbmNlLm5vdygpIC0gc3RhcnQpIC8gT1BfUlVOUztcblxuICB0ZXhNYW5hZ2VyLmRpc3Bvc2UoKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShiaW5hcnkud2ViR0xQcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuICByZXR1cm4gYXZnVGltZTtcbn07XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7TkRBcnJheU1hdGhDUFV9IGZyb20gJy4uLy4uL3NyYy9tYXRoL21hdGhfY3B1JztcbmltcG9ydCB7QXJyYXkyRCwgTkRBcnJheX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvbmRhcnJheSc7XG5cbmltcG9ydCB7QmVuY2htYXJrVGVzdH0gZnJvbSAnLi9iZW5jaG1hcmsnO1xuXG5jb25zdCBPUFNfUEVSX1JVTiA9IDEwO1xuXG5leHBvcnQgY29uc3QgQkVOQ0hNQVJLX1RFU1Q6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IG1hdGggPSBuZXcgTkRBcnJheU1hdGhDUFUoKTtcbiAgY29uc3QgYSA9IE5EQXJyYXkucmFuZFVuaWZvcm08QXJyYXkyRD4oW3NpemUsIHNpemVdLCAtMSwgMSk7XG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BTX1BFUl9SVU47IGkrKykge1xuICAgIG1hdGgubG9nU3VtRXhwKGEpO1xuICB9XG4gIGNvbnN0IGVuZCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICByZXR1cm4gKGVuZCAtIHN0YXJ0KSAvIE9QU19QRVJfUlVOO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtBcnJheTJELCBpbml0aWFsaXplR1BVLCBTY2FsYXJ9IGZyb20gJy4uLy4uL3NyYy9tYXRoL25kYXJyYXknO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgZ3BncHVfbWF0aCBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9ncGdwdV9tYXRoJztcbmltcG9ydCB7TG9nU3VtRXhwUHJvZ3JhbX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvbG9nc3VtZXhwX2dwdSc7XG5pbXBvcnQge1RleHR1cmVNYW5hZ2VyfSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC90ZXh0dXJlX21hbmFnZXInO1xuXG5pbXBvcnQge0JlbmNobWFya1Rlc3R9IGZyb20gJy4vYmVuY2htYXJrJztcblxuY29uc3QgT1BfUlVOUyA9IDI7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgZ3BncHUgPSBuZXcgR1BHUFVDb250ZXh0KCk7XG4gIGNvbnN0IHRleE1hbmFnZXIgPSBuZXcgVGV4dHVyZU1hbmFnZXIoZ3BncHUpO1xuICBpbml0aWFsaXplR1BVKGdwZ3B1LCB0ZXhNYW5hZ2VyKTtcbiAgY29uc3Qgb3V0ID0gbmV3IFNjYWxhcih7dGV4dHVyZTogdGV4TWFuYWdlci5hY3F1aXJlVGV4dHVyZShbMSwgMV0pfSk7XG4gIGNvbnN0IGEgPSBBcnJheTJELnJhbmRVbmlmb3JtKFtzaXplLCBzaXplXSwgLTEsIDEpO1xuICBjb25zdCBwcm9ncmFtID0gbmV3IExvZ1N1bUV4cFByb2dyYW0oYS5zaXplKTtcbiAgY29uc3QgYmluYXJ5ID0gZ3BncHVfbWF0aC5jb21waWxlUHJvZ3JhbShncGdwdSwgcHJvZ3JhbSwgW2FdLCBvdXQpO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgZ3BncHVfbWF0aC5ydW5Qcm9ncmFtKGJpbmFyeSwgW2FdLCBvdXQpO1xuICB9XG4gIG91dC5nZXRWYWx1ZXMoKTtcbiAgY29uc3QgYXZnVGltZSA9IChwZXJmb3JtYW5jZS5ub3coKSAtIHN0YXJ0KSAvIE9QX1JVTlM7XG4gIGEuZGlzcG9zZSgpO1xuICBvdXQuZGlzcG9zZSgpO1xuICB0ZXhNYW5hZ2VyLmRpc3Bvc2UoKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShiaW5hcnkud2ViR0xQcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtCZW5jaG1hcmtSdW4sIEJlbmNobWFya1J1bkdyb3VwfSBmcm9tICcuL2JlbmNobWFyayc7XG5pbXBvcnQgKiBhcyBjb252X2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9jb252X2dwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgY29udl90cmFuc3Bvc2VfZ3B1X2JlbmNobWFyayBmcm9tICcuL2NvbnZfdHJhbnNwb3NlX2dwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgbG9nc3VtZXhwX2NwdV9iZW5jaG1hcmsgZnJvbSAnLi9sb2dzdW1leHBfY3B1X2JlbmNobWFyayc7XG5pbXBvcnQgKiBhcyBsb2dzdW1leHBfZ3B1X2JlbmNobWFyayBmcm9tICcuL2xvZ3N1bWV4cF9ncHVfYmVuY2htYXJrJztcbmltcG9ydCAqIGFzIG1heF9wb29sX2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9tYXhfcG9vbF9ncHVfYmVuY2htYXJrJztcbmltcG9ydCAqIGFzIG11bG1hdF9jcHVfYmVuY2htYXJrIGZyb20gJy4vbXVsbWF0X2NwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgbXVsbWF0X2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9tdWxtYXRfZ3B1X2JlbmNobWFyayc7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfUlVOX0dST1VQUzogQmVuY2htYXJrUnVuR3JvdXBbXSA9IFtcbiAge1xuICAgIG5hbWU6XG4gICAgICAgICdNYXRyaXggTXVsdGlwbGljYXRpb24gKENQVSB2cyBHUFUpOiAnICtcbiAgICAgICAgICAgICdtYXRtdWwoW3NpemUsIHNpemVdLCBbc2l6ZSwgc2l6ZV0pJyxcbiAgICBtaW46IDAsXG4gICAgbWF4OiAxMDI0LFxuICAgIHN0ZXBTaXplOiA2NCxcbiAgICBzdGVwVG9TaXplVHJhbnNmb3JtYXRpb246IChzdGVwOiBudW1iZXIpID0+IE1hdGgubWF4KDEsIHN0ZXApLFxuICAgIGJlbmNobWFya1J1bnM6IFtcbiAgICAgIG5ldyBCZW5jaG1hcmtSdW4oJ211bG1hdF9ncHUnLCBtdWxtYXRfZ3B1X2JlbmNobWFyay5CRU5DSE1BUktfVEVTVCksXG4gICAgICBuZXcgQmVuY2htYXJrUnVuKCdtdWxtYXRfY3B1JywgbXVsbWF0X2NwdV9iZW5jaG1hcmsuQkVOQ0hNQVJLX1RFU1QpXG4gICAgXSxcbiAgfSxcbiAge1xuICAgIG5hbWU6ICdDb252b2x1dGlvbiAoR1BVKTogY29udiBvdmVyIGltYWdlIFtzaXplLCBzaXplLCAxXScsXG4gICAgbWluOiAwLFxuICAgIG1heDogMTAyNCxcbiAgICBzdGVwU2l6ZTogNjQsXG4gICAgc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uOiAoc3RlcDogbnVtYmVyKSA9PiBNYXRoLm1heCgxLCBzdGVwKSxcbiAgICBiZW5jaG1hcmtSdW5zOiBbbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgJ2QxPTEsIGQyPTEsIGY9MTEsIHM9MScsIGNvbnZfZ3B1X2JlbmNobWFyay5CRU5DSE1BUktfVEVTVCldLFxuICB9LFxuICB7XG4gICAgbmFtZTogJ0NvbnZvbHV0aW9uIFRyYW5zcG9zZWQgKEdQVSk6IGRlY29udiBvdmVyIGltYWdlIFtzaXplLCBzaXplLCAxXScsXG4gICAgbWluOiAwLFxuICAgIG1heDogMTAyNCxcbiAgICBzdGVwU2l6ZTogNjQsXG4gICAgc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uOiAoc3RlcDogbnVtYmVyKSA9PiBNYXRoLm1heCgxLCBzdGVwKSxcbiAgICBiZW5jaG1hcmtSdW5zOiBbbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgJ2QxPTEsIGQyPTEsIGY9MTEsIHM9MScsIGNvbnZfdHJhbnNwb3NlX2dwdV9iZW5jaG1hcmsuQkVOQ0hNQVJLX1RFU1QpXSxcbiAgfSxcbiAge1xuICAgIG5hbWU6ICdNYXggcG9vbCAoR1BVKScsXG4gICAgbWluOiAwLFxuICAgIG1heDogMTAyNCxcbiAgICBzdGVwU2l6ZTogNjQsXG4gICAgc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uOiAoc3RlcDogbnVtYmVyKSA9PiBNYXRoLm1heCgxLCBzdGVwKSxcbiAgICBiZW5jaG1hcmtSdW5zOiBbbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgJ2QxPTEsIGQyPTEsIGY9MTEsIHM9MScsXG4gICAgICAgIG1heF9wb29sX2dwdV9iZW5jaG1hcmsuTUFYX1BPT0xfQkVOQ0hNQVJLX1RFU1QpXSxcbiAgfSxcbiAge1xuICAgIG5hbWU6ICdMb2dTdW1FeHAgKENQVSB2cyBHUFUpOiBpbnB1dCBbc2l6ZSwgc2l6ZV0nLFxuICAgIG1pbjogMCxcbiAgICBtYXg6IDEwMjQsXG4gICAgc3RlcFNpemU6IDY0LFxuICAgIHN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbjogKHN0ZXA6IG51bWJlcikgPT4gTWF0aC5tYXgoMSwgc3RlcCksXG4gICAgYmVuY2htYXJrUnVuczogW1xuICAgICAgbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgICAnbG9nc3VtZXhwX2dwdScsIGxvZ3N1bWV4cF9ncHVfYmVuY2htYXJrLkJFTkNITUFSS19URVNUKSxcbiAgICAgIG5ldyBCZW5jaG1hcmtSdW4oJ2xvZ3N1bWV4cF9jcHUnLCBsb2dzdW1leHBfY3B1X2JlbmNobWFyay5CRU5DSE1BUktfVEVTVClcbiAgICBdLFxuICB9XG5dO1xuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgJy4uL2RlbW8taGVhZGVyJztcbmltcG9ydCAnLi4vZGVtby1mb290ZXInO1xuXG5pbXBvcnQge1BvbHltZXJFbGVtZW50LCBQb2x5bWVySFRNTEVsZW1lbnR9IGZyb20gJy4uL3BvbHltZXItc3BlYyc7XG5pbXBvcnQge0JlbmNobWFya1J1bkdyb3VwfSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmltcG9ydCB7QkVOQ0hNQVJLX1JVTl9HUk9VUFN9IGZyb20gJy4vbWF0aC1iZW5jaG1hcmstcnVuLWdyb3Vwcyc7XG5cbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTp2YXJpYWJsZS1uYW1lXG5leHBvcnQgbGV0IE1hdGhCZW5jaG1hcmtQb2x5bWVyOiBuZXcgKCkgPT4gUG9seW1lckhUTUxFbGVtZW50ID0gUG9seW1lckVsZW1lbnQoXG4gICAge2lzOiAnbWF0aC1iZW5jaG1hcmsnLCBwcm9wZXJ0aWVzOiB7YmVuY2htYXJrUnVuR3JvdXBOYW1lczogQXJyYXl9fSk7XG5cbmV4cG9ydCBjbGFzcyBNYXRoQmVuY2htYXJrIGV4dGVuZHMgTWF0aEJlbmNobWFya1BvbHltZXIge1xuICAvLyBQb2x5bWVyIHByb3BlcnRpZXMuXG4gIHByaXZhdGUgYmVuY2htYXJrUnVuR3JvdXBOYW1lczogc3RyaW5nW107XG4gIHByaXZhdGUgc3RvcE1lc3NhZ2VzOiBib29sZWFuW107XG5cbiAgcmVhZHkoKSB7XG4gICAgLy8gU2V0IHVwIHRoZSBiZW5jaG1hcmtzIFVJLlxuICAgIGNvbnN0IGJlbmNobWFya1J1bkdyb3VwTmFtZXM6IHN0cmluZ1tdID0gW107XG4gICAgdGhpcy5zdG9wTWVzc2FnZXMgPSBbXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IEJFTkNITUFSS19SVU5fR1JPVVBTLmxlbmd0aDsgaSsrKSB7XG4gICAgICBiZW5jaG1hcmtSdW5Hcm91cE5hbWVzLnB1c2goQkVOQ0hNQVJLX1JVTl9HUk9VUFNbaV0ubmFtZSk7XG4gICAgICB0aGlzLnN0b3BNZXNzYWdlcy5wdXNoKGZhbHNlKTtcbiAgICB9XG4gICAgdGhpcy5iZW5jaG1hcmtSdW5Hcm91cE5hbWVzID0gYmVuY2htYXJrUnVuR3JvdXBOYW1lcztcblxuICAgIC8vIEluIGEgc2V0VGltZW91dCB0byBsZXQgdGhlIFVJIHVwZGF0ZSBiZWZvcmUgd2UgYWRkIGV2ZW50IGxpc3RlbmVycy5cbiAgICBzZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgIGNvbnN0IHJ1bkJ1dHRvbnMgPSB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tdGVzdCcpO1xuICAgICAgY29uc3Qgc3RvcEJ1dHRvbnMgPSB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tc3RvcCcpO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBydW5CdXR0b25zLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIHJ1bkJ1dHRvbnNbaV0uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKSA9PiB7XG4gICAgICAgICAgdGhpcy5ydW5CZW5jaG1hcmtHcm91cChpKTtcbiAgICAgICAgfSk7XG4gICAgICAgIHN0b3BCdXR0b25zW2ldLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4ge1xuICAgICAgICAgIHRoaXMuc3RvcE1lc3NhZ2VzW2ldID0gdHJ1ZTtcbiAgICAgICAgfSk7XG4gICAgICB9XG4gICAgfSwgMCk7XG4gIH1cblxuICBwcml2YXRlIHJ1bkJlbmNobWFya0dyb3VwKGJlbmNobWFya1J1bkdyb3VwSW5kZXg6IG51bWJlcikge1xuICAgIGNvbnN0IGJlbmNobWFya1J1bkdyb3VwID0gQkVOQ0hNQVJLX1JVTl9HUk9VUFNbYmVuY2htYXJrUnVuR3JvdXBJbmRleF07XG5cbiAgICBjb25zdCBjYW52YXMgPSB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tcGxvdCcpW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdIGFzXG4gICAgICAgIEhUTUxDYW52YXNFbGVtZW50O1xuICAgIGNvbnN0IGNvbnRleHQgPSBjYW52YXMuZ2V0Q29udGV4dCgnMmQnKSBhcyBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQ7XG5cbiAgICBjb25zdCBkYXRhc2V0czogQ2hhcnREYXRhU2V0c1tdID0gW107XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBiZW5jaG1hcmtSdW5Hcm91cC5iZW5jaG1hcmtSdW5zLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBodWUgPSBNYXRoLmZsb29yKDM2MCAqIGkgLyBiZW5jaG1hcmtSdW5Hcm91cC5iZW5jaG1hcmtSdW5zLmxlbmd0aCk7XG4gICAgICBkYXRhc2V0cy5wdXNoKHtcbiAgICAgICAgZGF0YTogYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVuc1tpXS5jaGFydERhdGEsXG4gICAgICAgIGZpbGw6IGZhbHNlLFxuICAgICAgICBsYWJlbDogYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVuc1tpXS5uYW1lLFxuICAgICAgICBib3JkZXJDb2xvcjogJ2hzbCgnICsgaHVlICsgJywgMTAwJSwgNDAlKScsXG4gICAgICAgIGJhY2tncm91bmRDb2xvcjogJ2hzbCgnICsgaHVlICsgJywgMTAwJSwgNzAlKScsXG4gICAgICAgIHBvaW50UmFkaXVzOiAwLFxuICAgICAgICBwb2ludEhpdFJhZGl1czogNSxcbiAgICAgICAgYm9yZGVyV2lkdGg6IDEsXG4gICAgICAgIGxpbmVUZW5zaW9uOiAwXG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBjb25zdCBjaGFydCA9IG5ldyBDaGFydChjb250ZXh0LCB7XG4gICAgICB0eXBlOiAnbGluZScsXG4gICAgICBkYXRhOiB7ZGF0YXNldHN9LFxuICAgICAgb3B0aW9uczoge1xuICAgICAgICBhbmltYXRpb246IHtkdXJhdGlvbjogMH0sXG4gICAgICAgIHJlc3BvbnNpdmU6IGZhbHNlLFxuICAgICAgICBzY2FsZXM6IHtcbiAgICAgICAgICB4QXhlczogW3tcbiAgICAgICAgICAgIHR5cGU6ICdsaW5lYXInLFxuICAgICAgICAgICAgcG9zaXRpb246ICdib3R0b20nLFxuICAgICAgICAgICAgdGlja3M6IHtcbiAgICAgICAgICAgICAgbWluOiBiZW5jaG1hcmtSdW5Hcm91cC5taW4sXG4gICAgICAgICAgICAgIG1heDogYmVuY2htYXJrUnVuR3JvdXAubWF4LFxuICAgICAgICAgICAgICBzdGVwU2l6ZTogYmVuY2htYXJrUnVuR3JvdXAuc3RlcFNpemUsXG4gICAgICAgICAgICAgIGNhbGxiYWNrOiAobGFiZWw6IHN0cmluZykgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiBiZW5jaG1hcmtSdW5Hcm91cC5zdGVwVG9TaXplVHJhbnNmb3JtYXRpb24gIT0gbnVsbCA/XG4gICAgICAgICAgICAgICAgICAgIGJlbmNobWFya1J1bkdyb3VwLnN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbigrbGFiZWwpIDpcbiAgICAgICAgICAgICAgICAgICAgK2xhYmVsO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgICAgIH0gYXMgYW55ICAvLyBOb3RlOiB0aGUgdHlwaW5ncyBmb3IgdGhpcyBhcmUgaW5jb3JyZWN0LCBjYXN0IGFzIGFueS5cbiAgICAgICAgICB9XSxcbiAgICAgICAgICB5QXhlczogW3tcbiAgICAgICAgICAgIHRpY2tzOiB7XG4gICAgICAgICAgICAgIGNhbGxiYWNrOiAobGFiZWwsIGluZGV4LCBsYWJlbHMpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gbGFiZWwgKyAnbXMnO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9LFxuICAgICAgICAgIH1dXG4gICAgICAgIH0sXG4gICAgICAgIHRvb2x0aXBzOiB7bW9kZTogJ2xhYmVsJ30sXG4gICAgICAgIHRpdGxlOiB7dGV4dDogYmVuY2htYXJrUnVuR3JvdXAubmFtZX1cbiAgICAgIH1cbiAgICB9KTtcbiAgICBjYW52YXMuc3R5bGUuZGlzcGxheSA9ICdub25lJztcblxuICAgIGNvbnN0IHJ1bk1lc3NhZ2UgPVxuICAgICAgICB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tbWVzc2FnZScpW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdIGFzXG4gICAgICAgIEhUTUxFbGVtZW50O1xuICAgIHJ1bk1lc3NhZ2Uuc3R5bGUuZGlzcGxheSA9ICdibG9jayc7XG5cbiAgICBjb25zdCBydW5OdW1iZXJzVGFibGUgPVxuICAgICAgICB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tbnVtYmVycy10YWJsZScpW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdIGFzXG4gICAgICAgIEhUTUxFbGVtZW50O1xuICAgIHJ1bk51bWJlcnNUYWJsZS5pbm5lckhUTUwgPSAnJztcbiAgICBydW5OdW1iZXJzVGFibGUuc3R5bGUuZGlzcGxheSA9ICdub25lJztcblxuICAgIC8vIFNldCB1cCB0aGUgaGVhZGVyIGZvciB0aGUgdGFibGUuXG4gICAgY29uc3QgaGVhZGVycyA9IFsnc2l6ZSddO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVucy5sZW5ndGg7IGkrKykge1xuICAgICAgaGVhZGVycy5wdXNoKGJlbmNobWFya1J1bkdyb3VwLmJlbmNobWFya1J1bnNbaV0ubmFtZSk7XG4gICAgfVxuICAgIHJ1bk51bWJlcnNUYWJsZS5hcHBlbmRDaGlsZCh0aGlzLmJ1aWxkUnVuTnVtYmVyc1JvdyhoZWFkZXJzKSk7XG5cbiAgICB0aGlzLnJ1bkJlbmNobWFya1N0ZXBzKFxuICAgICAgICBjaGFydCwgYmVuY2htYXJrUnVuR3JvdXAsIGJlbmNobWFya1J1bkdyb3VwSW5kZXgsXG4gICAgICAgIGJlbmNobWFya1J1bkdyb3VwLm1pbik7XG4gIH1cblxuICBwcml2YXRlIGJ1aWxkUnVuTnVtYmVyc1Jvdyh2YWx1ZXM6IHN0cmluZ1tdKSB7XG4gICAgY29uc3QgcnVuTnVtYmVyUm93RWxlbWVudCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpO1xuICAgIHJ1bk51bWJlclJvd0VsZW1lbnQuY2xhc3NOYW1lID0gJ3J1bi1udW1iZXJzLXJvdyBtYXRoLWJlbmNobWFyayc7XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgcnVuTnVtYmVyQ2VsbEVsZW1lbnQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTtcbiAgICAgIHJ1bk51bWJlckNlbGxFbGVtZW50LmNsYXNzTmFtZSA9ICdydW4tbnVtYmVycy1jZWxsIG1hdGgtYmVuY2htYXJrJztcbiAgICAgIHJ1bk51bWJlckNlbGxFbGVtZW50LmlubmVyVGV4dCA9IHZhbHVlc1tpXTtcbiAgICAgIHJ1bk51bWJlclJvd0VsZW1lbnQuYXBwZW5kQ2hpbGQocnVuTnVtYmVyQ2VsbEVsZW1lbnQpO1xuICAgIH1cbiAgICByZXR1cm4gcnVuTnVtYmVyUm93RWxlbWVudDtcbiAgfVxuXG4gIHByaXZhdGUgcnVuQmVuY2htYXJrU3RlcHMoXG4gICAgICBjaGFydDogQ2hhcnQsIGJlbmNobWFya1J1bkdyb3VwOiBCZW5jaG1hcmtSdW5Hcm91cCxcbiAgICAgIGJlbmNobWFya1J1bkdyb3VwSW5kZXg6IG51bWJlciwgc3RlcDogbnVtYmVyKSB7XG4gICAgY29uc3QgcnVuTnVtYmVyc1RhYmxlID1cbiAgICAgICAgdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLW51bWJlcnMtdGFibGUnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICBIVE1MRWxlbWVudDtcbiAgICBpZiAoc3RlcCA+IGJlbmNobWFya1J1bkdyb3VwLm1heCB8fFxuICAgICAgICB0aGlzLnN0b3BNZXNzYWdlc1tiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSkge1xuICAgICAgdGhpcy5zdG9wTWVzc2FnZXNbYmVuY2htYXJrUnVuR3JvdXBJbmRleF0gPSBmYWxzZTtcblxuICAgICAgcnVuTnVtYmVyc1RhYmxlLnN0eWxlLmRpc3BsYXkgPSAnJztcblxuICAgICAgY29uc3QgY2FudmFzID1cbiAgICAgICAgICB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tcGxvdCcpW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdIGFzXG4gICAgICAgICAgSFRNTENhbnZhc0VsZW1lbnQ7XG4gICAgICBjYW52YXMuc3R5bGUuZGlzcGxheSA9ICdibG9jayc7XG4gICAgICBjaGFydC51cGRhdGUoKTtcblxuICAgICAgY29uc3QgcnVuTWVzc2FnZSA9XG4gICAgICAgICAgdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLW1lc3NhZ2UnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICAgIEhUTUxFbGVtZW50O1xuICAgICAgcnVuTWVzc2FnZS5zdHlsZS5kaXNwbGF5ID0gJ25vbmUnO1xuXG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3QgcnVuTnVtYmVyUm93RWxlbWVudCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpO1xuICAgIHJ1bk51bWJlclJvd0VsZW1lbnQuY2xhc3NOYW1lID0gJ3J1bi1udW1iZXJzLXJvdyBtYXRoLWJlbmNobWFyayc7XG5cbiAgICBjb25zdCByb3dWYWx1ZXM6IHN0cmluZ1tdID0gWycnICsgc3RlcF07XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBiZW5jaG1hcmtSdW5Hcm91cC5iZW5jaG1hcmtSdW5zLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBiZW5jaG1hcmtSdW4gPSBiZW5jaG1hcmtSdW5Hcm91cC5iZW5jaG1hcmtSdW5zW2ldO1xuICAgICAgY29uc3QgYmVuY2htYXJrVGVzdCA9IGJlbmNobWFya1J1bi5iZW5jaG1hcmtUZXN0O1xuXG4gICAgICBjb25zdCBzaXplID0gYmVuY2htYXJrUnVuR3JvdXAuc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uICE9IG51bGwgP1xuICAgICAgICAgIGJlbmNobWFya1J1bkdyb3VwLnN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbihzdGVwKSA6XG4gICAgICAgICAgc3RlcDtcblxuICAgICAgbGV0IHJlc3VsdFN0cmluZzogc3RyaW5nO1xuICAgICAgbGV0IGxvZ1N0cmluZzogc3RyaW5nO1xuICAgICAgbGV0IHRpbWUgPSAwO1xuICAgICAgbGV0IHN1Y2Nlc3MgPSB0cnVlO1xuXG4gICAgICB0cnkge1xuICAgICAgICB0aW1lID0gYmVuY2htYXJrVGVzdChzaXplKTtcbiAgICAgICAgcmVzdWx0U3RyaW5nID0gdGltZS50b0ZpeGVkKDMpICsgJ21zJztcbiAgICAgICAgbG9nU3RyaW5nID0gcmVzdWx0U3RyaW5nO1xuICAgICAgfSBjYXRjaCAoZSkge1xuICAgICAgICBzdWNjZXNzID0gZmFsc2U7XG4gICAgICAgIHJlc3VsdFN0cmluZyA9ICdFcnJvcic7XG4gICAgICAgIGxvZ1N0cmluZyA9IGUubWVzc2FnZTtcbiAgICAgIH1cblxuICAgICAgaWYgKHRpbWUgPj0gMCkge1xuICAgICAgICBpZiAoc3VjY2Vzcykge1xuICAgICAgICAgIGJlbmNobWFya1J1bi5jaGFydERhdGEucHVzaCh7eDogc3RlcCwgeTogdGltZX0pO1xuICAgICAgICB9XG4gICAgICAgIHJvd1ZhbHVlcy5wdXNoKHJlc3VsdFN0cmluZyk7XG4gICAgICB9XG4gICAgICBjb25zb2xlLmxvZyhiZW5jaG1hcmtSdW4ubmFtZSArICdbJyArIHN0ZXAgKyAnXTogJyArIGxvZ1N0cmluZyk7XG4gICAgfVxuICAgIHJ1bk51bWJlcnNUYWJsZS5hcHBlbmRDaGlsZCh0aGlzLmJ1aWxkUnVuTnVtYmVyc1Jvdyhyb3dWYWx1ZXMpKTtcblxuICAgIHN0ZXAgKz0gYmVuY2htYXJrUnVuR3JvdXAuc3RlcFNpemU7XG4gICAgLy8gQWxsb3cgdGhlIFVJIHRvIHVwZGF0ZS5cbiAgICBzZXRUaW1lb3V0KFxuICAgICAgICAoKSA9PiB0aGlzLnJ1bkJlbmNobWFya1N0ZXBzKFxuICAgICAgICAgICAgY2hhcnQsIGJlbmNobWFya1J1bkdyb3VwLCBiZW5jaG1hcmtSdW5Hcm91cEluZGV4LCBzdGVwKSxcbiAgICAgICAgMTAwKTtcbiAgfVxufVxuZG9jdW1lbnQucmVnaXN0ZXJFbGVtZW50KE1hdGhCZW5jaG1hcmsucHJvdG90eXBlLmlzLCBNYXRoQmVuY2htYXJrKTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uLy4uL3NyYy9tYXRoL2NvbnZfdXRpbCc7XG5pbXBvcnQge0FycmF5M0QsIGluaXRpYWxpemVHUFUsIE5EQXJyYXl9IGZyb20gJy4uLy4uL3NyYy9tYXRoL25kYXJyYXknO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgZ3BncHVfbWF0aCBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9ncGdwdV9tYXRoJztcbmltcG9ydCB7UG9vbDJEUHJvZ3JhbX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvcG9vbF9ncHUnO1xuaW1wb3J0IHtUZXh0dXJlTWFuYWdlcn0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvdGV4dHVyZV9tYW5hZ2VyJztcblxuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QX1JVTlMgPSA0MDtcblxuZXhwb3J0IGNvbnN0IE1BWF9QT09MX0JFTkNITUFSS19URVNUOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBjb25zdCBwb3NpdGlvbnMgPSBmYWxzZTtcbiAgcmV0dXJuIHRlc3RNYXhQb29sKHNpemUsIHBvc2l0aW9ucyk7XG59O1xuXG5leHBvcnQgY29uc3QgTUFYX1BPT0xfUE9TTlNfQkVOQ0hNQVJLX1RFU1Q6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IHBvc2l0aW9ucyA9IHRydWU7XG4gIHJldHVybiB0ZXN0TWF4UG9vbChzaXplLCBwb3NpdGlvbnMpO1xufTtcblxuZnVuY3Rpb24gdGVzdE1heFBvb2woc2l6ZTogbnVtYmVyLCBwb3NpdGlvbnM6IGJvb2xlYW4pOiBudW1iZXIge1xuICBjb25zdCBncGdwdSA9IG5ldyBHUEdQVUNvbnRleHQoKTtcbiAgY29uc3QgdGV4TWFuYWdlciA9IG5ldyBUZXh0dXJlTWFuYWdlcihncGdwdSk7XG4gIGluaXRpYWxpemVHUFUoZ3BncHUsIHRleE1hbmFnZXIpO1xuXG4gIGNvbnN0IG91dHB1dERlcHRoID0gMTtcbiAgY29uc3QgeFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbc2l6ZSwgc2l6ZSwgb3V0cHV0RGVwdGhdO1xuICBjb25zdCBmaWVsZFNpemUgPSAxMTtcbiAgY29uc3Qgc3RyaWRlID0gMTtcbiAgY29uc3QgemVyb1BhZCA9IGNvbnZfdXRpbC5jb21wdXRlRGVmYXVsdFBhZCh4U2hhcGUsIGZpZWxkU2l6ZSwgc3RyaWRlKTtcblxuICBjb25zdCBwcm9ncmFtID1cbiAgICAgIG5ldyBQb29sMkRQcm9ncmFtKHhTaGFwZSwgZmllbGRTaXplLCBzdHJpZGUsIHplcm9QYWQsICdtYXgnLCBwb3NpdGlvbnMpO1xuICBjb25zdCByZXMgPSBOREFycmF5Lnplcm9zKHByb2dyYW0ub3V0cHV0U2hhcGUpO1xuICBjb25zdCB4ID0gQXJyYXkzRC5yYW5kVW5pZm9ybSh4U2hhcGUsIC0xLCAxKTtcbiAgY29uc3QgYmluYXJ5ID0gZ3BncHVfbWF0aC5jb21waWxlUHJvZ3JhbShncGdwdSwgcHJvZ3JhbSwgW3hdLCByZXMpO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgZ3BncHVfbWF0aC5ydW5Qcm9ncmFtKGJpbmFyeSwgW3hdLCByZXMpO1xuICB9XG4gIHJlcy5nZXRWYWx1ZXMoKTtcbiAgY29uc3QgYXZnVGltZSA9IChwZXJmb3JtYW5jZS5ub3coKSAtIHN0YXJ0KSAvIE9QX1JVTlM7XG5cbiAgeC5kaXNwb3NlKCk7XG4gIHJlcy5kaXNwb3NlKCk7XG4gIHRleE1hbmFnZXIuZGlzcG9zZSgpO1xuICBncGdwdS5kZWxldGVQcm9ncmFtKGJpbmFyeS53ZWJHTFByb2dyYW0pO1xuICBncGdwdS5kaXNwb3NlKCk7XG5cbiAgcmV0dXJuIGF2Z1RpbWU7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7TkRBcnJheU1hdGhDUFV9IGZyb20gJy4uLy4uL3NyYy9tYXRoL21hdGhfY3B1JztcbmltcG9ydCB7QXJyYXkyRCwgTkRBcnJheX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvbmRhcnJheSc7XG5cbmltcG9ydCB7QmVuY2htYXJrVGVzdH0gZnJvbSAnLi9iZW5jaG1hcmsnO1xuXG5jb25zdCBPUFNfUEVSX1NNQUxMX1JVTiA9IDE7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgaWYgKHNpemUgPiA1MTIpIHtcbiAgICByZXR1cm4gLTE7XG4gIH1cbiAgY29uc3QgbWF0aCA9IG5ldyBOREFycmF5TWF0aENQVSgpO1xuICBjb25zdCBhID0gTkRBcnJheS5yYW5kVW5pZm9ybTxBcnJheTJEPihbc2l6ZSwgc2l6ZV0sIC0xLCAxKTtcbiAgY29uc3QgYiA9IE5EQXJyYXkucmFuZFVuaWZvcm08QXJyYXkyRD4oW3NpemUsIHNpemVdLCAtMSwgMSk7XG4gIGNvbnN0IHJ1bnMgPSAoc2l6ZSA8IDE5MikgPyBPUFNfUEVSX1NNQUxMX1JVTiA6IDE7XG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcnVuczsgaSsrKSB7XG4gICAgbWF0aC5tYXRNdWwoYSwgYik7XG4gIH1cbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIHJldHVybiAoZW5kIC0gc3RhcnQpIC8gcnVucztcbn07XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7TWF0cml4T3JpZW50YXRpb259IGZyb20gJy4uLy4uL3NyYy9tYXRoL21hdGgnO1xuaW1wb3J0IHtBcnJheTJEfSBmcm9tICcuLi8uLi9zcmMvbWF0aC9uZGFycmF5JztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9ncGdwdV9jb250ZXh0JztcbmltcG9ydCB7TWF0TXVsUHJvZ3JhbX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvbXVsbWF0X2dwdSc7XG5pbXBvcnQgKiBhcyBncGdwdV9tYXRoIGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X21hdGgnO1xuaW1wb3J0ICogYXMgbXVsbWF0X3BhY2tlZF9ncHUgZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvbXVsbWF0X3BhY2tlZF9ncHUnO1xuaW1wb3J0ICogYXMgdGVzdF91dGlsIGZyb20gJy4uLy4uL3NyYy90ZXN0X3V0aWwnO1xuXG5pbXBvcnQge0JlbmNobWFya1Rlc3R9IGZyb20gJy4vYmVuY2htYXJrJztcblxuY29uc3QgT1BfUlVOUyA9IDQwO1xuXG5leHBvcnQgY29uc3QgQkVOQ0hNQVJLX1RFU1Q6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCBhVGV4dHVyZSA9IGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUoc2l6ZSwgc2l6ZSk7XG4gIGNvbnN0IGJUZXh0dXJlID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShzaXplLCBzaXplKTtcbiAgY29uc3QgcmVzdWx0VGV4dHVyZSA9IGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUoc2l6ZSwgc2l6ZSk7XG5cbiAgY29uc3QgYUFyciA9IG5ldyBBcnJheTJEKFxuICAgICAgW3NpemUsIHNpemVdLCB7dGV4dHVyZTogYVRleHR1cmUsIHRleHR1cmVTaGFwZVJDOiBbc2l6ZSwgc2l6ZV19KTtcbiAgY29uc3QgYkFyciA9IG5ldyBBcnJheTJEKFxuICAgICAgW3NpemUsIHNpemVdLCB7dGV4dHVyZTogYlRleHR1cmUsIHRleHR1cmVTaGFwZVJDOiBbc2l6ZSwgc2l6ZV19KTtcbiAgY29uc3QgcmVzQXJyID0gbmV3IEFycmF5MkQoXG4gICAgICBbc2l6ZSwgc2l6ZV0sIHt0ZXh0dXJlOiByZXN1bHRUZXh0dXJlLCB0ZXh0dXJlU2hhcGVSQzogW3NpemUsIHNpemVdfSk7XG4gIGNvbnN0IHByb2dyYW0gPSBuZXcgTWF0TXVsUHJvZ3JhbShhQXJyLnNoYXBlLCBiQXJyLnNoYXBlKTtcbiAgY29uc3QgYmluYXJ5ID1cbiAgICAgIGdwZ3B1X21hdGguY29tcGlsZVByb2dyYW0oZ3BncHUsIHByb2dyYW0sIFthQXJyLCBiQXJyXSwgcmVzQXJyKTtcbiAgY29uc3QgYSA9IHRlc3RfdXRpbC5yYW5kb21BcnJheUluUmFuZ2Uoc2l6ZSAqIHNpemUsIC0xLCAxKTtcbiAgY29uc3QgYiA9IHRlc3RfdXRpbC5yYW5kb21BcnJheUluUmFuZ2Uoc2l6ZSAqIHNpemUsIC0xLCAxKTtcbiAgZ3BncHUudXBsb2FkTWF0cml4VG9UZXh0dXJlKGFUZXh0dXJlLCBzaXplLCBzaXplLCBhKTtcbiAgZ3BncHUudXBsb2FkTWF0cml4VG9UZXh0dXJlKGJUZXh0dXJlLCBzaXplLCBzaXplLCBiKTtcblxuICBjb25zdCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IE9QX1JVTlM7IGkrKykge1xuICAgIGdwZ3B1X21hdGgucnVuUHJvZ3JhbShiaW5hcnksIFthQXJyLCBiQXJyXSwgcmVzQXJyKTtcbiAgfVxuICBncGdwdS5kb3dubG9hZE1hdHJpeEZyb21UZXh0dXJlKHJlc3VsdFRleHR1cmUsIHNpemUsIHNpemUpO1xuICBjb25zdCBhdmdUaW1lID0gKHBlcmZvcm1hbmNlLm5vdygpIC0gc3RhcnQpIC8gT1BfUlVOUztcblxuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGFUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShiVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUocmVzdWx0VGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZVByb2dyYW0oYmluYXJ5LndlYkdMUHJvZ3JhbSk7XG4gIGdwZ3B1LmRpc3Bvc2UoKTtcblxuICByZXR1cm4gYXZnVGltZTtcbn07XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVF9QQUNLRUQ6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCBwcm9ncmFtOiBXZWJHTFByb2dyYW0gPVxuICAgICAgZ3BncHUuY3JlYXRlUHJvZ3JhbShtdWxtYXRfcGFja2VkX2dwdS5nZXRGcmFnbWVudFNoYWRlclNvdXJjZShcbiAgICAgICAgICBzaXplLCBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSLCBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSk7XG5cbiAgY29uc3QgYVRleHR1cmUgPSBncGdwdS5jcmVhdGVQYWNrZWRNYXRyaXhUZXh0dXJlKHNpemUsIHNpemUpO1xuICBjb25zdCBiVGV4dHVyZSA9IGdwZ3B1LmNyZWF0ZVBhY2tlZE1hdHJpeFRleHR1cmUoc2l6ZSwgc2l6ZSk7XG4gIGNvbnN0IHJlc3VsdFRleHR1cmUgPSBncGdwdS5jcmVhdGVQYWNrZWRNYXRyaXhUZXh0dXJlKHNpemUsIHNpemUpO1xuXG4gIGNvbnN0IGEgPSB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKHNpemUgKiBzaXplLCAtMSwgMSk7XG4gIGNvbnN0IGIgPSB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKHNpemUgKiBzaXplLCAtMSwgMSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShhVGV4dHVyZSwgc2l6ZSwgc2l6ZSwgYSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShiVGV4dHVyZSwgc2l6ZSwgc2l6ZSwgYik7XG5cbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUF9SVU5TOyBpKyspIHtcbiAgICBtdWxtYXRfcGFja2VkX2dwdS5tdWx0aXBseU1hdHJpeFBhY2tlZChcbiAgICAgICAgZ3BncHUsIHByb2dyYW0sIGFUZXh0dXJlLCBiVGV4dHVyZSwgcmVzdWx0VGV4dHVyZSwgW3NpemUsIHNpemVdKTtcbiAgfVxuXG4gIGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVBhY2tlZFRleHR1cmUocmVzdWx0VGV4dHVyZSwgc2l6ZSwgc2l6ZSk7XG4gIGNvbnN0IGF2Z1RpbWUgPSAocGVyZm9ybWFuY2Uubm93KCkgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoYVRleHR1cmUpO1xuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGJUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShyZXN1bHRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblBvbHltZXIoe2lzOiAnZGVtby1mb290ZXInfSk7XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5Qb2x5bWVyKHtpczogJ2RlbW8taGVhZGVyJ30pO1xuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG4vKipcbiAqIEBmaWxlb3ZlcnZpZXdcbiAqXG4gKiBEZWZpbmVzIGFuIGludGVyZmFjZSBmb3IgY3JlYXRpbmcgUG9seW1lciBlbGVtZW50cyBpbiBUeXBlc2NyaXB0IHdpdGggdGhlXG4gKiBjb3JyZWN0IHR5cGluZ3MuIEEgUG9seW1lciBlbGVtZW50IHNob3VsZCBiZSBkZWZpbmVkIGxpa2UgdGhpczpcbiAqXG4gKiBgYGBcbiAqIGxldCBNeUVsZW1lbnRQb2x5bWVyID0gUG9seW1lckVsZW1lbnQoe1xuICogICBpczogJ215LXBvbHltZXItZWxlbWVudCcsXG4gKiAgIHByb3BlcnRpZXM6IHtcbiAqICAgICBmb286IHN0cmluZyxcbiAqICAgICBiYXI6IEFycmF5XG4gKiAgIH1cbiAqIH0pO1xuICpcbiAqIGNsYXNzIE15RWxlbWVudCBleHRlbmRzIE15RWxlbWVudFBvbHltZXIge1xuICogICBmb286IHN0cmluZztcbiAqICAgYmFyOiBudW1iZXJbXTtcbiAqXG4gKiAgIHJlYWR5KCkge1xuICogICAgIGNvbnNvbGUubG9nKCdNeUVsZW1lbnQgaW5pdGlhbGl6ZWQhJyk7XG4gKiAgIH1cbiAqIH1cbiAqXG4gKiBkb2N1bWVudC5yZWdpc3RlckVsZW1lbnQoTXlFbGVtZW50LnByb3RvdHlwZS5pcywgTXlFbGVtZW50KTtcbiAqIGBgYFxuICovXG5cbmV4cG9ydCB0eXBlIFNwZWMgPSB7XG4gIGlzOiBzdHJpbmc7IHByb3BlcnRpZXM6IHtcbiAgICBba2V5OiBzdHJpbmddOiAoRnVuY3Rpb258e1xuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgdHlwZTogRnVuY3Rpb24sIHZhbHVlPzogYW55O1xuICAgICAgcmVmbGVjdFRvQXR0cmlidXRlPzogYm9vbGVhbjtcbiAgICAgIHJlYWRvbmx5PzogYm9vbGVhbjtcbiAgICAgIG5vdGlmeT86IGJvb2xlYW47XG4gICAgICBjb21wdXRlZD86IHN0cmluZztcbiAgICAgIG9ic2VydmVyPzogc3RyaW5nO1xuICAgIH0pXG4gIH07XG4gIG9ic2VydmVycz86IHN0cmluZ1tdO1xufTtcblxuZXhwb3J0IGZ1bmN0aW9uIFBvbHltZXJFbGVtZW50KHNwZWM6IFNwZWMpIHtcbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICByZXR1cm4gUG9seW1lci5DbGFzcyhzcGVjIGFzIGFueSkgYXMge25ldyAoKTogUG9seW1lckhUTUxFbGVtZW50fTtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBQb2x5bWVySFRNTEVsZW1lbnQgZXh0ZW5kcyBIVE1MRWxlbWVudCwgcG9seW1lci5CYXNlIHt9XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBhc3NlcnRDb25jYXQzRFNoYXBlc01hdGNoKFxuICAgIHgxU2hhcGU6IG51bWJlcltdLCB4MlNoYXBlOiBudW1iZXJbXSwgYXhpczogbnVtYmVyLFxuICAgIGVycm9yTWVzc2FnZVByZWZpeCA9ICcnKSB7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgeDFTaGFwZS5sZW5ndGggPT09IDMsXG4gICAgICBlcnJvck1lc3NhZ2VQcmVmaXggKyAnQ29uY2F0M0QgeDEgc2hhcGUgc2hvdWxkIGJlIG9mIHJhbmsgMy4nKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4MlNoYXBlLmxlbmd0aCA9PT0gMyxcbiAgICAgIGVycm9yTWVzc2FnZVByZWZpeCArICdDb25jYXQzRCB4MiBzaGFwZSBzaG91bGQgYmUgb2YgcmFuayAzLicpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgYXhpcyA+PSAwICYmIGF4aXMgPCAzLCAnQXhpcyBmb3IgY29uY2F0M0QgbXVzdCBiZSBiZXR3ZWVuIDAgYW5kIDIuJyk7XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCAzOyBpKyspIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgKGkgPT09IGF4aXMpIHx8ICh4MVNoYXBlW2ldID09PSB4MlNoYXBlW2ldKSxcbiAgICAgICAgZXJyb3JNZXNzYWdlUHJlZml4ICtcbiAgICAgICAgICAgIGBTaGFwZSAoJHt4MVNoYXBlfSkgZG9lcyBub3QgbWF0Y2ggKCR7eDJTaGFwZX0pIGFsb25nIGAgK1xuICAgICAgICAgICAgYG5vbi1jb25jYXRlbmF0ZWQgYXhpcy5gKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZUNvbmNhdDNET3V0cHV0U2hhcGUoXG4gICAgeDFTaGFwZTogbnVtYmVyW10sIHgyU2hhcGU6IG51bWJlcltdLFxuICAgIGF4aXM6IG51bWJlcik6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSB7XG4gIHV0aWwuYXNzZXJ0KHgxU2hhcGUubGVuZ3RoID09PSAzLCAnQ29uY2F0M0QgeDEgc2hhcGUgc2hvdWxkIGJlIG9mIHJhbmsgMy4nKTtcbiAgdXRpbC5hc3NlcnQoeDJTaGFwZS5sZW5ndGggPT09IDMsICdDb25jYXQzRCB4MnNoYXBlIHNob3VsZCBiZSBvZiByYW5rIDMuJyk7XG5cbiAgY29uc3Qgb3V0cHV0U2hhcGUgPSB4MVNoYXBlLnNsaWNlKCk7XG4gIG91dHB1dFNoYXBlW2F4aXNdICs9IHgyU2hhcGVbYXhpc107XG4gIHJldHVybiBvdXRwdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG59IiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5leHBvcnQgdHlwZSBPdXRwdXRJbmZvID0ge1xuICBzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICBwYWRkaW5nSW5mbzoge3RvcDogbnVtYmVyLCBsZWZ0OiBudW1iZXIsIHJpZ2h0OiBudW1iZXIsIGJvdHRvbTogbnVtYmVyfTtcbn07XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlT3V0cHV0SW5mbyhcbiAgICBpblNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZpbHRlckhlaWdodDogbnVtYmVyLFxuICAgIGZpbHRlcldpZHRoOiBudW1iZXIsIG91dERlcHRoOiBudW1iZXIsIHN0cmlkZUhlaWdodDogbnVtYmVyLFxuICAgIHN0cmlkZVdpZHRoOiBudW1iZXIsIHBhZGRpbmc6ICdzYW1lJ3wndmFsaWQnfG51bWJlcik6IE91dHB1dEluZm8ge1xuICBpZiAodHlwZW9mIHBhZGRpbmcgPT09ICdudW1iZXInKSB7XG4gICAgY29uc3Qgb3V0U2hhcGUgPSBjb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgICAgaW5TaGFwZSwgZmlsdGVySGVpZ2h0LCBvdXREZXB0aCwgc3RyaWRlSGVpZ2h0LCBwYWRkaW5nKTtcbiAgICByZXR1cm4ge1xuICAgICAgc2hhcGU6IG91dFNoYXBlLFxuICAgICAgcGFkZGluZ0luZm86XG4gICAgICAgICAge3RvcDogcGFkZGluZywgYm90dG9tOiBwYWRkaW5nLCBsZWZ0OiBwYWRkaW5nLCByaWdodDogcGFkZGluZ31cbiAgICB9O1xuICB9XG4gIGNvbnN0IGluSGVpZ2h0ID0gaW5TaGFwZVswXTtcbiAgY29uc3QgaW5XaWR0aCA9IGluU2hhcGVbMV07XG4gIGxldCBvdXRTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICBsZXQgcGFkZGluZ0luZm86IHtsZWZ0OiBudW1iZXIsIHRvcDogbnVtYmVyLCBib3R0b206IG51bWJlciwgcmlnaHQ6IG51bWJlcn07XG4gIGlmIChwYWRkaW5nID09PSAnc2FtZScpIHtcbiAgICBjb25zdCBvdXRIZWlnaHQgPSBNYXRoLmNlaWwoaW5IZWlnaHQgLyBzdHJpZGVIZWlnaHQpO1xuICAgIGNvbnN0IG91dFdpZHRoID0gTWF0aC5jZWlsKGluV2lkdGggLyBzdHJpZGVXaWR0aCk7XG4gICAgb3V0U2hhcGUgPSBbb3V0SGVpZ2h0LCBvdXRXaWR0aCwgb3V0RGVwdGhdO1xuICAgIGNvbnN0IHBhZEFsb25nSGVpZ2h0ID1cbiAgICAgICAgKG91dEhlaWdodCAtIDEpICogc3RyaWRlSGVpZ2h0ICsgZmlsdGVySGVpZ2h0IC0gaW5IZWlnaHQ7XG4gICAgY29uc3QgcGFkQWxvbmdXaWR0aCA9IChvdXRXaWR0aCAtIDEpICogc3RyaWRlV2lkdGggKyBmaWx0ZXJXaWR0aCAtIGluV2lkdGg7XG4gICAgY29uc3QgdG9wID0gTWF0aC5mbG9vcihwYWRBbG9uZ0hlaWdodCAvIDIpO1xuICAgIGNvbnN0IGJvdHRvbSA9IHBhZEFsb25nSGVpZ2h0IC0gdG9wO1xuICAgIGNvbnN0IGxlZnQgPSBNYXRoLmZsb29yKHBhZEFsb25nV2lkdGggLyAyKTtcbiAgICBjb25zdCByaWdodCA9IHBhZEFsb25nV2lkdGggLSBsZWZ0O1xuICAgIHBhZGRpbmdJbmZvID0ge3RvcCwgYm90dG9tLCBsZWZ0LCByaWdodH07XG4gIH0gZWxzZSBpZiAocGFkZGluZyA9PT0gJ3ZhbGlkJykge1xuICAgIGNvbnN0IG91dEhlaWdodCA9IE1hdGguY2VpbCgoaW5IZWlnaHQgLSBmaWx0ZXJIZWlnaHQgKyAxKSAvIHN0cmlkZUhlaWdodCk7XG4gICAgY29uc3Qgb3V0V2lkdGggPSBNYXRoLmNlaWwoKGluV2lkdGggLSBmaWx0ZXJXaWR0aCArIDEpIC8gc3RyaWRlV2lkdGgpO1xuICAgIG91dFNoYXBlID0gW291dEhlaWdodCwgb3V0V2lkdGgsIG91dERlcHRoXTtcbiAgICBwYWRkaW5nSW5mbyA9IHt0b3A6IDAsIGJvdHRvbTogMCwgbGVmdDogMCwgcmlnaHQ6IDB9O1xuICB9IGVsc2Uge1xuICAgIHRocm93IEVycm9yKGBVbmtub3duIHBhZGRpbmcgcGFyYW1ldGVyOiAke3BhZGRpbmd9YCk7XG4gIH1cbiAgcmV0dXJuIHtzaGFwZTogb3V0U2hhcGUsIHBhZGRpbmdJbmZvfTtcbn1cblxuLyoqXG4gKiBAZGVwcmVjYXRlZCBVc2UgYGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0SW5mb2AgaW5zdGVhZC5cbiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNvbXB1dGVPdXRwdXRTaGFwZTNEKFxuICAgIGluU2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZmllbGRTaXplOiBudW1iZXIsIG91dERlcHRoOiBudW1iZXIsXG4gICAgc3RyaWRlOiBudW1iZXIsIHplcm9QYWQ/OiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0ge1xuICBpZiAoemVyb1BhZCA9PSBudWxsKSB7XG4gICAgemVyb1BhZCA9IGNvbXB1dGVEZWZhdWx0UGFkKGluU2hhcGUsIGZpZWxkU2l6ZSwgc3RyaWRlKTtcbiAgfVxuICBjb25zdCBpbnB1dFJvd3MgPSBpblNoYXBlWzBdO1xuICBjb25zdCBpbnB1dENvbHMgPSBpblNoYXBlWzFdO1xuICBjb25zdCBvdXRwdXRSb3dzID0gKGlucHV0Um93cyAtIGZpZWxkU2l6ZSArIDIgKiB6ZXJvUGFkKSAvIHN0cmlkZSArIDE7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgdXRpbC5pc0ludChvdXRwdXRSb3dzKSxcbiAgICAgIGBUaGUgb3V0cHV0ICMgb2Ygcm93cyAoJHtvdXRwdXRSb3dzfSkgbXVzdCBiZSBhbiBpbnRlZ2VyLiBDaGFuZ2UgdGhlIGAgK1xuICAgICAgICAgIGBzdHJpZGUgYW5kL29yIHplcm8gcGFkIHBhcmFtZXRlcnNgKTtcblxuICBjb25zdCBvdXRwdXRDb2xzID0gKGlucHV0Q29scyAtIGZpZWxkU2l6ZSArIDIgKiB6ZXJvUGFkKSAvIHN0cmlkZSArIDE7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgdXRpbC5pc0ludChvdXRwdXRDb2xzKSxcbiAgICAgIGBUaGUgb3V0cHV0ICMgb2YgY29sdW1ucyAoJHtvdXRwdXRDb2xzfSkgbXVzdCBiZSBhbiBpbnRlZ2VyLiBDaGFuZ2UgYCArXG4gICAgICAgICAgYHRoZSBzdHJpZGUgYW5kL29yIHplcm8gcGFkIHBhcmFtZXRlcnNgKTtcblxuICByZXR1cm4gW291dHB1dFJvd3MsIG91dHB1dENvbHMsIG91dERlcHRoXTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbXB1dGVEZWZhdWx0UGFkKFxuICAgIGlucHV0U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZmllbGRTaXplOiBudW1iZXIsXG4gICAgc3RyaWRlOiBudW1iZXIpOiBudW1iZXIge1xuICByZXR1cm4gTWF0aC5mbG9vcigoaW5wdXRTaGFwZVswXSAqIChzdHJpZGUgLSAxKSAtIHN0cmlkZSArIGZpZWxkU2l6ZSkgLyAyKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbXB1dGVUZXhTaGFwZUZyb20zRChcbiAgICBzaGFwZVJvd0NvbERlcHRoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgcmV0dXJuIFtzaGFwZVJvd0NvbERlcHRoWzBdLCBzaGFwZVJvd0NvbERlcHRoWzFdICogc2hhcGVSb3dDb2xEZXB0aFsyXV07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlV2VpZ2h0c1NoYXBlNEQoXG4gICAgaW5wdXREZXB0aDogbnVtYmVyLCBvdXRwdXREZXB0aDogbnVtYmVyLFxuICAgIGZTaXplOiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSB7XG4gIHJldHVybiBbZlNpemUsIGZTaXplLCBpbnB1dERlcHRoLCBvdXRwdXREZXB0aF07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlRGlsYXRlZFJDKFxuICAgIHJjOiBbbnVtYmVyLCBudW1iZXJdLCBvcmlnU3RyaWRlOiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgY29uc3Qgcm93c0RpbGF0ZWQgPSAocmNbMF0gLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuICBjb25zdCBjb2xzRGlsYXRlZCA9IChyY1sxXSAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG4gIHJldHVybiBbcm93c0RpbGF0ZWQsIGNvbHNEaWxhdGVkXTtcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuZXhwb3J0IGZ1bmN0aW9uIHZhbGlkYXRlU2hhcGVzKFxuICAgIHNvdXJjZVNpemU6IFtudW1iZXIsIG51bWJlcl0sIGRlc3RTaXplOiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gIGNvbnN0IHNyY0FyZWEgPSBzb3VyY2VTaXplWzBdICogc291cmNlU2l6ZVsxXTtcbiAgY29uc3QgZHN0QXJlYSA9IGRlc3RTaXplWzBdICogZGVzdFNpemVbMV07XG4gIGlmIChzcmNBcmVhICE9PSBkc3RBcmVhKSB7XG4gICAgY29uc3Qgc3JjU3RyID0gJ1snICsgc291cmNlU2l6ZVswXSArICcsICcgKyBzb3VyY2VTaXplWzFdICsgJ10nO1xuICAgIGNvbnN0IGRzdFN0ciA9ICdbJyArIGRlc3RTaXplWzBdICsgJywgJyArIGRlc3RTaXplWzFdICsgJ10nO1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ2NvcHkyRCBzaGFwZXMgaGF2ZSBkaWZmZXJlbnQgYXJlYXM6XFxuICBzb3VyY2VTaXplICcgKyBzcmNTdHIgK1xuICAgICAgICAnLCBhcmVhICcgKyBzcmNBcmVhICsgJ1xcbiAgZGVzdFNpemUgJyArIGRzdFN0ciArICcsIGFyZWEgJyArIGRzdEFyZWEpO1xuICB9XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5pbXBvcnQgKiBhcyBjb25jYXQzZF91dGlsIGZyb20gJy4vY29uY2F0M2RfdXRpbCc7XG5pbXBvcnQgKiBhcyBjb252X3V0aWwgZnJvbSAnLi9jb252X3V0aWwnO1xuaW1wb3J0IHtPdXRwdXRJbmZvfSBmcm9tICcuL2NvbnZfdXRpbCc7XG5pbXBvcnQgKiBhcyBjb3B5MmRfdXRpbCBmcm9tICcuL2NvcHkyZF91dGlsJztcbmltcG9ydCB7QXJyYXkxRCwgQXJyYXkyRCwgQXJyYXkzRCwgQXJyYXk0RCwgTkRBcnJheSwgU2NhbGFyfSBmcm9tICcuL25kYXJyYXknO1xuXG5leHBvcnQgdHlwZSBTY29wZVJlc3VsdCA9IE5EQXJyYXlbXXxOREFycmF5fHZvaWQ7XG5cbmV4cG9ydCBpbnRlcmZhY2UgTFNUTUNlbGwge1xuICAoZGF0YTogQXJyYXkyRCwgYzogQXJyYXkyRCwgaDogQXJyYXkyRCk6IFtBcnJheTJELCBBcnJheTJEXTtcbn1cblxuXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgTkRBcnJheU1hdGgge1xuICBwcml2YXRlIG5kYXJyYXlTY29wZXM6IE5EQXJyYXlbXVtdID0gW107XG4gIHByaXZhdGUgYWN0aXZlU2NvcGU6IE5EQXJyYXlbXTtcblxuICBwcml2YXRlIG5kYXJyYXlzVG9LZWVwOiBOREFycmF5W11bXSA9IFtdO1xuICBwcml2YXRlIGFjdGl2ZVNjb3BlTkRBcnJheXNUb0tlZXA6IE5EQXJyYXlbXSA9IFtdO1xuXG4gIHByaXZhdGUgZGVidWdNb2RlID0gZmFsc2U7XG5cbiAgLyoqXG4gICAqIEBwYXJhbSBzYWZlTW9kZSBJbiBzYWZlIG1vZGUsIHlvdSBtdXN0IHVzZSBtYXRoIG9wZXJhdGlvbnMgaW5zaWRlXG4gICAqICAgICBhIG1hdGguc2NvcGUoKSB3aGljaCB3aWxsIGF1dG9tYXRpY2FsbHkgY2xlYW4gdXAgaW50ZXJtZWRpYXRlIE5EQXJyYXlzLlxuICAgKi9cbiAgY29uc3RydWN0b3IocHJpdmF0ZSBzYWZlTW9kZTogYm9vbGVhbikge31cblxuICAvKipcbiAgICogQ3JlYXRlIGEgbmV3IG1hdGggc2NvcGUuIFB1dCBjaGFpbmVkIG1hdGggb3BlcmF0aW9ucyBpbnNpZGUgYSBzY29wZVxuICAgKiBmdW5jdGlvbiBjbG9zdXJlIHNvIHRoYXQgdGhlIGxpYnJhcnkgYXV0b21hdGljYWxseSBjbGVhbnMgdXAgTkRBcnJheXNcbiAgICogZnJvbSBpbnRlcm1lZGlhdGUgbWF0aCBvcGVyYXRpb25zLiBZb3UgbXVzdCBjcmVhdGUgYSBzY29wZSBpbiBzYWZlIG1vZGVcbiAgICogdG8gY2FsbCBtYXRoIG9wZXJhdGlvbnMuIElmIGEgcmVzdWx0IGlzIHJldHVybmVkIGZyb20gdGhlIHNjb3BlLCBpdCB3aWxsXG4gICAqIGFsc28gYmUgdHJhY2tlZCwgd2hpY2ggbWVhbnMgdGhlcmUgbXVzdCBiZSB5ZXQgYW5vdGhlciB3cmFwcGluZyBzY29wZS5cbiAgICogQHBhcmFtIHNjb3BlRm4gVGhlIGZ1bmN0aW9uIHRvIGV4ZWN1dGUgd2l0aCBjaGFpbmVkIG1hdGggb3BlcmF0aW9ucy5cbiAgICovXG4gIHNjb3BlPFQgZXh0ZW5kcyBTY29wZVJlc3VsdD4oXG4gICAgICBzY29wZUZuOlxuICAgICAgICAgIChrZWVwOiA8VDEgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUMSkgPT4gVDEsXG4gICAgICAgICAgIHRyYWNrOiA8VDIgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUMikgPT4gVDIpID0+IFQpIHtcbiAgICB0aGlzLnN0YXJ0U2NvcGUoKTtcblxuICAgIGNvbnN0IGtlZXBGbiA9IDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQgPT4gdGhpcy5rZWVwKG5kYXJyYXkpO1xuICAgIGNvbnN0IHRyYWNrRm4gPSA8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUID0+IHRoaXMudHJhY2sobmRhcnJheSk7XG4gICAgY29uc3QgcmVzdWx0ID0gc2NvcGVGbihrZWVwRm4sIHRyYWNrRm4pO1xuXG4gICAgdGhpcy5lbmRTY29wZShyZXN1bHQpO1xuXG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG5cbiAgLyoqXG4gICAqIEluIGRlYnVnIG1vZGUsIHRoZSBvdXRwdXQgb2YgZXZlcnkgbWF0aCBjYWxsIHdpbGwgYmUgZG93bmxvYWRlZCB0byB0aGUgQ1BVXG4gICAqIGFuZCBjaGVja2VkIGZvciBOYU5zLiBUaGlzIHNpZ25pZmljYW50bHkgaW1wYWN0cyBwZXJmb3JtYW5jZS5cbiAgICovXG4gIGVuYWJsZURlYnVnTW9kZSgpIHtcbiAgICB0aGlzLmRlYnVnTW9kZSA9IHRydWU7XG4gICAgY29uc29sZS53YXJuKFxuICAgICAgICAnRGVidWdnaW5nIG1vZGUgaXMgT04uIFRoZSBvdXRwdXQgb2YgZXZlcnkgbWF0aCBjYWxsIHdpbGwgJyArXG4gICAgICAgICdiZSBkb3dubG9hZGVkIHRvIENQVSBhbmQgY2hlY2tlZCBmb3IgTmFOcy4gJyArXG4gICAgICAgICdUaGlzIHNpZ25pZmljYW50bHkgaW1wYWN0cyBwZXJmb3JtYW5jZS4nKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTdGFydCBhIHNjb3BlLiBVc2UgdGhpcyB3aXRoIGVuZFNjb3BlKCkgdG8gYWNoaWV2ZSB0aGUgc2FtZSBmdW5jdGlvbmFsaXR5XG4gICAqIGFzIHNjb3BlKCkgd2l0aG91dCB0aGUgbmVlZCBmb3IgYSBmdW5jdGlvbiBjbG9zdXJlLlxuICAgKi9cbiAgc3RhcnRTY29wZSgpIHtcbiAgICBjb25zdCBuZXdTY29wZTogTkRBcnJheVtdID0gW107XG4gICAgdGhpcy5uZGFycmF5U2NvcGVzLnB1c2gobmV3U2NvcGUpO1xuICAgIHRoaXMuYWN0aXZlU2NvcGUgPSBuZXdTY29wZTtcblxuICAgIGNvbnN0IG5ld05EQXJyYXlzVG9LZWVwOiBOREFycmF5W10gPSBbXTtcbiAgICB0aGlzLm5kYXJyYXlzVG9LZWVwLnB1c2gobmV3TkRBcnJheXNUb0tlZXApO1xuICAgIHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcCA9IG5ld05EQXJyYXlzVG9LZWVwO1xuICB9XG5cbiAgLyoqXG4gICAqIEVuZCBhIHNjb3BlLiBVc2UgdGhpcyB3aXRoIHN0YXJ0U2NvcGUoKSB0byBhY2hpZXZlIHRoZSBzYW1lIGZ1bmN0aW9uYWxpdHlcbiAgICogYXMgc2NvcGUoKSB3aXRob3V0IHRoZSBuZWVkIGZvciBhIGZ1bmN0aW9uIGNsb3N1cmUuXG4gICAqL1xuICBlbmRTY29wZShyZXN1bHQ6IFNjb3BlUmVzdWx0KSB7XG4gICAgbGV0IGFycmF5c1RvS2VlcCA9IHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcDtcbiAgICBpZiAocmVzdWx0ICE9IG51bGwpIHtcbiAgICAgIGFycmF5c1RvS2VlcCA9IGFycmF5c1RvS2VlcC5jb25jYXQocmVzdWx0IGFzIE5EQXJyYXkgfCBOREFycmF5W10pO1xuICAgIH1cbiAgICAvLyBEaXNwb3NlIHRoZSBjdXJyZW50IHNjb3BlLlxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5hY3RpdmVTY29wZS5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgbmRhcnJheSA9IHRoaXMuYWN0aXZlU2NvcGVbaV07XG4gICAgICBpZiAodGhpcy5pc05EQXJyYXlEYXRhSW5MaXN0KG5kYXJyYXksIGFycmF5c1RvS2VlcCkpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBuZGFycmF5LmRpc3Bvc2UoKTtcbiAgICB9XG5cbiAgICAvLyBQb3AgdGhlIGN1cnJlbnQgc2NvcGUuXG4gICAgdGhpcy5uZGFycmF5U2NvcGVzLnBvcCgpO1xuICAgIHRoaXMuYWN0aXZlU2NvcGUgPSB0aGlzLm5kYXJyYXlTY29wZXMubGVuZ3RoID09PSAwID9cbiAgICAgICAgbnVsbCA6XG4gICAgICAgIHRoaXMubmRhcnJheVNjb3Blc1t0aGlzLm5kYXJyYXlTY29wZXMubGVuZ3RoIC0gMV07XG5cbiAgICAvLyBUcmFjayB0aGUgY3VycmVudCByZXN1bHQgaW4gdGhlIHBhcmVudCBzY29wZS5cbiAgICBpZiAocmVzdWx0IGluc3RhbmNlb2YgTkRBcnJheSAmJlxuICAgICAgICAhdGhpcy5pc05EQXJyYXlEYXRhSW5MaXN0KHJlc3VsdCwgdGhpcy5hY3RpdmVTY29wZU5EQXJyYXlzVG9LZWVwKSkge1xuICAgICAgdGhpcy50cmFjayhyZXN1bHQpO1xuICAgIH0gZWxzZSBpZiAoQXJyYXkuaXNBcnJheShyZXN1bHQpKSB7XG4gICAgICByZXN1bHQuZm9yRWFjaChyID0+IHtcbiAgICAgICAgaWYgKHIgaW5zdGFuY2VvZiBOREFycmF5ICYmXG4gICAgICAgICAgICAhdGhpcy5pc05EQXJyYXlEYXRhSW5MaXN0KHIsIHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcCkpIHtcbiAgICAgICAgICB0aGlzLnRyYWNrKHIpO1xuICAgICAgICB9XG4gICAgICB9KTtcbiAgICB9XG5cbiAgICB0aGlzLm5kYXJyYXlzVG9LZWVwLnBvcCgpO1xuICAgIHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcCA9IHRoaXMubmRhcnJheXNUb0tlZXAubGVuZ3RoID09PSAwID9cbiAgICAgICAgbnVsbCA6XG4gICAgICAgIHRoaXMubmRhcnJheXNUb0tlZXBbdGhpcy5uZGFycmF5c1RvS2VlcC5sZW5ndGggLSAxXTtcbiAgfVxuXG4gIHByaXZhdGUgaXNOREFycmF5RGF0YUluTGlzdChuZGFycmF5OiBOREFycmF5LCBuZGFycmF5TGlzdDogTkRBcnJheVtdKSB7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBuZGFycmF5TGlzdC5sZW5ndGg7IGkrKykge1xuICAgICAgaWYgKG5kYXJyYXlMaXN0W2ldLmdldERhdGEoKSA9PT0gbmRhcnJheS5nZXREYXRhKCkpIHtcbiAgICAgICAgcmV0dXJuIHRydWU7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuXG4gIC8qKlxuICAgKiBLZWVwcyBhbiBOREFycmF5IGluIHRoZSBjdXJyZW50IHNjb3BlIGZyb20gYmVpbmcgZGlzcG9zZWQgYXV0b21hdGljYWxseS5cbiAgICogQHBhcmFtIHJlc3VsdCBUaGUgTkRBcnJheSB0byBrZWVwIGZyb20gYmVpbmcgZGlzcG9zZWQuXG4gICAqL1xuICBrZWVwPFQgZXh0ZW5kcyBOREFycmF5PihyZXN1bHQ6IFQpOiBUIHtcbiAgICBpZiAodGhpcy5hY3RpdmVTY29wZSA9PSBudWxsKSB7XG4gICAgICBpZiAodGhpcy5zYWZlTW9kZSkge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAnWW91IGFyZSB1c2luZyBtYXRoIGluIHNhZmUgbW9kZS4gRW5jbG9zZSBhbGwgJyArXG4gICAgICAgICAgICAnbWF0aC5tZXRob2QoKSBjYWxscyBpbnNpZGUgYSBzY29wZTogJyArXG4gICAgICAgICAgICAnbWF0aC5zY29wZSgoKSA9PiB7bWF0aC5tZXRob2QoKTsuLi59KSB0byBhdm9pZCBtZW1vcnkgJyArXG4gICAgICAgICAgICAnbGVha3MuJyk7XG4gICAgICB9XG4gICAgICByZXR1cm4gcmVzdWx0O1xuICAgIH1cbiAgICB0aGlzLmFjdGl2ZVNjb3BlTkRBcnJheXNUb0tlZXAucHVzaChyZXN1bHQpO1xuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICBwcml2YXRlIGNoZWNrRm9yTmFOKGFycjogTkRBcnJheSk6IHZvaWQge1xuICAgIGNvbnN0IHZhbHMgPSBhcnIuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWxzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBpZiAoaXNOYU4odmFsc1tpXSkpIHtcbiAgICAgICAgdGhyb3cgRXJyb3IoJ1RoZSByZXN1bHQgTkRBcnJheSBvZiB0aGUgbGFzdCBtYXRoIGNhbGwgaGFzIE5hTnMuJyk7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgLyoqXG4gICAqIFRyYWNrcyBhbiBOREFycmF5IGluIHRoZSBjdXJyZW50IHNjb3BlIHRvIGJlIGF1dG9tYXRpY2FsbHkgY2xlYW5lZCB1cCB3aGVuXG4gICAqIHRoZSBjdXJyZW50IHNjb3BlIGVuZHMsIGFuZCByZXR1cm5zIHRoZSB2YWx1ZS5cbiAgICogQHBhcmFtIHJlc3VsdCBUaGUgTkRBcnJheSB0byB0cmFjayBpbiB0aGUgY3VycmVudCBzY29wZS5cbiAgICovXG4gIHRyYWNrPFQgZXh0ZW5kcyBOREFycmF5PihyZXN1bHQ6IFQpOiBUIHtcbiAgICBpZiAodGhpcy5kZWJ1Z01vZGUpIHtcbiAgICAgIHRoaXMuY2hlY2tGb3JOYU4ocmVzdWx0KTtcbiAgICB9XG4gICAgaWYgKHRoaXMuYWN0aXZlU2NvcGUgPT0gbnVsbCkge1xuICAgICAgaWYgKHRoaXMuc2FmZU1vZGUpIHtcbiAgICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgICAgJ1lvdSBhcmUgdXNpbmcgbWF0aCBpbiBzYWZlIG1vZGUuIEVuY2xvc2UgYWxsICcgK1xuICAgICAgICAgICAgJ21hdGgubWV0aG9kKCkgY2FsbHMgaW5zaWRlIGEgc2NvcGU6ICcgK1xuICAgICAgICAgICAgJ21hdGguc2NvcGUoKCkgPT4ge21hdGgubWV0aG9kKCk7Li4ufSkgdG8gYXZvaWQgbWVtb3J5ICcgK1xuICAgICAgICAgICAgJ2xlYWtzLicpO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICB9XG4gICAgdGhpcy5hY3RpdmVTY29wZS5wdXNoKHJlc3VsdCk7XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgZG90IHByb2R1Y3Qgb2YgdHdvIG1hdHJpY2VzLCBBICogQi4gVGhlc2UgbXVzdCBiZSBtYXRyaWNlcyxcbiAgICogdXNlIG1hdHJpeFRpbWVzVmVjdG9yIGFuZCB2ZWN0b3JUaW1lc01hdHJpeCwgZG90UHJvZHVjdCwgYW5kIG91dGVyUHJvZHVjdFxuICAgKiBpbiBvdGhlciBjYXNlcy5cbiAgICogQHBhcmFtIGEgRmlyc3QgbWF0cml4IGluIGRvdCBwcm9kdWN0IG9wZXJhdGlvbi5cbiAgICogQHBhcmFtIGIgU2Vjb25kIG1hdHJpeCBpbiBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSBhT3JpZW50YXRpb24gVGhlIE1hdHJpeE9yaWVudGF0aW9uIG9mIEEuIElmIHVzaW5nIFRSQU5TUE9TRUQsIHdpbGxcbiAgICogY29tcHV0ZSBBXlQgKiBCLlxuICAgKiBAcGFyYW0gYk9yaWVudGF0aW9uIFRoZSBNYXRyaXhPcmllbnRhdGlvbiBvZiBCLiBJZiB1c2luZyBUUkFOU1BPU0VELCB3aWxsXG4gICAqIGNvbXB1dGUgQSAqIEJeVC5cbiAgICovXG4gIG1hdE11bChcbiAgICAgIGE6IEFycmF5MkQsIGI6IEFycmF5MkQsIGFPcmllbnRhdGlvbiA9IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIsXG4gICAgICBiT3JpZW50YXRpb24gPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKTogQXJyYXkyRCB7XG4gICAgY29uc3QgaW5uZXJTaGFwZUEgPVxuICAgICAgICAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IGEuc2hhcGVbMV0gOiBhLnNoYXBlWzBdO1xuICAgIGNvbnN0IGlubmVyU2hhcGVCID1cbiAgICAgICAgKGJPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyBiLnNoYXBlWzBdIDogYi5zaGFwZVsxXTtcblxuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBhLnJhbmsgPT09IDIgJiYgYi5yYW5rID09PSAyLFxuICAgICAgICBgRXJyb3IgaW4gbWF0TXVsOiBpbnB1dHMgbXVzdCBiZSByYW5rIDIsIGdvdCByYW5rcyAke2EucmFua31gICtcbiAgICAgICAgICAgIGBhbmQgJHtiLnJhbmt9LmApO1xuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGlubmVyU2hhcGVBID09PSBpbm5lclNoYXBlQixcbiAgICAgICAgYEVycm9yIGluIG1hdE11bDogaW5uZXIgc2hhcGVzICgke2lubmVyU2hhcGVBfSkgYW5kIChgICtcbiAgICAgICAgICAgIGAke2lubmVyU2hhcGVCfSkgb2YgTkRBcnJheXMgd2l0aCBzaGFwZXMgJHthLnNoYXBlfSBhbmQgYCArXG4gICAgICAgICAgICBgJHtiLnNoYXBlfSBhbmQgb3JpZW50YXRpb25zICR7TWF0cml4T3JpZW50YXRpb25bYU9yaWVudGF0aW9uXX1gICtcbiAgICAgICAgICAgIGAgYW5kICR7TWF0cml4T3JpZW50YXRpb25bYk9yaWVudGF0aW9uXX0gbXVzdCBtYXRjaC5gKTtcblxuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMubWF0TXVsSW50ZXJuYWwoYSwgYiwgYU9yaWVudGF0aW9uLCBiT3JpZW50YXRpb24pKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgbWF0TXVsSW50ZXJuYWwoXG4gICAgICBhOiBBcnJheTJELCBiOiBBcnJheTJELCBhT3JpZW50YXRpb246IE1hdHJpeE9yaWVudGF0aW9uLFxuICAgICAgYk9yaWVudGF0aW9uOiBNYXRyaXhPcmllbnRhdGlvbik6IEFycmF5MkQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBkb3QgcHJvZHVjdCBvZiBhIHZlY3RvciBhbmQgYSBtYXRyaXgsIHYgKiBCLlxuICAgKiBAcGFyYW0gdiBUaGUgdmVjdG9yIGluIGRvdCBwcm9kdWN0IG9wZXJhdGlvbi5cbiAgICogQHBhcmFtIG1hdHJpeCBUaGUgbWF0cml4IGluIGRvdCBwcm9kdWN0IG9wZXJhdGlvbi5cbiAgICovXG4gIHZlY3RvclRpbWVzTWF0cml4KHY6IEFycmF5MUQsIG1hdHJpeDogQXJyYXkyRCk6IEFycmF5MUQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB2LnJhbmsgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiB2ZWN0b3JUaW1lc01hdHJpeDogZmlyc3QgaW5wdXQgbXVzdCBiZSByYW5rIDEsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke3YucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIG1hdHJpeC5yYW5rID09PSAyLFxuICAgICAgICBgRXJyb3IgaW4gdmVjdG9yVGltZXNNYXRyaXg6IHNlY29uZCBpbnB1dCBtdXN0IGJlIHJhbmsgMiwgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGByYW5rICR7bWF0cml4LnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB2LnNpemUgPT09IG1hdHJpeC5zaGFwZVswXSxcbiAgICAgICAgYEVycm9yIGluIHZlY3RvclRpbWVzTWF0cml4OiBzaXplIG9mIGZpcnN0IHJhbmsgMSBpbnB1dCAoJHt2LnNpemV9KSBgICtcbiAgICAgICAgICAgIGBtdXN0IG1hdGNoIGlubmVyIGRpbWVuc2lvbiBvZiBzZWNvbmQgcmFuayAyIGlucHV0LCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYHJhbmsgJHttYXRyaXgucmFua30uYCk7XG5cbiAgICByZXR1cm4gdGhpcy5tYXRNdWwodi5hczJEKDEsIHYuc2l6ZSksIG1hdHJpeCkuYXMxRCgpO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBkb3QgcHJvZHVjdCBvZiBhIG1hdHJpeCBhbmQgdmVjdG9yLCBBICogdi5cbiAgICogQHBhcmFtIG1hdHJpeCBUaGUgbWF0cml4IGluIGRvdCBwcm9kdWN0IG9wZXJhdGlvbi5cbiAgICogQHBhcmFtIHYgVGhlIHZlY3RvciBpbiBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqL1xuICBtYXRyaXhUaW1lc1ZlY3RvcihtYXRyaXg6IEFycmF5MkQsIHY6IEFycmF5MUQpOiBBcnJheTFEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdi5yYW5rID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gdmVjdG9yVGltZXNNYXRyaXg6IHNlY29uZCBpbnB1dCBtdXN0IHJhbmsgMSwgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGByYW5rICR7di5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgbWF0cml4LnJhbmsgPT09IDIsXG4gICAgICAgIGBFcnJvciBpbiB2ZWN0b3JUaW1lc01hdHJpeDogZmlyc3QgaW5wdXQgbXVzdCBiZSBhIHJhbmsgMiwgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGByYW5rICR7bWF0cml4LnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB2LnNpemUgPT09IG1hdHJpeC5zaGFwZVsxXSxcbiAgICAgICAgYEVycm9yIGluIHZlY3RvclRpbWVzTWF0cml4OiBzaXplIG9mIGZpcnN0IHJhbmsgMSBpbnB1dCAke3Yuc2l6ZX0gYCArXG4gICAgICAgICAgICBgbXVzdCBtYXRjaCBpbm5lciBkaW1lbnNpb24gb2Ygc2Vjb25kIHJhbmsgMiBpbnB1dCwgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGBzaGFwZSAke21hdHJpeC5zaGFwZX0uYCk7XG5cbiAgICByZXR1cm4gdGhpcy5tYXRNdWwobWF0cml4LCB2LmFzMkQodi5zaXplLCAxKSkuYXMxRCgpO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBkb3QgcHJvZHVjdCBvZiB0d28gdmVjdG9ycywgdjEgKiB2Mi5cbiAgICogQHBhcmFtIHYxIFRoZSBmaXJzdCB2ZWN0b3IgaW4gdGhlIGRvdCBwcm9kdWN0IG9wZXJhdGlvbi5cbiAgICogQHBhcmFtIHYyIFRoZSBzZWNvbmQgdmVjdG9yIGluIHRoZSBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqL1xuICBkb3RQcm9kdWN0KHYxOiBBcnJheTFELCB2MjogQXJyYXkxRCk6IFNjYWxhciB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHYxLnJhbmsgPT09IDEgJiYgdjIucmFuayA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIGRvdFByb2R1Y3Q6IGlucHV0cyBtdXN0IGJlIHJhbmsgMSwgYnV0IGdvdCByYW5rcyBgICtcbiAgICAgICAgICAgIGAke3YxLnJhbmt9IGFuZCAke3YyLnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB2MS5zaXplID09PSB2Mi5zaXplLFxuICAgICAgICBgRXJyb3IgaW4gZG90UHJvZHVjdDogc2l6ZSBvZiBpbnB1dHMgKCR7djEuc2l6ZX0pIGFuZCAoYCArXG4gICAgICAgICAgICBgJHt2Mi5zaXplfSkgbXVzdCBtYXRjaC5gKTtcbiAgICByZXR1cm4gdGhpcy5tYXRNdWwodjEuYXMyRCgxLCB2MS5zaXplKSwgdjIuYXMyRCh2Mi5zaXplLCAxKSkuYXNTY2FsYXIoKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgb3V0ZXIgcHJvZHVjdCBvZiB0d28gdmVjdG9ycywgdjEgYW5kIHYyLlxuICAgKiBAcGFyYW0gdjEgVGhlIGZpcnN0IHZlY3RvciBpbiB0aGUgb3V0ZXIgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSB2MiBUaGUgc2Vjb25kIHZlY3RvciBpbiB0aGUgZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKi9cbiAgb3V0ZXJQcm9kdWN0KHYxOiBBcnJheTFELCB2MjogQXJyYXkxRCk6IEFycmF5MkQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB2MS5yYW5rID09PSAxICYmIHYyLnJhbmsgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBvdXRlclByb2R1Y3Q6IGlucHV0cyBtdXN0IGJlIHJhbmsgMSwgYnV0IGdvdCByYW5rcyBgICtcbiAgICAgICAgICAgIGAke3YxLnJhbmt9IGFuZCAke3YyLnJhbmt9LmApO1xuXG4gICAgcmV0dXJuIHRoaXMubWF0TXVsKHYxLmFzMkQodjEuc2l6ZSwgMSksIHYyLmFzMkQoMSwgdjIuc2l6ZSkpO1xuICB9XG5cbiAgLy8vLy8vLy8vLy8vLy8vXG4gIC8vIFNoYXBlIG9wcyAvL1xuICAvLy8vLy8vLy8vLy8vLy9cblxuICAvKipcbiAgICogQ2xvbmVzIGFuIE5EQXJyYXkgb2YgYW55IHNoYXBlLlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgTkRBcnJheSB0byBjbG9uZS5cbiAgICovXG4gIGNsb25lPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5jbG9uZUludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgY2xvbmVJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIEBkZXByZWNhdGVkIFBsZWFzZSBjYWxsIHJlc2hhcGUoKSBkaXJlY3RseSBvbiB0aGUgbmRhcnJheSBvYmplY3QuXG4gICAqL1xuICByZXNoYXBlPFQxIGV4dGVuZHMgTkRBcnJheSwgVDIgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIG5kYXJyYXk6IFQxLCBuZXdTaGFwZTogbnVtYmVyW10pOiBUMiB7XG4gICAgY29uc29sZS53YXJuKFxuICAgICAgICAnbWF0aC5yZXNoYXBlKCkgaXMgZGVwcmVjYXRlZC4gUGxlYXNlIGNhbGwgcmVzaGFwZSgpICcgK1xuICAgICAgICAnZGlyZWN0bHkgb24gdGhlIG5kYXJyYXkgb2JqZWN0Jyk7XG4gICAgcmV0dXJuIG5kYXJyYXkucmVzaGFwZShuZXdTaGFwZSk7XG4gIH1cblxuICAvKipcbiAgICogRXh0cmFjdHMgYSBzbGljZSBmcm9tIGEgbWF0cml4LiBUaGUgb3BlcmF0aW9uIGV4dHJhY2VzIGEgc2xpY2UgZnJvbSBpbnB1dFxuICAgKiB0aGF0IHN0YXJ0cyBhdCBjb29yZGluYXRlcyBgYmVnaW5gIGFuZCBpcyBvZiBzaXplIGBzaXplYC5cbiAgICogQHBhcmFtIGlucHV0IFRoZSBpbnB1dCBtYXRyaXggdG8gc2xpY2UgZnJvbS5cbiAgICogQHBhcmFtIGJlZ2luIFRoZSAyRCBjb29yZGluYXRlcyBpbiB0aGUgaW5wdXQgbWF0cml4IHRvIHN0YXJ0IHRoZSBzbGljZVxuICAgKiBmcm9tLlxuICAgKiBAcGFyYW0gc2l6ZSBUaGUgc2ljZSBvZiB0aGUgMkQgd2luZG93IHRvIHNsaWNlLlxuICAgKi9cbiAgc2xpY2UyRChpbnB1dDogQXJyYXkyRCwgYmVnaW46IFtudW1iZXIsIG51bWJlcl0sIHNpemU6IFtudW1iZXIsIG51bWJlcl0pOlxuICAgICAgQXJyYXkyRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGJlZ2luWzBdICsgc2l6ZVswXSA8PSBpbnB1dC5zaGFwZVswXSAmJlxuICAgICAgICAgICAgYmVnaW5bMV0gKyBzaXplWzFdIDw9IGlucHV0LnNoYXBlWzFdLFxuICAgICAgICBgRXJyb3IgaW4gc2xpY2UyRDogcmVxdWVzdGVkIHN0YXJ0IHBvc2l0aW9uICR7YmVnaW59IGFuZCBzaXplIGAgK1xuICAgICAgICAgICAgYCR7c2l6ZX0gd291bGQgb3ZlcmZsb3cgaW5wdXQgb2Ygc2hhcGUgJHtpbnB1dC5zaGFwZX0uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zbGljZTJESW50ZXJuYWwoaW5wdXQsIGJlZ2luLCBzaXplKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHNsaWNlMkRJbnRlcm5hbChcbiAgICAgIGlucHV0OiBBcnJheTJELCBiZWdpbjogW251bWJlciwgbnVtYmVyXSwgc2l6ZTogW251bWJlciwgbnVtYmVyXSk6IEFycmF5MkQ7XG5cbiAgLyoqXG4gICAqIENvcGllcyBhIHdpbmRvdyBmcm9tIHRoZSBgc291cmNlYCBtYXRyaXggc3RhcnRpbmcgYXQgYHNvdXJjZUJlZ2luYCBhbmQgaXNcbiAgICogb2Ygc2l6ZSBgc291cmNlU2l6ZWAgdG8gYSB3aW5kb3cgaW4gdGhlIGBkZXN0YCBtYXRyaXggc3RhcnRpbmcgYXRcbiAgICogYGRlc3RCZWdpbmAgYW5kIGlzIG9mIHNpemUgYGRlc3RTaXplYC9cbiAgICogQHBhcmFtIHNvdXJjZSBUaGUgc291cmNlIG1hdHJpeCB0byBjb3B5IGZyb20uXG4gICAqIEBwYXJhbSBzb3VyY2VCZWdpbiBUaGUgY29vcmRpbmF0ZXMgdG8gc3RhcnQgdGhlIGNvcHkgZnJvbS5cbiAgICogQHBhcmFtIHNvdXJjZVNpemUgVGhlIHNpemUgb2YgdGhlIGNvcHkgd2luZG93LlxuICAgKiBAcGFyYW0gZGVzdCBUaGUgZGVzdGluYXRpb24gbWF0cml4IHRvIGNvcHkgdG8uXG4gICAqIEBwYXJhbSBkZXN0QmVnaW4gVGhlIGNvb3JkaW5hdGVzIGluIGBkZXN0YCB0byBjb3B5IHRvLlxuICAgKiBAcGFyYW0gZGVzdFNpemUgVGhlIHNpemUgb2YgdGhlIGRlc3RpbmF0aW9uIHdpbmRvdy5cbiAgICovXG4gIGNvcHkyRChcbiAgICAgIHNvdXJjZTogQXJyYXkyRCwgc291cmNlQmVnaW46IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBzb3VyY2VTaXplOiBbbnVtYmVyLCBudW1iZXJdLCBkZXN0OiBBcnJheTJELCBkZXN0QmVnaW46IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBkZXN0U2l6ZTogW251bWJlciwgbnVtYmVyXSkge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBzb3VyY2VCZWdpblswXSArIHNvdXJjZVNpemVbMF0gPD0gc291cmNlLnNoYXBlWzBdICYmXG4gICAgICAgICAgICBzb3VyY2VCZWdpblsxXSArIHNvdXJjZVNpemVbMV0gPD0gc291cmNlLnNoYXBlWzFdLFxuICAgICAgICBgRXJyb3IgaW4gY29weTJEOiByZXF1ZXN0ZWQgc291cmNlIHN0YXJ0IHBvc2l0aW9uICR7c291cmNlQmVnaW59IGAgK1xuICAgICAgICAgICAgYGFuZCBzb3VyY2Ugc2l6ZSAke3NvdXJjZVNpemV9IHdvdWxkIG92ZXJmbG93IHNvdXJjZSBOREFycmF5YCArXG4gICAgICAgICAgICBgb2Ygc2hhcGUgJHtzb3VyY2Uuc2hhcGV9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBkZXN0QmVnaW5bMF0gKyBkZXN0U2l6ZVswXSA8PSBkZXN0LnNoYXBlWzBdICYmXG4gICAgICAgICAgICBkZXN0QmVnaW5bMV0gKyBkZXN0U2l6ZVsxXSA8PSBkZXN0LnNoYXBlWzFdLFxuICAgICAgICBgRXJyb3IgaW4gY29weTJEOiByZXF1ZXN0ZWQgZGVzdCBzdGFydCBwb3NpdGlvbiAke2Rlc3RCZWdpbn0gYCArXG4gICAgICAgICAgICBgYW5kIHNvdXJjZSBzaXplICR7ZGVzdFNpemV9IHdvdWxkIG92ZXJmbG93IGRlc3QgTkRBcnJheSBvZmAgK1xuICAgICAgICAgICAgYHNoYXBlICR7ZGVzdC5zaGFwZX0uYCk7XG4gICAgY29weTJkX3V0aWwudmFsaWRhdGVTaGFwZXMoc291cmNlU2l6ZSwgZGVzdFNpemUpO1xuXG4gICAgcmV0dXJuIHRoaXMuY29weTJESW50ZXJuYWwoXG4gICAgICAgIHNvdXJjZSwgc291cmNlQmVnaW4sIHNvdXJjZVNpemUsIGRlc3QsIGRlc3RCZWdpbiwgZGVzdFNpemUpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBjb3B5MkRJbnRlcm5hbChcbiAgICAgIHNvdXJjZTogQXJyYXkyRCwgc291cmNlQmVnaW46IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBzb3VyY2VTaXplOiBbbnVtYmVyLCBudW1iZXJdLCBkZXN0OiBBcnJheTJELCBkZXN0QmVnaW46IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBkZXN0U2l6ZTogW251bWJlciwgbnVtYmVyXSk6IHZvaWQ7XG5cbiAgLyoqXG4gICAqIENvbmNhdGVuYXRlcyB0d28gM0QgbmRhcnJheXMgYWxvbmcgYSBnaXZlbiBheGlzLlxuICAgKlxuICAgKiBGb3IgZXhhbXBsZSwgaWY6XG4gICAqIEE6IHNoYXBlKDIsIDEsIDMpID0gfCByMSwgZzEsIGIxIHxcbiAgICogICAgICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIgfFxuICAgKlxuICAgKiBCOiBzaGFwZSgyLCAxLCAzKSA9IHwgcjMsIGczLCBiMyB8XG4gICAqICAgICAgICAgICAgICAgICAgICAgfCByNCwgZzQsIGI0IHxcbiAgICpcbiAgICogQyA9IGNvbmNhdDNEKEEsIEIsIGF4aXMpXG4gICAqXG4gICAqIGlmIGF4aXMgPSAwOlxuICAgKiBDOiBzaGFwZSg0LCAxLCAzKSA9IHwgcjEsIGcxLCBiMSB8XG4gICAqICAgICAgICAgICAgICAgICAgICAgfCByMiwgZzIsIGIyIHxcbiAgICogICAgICAgICAgICAgICAgICAgICB8IHIzLCBnMywgYjMgfFxuICAgKiAgICAgICAgICAgICAgICAgICAgIHwgcjQsIGc0LCBiNCB8XG4gICAqXG4gICAqIGlmIGF4aXMgPSAxOlxuICAgKiBDOiBzaGFwZSgyLCAyLCAzKSA9IHwgcjEsIGcxLCBiMSwgcjMsIGczLCBiMyB8XG4gICAqICAgICAgICAgICAgICAgICAgICAgfCByMiwgZzIsIGIyLCByNCwgZzQsIGI0IHxcbiAgICpcbiAgICogaWYgYXhpcyA9IDI6XG4gICAqIEMgPSBzaGFwZSgyLCAxLCA2KSA9IHwgcjEsIGcxLCBiMSwgcjMsIGczLCBiMyB8XG4gICAqICAgICAgICAgICAgICAgICAgICAgIHwgcjIsIGcyLCBiMiwgcjQsIGc0LCBiNCB8XG4gICAqXG4gICAqIEBwYXJhbSBuZGFycmF5MSBUaGUgZmlyc3QgYXJyYXkgdG8gY29uY2F0LlxuICAgKiBAcGFyYW0gbmRhcnJheTIgVGhlIHNlY29uZCBhcnJheSB0byBjb25hdC5cbiAgICogQHBhcmFtIGF4aXMgVGhlIGF4aXMgdG8gY29uY2F0ZSBhbG9uZy5cbiAgICovXG4gIGNvbmNhdDNEKG5kYXJyYXkxOiBBcnJheTNELCBuZGFycmF5MjogQXJyYXkzRCwgYXhpczogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgY29uY2F0M2RfdXRpbC5hc3NlcnRDb25jYXQzRFNoYXBlc01hdGNoKFxuICAgICAgICBuZGFycmF5MS5zaGFwZSwgbmRhcnJheTIuc2hhcGUsIGF4aXMsICdFcnJvciBpbiBjb25jYXQzZDogJyk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5jb25jYXQzREludGVybmFsKG5kYXJyYXkxLCBuZGFycmF5MiwgYXhpcykpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBjb25jYXQzREludGVybmFsKFxuICAgICAgbmRhcnJheTE6IEFycmF5M0QsIG5kYXJyYXkyOiBBcnJheTNELCBheGlzOiBudW1iZXIpOiBBcnJheTNEO1xuXG4gIC8vLy8vLy8vLy8vLy8vLy8vLy9cbiAgLy8gUmVkdWN0aW9uIG9wcyAvL1xuICAvLy8vLy8vLy8vLy8vLy8vLy8vXG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSB0aGUgbG9nKHN1bShlIF4geCkpIGZvciBlYWNoIHggaW4gdGhlIGlucHV0IG5kYXJyYXkuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5IHRvIGNvbXB1dGUgdGhlIGxvZ1N1bUV4cCBvdmVyLlxuICAgKi9cbiAgbG9nU3VtRXhwKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMubG9nU3VtRXhwSW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBsb2dTdW1FeHBJbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgc3VtIG9mIGFsbCB0aGUgZW50cmllcyBpbiB0aGUgaW5wdXQgTkRBcnJheS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkgdG8gY29tcHV0ZSB0aGUgc3VtIG92ZXIuXG4gICAqL1xuICBzdW0obmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zdW1JbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHN1bUludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXI7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBmbGF0dGVuZWQgaW5kZXggb2YgdGhlIG1pbmltdW0gZWxlbWVudCBpbiB0aGUgbmRhcnJheS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICBhcmdNaW4obmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5hcmdNaW5JbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGFyZ01pbkludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXI7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBmbGF0dGVuZWQgaW5kZXggb2YgdGhlIG1heGltdW0gZWxlbWVudCBpbiB0aGUgbmRhcnJheS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICBhcmdNYXgobmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5hcmdNYXhJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGFyZ01heEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXI7XG5cbiAgLyoqXG4gICAqIFJldHVybnMgYSAxIGlmIHRoZSBhcmdNYXggb2YgeDEgYW5kIHgyIGFyZSB0aGUgc2FtZSwgb3RoZXJ3aXNlIDAuXG4gICAqIEBwYXJhbSB4MSBUaGUgZmlyc3QgaW5wdXQgTkRBcnJheS5cbiAgICogQHBhcmFtIHgyIFRoZSBzZWNvbmQgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIGFyZ01heEVxdWFscyh4MTogTkRBcnJheSwgeDI6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIHV0aWwuYXNzZXJ0U2hhcGVzTWF0Y2goeDEuc2hhcGUsIHgyLnNoYXBlLCAnRXJyb3IgaW4gYXJnTWF4RXF1YWxzOiAnKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmFyZ01heEVxdWFsc0ludGVybmFsKHgxLCB4MikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBhcmdNYXhFcXVhbHNJbnRlcm5hbCh4MTogTkRBcnJheSwgeDI6IE5EQXJyYXkpOiBTY2FsYXI7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSB0b3AgSyB2YWx1ZXMgYW5kIGZsYXR0ZW5lZCBpbmRpY2VzLlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICogQHBhcmFtIGsgSG93IG1hbnkgdG9wIHZhbHVlcyB0byBjb21wdXRlLlxuICAgKi9cbiAgdG9wSyhuZGFycmF5OiBOREFycmF5LCBrOiBudW1iZXIpOiB7dmFsdWVzOiBBcnJheTFELCBpbmRpY2VzOiBBcnJheTFEfSB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGsgPD0gbmRhcnJheS5zaXplLFxuICAgICAgICBgRXJyb3IgaW4gdG9wSzogayB2YWx1ZSAoJHtrfSkgbXVzdCBiZSBsZXNzIHRoYW4gc2l6ZSBvZiBpbnB1dCBgICtcbiAgICAgICAgICAgIGBuZGFycmF5LCBnb3Qgc2hhcGUgJHtuZGFycmF5LnNoYXBlfS5gKTtcbiAgICBjb25zdCByZXN1bHQgPSB0aGlzLnRvcEtJbnRlcm5hbChuZGFycmF5LCBrKTtcbiAgICB0aGlzLnRyYWNrKHJlc3VsdC52YWx1ZXMpO1xuICAgIHRoaXMudHJhY2socmVzdWx0LmluZGljZXMpO1xuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHRvcEtJbnRlcm5hbChuZGFycmF5OiBOREFycmF5LCBrOiBudW1iZXIpOlxuICAgICAge3ZhbHVlczogQXJyYXkxRCwgaW5kaWNlczogQXJyYXkxRH07XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBtaW5pbXVtIHZhbHVlIGZyb20gdGhlIGlucHV0LlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIG1pbihuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLm1pbkludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgbWluSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIG1heGltdW0gdmFsdWUgZnJvbSB0aGUgaW5wdXQuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgbWF4KG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMubWF4SW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBtYXhJbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgc29mdG1heCBub3JtYWxpemVkIHZlY3RvciBmcm9tIHRoZSBpbnB1dCB2ZWN0b3IuXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCB2ZWN0b3IuXG4gICAqL1xuICBzb2Z0bWF4KHg6IEFycmF5MUQpOiBBcnJheTFEIHtcbiAgICByZXR1cm4gdGhpcy5zY29wZSgoKSA9PiB7XG4gICAgICAvLyBEbyBpdCBpbiBsb2cgc3BhY2UgZm9yIG51bWVyaWNhbCBzdGFiaWxpdHkuXG4gICAgICAvLyBleHAoWCAtIGxvZ1N1bUV4cChYKSlcbiAgICAgIGNvbnN0IGxzZSA9IHRoaXMubG9nU3VtRXhwKHgpO1xuICAgICAgY29uc3QgbG9nUmVzdWx0ID0gdGhpcy5hcnJheU1pbnVzU2NhbGFyKHgsIGxzZSk7XG4gICAgICByZXR1cm4gdGhpcy5leHAobG9nUmVzdWx0KTtcbiAgICB9KTtcbiAgfVxuXG4gIC8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cbiAgLy8gRWxlbWVudC13aXNlIG9wcyAvL1xuICAvLy8vLy8vLy8vLy8vLy8vLy8vLy8vXG5cbiAgLyoqXG4gICAqIFN3aXRjaGVzIGRpbWVuc2lvbnMgb2YgdGhlIGlucHV0IE5EQXJyYXkuXG4gICAqIEBwYXJhbSBhIFRoZSBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0gbmV3RGltIFRoZSBuZXcgaW5kaWNlcyB0aGF0IGRlZmluZSB3aGljaCBzaGFwZXMgdmFsdWVzIHRvIHN3aXRjaC5cbiAgICovXG4gIHN3aXRjaERpbTxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgbmV3RGltOiBudW1iZXJbXSk6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBhLnJhbmsgPT09IG5ld0RpbS5sZW5ndGgsXG4gICAgICAgIGBFcnJvciBpbiBzd2l0Y2hEaW06IGxlbmd0aCBvZiBpbnB1dCBzaGFwZSAke2Euc2hhcGV9IGAgK1xuICAgICAgICAgICAgYG11c3QgbWF0Y2ggc2l6ZSBvZiBuZXdEaW0gYXJyYXkgJHtuZXdEaW19LmApO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuc3dpdGNoRGltSW50ZXJuYWwoYSwgbmV3RGltKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHN3aXRjaERpbUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIGE6IFQsIG5ld0RpbTogbnVtYmVyW10pOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBhIHNjYWxhciBwbHVzIE5EQXJyYXksIGMgKyBBLlxuICAgKiBAcGFyYW0gYyBUaGUgc2NhbGFyIGMgaW4gYyArIEEuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IEEgaW4gYyArIEEuXG4gICAqL1xuICBzY2FsYXJQbHVzQXJyYXk8VCBleHRlbmRzIE5EQXJyYXk+KGM6IFNjYWxhciwgYTogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBzY2FsYXJQbHVzQXJyYXk6IGZpcnN0IGFyZ3VtZW50IG11c3QgYmUgcmFuayAwLCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYHJhbmsgJHtjLnJhbmt9LmApO1xuICAgIHJldHVybiB0aGlzLmFkZChjLCBhKSBhcyBUO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgc2NhbGFyIG1pbnVzIE5EQXJyYXksIGMgLSBBLlxuICAgKiBAcGFyYW0gYyBUaGUgc2NhbGFyIGMgaW4gYyAtIEEuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IEEgaW4gYyAtIEEuXG4gICAqL1xuICBzY2FsYXJNaW51c0FycmF5PFQgZXh0ZW5kcyBOREFycmF5PihjOiBTY2FsYXIsIGE6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYy5zaXplID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gc2NhbGFyTWludXNBcnJheTogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMuc3ViKGMsIGEpIGFzIFQ7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgQSAtIGMuIEEgaXMgTkRBcnJheSwgYyBpcyBTY2FsYXIuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IEEgaW4gQSAtIGMuXG4gICAqIEBwYXJhbSBjIFRoZSBTY2FsYXIgYyBpbiBBIC0gYy5cbiAgICovXG4gIGFycmF5TWludXNTY2FsYXI8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGM6IFNjYWxhcik6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBhcnJheU1pbnVzU2NhbGFyOiBzZWNvbmQgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBgICtcbiAgICAgICAgICAgIGBnb3QgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMuc3ViKGEsIGMpIGFzIFQ7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgLTEgKiBBIGVsZW1lbnQtd2lzZS5cbiAgICogQHBhcmFtIGEgVGhlIGlucHV0IGFycmF5LlxuICAgKi9cbiAgbmVnPFQgZXh0ZW5kcyBOREFycmF5PihhOiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5uZWdJbnRlcm5hbChhKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG5lZ0ludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBUKTogVDtcblxuICAvKipcbiAgICogQWRkcyB0d28gTkRBcnJheXMgZWxlbWVudC13aXNlLCBBICsgQi4gU3VwcG9ydHMgYnJvYWRjYXN0aW5nLlxuICAgKiBGb3IgYSBzdHJpY3RlciB2ZXJzaW9uIHdpdGhvdXQgYnJvYWRjYXN0aW5nIHVzZSBtYXRoLmFkZFN0cmljdCgpLlxuICAgKlxuICAgKiBAcGFyYW0gYSBUaGUgZmlyc3QgTkRBcnJheSB0byBhZGQgZWxlbWVudC13aXNlLlxuICAgKiBAcGFyYW0gYiBUaGUgc2Vjb25kIE5EQXJyYXkgdG8gYWRkIGVsZW1lbnQtd2lzZS5cbiAgICovXG4gIGFkZChhOiBOREFycmF5LCBiOiBOREFycmF5KTogTkRBcnJheSB7XG4gICAgdXRpbC5hc3NlcnRBbmRHZXRCcm9hZGNhc3RlZFNoYXBlKGEuc2hhcGUsIGIuc2hhcGUpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuYWRkSW50ZXJuYWwoYSwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBhZGRJbnRlcm5hbChhOiBOREFycmF5LCBiOiBOREFycmF5KTogTkRBcnJheTtcblxuICAvKipcbiAgICogQWRkcyB0d28gTkRBcnJheXMgZWxlbWVudC13aXNlLCBBICsgQi4gSW5wdXRzIG11c3RcbiAgICogYmUgdGhlIHNhbWUgc2hhcGUuIEZvciBicm9hZGNhc3Rpbmcgc3VwcG9ydCwgdXNlIG1hdGguYWRkKCkgaW5zdGVhZC5cbiAgICpcbiAgICogQHBhcmFtIGEgVGhlIGZpcnN0IE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKiBAcGFyYW0gYiBUaGUgc2Vjb25kIE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKi9cbiAgYWRkU3RyaWN0PFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChhLnNoYXBlLCBiLnNoYXBlLCAnRXJyb3IgaW4gYWRkU3RyaWN0OiAnKTtcbiAgICByZXR1cm4gdGhpcy5hZGQoYSwgYikgYXMgVDtcbiAgfVxuXG4gIC8qKlxuICAgKiBTdWJ0cmFjdHMgdHdvIE5EQXJyYXlzIGVsZW1lbnQtd2lzZSwgQSAtIEIuIFN1cHBvcnRzIGJyb2FkY2FzdGluZy5cbiAgICogRm9yIGEgc3RyaWN0ZXIgdmVyc2lvbiB3aXRob3V0IGJyb2FkY2FzdGluZyB1c2UgbWF0aC5zdWJTdHJpY3QoKS5cbiAgICpcbiAgICogQHBhcmFtIGEgVGhlIGZpcnN0IE5EQXJyYXkgdG8gc3VidHJhY3QgZWxlbWVudC13aXNlLlxuICAgKiBAcGFyYW0gYiBUaGUgc2Vjb25kIE5EQXJyYXkgdG8gc3VidHJhY3QgZWxlbWVudC13aXNlLlxuICAgKi9cbiAgc3ViKGE6IE5EQXJyYXksIGI6IE5EQXJyYXkpOiBOREFycmF5IHtcbiAgICB1dGlsLmFzc2VydEFuZEdldEJyb2FkY2FzdGVkU2hhcGUoYS5zaGFwZSwgYi5zaGFwZSk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zdWJJbnRlcm5hbChhLCBiKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHN1YkludGVybmFsKGE6IE5EQXJyYXksIGI6IE5EQXJyYXkpOiBOREFycmF5O1xuXG4gIC8qKlxuICAgKiBTdWJ0cmFjdHMgdHdvIE5EQXJyYXlzIGVsZW1lbnQtd2lzZSwgQSAtIEIuIElucHV0cyBtdXN0XG4gICAqIGJlIHRoZSBzYW1lIHNoYXBlLiBGb3IgYnJvYWRjYXN0aW5nIHN1cHBvcnQsIHVzZSBtYXRoLnN1YigpIGluc3RlYWQuXG4gICAqXG4gICAqIEBwYXJhbSBhIFRoZSBmaXJzdCBOREFycmF5IHRvIG11bHRpcGx5IGVsZW1lbnQtd2lzZS5cbiAgICogQHBhcmFtIGIgVGhlIHNlY29uZCBOREFycmF5IHRvIG11bHRpcGx5IGVsZW1lbnQtd2lzZS5cbiAgICovXG4gIHN1YlN0cmljdDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0U2hhcGVzTWF0Y2goYS5zaGFwZSwgYi5zaGFwZSwgJ0Vycm9yIGluIHN1YlN0cmljdDogJyk7XG4gICAgcmV0dXJuIHRoaXMuc3ViKGEsIGIpIGFzIFQ7XG4gIH1cblxuICAvKipcbiAgICogTXVsdGlwbGllcyB0d28gTkRBcnJheXMgZWxlbWVudC13aXNlLCBBICogQi4gU3VwcG9ydHMgYnJvYWRjYXN0aW5nLlxuICAgKiBGb3IgYSBzdHJpY3RlciB2ZXJzaW9uIHdpdGhvdXQgYnJvYWRjYXN0aW5nIHVzZSBtYXRoLm11bHRpcGx5U3RyaWN0KCkuXG4gICAqXG4gICAqIEBwYXJhbSBhIFRoZSBmaXJzdCBOREFycmF5IHRvIG11bHRpcGx5IGVsZW1lbnQtd2lzZS5cbiAgICogQHBhcmFtIGIgVGhlIHNlY29uZCBOREFycmF5IHRvIG11bHRpcGx5IGVsZW1lbnQtd2lzZS5cbiAgICovXG4gIG11bHRpcGx5KGE6IE5EQXJyYXksIGI6IE5EQXJyYXkpOiBOREFycmF5IHtcbiAgICB1dGlsLmFzc2VydEFuZEdldEJyb2FkY2FzdGVkU2hhcGUoYS5zaGFwZSwgYi5zaGFwZSk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5tdWx0aXBseUludGVybmFsKGEsIGIpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgbXVsdGlwbHlJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIEBkZXByZWNhdGVkIFVzZSBtYXRoLm11bHRpcGx5U3RyaWN0KCkgaW5zdGVhZC5cbiAgICovXG4gIGVsZW1lbnRXaXNlTXVsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMubXVsdGlwbHlTdHJpY3QoYSwgYik7XG4gIH1cblxuICAvKipcbiAgICogTXVsdGlwbGllcyB0d28gTkRBcnJheXMgZWxlbWVudC13aXNlLCBBICogQi4gSW5wdXRzIG11c3RcbiAgICogYmUgdGhlIHNhbWUgc2hhcGUuIEZvciBicm9hZGNhc3Rpbmcgc3VwcG9ydCwgdXNlIG1hdGgubXVsdGlwbHkoKSBpbnN0ZWFkLlxuICAgKlxuICAgKiBAcGFyYW0gYSBUaGUgZmlyc3QgTkRBcnJheSB0byBtdWx0aXBseSBlbGVtZW50LXdpc2UuXG4gICAqIEBwYXJhbSBiIFRoZSBzZWNvbmQgTkRBcnJheSB0byBtdWx0aXBseSBlbGVtZW50LXdpc2UuXG4gICAqL1xuICBtdWx0aXBseVN0cmljdDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0U2hhcGVzTWF0Y2goYS5zaGFwZSwgYi5zaGFwZSwgJ0Vycm9yIGluIG11bHRpcGx5U3RyaWN0OiAnKTtcbiAgICByZXR1cm4gdGhpcy5tdWx0aXBseShhLCBiKSBhcyBUO1xuICB9XG5cbiAgLyoqXG4gICAqIERpdmlkZXMgdHdvIE5EQXJyYXlzIGVsZW1lbnQtd2lzZSwgQSAvIEIuIFN1cHBvcnRzIGJyb2FkY2FzdGluZy5cbiAgICogRm9yIGEgc3RyaWN0ZXIgdmVyc2lvbiB3aXRob3V0IGJyb2FkY2FzdGluZyB1c2UgbWF0aC5kaXZpZGVTdHJpY3QoKS5cbiAgICpcbiAgICogQHBhcmFtIGEgVGhlIGZpcnN0IE5EQXJyYXkgdG8gZGl2aWRlIGVsZW1lbnQtd2lzZS5cbiAgICogQHBhcmFtIGIgVGhlIHNlY29uZCBOREFycmF5IHRvIGRpdmlkZSBlbGVtZW50LXdpc2UuXG4gICAqL1xuICBkaXZpZGUoYTogTkRBcnJheSwgYjogTkRBcnJheSk6IE5EQXJyYXkge1xuICAgIHV0aWwuYXNzZXJ0QW5kR2V0QnJvYWRjYXN0ZWRTaGFwZShhLnNoYXBlLCBiLnNoYXBlKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmRpdmlkZUludGVybmFsKGEsIGIpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgZGl2aWRlSW50ZXJuYWwoYTogTkRBcnJheSwgYjogTkRBcnJheSk6IE5EQXJyYXk7XG5cbiAgLyoqXG4gICAqIERpdmlkZXMgdHdvIE5EQXJyYXlzIGVsZW1lbnQtd2lzZSwgQSAvIEIuIElucHV0cyBtdXN0XG4gICAqIGJlIHRoZSBzYW1lIHNoYXBlLiBGb3IgYnJvYWRjYXN0aW5nIHN1cHBvcnQsIHVzZSBtYXRoLmRpdmlkZSgpIGluc3RlYWQuXG4gICAqXG4gICAqIEBwYXJhbSBhIFRoZSBmaXJzdCBOREFycmF5IHRvIG11bHRpcGx5IGVsZW1lbnQtd2lzZS5cbiAgICogQHBhcmFtIGIgVGhlIHNlY29uZCBOREFycmF5IHRvIG11bHRpcGx5IGVsZW1lbnQtd2lzZS5cbiAgICovXG4gIGRpdmlkZVN0cmljdDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0U2hhcGVzTWF0Y2goYS5zaGFwZSwgYi5zaGFwZSwgJ0Vycm9yIGluIGRpdmlkZVN0cmljdDogJyk7XG4gICAgcmV0dXJuIHRoaXMuZGl2aWRlKGEsIGIpIGFzIFQ7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgYSBzY2FsYXIgZGl2aWRlZCBieSBhbiBOREFycmF5LCBicm9hZGNhc3RlZCBvdmVyIHRoZSBOREFycmF5LCBjIC9cbiAgICogQS5cbiAgICogQHBhcmFtIGMgVGhlIHNjYWxhciB2YWx1ZSBpbiBjIC8gQS5cbiAgICogQHBhcmFtIGEgVGhlIE5EQXJyYXkgdmFsdWUgaW4gYyAvIEEuXG4gICAqL1xuICBzY2FsYXJEaXZpZGVkQnlBcnJheTxUIGV4dGVuZHMgTkRBcnJheT4oYzogU2NhbGFyLCBhOiBUKTogVCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGMuc2l6ZSA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIHNjYWxhckRpdmlkZWRCeUFycmF5OiBmaXJzdCBhcmd1bWVudCBtdXN0IGJlIHJhbmsgMCwgYnV0IGAgK1xuICAgICAgICAgICAgYGdvdCBOREFycmF5IG9mIHJhbmsgJHtjLnJhbmt9LmApO1xuICAgIHJldHVybiB0aGlzLmRpdmlkZShjLCBhKSBhcyBUO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGFuIE5EQXJyYXkgZGl2aWRlZCBieSBhIHNjYWxhciwgYnJvYWRjYXN0ZWQgb3ZlciB0aGUgTkRBcnJheSwgQSAvXG4gICAqIGMuXG4gICAqIEBwYXJhbSBhIFRoZSBOREFycmF5IHZhbHVlIGluIEEgLyBjLlxuICAgKiBAcGFyYW0gYyBUaGUgc2NhbGFyIHZhbHVlIGluIEEgLyBjLlxuICAgKi9cbiAgYXJyYXlEaXZpZGVkQnlTY2FsYXI8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGM6IFNjYWxhcik6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBhcnJheURpdmlkZWRCeVNjYWxhcjogc2Vjb25kIGFyZ3VtZW50IG11c3QgYmUgcmFuayAwLCBgICtcbiAgICAgICAgICAgIGBidXQgZ290IE5EQXJyYXkgb2YgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMuZGl2aWRlKGEsIGMpIGFzIFQ7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgZXhwb25lbnRpYWwgb2YgdGhlIGlucHV0IE5EQXJyYXkgZWxlbWVudC13aXNlLiB5ID0gZSBeIHhcbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICBleHA8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmV4cEludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgZXhwSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBuYXR1cmFsIGxvZ2FyaXRobSBvZiB0aGUgaW5wdXQgTkRBcnJheSBlbGVtZW50LXdpc2UuIHkgPSBsbih4KVxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIGxvZzxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMubG9nSW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBsb2dJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHJlY3RpZmllZCBsaW5lYXIgZWxlbWVudC13aXNlLCBtYXgoeCwgMCkuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgcmVsdTxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMucmVsdUludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgcmVsdUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgc2lnbW9pZCBlbGVtZW50LXdpc2UsIHkgPSAxIC8gKDEgKyBleHAoLXgpKS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICBzaWdtb2lkPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zaWdtb2lkSW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzaWdtb2lkSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBoeXBlcmJvbGljIHRhbmdlbnQgb2YgdGhlIGlucHV0IE5EQXJyYXkgZWxlbWVudC13aXNlLlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIHRhbmg8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnRhbmhJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHRhbmhJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHNpbiBvZiB0aGUgaW5wdXQgTkRBcnJheSBlbGVtZW50LXdpc2UsIHkgPSBzaW4oeCkuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgc2luPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zaW5JbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHNpbkludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgc3RlcCBvZiB0aGUgaW5wdXQgTkRBcnJheSBlbGVtZW50LXdpc2UsIHkgPSAxIGlmIHggPiAwIHwgMCBpZiB4IDw9XG4gICAqIDBcbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICBzdGVwPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zdGVwSW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzdGVwSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBhIHNjYWxlZCBhcnJheSBhZGQgb3BlcmF0aW9uLCBjMSAqIEEgKyBjMiAqIEIuXG4gICAqIEBwYXJhbSBjMSBUaGUgZmlyc3Qgc2NhbGFyIGluIHRoZSBzY2FsZWQgYXJyYXkgYWRkIGNvbXB1dGF0aW9uLlxuICAgKiBAcGFyYW0gYSBUaGUgZmlyc3QgTkRBcnJheSBpbiB0aGUgc2NhbGVkIGFycmF5IGFkZCBjb21wdXRhdGlvbi5cbiAgICogQHBhcmFtIGMyIFRoZSBzZWNvbmQgc2NhbGFyIGluIHRoZSBzY2FsZWQgYXJyYXkgYWRkIGNvbXB1dGF0aW9uLlxuICAgKiBAcGFyYW0gY2IgVGhlIHNlY29uZCBOREFycmF5IGluIHRoZSBzY2FsZWQgYXJyYXkgYWRkIGNvbXB1dGF0aW9uLlxuICAgKi9cbiAgc2NhbGVkQXJyYXlBZGQ8VCBleHRlbmRzIE5EQXJyYXk+KGMxOiBTY2FsYXIsIGE6IFQsIGMyOiBTY2FsYXIsIGI6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYzEuc2l6ZSA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIHNjYWxlZEFycmF5QWRkOiBmaXJzdCBhcmd1bWVudCBtdXN0IHJhbmsgMCwgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGAgcmFuayAke2MxLnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjMi5zaXplID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gc2NhbGVkQXJyYXlBZGQ6IHRoaXJkIGFyZ3VtZW50IG11c3QgYmUgcmFuayAwLCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYE5EQXJyYXkgb2YgcmFuayAke2MyLnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0U2hhcGVzTWF0Y2goYS5zaGFwZSwgYi5zaGFwZSwgJ0Vycm9yIGluIHNjYWxlZEFycmF5QWRkOiAnKTtcblxuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuc2NhbGVkQXJyYXlBZGRJbnRlcm5hbChjMSwgYSwgYzIsIGIpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc2NhbGVkQXJyYXlBZGRJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oXG4gICAgICBjMTogU2NhbGFyLCBhOiBULCBjMjogU2NhbGFyLCBiOiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgYSBzY2FsYXIgdGltZXMgYXJyYXkgb3BlcmF0aW9uIGJyb2FkY2FzdGVkIG92ZXIgdGhlIE5EQXJyYXksIGMgKlxuICAgKiBBLlxuICAgKiBAcGFyYW0gYyBUaGUgc2NhbGFyIGluIHRoZSBvcGVyYXRpb24uXG4gICAqIEBwYXJhbSBBIHRoZSBOREFycmF5IGluIHRoZSBvcGVyYXRpb24gdGhhdCB3aWxsIGJlIGJyb2FkY2FzdGVkIG92ZXIuXG4gICAqL1xuICBzY2FsYXJUaW1lc0FycmF5PFQgZXh0ZW5kcyBOREFycmF5PihjOiBTY2FsYXIsIGE6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYy5zaXplID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gYXJyYXlEaXZpZGVkQnlTY2FsYXI6IGZpcnN0IGFyZ3VtZW50IG11c3QgYmUgcmFuayAwLCBidXQgYCArXG4gICAgICAgICAgICBgZ290IHJhbmsgJHtjLnJhbmt9LmApO1xuICAgIHJldHVybiB0aGlzLm11bHRpcGx5KGMsIGEpIGFzIFQ7XG4gIH1cblxuICAvKipcbiAgICogQGRlcHJlY2F0ZWQgVXNlIG1hdGgubXVsdGlwbHkoKSBpbnN0ZWFkLlxuICAgKi9cbiAgZWxlbWVudFdpc2VNdWxCcm9hZGNhc3QoYTogQXJyYXkyRCwgYjogQXJyYXkyRCk6IEFycmF5MkQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBhLnJhbmsgPT09IDIsXG4gICAgICAgIGBFcnJvciBpbiBlbGVtZW50V2lzZU11bEJyb2FkY2FzdDogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSBgICtcbiAgICAgICAgICAgIGByYW5rIDIsIGJ1dCBnb3QgcmFuayAke2EucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGIucmFuayA9PT0gMixcbiAgICAgICAgYEVycm9yIGluIGVsZW1lbnRXaXNlTXVsQnJvYWRjYXN0OiBzZWNvbmQgYXJndW1lbnQgbXVzdCBiZSBgICtcbiAgICAgICAgICAgIGByYW5rIDIsIGJ1dCBnb3QgcmFuayAke2IucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMubXVsdGlwbHkoYSwgYikgYXMgQXJyYXkyRDtcbiAgfVxuXG4gIC8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuICAvLyBDb252b2x1dGlvbiBvcHMgLy9cbiAgLy8vLy8vLy8vLy8vLy8vLy8vLy8vXG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgMkQgY29udm9sdXRpb24gb3ZlciB0aGUgaW5wdXQgeC5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IGltYWdlLCByYW5rIDMsIG9mIHNoYXBlIFtoZWlnaHQsIHdpZHRoLCBpbkRlcHRoXS5cbiAgICogQHBhcmFtIHdlaWdodHMgVGhlIHdlaWdodHMsIHJhbmsgNCwgb2Ygc2hhcGVcbiAgICogICAgIFtmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoLCBpbkRlcHRoLCBvdXREZXB0aF0uXG4gICAqIEBwYXJhbSBiaWFzIE9wdGlvbmFsIGJpYXMsIHJhbmsgMSBvZiBzaGFwZSBbb3V0RGVwdGhdLlxuICAgKiBAcGFyYW0gc3RyaWRlcyBUaGUgc3RyaWRlcyBvZiB0aGUgY29udm9sdXRpb246IFtzdHJpZGVIZWlnaHQsIHN0cmlkZVdpZHRoXS5cbiAgICogQHBhcmFtIHBhZCBBIHN0cmluZyBmcm9tOiAnc2FtZScsICd2YWxpZCcuIFRoZSB0eXBlIG9mIHBhZGRpbmcgYWxnb3JpdGhtLlxuICAgKi9cbiAgY29udjJkKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhczogQXJyYXkxRHxudWxsLFxuICAgICAgc3RyaWRlczogW251bWJlciwgbnVtYmVyXXxudW1iZXIsIHBhZDogJ3ZhbGlkJ3wnc2FtZSd8bnVtYmVyKTogQXJyYXkzRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZDogeCBtdXN0IGJlIHJhbmsgMywgYnV0IGdvdCByYW5rICR7eC5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgd2VpZ2h0cy5yYW5rID09PSA0LFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkOiB3ZWlnaHRzIG11c3QgYmUgcmFuayA0LCBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgICBgJHt3ZWlnaHRzLnJhbmt9LmApO1xuICAgIGlmIChiaWFzICE9IG51bGwpIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGJpYXMucmFuayA9PT0gMSxcbiAgICAgICAgICBgRXJyb3IgaW4gY29udjJkOiBiaWFzIG11c3QgYmUgcmFuayAxLCBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgICAgIGAke2JpYXMucmFua30uYCk7XG4gICAgfVxuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHguc2hhcGVbMl0gPT09IHdlaWdodHMuc2hhcGVbMl0sXG4gICAgICAgIGBFcnJvciBpbiBjb252MmQ6IGRlcHRoIG9mIGlucHV0ICgke3guc2hhcGVbMl19KSBtdXN0IG1hdGNoICBgICtcbiAgICAgICAgICAgIGBpbnB1dCBkZXB0aCBmb3Igd2VpZ2h0cyAke3dlaWdodHMuc2hhcGVbMl19LmApO1xuXG4gICAgY29uc3QgZmlsdGVySGVpZ2h0ID0gd2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBmaWx0ZXJXaWR0aCA9IHdlaWdodHMuc2hhcGVbMV07XG4gICAgY29uc3Qgb3V0RGVwdGggPSB3ZWlnaHRzLnNoYXBlWzNdO1xuICAgIGxldCBzdHJpZGVIZWlnaHQ6IG51bWJlcjtcbiAgICBsZXQgc3RyaWRlV2lkdGg6IG51bWJlcjtcbiAgICBpZiAodHlwZW9mIHN0cmlkZXMgPT09ICdudW1iZXInKSB7XG4gICAgICBzdHJpZGVIZWlnaHQgPSBzdHJpZGVzO1xuICAgICAgc3RyaWRlV2lkdGggPSBzdHJpZGVzO1xuICAgIH0gZWxzZSB7XG4gICAgICBzdHJpZGVIZWlnaHQgPSBzdHJpZGVzWzBdO1xuICAgICAgc3RyaWRlV2lkdGggPSBzdHJpZGVzWzFdO1xuICAgIH1cbiAgICBjb25zdCBvdXRwdXRJbmZvID0gY29udl91dGlsLmNvbXB1dGVPdXRwdXRJbmZvKFxuICAgICAgICB4LnNoYXBlLCBmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoLCBvdXREZXB0aCwgc3RyaWRlSGVpZ2h0LCBzdHJpZGVXaWR0aCxcbiAgICAgICAgcGFkKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmNvbnYyZEludGVybmFsKFxuICAgICAgICB4LCB3ZWlnaHRzLCBiaWFzLCBzdHJpZGVIZWlnaHQsIHN0cmlkZVdpZHRoLCBvdXRwdXRJbmZvKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvbnYyZEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhczogQXJyYXkxRHxudWxsLCBzdHJpZGVIZWlnaHQ6IG51bWJlcixcbiAgICAgIHN0cmlkZVdpZHRoOiBudW1iZXIsIG91dHB1dEluZm86IE91dHB1dEluZm8pOiBBcnJheTNEO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgYmFja3Byb3Agb2YgYSAyRCBjb252b2x1dGlvbi5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IGltYWdlLCByYW5rIDMsIG9mIHNoYXBlIFtoZWlnaHQsIHdpZHRoLCBpbkRlcHRoXS5cbiAgICogQHBhcmFtIGR5IFRoZSBkeSBpbWFnZSwgcmFuayAzLCBvZiBzaGFwZSBbaGVpZ2h0LCB3aWR0aCwgb3V0RGVwdGhdLlxuICAgKiBAcGFyYW0gd2VpZ2h0cyBUaGUgd2VpZ2h0cywgcmFuayA0LCBvZiBzaGFwZVxuICAgKiAgICAgW2ZpbHRlckhlaWdodCwgZmlsdGVyV2lkdGgsIGluRGVwdGgsIG91dERlcHRoXS5cbiAgICogQHBhcmFtIHN0cmlkZXMgVGhlIHN0cmlkZXMgb2YgdGhlIGNvbnZvbHV0aW9uOiBbc3RyaWRlSGVpZ2h0LCBzdHJpZGVXaWR0aF0uXG4gICAqIEBwYXJhbSBwYWQgQSBzdHJpbmcgZnJvbTogJ3NhbWUnLCAndmFsaWQnLiBUaGUgdHlwZSBvZiBwYWRkaW5nIGFsZ29yaXRobS5cbiAgICovXG4gIGNvbnYyZEJhY2tQcm9wKFxuICAgICAgeDogQXJyYXkzRCwgZHk6IEFycmF5M0QsIHdlaWdodHM6IEFycmF5NEQsXG4gICAgICBzdHJpZGVzOiBbbnVtYmVyLCBudW1iZXJdfG51bWJlcixcbiAgICAgIHBhZDogJ3ZhbGlkJ3wnc2FtZSd8bnVtYmVyKToge2R4OiBBcnJheTNELCBkdzogQXJyYXk0RCwgZGI6IEFycmF5MUR9IHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkQmFja1Byb3A6IHggbXVzdCBiZSByYW5rIDMsIGJ1dCBnb3Qgc2hhcGUgYCArXG4gICAgICAgICAgICBgJHt4LnNoYXBlfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgZHkucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZEJhY2tQcm9wOiBkeSBtdXN0IGJlIHJhbmsgMywgYnV0IGdvdCBzaGFwZSBgICtcbiAgICAgICAgICAgIGAke2R5LnNoYXBlfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgd2VpZ2h0cy5yYW5rID09PSA0LFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkQmFja1Byb3A6IHdlaWdodHMgbXVzdCBiZSByYW5rIDQsIGJ1dCBnb3Qgc2hhcGUgYCArXG4gICAgICAgICAgICBgJHt3ZWlnaHRzLnNoYXBlfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5zaGFwZVsyXSA9PT0gd2VpZ2h0cy5zaGFwZVsyXSxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZEJhY2tQcm9wOiBkZXB0aCBvZiB4ICR7eC5zaGFwZVsyXX0pIG11c3QgYCArXG4gICAgICAgICAgICBgbWF0Y2ggaW5wdXQgZGVwdGggZm9yIHdlaWdodHMgKCR7d2VpZ2h0cy5zaGFwZVsyXX0uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGR5LnNoYXBlWzJdID09PSB3ZWlnaHRzLnNoYXBlWzNdLFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkQmFja1Byb3A6IGRlcHRoIG9mIGR5ICgke2R5LnNoYXBlWzJdfSkgbXVzdCBgICtcbiAgICAgICAgICAgIGBtYXRjaCBvdXRwdXQgZGVwdGggZm9yIHdlaWdodHMgKCR7d2VpZ2h0cy5zaGFwZVszXX0pLmApO1xuXG4gICAgY29uc3QgZmlsdGVySGVpZ2h0ID0gd2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBmaWx0ZXJXaWR0aCA9IHdlaWdodHMuc2hhcGVbMV07XG4gICAgY29uc3Qgb3V0RGVwdGggPSB3ZWlnaHRzLnNoYXBlWzNdO1xuICAgIGxldCBzdHJpZGVIZWlnaHQ6IG51bWJlcjtcbiAgICBsZXQgc3RyaWRlV2lkdGg6IG51bWJlcjtcbiAgICBpZiAodHlwZW9mIHN0cmlkZXMgPT09ICdudW1iZXInKSB7XG4gICAgICBzdHJpZGVIZWlnaHQgPSBzdHJpZGVzO1xuICAgICAgc3RyaWRlV2lkdGggPSBzdHJpZGVzO1xuICAgIH0gZWxzZSB7XG4gICAgICBzdHJpZGVIZWlnaHQgPSBzdHJpZGVzWzBdO1xuICAgICAgc3RyaWRlV2lkdGggPSBzdHJpZGVzWzFdO1xuICAgIH1cbiAgICBjb25zdCBvdXRwdXRJbmZvID0gY29udl91dGlsLmNvbXB1dGVPdXRwdXRJbmZvKFxuICAgICAgICB4LnNoYXBlLCBmaWx0ZXJIZWlnaHQsIGZpbHRlcldpZHRoLCBvdXREZXB0aCwgc3RyaWRlSGVpZ2h0LCBzdHJpZGVXaWR0aCxcbiAgICAgICAgcGFkKTtcbiAgICBjb25zdCBiYWNrcHJvcFJlc3VsdCA9IHRoaXMuY29udjJkQmFja1Byb3BJbnRlcm5hbChcbiAgICAgICAgeCwgZHksIHdlaWdodHMsIHN0cmlkZUhlaWdodCwgc3RyaWRlV2lkdGgsIG91dHB1dEluZm8pO1xuXG4gICAgdGhpcy50cmFjayhiYWNrcHJvcFJlc3VsdC5kYik7XG4gICAgdGhpcy50cmFjayhiYWNrcHJvcFJlc3VsdC5kdyk7XG4gICAgdGhpcy50cmFjayhiYWNrcHJvcFJlc3VsdC5keCk7XG5cbiAgICByZXR1cm4gYmFja3Byb3BSZXN1bHQ7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvbnYyZEJhY2tQcm9wSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBkeTogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgc3RyaWRlSGVpZ2h0OiBudW1iZXIsXG4gICAgICBzdHJpZGVXaWR0aDogbnVtYmVyLFxuICAgICAgb3V0cHV0SW5mbzogT3V0cHV0SW5mbyk6IHtkeDogQXJyYXkzRCwgZHc6IEFycmF5NEQsIGRiOiBBcnJheTFEfTtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIHRyYW5zcG9zZWQgMkQgY29udm9sdXRpb24gb2YgYW4gaW1hZ2UsIGFsc28ga25vd24gYXMgYVxuICAgKiBkZWNvbnZvbHV0aW9uLlxuICAgKlxuICAgKiBAcGFyYW0geCBUaGUgaW5wdXQgaW1hZ2UsIHJhbmsgMywgb2Ygc2hhcGUgW2hlaWdodCwgd2lkdGgsIGluRGVwdGhdLlxuICAgKiBAcGFyYW0gd2VpZ2h0cyBUaGUgd2VpZ2h0cywgcmFuayA0LCBvZiBzaGFwZVxuICAgKiAgICAgW2ZpbHRlckhlaWdodCwgZmlsdGVyV2lkdGgsIG91dERlcHRoLCBpbkRlcHRoXS5cbiAgICogICAgIGluRGVwdGggbXVzdCBtYXRjaCB4J3MgaW5EZXB0aC5cbiAgICogQHBhcmFtIGJpYXMgT3B0aW9uYWwgYmlhcywgcmFuayAxLCBvZiBzaGFwZSBbb3V0RGVwdGhdLlxuICAgKiBAcGFyYW0gc3RyaWRlcyBUaGUgc3RyaWRlcyBvZiB0aGUgY29udm9sdXRpb246IFtzdHJpZGVIZWlnaHQsIHN0cmlkZVdpZHRoXS5cbiAgICogQHBhcmFtIHBhZCBBIHN0cmluZyBmcm9tOiAnc2FtZScsICd2YWxpZCcuIFRoZSB0eXBlIG9mIHBhZGRpbmcgYWxnb3JpdGhtLlxuICAgKi9cbiAgY29udjJkVHJhbnNwb3NlKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhczogQXJyYXkxRHxudWxsLFxuICAgICAgc3RyaWRlczogW251bWJlciwgbnVtYmVyXXxudW1iZXIsIHBhZDogJ3ZhbGlkJ3wnc2FtZSd8bnVtYmVyKTogQXJyYXkzRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZFRyYW5zcG9zZTogeCBtdXN0IGJlIHJhbmsgMywgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgICAgYCR7eC5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgd2VpZ2h0cy5yYW5rID09PSA0LFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkVHJhbnNwb3NlOiB3ZWlnaHRzIG11c3QgYmUgcmFuayA0LCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYHJhbmsgJHt3ZWlnaHRzLnJhbmt9YCk7XG4gICAgaWYgKGJpYXMgIT0gbnVsbCkge1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgYmlhcy5yYW5rID09PSAxLFxuICAgICAgICAgIGBFcnJvciBpbiBjb252MmRUcmFuc3Bvc2U6IGJpYXMgbXVzdCBiZSByYW5rIDEsIGJ1dCBnb3QgJyArXG4gICAgICAgICAgICAgICdyYW5rICR7Ymlhcy5yYW5rfS5gKTtcbiAgICB9XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHguc2hhcGVbMl0gPT09IHdlaWdodHMuc2hhcGVbM10sXG4gICAgICAgIGBFcnJvciBpbiBjb252MmRUcmFuc3Bvc2U6IGRlcHRoIG9mIGlucHV0ICgke3guc2hhcGVbMl19KSBtdXN0IGAgK1xuICAgICAgICAgICAgYG1hdGNoIGlucHV0IGRlcHRoIGZvciB3ZWlnaHRzICR7d2VpZ2h0cy5zaGFwZVszXX0uYCk7XG5cbiAgICBjb25zdCBmaWx0ZXJIZWlnaHQgPSB3ZWlnaHRzLnNoYXBlWzBdO1xuICAgIGNvbnN0IGZpbHRlcldpZHRoID0gd2VpZ2h0cy5zaGFwZVsxXTtcbiAgICBjb25zdCBvdXREZXB0aCA9IHdlaWdodHMuc2hhcGVbM107XG4gICAgbGV0IHN0cmlkZUhlaWdodDogbnVtYmVyO1xuICAgIGxldCBzdHJpZGVXaWR0aDogbnVtYmVyO1xuICAgIGlmICh0eXBlb2Ygc3RyaWRlcyA9PT0gJ251bWJlcicpIHtcbiAgICAgIHN0cmlkZUhlaWdodCA9IHN0cmlkZXM7XG4gICAgICBzdHJpZGVXaWR0aCA9IHN0cmlkZXM7XG4gICAgfSBlbHNlIHtcbiAgICAgIHN0cmlkZUhlaWdodCA9IHN0cmlkZXNbMF07XG4gICAgICBzdHJpZGVXaWR0aCA9IHN0cmlkZXNbMV07XG4gICAgfVxuICAgIGNvbnN0IG91dHB1dEluZm8gPSBjb252X3V0aWwuY29tcHV0ZU91dHB1dEluZm8oXG4gICAgICAgIHguc2hhcGUsIGZpbHRlckhlaWdodCwgZmlsdGVyV2lkdGgsIG91dERlcHRoLCBzdHJpZGVIZWlnaHQsIHN0cmlkZVdpZHRoLFxuICAgICAgICBwYWQpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuY29udjJkVHJhbnNwb3NlSW50ZXJuYWwoXG4gICAgICAgIHgsIHdlaWdodHMsIGJpYXMsIHN0cmlkZUhlaWdodCwgc3RyaWRlV2lkdGgsIG91dHB1dEluZm8pKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgY29udjJkVHJhbnNwb3NlSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCB3ZWlnaHRzOiBBcnJheTRELCBiaWFzOiBBcnJheTFEfG51bGwsIHN0cmlkZUhlaWdodDogbnVtYmVyLFxuICAgICAgc3RyaWRlV2lkdGg6IG51bWJlciwgb3V0cHV0SW5mbzogT3V0cHV0SW5mbyk6IEFycmF5M0Q7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSAyRCBtYXggcG9vbGluZyBvZiBhbiBpbWFnZS5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IGltYWdlLCBtdXN0IGJlIHJhbmsgMy5cbiAgICogQHBhcmFtIGZTaXplIFRoZSBmaWVsZCBzaXplIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHN0cmlkZSBUaGUgc3RyaWRlIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHBhZCBUaGUgcGFkZGluZyBvZiBlYWNoIHNpZGUgb2YgdGhlIGlucHV0IE5EQXJyYXkuIFdpbGwgcGFkIGVxdWFsbHlcbiAgICogb24gYWxsIHNpZGVzLlxuICAgKi9cbiAgbWF4UG9vbCh4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICAnRXJyb3IgaW4gbWF4UG9vbDogeCBtdXN0IGJlIHJhbmsgMyBidXQgZ290IHJhbmsgJyArIHgucmFuayArICcuJyk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5tYXhQb29sSW50ZXJuYWwoeCwgZlNpemUsIHN0cmlkZSwgcGFkKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG1heFBvb2xJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcik6IEFycmF5M0Q7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBiYWNrcHJvcCBvZiBhIG1heCBwb29sLlxuICAgKiBAcGFyYW0gZHkgVGhlIGR5IGVycm9yLlxuICAgKiBAcGFyYW0geCBUaGUgaW5wdXQgaW1hZ2UsIG11c3QgYmUgcmFuayAzLlxuICAgKiBAcGFyYW0gZlNpemUgVGhlIGZpZWxkIHNpemUgb2YgdGhlIG1heCBwb29sLlxuICAgKiBAcGFyYW0gc3RyaWRlIFRoZSBzdHJpZGUgb2YgdGhlIG1heCBwb29sLlxuICAgKiBAcGFyYW0gcGFkIFRoZSBwYWRkaW5nIG9mIGVhY2ggc2lkZSBvZiB0aGUgaW5wdXQgTkRBcnJheS4gV2lsbCBwYWQgZXF1YWxseVxuICAgKiBvbiBhbGwgc2lkZXMuXG4gICAqL1xuICBtYXhQb29sQmFja3Byb3AoXG4gICAgICBkeTogQXJyYXkzRCwgeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsXG4gICAgICBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBkeS5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gbWF4UG9vbEJhY2twcm9wOiBkeSBtdXN0IGJlIHJhbmsgMyBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgICBgJHtkeS5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gbWF4UG9vbEJhY2twcm9wOiB4IG11c3QgYmUgcmFuayAzIGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICAgIGAke3gucmFua30uYCk7XG5cbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLm1heFBvb2xCYWNrcHJvcEludGVybmFsKGR5LCB4LCBmU2l6ZSwgc3RyaWRlLCBwYWQpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgbWF4UG9vbEJhY2twcm9wSW50ZXJuYWwoXG4gICAgICBkeTogQXJyYXkzRCwgeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsXG4gICAgICBwYWQ6IG51bWJlcik6IEFycmF5M0Q7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSAyRCBtaW4gcG9vbGluZyBvZiBhbiBpbWFnZS5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IGltYWdlLCBtdXN0IGJlIHJhbmsgMy5cbiAgICogQHBhcmFtIGZTaXplIFRoZSBmaWVsZCBzaXplIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHN0cmlkZSBUaGUgc3RyaWRlIG9mIHRoZSBtYXggcG9vbC5cbiAgICogQHBhcmFtIHBhZCBUaGUgcGFkZGluZyBvZiBlYWNoIHNpZGUgb2YgdGhlIGlucHV0IE5EQXJyYXkuIFdpbGwgcGFkIGVxdWFsbHlcbiAgICogb24gYWxsIHNpZGVzLlxuICAgKi9cbiAgbWluUG9vbCh4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gbWluUG9vbDogeCBtdXN0IGJlIHJhbmsgMyBidXQgZ290IHJhbmsgJHt4LnJhbmt9LmApO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMubWluUG9vbEludGVybmFsKHgsIGZTaXplLCBzdHJpZGUsIHBhZCkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBtaW5Qb29sSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgMkQgYXZlcmFnZSBwb29saW5nIG9mIGFuIGltYWdlLlxuICAgKiBAcGFyYW0geCBUaGUgaW5wdXQgaW1hZ2UsIG11c3QgYmUgcmFuayAzLlxuICAgKiBAcGFyYW0gZlNpemUgVGhlIGZpZWxkIHNpemUgb2YgdGhlIG1heCBwb29sLlxuICAgKiBAcGFyYW0gc3RyaWRlIFRoZSBzdHJpZGUgb2YgdGhlIG1heCBwb29sLlxuICAgKiBAcGFyYW0gcGFkIFRoZSBwYWRkaW5nIG9mIGVhY2ggc2lkZSBvZiB0aGUgaW5wdXQgTkRBcnJheS4gV2lsbCBwYWQgZXF1YWxseVxuICAgKiBvbiBhbGwgc2lkZXMuXG4gICAqL1xuICBhdmdQb29sKHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnJhbmsgPT09IDMsXG4gICAgICAgIGBFcnJvciBpbiBhdmdQb29sOiB4IG11c3QgYmUgcmFuayAzIGJ1dCBnb3QgcmFuayAke3gucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5hdmdQb29sSW50ZXJuYWwoeCwgZlNpemUsIHN0cmlkZSwgcGFkKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGF2Z1Bvb2xJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcik6IEFycmF5M0Q7XG5cbiAgLypcbiAgICogQmlsaW5lYXIgcmVzaXplIGEgM0QgYXJyYXkgcGVyIGVhY2ggY2hhbm5lbCB0byBhIG5ldyAyRCBzaGFwZS5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IEFycmF5M0QuXG4gICAqIEBwYXJhbSBuZXdTaGFwZTJEIFRoZSBuZXcgc2hhcGUgdG8gcmVzaXplIHRoZSBBcnJheTNEIHRvLiBFYWNoIGNoYW5uZWwgaXNcbiAgICogcmVzaXplZCBpbmRpdmlkdWFsbHkuXG4gICAqIEBwYXJhbSBhbGlnbkNvcm5lcnMgQW4gb3B0aW9uYWwgYm9vbC4gRGVmYXVsdHMgdG8gRmFsc2UuIElmIHRydWUsIHJlc2NhbGVcbiAgICogaW5wdXQgYnkgKG5ld19oZWlnaHQgLSAxKSAvIChoZWlnaHQgLSAxKSwgd2hpY2ggZXhhY3RseSBhbGlnbnMgdGhlIDRcbiAgICogY29ybmVycyBvZiBpbWFnZXMgYW5kIHJlc2l6ZWQgaW1hZ2VzLiBJZiBmYWxzZSwgcmVzY2FsZSBieSBuZXdfaGVpZ2h0IC9cbiAgICogaGVpZ2h0LiBUcmVhdCBzaW1pbGFybHkgdGhlIHdpZHRoIGRpbWVuc2lvbi5cbiAgICovXG4gIHJlc2l6ZUJpbGluZWFyM0QoXG4gICAgICB4OiBBcnJheTNELCBuZXdTaGFwZTJEOiBbbnVtYmVyLCBudW1iZXJdLCBhbGlnbkNvcm5lcnMgPSBmYWxzZSk6IEFycmF5M0Qge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnJhbmsgPT09IDMsXG4gICAgICAgIGBFcnJvciBpbiByZXNpemVCaWxpbmVhcjNEOiB4IG11c3QgYmUgcmFuayAzIGJ1dCBnb3QgcmFuayAke3gucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIG5ld1NoYXBlMkQubGVuZ3RoID09PSAyLFxuICAgICAgICBgRXJyb3IgaW4gcmVzaXplQmlsaW5lYXIzRDogbmV3IHNoYXBlIG11c3QgMkQsIGJ1dCBnb3Qgc2hhcGUgYCArXG4gICAgICAgICAgICBgJHtuZXdTaGFwZTJEfS5gKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayhcbiAgICAgICAgdGhpcy5yZXNpemVCaWxpbmVhcjNESW50ZXJuYWwoeCwgbmV3U2hhcGUyRCwgYWxpZ25Db3JuZXJzKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHJlc2l6ZUJpbGluZWFyM0RJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIG5ld1NoYXBlMkQ6IFtudW1iZXIsIG51bWJlcl0sIGFsaWduQ29ybmVyczogYm9vbGVhbik6IEFycmF5M0Q7XG5cbiAgLyoqXG4gICAqIEJhdGNoIG5vcm1hbGl6YXRpb24gM0QuIE1lYW4sIHZhcmlhbmNlLCBzY2FsZSwgYW5kIG9mZnNldCBjYW4gYmUgb2YgdHdvXG4gICAqIHNoYXBlczogMSkgVGhlIHNhbWUgc2hhcGUgYXMgdGhlIGlucHV0OiBhbiBBcnJheTNELiAyKSBJbiB0aGUgY29tbW9uIGNhc2UsXG4gICAqIHRoZSBkZXB0aCBkaW1lbnNpb24gaXMgdGhlIGxhc3QgZGltZW5zaW9uIG9mIHgsIHNvIHRoZSB2YWx1ZXMgd291bGQgYmUgYW5cbiAgICogQXJyYXkxRCBvZiBzaGFwZSBbZGVwdGhdLlxuICAgKiBAcGFyYW0geCBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICogQHBhcmFtIG1lYW4gQSBtZWFuIE5EQXJyYXkuXG4gICAqIEBwYXJhbSB2YXJpYW5jZSBBIHZhcmlhbmNlIE5EQXJyYXkuXG4gICAqIEBwYXJhbSB2YXJpYW5jZUVwc2lsb24gQSBzbWFsbCBmbG9hdCBudW1iZXIgdG8gYXZvaWQgZGl2aWRpbmcgYnkgMC5cbiAgICogQHBhcmFtIHNjYWxlIEEgc2NhbGUgTkRBcnJheS5cbiAgICogQHBhcmFtIG9mZnNldCBBbiBvZmZzZXQgTkRBcnJheS5cbiAgICovXG4gIGJhdGNoTm9ybWFsaXphdGlvbjNEKFxuICAgICAgeDogQXJyYXkzRCwgbWVhbjogQXJyYXkzRHxBcnJheTFELCB2YXJpYW5jZTogQXJyYXkzRHxBcnJheTFELFxuICAgICAgdmFyaWFuY2VFcHNpbG9uID0gLjAwMSwgc2NhbGU/OiBBcnJheTNEfEFycmF5MUQsXG4gICAgICBvZmZzZXQ/OiBBcnJheTNEfEFycmF5MUQpOiBBcnJheTNEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gYmF0Y2hOb3JtYWxpemF0aW9uM0Q6IHggbXVzdCBiZSByYW5rIDMgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgICAgYCR7eC5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgbWVhbi5yYW5rID09PSAzIHx8IG1lYW4ucmFuayA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIGJhdGNoTm9ybWFsaXphdGlvbjNEOiBtZWFuIG11c3QgYmUgcmFuayAzIG9yIHJhbmsgMSBidXQgYCArXG4gICAgICAgICAgICBgZ290IHJhbmsgJHttZWFuLnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB2YXJpYW5jZS5yYW5rID09PSAzIHx8IHZhcmlhbmNlLnJhbmsgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBiYXRjaE5vcm1hbGl6YXRpb24zRDogdmFyaWFuY2UgbXVzdCBiZSByYW5rIDMgb3IgcmFuayAxIGAgK1xuICAgICAgICAgICAgYGJ1dCBnb3QgcmFuayAke3ZhcmlhbmNlLnJhbmt9LmApO1xuICAgIGlmIChzY2FsZSAhPSBudWxsKSB7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICBzY2FsZS5yYW5rID09PSAzIHx8IHNjYWxlLnJhbmsgPT09IDEsXG4gICAgICAgICAgYEVycm9yIGluIGJhdGNoTm9ybWFsaXphdGlvbjNEOiBzY2FsZSBtdXN0IGJlIHJhbmsgMyBvciByYW5rIDEgYCArXG4gICAgICAgICAgICAgIGBidXQgZ290IHJhbmsgJHtzY2FsZS5yYW5rfS5gKTtcbiAgICB9XG4gICAgaWYgKG9mZnNldCAhPSBudWxsKSB7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICBvZmZzZXQucmFuayA9PT0gMyB8fCBvZmZzZXQucmFuayA9PT0gMSxcbiAgICAgICAgICBgRXJyb3IgaW4gYmF0Y2hOb3JtYWxpemF0aW9uM0Q6IG9mZnNldCBtdXN0IGJlIHJhbmsgMyBvciByYW5rIDEgYCArXG4gICAgICAgICAgICAgIGBidXQgZ290IHJhbmsgJHtvZmZzZXQucmFua30uYCk7XG4gICAgfVxuXG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5iYXRjaE5vcm1hbGl6YXRpb24zREludGVybmFsKFxuICAgICAgICB4LCBtZWFuLCB2YXJpYW5jZSwgdmFyaWFuY2VFcHNpbG9uLCBzY2FsZSwgb2Zmc2V0KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGJhdGNoTm9ybWFsaXphdGlvbjNESW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBtZWFuOiBBcnJheTNEfEFycmF5MUQsIHZhcmlhbmNlOiBBcnJheTNEfEFycmF5MUQsXG4gICAgICB2YXJpYW5jZUVwc2lsb246IG51bWJlciwgc2NhbGU/OiBBcnJheTNEfEFycmF5MUQsXG4gICAgICBvZmZzZXQ/OiBBcnJheTNEfEFycmF5MUQpOiBBcnJheTNEO1xuXG4gIC8vLy8vLy8vLy8vLy8vXG4gIC8vIExTVE0gb3BzIC8vXG4gIC8vLy8vLy8vLy8vLy8vXG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBuZXh0IHN0YXRlcyBhbmQgb3V0cHV0cyBvZiBhIHN0YWNrIG9mIExTVE1DZWxscy5cbiAgICogRWFjaCBjZWxsIG91dHB1dCBpcyB1c2VkIGFzIGlucHV0IHRvIHRoZSBuZXh0IGNlbGwuXG4gICAqIFRoaXMgaXMgb25seSB0aGUgZm9yd2FyZCBtb2RlLlxuICAgKiBEZXJpdmVkIGZyb20gdGYuY29udHJpYi5ybi5NdWx0aVJOTkNlbGwuXG4gICAqIEBwYXJhbSBsc3RtQ2VsbHMgQXJyYXkgb2YgTFNUTUNlbGwgZnVuY3Rpb25zLlxuICAgKiBAcGFyYW0gZGF0YSBUaGUgaW5wdXQgdG8gdGhlIGNlbGwuXG4gICAqIEBwYXJhbSBjIEFycmF5IG9mIHByZXZpb3VzIGNlbGwgc3RhdGVzLlxuICAgKiBAcGFyYW0gaCBBcnJheSBvZiBwcmV2aW91cyBjZWxsIG91dHB1dHMuXG4gICAqIEByZXR1cm4gVHVwbGUgW25leHRDZWxsU3RhdGVzLCBjZWxsT3V0cHV0c11cbiAgICovXG4gIG11bHRpUk5OQ2VsbChcbiAgICAgIGxzdG1DZWxsczogTFNUTUNlbGxbXSwgZGF0YTogQXJyYXkyRCwgYzogQXJyYXkyRFtdLFxuICAgICAgaDogQXJyYXkyRFtdKTogW0FycmF5MkRbXSwgQXJyYXkyRFtdXSB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGRhdGEuc2hhcGVbMF0gPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBtdWx0aVJOTkNlbGw6IGZpcnN0IGRpbWVuc2lvbiBvZiBkYXRhIGlzICR7ZGF0YS5zaGFwZVswXX0sIGAgK1xuICAgICAgICAgICAgYGJ1dCBiYXRjaCBzaXplcyA+IDEgYXJlIG5vdCB5ZXQgc3VwcG9ydGVkLmApO1xuICAgIGNvbnN0IHJlcyA9IHRoaXMuc2NvcGUoKCkgPT4ge1xuICAgICAgbGV0IGlucHV0ID0gZGF0YTtcbiAgICAgIGNvbnN0IG5ld1N0YXRlcyA9IFtdO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsc3RtQ2VsbHMubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgY29uc3Qgb3V0cHV0ID0gbHN0bUNlbGxzW2ldKGlucHV0LCBjW2ldLCBoW2ldKTtcbiAgICAgICAgbmV3U3RhdGVzLnB1c2gob3V0cHV0WzBdKTtcbiAgICAgICAgbmV3U3RhdGVzLnB1c2gob3V0cHV0WzFdKTtcbiAgICAgICAgaW5wdXQgPSBvdXRwdXRbMV07XG4gICAgICB9XG5cbiAgICAgIHJldHVybiBuZXdTdGF0ZXM7XG4gICAgfSk7XG4gICAgY29uc3QgbmV3QzogQXJyYXkyRFtdID0gW107XG4gICAgY29uc3QgbmV3SDogQXJyYXkyRFtdID0gW107XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCByZXMubGVuZ3RoOyBpICs9IDIpIHtcbiAgICAgIG5ld0MucHVzaChyZXNbaV0gYXMgQXJyYXkyRCk7XG4gICAgICBuZXdILnB1c2gocmVzW2kgKyAxXSBhcyBBcnJheTJEKTtcbiAgICB9XG4gICAgcmV0dXJuIFtuZXdDLCBuZXdIXTtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgbmV4dCBzdGF0ZSBhbmQgb3V0cHV0IG9mIGEgQmFzaWNMU1RNQ2VsbC5cbiAgICogVGhpcyBpcyBvbmx5IHRoZSBmb3J3YXJkIG1vZGUuXG4gICAqIERlcml2ZWQgZnJvbSB0Zi5jb250cmliLnJubi5CYXNpY0xTVE1DZWxsLlxuICAgKiBAcGFyYW0gZm9yZ2V0QmlhcyBGb3JnZXQgYmlhcyBmb3IgdGhlIGNlbGwuXG4gICAqIEBwYXJhbSBsc3RtS2VybmVsIFRoZSB3ZWlnaHRzIGZvciB0aGUgY2VsbC5cbiAgICogQHBhcmFtIGxzdG1CaWFzIFRoZSBiaWFzIGZvciB0aGUgY2VsbC5cbiAgICogQHBhcmFtIGRhdGEgVGhlIGlucHV0IHRvIHRoZSBjZWxsLlxuICAgKiBAcGFyYW0gYyBQcmV2aW91cyBjZWxsIHN0YXRlLlxuICAgKiBAcGFyYW0gaCBQcmV2aW91cyBjZWxsIG91dHB1dC5cbiAgICogQHJldHVybiBUdXBsZSBbbmV4dENlbGxTdGF0ZSwgY2VsbE91dHB1dF1cbiAgICovXG4gIGJhc2ljTFNUTUNlbGwoXG4gICAgICBmb3JnZXRCaWFzOiBTY2FsYXIsIGxzdG1LZXJuZWw6IEFycmF5MkQsIGxzdG1CaWFzOiBBcnJheTFELCBkYXRhOiBBcnJheTJELFxuICAgICAgYzogQXJyYXkyRCwgaDogQXJyYXkyRCk6IFtBcnJheTJELCBBcnJheTJEXSB7XG4gICAgY29uc3QgcmVzID0gdGhpcy5zY29wZSgoKSA9PiB7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICBkYXRhLnNoYXBlWzBdID09PSAxLFxuICAgICAgICAgIGBFcnJvciBpbiBtdWx0aVJOTkNlbGw6IGZpcnN0IGRpbWVuc2lvbiBvZiBkYXRhIGlzIGAgK1xuICAgICAgICAgICAgICBgJHtkYXRhLnNoYXBlWzBdfSwgYnV0IGJhdGNoIHNpemVzID4gMSBhcmUgbm90IHlldCBzdXBwb3J0ZWQuYCk7XG4gICAgICAvLyBjb25jYXQoaW5wdXRzLCBoLCAxKVxuICAgICAgLy8gVGhlcmUgaXMgbm8gY29uY2F0MWQsIHNvIHJlc2hhcGUgaW5wdXRzIGFuZCBoIHRvIDNkLCBjb25jYXQsIHRoZW5cbiAgICAgIC8vIHJlc2hhcGUgYmFjayB0byAyZC5cbiAgICAgIGNvbnN0IGRhdGEzRCA9IGRhdGEuYXMzRCgxLCAxLCBkYXRhLnNoYXBlWzFdKTtcbiAgICAgIGNvbnN0IGgzRCA9IGguYXMzRCgxLCAxLCBoLnNoYXBlWzFdKTtcbiAgICAgIGNvbnN0IGNvbWJpbmVkM0QgPSB0aGlzLmNvbmNhdDNEKGRhdGEzRCwgaDNELCAyKTtcbiAgICAgIGNvbnN0IGNvbWJpbmVkMkQgPSBjb21iaW5lZDNELmFzMkQoMSwgZGF0YS5zaGFwZVsxXSArIGguc2hhcGVbMV0pO1xuXG4gICAgICBjb25zdCB3ZWlnaHRlZCA9IHRoaXMubWF0TXVsKGNvbWJpbmVkMkQsIGxzdG1LZXJuZWwpO1xuICAgICAgY29uc3QgcmVzID0gdGhpcy5hZGQod2VpZ2h0ZWQsIGxzdG1CaWFzKSBhcyBBcnJheTJEO1xuXG4gICAgICAvLyBpID0gaW5wdXRfZ2F0ZSwgaiA9IG5ld19pbnB1dCwgZiA9IGZvcmdldF9nYXRlLCBvID0gb3V0cHV0X2dhdGVcbiAgICAgIGNvbnN0IGkgPSB0aGlzLnNsaWNlMkQocmVzLCBbMCwgMF0sIFtyZXMuc2hhcGVbMF0sIHJlcy5zaGFwZVsxXSAvIDRdKTtcbiAgICAgIGNvbnN0IGogPSB0aGlzLnNsaWNlMkQoXG4gICAgICAgICAgcmVzLCBbMCwgcmVzLnNoYXBlWzFdIC8gNCAqIDFdLCBbcmVzLnNoYXBlWzBdLCByZXMuc2hhcGVbMV0gLyA0XSk7XG4gICAgICBjb25zdCBmID0gdGhpcy5zbGljZTJEKFxuICAgICAgICAgIHJlcywgWzAsIHJlcy5zaGFwZVsxXSAvIDQgKiAyXSwgW3Jlcy5zaGFwZVswXSwgcmVzLnNoYXBlWzFdIC8gNF0pO1xuICAgICAgY29uc3QgbyA9IHRoaXMuc2xpY2UyRChcbiAgICAgICAgICByZXMsIFswLCByZXMuc2hhcGVbMV0gLyA0ICogM10sIFtyZXMuc2hhcGVbMF0sIHJlcy5zaGFwZVsxXSAvIDRdKTtcblxuICAgICAgY29uc3QgbmV3QyA9XG4gICAgICAgICAgdGhpcy5hZGQoXG4gICAgICAgICAgICAgIHRoaXMubXVsdGlwbHlTdHJpY3QoXG4gICAgICAgICAgICAgICAgICBjLCB0aGlzLnNpZ21vaWQodGhpcy5zY2FsYXJQbHVzQXJyYXkoZm9yZ2V0QmlhcywgZikpKSxcbiAgICAgICAgICAgICAgdGhpcy5tdWx0aXBseVN0cmljdCh0aGlzLnNpZ21vaWQoaSksIHRoaXMudGFuaChqKSkpIGFzIEFycmF5MkQ7XG4gICAgICBjb25zdCBuZXdIID1cbiAgICAgICAgICB0aGlzLm11bHRpcGx5U3RyaWN0KHRoaXMudGFuaChuZXdDKSwgdGhpcy5zaWdtb2lkKG8pKSBhcyBBcnJheTJEO1xuXG4gICAgICByZXR1cm4gW25ld0MsIG5ld0hdO1xuICAgIH0pO1xuICAgIHJldHVybiBbcmVzWzBdLCByZXNbMV1dO1xuICB9XG59XG5cbmV4cG9ydCBlbnVtIE1hdHJpeE9yaWVudGF0aW9uIHtcbiAgUkVHVUxBUixcbiAgVFJBTlNQT1NFRFxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQgKiBhcyBjb25jYXQzZF91dGlsIGZyb20gJy4vY29uY2F0M2RfdXRpbCc7XG5pbXBvcnQgKiBhcyBjb252X3V0aWwgZnJvbSAnLi9jb252X3V0aWwnO1xuaW1wb3J0IHtPdXRwdXRJbmZvfSBmcm9tICcuL2NvbnZfdXRpbCc7XG5pbXBvcnQgKiBhcyBjb3B5MkRfdXRpbCBmcm9tICcuL2NvcHkyZF91dGlsJztcbmltcG9ydCB7TWF0cml4T3JpZW50YXRpb24sIE5EQXJyYXlNYXRofSBmcm9tICcuL21hdGgnO1xuaW1wb3J0IHtBcnJheTFELCBBcnJheTJELCBBcnJheTNELCBBcnJheTRELCBOREFycmF5LCBTY2FsYXJ9IGZyb20gJy4vbmRhcnJheSc7XG5cbmV4cG9ydCBjbGFzcyBOREFycmF5TWF0aENQVSBleHRlbmRzIE5EQXJyYXlNYXRoIHtcbiAgY29uc3RydWN0b3Ioc2FmZU1vZGUgPSBmYWxzZSkge1xuICAgIHN1cGVyKHNhZmVNb2RlKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBjbG9uZUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihcbiAgICAgICAgbmRhcnJheS5zaGFwZSwge3ZhbHVlczogbmV3IEZsb2F0MzJBcnJheShuZGFycmF5LmdldFZhbHVlcygpKX0pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHNsaWNlMkRJbnRlcm5hbChcbiAgICAgIGlucHV0OiBBcnJheTJELCBiZWdpblJvd0NvbDogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHNpemVSb3dDb2w6IFtudW1iZXIsIG51bWJlcl0pOiBBcnJheTJEIHtcbiAgICBjb25zdCByZXN1bHQgPSBBcnJheTJELnplcm9zKHNpemVSb3dDb2wpO1xuICAgIHRoaXMuY29weTJESW50ZXJuYWwoXG4gICAgICAgIGlucHV0LCBiZWdpblJvd0NvbCwgc2l6ZVJvd0NvbCwgcmVzdWx0LCBbMCwgMF0sIHNpemVSb3dDb2wpO1xuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICBwcm90ZWN0ZWQgY29weTJESW50ZXJuYWwoXG4gICAgICBzb3VyY2U6IEFycmF5MkQsIHNvdXJjZUJlZ2luUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgc291cmNlU2l6ZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSwgZGVzdDogQXJyYXkyRCxcbiAgICAgIGRlc3RCZWdpblJvd0NvbDogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIGRlc3RTaXplUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdKTogdm9pZCB7XG4gICAgY29weTJEX3V0aWwudmFsaWRhdGVTaGFwZXMoc291cmNlU2l6ZVJvd0NvbCwgZGVzdFNpemVSb3dDb2wpO1xuICAgIGNvbnN0IHNyY1ZhbHVlcyA9IHNvdXJjZS5nZXRWYWx1ZXMoKTtcbiAgICBjb25zdCBkc3RWYWx1ZXMgPSBkZXN0LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IG4gPSBzb3VyY2VTaXplUm93Q29sWzBdICogc291cmNlU2l6ZVJvd0NvbFsxXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG47ICsraSkge1xuICAgICAgY29uc3Qgc3JjUm93ID0gc291cmNlQmVnaW5Sb3dDb2xbMF0gKyBNYXRoLmZsb29yKGkgLyBzb3VyY2VTaXplUm93Q29sWzFdKTtcbiAgICAgIGNvbnN0IHNyY0NvbCA9IHNvdXJjZUJlZ2luUm93Q29sWzFdICsgKGkgJSBzb3VyY2VTaXplUm93Q29sWzFdKTtcbiAgICAgIGNvbnN0IHNyY09mZiA9IHNyY1JvdyAqIHNvdXJjZS5zaGFwZVsxXSArIHNyY0NvbDtcbiAgICAgIGNvbnN0IGRzdFJvdyA9IGRlc3RCZWdpblJvd0NvbFswXSArIE1hdGguZmxvb3IoaSAvIGRlc3RTaXplUm93Q29sWzFdKTtcbiAgICAgIGNvbnN0IGRzdENvbCA9IGRlc3RCZWdpblJvd0NvbFsxXSArIChpICUgZGVzdFNpemVSb3dDb2xbMV0pO1xuICAgICAgY29uc3QgZHN0T2ZmID0gZHN0Um93ICogZGVzdC5zaGFwZVsxXSArIGRzdENvbDtcbiAgICAgIGRzdFZhbHVlc1tkc3RPZmZdID0gc3JjVmFsdWVzW3NyY09mZl07XG4gICAgfVxuICB9XG5cbiAgcHJvdGVjdGVkIGNvbmNhdDNESW50ZXJuYWwoeDE6IEFycmF5M0QsIHgyOiBBcnJheTNELCBheGlzOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICBjb25zdCBvdXRwdXRTaGFwZSA9XG4gICAgICAgIGNvbmNhdDNkX3V0aWwuY29tcHV0ZUNvbmNhdDNET3V0cHV0U2hhcGUoeDEuc2hhcGUsIHgyLnNoYXBlLCBheGlzKTtcblxuICAgIGNvbnN0IHZhbHVlcyA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBvdXRwdXRTaGFwZVswXTsgaSsrKSB7XG4gICAgICBmb3IgKGxldCBqID0gMDsgaiA8IG91dHB1dFNoYXBlWzFdOyBqKyspIHtcbiAgICAgICAgZm9yIChsZXQgayA9IDA7IGsgPCBvdXRwdXRTaGFwZVsyXTsgaysrKSB7XG4gICAgICAgICAgLy8gU2hhZGVyIGJlZ2lucy5cbiAgICAgICAgICBjb25zdCBpbmRleDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gW2ksIGosIGtdO1xuICAgICAgICAgIGxldCB2YWx1ZTogbnVtYmVyO1xuICAgICAgICAgIGlmIChpbmRleFtheGlzXSA8IHgxLnNoYXBlW2F4aXNdKSB7XG4gICAgICAgICAgICB2YWx1ZSA9IHgxLmdldChpLCBqLCBrKTtcbiAgICAgICAgICB9IGVsc2Uge1xuICAgICAgICAgICAgaW5kZXhbYXhpc10gLT0geDEuc2hhcGVbYXhpc107XG4gICAgICAgICAgICBjb25zdCBbaTIsIGoyLCBrMl0gPSBpbmRleDtcbiAgICAgICAgICAgIHZhbHVlID0geDIuZ2V0KGkyLCBqMiwgazIpO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIHZhbHVlcy5zZXQodmFsdWUsIGksIGosIGspO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHZhbHVlcztcbiAgfVxuXG4gIHByb3RlY3RlZCBzY2FsZWRBcnJheUFkZEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIGMxOiBTY2FsYXIsIGE6IFQsIGMyOiBTY2FsYXIsIGI6IFQpIHtcbiAgICBjb25zdCBuZXdTaGFwZSA9IHV0aWwuYXNzZXJ0QW5kR2V0QnJvYWRjYXN0ZWRTaGFwZShhLnNoYXBlLCBiLnNoYXBlKTtcbiAgICBjb25zdCBuZXdWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHV0aWwuc2l6ZUZyb21TaGFwZShuZXdTaGFwZSkpO1xuXG4gICAgY29uc3QgYVZhbHVlcyA9IGEuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgYlZhbHVlcyA9IGIuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgYzFWYWwgPSBjMS5nZXQoKTtcbiAgICBjb25zdCBjMlZhbCA9IGMyLmdldCgpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbmV3VmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBuZXdWYWx1ZXNbaV0gPSBjMVZhbCAqIGFWYWx1ZXNbaSAlIGEuc2l6ZV0gKyBjMlZhbCAqIGJWYWx1ZXNbaSAlIGIuc2l6ZV07XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmV3U2hhcGUsIHt2YWx1ZXM6IG5ld1ZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIG5lZ0ludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMuc2NhbGFyVGltZXNBcnJheShTY2FsYXIuTkVHX09ORSwgYSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgYWRkSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy5zY2FsZWRBcnJheUFkZEludGVybmFsPFQ+KFNjYWxhci5PTkUsIGEsIFNjYWxhci5PTkUsIGIpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHN1YkludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMuc2NhbGVkQXJyYXlBZGRJbnRlcm5hbDxUPihTY2FsYXIuT05FLCBhLCBTY2FsYXIuTkVHX09ORSwgYik7XG4gIH1cblxuICBwcm90ZWN0ZWQgbWF0TXVsSW50ZXJuYWwoXG4gICAgICBhOiBBcnJheTJELCBiOiBBcnJheTJELCBhT3JpZW50YXRpb24gPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSLFxuICAgICAgYk9yaWVudGF0aW9uID0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUik6IEFycmF5MkQge1xuICAgIGNvbnN0IHNoYXJlZERpbSA9XG4gICAgICAgIChhT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID8gYS5zaGFwZVsxXSA6IGEuc2hhcGVbMF07XG5cbiAgICBjb25zdCBsZWZ0RGltID1cbiAgICAgICAgKGFPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyBhLnNoYXBlWzBdIDogYS5zaGFwZVsxXTtcbiAgICBjb25zdCByaWdodERpbSA9XG4gICAgICAgIChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID8gYi5zaGFwZVsxXSA6IGIuc2hhcGVbMF07XG5cbiAgICBjb25zdCBub3JtYWxHZXR0ZXIgPSAobWF0cml4OiBBcnJheTJELCBpOiBudW1iZXIsIGo6IG51bWJlcikgPT5cbiAgICAgICAgbWF0cml4LmdldChpLCBqKTtcbiAgICBjb25zdCB0cmFuc3Bvc2VkR2V0dGVyID0gKG1hdHJpeDogQXJyYXkyRCwgaTogbnVtYmVyLCBqOiBudW1iZXIpID0+XG4gICAgICAgIG1hdHJpeC5nZXQoaiwgaSk7XG5cbiAgICBjb25zdCBhR2V0dGVyID0gKGFPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgP1xuICAgICAgICBub3JtYWxHZXR0ZXIgOlxuICAgICAgICB0cmFuc3Bvc2VkR2V0dGVyO1xuICAgIGNvbnN0IGJHZXR0ZXIgPSAoYk9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICAgIG5vcm1hbEdldHRlciA6XG4gICAgICAgIHRyYW5zcG9zZWRHZXR0ZXI7XG4gICAgY29uc3QgdmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShsZWZ0RGltICogcmlnaHREaW0pO1xuICAgIGxldCBpbmRleCA9IDA7XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGxlZnREaW07ICsraSkge1xuICAgICAgZm9yIChsZXQgaiA9IDA7IGogPCByaWdodERpbTsgKytqKSB7XG4gICAgICAgIGxldCBzdW0gPSAwO1xuICAgICAgICBmb3IgKGxldCBrID0gMDsgayA8IHNoYXJlZERpbTsgKytrKSB7XG4gICAgICAgICAgLy8gVE9ETzogb3B0aW1pemUgQ1BVIG1hdG11bC5cbiAgICAgICAgICBzdW0gKz0gYUdldHRlcihhLCBpLCBrKSAqIGJHZXR0ZXIoYiwgaywgaik7XG4gICAgICAgIH1cbiAgICAgICAgdmFsdWVzW2luZGV4KytdID0gc3VtO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gQXJyYXkyRC5uZXcoW2xlZnREaW0sIHJpZ2h0RGltXSwgdmFsdWVzKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBtdWx0aXBseUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgY29uc3QgbmV3U2hhcGUgPSB1dGlsLmFzc2VydEFuZEdldEJyb2FkY2FzdGVkU2hhcGUoYS5zaGFwZSwgYi5zaGFwZSk7XG4gICAgY29uc3QgbmV3VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheSh1dGlsLnNpemVGcm9tU2hhcGUobmV3U2hhcGUpKTtcblxuICAgIGNvbnN0IGFWYWx1ZXMgPSBhLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IGJWYWx1ZXMgPSBiLmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbmV3VmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBuZXdWYWx1ZXNbaV0gPSBhVmFsdWVzW2kgJSBhLnNpemVdICogYlZhbHVlc1tpICUgYi5zaXplXTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihuZXdTaGFwZSwge3ZhbHVlczogbmV3VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgZGl2aWRlSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICBjb25zdCBuZXdTaGFwZSA9IHV0aWwuYXNzZXJ0QW5kR2V0QnJvYWRjYXN0ZWRTaGFwZShhLnNoYXBlLCBiLnNoYXBlKTtcbiAgICBjb25zdCBuZXdWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHV0aWwuc2l6ZUZyb21TaGFwZShuZXdTaGFwZSkpO1xuXG4gICAgY29uc3QgYVZhbHVlcyA9IGEuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgYlZhbHVlcyA9IGIuZ2V0VmFsdWVzKCk7XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG5ld1ZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgbmV3VmFsdWVzW2ldID0gYVZhbHVlc1tpICUgYS5zaXplXSAvIGJWYWx1ZXNbaSAlIGIuc2l6ZV07XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmV3U2hhcGUsIHt2YWx1ZXM6IG5ld1ZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHN1bUludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGxldCBzdW0gPSAwO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIHN1bSArPSB2YWx1ZXNbaV07XG4gICAgfVxuICAgIHJldHVybiBTY2FsYXIubmV3KHN1bSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgYXJnTWluSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgbGV0IG1pbiA9IE51bWJlci5NQVhfVkFMVUU7XG4gICAgbGV0IG1pbkluZGV4ID0gLTE7XG4gICAgY29uc3QgdmFsdWVzID0gbmRhcnJheS5nZXRWYWx1ZXMoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3QgdmFsdWUgPSB2YWx1ZXNbaV07XG4gICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgICB9XG4gICAgICBpZiAodmFsdWUgPCBtaW4pIHtcbiAgICAgICAgbWluID0gdmFsdWU7XG4gICAgICAgIG1pbkluZGV4ID0gaTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIFNjYWxhci5uZXcobWluSW5kZXgpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGFyZ01heEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGxldCBtYXggPSBOdW1iZXIuTkVHQVRJVkVfSU5GSU5JVFk7XG4gICAgbGV0IG1heEluZGV4ID0gLTE7XG4gICAgY29uc3QgdmFsdWVzID0gbmRhcnJheS5nZXRWYWx1ZXMoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3QgdmFsdWUgPSB2YWx1ZXNbaV07XG4gICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgICB9XG4gICAgICBpZiAodmFsdWUgPiBtYXgpIHtcbiAgICAgICAgbWF4ID0gdmFsdWU7XG4gICAgICAgIG1heEluZGV4ID0gaTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIFNjYWxhci5uZXcobWF4SW5kZXgpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGFyZ01heEVxdWFsc0ludGVybmFsKHgxOiBOREFycmF5LCB4MjogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgY29uc3QgYXJnTWF4MSA9IHRoaXMuYXJnTWF4SW50ZXJuYWwoeDEpLmdldCgpO1xuICAgIGNvbnN0IGFyZ01heDIgPSB0aGlzLmFyZ01heEludGVybmFsKHgyKS5nZXQoKTtcbiAgICBpZiAoaXNOYU4oYXJnTWF4MSkgfHwgaXNOYU4oYXJnTWF4MikpIHtcbiAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgfVxuICAgIHJldHVybiBTY2FsYXIubmV3KCsoYXJnTWF4MSA9PT0gYXJnTWF4MikpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHRvcEtJbnRlcm5hbChuZGFycmF5OiBOREFycmF5LCBrOiBudW1iZXIpOlxuICAgICAge3ZhbHVlczogQXJyYXkxRCwgaW5kaWNlczogQXJyYXkxRH0ge1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgdmFsdWVzQW5kSW5kaWNlczogQXJyYXk8e3ZhbHVlOiBudW1iZXIsIGluZGV4OiBudW1iZXJ9PiA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICB2YWx1ZXNBbmRJbmRpY2VzLnB1c2goe3ZhbHVlOiB2YWx1ZXNbaV0sIGluZGV4OiBpfSk7XG4gICAgfVxuICAgIHZhbHVlc0FuZEluZGljZXMuc29ydCgoYSwgYikgPT4ge1xuICAgICAgcmV0dXJuIGIudmFsdWUgLSBhLnZhbHVlO1xuICAgIH0pO1xuICAgIGNvbnN0IHRvcGtWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KGspO1xuICAgIGNvbnN0IHRvcGtJbmRpY2VzID0gbmV3IEZsb2F0MzJBcnJheShrKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGs7IGkrKykge1xuICAgICAgdG9wa1ZhbHVlc1tpXSA9IHZhbHVlc0FuZEluZGljZXNbaV0udmFsdWU7XG4gICAgICB0b3BrSW5kaWNlc1tpXSA9IHZhbHVlc0FuZEluZGljZXNbaV0uaW5kZXg7XG4gICAgfVxuICAgIHJldHVybiB7dmFsdWVzOiBBcnJheTFELm5ldyh0b3BrVmFsdWVzKSwgaW5kaWNlczogQXJyYXkxRC5uZXcodG9wa0luZGljZXMpfTtcbiAgfVxuXG4gIHByb3RlY3RlZCBtaW5JbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGxldCBtaW4gPSB2YWx1ZXNbMF07XG4gICAgZm9yIChsZXQgaSA9IDE7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgaWYgKGlzTmFOKHZhbHVlKSkge1xuICAgICAgICByZXR1cm4gU2NhbGFyLm5ldyhOYU4pO1xuICAgICAgfVxuICAgICAgaWYgKHZhbHVlIDwgbWluKSB7XG4gICAgICAgIG1pbiA9IHZhbHVlO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gU2NhbGFyLm5ldyhtaW4pO1xuICB9XG5cbiAgcHJvdGVjdGVkIG1heEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgbGV0IG1heCA9IHZhbHVlc1swXTtcbiAgICBmb3IgKGxldCBpID0gMTsgaSA8IHZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgY29uc3QgdmFsdWUgPSB2YWx1ZXNbaV07XG4gICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgIHJldHVybiBTY2FsYXIubmV3KE5hTik7XG4gICAgICB9XG4gICAgICBpZiAodmFsdWUgPiBtYXgpIHtcbiAgICAgICAgbWF4ID0gdmFsdWU7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBTY2FsYXIubmV3KG1heCk7XG4gIH1cblxuICBwcm90ZWN0ZWQgZXhwSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IG5ld1ZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkodmFsdWVzLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIG5ld1ZhbHVlc1tpXSA9IE1hdGguZXhwKHZhbHVlc1tpXSk7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmRhcnJheS5zaGFwZSwge3ZhbHVlczogbmV3VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgbG9nSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IG5ld1ZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkodmFsdWVzLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgbmV3VmFsdWVzW2ldID0gTWF0aC5sb2codmFsdWUpO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IG5ld1ZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIGxvZ1N1bUV4cEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGNvbnN0IHhNYXggPSB0aGlzLm1heChuZGFycmF5KTtcbiAgICBjb25zdCBhID0gdGhpcy5hcnJheU1pbnVzU2NhbGFyKG5kYXJyYXksIHhNYXgpO1xuICAgIGNvbnN0IGIgPSB0aGlzLmV4cChhKTtcbiAgICBjb25zdCBjID0gdGhpcy5zdW0oYik7XG4gICAgY29uc3QgZCA9IHRoaXMubG9nKGMpO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRoaXMuYWRkKHhNYXgsIGQpO1xuXG4gICAgeE1heC5kaXNwb3NlKCk7XG4gICAgYS5kaXNwb3NlKCk7XG4gICAgYi5kaXNwb3NlKCk7XG4gICAgYy5kaXNwb3NlKCk7XG4gICAgZC5kaXNwb3NlKCk7XG5cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHJvdGVjdGVkIHJlbHVJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIGNvbnN0IHJlc3VsdFZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobmRhcnJheS5zaXplKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICByZXN1bHRWYWx1ZXNbaV0gPSBNYXRoLm1heCgwLCB2YWx1ZXNbaV0pO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHNpZ21vaWRJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIGNvbnN0IHJlc3VsdFZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobmRhcnJheS5zaXplKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICByZXN1bHRWYWx1ZXNbaV0gPSAxIC8gKDEgKyBNYXRoLmV4cCgtdmFsdWVzW2ldKSk7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmRhcnJheS5zaGFwZSwge3ZhbHVlczogcmVzdWx0VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgdGFuaEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgY29uc3QgcmVzdWx0VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShuZGFycmF5LnNpemUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIHJlc3VsdFZhbHVlc1tpXSA9IHV0aWwudGFuaCh2YWx1ZXNbaV0pO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHNpbkludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgY29uc3QgcmVzdWx0VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShuZGFycmF5LnNpemUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIHJlc3VsdFZhbHVlc1tpXSA9IE1hdGguc2luKHZhbHVlc1tpXSk7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmRhcnJheS5zaGFwZSwge3ZhbHVlczogcmVzdWx0VmFsdWVzfSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgc3RlcEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgY29uc3QgcmVzdWx0VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShuZGFycmF5LnNpemUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgcmVzdWx0VmFsdWVzW2ldID0gdmFsdWUgPiAwID8gMSA6ICh2YWx1ZSA8IDAgPyAwIDogdmFsdWUpO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIGNvbnYyZEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhc2VzOiBBcnJheTFEfG51bGwsIHN0cmlkZUhlaWdodDogbnVtYmVyLFxuICAgICAgc3RyaWRlV2lkdGg6IG51bWJlciwgb3V0cHV0SW5mbzogT3V0cHV0SW5mbyk6IEFycmF5M0Qge1xuICAgIGNvbnN0IFt4Um93cywgeENvbHMsIGlucHV0RGVwdGhdID0geC5zaGFwZTtcbiAgICBjb25zdCBmaWx0ZXJIZWlnaHQgPSB3ZWlnaHRzLnNoYXBlWzBdO1xuICAgIGNvbnN0IGZpbHRlcldpZHRoID0gd2VpZ2h0cy5zaGFwZVsxXTtcbiAgICBjb25zdCBvdXREZXB0aCA9IHdlaWdodHMuc2hhcGVbM107XG4gICAgY29uc3QgcGFkTGVmdCA9IG91dHB1dEluZm8ucGFkZGluZ0luZm8ubGVmdDtcbiAgICBjb25zdCBwYWRUb3AgPSBvdXRwdXRJbmZvLnBhZGRpbmdJbmZvLnRvcDtcblxuICAgIGNvbnN0IHkgPSBBcnJheTNELnplcm9zKG91dHB1dEluZm8uc2hhcGUpO1xuICAgIGZvciAobGV0IGQyID0gMDsgZDIgPCBvdXREZXB0aDsgKytkMikge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IHkuc2hhcGVbMF07ICsreVIpIHtcbiAgICAgICAgY29uc3QgeFJDb3JuZXIgPSB5UiAqIHN0cmlkZUhlaWdodCAtIHBhZExlZnQ7XG4gICAgICAgIGNvbnN0IHhSTWluID0gTWF0aC5tYXgoMCwgeFJDb3JuZXIpO1xuICAgICAgICBjb25zdCB4Uk1heCA9IE1hdGgubWluKHhSb3dzLCBmaWx0ZXJIZWlnaHQgKyB4UkNvcm5lcik7XG4gICAgICAgIGZvciAobGV0IHlDID0gMDsgeUMgPCB5LnNoYXBlWzFdOyArK3lDKSB7XG4gICAgICAgICAgY29uc3QgeENDb3JuZXIgPSB5QyAqIHN0cmlkZVdpZHRoIC0gcGFkVG9wO1xuICAgICAgICAgIGNvbnN0IHhDTWluID0gTWF0aC5tYXgoMCwgeENDb3JuZXIpO1xuICAgICAgICAgIGNvbnN0IHhDTWF4ID0gTWF0aC5taW4oeENvbHMsIGZpbHRlcldpZHRoICsgeENDb3JuZXIpO1xuICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB4UiA9IHhSTWluOyB4UiA8IHhSTWF4OyArK3hSKSB7XG4gICAgICAgICAgICBjb25zdCB3UiA9IHhSIC0geFJDb3JuZXI7XG4gICAgICAgICAgICBmb3IgKGxldCB4QyA9IHhDTWluOyB4QyA8IHhDTWF4OyArK3hDKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHdDID0geEMgLSB4Q0Nvcm5lcjtcbiAgICAgICAgICAgICAgZm9yIChsZXQgZDEgPSAwOyBkMSA8IGlucHV0RGVwdGg7ICsrZDEpIHtcbiAgICAgICAgICAgICAgICBjb25zdCBwaXhlbCA9IHguZ2V0KHhSLCB4QywgZDEpO1xuICAgICAgICAgICAgICAgIGNvbnN0IHdlaWdodCA9IHdlaWdodHMuZ2V0KHdSLCB3QywgZDEsIGQyKTtcbiAgICAgICAgICAgICAgICBkb3RQcm9kICs9IHBpeGVsICogd2VpZ2h0O1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIGNvbnN0IGJpYXMgPSAoYmlhc2VzICE9IG51bGwpID8gYmlhc2VzLmdldChkMikgOiAwO1xuICAgICAgICAgIHkuc2V0KGRvdFByb2QgKyBiaWFzLCB5UiwgeUMsIGQyKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfVxuXG4gIHByb3RlY3RlZCBjb252MmRCYWNrUHJvcEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZHk6IEFycmF5M0QsIHdlaWdodHM6IEFycmF5NEQsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiB7ZHg6IEFycmF5M0QsIGR3OiBBcnJheTRELCBkYjogQXJyYXkxRH0ge1xuICAgIGNvbnN0IGZTaXplID0gd2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBkdyA9IHRoaXMuY29udjJkRGVyV2VpZ2h0cyh4LCBkeSwgZlNpemUsIHN0cmlkZSwgcGFkKTtcbiAgICBjb25zdCBkYiA9IHRoaXMuY29udjJkRGVyQmlhcyhkeSk7XG4gICAgY29uc3QgZHggPSB0aGlzLmNvbnYyZFRyYW5zcG9zZUludGVybmFsKGR5LCB3ZWlnaHRzLCBudWxsLCBzdHJpZGUsIHBhZCk7XG4gICAgcmV0dXJuIHtkeCwgZGIsIGR3fTtcbiAgfVxuXG4gIC8qKlxuICAgKiBpbWFnZSBpcyBvZiBzaGFwZSBbciwgYywgZDFdLlxuICAgKiB3ZWlnaHRzIGlzIG9mIHNoYXBlIFtGLCBGLCBkMSwgZDJdLlxuICAgKi9cbiAgcHJvdGVjdGVkIGNvbnYyZFRyYW5zcG9zZUludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhc2VzOiBBcnJheTFEfG51bGwsIG9yaWdTdHJpZGU6IG51bWJlcixcbiAgICAgIG9yaWdQYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIGNvbnN0IGZTaXplID0gd2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBwYWQgPSBmU2l6ZSAtIDEgLSBvcmlnUGFkO1xuICAgIGNvbnN0IG9yaWdJbnB1dERlcHRoID0gd2VpZ2h0cy5zaGFwZVsyXTtcbiAgICBjb25zdCBvcmlnT3V0cHV0RGVwdGggPSB3ZWlnaHRzLnNoYXBlWzNdO1xuICAgIGNvbnN0IHhSb3dzID0geC5zaGFwZVswXTtcbiAgICBjb25zdCB4Q29scyA9IHguc2hhcGVbMV07XG5cbiAgICAvLyBEaWxhdGUgdGhlIGlucHV0LlxuICAgIGNvbnN0IHhSb3dzRGlsYXRlZCA9ICh4Um93cyAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG4gICAgY29uc3QgeENvbHNEaWxhdGVkID0gKHhDb2xzIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcblxuICAgIGNvbnN0IG91dHB1dFNoYXBlID0gY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKFxuICAgICAgICBbeFJvd3NEaWxhdGVkLCB4Q29sc0RpbGF0ZWQsIG9yaWdPdXRwdXREZXB0aF0sIGZTaXplLCBvcmlnSW5wdXREZXB0aCwgMSxcbiAgICAgICAgcGFkKTtcbiAgICBjb25zdCB5ID0gQXJyYXkzRC56ZXJvcyhvdXRwdXRTaGFwZSk7XG4gICAgZm9yIChsZXQgZDIgPSAwOyBkMiA8IG9yaWdJbnB1dERlcHRoOyArK2QyKSB7XG4gICAgICBmb3IgKGxldCB5UiA9IDA7IHlSIDwgeS5zaGFwZVswXTsgKyt5Uikge1xuICAgICAgICBjb25zdCB4UkNvcm5lciA9IHlSIC0gcGFkO1xuICAgICAgICBjb25zdCB4Uk1pbiA9IE1hdGgubWF4KDAsIE1hdGguY2VpbCh4UkNvcm5lciAvIG9yaWdTdHJpZGUpKTtcbiAgICAgICAgY29uc3QgeFJNYXggPSBNYXRoLm1pbih4Um93cywgKGZTaXplICsgeFJDb3JuZXIpIC8gb3JpZ1N0cmlkZSk7XG5cbiAgICAgICAgZm9yIChsZXQgeUMgPSAwOyB5QyA8IHkuc2hhcGVbMV07ICsreUMpIHtcbiAgICAgICAgICBjb25zdCB4Q0Nvcm5lciA9IHlDIC0gcGFkO1xuICAgICAgICAgIGNvbnN0IHhDTWluID0gTWF0aC5tYXgoMCwgTWF0aC5jZWlsKHhDQ29ybmVyIC8gb3JpZ1N0cmlkZSkpO1xuICAgICAgICAgIGNvbnN0IHhDTWF4ID0gTWF0aC5taW4oeENvbHMsIChmU2l6ZSArIHhDQ29ybmVyKSAvIG9yaWdTdHJpZGUpO1xuXG4gICAgICAgICAgbGV0IGRvdFByb2QgPSAwO1xuICAgICAgICAgIGZvciAobGV0IHhSID0geFJNaW47IHhSIDwgeFJNYXg7ICsreFIpIHtcbiAgICAgICAgICAgIGNvbnN0IHdSID0geFIgKiBvcmlnU3RyaWRlIC0geFJDb3JuZXI7XG5cbiAgICAgICAgICAgIGZvciAobGV0IHhDID0geENNaW47IHhDIDwgeENNYXg7ICsreEMpIHtcbiAgICAgICAgICAgICAgY29uc3Qgd0MgPSB4QyAqIG9yaWdTdHJpZGUgLSB4Q0Nvcm5lcjtcblxuICAgICAgICAgICAgICBmb3IgKGxldCBkMSA9IDA7IGQxIDwgb3JpZ091dHB1dERlcHRoOyArK2QxKSB7XG4gICAgICAgICAgICAgICAgY29uc3QgcGl4ZWwgPSB4LmdldCh4UiwgeEMsIGQxKTtcbiAgICAgICAgICAgICAgICBjb25zdCB3ZWlnaHQgPVxuICAgICAgICAgICAgICAgICAgICB3ZWlnaHRzLmdldChmU2l6ZSAtIDEgLSB3UiwgZlNpemUgLSAxIC0gd0MsIGQyLCBkMSk7XG4gICAgICAgICAgICAgICAgZG90UHJvZCArPSBwaXhlbCAqIHdlaWdodDtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgICBjb25zdCBiaWFzID0gYmlhc2VzICE9IG51bGwgPyBiaWFzZXMuZ2V0KGQyKSA6IDA7XG4gICAgICAgICAgeS5zZXQoZG90UHJvZCArIGJpYXMsIHlSLCB5QywgZDIpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiB5O1xuICB9XG5cbiAgLyoqXG4gICAqIGltYWdlIGlzIG9mIHNoYXBlIFtyLCBjLCBkMV0uXG4gICAqIHdlaWdodHMgaXMgb2Ygc2hhcGUgW0YsIEYsIGQxLCBkMl0uXG4gICAqL1xuICBwcm90ZWN0ZWQgY29udjJkVHJhbnNwb3NlU2hhZGVyTGlrZShcbiAgICAgIHg6IEFycmF5M0QsIG9yaWdXZWlnaHRzOiBBcnJheTRELCBvcmlnU3RyaWRlOiBudW1iZXIsXG4gICAgICBvcmlnUGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICBjb25zdCBmU2l6ZSA9IG9yaWdXZWlnaHRzLnNoYXBlWzBdO1xuICAgIGNvbnN0IHBhZCA9IGZTaXplIC0gMSAtIG9yaWdQYWQ7XG4gICAgY29uc3Qgb3JpZ0lucHV0RGVwdGggPSBvcmlnV2VpZ2h0cy5zaGFwZVsyXTtcbiAgICBjb25zdCBvcmlnT3V0cHV0RGVwdGggPSBvcmlnV2VpZ2h0cy5zaGFwZVszXTtcbiAgICBjb25zdCB4Um93cyA9IHguc2hhcGVbMF07XG4gICAgY29uc3QgeENvbHMgPSB4LnNoYXBlWzFdO1xuXG4gICAgLy8gRGlsYXRlIHRoZSBpbnB1dC5cbiAgICBjb25zdCB4Um93c0RpbGF0ZWQgPSAoeFJvd3MgLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuICAgIGNvbnN0IHhDb2xzRGlsYXRlZCA9ICh4Q29scyAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG5cbiAgICBjb25zdCBvdXRwdXRTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgICAgW3hSb3dzRGlsYXRlZCwgeENvbHNEaWxhdGVkLCBvcmlnT3V0cHV0RGVwdGhdLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGgsIDEsXG4gICAgICAgIHBhZCk7XG4gICAgY29uc3QgeSA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuXG4gICAgZm9yIChsZXQgZDIgPSAwOyBkMiA8IG9yaWdJbnB1dERlcHRoOyArK2QyKSB7XG4gICAgICBmb3IgKGxldCB5UiA9IDA7IHlSIDwgeS5zaGFwZVswXTsgKyt5Uikge1xuICAgICAgICBmb3IgKGxldCB5QyA9IDA7IHlDIDwgeS5zaGFwZVsxXTsgKyt5Qykge1xuICAgICAgICAgIC8vIFNoYWRlciBjb2RlIGJlZ2lucy5cbiAgICAgICAgICBjb25zdCB4UkNvcm5lciA9IHlSIC0gcGFkO1xuICAgICAgICAgIGNvbnN0IHhDQ29ybmVyID0geUMgLSBwYWQ7XG4gICAgICAgICAgbGV0IGRvdFByb2QgPSAwO1xuICAgICAgICAgIGZvciAobGV0IHdSID0gMDsgd1IgPCBmU2l6ZTsgKyt3Uikge1xuICAgICAgICAgICAgY29uc3QgeFIgPSAoeFJDb3JuZXIgKyB3UikgLyBvcmlnU3RyaWRlO1xuICAgICAgICAgICAgaWYgKHhSIDwgMCB8fCB4UiA+PSB4Um93cyB8fCBNYXRoLmZsb29yKHhSKSAhPT0geFIpIHtcbiAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBmb3IgKGxldCB3QyA9IDA7IHdDIDwgZlNpemU7ICsrd0MpIHtcbiAgICAgICAgICAgICAgY29uc3QgeEMgPSAoeENDb3JuZXIgKyB3QykgLyBvcmlnU3RyaWRlO1xuICAgICAgICAgICAgICBpZiAoeEMgPCAwIHx8IHhDID49IHhDb2xzIHx8IE1hdGguZmxvb3IoeEMpICE9PSB4Qykge1xuICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIGZvciAobGV0IGQxID0gMDsgZDEgPCBvcmlnT3V0cHV0RGVwdGg7ICsrZDEpIHtcbiAgICAgICAgICAgICAgICBjb25zdCBwaXhlbCA9IHguZ2V0KHhSLCB4QywgZDEpO1xuICAgICAgICAgICAgICAgIGNvbnN0IHdlaWdodCA9XG4gICAgICAgICAgICAgICAgICAgIG9yaWdXZWlnaHRzLmdldChmU2l6ZSAtIDEgLSB3UiwgZlNpemUgLSAxIC0gd0MsIGQyLCBkMSk7XG4gICAgICAgICAgICAgICAgZG90UHJvZCArPSBwaXhlbCAqIHdlaWdodDtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgICB5LnNldChkb3RQcm9kLCB5UiwgeUMsIGQyKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfVxuXG4gIGNvbnYyZERlcldlaWdodHMoXG4gICAgICB4OiBBcnJheTNELCBkWTogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsXG4gICAgICB6ZXJvUGFkOiBudW1iZXIpOiBBcnJheTREIHtcbiAgICBjb25zdCBpbnB1dERlcHRoID0geC5zaGFwZVsyXTtcbiAgICBjb25zdCBvdXRwdXREZXB0aCA9IGRZLnNoYXBlWzJdO1xuICAgIGNvbnN0IHdlaWdodHNTaGFwZSA9XG4gICAgICAgIGNvbnZfdXRpbC5jb21wdXRlV2VpZ2h0c1NoYXBlNEQoaW5wdXREZXB0aCwgb3V0cHV0RGVwdGgsIGZTaXplKTtcbiAgICBjb25zdCBkVyA9IEFycmF5NEQuemVyb3Mod2VpZ2h0c1NoYXBlKTtcblxuICAgIGNvbnN0IHlOdW1Sb3dzID0gZFkuc2hhcGVbMF07XG4gICAgY29uc3QgeU51bUNvbHMgPSBkWS5zaGFwZVsxXTtcbiAgICBjb25zdCB4TnVtUm93cyA9IHguc2hhcGVbMF07XG4gICAgY29uc3QgeE51bUNvbHMgPSB4LnNoYXBlWzFdO1xuXG4gICAgZm9yIChsZXQgd1IgPSAwOyB3UiA8IGZTaXplOyArK3dSKSB7XG4gICAgICBjb25zdCB5Uk1pbiA9IE1hdGgubWF4KDAsIE1hdGguY2VpbCgoemVyb1BhZCAtIHdSKSAvIHN0cmlkZSkpO1xuICAgICAgY29uc3QgeVJNYXggPSBNYXRoLm1pbih5TnVtUm93cywgKHhOdW1Sb3dzICsgemVyb1BhZCAtIHdSKSAvIHN0cmlkZSk7XG5cbiAgICAgIGZvciAobGV0IHdDID0gMDsgd0MgPCBmU2l6ZTsgKyt3Qykge1xuICAgICAgICBjb25zdCB5Q01pbiA9IE1hdGgubWF4KDAsIE1hdGguY2VpbCgoemVyb1BhZCAtIHdDKSAvIHN0cmlkZSkpO1xuICAgICAgICBjb25zdCB5Q01heCA9IE1hdGgubWluKHlOdW1Db2xzLCAoeE51bUNvbHMgKyB6ZXJvUGFkIC0gd0MpIC8gc3RyaWRlKTtcblxuICAgICAgICBmb3IgKGxldCBkMSA9IDA7IGQxIDwgaW5wdXREZXB0aDsgKytkMSkge1xuICAgICAgICAgIGZvciAobGV0IGQyID0gMDsgZDIgPCBvdXRwdXREZXB0aDsgKytkMikge1xuICAgICAgICAgICAgLy8gTmVlZCB0byBjb252b2x2ZS5cbiAgICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICAgIGZvciAobGV0IHlSID0geVJNaW47IHlSIDwgeVJNYXg7ICsreVIpIHtcbiAgICAgICAgICAgICAgY29uc3QgeFIgPSB3UiArIHlSICogc3RyaWRlIC0gemVyb1BhZDtcbiAgICAgICAgICAgICAgZm9yIChsZXQgeUMgPSB5Q01pbjsgeUMgPCB5Q01heDsgKyt5Qykge1xuICAgICAgICAgICAgICAgIGNvbnN0IHhDID0gd0MgKyB5QyAqIHN0cmlkZSAtIHplcm9QYWQ7XG4gICAgICAgICAgICAgICAgZG90UHJvZCArPSB4LmdldCh4UiwgeEMsIGQxKSAqIGRZLmdldCh5UiwgeUMsIGQyKTtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgZFcuc2V0KGRvdFByb2QsIHdSLCB3QywgZDEsIGQyKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGRXO1xuICB9XG5cbiAgY29udjJkRGVyQmlhcyhkWTogQXJyYXkzRCk6IEFycmF5MUQge1xuICAgIGNvbnN0IG91dHB1dERlcHRoID0gZFkuc2hhcGVbMl07XG4gICAgY29uc3QgbnVtUm93cyA9IGRZLnNoYXBlWzBdO1xuICAgIGNvbnN0IG51bUNvbHMgPSBkWS5zaGFwZVsxXTtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KG91dHB1dERlcHRoKTtcbiAgICBmb3IgKGxldCBkMiA9IDA7IGQyIDwgb3V0cHV0RGVwdGg7ICsrZDIpIHtcbiAgICAgIGxldCBzdW0gPSAwO1xuICAgICAgZm9yIChsZXQgciA9IDA7IHIgPCBudW1Sb3dzOyArK3IpIHtcbiAgICAgICAgZm9yIChsZXQgYyA9IDA7IGMgPCBudW1Db2xzOyArK2MpIHtcbiAgICAgICAgICBzdW0gKz0gZFkuZ2V0KHIsIGMsIGQyKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgdmFsdWVzW2QyXSA9IHN1bTtcbiAgICB9XG4gICAgcmV0dXJuIEFycmF5MUQubmV3KHZhbHVlcyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgc3dpdGNoRGltSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KHQ6IFQsIG5ld0RpbTogbnVtYmVyW10pOiBUIHtcbiAgICBjb25zdCBuZXdTaGFwZTogbnVtYmVyW10gPSBuZXcgQXJyYXkodC5yYW5rKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG5ld1NoYXBlLmxlbmd0aDsgaSsrKSB7XG4gICAgICBuZXdTaGFwZVtpXSA9IHQuc2hhcGVbbmV3RGltW2ldXTtcbiAgICB9XG4gICAgY29uc3QgcmVzdWx0VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheSh0LnNpemUpO1xuICAgIGNvbnN0IHZhbHVlcyA9IHQuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgcmVzdWx0ID0gTkRBcnJheS5tYWtlPFQ+KG5ld1NoYXBlLCB7dmFsdWVzOiByZXN1bHRWYWx1ZXN9KTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHQuc2l6ZTsgKytpKSB7XG4gICAgICBjb25zdCBsb2MgPSB0LmluZGV4VG9Mb2MoaSk7XG5cbiAgICAgIC8vIFBlcm11dGUgbG9jYXRpb24uXG4gICAgICBjb25zdCBuZXdMb2M6IG51bWJlcltdID0gbmV3IEFycmF5KGxvYy5sZW5ndGgpO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBuZXdMb2MubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgbmV3TG9jW2ldID0gbG9jW25ld0RpbVtpXV07XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IG5ld0luZGV4ID0gcmVzdWx0LmxvY1RvSW5kZXgobmV3TG9jKTtcbiAgICAgIHJlc3VsdFZhbHVlc1tuZXdJbmRleF0gPSB2YWx1ZXNbaV07XG4gICAgfVxuICAgIHJldHVybiByZXN1bHQ7XG4gIH1cblxuICBwcml2YXRlIHBvb2woXG4gICAgICB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIsXG4gICAgICBwb29sVHlwZTogJ21heCd8J21pbid8J2F2ZycpIHtcbiAgICBjb25zdCBbeFJvd3MsIHhDb2xzLCBkZXB0aF0gPSB4LnNoYXBlO1xuICAgIGNvbnN0IG91dHB1dFNoYXBlID0gY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKFxuICAgICAgICBbeFJvd3MsIHhDb2xzLCBkZXB0aF0sIGZTaXplLCBkZXB0aCwgc3RyaWRlLCBwYWQpO1xuICAgIGNvbnN0IHkgPSBBcnJheTNELnplcm9zKG91dHB1dFNoYXBlKTtcbiAgICBmb3IgKGxldCBkID0gMDsgZCA8IGRlcHRoOyArK2QpIHtcbiAgICAgIGZvciAobGV0IHlSID0gMDsgeVIgPCB5LnNoYXBlWzBdOyArK3lSKSB7XG4gICAgICAgIGNvbnN0IHhSQ29ybmVyID0geVIgKiBzdHJpZGUgLSBwYWQ7XG4gICAgICAgIGNvbnN0IHhSTWluID0gTWF0aC5tYXgoMCwgeFJDb3JuZXIpO1xuICAgICAgICBjb25zdCB4Uk1heCA9IE1hdGgubWluKHhSb3dzLCBmU2l6ZSArIHhSQ29ybmVyKTtcbiAgICAgICAgZm9yIChsZXQgeUMgPSAwOyB5QyA8IHkuc2hhcGVbMV07ICsreUMpIHtcbiAgICAgICAgICBjb25zdCB4Q0Nvcm5lciA9IHlDICogc3RyaWRlIC0gcGFkO1xuICAgICAgICAgIGNvbnN0IHhDTWluID0gTWF0aC5tYXgoMCwgeENDb3JuZXIpO1xuICAgICAgICAgIGNvbnN0IHhDTWF4ID0gTWF0aC5taW4oeENvbHMsIGZTaXplICsgeENDb3JuZXIpO1xuXG5cbiAgICAgICAgICBsZXQgbWluTWF4VmFsdWUgPVxuICAgICAgICAgICAgICAocG9vbFR5cGUgPT09ICdtYXgnID8gTnVtYmVyLk5FR0FUSVZFX0lORklOSVRZIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIE51bWJlci5QT1NJVElWRV9JTkZJTklUWSk7XG4gICAgICAgICAgbGV0IGF2Z1ZhbHVlID0gMDtcblxuICAgICAgICAgIGZvciAobGV0IHhSID0geFJNaW47IHhSIDwgeFJNYXg7ICsreFIpIHtcbiAgICAgICAgICAgIGZvciAobGV0IHhDID0geENNaW47IHhDIDwgeENNYXg7ICsreEMpIHtcbiAgICAgICAgICAgICAgY29uc3QgcGl4ZWwgPSB4LmdldCh4UiwgeEMsIGQpO1xuICAgICAgICAgICAgICBpZiAoaXNOYU4ocGl4ZWwpKSB7XG4gICAgICAgICAgICAgICAgbWluTWF4VmFsdWUgPSBOYU47XG4gICAgICAgICAgICAgICAgYXZnVmFsdWUgPSBOYU47XG4gICAgICAgICAgICAgICAgYnJlYWs7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgaWYgKChwb29sVHlwZSA9PT0gJ21heCcgJiYgcGl4ZWwgPiBtaW5NYXhWYWx1ZSkgfHxcbiAgICAgICAgICAgICAgICAgIChwb29sVHlwZSA9PT0gJ21pbicgJiYgcGl4ZWwgPCBtaW5NYXhWYWx1ZSkpIHtcbiAgICAgICAgICAgICAgICBtaW5NYXhWYWx1ZSA9IHBpeGVsO1xuICAgICAgICAgICAgICB9IGVsc2UgaWYgKHBvb2xUeXBlID09PSAnYXZnJykge1xuICAgICAgICAgICAgICAgIGF2Z1ZhbHVlICs9IHBpeGVsIC8gKGZTaXplICogZlNpemUpO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgICBpZiAoaXNOYU4obWluTWF4VmFsdWUpKSB7XG4gICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgICB5LnNldChwb29sVHlwZSA9PT0gJ2F2ZycgPyBhdmdWYWx1ZSA6IG1pbk1heFZhbHVlLCB5UiwgeUMsIGQpO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiB5O1xuICB9XG5cbiAgcHJvdGVjdGVkIG1heFBvb2xJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHJldHVybiB0aGlzLnBvb2woeCwgZlNpemUsIHN0cmlkZSwgcGFkLCAnbWF4Jyk7XG4gIH1cblxuICBtYXhQb29sUG9zaXRpb25zKHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcikge1xuICAgIGNvbnN0IFt4Um93cywgeENvbHMsIGRlcHRoXSA9IHguc2hhcGU7XG4gICAgY29uc3Qgb3V0cHV0U2hhcGUgPVxuICAgICAgICBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoeC5zaGFwZSwgZlNpemUsIGRlcHRoLCBzdHJpZGUsIHBhZCk7XG4gICAgY29uc3QgbWF4UG9zaXRpb25zID0gQXJyYXkzRC56ZXJvcyhvdXRwdXRTaGFwZSk7XG4gICAgZm9yIChsZXQgZCA9IDA7IGQgPCBkZXB0aDsgKytkKSB7XG4gICAgICBmb3IgKGxldCB5UiA9IDA7IHlSIDwgb3V0cHV0U2hhcGVbMF07ICsreVIpIHtcbiAgICAgICAgY29uc3QgeFJDb3JuZXIgPSB5UiAqIHN0cmlkZSAtIHBhZDtcbiAgICAgICAgY29uc3QgeFJNaW4gPSBNYXRoLm1heCgwLCB4UkNvcm5lcik7XG4gICAgICAgIGNvbnN0IHhSTWF4ID0gTWF0aC5taW4oeFJvd3MsIGZTaXplICsgeFJDb3JuZXIpO1xuICAgICAgICBmb3IgKGxldCB5QyA9IDA7IHlDIDwgb3V0cHV0U2hhcGVbMV07ICsreUMpIHtcbiAgICAgICAgICBjb25zdCB4Q0Nvcm5lciA9IHlDICogc3RyaWRlIC0gcGFkO1xuICAgICAgICAgIGNvbnN0IHhDTWluID0gTWF0aC5tYXgoMCwgeENDb3JuZXIpO1xuICAgICAgICAgIGNvbnN0IHhDTWF4ID0gTWF0aC5taW4oeENvbHMsIGZTaXplICsgeENDb3JuZXIpO1xuICAgICAgICAgIGxldCBtYXhWYWx1ZSA9IE51bWJlci5ORUdBVElWRV9JTkZJTklUWTtcbiAgICAgICAgICBsZXQgbWF4UG9zaXRpb24gPSAtMTtcbiAgICAgICAgICBmb3IgKGxldCB4UiA9IHhSTWluOyB4UiA8IHhSTWF4OyArK3hSKSB7XG4gICAgICAgICAgICBjb25zdCB3UiA9IHhSIC0geFJDb3JuZXI7XG4gICAgICAgICAgICBmb3IgKGxldCB4QyA9IHhDTWluOyB4QyA8IHhDTWF4OyArK3hDKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHdDID0geEMgLSB4Q0Nvcm5lcjtcbiAgICAgICAgICAgICAgY29uc3QgcGl4ZWwgPSB4LmdldCh4UiwgeEMsIGQpO1xuICAgICAgICAgICAgICBpZiAocGl4ZWwgPiBtYXhWYWx1ZSkge1xuICAgICAgICAgICAgICAgIG1heFZhbHVlID0gcGl4ZWw7XG4gICAgICAgICAgICAgICAgbWF4UG9zaXRpb24gPSB3UiAqIGZTaXplICsgd0M7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgbWF4UG9zaXRpb25zLnNldChtYXhQb3NpdGlvbiwgeVIsIHlDLCBkKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gbWF4UG9zaXRpb25zO1xuICB9XG5cbiAgcHJvdGVjdGVkIG1heFBvb2xCYWNrcHJvcEludGVybmFsKFxuICAgICAgZHk6IEFycmF5M0QsIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIG9yaWdTdHJpZGU6IG51bWJlcixcbiAgICAgIG9yaWdQYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIGNvbnN0IG1heFBvc2l0aW9ucyA9IHRoaXMubWF4UG9vbFBvc2l0aW9ucyh4LCBmU2l6ZSwgb3JpZ1N0cmlkZSwgb3JpZ1BhZCk7XG4gICAgY29uc3QgcGFkID0gZlNpemUgLSAxIC0gb3JpZ1BhZDtcbiAgICBjb25zdCBbZHlSb3dzLCBkeUNvbHMsIGRlcHRoXSA9IGR5LnNoYXBlO1xuXG4gICAgLy8gRGlsYXRlIHRoZSBpbnB1dC5cbiAgICBjb25zdCBkeVJvd3NEaWxhdGVkID0gKGR5Um93cyAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG4gICAgY29uc3QgZHhDb2xzRGlsYXRlZCA9IChkeUNvbHMgLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuXG4gICAgY29uc3Qgb3V0cHV0U2hhcGUgPSBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgIFtkeVJvd3NEaWxhdGVkLCBkeENvbHNEaWxhdGVkLCBkZXB0aF0sIGZTaXplLCBkZXB0aCwgMSwgcGFkKTtcbiAgICBjb25zdCBkeCA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuXG4gICAgZm9yIChsZXQgZCA9IDA7IGQgPCBkZXB0aDsgKytkKSB7XG4gICAgICBmb3IgKGxldCBkeFIgPSAwOyBkeFIgPCBkeC5zaGFwZVswXTsgKytkeFIpIHtcbiAgICAgICAgZm9yIChsZXQgZHhDID0gMDsgZHhDIDwgZHguc2hhcGVbMV07ICsrZHhDKSB7XG4gICAgICAgICAgLy8gU2hhZGVyIGNvZGUgYmVnaW5zLlxuICAgICAgICAgIGNvbnN0IGR5UkNvcm5lciA9IGR4UiAtIHBhZDtcbiAgICAgICAgICBjb25zdCBkeUNDb3JuZXIgPSBkeEMgLSBwYWQ7XG4gICAgICAgICAgbGV0IGRvdFByb2QgPSAwO1xuICAgICAgICAgIGZvciAobGV0IHdSID0gMDsgd1IgPCBmU2l6ZTsgKyt3Uikge1xuICAgICAgICAgICAgY29uc3QgZHlSID0gKGR5UkNvcm5lciArIHdSKSAvIG9yaWdTdHJpZGU7XG4gICAgICAgICAgICBpZiAoZHlSIDwgMCB8fCBkeVIgPj0gZHlSb3dzIHx8IE1hdGguZmxvb3IoZHlSKSAhPT0gZHlSKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgZm9yIChsZXQgd0MgPSAwOyB3QyA8IGZTaXplOyArK3dDKSB7XG4gICAgICAgICAgICAgIGNvbnN0IGR5QyA9IChkeUNDb3JuZXIgKyB3QykgLyBvcmlnU3RyaWRlO1xuICAgICAgICAgICAgICBpZiAoZHlDIDwgMCB8fCBkeUMgPj0gZHlDb2xzIHx8IE1hdGguZmxvb3IoZHlDKSAhPT0gZHlDKSB7XG4gICAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgICAgY29uc3QgbWF4UG9zID0gZlNpemUgKiBmU2l6ZSAtIDEgLSBtYXhQb3NpdGlvbnMuZ2V0KGR5UiwgZHlDLCBkKTtcbiAgICAgICAgICAgICAgY29uc3QgY3VyUG9zID0gd1IgKiBmU2l6ZSArIHdDO1xuXG4gICAgICAgICAgICAgIGNvbnN0IG1hc2sgPSBtYXhQb3MgPT09IGN1clBvcyA/IDEgOiAwO1xuICAgICAgICAgICAgICBpZiAobWFzayA9PT0gMCkge1xuICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICB9XG5cbiAgICAgICAgICAgICAgY29uc3QgcGl4ZWwgPSBkeS5nZXQoZHlSLCBkeUMsIGQpO1xuICAgICAgICAgICAgICBkb3RQcm9kICs9IHBpeGVsICogbWFzaztcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgZHguc2V0KGRvdFByb2QsIGR4UiwgZHhDLCBkKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gZHg7XG4gIH1cblxuICBwcm90ZWN0ZWQgbWluUG9vbEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgcmV0dXJuIHRoaXMucG9vbCh4LCBmU2l6ZSwgc3RyaWRlLCBwYWQsICdtaW4nKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBhdmdQb29sSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICByZXR1cm4gdGhpcy5wb29sKHgsIGZTaXplLCBzdHJpZGUsIHBhZCwgJ2F2ZycpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHJlc2l6ZUJpbGluZWFyM0RJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIG5ld1NoYXBlMkQ6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBhbGlnbkNvcm5lcnM6IGJvb2xlYW4pOiBBcnJheTNEIHtcbiAgICBjb25zdCBvdXRwdXQgPSBBcnJheTNELnplcm9zKFtuZXdTaGFwZTJEWzBdLCBuZXdTaGFwZTJEWzFdLCB4LnNoYXBlWzJdXSk7XG5cbiAgICBjb25zdCBlZmZlY3RpdmVJbnB1dFNpemUgPVxuICAgICAgICBhbGlnbkNvcm5lcnMgPyBbeC5zaGFwZVswXSAtIDEsIHguc2hhcGVbMV0gLSAxLCB4LnNoYXBlWzJdXSA6IHguc2hhcGU7XG4gICAgY29uc3QgZWZmZWN0aXZlT3V0cHV0U2l6ZSA9IGFsaWduQ29ybmVycyA/XG4gICAgICAgIFtvdXRwdXQuc2hhcGVbMF0gLSAxLCBvdXRwdXQuc2hhcGVbMV0gLSAxLCBvdXRwdXQuc2hhcGVbMl1dIDpcbiAgICAgICAgb3V0cHV0LnNoYXBlO1xuICAgIGZvciAobGV0IHIgPSAwOyByIDwgb3V0cHV0LnNoYXBlWzBdOyByKyspIHtcbiAgICAgIGZvciAobGV0IGMgPSAwOyBjIDwgb3V0cHV0LnNoYXBlWzFdOyBjKyspIHtcbiAgICAgICAgZm9yIChsZXQgZCA9IDA7IGQgPCBvdXRwdXQuc2hhcGVbMl07IGQrKykge1xuICAgICAgICAgIC8vIEJlZ2luIHNoYWRlci5cblxuICAgICAgICAgIC8vIENvbXB1dGUgdGhlIGZyYWN0aW9uYWwgaW5kZXggb2YgdGhlIHNvdXJjZS5cbiAgICAgICAgICBjb25zdCBzb3VyY2VGcmFjUm93ID1cbiAgICAgICAgICAgICAgKGVmZmVjdGl2ZUlucHV0U2l6ZVswXSkgKiByIC8gKGVmZmVjdGl2ZU91dHB1dFNpemVbMF0pO1xuICAgICAgICAgIGNvbnN0IHNvdXJjZUZyYWNDb2wgPVxuICAgICAgICAgICAgICAoZWZmZWN0aXZlSW5wdXRTaXplWzFdKSAqIGMgLyAoZWZmZWN0aXZlT3V0cHV0U2l6ZVsxXSk7XG5cbiAgICAgICAgICBjb25zdCBzb3VyY2VSb3dGbG9vciA9IE1hdGguZmxvb3Ioc291cmNlRnJhY1Jvdyk7XG4gICAgICAgICAgY29uc3Qgc291cmNlUm93Q2VpbCA9XG4gICAgICAgICAgICAgIE1hdGgubWluKHguc2hhcGVbMF0gLSAxLCBNYXRoLmNlaWwoc291cmNlRnJhY1JvdykpO1xuICAgICAgICAgIGNvbnN0IHNvdXJjZUNvbEZsb29yID0gTWF0aC5mbG9vcihzb3VyY2VGcmFjQ29sKTtcbiAgICAgICAgICBjb25zdCBzb3VyY2VDb2xDZWlsID1cbiAgICAgICAgICAgICAgTWF0aC5taW4oeC5zaGFwZVsxXSAtIDEsIE1hdGguY2VpbChzb3VyY2VGcmFjQ29sKSk7XG5cbiAgICAgICAgICBjb25zdCB0b3BMZWZ0ID0geC5nZXQoc291cmNlUm93Rmxvb3IsIHNvdXJjZUNvbEZsb29yLCBkKTtcbiAgICAgICAgICBjb25zdCBib3R0b21MZWZ0ID0geC5nZXQoc291cmNlUm93Q2VpbCwgc291cmNlQ29sRmxvb3IsIGQpO1xuICAgICAgICAgIGNvbnN0IHRvcFJpZ2h0ID0geC5nZXQoc291cmNlUm93Rmxvb3IsIHNvdXJjZUNvbENlaWwsIGQpO1xuICAgICAgICAgIGNvbnN0IGJvdHRvbVJpZ2h0ID0geC5nZXQoc291cmNlUm93Q2VpbCwgc291cmNlQ29sQ2VpbCwgZCk7XG5cbiAgICAgICAgICBjb25zdCByb3dGcmFjID0gc291cmNlRnJhY1JvdyAtIHNvdXJjZVJvd0Zsb29yO1xuICAgICAgICAgIGNvbnN0IGNvbEZyYWMgPSBzb3VyY2VGcmFjQ29sIC0gc291cmNlQ29sRmxvb3I7XG5cbiAgICAgICAgICBjb25zdCB0b3AgPSB0b3BMZWZ0ICsgKHRvcFJpZ2h0IC0gdG9wTGVmdCkgKiBjb2xGcmFjO1xuICAgICAgICAgIGNvbnN0IGJvdHRvbSA9IGJvdHRvbUxlZnQgKyAoYm90dG9tUmlnaHQgLSBib3R0b21MZWZ0KSAqIGNvbEZyYWM7XG4gICAgICAgICAgY29uc3QgbmV3VmFsdWUgPSB0b3AgKyAoYm90dG9tIC0gdG9wKSAqIHJvd0ZyYWM7XG5cbiAgICAgICAgICBvdXRwdXQuc2V0KG5ld1ZhbHVlLCByLCBjLCBkKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiBvdXRwdXQ7XG4gIH1cblxuICBwcm90ZWN0ZWQgYmF0Y2hOb3JtYWxpemF0aW9uM0RJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIG1lYW46IEFycmF5M0R8QXJyYXkxRCwgdmFyaWFuY2U6IEFycmF5M0R8QXJyYXkxRCxcbiAgICAgIHZhcmlhbmNlRXBzaWxvbiA9IC4wMDEsIHNjYWxlPzogQXJyYXkzRHxBcnJheTFELFxuICAgICAgb2Zmc2V0PzogQXJyYXkzRHxBcnJheTFEKTogQXJyYXkzRCB7XG4gICAgY29uc3QgeFZhbHVlcyA9IHguZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgbWVhblZhbHVlcyA9IG1lYW4uZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgdmFyaWFuY2VWYWx1ZXMgPSB2YXJpYW5jZS5nZXRWYWx1ZXMoKTtcbiAgICBjb25zdCBzY2FsZVZhbHVlcyA9IHNjYWxlID8gc2NhbGUuZ2V0VmFsdWVzKCkgOiBuZXcgRmxvYXQzMkFycmF5KFsxXSk7XG4gICAgY29uc3Qgb2Zmc2V0VmFsdWVzID0gb2Zmc2V0ID8gb2Zmc2V0LmdldFZhbHVlcygpIDogbmV3IEZsb2F0MzJBcnJheShbMF0pO1xuICAgIGNvbnN0IG91dFZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkoeFZhbHVlcy5sZW5ndGgpO1xuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB4VmFsdWVzLmxlbmd0aDsgaSsrKSB7XG4gICAgICBvdXRWYWx1ZXNbaV0gPSBvZmZzZXRWYWx1ZXNbaSAlIG9mZnNldFZhbHVlcy5sZW5ndGhdICtcbiAgICAgICAgICAoeFZhbHVlc1tpXSAtIG1lYW5WYWx1ZXNbaSAlIG1lYW5WYWx1ZXMubGVuZ3RoXSkgKlxuICAgICAgICAgICAgICBzY2FsZVZhbHVlc1tpICUgc2NhbGVWYWx1ZXMubGVuZ3RoXSAvXG4gICAgICAgICAgICAgIE1hdGguc3FydChcbiAgICAgICAgICAgICAgICAgIHZhcmlhbmNlVmFsdWVzW2kgJSB2YXJpYW5jZVZhbHVlcy5sZW5ndGhdICsgdmFyaWFuY2VFcHNpbG9uKTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxBcnJheTNEPih4LnNoYXBlLCB7dmFsdWVzOiBvdXRWYWx1ZXN9KTtcbiAgfVxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5pbXBvcnQge0dQR1BVQ29udGV4dH0gZnJvbSAnLi93ZWJnbC9ncGdwdV9jb250ZXh0JztcbmltcG9ydCB7VGV4dHVyZU1hbmFnZXJ9IGZyb20gJy4vd2ViZ2wvdGV4dHVyZV9tYW5hZ2VyJztcbmltcG9ydCAqIGFzIHdlYmdsX3V0aWwgZnJvbSAnLi93ZWJnbC93ZWJnbF91dGlsJztcblxuLy8gVGhlc2UgZ2xvYmFsIHZhcmlhYmxlcyBuZWVkIHRvIGJlIGluaXRpYWxpemVkIHRvIG51bGwgc28gdGhhdCBjbG9zdXJlIGtub3dzXG4vLyBub3QgdG8gc2VhbCB0aGVtLlxuLyoqIEBoaWRkZW4gKi9cbmV4cG9ydCBsZXQgR1BHUFU6IEdQR1BVQ29udGV4dCA9IG51bGw7XG4vKiogQGhpZGRlbiAqL1xuZXhwb3J0IGxldCBURVhUVVJFX01BTkFHRVI6IFRleHR1cmVNYW5hZ2VyID0gbnVsbDtcblxuLyoqIEBoaWRkZW4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgTkRBcnJheURhdGEge1xuICB2YWx1ZXM/OiBGbG9hdDMyQXJyYXk7XG4gIHRleHR1cmU/OiBXZWJHTFRleHR1cmU7XG4gIC8qKiBbcm93cywgY29sdW1uc10gc2hhcGUgb2YgdGhlIHRleHR1cmUuICovXG4gIHRleHR1cmVTaGFwZVJDPzogW251bWJlciwgbnVtYmVyXTtcbn1cblxuLyoqIEBoaWRkZW4gKi9cbmV4cG9ydCBmdW5jdGlvbiBpbml0aWFsaXplR1BVKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHRleHR1cmVNYW5hZ2VyOiBUZXh0dXJlTWFuYWdlcikge1xuICBHUEdQVSA9IGdwZ3B1O1xuICBURVhUVVJFX01BTkFHRVIgPSB0ZXh0dXJlTWFuYWdlcjtcbn1cblxuZnVuY3Rpb24gdGhyb3dJZkdQVU5vdEluaXRpYWxpemVkKCkge1xuICBpZiAoR1BHUFUgPT0gbnVsbCB8fCBURVhUVVJFX01BTkFHRVIgPT0gbnVsbCkge1xuICAgIHRocm93IG5ldyBFcnJvcignR1BVIG5vdCBpbnRpYWxpemVkLicpO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBOREFycmF5IHtcbiAgLyoqIFRoZSBzaGFwZSBvZiB0aGUgbmRhcnJheS4gKi9cbiAgc2hhcGU6IG51bWJlcltdO1xuICAvKiogTnVtYmVyIG9mIGVsZW1lbnRzIGluIHRoZSBuZGFycmF5LiAqL1xuICBzaXplOiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIE51bWJlciBvZiBlbGVtZW50cyB0byBza2lwIGluIGVhY2ggZGltZW5zaW9uIHdoZW4gaW5kZXhpbmcuIFNlZVxuICAgKiBodHRwczovL2RvY3Muc2NpcHkub3JnL2RvYy9udW1weS9yZWZlcmVuY2UvZ2VuZXJhdGVkXG4gICAqICAgICAvbnVtcHkubmRhcnJheS5zdHJpZGVzLmh0bWxcbiAgICovXG4gIHByb3RlY3RlZCBzdHJpZGVzOiBudW1iZXJbXTtcblxuICBwcml2YXRlIGRhdGE6IE5EQXJyYXlEYXRhO1xuXG4gIHByb3RlY3RlZCBjb25zdHJ1Y3RvcihzaGFwZTogbnVtYmVyW10sIGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgLy8gU2FuaXR5IGNoZWNrcy5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgZGF0YS52YWx1ZXMgIT0gbnVsbCB8fCBkYXRhLnRleHR1cmUgIT0gbnVsbCxcbiAgICAgICAgJ0VpdGhlciBgdmFsdWVzYCBvciBgdGV4dHVyZWAgbXVzdCBiZSBkZWZpbmVkJyk7XG5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgZGF0YS50ZXh0dXJlID09IG51bGwgfHwgKGRhdGEudGV4dHVyZVNoYXBlUkMgIT0gbnVsbCksXG4gICAgICAgICdgdGV4dHVyZVNoYXBlYCBtdXN0IGJlIGRlZmluZWQgd2hlbiBgdGV4dHVyZWAgaXMgZGVmaW5lZCcpO1xuXG4gICAgdGhpcy5zaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKTtcblxuICAgIGlmIChkYXRhLnZhbHVlcyAhPSBudWxsKSB7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICB0aGlzLnNpemUgPT09IGRhdGEudmFsdWVzLmxlbmd0aCxcbiAgICAgICAgICAnQ29uc3RydWN0aW5nIG5kYXJyYXkgb2Ygc2hhcGUgKCcgKyB0aGlzLnNpemUgKyAnKSBzaG91bGQgbWF0Y2ggdGhlJyArXG4gICAgICAgICAgICAgICcgbGVuZ3RoIG9mIHZhbHVlcyAoJyArIGRhdGEudmFsdWVzLmxlbmd0aCArICcpJyk7XG4gICAgfVxuXG4gICAgdGhpcy5zaGFwZSA9IHNoYXBlO1xuICAgIHRoaXMuZGF0YSA9IGRhdGE7XG4gICAgY29uc3QgZGltID0gdGhpcy5zaGFwZS5sZW5ndGg7XG5cbiAgICBpZiAoZGltIDwgMikge1xuICAgICAgdGhpcy5zdHJpZGVzID0gW107XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIExhc3QgZGltZW5zaW9uIGhhcyBpbXBsaWNpdCBzdHJpZGUgb2YgMSwgdGh1cyBoYXZpbmcgRC0xIChpbnN0ZWFkIG9mIEQpXG4gICAgICAvLyBzdHJpZGVzLlxuICAgICAgdGhpcy5zdHJpZGVzID0gbmV3IEFycmF5KGRpbSAtIDEpO1xuICAgICAgdGhpcy5zdHJpZGVzW2RpbSAtIDJdID0gdGhpcy5zaGFwZVtkaW0gLSAxXTtcbiAgICAgIGZvciAobGV0IGkgPSBkaW0gLSAzOyBpID49IDA7IC0taSkge1xuICAgICAgICB0aGlzLnN0cmlkZXNbaV0gPSB0aGlzLnN0cmlkZXNbaSArIDFdICogdGhpcy5zaGFwZVtpICsgMV07XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgLyoqIENyZWF0ZXMgYSBuZGFycmF5IG9mIHplcm9zIHdpdGggdGhlIHNwZWNpZmllZCBzaGFwZS4gKi9cbiAgc3RhdGljIHplcm9zKHNoYXBlOiBudW1iZXJbXSk6IE5EQXJyYXkge1xuICAgIGNvbnN0IHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkodXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKSk7XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZShzaGFwZSwge3ZhbHVlc30pO1xuICB9XG5cbiAgLyoqXG4gICAqIENyZWF0ZXMgYSBuZGFycmF5IG9mIHplcm9zIHdpdGggdGhlIHNhbWUgc2hhcGUgYXMgdGhlIHNwZWNpZmllZCBuZGFycmF5LlxuICAgKi9cbiAgc3RhdGljIHplcm9zTGlrZTxUIGV4dGVuZHMgTkRBcnJheT4oYW5vdGhlcjogVCk6IFQge1xuICAgIHJldHVybiBOREFycmF5Lnplcm9zKGFub3RoZXIuc2hhcGUpIGFzIFQ7XG4gIH1cblxuICAvKiogQ3JlYXRlcyBhIG5kYXJyYXkgd2l0aCB0aGUgc2FtZSB2YWx1ZXMvc2hhcGUgYXMgdGhlIHNwZWNpZmllZCBuZGFycmF5LiAqL1xuICBzdGF0aWMgbGlrZTxUIGV4dGVuZHMgTkRBcnJheT4oYW5vdGhlcjogVCk6IFQge1xuICAgIGNvbnN0IHZhbHVlcyA9IGFub3RoZXIuZ2V0VmFsdWVzKCk7XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihhbm90aGVyLnNoYXBlLCB7dmFsdWVzOiBuZXcgRmxvYXQzMkFycmF5KHZhbHVlcyl9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBNYWtlcyBhIG5ldyBuZGFycmF5IHdpdGggdGhlIHByb3ZpZGVkIHNoYXBlIGFuZCB2YWx1ZXMuIFZhbHVlcyBzaG91bGQgYmUgaW5cbiAgICogYSBmbGF0IGFycmF5LlxuICAgKi9cbiAgc3RhdGljIG1ha2U8VCBleHRlbmRzIE5EQXJyYXk+KHNoYXBlOiBudW1iZXJbXSwgZGF0YTogTkRBcnJheURhdGEpOiBUIHtcbiAgICBzd2l0Y2ggKHNoYXBlLmxlbmd0aCkge1xuICAgICAgY2FzZSAwOlxuICAgICAgICByZXR1cm4gbmV3IFNjYWxhcihkYXRhKSBhcyBUO1xuICAgICAgY2FzZSAxOlxuICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICAgIHJldHVybiBuZXcgQXJyYXkxRChkYXRhKSBhcyBhbnk7XG4gICAgICBjYXNlIDI6XG4gICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgcmV0dXJuIG5ldyBBcnJheTJEKHNoYXBlIGFzIFtudW1iZXIsIG51bWJlcl0sIGRhdGEpIGFzIGFueTtcbiAgICAgIGNhc2UgMzpcbiAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgICByZXR1cm4gbmV3IEFycmF5M0Qoc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBkYXRhKSBhcyBhbnk7XG4gICAgICBjYXNlIDQ6XG4gICAgICAgIHJldHVybiBuZXcgQXJyYXk0RChcbiAgICAgICAgICAgICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICAgICAgICAgICAgICAgc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGRhdGEpIGFzIGFueTtcbiAgICAgIGRlZmF1bHQ6XG4gICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgcmV0dXJuIG5ldyBOREFycmF5KHNoYXBlLCBkYXRhKSBhcyBhbnk7XG4gICAgfVxuICB9XG5cbiAgLyoqIFJlc2hhcGVzIHRoZSBjdXJyZW50IG5kYXJyYXkgaW50byB0aGUgcHJvdmlkZWQgc2hhcGUuICovXG4gIHJlc2hhcGU8VCBleHRlbmRzIE5EQXJyYXk+KG5ld1NoYXBlOiBudW1iZXJbXSk6IFQge1xuICAgIGlmICh1dGlsLmFycmF5c0VxdWFsKHRoaXMuc2hhcGUsIG5ld1NoYXBlKSkge1xuICAgICAgLy8gTm8tb3AuXG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICByZXR1cm4gdGhpcyBhcyBhbnk7XG4gICAgfVxuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHRoaXMuc2l6ZSA9PT0gdXRpbC5zaXplRnJvbVNoYXBlKG5ld1NoYXBlKSxcbiAgICAgICAgJ25ldyBzaGFwZSBhbmQgb2xkIHNoYXBlIG11c3QgaGF2ZSB0aGUgc2FtZSBudW1iZXIgb2YgZWxlbWVudHMuJyk7XG5cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5ld1NoYXBlLCB0aGlzLmRhdGEpO1xuICB9XG5cbiAgYXNTY2FsYXIoKTogU2NhbGFyIHtcbiAgICB1dGlsLmFzc2VydCh0aGlzLnNpemUgPT09IDEsICdUaGUgYXJyYXkgbXVzdCBoYXZlIG9ubHkgMSBlbGVtZW50LicpO1xuICAgIHJldHVybiB0aGlzLnJlc2hhcGU8U2NhbGFyPihbXSk7XG4gIH1cblxuICBhczFEKCk6IEFycmF5MUQge1xuICAgIHJldHVybiB0aGlzLnJlc2hhcGU8QXJyYXkxRD4oW3RoaXMuc2l6ZV0pO1xuICB9XG5cbiAgYXMyRChyb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IEFycmF5MkQge1xuICAgIHJldHVybiB0aGlzLnJlc2hhcGU8QXJyYXkyRD4oW3Jvd3MsIGNvbHVtbnNdKTtcbiAgfVxuXG4gIGFzM0Qocm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIsIGRlcHRoOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICByZXR1cm4gdGhpcy5yZXNoYXBlPEFycmF5M0Q+KFtyb3dzLCBjb2x1bW5zLCBkZXB0aF0pO1xuICB9XG5cbiAgYXM0RChyb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlciwgZGVwdGg6IG51bWJlciwgZGVwdGgyOiBudW1iZXIpOiBBcnJheTREIHtcbiAgICByZXR1cm4gdGhpcy5yZXNoYXBlPEFycmF5NEQ+KFtyb3dzLCBjb2x1bW5zLCBkZXB0aCwgZGVwdGgyXSk7XG4gIH1cblxuICBnZXQgcmFuaygpOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLnNoYXBlLmxlbmd0aDtcbiAgfVxuXG4gIGdldCguLi5sb2NzOiBudW1iZXJbXSkge1xuICAgIGxldCBpbmRleCA9IGxvY3NbbG9jcy5sZW5ndGggLSAxXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGxvY3MubGVuZ3RoIC0gMTsgKytpKSB7XG4gICAgICBpbmRleCArPSB0aGlzLnN0cmlkZXNbaV0gKiBsb2NzW2ldO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5nZXRWYWx1ZXMoKVtpbmRleF07XG4gIH1cblxuICBhZGQodmFsdWU6IG51bWJlciwgLi4ubG9jczogbnVtYmVyW10pIHtcbiAgICB0aGlzLnNldCh0aGlzLmdldCguLi5sb2NzKSArIHZhbHVlLCAuLi5sb2NzKTtcbiAgfVxuXG4gIHNldCh2YWx1ZTogbnVtYmVyLCAuLi5sb2NzOiBudW1iZXJbXSkge1xuICAgIGxldCBpbmRleCA9IGxvY3NbbG9jcy5sZW5ndGggLSAxXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGxvY3MubGVuZ3RoIC0gMTsgKytpKSB7XG4gICAgICBpbmRleCArPSB0aGlzLnN0cmlkZXNbaV0gKiBsb2NzW2ldO1xuICAgIH1cbiAgICB0aGlzLmdldFZhbHVlcygpW2luZGV4XSA9IHZhbHVlO1xuICB9XG5cbiAgbG9jVG9JbmRleChsb2NzOiBudW1iZXJbXSk6IG51bWJlciB7XG4gICAgbGV0IGluZGV4ID0gbG9jc1tsb2NzLmxlbmd0aCAtIDFdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbG9jcy5sZW5ndGggLSAxOyArK2kpIHtcbiAgICAgIGluZGV4ICs9IHRoaXMuc3RyaWRlc1tpXSAqIGxvY3NbaV07XG4gICAgfVxuICAgIHJldHVybiBpbmRleDtcbiAgfVxuXG4gIGluZGV4VG9Mb2MoaW5kZXg6IG51bWJlcik6IG51bWJlcltdIHtcbiAgICBjb25zdCBsb2NzOiBudW1iZXJbXSA9IG5ldyBBcnJheSh0aGlzLnNoYXBlLmxlbmd0aCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsb2NzLmxlbmd0aCAtIDE7ICsraSkge1xuICAgICAgbG9jc1tpXSA9IE1hdGguZmxvb3IoaW5kZXggLyB0aGlzLnN0cmlkZXNbaV0pO1xuICAgICAgaW5kZXggLT0gbG9jc1tpXSAqIHRoaXMuc3RyaWRlc1tpXTtcbiAgICB9XG4gICAgbG9jc1tsb2NzLmxlbmd0aCAtIDFdID0gaW5kZXg7XG4gICAgcmV0dXJuIGxvY3M7XG4gIH1cblxuICBmaWxsKHZhbHVlOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpLmZpbGwodmFsdWUpO1xuICB9XG5cbiAgZ2V0RGF0YSgpOiBOREFycmF5RGF0YSB7XG4gICAgcmV0dXJuIHRoaXMuZGF0YTtcbiAgfVxuXG4gIGdldFZhbHVlcygpOiBGbG9hdDMyQXJyYXkge1xuICAgIGlmICh0aGlzLmRhdGEudmFsdWVzID09IG51bGwpIHtcbiAgICAgIHRocm93SWZHUFVOb3RJbml0aWFsaXplZCgpO1xuICAgICAgdGhpcy5kYXRhLnZhbHVlcyA9IEdQR1BVLmRvd25sb2FkTWF0cml4RnJvbVRleHR1cmUoXG4gICAgICAgICAgdGhpcy5kYXRhLnRleHR1cmUsIHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQ1swXSxcbiAgICAgICAgICB0aGlzLmRhdGEudGV4dHVyZVNoYXBlUkNbMV0pO1xuICAgICAgdGhpcy5kaXNwb3NlVGV4dHVyZSgpO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5kYXRhLnZhbHVlcztcbiAgfVxuXG4gIHByaXZhdGUgdXBsb2FkVG9HUFUocHJlZmVycmVkVGV4U2hhcGU/OiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gICAgdGhyb3dJZkdQVU5vdEluaXRpYWxpemVkKCk7XG4gICAgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDID0gd2ViZ2xfdXRpbC5nZXRUZXh0dXJlU2hhcGVGcm9tTG9naWNhbFNoYXBlKFxuICAgICAgICBHUEdQVS5nbCwgdGhpcy5zaGFwZSwgcHJlZmVycmVkVGV4U2hhcGUpO1xuICAgIHRoaXMuZGF0YS50ZXh0dXJlID1cbiAgICAgICAgVEVYVFVSRV9NQU5BR0VSLmFjcXVpcmVUZXh0dXJlKHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQyk7XG5cbiAgICBHUEdQVS51cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgICAgIHRoaXMuZGF0YS50ZXh0dXJlLCB0aGlzLmRhdGEudGV4dHVyZVNoYXBlUkNbMF0sXG4gICAgICAgIHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQ1sxXSwgdGhpcy5kYXRhLnZhbHVlcyk7XG5cbiAgICB0aGlzLmRhdGEudmFsdWVzID0gbnVsbDtcbiAgfVxuXG4gIGdldFRleHR1cmUocHJlZmVycmVkU2hhcGVSQz86IFtudW1iZXIsIG51bWJlcl0pOiBXZWJHTFRleHR1cmUge1xuICAgIGlmICh0aGlzLmRhdGEudGV4dHVyZSA9PSBudWxsKSB7XG4gICAgICB0aGlzLnVwbG9hZFRvR1BVKHByZWZlcnJlZFNoYXBlUkMpO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5kYXRhLnRleHR1cmU7XG4gIH1cblxuICBnZXRUZXh0dXJlU2hhcGVSQyhwcmVmZXJyZWRTaGFwZVJDPzogW251bWJlciwgbnVtYmVyXSk6IFtudW1iZXIsIG51bWJlcl0ge1xuICAgIGlmICh0aGlzLmRhdGEudGV4dHVyZVNoYXBlUkMgPT0gbnVsbCkge1xuICAgICAgdGhpcy51cGxvYWRUb0dQVShwcmVmZXJyZWRTaGFwZVJDKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQztcbiAgfVxuXG4gIGRpc3Bvc2UoKTogdm9pZCB7XG4gICAgdGhpcy5kYXRhLnZhbHVlcyA9IG51bGw7XG4gICAgdGhpcy5zaGFwZSA9IG51bGw7XG4gICAgaWYgKHRoaXMuZGF0YS50ZXh0dXJlICE9IG51bGwpIHtcbiAgICAgIHRoaXMuZGlzcG9zZVRleHR1cmUoKTtcbiAgICB9XG4gIH1cblxuICBwcml2YXRlIGRpc3Bvc2VUZXh0dXJlKCkge1xuICAgIHRocm93SWZHUFVOb3RJbml0aWFsaXplZCgpO1xuICAgIFRFWFRVUkVfTUFOQUdFUi5yZWxlYXNlVGV4dHVyZSh0aGlzLmRhdGEudGV4dHVyZSwgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDKTtcbiAgICB0aGlzLmRhdGEudGV4dHVyZSA9IG51bGw7XG4gICAgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDID0gbnVsbDtcbiAgfVxuXG4gIGluR1BVKCk6IGJvb2xlYW4ge1xuICAgIHJldHVybiB0aGlzLmRhdGEudGV4dHVyZSAhPSBudWxsO1xuICB9XG5cbiAgZXF1YWxzKHQ6IE5EQXJyYXkpOiBib29sZWFuIHtcbiAgICByZXR1cm4gdXRpbC5hcnJheXNFcXVhbCh0aGlzLnNoYXBlLCB0LnNoYXBlKSAmJlxuICAgICAgICB1dGlsLmFycmF5c0VxdWFsKHRoaXMuZ2V0VmFsdWVzKCksIHQuZ2V0VmFsdWVzKCkpO1xuICB9XG5cbiAgc3RhdGljIHJhbmQ8VCBleHRlbmRzIE5EQXJyYXk+KHNoYXBlOiBudW1iZXJbXSwgcmFuZEZ1bmN0aW9uOiAoKSA9PiBudW1iZXIpOlxuICAgICAgVCB7XG4gICAgY29uc3Qgc2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZShzaGFwZSk7XG4gICAgY29uc3QgdmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShzaXplKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHNpemU7IGkrKykge1xuICAgICAgdmFsdWVzW2ldID0gcmFuZEZ1bmN0aW9uKCk7XG4gICAgfVxuXG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihzaGFwZSwge3ZhbHVlc30pO1xuICB9XG5cbiAgc3RhdGljIHJhbmROb3JtYWw8VCBleHRlbmRzIE5EQXJyYXk+KHNoYXBlOiBudW1iZXJbXSwgbWVhbiA9IDAsIHN0ZERldiA9IDEpIHtcbiAgICByZXR1cm4gTkRBcnJheS5yYW5kPFQ+KHNoYXBlLCAoKSA9PiB1dGlsLnJhbmRHYXVzcyhtZWFuLCBzdGREZXYpKTtcbiAgfVxuXG4gIHN0YXRpYyByYW5kVHJ1bmNhdGVkTm9ybWFsPFQgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIHNoYXBlOiBudW1iZXJbXSwgbWVhbiA9IDAsIHN0ZERldiA9IDEpIHtcbiAgICByZXR1cm4gTkRBcnJheS5yYW5kPFQ+KHNoYXBlLCAoKSA9PiB1dGlsLnJhbmRHYXVzcyhtZWFuLCBzdGREZXYsIHRydWUpKTtcbiAgfVxuXG4gIHN0YXRpYyByYW5kVW5pZm9ybTxUIGV4dGVuZHMgTkRBcnJheT4oc2hhcGU6IG51bWJlcltdLCBhOiBudW1iZXIsIGI6IG51bWJlcikge1xuICAgIHJldHVybiBOREFycmF5LnJhbmQ8VD4oc2hhcGUsICgpID0+IHV0aWwucmFuZFVuaWZvcm0oYSwgYikpO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBTY2FsYXIgZXh0ZW5kcyBOREFycmF5IHtcbiAgY29uc3RydWN0b3IoZGF0YTogTkRBcnJheURhdGEpIHtcbiAgICBpZiAoZGF0YS50ZXh0dXJlICE9IG51bGwpIHtcbiAgICAgIGRhdGEudGV4dHVyZVNoYXBlUkMgPSBbMSwgMV07XG4gICAgfVxuICAgIHN1cGVyKFtdLCBkYXRhKTtcbiAgfVxuXG4gIHN0YXRpYyBuZXcodmFsdWU6IG51bWJlcikge1xuICAgIHJldHVybiBuZXcgU2NhbGFyKHt2YWx1ZXM6IG5ldyBGbG9hdDMyQXJyYXkoW3ZhbHVlXSl9KTtcbiAgfVxuXG4gIHN0YXRpYyBaRVJPID0gU2NhbGFyLm5ldygwKTtcbiAgc3RhdGljIE9ORSA9IFNjYWxhci5uZXcoMSk7XG4gIHN0YXRpYyBUV08gPSBTY2FsYXIubmV3KDIpO1xuICBzdGF0aWMgTkVHX09ORSA9IFNjYWxhci5uZXcoLTEpO1xuXG4gIGdldCgpOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLmdldFZhbHVlcygpWzBdO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpWzBdID0gdmFsdWU7XG4gIH1cblxuICBhZGQodmFsdWU6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClbMF0gKz0gdmFsdWU7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIEFycmF5MUQgZXh0ZW5kcyBOREFycmF5IHtcbiAgc2hhcGU6IFtudW1iZXJdO1xuXG4gIGNvbnN0cnVjdG9yKGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgY29uc3Qgc2hhcGUgPSAoZGF0YS52YWx1ZXMgIT0gbnVsbCkgP1xuICAgICAgICBbZGF0YS52YWx1ZXMubGVuZ3RoXSA6XG4gICAgICAgIFt1dGlsLnNpemVGcm9tU2hhcGUoZGF0YS50ZXh0dXJlU2hhcGVSQyldO1xuICAgIHN1cGVyKHNoYXBlLCBkYXRhKTtcbiAgfVxuXG4gIHN0YXRpYyBuZXcodmFsdWVzOiBGbG9hdDMyQXJyYXl8bnVtYmVyW10pIHtcbiAgICBpZiAoISh2YWx1ZXMgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpKSB7XG4gICAgICBjb25zdCBpbmZlcnJlZFNoYXBlID0gdXRpbC5pbmZlclNoYXBlKHZhbHVlcyk7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICBpbmZlcnJlZFNoYXBlLmxlbmd0aCA9PT0gMSxcbiAgICAgICAgICBgRXJyb3IgY29uc3RydWN0aW5nIEFycmF5MUQuIFNoYXBlIG9mIHZhbHVlcyAke2luZmVycmVkU2hhcGV9IGlzIGAgK1xuICAgICAgICAgICAgICBgbm90IDEgZGltZW5zaW9uYWwuYCk7XG4gICAgfVxuICAgIHJldHVybiBuZXcgQXJyYXkxRCh7dmFsdWVzOiB0b1R5cGVkQXJyYXkodmFsdWVzKX0pO1xuICB9XG5cbiAgZ2V0KGk6IG51bWJlcik6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMuZ2V0VmFsdWVzKClbaV07XG4gIH1cblxuICBzZXQodmFsdWU6IG51bWJlciwgaTogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVtpXSA9IHZhbHVlO1xuICB9XG5cbiAgYWRkKHZhbHVlOiBudW1iZXIsIGk6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClbaV0gKz0gdmFsdWU7XG4gIH1cblxuICBsb2NUb0luZGV4KGxvYzogW251bWJlcl0pOiBudW1iZXIge1xuICAgIHJldHVybiBsb2NbMF07XG4gIH1cblxuICBpbmRleFRvTG9jKGluZGV4OiBudW1iZXIpOiBbbnVtYmVyXSB7XG4gICAgcmV0dXJuIFtpbmRleF07XG4gIH1cblxuICBzdGF0aWMgemVyb3Moc2hhcGU6IFtudW1iZXJdKTogQXJyYXkxRCB7XG4gICAgcmV0dXJuIE5EQXJyYXkuemVyb3Moc2hhcGUpIGFzIEFycmF5MUQ7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIEFycmF5MkQgZXh0ZW5kcyBOREFycmF5IHtcbiAgc2hhcGU6IFtudW1iZXIsIG51bWJlcl07XG5cbiAgcHJpdmF0ZSBzdHJpZGUwOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3Ioc2hhcGU6IFtudW1iZXIsIG51bWJlcl0sIGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgdXRpbC5hc3NlcnQoc2hhcGUubGVuZ3RoID09PSAyLCAnU2hhcGUgc2hvdWxkIGJlIG9mIGxlbmd0aCAyJyk7XG4gICAgc3VwZXIoc2hhcGUsIGRhdGEpO1xuICAgIHRoaXMuc3RyaWRlMCA9IHRoaXMuc3RyaWRlc1swXTtcbiAgfVxuXG4gIHN0YXRpYyBuZXcoXG4gICAgICBzaGFwZTogW251bWJlciwgbnVtYmVyXSwgdmFsdWVzOiBGbG9hdDMyQXJyYXl8bnVtYmVyW118bnVtYmVyW11bXSkge1xuICAgIGlmICghKHZhbHVlcyBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkpIHtcbiAgICAgIGNvbnN0IGluZmVycmVkU2hhcGUgPSB1dGlsLmluZmVyU2hhcGUodmFsdWVzKTtcbiAgICAgIGlmIChpbmZlcnJlZFNoYXBlLmxlbmd0aCA+IDEpIHtcbiAgICAgICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChcbiAgICAgICAgICAgIHNoYXBlLCBpbmZlcnJlZFNoYXBlLFxuICAgICAgICAgICAgYEVycm9yIHdoZW4gY29uc3RydWN0aW5nIEFycmF5MkQuIFNoYXBlIG9mIHZhbHVlcyBgICtcbiAgICAgICAgICAgICAgICBgJHtpbmZlcnJlZFNoYXBlfSBkb2VzIG5vdCBtYXRjaCB0aGUgcHJvdmlkZWQgc2hhcGUgYCArXG4gICAgICAgICAgICAgICAgYCR7c2hhcGV9LiBgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG5ldyBBcnJheTJEKHNoYXBlLCB7dmFsdWVzOiB0b1R5cGVkQXJyYXkodmFsdWVzKX0pO1xuICB9XG5cbiAgZ2V0KGk6IG51bWJlciwgajogbnVtYmVyKSB7XG4gICAgcmV0dXJuIHRoaXMuZ2V0VmFsdWVzKClbdGhpcy5zdHJpZGUwICogaSArIGpdO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVt0aGlzLnN0cmlkZTAgKiBpICsgal0gPSB2YWx1ZTtcbiAgfVxuXG4gIGFkZCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIsIGo6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClbdGhpcy5zdHJpZGUwICogaSArIGpdICs9IHZhbHVlO1xuICB9XG5cbiAgbG9jVG9JbmRleChsb2NzOiBbbnVtYmVyLCBudW1iZXJdKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5zdHJpZGUwICogbG9jc1swXSArIGxvY3NbMV07XG4gIH1cblxuICBpbmRleFRvTG9jKGluZGV4OiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgICByZXR1cm4gW01hdGguZmxvb3IoaW5kZXggLyB0aGlzLnN0cmlkZTApLCBpbmRleCAlIHRoaXMuc3RyaWRlMF07XG4gIH1cblxuICBzdGF0aWMgemVyb3Moc2hhcGU6IFtudW1iZXIsIG51bWJlcl0pOiBBcnJheTJEIHtcbiAgICByZXR1cm4gTkRBcnJheS56ZXJvcyhzaGFwZSkgYXMgQXJyYXkyRDtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgQXJyYXkzRCBleHRlbmRzIE5EQXJyYXkge1xuICBzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICBwcml2YXRlIHN0cmlkZTA6IG51bWJlcjtcbiAgcHJpdmF0ZSBzdHJpZGUxOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3Ioc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZGF0YTogTkRBcnJheURhdGEpIHtcbiAgICB1dGlsLmFzc2VydChzaGFwZS5sZW5ndGggPT09IDMsICdTaGFwZSBzaG91bGQgYmUgb2YgbGVuZ3RoIDMnKTtcbiAgICBzdXBlcihzaGFwZSwgZGF0YSk7XG4gICAgdGhpcy5zdHJpZGUwID0gdGhpcy5zdHJpZGVzWzBdO1xuICAgIHRoaXMuc3RyaWRlMSA9IHRoaXMuc3RyaWRlc1sxXTtcbiAgfVxuXG4gIHN0YXRpYyBuZXcoXG4gICAgICBzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgICAgdmFsdWVzOiBGbG9hdDMyQXJyYXl8bnVtYmVyW118bnVtYmVyW11bXVtdKSB7XG4gICAgaWYgKCEodmFsdWVzIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KSkge1xuICAgICAgY29uc3QgaW5mZXJyZWRTaGFwZSA9IHV0aWwuaW5mZXJTaGFwZSh2YWx1ZXMpO1xuICAgICAgaWYgKGluZmVycmVkU2hhcGUubGVuZ3RoID4gMSkge1xuICAgICAgICB1dGlsLmFzc2VydFNoYXBlc01hdGNoKFxuICAgICAgICAgICAgc2hhcGUsIGluZmVycmVkU2hhcGUsXG4gICAgICAgICAgICBgRXJyb3Igd2hlbiBjb25zdHJ1Y3RpbmcgQXJyYXkzRC4gU2hhcGUgb2YgdmFsdWVzIGAgK1xuICAgICAgICAgICAgICAgIGAke2luZmVycmVkU2hhcGV9IGRvZXMgbm90IG1hdGNoIHRoZSBwcm92aWRlZCBzaGFwZSBgICtcbiAgICAgICAgICAgICAgICBgJHtzaGFwZX0uIGApO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gbmV3IEFycmF5M0Qoc2hhcGUsIHt2YWx1ZXM6IHRvVHlwZWRBcnJheSh2YWx1ZXMpfSk7XG4gIH1cblxuICBnZXQoaTogbnVtYmVyLCBqOiBudW1iZXIsIGs6IG51bWJlcikge1xuICAgIHJldHVybiB0aGlzLmdldFZhbHVlcygpW3RoaXMuc3RyaWRlMCAqIGkgKyB0aGlzLnN0cmlkZTEgKiBqICsga107XG4gIH1cblxuICBzZXQodmFsdWU6IG51bWJlciwgaTogbnVtYmVyLCBqOiBudW1iZXIsIGs6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClbdGhpcy5zdHJpZGUwICogaSArIHRoaXMuc3RyaWRlMSAqIGogKyBrXSA9IHZhbHVlO1xuICB9XG5cbiAgYWRkKHZhbHVlOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyLCBrOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpW3RoaXMuc3RyaWRlMCAqIGkgKyB0aGlzLnN0cmlkZTEgKiBqICsga10gKz0gdmFsdWU7XG4gIH1cblxuICBsb2NUb0luZGV4KGxvY3M6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSk6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMuc3RyaWRlMCAqIGxvY3NbMF0gKyB0aGlzLnN0cmlkZTEgKiBsb2NzWzFdICsgbG9jc1syXTtcbiAgfVxuXG4gIGluZGV4VG9Mb2MoaW5kZXg6IG51bWJlcik6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSB7XG4gICAgY29uc3QgaSA9IE1hdGguZmxvb3IoaW5kZXggLyB0aGlzLnN0cmlkZTApO1xuICAgIGluZGV4IC09IGkgKiB0aGlzLnN0cmlkZTA7XG4gICAgcmV0dXJuIFtpLCBNYXRoLmZsb29yKGluZGV4IC8gdGhpcy5zdHJpZGUxKSwgaW5kZXggJSB0aGlzLnN0cmlkZTFdO1xuICB9XG5cbiAgc3RhdGljIHplcm9zKHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pOiBBcnJheTNEIHtcbiAgICByZXR1cm4gTkRBcnJheS56ZXJvcyhzaGFwZSkgYXMgQXJyYXkzRDtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgQXJyYXk0RCBleHRlbmRzIE5EQXJyYXkge1xuICBzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIHByaXZhdGUgc3RyaWRlMDogbnVtYmVyO1xuICBwcml2YXRlIHN0cmlkZTE6IG51bWJlcjtcbiAgcHJpdmF0ZSBzdHJpZGUyOiBudW1iZXI7XG5cbiAgY29uc3RydWN0b3Ioc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLCBkYXRhOiBOREFycmF5RGF0YSkge1xuICAgIHV0aWwuYXNzZXJ0KHNoYXBlLmxlbmd0aCA9PT0gNCwgJ1NoYXBlIHNob3VsZCBiZSBvZiBsZW5ndGggNCcpO1xuICAgIHN1cGVyKHNoYXBlLCBkYXRhKTtcbiAgICB0aGlzLnN0cmlkZTAgPSB0aGlzLnN0cmlkZXNbMF07XG4gICAgdGhpcy5zdHJpZGUxID0gdGhpcy5zdHJpZGVzWzFdO1xuICAgIHRoaXMuc3RyaWRlMiA9IHRoaXMuc3RyaWRlc1syXTtcbiAgfVxuXG4gIHN0YXRpYyBuZXcoXG4gICAgICBzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sXG4gICAgICB2YWx1ZXM6IEZsb2F0MzJBcnJheXxudW1iZXJbXXxudW1iZXJbXVtdW11bXSkge1xuICAgIGlmICghKHZhbHVlcyBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkpIHtcbiAgICAgIGNvbnN0IGluZmVycmVkU2hhcGUgPSB1dGlsLmluZmVyU2hhcGUodmFsdWVzKTtcbiAgICAgIGlmIChpbmZlcnJlZFNoYXBlLmxlbmd0aCA+IDEpIHtcbiAgICAgICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChcbiAgICAgICAgICAgIHNoYXBlLCBpbmZlcnJlZFNoYXBlLFxuICAgICAgICAgICAgYEVycm9yIHdoZW4gY29uc3RydWN0aW5nIEFycmF5NEQuIFNoYXBlIG9mIHZhbHVlcyBgICtcbiAgICAgICAgICAgICAgICBgJHtpbmZlcnJlZFNoYXBlfSBkb2VzIG5vdCBtYXRjaCB0aGUgcHJvdmlkZWQgc2hhcGUgYCArXG4gICAgICAgICAgICAgICAgYCR7c2hhcGV9LiBgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG5ldyBBcnJheTREKHNoYXBlLCB7dmFsdWVzOiB0b1R5cGVkQXJyYXkodmFsdWVzKX0pO1xuICB9XG5cbiAgZ2V0KGk6IG51bWJlciwgajogbnVtYmVyLCBrOiBudW1iZXIsIGw6IG51bWJlcikge1xuICAgIHJldHVybiB0aGlzLmdldFZhbHVlcygpXG4gICAgICAgIFt0aGlzLnN0cmlkZTAgKiBpICsgdGhpcy5zdHJpZGUxICogaiArIHRoaXMuc3RyaWRlMiAqIGsgKyBsXTtcbiAgfVxuXG4gIHNldCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIsIGo6IG51bWJlciwgazogbnVtYmVyLCBsOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpXG4gICAgICAgIFt0aGlzLnN0cmlkZTAgKiBpICsgdGhpcy5zdHJpZGUxICogaiArIHRoaXMuc3RyaWRlMiAqIGsgKyBsXSA9IHZhbHVlO1xuICB9XG5cbiAgYWRkKHZhbHVlOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyLCBrOiBudW1iZXIsIGw6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClcbiAgICAgICAgW3RoaXMuc3RyaWRlMCAqIGkgKyB0aGlzLnN0cmlkZTEgKiBqICsgdGhpcy5zdHJpZGUyICogayArIGxdICs9IHZhbHVlO1xuICB9XG5cbiAgbG9jVG9JbmRleChsb2NzOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSk6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMuc3RyaWRlMCAqIGxvY3NbMF0gKyB0aGlzLnN0cmlkZTEgKiBsb2NzWzFdICtcbiAgICAgICAgdGhpcy5zdHJpZGUyICogbG9jc1syXSArIGxvY3NbM107XG4gIH1cblxuICBpbmRleFRvTG9jKGluZGV4OiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSB7XG4gICAgY29uc3QgaSA9IE1hdGguZmxvb3IoaW5kZXggLyB0aGlzLnN0cmlkZTApO1xuICAgIGluZGV4IC09IGkgKiB0aGlzLnN0cmlkZTA7XG4gICAgY29uc3QgaiA9IE1hdGguZmxvb3IoaW5kZXggLyB0aGlzLnN0cmlkZTEpO1xuICAgIGluZGV4IC09IGogKiB0aGlzLnN0cmlkZTE7XG4gICAgcmV0dXJuIFtpLCBqLCBNYXRoLmZsb29yKGluZGV4IC8gdGhpcy5zdHJpZGUyKSwgaW5kZXggJSB0aGlzLnN0cmlkZTJdO1xuICB9XG5cbiAgc3RhdGljIHplcm9zKHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSk6IEFycmF5NEQge1xuICAgIHJldHVybiBOREFycmF5Lnplcm9zKHNoYXBlKSBhcyBBcnJheTREO1xuICB9XG59XG5cbnR5cGUgQXJyYXlEYXRhID0gRmxvYXQzMkFycmF5fG51bWJlcltdfG51bWJlcltdW118bnVtYmVyW11bXVtdfG51bWJlcltdW11bXVtdO1xuXG5mdW5jdGlvbiB0b1R5cGVkQXJyYXkoYTogQXJyYXlEYXRhKTogRmxvYXQzMkFycmF5IHtcbiAgcmV0dXJuIChhIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KSA/IGEgOiBuZXcgRmxvYXQzMkFycmF5KHV0aWwuZmxhdHRlbihhKSk7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIGNvbnZfdXRpbCBmcm9tICcuLi9jb252X3V0aWwnO1xuaW1wb3J0IHtHUEdQVVByb2dyYW19IGZyb20gJy4vZ3BncHVfbWF0aCc7XG5cbmV4cG9ydCBjbGFzcyBDb252MkREZXJXZWlnaHRzUHJvZ3JhbSBpbXBsZW1lbnRzIEdQR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnLCAnZHknXTtcbiAgcGFyYW1zOiBBcnJheTx7fT47XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgdXNlckNvZGU6IHN0cmluZztcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIHhTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBmU2l6ZTogbnVtYmVyLCBvdXRwdXREZXB0aDogbnVtYmVyLFxuICAgICAgc3RyaWRlOiBudW1iZXIsIHplcm9QYWQ6IG51bWJlcikge1xuICAgIGNvbnN0IHlTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgICAgeFNoYXBlLCBmU2l6ZSwgb3V0cHV0RGVwdGgsIHN0cmlkZSwgemVyb1BhZCk7XG4gICAgY29uc3QgeU51bVJvd3MgPSB5U2hhcGVbMF07XG4gICAgY29uc3QgeU51bUNvbHMgPSB5U2hhcGVbMV07XG4gICAgY29uc3QgeE51bVJvd3MgPSB4U2hhcGVbMF07XG4gICAgY29uc3QgeE51bUNvbHMgPSB4U2hhcGVbMV07XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9XG4gICAgICAgIGNvbnZfdXRpbC5jb21wdXRlV2VpZ2h0c1NoYXBlNEQoeFNoYXBlWzJdLCBvdXRwdXREZXB0aCwgZlNpemUpO1xuICAgIHRoaXMucGFyYW1zID0gW3N0cmlkZSwgemVyb1BhZF07XG4gICAgdGhpcy51c2VyQ29kZSA9IGBcbiAgICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgdmVjNCBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgZmxvYXQgd1IgPSBjb29yZHMueDtcbiAgICAgICAgZmxvYXQgd0MgPSBjb29yZHMueTtcbiAgICAgICAgZmxvYXQgZDEgPSBjb29yZHMuejtcbiAgICAgICAgZmxvYXQgZDIgPSBjb29yZHMudztcblxuICAgICAgICAvLyBDb252b2x2ZSB4KD8sID8sIGQxKSB3aXRoIGR5KDosIDosIGQyKSB0byBnZXQgZHcod1IsIHdDLCBkMSwgZDIpLlxuICAgICAgICAvLyA/ID0gdG8gYmUgZGV0ZXJtaW5lZC4gOiA9IGFjcm9zcyBhbGwgdmFsdWVzIGluIHRoYXQgYXhpcy5cbiAgICAgICAgZmxvYXQgZG90UHJvZCA9IDAuMDtcbiAgICAgICAgZm9yIChpbnQgaXlSID0gMDsgaXlSIDwgJHt5TnVtUm93c307IGl5UisrKSB7XG4gICAgICAgICAgZmxvYXQgeVIgPSBmbG9hdChpeVIpO1xuICAgICAgICAgIGZsb2F0IHhSID0gd1IgKyB5UiAqICR7c3RyaWRlfS4wIC0gJHt6ZXJvUGFkfS4wO1xuXG4gICAgICAgICAgaWYgKHhSIDwgMC4wIHx8IHhSID49ICR7eE51bVJvd3N9LjApIHtcbiAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgIH1cblxuICAgICAgICAgIGZvciAoaW50IGl5QyA9IDA7IGl5QyA8ICR7eU51bUNvbHN9OyBpeUMrKykge1xuICAgICAgICAgICAgZmxvYXQgeUMgPSBmbG9hdChpeUMpO1xuICAgICAgICAgICAgZmxvYXQgeEMgPSB3QyArIHlDICogJHtzdHJpZGV9LjAgLSAke3plcm9QYWR9LjA7XG5cbiAgICAgICAgICAgIGlmICh4QyA8IDAuMCB8fCB4QyA+PSAke3hOdW1Db2xzfS4wKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBmbG9hdCBkeVZhbHVlID0gZ2V0RHkoeVIsIHlDLCBkMik7XG4gICAgICAgICAgICBmbG9hdCB4VmFsdWUgPSBnZXRYKHhSLCB4QywgZDEpO1xuICAgICAgICAgICAgZG90UHJvZCArPSAoeFZhbHVlICogZHlWYWx1ZSk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHNldE91dHB1dChkb3RQcm9kKTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBDb252MkRUcmFuc3Bvc2VQcm9ncmFtIGltcGxlbWVudHMgR1BHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lcyA9IFsneCcsICdXJywgJ2JpYXMnXTtcbiAgcGFyYW1zOiBBcnJheTx7fT47XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgdXNlckNvZGU6IHN0cmluZztcblxuICBjb25zdHJ1Y3RvcihcbiAgICAgIHhTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBmU2l6ZTogbnVtYmVyLCBvcmlnSW5wdXREZXB0aDogbnVtYmVyLFxuICAgICAgb3JpZ1N0cmlkZTogbnVtYmVyLCBvcmlnUGFkOiBudW1iZXIsIGhhc0JpYXM6IGJvb2xlYW4pIHtcbiAgICBjb25zdCBbeFJvd3MsIHhDb2xzLCBvcmlnT3V0cHV0RGVwdGhdID0geFNoYXBlO1xuICAgIGNvbnN0IGJpYXNTbmlwcGV0ID0gaGFzQmlhcyA/ICdkb3RQcm9kICs9IGdldEJpYXMoZDIpOycgOiAnJztcblxuICAgIC8vIEZpZ3VyZSBvdXQgdGhlIG91dHB1dCBzaGFwZSBieSBkaWxhdGluZyB0aGUgaW5wdXQuXG4gICAgY29uc3QgeFJvd3NEaWxhdGVkID0gKHhSb3dzIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcbiAgICBjb25zdCB4Q29sc0RpbGF0ZWQgPSAoeENvbHMgLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuICAgIGNvbnN0IHBhZCA9IGZTaXplIC0gMSAtIG9yaWdQYWQ7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgICAgW3hSb3dzRGlsYXRlZCwgeENvbHNEaWxhdGVkLCBvcmlnT3V0cHV0RGVwdGhdLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGgsIDEsXG4gICAgICAgIHBhZCk7XG4gICAgdGhpcy5wYXJhbXMgPSBbcGFkLCBmU2l6ZSwgb3JpZ1N0cmlkZSwgaGFzQmlhc107XG5cbiAgICB0aGlzLnVzZXJDb2RlID0gYFxuICAgICAgdm9pZCBtYWluKCkge1xuICAgICAgICB2ZWMzIGNvb3JkcyA9IGdldE91dHB1dENvb3JkcygpO1xuICAgICAgICBmbG9hdCB5UiA9IGNvb3Jkcy54O1xuICAgICAgICBmbG9hdCB5QyA9IGNvb3Jkcy55O1xuICAgICAgICBmbG9hdCBkMiA9IGNvb3Jkcy56O1xuXG4gICAgICAgIHZlYzIgeFJDQ29ybmVyID0gdmVjMih5UiwgeUMpIC0gdmVjMigke3BhZH0uMCwgJHtwYWR9LjApO1xuICAgICAgICBmbG9hdCB4UkNvcm5lciA9IHhSQ0Nvcm5lci54O1xuICAgICAgICBmbG9hdCB4Q0Nvcm5lciA9IHhSQ0Nvcm5lci55O1xuXG4gICAgICAgIC8vIENvbnZvbHZlIHgoPywgPywgZDEpIHdpdGggdyg6LCA6LCBkMiwgZDEpIHRvIGdldCB5KHlSLCB5QywgZDIpLlxuICAgICAgICAvLyA/ID0gdG8gYmUgZGV0ZXJtaW5lZC4gOiA9IGFjcm9zcyBhbGwgdmFsdWVzIGluIHRoYXQgYXhpcy5cbiAgICAgICAgZmxvYXQgZG90UHJvZCA9IDAuMDtcbiAgICAgICAgZm9yIChpbnQgaXdSID0gMDsgaXdSIDwgJHtmU2l6ZX07IGl3UisrKSB7XG4gICAgICAgICAgZmxvYXQgd1IgPSBmbG9hdChpd1IpO1xuICAgICAgICAgIGZsb2F0IHhSID0gKHhSQ29ybmVyICsgd1IpIC8gJHtvcmlnU3RyaWRlfS4wO1xuXG4gICAgICAgICAgaWYgKHhSIDwgMC4wIHx8IHhSID49ICR7eFJvd3N9LjAgfHwgZnJhY3QoeFIpID4gMC4wKSB7XG4gICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBmbG9hdCB3UlBlcm0gPSAke2ZTaXplfS4wIC0gMS4wIC0gd1I7XG5cbiAgICAgICAgICBmb3IgKGludCBpd0MgPSAwOyBpd0MgPCAke2ZTaXplfTsgaXdDKyspIHtcbiAgICAgICAgICAgIGZsb2F0IHdDID0gZmxvYXQoaXdDKTtcbiAgICAgICAgICAgIGZsb2F0IHhDID0gKHhDQ29ybmVyICsgd0MpIC8gJHtvcmlnU3RyaWRlfS4wO1xuXG4gICAgICAgICAgICBpZiAoeEMgPCAwLjAgfHwgeEMgPj0gJHt4Q29sc30uMCB8fCBmcmFjdCh4QykgPiAwLjApIHtcbiAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGZsb2F0IHdDUGVybSA9ICR7ZlNpemV9LjAgLSAxLjAgLSB3QztcblxuICAgICAgICAgICAgZm9yIChpbnQgaWQxID0gMDsgaWQxIDwgJHtvcmlnT3V0cHV0RGVwdGh9OyBpZDErKykge1xuICAgICAgICAgICAgICBmbG9hdCBkMSA9IGZsb2F0KGlkMSk7XG4gICAgICAgICAgICAgIGZsb2F0IHhWYWx1ZSA9IGdldFgoeFIsIHhDLCBkMSk7XG4gICAgICAgICAgICAgIGZsb2F0IHdWYWx1ZSA9IGdldFcod1JQZXJtLCB3Q1Blcm0sIGQyLCBkMSk7XG4gICAgICAgICAgICAgIGRvdFByb2QgKz0geFZhbHVlICogd1ZhbHVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICAke2JpYXNTbmlwcGV0fVxuICAgICAgICBzZXRPdXRwdXQoZG90UHJvZCk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgQ29udjJERGVyQmlhc1Byb2dyYW0gaW1wbGVtZW50cyBHUEdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWydkeSddO1xuICBwYXJhbXM6IEFycmF5PHt9PiA9IFtdO1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHVzZXJDb2RlOiBzdHJpbmc7XG5cbiAgY29uc3RydWN0b3IoeVNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pIHtcbiAgICBjb25zdCBbeU51bVJvd3MsIHlOdW1Db2xzLCBvdXRwdXREZXB0aF0gPSB5U2hhcGU7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IFtvdXRwdXREZXB0aF07XG4gICAgdGhpcy51c2VyQ29kZSA9IGBcbiAgICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgZmxvYXQgZDIgPSBnZXRPdXRwdXRDb29yZHMoKTtcblxuICAgICAgICBmbG9hdCBkZXJCaWFzID0gMC4wO1xuICAgICAgICBmb3IgKGludCBpeVIgPSAwOyBpeVIgPCAke3lOdW1Sb3dzfTsgaXlSKyspIHtcbiAgICAgICAgICBmbG9hdCB5UiA9IGZsb2F0KGl5Uik7XG4gICAgICAgICAgZm9yIChpbnQgaXlDID0gMDsgaXlDIDwgJHt5TnVtQ29sc307IGl5QysrKSB7XG4gICAgICAgICAgICBmbG9hdCB5QyA9IGZsb2F0KGl5Qyk7XG4gICAgICAgICAgICBkZXJCaWFzICs9IGdldER5KHlSLCB5QywgZDIpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBzZXRPdXRwdXQoZGVyQmlhcyk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge091dHB1dEluZm99IGZyb20gJy4uL2NvbnZfdXRpbCc7XG5pbXBvcnQge0dQR1BVUHJvZ3JhbX0gZnJvbSAnLi9ncGdwdV9tYXRoJztcblxuZXhwb3J0IGNsYXNzIENvbnYyRFByb2dyYW0gaW1wbGVtZW50cyBHUEdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWyd4JywgJ1cnLCAnYmlhcyddO1xuICBwYXJhbXM6IEFycmF5PHt9PjtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICB1c2VyQ29kZTogc3RyaW5nO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgeFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZpbHRlckhlaWdodDogbnVtYmVyLFxuICAgICAgZmlsdGVyV2lkdGg6IG51bWJlciwgc3RyaWRlSGVpZ2h0OiBudW1iZXIsIHN0cmlkZVdpZHRoOiBudW1iZXIsXG4gICAgICBvdXRwdXRJbmZvOiBPdXRwdXRJbmZvLCBoYXNCaWFzOiBib29sZWFuKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IG91dHB1dEluZm8uc2hhcGU7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IHhTaGFwZVsyXTtcbiAgICB0aGlzLnBhcmFtcyA9XG4gICAgICAgIFtmaWx0ZXJXaWR0aCwgZmlsdGVySGVpZ2h0LCBzdHJpZGVIZWlnaHQsIHN0cmlkZVdpZHRoLCBoYXNCaWFzXTtcbiAgICBjb25zdCBiaWFzU25pcHBldCA9IGhhc0JpYXMgPyAnZG90UHJvZCArPSBnZXRCaWFzKGQyKTsnIDogJyc7XG4gICAgY29uc3QgeE51bVJvd3MgPSB4U2hhcGVbMF07XG4gICAgY29uc3QgeE51bUNvbHMgPSB4U2hhcGVbMV07XG4gICAgY29uc3QgcGFkVG9wID0gb3V0cHV0SW5mby5wYWRkaW5nSW5mby50b3A7XG4gICAgY29uc3QgcGFkTGVmdCA9IG91dHB1dEluZm8ucGFkZGluZ0luZm8ubGVmdDtcblxuICAgIHRoaXMudXNlckNvZGUgPSBgXG4gICAgICBjb25zdCB2ZWMyIHN0cmlkZXMgPSB2ZWMyKCR7c3RyaWRlSGVpZ2h0fS4wLCAke3N0cmlkZVdpZHRofS4wKTtcbiAgICAgIGNvbnN0IHZlYzIgcGFkcyA9IHZlYzIoJHtwYWRUb3B9LjAsICR7cGFkTGVmdH0uMCk7XG5cbiAgICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgdmVjMyBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgZmxvYXQgeVIgPSBjb29yZHMueDtcbiAgICAgICAgZmxvYXQgeUMgPSBjb29yZHMueTtcbiAgICAgICAgZmxvYXQgZDIgPSBjb29yZHMuejtcblxuICAgICAgICB2ZWMyIHhSQ0Nvcm5lciA9IHZlYzIoeVIsIHlDKSAqIHN0cmlkZXMgLSBwYWRzO1xuICAgICAgICBmbG9hdCB4UkNvcm5lciA9IHhSQ0Nvcm5lci54O1xuICAgICAgICBmbG9hdCB4Q0Nvcm5lciA9IHhSQ0Nvcm5lci55O1xuXG4gICAgICAgIC8vIENvbnZvbHZlIHgoPywgPywgZDEpIHdpdGggdyg6LCA6LCBkMSwgZDIpIHRvIGdldCB5KHlSLCB5QywgZDIpLlxuICAgICAgICAvLyA/ID0gdG8gYmUgZGV0ZXJtaW5lZC4gOiA9IGFjcm9zcyBhbGwgdmFsdWVzIGluIHRoYXQgYXhpcy5cbiAgICAgICAgZmxvYXQgZG90UHJvZCA9IDAuMDtcbiAgICAgICAgZm9yIChpbnQgaXdSID0gMDsgaXdSIDwgJHtmaWx0ZXJIZWlnaHR9OyBpd1IrKykge1xuICAgICAgICAgIGZsb2F0IHdSID0gZmxvYXQoaXdSKTtcbiAgICAgICAgICBmbG9hdCB4UiA9IHhSQ29ybmVyICsgd1I7XG5cbiAgICAgICAgICBpZiAoeFIgPCAwLjAgfHwgeFIgPj0gJHt4TnVtUm93c30uMCkge1xuICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgZm9yIChpbnQgaXdDID0gMDsgaXdDIDwgJHtmaWx0ZXJXaWR0aH07IGl3QysrKSB7XG4gICAgICAgICAgICBmbG9hdCB3QyA9IGZsb2F0KGl3Qyk7XG4gICAgICAgICAgICBmbG9hdCB4QyA9IHhDQ29ybmVyICsgd0M7XG5cbiAgICAgICAgICAgIGlmICh4QyA8IDAuMCB8fCB4QyA+PSAke3hOdW1Db2xzfS4wKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBmb3IgKGludCBpZDEgPSAwOyBpZDEgPCAke2lucHV0RGVwdGh9OyBpZDErKykge1xuICAgICAgICAgICAgICBmbG9hdCBkMSA9IGZsb2F0KGlkMSk7XG4gICAgICAgICAgICAgIGZsb2F0IHhWYWx1ZSA9IGdldFgoeFIsIHhDLCBkMSk7XG4gICAgICAgICAgICAgIGZsb2F0IHdWYWx1ZSA9IGdldFcod1IsIHdDLCBkMSwgZDIpO1xuICAgICAgICAgICAgICBkb3RQcm9kICs9IHhWYWx1ZSAqIHdWYWx1ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgJHtiaWFzU25pcHBldH1cbiAgICAgICAgc2V0T3V0cHV0KGRvdFByb2QpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgZ3BncHVfdXRpbCBmcm9tICcuL2dwZ3B1X3V0aWwnO1xuaW1wb3J0ICogYXMgdGV4X3V0aWwgZnJvbSAnLi90ZXhfdXRpbCc7XG5pbXBvcnQgKiBhcyB3ZWJnbF91dGlsIGZyb20gJy4vd2ViZ2xfdXRpbCc7XG5cbmltcG9ydCB7V2ViR0xMb3NlQ29udGV4dEV4dGVuc2lvbn0gZnJvbSAnLi93ZWJnbF91dGlsJztcblxuZXhwb3J0IGNsYXNzIEdQR1BVQ29udGV4dCB7XG4gIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQ7XG4gIHRleHR1cmVGbG9hdEV4dGVuc2lvbjoge307XG4gIGNvbG9yQnVmZmVyRmxvYXRFeHRlbnNpb246IHt9O1xuICBsb3NlQ29udGV4dEV4dGVuc2lvbjogV2ViR0xMb3NlQ29udGV4dEV4dGVuc2lvbjtcbiAgdmVydGV4QnVmZmVyOiBXZWJHTEJ1ZmZlcjtcbiAgaW5kZXhCdWZmZXI6IFdlYkdMQnVmZmVyO1xuICBmcmFtZWJ1ZmZlcjogV2ViR0xGcmFtZWJ1ZmZlcjtcbiAgb3V0cHV0VGV4dHVyZTogV2ViR0xUZXh0dXJlfG51bGwgPSBudWxsO1xuICBwcm9ncmFtOiBXZWJHTFByb2dyYW18bnVsbCA9IG51bGw7XG4gIHByaXZhdGUgZGlzcG9zZWQgPSBmYWxzZTtcbiAgcHJpdmF0ZSBhdXRvRGVidWdWYWxpZGF0ZSA9IGZhbHNlO1xuXG4gIGNvbnN0cnVjdG9yKGdsPzogV2ViR0xSZW5kZXJpbmdDb250ZXh0KSB7XG4gICAgaWYgKGdsICE9IG51bGwpIHtcbiAgICAgIHRoaXMuZ2wgPSBnbDtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5nbCA9IGdwZ3B1X3V0aWwuY3JlYXRlV2ViR0xDb250ZXh0KCk7XG4gICAgfVxuXG4gICAgLy8gV2ViR0wgMi4wIGVuYWJsZXMgdGV4dHVyZSBmbG9hdHMgd2l0aG91dCBhbiBleHRlbnNpb24uXG4gICAgaWYgKCF3ZWJnbF91dGlsLmlzV2ViR0wyRW5hYmxlZCgpKSB7XG4gICAgICB0aGlzLnRleHR1cmVGbG9hdEV4dGVuc2lvbiA9XG4gICAgICAgICAgd2ViZ2xfdXRpbC5nZXRFeHRlbnNpb25PclRocm93KHRoaXMuZ2wsICdPRVNfdGV4dHVyZV9mbG9hdCcpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmNvbG9yQnVmZmVyRmxvYXRFeHRlbnNpb24gPVxuICAgICAgICAgIHdlYmdsX3V0aWwuZ2V0RXh0ZW5zaW9uT3JUaHJvdyh0aGlzLmdsLCAnRVhUX2NvbG9yX2J1ZmZlcl9mbG9hdCcpO1xuICAgIH1cblxuICAgIHRoaXMubG9zZUNvbnRleHRFeHRlbnNpb24gPVxuICAgICAgICB3ZWJnbF91dGlsLmdldEV4dGVuc2lvbk9yVGhyb3codGhpcy5nbCwgJ1dFQkdMX2xvc2VfY29udGV4dCcpIGFzXG4gICAgICAgIFdlYkdMTG9zZUNvbnRleHRFeHRlbnNpb247XG4gICAgdGhpcy52ZXJ0ZXhCdWZmZXIgPSBncGdwdV91dGlsLmNyZWF0ZVZlcnRleEJ1ZmZlcih0aGlzLmdsKTtcbiAgICB0aGlzLmluZGV4QnVmZmVyID0gZ3BncHVfdXRpbC5jcmVhdGVJbmRleEJ1ZmZlcih0aGlzLmdsKTtcbiAgICB0aGlzLmZyYW1lYnVmZmVyID0gd2ViZ2xfdXRpbC5jcmVhdGVGcmFtZWJ1ZmZlcih0aGlzLmdsKTtcbiAgfVxuXG4gIHB1YmxpYyBkaXNwb3NlKCkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgaWYgKHRoaXMucHJvZ3JhbSAhPSBudWxsKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgJ0Rpc3Bvc2luZyBhIEdQR1BVQ29udGV4dCB0aGF0IHN0aWxsIGhhcyBhIGJvdW5kIFdlYkdMUHJvZ3JhbS4nICtcbiAgICAgICAgICAnIFRoaXMgaXMgcHJvYmFibHkgYSByZXNvdXJjZSBsZWFrLCBkZWxldGUgdGhlIHByb2dyYW0gd2l0aCAnICtcbiAgICAgICAgICAnR1BHUFVDb250ZXh0LmRlbGV0ZVByb2dyYW0gYmVmb3JlIGRpc3Bvc2luZy4nKTtcbiAgICB9XG4gICAgaWYgKHRoaXMub3V0cHV0VGV4dHVyZSAhPSBudWxsKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgJ0Rpc3Bvc2luZyBhIEdQR1BVQ29udGV4dCB0aGF0IHN0aWxsIGhhcyBhIGJvdW5kIG91dHB1dCBtYXRyaXggJyArXG4gICAgICAgICAgJ3RleHR1cmUuICBUaGlzIGlzIHByb2JhYmx5IGEgcmVzb3VyY2UgbGVhaywgZGVsZXRlIHRoZSBvdXRwdXQgJyArXG4gICAgICAgICAgJ21hdHJpeCB0ZXh0dXJlIHdpdGggR1BHUFVDb250ZXh0LmRlbGV0ZU1hdHJpeFRleHR1cmUgYmVmb3JlICcgK1xuICAgICAgICAgICdkaXNwb3NpbmcuJyk7XG4gICAgfVxuICAgIGNvbnN0IGdsID0gdGhpcy5nbDtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZmluaXNoKCkpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIG51bGwpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGVsZXRlRnJhbWVidWZmZXIodGhpcy5mcmFtZWJ1ZmZlcikpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kQnVmZmVyKGdsLkFSUkFZX0JVRkZFUiwgbnVsbCkpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kZWxldGVCdWZmZXIodGhpcy52ZXJ0ZXhCdWZmZXIpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgICAgZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuRUxFTUVOVF9BUlJBWV9CVUZGRVIsIG51bGwpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGVsZXRlQnVmZmVyKHRoaXMuaW5kZXhCdWZmZXIpKTtcbiAgICB0aGlzLmxvc2VDb250ZXh0RXh0ZW5zaW9uLmxvc2VDb250ZXh0KCk7XG4gICAgdGhpcy5kaXNwb3NlZCA9IHRydWU7XG4gIH1cblxuICBwdWJsaWMgZW5hYmxlQXV0b21hdGljRGVidWdWYWxpZGF0aW9uKGVuYWJsZWQ6IGJvb2xlYW4pIHtcbiAgICB0aGlzLmF1dG9EZWJ1Z1ZhbGlkYXRlID0gZW5hYmxlZDtcbiAgICB3ZWJnbF91dGlsLmVuYWJsZURlYnVnV2ViR0xFcnJvckNoZWNraW5nKGVuYWJsZWQpO1xuICB9XG5cbiAgcHVibGljIGNyZWF0ZU1hdHJpeFRleHR1cmUocm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBXZWJHTFRleHR1cmUge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwuY3JlYXRlTWF0cml4VGV4dHVyZSh0aGlzLmdsLCByb3dzLCBjb2x1bW5zKTtcbiAgfVxuXG4gIHB1YmxpYyB1cGxvYWRQaXhlbERhdGFUb1RleHR1cmUoXG4gICAgICB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsXG4gICAgICBwaXhlbHM6IEltYWdlRGF0YXxIVE1MSW1hZ2VFbGVtZW50fEhUTUxDYW52YXNFbGVtZW50fEhUTUxWaWRlb0VsZW1lbnQpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGdwZ3B1X3V0aWwudXBsb2FkUGl4ZWxEYXRhVG9UZXh0dXJlKHRoaXMuZ2wsIHRleHR1cmUsIHBpeGVscyk7XG4gIH1cblxuICBwdWJsaWMgY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShyb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6XG4gICAgICBXZWJHTFRleHR1cmUge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwuY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZSh0aGlzLmdsLCByb3dzLCBjb2x1bW5zKTtcbiAgfVxuXG4gIHB1YmxpYyBkZWxldGVNYXRyaXhUZXh0dXJlKHRleHR1cmU6IFdlYkdMVGV4dHVyZSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgaWYgKHRoaXMub3V0cHV0VGV4dHVyZSA9PT0gdGV4dHVyZSkge1xuICAgICAgd2ViZ2xfdXRpbC51bmJpbmRDb2xvclRleHR1cmVGcm9tRnJhbWVidWZmZXIodGhpcy5nbCwgdGhpcy5mcmFtZWJ1ZmZlcik7XG4gICAgICB0aGlzLm91dHB1dFRleHR1cmUgPSBudWxsO1xuICAgIH1cbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayh0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLmRlbGV0ZVRleHR1cmUodGV4dHVyZSkpO1xuICB9XG5cbiAgcHVibGljIHVwbG9hZE1hdHJpeFRvVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIsXG4gICAgICBtYXRyaXg6IEZsb2F0MzJBcnJheSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgbnVtQ2hhbm5lbHMgPSAxO1xuICAgIHJldHVybiBncGdwdV91dGlsLnVwbG9hZE1hdHJpeFRvVGV4dHVyZShcbiAgICAgICAgdGhpcy5nbCwgdGV4dHVyZSwgcm93cywgY29sdW1ucywgbWF0cml4LCBudW1DaGFubmVscyk7XG4gIH1cblxuICBwdWJsaWMgdXBsb2FkTWF0cml4VG9QYWNrZWRUZXh0dXJlKFxuICAgICAgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcixcbiAgICAgIG1hdHJpeDogRmxvYXQzMkFycmF5KSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICByZXR1cm4gZ3BncHVfdXRpbC51cGxvYWRNYXRyaXhUb1BhY2tlZFRleHR1cmUoXG4gICAgICAgIHRoaXMuZ2wsIHRleHR1cmUsIHJvd3MsIGNvbHVtbnMsIG1hdHJpeCk7XG4gIH1cblxuICBwdWJsaWMgZG93bmxvYWRNYXRyaXhGcm9tVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBGbG9hdDMyQXJyYXkge1xuICAgIHJldHVybiB0aGlzLmRvd25sb2FkTWF0cml4RHJpdmVyKFxuICAgICAgICB0ZXh0dXJlLFxuICAgICAgICAoKSA9PlxuICAgICAgICAgICAgZ3BncHVfdXRpbC5kb3dubG9hZE1hdHJpeEZyb21PdXRwdXRUZXh0dXJlKHRoaXMuZ2wsIHJvd3MsIGNvbHVtbnMpKTtcbiAgfVxuXG4gIHB1YmxpYyBkb3dubG9hZE1hdHJpeEZyb21QYWNrZWRUZXh0dXJlKFxuICAgICAgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gICAgcmV0dXJuIHRoaXMuZG93bmxvYWRNYXRyaXhEcml2ZXIoXG4gICAgICAgIHRleHR1cmUsXG4gICAgICAgICgpID0+IGdwZ3B1X3V0aWwuZG93bmxvYWRNYXRyaXhGcm9tUGFja2VkT3V0cHV0VGV4dHVyZShcbiAgICAgICAgICAgIHRoaXMuZ2wsIHJvd3MsIGNvbHVtbnMpKTtcbiAgfVxuXG4gIHB1YmxpYyBjcmVhdGVQcm9ncmFtKGZyYWdtZW50U2hhZGVyU291cmNlOiBzdHJpbmcpOiBXZWJHTFByb2dyYW0ge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgZ2wgPSB0aGlzLmdsO1xuICAgIGNvbnN0IGZyYWdtZW50U2hhZGVyOiBXZWJHTFNoYWRlciA9XG4gICAgICAgIHdlYmdsX3V0aWwuY3JlYXRlRnJhZ21lbnRTaGFkZXIoZ2wsIGZyYWdtZW50U2hhZGVyU291cmNlKTtcbiAgICBjb25zdCB2ZXJ0ZXhTaGFkZXI6IFdlYkdMU2hhZGVyID0gZ3BncHVfdXRpbC5jcmVhdGVWZXJ0ZXhTaGFkZXIoZ2wpO1xuICAgIGNvbnN0IHByb2dyYW06IFdlYkdMUHJvZ3JhbSA9IHdlYmdsX3V0aWwuY3JlYXRlUHJvZ3JhbShnbCk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmF0dGFjaFNoYWRlcihwcm9ncmFtLCB2ZXJ0ZXhTaGFkZXIpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYXR0YWNoU2hhZGVyKHByb2dyYW0sIGZyYWdtZW50U2hhZGVyKSk7XG4gICAgd2ViZ2xfdXRpbC5saW5rUHJvZ3JhbShnbCwgcHJvZ3JhbSk7XG4gICAgaWYgKHRoaXMuYXV0b0RlYnVnVmFsaWRhdGUpIHtcbiAgICAgIHdlYmdsX3V0aWwudmFsaWRhdGVQcm9ncmFtKGdsLCBwcm9ncmFtKTtcbiAgICB9XG5cbiAgICByZXR1cm4gcHJvZ3JhbTtcbiAgfVxuXG4gIHB1YmxpYyBkZWxldGVQcm9ncmFtKHByb2dyYW06IFdlYkdMUHJvZ3JhbSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgaWYgKHByb2dyYW0gPT09IHRoaXMucHJvZ3JhbSkge1xuICAgICAgdGhpcy5wcm9ncmFtID0gbnVsbDtcbiAgICB9XG4gICAgaWYgKHByb2dyYW0gIT0gbnVsbCkge1xuICAgICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2sodGhpcy5nbCwgKCkgPT4gdGhpcy5nbC5kZWxldGVQcm9ncmFtKHByb2dyYW0pKTtcbiAgICB9XG4gIH1cblxuICBwdWJsaWMgc2V0UHJvZ3JhbShwcm9ncmFtOiBXZWJHTFByb2dyYW18bnVsbCkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgdGhpcy5wcm9ncmFtID0gcHJvZ3JhbTtcbiAgICBpZiAoKHRoaXMucHJvZ3JhbSAhPSBudWxsKSAmJiB0aGlzLmF1dG9EZWJ1Z1ZhbGlkYXRlKSB7XG4gICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlUHJvZ3JhbSh0aGlzLmdsLCB0aGlzLnByb2dyYW0pO1xuICAgIH1cbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayh0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLnVzZVByb2dyYW0ocHJvZ3JhbSkpO1xuICB9XG5cbiAgcHVibGljIGdldFVuaWZvcm1Mb2NhdGlvbih1bmlmb3JtTmFtZTogc3RyaW5nKTogV2ViR0xVbmlmb3JtTG9jYXRpb24ge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgdGhpcy50aHJvd0lmTm9Qcm9ncmFtKCk7XG4gICAgcmV0dXJuIHdlYmdsX3V0aWwuZ2V0UHJvZ3JhbVVuaWZvcm1Mb2NhdGlvbk9yVGhyb3coXG4gICAgICAgIHRoaXMuZ2wsIHRoaXMucHJvZ3JhbSwgdW5pZm9ybU5hbWUpO1xuICB9XG5cbiAgcHVibGljIHNldElucHV0TWF0cml4VGV4dHVyZShcbiAgICAgIGlucHV0TWF0cml4VGV4dHVyZTogV2ViR0xUZXh0dXJlLCB1bmlmb3JtTmFtZTogc3RyaW5nLFxuICAgICAgdGV4dHVyZVVuaXQ6IG51bWJlcikge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgdGhpcy50aHJvd0lmTm9Qcm9ncmFtKCk7XG4gICAgd2ViZ2xfdXRpbC5iaW5kVGV4dHVyZVRvUHJvZ3JhbVVuaWZvcm1TYW1wbGVyKFxuICAgICAgICB0aGlzLmdsLCB0aGlzLnByb2dyYW0sIGlucHV0TWF0cml4VGV4dHVyZSwgdW5pZm9ybU5hbWUsIHRleHR1cmVVbml0KTtcbiAgfVxuXG4gIHB1YmxpYyBzZXRPdXRwdXRNYXRyaXhUZXh0dXJlKFxuICAgICAgb3V0cHV0TWF0cml4VGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcikge1xuICAgIHRoaXMuc2V0T3V0cHV0TWF0cml4VGV4dHVyZURyaXZlcihvdXRwdXRNYXRyaXhUZXh0dXJlLCBjb2x1bW5zLCByb3dzKTtcbiAgfVxuXG4gIHB1YmxpYyBzZXRPdXRwdXRQYWNrZWRNYXRyaXhUZXh0dXJlKFxuICAgICAgb3V0cHV0UGFja2VkTWF0cml4VGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcikge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgW3dpZHRoLCBoZWlnaHRdID1cbiAgICAgICAgdGV4X3V0aWwuZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gICAgdGhpcy5zZXRPdXRwdXRNYXRyaXhUZXh0dXJlRHJpdmVyKG91dHB1dFBhY2tlZE1hdHJpeFRleHR1cmUsIHdpZHRoLCBoZWlnaHQpO1xuICB9XG5cbiAgcHVibGljIHNldE91dHB1dE1hdHJpeFdyaXRlUmVnaW9uKFxuICAgICAgc3RhcnRSb3c6IG51bWJlciwgbnVtUm93czogbnVtYmVyLCBzdGFydENvbHVtbjogbnVtYmVyLFxuICAgICAgbnVtQ29sdW1uczogbnVtYmVyKSB7XG4gICAgdGhpcy5zZXRPdXRwdXRNYXRyaXhXcml0ZVJlZ2lvbkRyaXZlcihcbiAgICAgICAgc3RhcnRDb2x1bW4sIHN0YXJ0Um93LCBudW1Db2x1bW5zLCBudW1Sb3dzKTtcbiAgfVxuXG4gIHB1YmxpYyBzZXRPdXRwdXRQYWNrZWRNYXRyaXhXcml0ZVJlZ2lvbihcbiAgICAgIHN0YXJ0Um93OiBudW1iZXIsIG51bVJvd3M6IG51bWJlciwgc3RhcnRDb2x1bW46IG51bWJlcixcbiAgICAgIG51bUNvbHVtbnM6IG51bWJlcikge1xuICAgIHRocm93IG5ldyBFcnJvcignc2V0T3V0cHV0UGFja2VkTWF0cml4V3JpdGVSZWdpb24gbm90IGltcGxlbWVudGVkLicpO1xuICB9XG5cbiAgcHVibGljIGRlYnVnVmFsaWRhdGUoKSB7XG4gICAgaWYgKHRoaXMucHJvZ3JhbSAhPSBudWxsKSB7XG4gICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlUHJvZ3JhbSh0aGlzLmdsLCB0aGlzLnByb2dyYW0pO1xuICAgIH1cbiAgICB3ZWJnbF91dGlsLnZhbGlkYXRlRnJhbWVidWZmZXIodGhpcy5nbCk7XG4gIH1cblxuICBwdWJsaWMgZXhlY3V0ZVByb2dyYW0oKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB0aGlzLnRocm93SWZOb1Byb2dyYW0oKTtcbiAgICBjb25zdCBnbCA9IHRoaXMuZ2w7XG4gICAgZ3BncHVfdXRpbC5iaW5kVmVydGV4UHJvZ3JhbUF0dHJpYnV0ZVN0cmVhbXMoXG4gICAgICAgIGdsLCB0aGlzLnByb2dyYW0sIHRoaXMudmVydGV4QnVmZmVyKTtcbiAgICBpZiAodGhpcy5hdXRvRGVidWdWYWxpZGF0ZSkge1xuICAgICAgdGhpcy5kZWJ1Z1ZhbGlkYXRlKCk7XG4gICAgfVxuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgICBnbCwgKCkgPT4gZ2wuZHJhd0VsZW1lbnRzKGdsLlRSSUFOR0xFUywgNiwgZ2wuVU5TSUdORURfU0hPUlQsIDApKTtcbiAgfVxuXG4gIHB1YmxpYyBibG9ja1VudGlsQWxsUHJvZ3JhbXNDb21wbGV0ZWQoKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayh0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLmZpbmlzaCgpKTtcbiAgfVxuXG4gIHByaXZhdGUgZG93bmxvYWRNYXRyaXhEcml2ZXIoXG4gICAgICB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsXG4gICAgICBkb3dubG9hZEFuZERlY29kZTogKCkgPT4gRmxvYXQzMkFycmF5KTogRmxvYXQzMkFycmF5IHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHdlYmdsX3V0aWwuYmluZENvbG9yVGV4dHVyZVRvRnJhbWVidWZmZXIoXG4gICAgICAgIHRoaXMuZ2wsIHRleHR1cmUsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgIGNvbnN0IHJlc3VsdCA9IGRvd25sb2FkQW5kRGVjb2RlKCk7XG4gICAgaWYgKHRoaXMub3V0cHV0VGV4dHVyZSAhPSBudWxsKSB7XG4gICAgICB3ZWJnbF91dGlsLmJpbmRDb2xvclRleHR1cmVUb0ZyYW1lYnVmZmVyKFxuICAgICAgICAgIHRoaXMuZ2wsIHRoaXMub3V0cHV0VGV4dHVyZSwgdGhpcy5mcmFtZWJ1ZmZlcik7XG4gICAgICBpZiAodGhpcy5hdXRvRGVidWdWYWxpZGF0ZSkge1xuICAgICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlRnJhbWVidWZmZXIodGhpcy5nbCk7XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIHdlYmdsX3V0aWwudW5iaW5kQ29sb3JUZXh0dXJlRnJvbUZyYW1lYnVmZmVyKHRoaXMuZ2wsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgIH1cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHJpdmF0ZSBzZXRPdXRwdXRNYXRyaXhUZXh0dXJlRHJpdmVyKFxuICAgICAgb3V0cHV0TWF0cml4VGV4dHVyZU1heWJlUGFja2VkOiBXZWJHTFRleHR1cmUsIHdpZHRoOiBudW1iZXIsXG4gICAgICBoZWlnaHQ6IG51bWJlcikge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgZ2wgPSB0aGlzLmdsO1xuICAgIHdlYmdsX3V0aWwuYmluZENvbG9yVGV4dHVyZVRvRnJhbWVidWZmZXIoXG4gICAgICAgIGdsLCBvdXRwdXRNYXRyaXhUZXh0dXJlTWF5YmVQYWNrZWQsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgIGlmICh0aGlzLmF1dG9EZWJ1Z1ZhbGlkYXRlKSB7XG4gICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlRnJhbWVidWZmZXIoZ2wpO1xuICAgIH1cbiAgICB0aGlzLm91dHB1dFRleHR1cmUgPSBvdXRwdXRNYXRyaXhUZXh0dXJlTWF5YmVQYWNrZWQ7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnZpZXdwb3J0KDAsIDAsIHdpZHRoLCBoZWlnaHQpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuc2Npc3NvcigwLCAwLCB3aWR0aCwgaGVpZ2h0KSk7XG4gIH1cblxuICBwcml2YXRlIHNldE91dHB1dE1hdHJpeFdyaXRlUmVnaW9uRHJpdmVyKFxuICAgICAgeDogbnVtYmVyLCB5OiBudW1iZXIsIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgICAgdGhpcy5nbCwgKCkgPT4gdGhpcy5nbC5zY2lzc29yKHgsIHksIHdpZHRoLCBoZWlnaHQpKTtcbiAgfVxuXG4gIHByaXZhdGUgdGhyb3dJZkRpc3Bvc2VkKCkge1xuICAgIGlmICh0aGlzLmRpc3Bvc2VkKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0F0dGVtcHRlZCB0byB1c2UgZGlzcG9zZWQgR1BHUFVDb250ZXh0LicpO1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgdGhyb3dJZk5vUHJvZ3JhbSgpIHtcbiAgICBpZiAodGhpcy5wcm9ncmFtID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignTm8gR1BVIHByb2dyYW0gaXMgY3VycmVudGx5IHNldC4nKTtcbiAgICB9XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi8uLi91dGlsJztcbmltcG9ydCB7TkRBcnJheX0gZnJvbSAnLi4vbmRhcnJheSc7XG5cbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgc2hhZGVyX2NvbXBpbGVyIGZyb20gJy4vc2hhZGVyX2NvbXBpbGVyJztcbmltcG9ydCB7U2hhcGVJbmZvfSBmcm9tICcuL3NoYWRlcl9jb21waWxlcic7XG5cbmV4cG9ydCBpbnRlcmZhY2UgR1BHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lczogc3RyaW5nW107XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgcGFyYW1zOiBBcnJheTx7fT47XG4gIHVzZXJDb2RlOiBzdHJpbmc7XG4gIHN1cHBvcnRzQnJvYWRjYXN0aW5nPzogYm9vbGVhbjtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBHUEdQVUJpbmFyeSB7XG4gIHdlYkdMUHJvZ3JhbTogV2ViR0xQcm9ncmFtO1xuICBwcm9ncmFtOiBHUEdQVVByb2dyYW07XG4gIGdwZ3B1OiBHUEdQVUNvbnRleHQ7XG4gIHNvdXJjZTogc3RyaW5nO1xuICBpblNoYXBlSW5mb3M6IFNoYXBlSW5mb1tdO1xuICBvdXRTaGFwZUluZm86IFNoYXBlSW5mbztcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbXBpbGVQcm9ncmFtPFQgZXh0ZW5kcyBOREFycmF5LCBLIGV4dGVuZHMgTkRBcnJheT4oXG4gICAgZ3BncHU6IEdQR1BVQ29udGV4dCwgcHJvZ3JhbTogR1BHUFVQcm9ncmFtLCBpbnB1dHM6IFRbXSxcbiAgICBvdXRwdXQ6IEspOiBHUEdQVUJpbmFyeSB7XG4gIGNvbnN0IHVzZXJDb2RlID0gcHJvZ3JhbS51c2VyQ29kZTtcbiAgY29uc3QgaW5wdXRJbmZvcyA9IGlucHV0cy5tYXAoKGlucHV0LCBpKSA9PiB7XG4gICAgY29uc3Qgc2hhcGVJbmZvID0ge1xuICAgICAgbG9naWNhbFNoYXBlOiBpbnB1dC5zaGFwZSxcbiAgICAgIHRleFNoYXBlOiBpbnB1dC5nZXRUZXh0dXJlU2hhcGVSQygpXG4gICAgfTtcbiAgICByZXR1cm4ge25hbWU6IHByb2dyYW0udmFyaWFibGVOYW1lc1tpXSwgc2hhcGVJbmZvfTtcbiAgfSk7XG4gIGNvbnN0IGluU2hhcGVJbmZvcyA9IGlucHV0SW5mb3MubWFwKHggPT4geC5zaGFwZUluZm8pO1xuICBjb25zdCBvdXRTaGFwZUluZm8gPSB7XG4gICAgbG9naWNhbFNoYXBlOiBvdXRwdXQuc2hhcGUsXG4gICAgdGV4U2hhcGU6IG91dHB1dC5nZXRUZXh0dXJlU2hhcGVSQygpXG4gIH07XG4gIGNvbnN0IHNvdXJjZSA9IHNoYWRlcl9jb21waWxlci5tYWtlU2hhZGVyKFxuICAgICAgaW5wdXRJbmZvcywgb3V0U2hhcGVJbmZvLCB1c2VyQ29kZSxcbiAgICAgIHByb2dyYW0uc3VwcG9ydHNCcm9hZGNhc3RpbmcgPT09IHRydWUpO1xuICByZXR1cm4ge1xuICAgIHByb2dyYW0sXG4gICAgc291cmNlLFxuICAgIHdlYkdMUHJvZ3JhbTogZ3BncHUuY3JlYXRlUHJvZ3JhbShzb3VyY2UpLCBncGdwdSwgaW5TaGFwZUluZm9zLCBvdXRTaGFwZUluZm9cbiAgfTtcbn1cblxuZnVuY3Rpb24gdmFsaWRhdGVCaW5hcnlBbmRQcm9ncmFtKHNoYXBlSW5mb3M6IFNoYXBlSW5mb1tdLCBpbnB1dHM6IE5EQXJyYXlbXSkge1xuICBpZiAoc2hhcGVJbmZvcy5sZW5ndGggIT09IGlucHV0cy5sZW5ndGgpIHtcbiAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgYEJpbmFyeSB3YXMgY29tcGlsZWQgd2l0aCAke3NoYXBlSW5mb3MubGVuZ3RofSBpbnB1dHMsIGJ1dCBgICtcbiAgICAgICAgYHdhcyBleGVjdXRlZCB3aXRoICR7aW5wdXRzLmxlbmd0aH0gaW5wdXRzYCk7XG4gIH1cblxuICBzaGFwZUluZm9zLmZvckVhY2goKHMsIGkpID0+IHtcbiAgICBjb25zdCBzaGFwZUEgPSBzLmxvZ2ljYWxTaGFwZTtcbiAgICBjb25zdCB0ZXhTaGFwZUEgPSBzLnRleFNoYXBlO1xuICAgIGNvbnN0IHNoYXBlQiA9IGlucHV0c1tpXS5zaGFwZTtcbiAgICBjb25zdCB0ZXhTaGFwZUIgPSBpbnB1dHNbaV0uZ2V0VGV4dHVyZVNoYXBlUkMoKTtcblxuICAgIGlmICghdXRpbC5hcnJheXNFcXVhbChzaGFwZUEsIHNoYXBlQikpIHtcbiAgICAgIHRocm93IEVycm9yKFxuICAgICAgICAgIGBCaW5hcnkgd2FzIGNvbXBpbGVkIHdpdGggZGlmZmVyZW50IHNoYXBlcyB0aGFuIGAgK1xuICAgICAgICAgIGB0aGUgY3VycmVudCBhcmdzLiBTaGFwZXMgJHtzaGFwZUF9IGFuZCAke3NoYXBlQn0gbXVzdCBtYXRjaGApO1xuICAgIH1cbiAgICBpZiAoIXV0aWwuYXJyYXlzRXF1YWwodGV4U2hhcGVBLCB0ZXhTaGFwZUIpKSB7XG4gICAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgICBgQmluYXJ5IHdhcyBjb21waWxlZCB3aXRoIGRpZmZlcmVudCB0ZXh0dXJlIHNoYXBlcyB0aGFuIHRoZWAgK1xuICAgICAgICAgIGAgY3VycmVudCBhcmdzLiBTaGFwZSAke3RleFNoYXBlQX0gYW5kICR7dGV4U2hhcGVCfSBtdXN0IG1hdGNoYCk7XG4gICAgfVxuICB9KTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHJ1blByb2dyYW08VCBleHRlbmRzIE5EQXJyYXksIEsgZXh0ZW5kcyBOREFycmF5PihcbiAgICBiaW5hcnk6IEdQR1BVQmluYXJ5LCBpbnB1dHM6IFRbXSwgb3V0cHV0OiBLLFxuICAgIGN1c3RvbVNldHVwPzogKGdwZ3B1OiBHUEdQVUNvbnRleHQpID0+IHZvaWQpOiB2b2lkIHtcbiAgdmFsaWRhdGVCaW5hcnlBbmRQcm9ncmFtKGJpbmFyeS5pblNoYXBlSW5mb3MsIGlucHV0cyk7XG4gIHZhbGlkYXRlQmluYXJ5QW5kUHJvZ3JhbShbYmluYXJ5Lm91dFNoYXBlSW5mb10sIFtvdXRwdXRdKTtcblxuICBjb25zdCBvdXRUZXggPSBvdXRwdXQuZ2V0VGV4dHVyZSgpO1xuICBjb25zdCBvdXRUZXhTaGFwZSA9IG91dHB1dC5nZXRUZXh0dXJlU2hhcGVSQygpO1xuICBjb25zdCBncGdwdSA9IGJpbmFyeS5ncGdwdTtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShvdXRUZXgsIG91dFRleFNoYXBlWzBdLCBvdXRUZXhTaGFwZVsxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0oYmluYXJ5LndlYkdMUHJvZ3JhbSk7XG4gIGlucHV0cy5mb3JFYWNoKChpbnB1dCwgaSkgPT4ge1xuICAgIGNvbnN0IHRleCA9IGlucHV0LmdldFRleHR1cmUoKTtcbiAgICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUodGV4LCBiaW5hcnkucHJvZ3JhbS52YXJpYWJsZU5hbWVzW2ldLCBpKTtcbiAgfSk7XG4gIGlmIChjdXN0b21TZXR1cCAhPSBudWxsKSB7XG4gICAgY3VzdG9tU2V0dXAoZ3BncHUpO1xuICB9XG4gIGdwZ3B1LmV4ZWN1dGVQcm9ncmFtKCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlU2hhZGVyS2V5KFxuICAgIHByb2dyYW06IEdQR1BVUHJvZ3JhbSwgaW5wdXRzOiBOREFycmF5W10sIG91dHB1dDogTkRBcnJheSk6IHN0cmluZyB7XG4gIGNvbnN0IHBhcmFtcyA9IHByb2dyYW0ucGFyYW1zO1xuICBjb25zdCBrZXlTdGFydCA9XG4gICAgICBpbnB1dHMuY29uY2F0KG91dHB1dCkubWFwKHggPT4geC5zaGFwZSArICdfJyArIHguZ2V0VGV4dHVyZVNoYXBlUkMoKSk7XG4gIGNvbnN0IGtleUVuZCA9IHBhcmFtcy5tYXAocCA9PiBwLnRvU3RyaW5nKCkpO1xuICBsZXQga2V5ID0gW3Byb2dyYW0uY29uc3RydWN0b3IubmFtZV07XG4gIGtleS5wdXNoKChwcm9ncmFtLnN1cHBvcnRzQnJvYWRjYXN0aW5nID09PSB0cnVlKS50b1N0cmluZygpKTtcbiAga2V5ID0ga2V5LmNvbmNhdChrZXlTdGFydCwga2V5RW5kKTtcbiAgcmV0dXJuIGtleS5qb2luKCdfJyk7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIHRleF91dGlsIGZyb20gJy4vdGV4X3V0aWwnO1xuaW1wb3J0ICogYXMgd2ViZ2xfdXRpbCBmcm9tICcuL3dlYmdsX3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0V2ViR0xDb250ZXh0QXR0cmlidXRlcygpOiBXZWJHTENvbnRleHRBdHRyaWJ1dGVzIHtcbiAgcmV0dXJuIHtcbiAgICBhbHBoYTogZmFsc2UsXG4gICAgYW50aWFsaWFzOiBmYWxzZSxcbiAgICBwcmVtdWx0aXBsaWVkQWxwaGE6IGZhbHNlLFxuICAgIHByZXNlcnZlRHJhd2luZ0J1ZmZlcjogZmFsc2UsXG4gICAgZGVwdGg6IGZhbHNlLFxuICAgIHN0ZW5jaWw6IGZhbHNlLFxuICAgIGZhaWxJZk1ham9yUGVyZm9ybWFuY2VDYXZlYXQ6IHRydWVcbiAgfTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVdlYkdMQ29udGV4dChjYW52YXM/OiBIVE1MQ2FudmFzRWxlbWVudCkge1xuICBjb25zdCBhdHRyaWJ1dGVzID0gZ2V0V2ViR0xDb250ZXh0QXR0cmlidXRlcygpO1xuICBsZXQgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dDtcbiAgaWYgKGNhbnZhcyAhPSBudWxsKSB7XG4gICAgZ2wgPSB3ZWJnbF91dGlsLmNyZWF0ZVdlYkdMUmVuZGVyaW5nQ29udGV4dEZyb21DYW52YXMoY2FudmFzLCBhdHRyaWJ1dGVzKTtcbiAgfSBlbHNlIHtcbiAgICBnbCA9IHdlYmdsX3V0aWwuY3JlYXRlV2ViR0xSZW5kZXJpbmdDb250ZXh0KGF0dHJpYnV0ZXMpO1xuICB9XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLkRFUFRIX1RFU1QpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRpc2FibGUoZ2wuU1RFTkNJTF9URVNUKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLkJMRU5EKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLkRJVEhFUikpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGlzYWJsZShnbC5QT0xZR09OX09GRlNFVF9GSUxMKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLlNBTVBMRV9DT1ZFUkFHRSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZW5hYmxlKGdsLlNDSVNTT1JfVEVTVCkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZW5hYmxlKGdsLkNVTExfRkFDRSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuY3VsbEZhY2UoZ2wuQkFDSykpO1xuICByZXR1cm4gZ2w7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVWZXJ0ZXhTaGFkZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMU2hhZGVyIHtcbiAgY29uc3QgdmVydGV4U2hhZGVyU291cmNlID0gYFxuICAgIHByZWNpc2lvbiBoaWdocCBmbG9hdDtcbiAgICBhdHRyaWJ1dGUgdmVjMyBjbGlwU3BhY2VQb3M7XG4gICAgYXR0cmlidXRlIHZlYzIgdXY7XG4gICAgdmFyeWluZyB2ZWMyIHJlc3VsdFVWO1xuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgZ2xfUG9zaXRpb24gPSB2ZWM0KGNsaXBTcGFjZVBvcywgMSk7XG4gICAgICByZXN1bHRVViA9IHV2O1xuICAgIH1gO1xuICByZXR1cm4gd2ViZ2xfdXRpbC5jcmVhdGVWZXJ0ZXhTaGFkZXIoZ2wsIHZlcnRleFNoYWRlclNvdXJjZSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVWZXJ0ZXhCdWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMQnVmZmVyIHtcbiAgLy8gW3ggeSB6IHUgdl0gKiBbdXBwZXItbGVmdCwgbG93ZXItbGVmdCwgdXBwZXItcmlnaHQsIGxvd2VyLXJpZ2h0XVxuICBjb25zdCB2ZXJ0ZXhBcnJheSA9IG5ldyBGbG9hdDMyQXJyYXkoXG4gICAgICBbLTEsIDEsIDAsIDAsIDEsIC0xLCAtMSwgMCwgMCwgMCwgMSwgMSwgMCwgMSwgMSwgMSwgLTEsIDAsIDEsIDBdKTtcbiAgcmV0dXJuIHdlYmdsX3V0aWwuY3JlYXRlU3RhdGljVmVydGV4QnVmZmVyKGdsLCB2ZXJ0ZXhBcnJheSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVJbmRleEJ1ZmZlcihnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogV2ViR0xCdWZmZXIge1xuICAvLyBPcGVuR0wgKGFuZCBXZWJHTCkgaGF2ZSBcIkNDVyA9PSBmcm9udFwiIHdpbmRpbmdcbiAgY29uc3QgdHJpYW5nbGVWZXJ0ZXhJbmRpY2VzID0gbmV3IFVpbnQxNkFycmF5KFswLCAxLCAyLCAyLCAxLCAzXSk7XG4gIHJldHVybiB3ZWJnbF91dGlsLmNyZWF0ZVN0YXRpY0luZGV4QnVmZmVyKGdsLCB0cmlhbmdsZVZlcnRleEluZGljZXMpO1xufVxuXG5mdW5jdGlvbiBnZXRUZXh0dXJlSW50ZXJuYWxGb3JtYXQoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgbnVtQ2hhbm5lbHM6IG51bWJlcik6IG51bWJlciB7XG4gIGlmICh3ZWJnbF91dGlsLmlzV2ViR0wyRW5hYmxlZCgpKSB7XG4gICAgaWYgKG51bUNoYW5uZWxzID09PSA0KSB7XG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICByZXR1cm4gKGdsIGFzIGFueSkuUkdCQTMyRjtcbiAgICB9XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIHJldHVybiAoZ2wgYXMgYW55KS5SMzJGO1xuICB9XG4gIHJldHVybiBnbC5SR0JBO1xufVxuXG5mdW5jdGlvbiBnZXRUZXh0dXJlRm9ybWF0KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIG51bUNoYW5uZWxzOiBudW1iZXIpOiBudW1iZXIge1xuICBpZiAod2ViZ2xfdXRpbC5pc1dlYkdMMkVuYWJsZWQoKSAmJiBudW1DaGFubmVscyA9PT0gMSkge1xuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICByZXR1cm4gKGdsIGFzIGFueSkuUkVEO1xuICB9XG4gIHJldHVybiBnbC5SR0JBO1xufVxuXG5mdW5jdGlvbiBjcmVhdGVBbmRDb25maWd1cmVUZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLFxuICAgIG51bUNoYW5uZWxzOiBudW1iZXIpOiBXZWJHTFRleHR1cmUge1xuICB3ZWJnbF91dGlsLnZhbGlkYXRlVGV4dHVyZVNpemUoZ2wsIHdpZHRoLCBoZWlnaHQpO1xuICBjb25zdCB0ZXh0dXJlID0gd2ViZ2xfdXRpbC5jcmVhdGVUZXh0dXJlKGdsKTtcblxuICBjb25zdCB0ZXgyZCA9IGdsLlRFWFRVUkVfMkQ7XG4gIGNvbnN0IGludGVybmFsRm9ybWF0ID0gZ2V0VGV4dHVyZUludGVybmFsRm9ybWF0KGdsLCBudW1DaGFubmVscyk7XG4gIGNvbnN0IGZvcm1hdCA9IGdldFRleHR1cmVGb3JtYXQoZ2wsIG51bUNoYW5uZWxzKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRUZXh0dXJlKHRleDJkLCB0ZXh0dXJlKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnRleFBhcmFtZXRlcmkodGV4MmQsIGdsLlRFWFRVUkVfV1JBUF9TLCBnbC5DTEFNUF9UT19FREdFKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnRleFBhcmFtZXRlcmkodGV4MmQsIGdsLlRFWFRVUkVfV1JBUF9ULCBnbC5DTEFNUF9UT19FREdFKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnRleFBhcmFtZXRlcmkodGV4MmQsIGdsLlRFWFRVUkVfTUlOX0ZJTFRFUiwgZ2wuTkVBUkVTVCkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC50ZXhQYXJhbWV0ZXJpKHRleDJkLCBnbC5URVhUVVJFX01BR19GSUxURVIsIGdsLk5FQVJFU1QpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICBnbCxcbiAgICAgICgpID0+IGdsLnRleEltYWdlMkQoXG4gICAgICAgICAgdGV4MmQsIDAsIGludGVybmFsRm9ybWF0LCB3aWR0aCwgaGVpZ2h0LCAwLCBmb3JtYXQsIGdsLkZMT0FULCBudWxsKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKSk7XG4gIHJldHVybiB0ZXh0dXJlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlTWF0cml4VGV4dHVyZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFdlYkdMVGV4dHVyZSB7XG4gIGNvbnN0IFt3aWR0aCwgaGVpZ2h0XSA9XG4gICAgICB0ZXhfdXRpbC5nZXRVbnBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KHJvd3MsIGNvbHVtbnMpO1xuICBjb25zdCBudW1DaGFubmVscyA9IDE7XG4gIHJldHVybiBjcmVhdGVBbmRDb25maWd1cmVUZXh0dXJlKGdsLCB3aWR0aCwgaGVpZ2h0LCBudW1DaGFubmVscyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVDb2xvck1hdHJpeFRleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBXZWJHTFRleHR1cmUge1xuICBjb25zdCBbd2lkdGgsIGhlaWdodF0gPVxuICAgICAgdGV4X3V0aWwuZ2V0Q29sb3JNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcbiAgY29uc3QgbnVtQ2hhbm5lbHMgPSA0O1xuICByZXR1cm4gY3JlYXRlQW5kQ29uZmlndXJlVGV4dHVyZShnbCwgd2lkdGgsIGhlaWdodCwgbnVtQ2hhbm5lbHMpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFdlYkdMVGV4dHVyZSB7XG4gIGNvbnN0IFt3aWR0aCwgaGVpZ2h0XSA9XG4gICAgICB0ZXhfdXRpbC5nZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcbiAgY29uc3QgbnVtQ2hhbm5lbHMgPSA0O1xuICByZXR1cm4gY3JlYXRlQW5kQ29uZmlndXJlVGV4dHVyZShnbCwgd2lkdGgsIGhlaWdodCwgbnVtQ2hhbm5lbHMpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZFZlcnRleFByb2dyYW1BdHRyaWJ1dGVTdHJlYW1zKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSxcbiAgICB2ZXJ0ZXhCdWZmZXI6IFdlYkdMQnVmZmVyKSB7XG4gIGNvbnN0IHBvc09mZnNldCA9IDA7ICAgICAgICAgICAgICAgLy8geCBpcyB0aGUgZmlyc3QgYnVmZmVyIGVsZW1lbnRcbiAgY29uc3QgdXZPZmZzZXQgPSAzICogNDsgICAgICAgICAgICAvLyB1diBjb21lcyBhZnRlciBbeCB5IHpdXG4gIGNvbnN0IHN0cmlkZSA9ICgzICogNCkgKyAoMiAqIDQpOyAgLy8geHl6ICsgdXYsIGVhY2ggZW50cnkgaXMgNC1ieXRlIGZsb2F0LlxuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC5iaW5kQnVmZmVyKGdsLkFSUkFZX0JVRkZFUiwgdmVydGV4QnVmZmVyKSk7XG4gIHdlYmdsX3V0aWwuYmluZFZlcnRleEJ1ZmZlclRvUHJvZ3JhbUF0dHJpYnV0ZShcbiAgICAgIGdsLCBwcm9ncmFtLCAnY2xpcFNwYWNlUG9zJywgdmVydGV4QnVmZmVyLCAzLCBzdHJpZGUsIHBvc09mZnNldCk7XG4gIHRyeSB7XG4gICAgd2ViZ2xfdXRpbC5iaW5kVmVydGV4QnVmZmVyVG9Qcm9ncmFtQXR0cmlidXRlKFxuICAgICAgICBnbCwgcHJvZ3JhbSwgJ3V2JywgdmVydGV4QnVmZmVyLCAyLCBzdHJpZGUsIHV2T2Zmc2V0KTtcbiAgfSBjYXRjaCAoZSkge1xuICAgIC8vIFByb2dyYW1zIHdpdGggMXgxIG91dHB1dCB0ZXh0dXJlcyBkb24ndCB1c2UgdGhlIHV2IGF0dHJpYnV0ZS5cbiAgICAvLyBUaGlzIGNhbiBjYXVzZSB0aGUgc2hhZGVyIGxpbmtlciB0byBkZWFkLXN0cmlwIGl0LCBzbyB3ZSBzaG91bGRuJ3RcbiAgICAvLyBjb21wbGFpbiBvciBmYWlsIGlmIGl0J3Mgbm90IHByZXNlbnQuXG4gICAgaWYgKCFlLmhhc093blByb3BlcnR5KCduYW1lZFZlcnRleEF0dHJpYnV0ZU5vdEZvdW5kJykpIHtcbiAgICAgIHRocm93IGU7XG4gICAgfVxuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1cGxvYWRQaXhlbERhdGFUb1RleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgIHBpeGVsczogSW1hZ2VEYXRhfEhUTUxJbWFnZUVsZW1lbnR8SFRNTENhbnZhc0VsZW1lbnR8SFRNTFZpZGVvRWxlbWVudCkge1xuICBjb25zdCBudW1DaGFubmVscyA9IDQ7XG4gIGNvbnN0IGludGVybmFsRm9ybWF0ID0gZ2V0VGV4dHVyZUludGVybmFsRm9ybWF0KGdsLCBudW1DaGFubmVscyk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0ZXh0dXJlKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsXG4gICAgICAoKSA9PiBnbC50ZXhJbWFnZTJEKFxuICAgICAgICAgIGdsLlRFWFRVUkVfMkQsIDAsIGludGVybmFsRm9ybWF0LCBnbC5SR0JBLCBnbC5GTE9BVCwgcGl4ZWxzKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKSk7XG59XG5cbmZ1bmN0aW9uIHVwbG9hZERhdGFUb1RleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCB3aWR0aDogbnVtYmVyLFxuICAgIGhlaWdodDogbnVtYmVyLCBkYXRhOiBGbG9hdDMyQXJyYXksIG51bUNoYW5uZWxzOiBudW1iZXIpIHtcbiAgY29uc3QgdGV4dHVyZUZvcm1hdCA9IGdldFRleHR1cmVGb3JtYXQoZ2wsIG51bUNoYW5uZWxzKTtcblxuICB3ZWJnbF91dGlsLnZhbGlkYXRlVGV4dHVyZVNpemUoZ2wsIHdpZHRoLCBoZWlnaHQpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLFxuICAgICAgKCkgPT4gZ2wudGV4U3ViSW1hZ2UyRChcbiAgICAgICAgICBnbC5URVhUVVJFXzJELCAwLCAwLCAwLCB3aWR0aCwgaGVpZ2h0LCB0ZXh0dXJlRm9ybWF0LCBnbC5GTE9BVCxcbiAgICAgICAgICBkYXRhKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsXG4gICAgY29sdW1uczogbnVtYmVyLCBtYXRyaXg6IEZsb2F0MzJBcnJheSwgbnVtQ2hhbm5lbHM6IG51bWJlcikge1xuICBjb25zdCBbdywgaF0gPVxuICAgICAgdGV4X3V0aWwuZ2V0VW5wYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcblxuICBjb25zdCBjaGFubmVsc1BlclRleHR1cmUgPVxuICAgICAgbnVtQ2hhbm5lbHMgPT09IDEgPyB3ZWJnbF91dGlsLmdldENoYW5uZWxzUGVyVGV4dHVyZSgpIDogbnVtQ2hhbm5lbHM7XG4gIGNvbnN0IHVucGFja2VkQXJyYXkgPVxuICAgICAgbmV3IEZsb2F0MzJBcnJheSh0ZXhfdXRpbC5nZXRVbnBhY2tlZEFycmF5U2l6ZUZyb21NYXRyaXhTaXplKFxuICAgICAgICAgIG1hdHJpeC5sZW5ndGgsIGNoYW5uZWxzUGVyVGV4dHVyZSkpO1xuICB0ZXhfdXRpbC5lbmNvZGVNYXRyaXhUb1VucGFja2VkQXJyYXkoXG4gICAgICBtYXRyaXgsIHVucGFja2VkQXJyYXksIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG5cbiAgdXBsb2FkRGF0YVRvVGV4dHVyZShnbCwgdGV4dHVyZSwgdywgaCwgdW5wYWNrZWRBcnJheSwgbnVtQ2hhbm5lbHMpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdXBsb2FkTWF0cml4VG9QYWNrZWRUZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLFxuICAgIGNvbHVtbnM6IG51bWJlciwgbWF0cml4OiBGbG9hdDMyQXJyYXkpIHtcbiAgY29uc3QgW3csIGhdID0gdGV4X3V0aWwuZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gIGNvbnN0IHBhY2tlZFJHQkEgPSBuZXcgRmxvYXQzMkFycmF5KFxuICAgICAgdGV4X3V0aWwuZ2V0UGFja2VkUkdCQUFycmF5U2l6ZUZyb21NYXRyaXhTaGFwZShyb3dzLCBjb2x1bW5zKSk7XG4gIHRleF91dGlsLmVuY29kZU1hdHJpeFRvUGFja2VkUkdCQShtYXRyaXgsIHJvd3MsIGNvbHVtbnMsIHBhY2tlZFJHQkEpO1xuICBjb25zdCBudW1DaGFubmVscyA9IDQ7XG4gIHVwbG9hZERhdGFUb1RleHR1cmUoZ2wsIHRleHR1cmUsIHcsIGgsIHBhY2tlZFJHQkEsIG51bUNoYW5uZWxzKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRvd25sb2FkTWF0cml4RnJvbU91dHB1dFRleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBGbG9hdDMyQXJyYXkge1xuICBjb25zdCBbdywgaF0gPVxuICAgICAgdGV4X3V0aWwuZ2V0VW5wYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcblxuICBjb25zdCBjaGFubmVsc1BlclRleHR1cmUgPSA0O1xuICBjb25zdCB1bnBhY2tlZEFycmF5ID1cbiAgICAgIG5ldyBGbG9hdDMyQXJyYXkodGV4X3V0aWwuZ2V0VW5wYWNrZWRBcnJheVNpemVGcm9tTWF0cml4U2l6ZShcbiAgICAgICAgICByb3dzICogY29sdW1ucywgY2hhbm5lbHNQZXJUZXh0dXJlKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnJlYWRQaXhlbHMoMCwgMCwgdywgaCwgZ2wuUkdCQSwgZ2wuRkxPQVQsIHVucGFja2VkQXJyYXkpKTtcblxuICBjb25zdCBtYXRyaXggPSBuZXcgRmxvYXQzMkFycmF5KHJvd3MgKiBjb2x1bW5zKTtcbiAgdGV4X3V0aWwuZGVjb2RlTWF0cml4RnJvbVVucGFja2VkQXJyYXkoXG4gICAgICB1bnBhY2tlZEFycmF5LCBtYXRyaXgsIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG4gIHJldHVybiBtYXRyaXg7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBkb3dubG9hZE1hdHJpeEZyb21QYWNrZWRPdXRwdXRUZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogRmxvYXQzMkFycmF5IHtcbiAgY29uc3QgW3csIGhdID0gdGV4X3V0aWwuZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gIGNvbnN0IHBhY2tlZFJHQkEgPSBuZXcgRmxvYXQzMkFycmF5KFxuICAgICAgdGV4X3V0aWwuZ2V0UGFja2VkUkdCQUFycmF5U2l6ZUZyb21NYXRyaXhTaGFwZShyb3dzLCBjb2x1bW5zKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnJlYWRQaXhlbHMoMCwgMCwgdywgaCwgZ2wuUkdCQSwgZ2wuRkxPQVQsIHBhY2tlZFJHQkEpKTtcbiAgY29uc3QgbWF0cml4ID0gbmV3IEZsb2F0MzJBcnJheShyb3dzICogY29sdW1ucyk7XG4gIHJldHVybiB0ZXhfdXRpbC5kZWNvZGVNYXRyaXhGcm9tUGFja2VkUkdCQShwYWNrZWRSR0JBLCByb3dzLCBjb2x1bW5zLCBtYXRyaXgpO1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge0dQR1BVUHJvZ3JhbX0gZnJvbSAnLi9ncGdwdV9tYXRoJztcblxuZXhwb3J0IGNsYXNzIExvZ1N1bUV4cFByb2dyYW0gaW1wbGVtZW50cyBHUEdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWydBJ107XG4gIHBhcmFtczogQXJyYXk8e30+ID0gW107XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXSA9IFtdO1xuICB1c2VyQ29kZTogc3RyaW5nO1xuXG4gIGNvbnN0cnVjdG9yKGFTaXplOiBudW1iZXIpIHtcbiAgICB0aGlzLnVzZXJDb2RlID0gYFxuICAgICAgdm9pZCBtYWluKCkge1xuICAgICAgICBmbG9hdCBhTWF4ID0gZ2V0QUZsYXQoMC4wKTtcbiAgICAgICAgZm9yIChpbnQgaSA9IDA7IGkgPCAke2FTaXplfTsgaSsrKSB7XG4gICAgICAgICAgYU1heCA9IG1heChhTWF4LCBnZXRBRmxhdChmbG9hdChpKSkpO1xuICAgICAgICB9XG5cbiAgICAgICAgZmxvYXQgZXhwU3VtID0gMC4wO1xuICAgICAgICBmb3IgKGludCBpID0gMDsgaSA8ICR7YVNpemV9OyBpKyspIHtcbiAgICAgICAgICBleHBTdW0gKz0gZXhwKGdldEFGbGF0KGZsb2F0KGkpKSAtIGFNYXgpO1xuICAgICAgICB9XG5cbiAgICAgICAgc2V0T3V0cHV0KGFNYXggKyBsb2coZXhwU3VtKSk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge01hdHJpeE9yaWVudGF0aW9ufSBmcm9tICcuLi9tYXRoJztcbmltcG9ydCB7R1BHUFVQcm9ncmFtfSBmcm9tICcuL2dwZ3B1X21hdGgnO1xuXG5leHBvcnQgY2xhc3MgTWF0TXVsUHJvZ3JhbSBpbXBsZW1lbnRzIEdQR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ21hdHJpeEEnLCAnbWF0cml4QiddO1xuICBwYXJhbXM6IEFycmF5PHt9PjtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICB1c2VyQ29kZTogc3RyaW5nO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgYVNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLCBiU2hhcGU6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBhT3JpZW50ID0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUixcbiAgICAgIGJPcmllbnQgPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSB7XG4gICAgdGhpcy5wYXJhbXMgPSBbYU9yaWVudCwgYk9yaWVudF07XG5cbiAgICBjb25zdCBvdXRlclNoYXBlQSA9XG4gICAgICAgIChhT3JpZW50ID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IGFTaGFwZVswXSA6IGFTaGFwZVsxXTtcbiAgICBjb25zdCBvdXRlclNoYXBlQiA9XG4gICAgICAgIChiT3JpZW50ID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IGJTaGFwZVsxXSA6IGJTaGFwZVswXTtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gW291dGVyU2hhcGVBLCBvdXRlclNoYXBlQl07XG5cbiAgICBjb25zdCBzaGFyZWREaW0gPVxuICAgICAgICAoYU9yaWVudCA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUiA/IGFTaGFwZVsxXSA6IGFTaGFwZVswXSk7XG4gICAgY29uc3QgYVNuaXBwZXQgPVxuICAgICAgICAoYU9yaWVudCA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyAnYVJvdywgaScgOiAnaSwgYVJvdyc7XG4gICAgY29uc3QgYlNuaXBwZXQgPVxuICAgICAgICAoYk9yaWVudCA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyAnaSwgYkNvbCcgOiAnYkNvbCwgaSc7XG5cbiAgICB0aGlzLnVzZXJDb2RlID0gYFxuICAgICAgY29uc3QgaW50IHNoYXJlZERpbSA9ICR7c2hhcmVkRGltfTtcblxuICAgICAgZmxvYXQgZG90QVJvd0JDb2woZmxvYXQgYVJvdywgZmxvYXQgYkNvbCkge1xuICAgICAgICBmbG9hdCByZXN1bHQgPSAwLjA7XG4gICAgICAgIGZvciAoaW50IGlpID0gMDsgaWkgPCBzaGFyZWREaW07IGlpKyspIHtcbiAgICAgICAgICBmbG9hdCBpID0gZmxvYXQoaWkpO1xuICAgICAgICAgIGZsb2F0IGEgPSBnZXRNYXRyaXhBKCR7YVNuaXBwZXR9KTtcbiAgICAgICAgICBmbG9hdCBiID0gZ2V0TWF0cml4Qigke2JTbmlwcGV0fSk7XG4gICAgICAgICAgcmVzdWx0ICs9IChhICogYik7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICAgIH1cblxuICAgICAgdm9pZCBtYWluKCkge1xuICAgICAgICB2ZWMyIHJlc1JDID0gZ2V0T3V0cHV0Q29vcmRzKCk7XG4gICAgICAgIHNldE91dHB1dChkb3RBUm93QkNvbChyZXNSQy54LCByZXNSQy55KSk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge01hdHJpeE9yaWVudGF0aW9ufSBmcm9tICcuLi9tYXRoJztcblxuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vZ3BncHVfY29udGV4dCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlclNvdXJjZShcbiAgICBzaGFyZWREaW1lbnNpb246IG51bWJlciwgYU9yaWVudGF0aW9uOiBNYXRyaXhPcmllbnRhdGlvbixcbiAgICBiT3JpZW50YXRpb246IE1hdHJpeE9yaWVudGF0aW9uKTogc3RyaW5nIHtcbiAgLypcbiAgICAgIEEgPSBbMCAxICAgQiA9IFswIDEgIG91dCA9IFtBMCpCMCtBMSpCMiBBMCpCMStBMSpCM1xuICAgICAgICAgICAyIDNdICAgICAgIDIgM10gICAgICAgIEEyKkIwK0ExKkIyIEEyKkIxK0F3KkIzXVxuICAgICAgb3V0LjAgPSBBMCAqIEIwICsgQTEgKiBCMlxuICAgICAgb3V0LjEgPSBBMCAqIEIxICsgQTEgKiBCM1xuICAgICAgb3V0LjIgPSBBMiAqIEIwICsgQTMgKiBCMlxuICAgICAgb3V0LjMgPSBBMiAqIEIxICsgQTMgKiBCM1xuXG4gICAgICBBKkIgICAgID0gQS54eHp6ICogQi54eXh5ICsgQS55eXd3ICogQi56d3p3XG4gICAgICBBXnQqQiAgID0gQS54eHl5ICogQi54eXh5ICsgQS56end3ICogQi56d3p3XG4gICAgICBBKkJedCAgID0gQS54eHp6ICogQi54enh6ICsgQS55eXd3ICogQi55d3l3XG4gICAgICBBXnQqQl50ID0gQS54eHl5ICogQi54enh6ICsgQS56end3ICogQi55d3l3XG4gICAqL1xuICBjb25zdCBzaGFyZWREaW1lbnNpb25QYWNrZWQgPSBNYXRoLmNlaWwoc2hhcmVkRGltZW5zaW9uIC8gMik7XG4gIGNvbnN0IGFTYW1wbGUgPSAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICAnY2VudGVyLCByZXN1bHRVVi50JyA6XG4gICAgICAncmVzdWx0VVYudCwgY2VudGVyJztcbiAgY29uc3QgYlNhbXBsZSA9IChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgICdyZXN1bHRVVi5zLCBjZW50ZXInIDpcbiAgICAgICdjZW50ZXIsIHJlc3VsdFVWLnMnO1xuICBjb25zdCBhU3dpenpsZTogW3N0cmluZywgc3RyaW5nXSA9XG4gICAgICAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IFsnYS54eHp6JywgJ2EueXl3dyddIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgWydhLnh4eXknLCAnYS56end3J107XG4gIGNvbnN0IGJTd2l6emxlOiBbc3RyaW5nLCBzdHJpbmddID1cbiAgICAgIChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID8gWydiLnh5eHknLCAnYi56d3p3J10gOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBbJ2IueHp4eicsICdiLnl3eXcnXTtcbiAgcmV0dXJuIGBcbiAgICBwcmVjaXNpb24gaGlnaHAgZmxvYXQ7XG4gICAgdW5pZm9ybSBzYW1wbGVyMkQgbWF0cml4QTtcbiAgICB1bmlmb3JtIHNhbXBsZXIyRCBtYXRyaXhCO1xuICAgIHZhcnlpbmcgdmVjMiByZXN1bHRVVjtcblxuICAgIGNvbnN0IGZsb2F0IHNoYXJlZERpbWVuc2lvbiA9ICR7c2hhcmVkRGltZW5zaW9uUGFja2VkfS4wO1xuXG4gICAgdmVjNCBkb3QyeDJBUm93QkNvbCgpIHtcbiAgICAgIHZlYzQgcmVzdWx0ID0gdmVjNCgwLCAwLCAwLCAwKTtcbiAgICAgIGZvciAoaW50IGlpID0gMDsgaWkgPCAke3NoYXJlZERpbWVuc2lvblBhY2tlZH07IGlpKyspIHtcbiAgICAgICAgZmxvYXQgaSA9IGZsb2F0KGlpKTtcbiAgICAgICAgZmxvYXQgY2VudGVyID0gKGkgKyAwLjUpIC8gc2hhcmVkRGltZW5zaW9uO1xuICAgICAgICB2ZWM0IGEgPSB0ZXh0dXJlMkQobWF0cml4QSwgdmVjMigke2FTYW1wbGV9KSk7XG4gICAgICAgIHZlYzQgYiA9IHRleHR1cmUyRChtYXRyaXhCLCB2ZWMyKCR7YlNhbXBsZX0pKTtcbiAgICAgICAgcmVzdWx0ICs9XG4gICAgICAgICAgKCR7YVN3aXp6bGVbMF19ICogJHtiU3dpenpsZVswXX0pICsgKCR7YVN3aXp6bGVbMV19ICogJHtiU3dpenpsZVsxXX0pO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICB9XG5cbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICBnbF9GcmFnQ29sb3IgPSBkb3QyeDJBUm93QkNvbCgpO1xuICAgIH1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbXVsdGlwbHlNYXRyaXhQYWNrZWQoXG4gICAgZ3BncHU6IEdQR1BVQ29udGV4dCwgbXVsdGlwbHlQcm9ncmFtOiBXZWJHTFByb2dyYW0sIGE6IFdlYkdMVGV4dHVyZSxcbiAgICBiOiBXZWJHTFRleHR1cmUsIHJlc3VsdDogV2ViR0xUZXh0dXJlLFxuICAgIHJlc3VsdFNoYXBlUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gIGdwZ3B1LnNldE91dHB1dFBhY2tlZE1hdHJpeFRleHR1cmUoXG4gICAgICByZXN1bHQsIHJlc3VsdFNoYXBlUm93Q29sWzBdLCByZXN1bHRTaGFwZVJvd0NvbFsxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0obXVsdGlwbHlQcm9ncmFtKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGEsICdtYXRyaXhBJywgMCk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZShiLCAnbWF0cml4QicsIDEpO1xuICBncGdwdS5leGVjdXRlUHJvZ3JhbSgpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdXBsb2FkTXVsdGlwbHlNYXRyaXhQYWNrZWREb3dubG9hZChcbiAgICBhOiBGbG9hdDMyQXJyYXksIGFTaGFwZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSwgYjogRmxvYXQzMkFycmF5LFxuICAgIGJTaGFwZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSwgYU9yaWVudGF0aW9uID0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUixcbiAgICBiT3JpZW50YXRpb24gPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKTogRmxvYXQzMkFycmF5IHtcbiAgY29uc3QgcmVzdWx0TnVtUm93cyA9IChhT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgIGFTaGFwZVJvd0NvbFswXSA6XG4gICAgICBhU2hhcGVSb3dDb2xbMV07XG4gIGNvbnN0IHJlc3VsdE51bUNvbHMgPSAoYk9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICBiU2hhcGVSb3dDb2xbMV0gOlxuICAgICAgYlNoYXBlUm93Q29sWzBdO1xuICBjb25zdCBzaGFyZWREaW1lbnNpb24gPSAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICBhU2hhcGVSb3dDb2xbMV0gOlxuICAgICAgYVNoYXBlUm93Q29sWzBdO1xuXG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCBwcm9ncmFtOiBXZWJHTFByb2dyYW0gPSBncGdwdS5jcmVhdGVQcm9ncmFtKFxuICAgICAgZ2V0RnJhZ21lbnRTaGFkZXJTb3VyY2Uoc2hhcmVkRGltZW5zaW9uLCBhT3JpZW50YXRpb24sIGJPcmllbnRhdGlvbikpO1xuXG4gIGNvbnN0IGFUZXh0dXJlOiBXZWJHTFRleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShhU2hhcGVSb3dDb2xbMF0sIGFTaGFwZVJvd0NvbFsxXSk7XG4gIGNvbnN0IGJUZXh0dXJlOiBXZWJHTFRleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShiU2hhcGVSb3dDb2xbMF0sIGJTaGFwZVJvd0NvbFsxXSk7XG4gIGNvbnN0IHJlc3VsdFRleHR1cmU6IFdlYkdMVGV4dHVyZSA9XG4gICAgICBncGdwdS5jcmVhdGVQYWNrZWRNYXRyaXhUZXh0dXJlKHJlc3VsdE51bVJvd3MsIHJlc3VsdE51bUNvbHMpO1xuXG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShcbiAgICAgIGFUZXh0dXJlLCBhU2hhcGVSb3dDb2xbMF0sIGFTaGFwZVJvd0NvbFsxXSwgYSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShcbiAgICAgIGJUZXh0dXJlLCBiU2hhcGVSb3dDb2xbMF0sIGJTaGFwZVJvd0NvbFsxXSwgYik7XG5cbiAgbXVsdGlwbHlNYXRyaXhQYWNrZWQoXG4gICAgICBncGdwdSwgcHJvZ3JhbSwgYVRleHR1cmUsIGJUZXh0dXJlLCByZXN1bHRUZXh0dXJlLFxuICAgICAgW3Jlc3VsdE51bVJvd3MsIHJlc3VsdE51bUNvbHNdKTtcblxuICBjb25zdCByZXN1bHQgPSBncGdwdS5kb3dubG9hZE1hdHJpeEZyb21QYWNrZWRUZXh0dXJlKFxuICAgICAgcmVzdWx0VGV4dHVyZSwgcmVzdWx0TnVtUm93cywgcmVzdWx0TnVtQ29scyk7XG5cbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShhVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoYlRleHR1cmUpO1xuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKHJlc3VsdFRleHR1cmUpO1xuICBncGdwdS5kZWxldGVQcm9ncmFtKHByb2dyYW0pO1xuICBncGdwdS5kaXNwb3NlKCk7XG5cbiAgcmV0dXJuIHJlc3VsdDtcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uL2NvbnZfdXRpbCc7XG5pbXBvcnQge0dQR1BVUHJvZ3JhbX0gZnJvbSAnLi9ncGdwdV9tYXRoJztcblxuZXhwb3J0IGNsYXNzIFBvb2wyRFByb2dyYW0gaW1wbGVtZW50cyBHUEdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWyd4J107XG4gIHBhcmFtczogQXJyYXk8e30+O1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHVzZXJDb2RlOiBzdHJpbmc7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICB4U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsXG4gICAgICBwYWQ6IG51bWJlciwgcG9vbFR5cGU6ICdtYXgnfCdtaW4nfCdhdmcnLCBjb21wdXRlUG9zaXRpb25zOiBib29sZWFuKSB7XG4gICAgaWYgKHBvb2xUeXBlID09PSAnYXZnJyAmJiBjb21wdXRlUG9zaXRpb25zKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0Nhbm5vdCBjb21wdXRlIHBvc2l0aW9ucyBmb3IgYXZlcmFnZSBwb29sLicpO1xuICAgIH1cblxuICAgIGxldCByZXR1cm5WYWx1ZSA9ICdtaW5NYXhWYWx1ZSc7XG4gICAgaWYgKGNvbXB1dGVQb3NpdGlvbnMpIHtcbiAgICAgIHJldHVyblZhbHVlID0gJ21pbk1heFBvc2l0aW9uJztcbiAgICB9IGVsc2UgaWYgKHBvb2xUeXBlID09PSAnYXZnJykge1xuICAgICAgcmV0dXJuVmFsdWUgPSBgYXZnVmFsdWUgLyAke2ZTaXplICogZlNpemV9LjBgO1xuICAgIH1cbiAgICBjb25zdCB4Um93c0xpbWl0ID0geFNoYXBlWzBdIC0gMC41O1xuICAgIGNvbnN0IHhDb2xzTGltaXQgPSB4U2hhcGVbMV0gLSAwLjU7XG4gICAgdGhpcy5wYXJhbXMgPSBbc3RyaWRlLCBwYWQsIGZTaXplLCBjb21wdXRlUG9zaXRpb25zXTtcbiAgICB0aGlzLm91dHB1dFNoYXBlID1cbiAgICAgICAgY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKHhTaGFwZSwgZlNpemUsIHhTaGFwZVsyXSwgc3RyaWRlLCBwYWQpO1xuXG4gICAgdGhpcy51c2VyQ29kZSA9IGBcbiAgICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgdmVjMyBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgZmxvYXQgeVIgPSBjb29yZHMueDtcbiAgICAgICAgZmxvYXQgeUMgPSBjb29yZHMueTtcbiAgICAgICAgZmxvYXQgZCA9IGNvb3Jkcy56O1xuXG4gICAgICAgIHZlYzIgeFJDQ29ybmVyID0gdmVjMih5UiwgeUMpICogdmVjMigke3N0cmlkZX0uMCwgJHtzdHJpZGV9LjApIC1cbiAgICAgICAgICAgIHZlYzIoJHtwYWR9LjAsICR7cGFkfS4wKTtcbiAgICAgICAgZmxvYXQgeFJDb3JuZXIgPSB4UkNDb3JuZXIueDtcbiAgICAgICAgZmxvYXQgeENDb3JuZXIgPSB4UkNDb3JuZXIueTtcblxuICAgICAgICAvLyBtYXgvbWluIHgoPywgPywgZCkgdG8gZ2V0IHkoeVIsIHlDLCBkKS5cbiAgICAgICAgLy8gPyA9IHRvIGJlIGRldGVybWluZWRcbiAgICAgICAgZmxvYXQgbWluTWF4VmFsdWUgPSAwLjA7XG4gICAgICAgIGZsb2F0IG1pbk1heFZhbHVlRm91bmQgPSAwLjA7XG4gICAgICAgIGZsb2F0IG1pbk1heFBvc2l0aW9uID0gMC4wO1xuICAgICAgICBmbG9hdCBhdmdWYWx1ZSA9IDAuMDtcblxuICAgICAgICBmb3IgKGludCBpd1IgPSAwOyBpd1IgPCAke2ZTaXplfTsgaXdSKyspIHtcbiAgICAgICAgICBmbG9hdCB3UiA9IGZsb2F0KGl3Uik7XG4gICAgICAgICAgZmxvYXQgeFIgPSB4UkNvcm5lciArIHdSO1xuXG4gICAgICAgICAgaWYgKHhSIDwgMC4wIHx8IHhSID4gJHt4Um93c0xpbWl0fSkge1xuICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgZm9yIChpbnQgaXdDID0gMDsgaXdDIDwgJHtmU2l6ZX07IGl3QysrKSB7XG4gICAgICAgICAgICBmbG9hdCB3QyA9IGZsb2F0KGl3Qyk7XG4gICAgICAgICAgICBmbG9hdCB4QyA9IHhDQ29ybmVyICsgd0M7XG5cbiAgICAgICAgICAgIGlmICh4QyA8IDAuMCB8fCB4QyA+ICR7eENvbHNMaW1pdH0pIHtcbiAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGZsb2F0IHZhbHVlID0gZ2V0WCh4UiwgeEMsIGQpO1xuXG4gICAgICAgICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgICAgICAgIHNldE91dHB1dCh2YWx1ZSk7XG4gICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgaWYgKCR7cG9vbFR5cGUgPT09ICdhdmcnfSkge1xuICAgICAgICAgICAgICBhdmdWYWx1ZSArPSB2YWx1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIC8vIElmIGEgbWluIC8gbWF4IHZhbHVlIGhhcyBhbHJlYWR5IGJlZW4gZm91bmQsIHVzZSBpdC4gSWYgbm90LFxuICAgICAgICAgICAgICAvLyB1c2UgdGhlIGN1cnJlbnQgdmFsdWUuXG4gICAgICAgICAgICAgIGZsb2F0IGN1cnJNaW5NYXhWYWx1ZSA9IG1peChcbiAgICAgICAgICAgICAgICAgIHZhbHVlLCBtaW5NYXhWYWx1ZSwgbWluTWF4VmFsdWVGb3VuZCk7XG4gICAgICAgICAgICAgIGlmICh2YWx1ZSAke3Bvb2xUeXBlID09PSAnbWluJyA/ICc8PScgOiAnPj0nfSBjdXJyTWluTWF4VmFsdWUpIHtcbiAgICAgICAgICAgICAgICBtaW5NYXhWYWx1ZSA9IHZhbHVlO1xuICAgICAgICAgICAgICAgIG1pbk1heFZhbHVlRm91bmQgPSAxLjA7XG4gICAgICAgICAgICAgICAgaWYgKCR7Y29tcHV0ZVBvc2l0aW9uc30pIHtcbiAgICAgICAgICAgICAgICAgIG1pbk1heFBvc2l0aW9uID0gd1IgKiAke2ZTaXplfS4wICsgd0M7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHNldE91dHB1dCgke3JldHVyblZhbHVlfSk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uLy4uL3V0aWwnO1xuXG5leHBvcnQgdHlwZSBTaGFwZUluZm8gPSB7XG4gIGxvZ2ljYWxTaGFwZTogbnVtYmVyW10sXG4gIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdXG59O1xuXG5leHBvcnQgdHlwZSBJbnB1dEluZm8gPSB7XG4gIG5hbWU6IHN0cmluZyxcbiAgc2hhcGVJbmZvOiBTaGFwZUluZm9cbn07XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlU2hhZGVyKFxuICAgIGlucHV0c0luZm86IElucHV0SW5mb1tdLCBvdXRwdXRTaGFwZTogU2hhcGVJbmZvLCB1c2VyQ29kZTogc3RyaW5nLFxuICAgIGJyb2FkY2FzdDogYm9vbGVhbik6IHN0cmluZyB7XG4gIGNvbnN0IGlucHV0UHJlZml4U25pcHBldCA9XG4gICAgICBpbnB1dHNJbmZvLm1hcCh4ID0+IGB1bmlmb3JtIHNhbXBsZXIyRCAke3gubmFtZX07YCkuam9pbignXFxuJyk7XG4gIGNvbnN0IGlucHV0U2FtcGxpbmdTbmlwcGV0ID1cbiAgICAgIGlucHV0c0luZm8ubWFwKHggPT4gZ2V0SW5wdXRTYW1wbGluZ1NuaXBwZXQoeCwgb3V0cHV0U2hhcGUsIGJyb2FkY2FzdCkpXG4gICAgICAgICAgLmpvaW4oJ1xcbicpO1xuICBjb25zdCBvdXRUZXhTaGFwZSA9IG91dHB1dFNoYXBlLnRleFNoYXBlO1xuICBjb25zdCBvdXRwdXRTYW1wbGluZ1NuaXBwZXQgPVxuICAgICAgZ2V0T3V0cHV0U2FtcGxpbmdTbmlwcGV0KG91dHB1dFNoYXBlLmxvZ2ljYWxTaGFwZSwgb3V0VGV4U2hhcGUpO1xuICBjb25zdCBzb3VyY2UgPSBbXG4gICAgU0hBREVSX1BSRUZJWCwgaW5wdXRQcmVmaXhTbmlwcGV0LCBpbnB1dFNhbXBsaW5nU25pcHBldCxcbiAgICBvdXRwdXRTYW1wbGluZ1NuaXBwZXQsIHVzZXJDb2RlXG4gIF0uam9pbignXFxuJyk7XG4gIHJldHVybiBzb3VyY2U7XG59XG5cbmZ1bmN0aW9uIGdldElucHV0U2FtcGxpbmdTbmlwcGV0KFxuICAgIGluSW5mbzogSW5wdXRJbmZvLCBvdXRTaGFwZUluZm86IFNoYXBlSW5mbywgYnJvYWRjYXN0OiBib29sZWFuKSB7XG4gIGNvbnN0IHNoYXBlID0gaW5JbmZvLnNoYXBlSW5mby5sb2dpY2FsU2hhcGU7XG4gIGNvbnN0IHRleFNoYXBlID0gaW5JbmZvLnNoYXBlSW5mby50ZXhTaGFwZTtcbiAgY29uc3Qgb3V0VGV4U2hhcGUgPSBvdXRTaGFwZUluZm8udGV4U2hhcGU7XG5cbiAgbGV0IHJlcyA9ICcnO1xuICBzd2l0Y2ggKHNoYXBlLmxlbmd0aCkge1xuICAgIGNhc2UgMDpcbiAgICAgIHJlcyArPSBnZXRTYW1wbGVyU2NhbGFyKGluSW5mby5uYW1lKTtcbiAgICAgIGJyZWFrO1xuICAgIGNhc2UgMTpcbiAgICAgIHJlcyArPSBnZXRTYW1wbGVyMUQoaW5JbmZvLm5hbWUsIHRleFNoYXBlKTtcbiAgICAgIGJyZWFrO1xuICAgIGNhc2UgMjpcbiAgICAgIHJlcyArPSBnZXRTYW1wbGVyMkQoaW5JbmZvLm5hbWUsIHNoYXBlIGFzIFtudW1iZXIsIG51bWJlcl0sIHRleFNoYXBlKTtcbiAgICAgIGJyZWFrO1xuICAgIGNhc2UgMzpcbiAgICAgIHJlcyArPSBnZXRTYW1wbGVyM0QoXG4gICAgICAgICAgaW5JbmZvLm5hbWUsIHNoYXBlIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgdGV4U2hhcGUpO1xuICAgICAgYnJlYWs7XG4gICAgY2FzZSA0OlxuICAgICAgcmVzICs9IGdldFNhbXBsZXI0RChcbiAgICAgICAgICBpbkluZm8ubmFtZSwgc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIHRleFNoYXBlKTtcbiAgICAgIGJyZWFrO1xuICAgIGRlZmF1bHQ6XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYCR7c2hhcGUubGVuZ3RofS1EIGlucHV0IHNhbXBsaW5nYCArXG4gICAgICAgICAgYCBpcyBub3QgeWV0IHN1cHBvcnRlZGApO1xuICB9XG4gIC8vIElmIGlucHV0IGFuZCBvdXRwdXQgaGF2ZSBtYXRjaGluZyBsb2dpY2FsIHNoYXBlcywgYWRkXG4gIC8vIGdldFRleE5hbWVBdE91dENvb3JkKCkgbWV0aG9kIHRoYXQgc2FtcGxlcyB0aGUgaW5wdXQgdGV4dHVyZSB1c2luZyB0aGVcbiAgLy8gb3V0cHV0IGNvb3JkaW5hdGVzLlxuICBpZiAoYnJvYWRjYXN0IHx8XG4gICAgICB1dGlsLmFycmF5c0VxdWFsKFxuICAgICAgICAgIGluSW5mby5zaGFwZUluZm8ubG9naWNhbFNoYXBlLCBvdXRTaGFwZUluZm8ubG9naWNhbFNoYXBlKSkge1xuICAgIHJlcyArPVxuICAgICAgICBnZXRTYW1wbGVyQXRPdXRwdXRDb29yZHMoaW5JbmZvLm5hbWUsIHRleFNoYXBlLCBvdXRUZXhTaGFwZSwgYnJvYWRjYXN0KTtcbiAgfVxuICByZXMgKz0gZ2V0U2FtcGxlckZsYXQoaW5JbmZvLm5hbWUsIHRleFNoYXBlKTtcbiAgcmV0dXJuIHJlcztcbn1cblxuZnVuY3Rpb24gZ2V0T3V0cHV0U2FtcGxpbmdTbmlwcGV0KFxuICAgIG91dFNoYXBlOiBudW1iZXJbXSwgb3V0VGV4U2hhcGU6IFtudW1iZXIsIG51bWJlcl0pOiBzdHJpbmcge1xuICBzd2l0Y2ggKG91dFNoYXBlLmxlbmd0aCkge1xuICAgIGNhc2UgMDpcbiAgICAgIC8vIERvZXNuJ3QgbWFrZSBzZW5zZSB0byBjYWxsIGdldE91dHB1dENvb3JkcygpIHdoZW4gb3V0cHV0IGlzIHNjYWxhci5cbiAgICAgIHJldHVybiAnJztcbiAgICBjYXNlIDE6XG4gICAgICByZXR1cm4gZ2V0T3V0cHV0MURDb29yZHMob3V0U2hhcGUgYXMgW251bWJlcl0sIG91dFRleFNoYXBlKTtcbiAgICBjYXNlIDI6XG4gICAgICByZXR1cm4gZ2V0T3V0cHV0MkRDb29yZHMob3V0U2hhcGUgYXMgW251bWJlciwgbnVtYmVyXSwgb3V0VGV4U2hhcGUpO1xuICAgIGNhc2UgMzpcbiAgICAgIHJldHVybiBnZXRPdXRwdXQzRENvb3JkcyhcbiAgICAgICAgICBvdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIG91dFRleFNoYXBlKTtcbiAgICBjYXNlIDQ6XG4gICAgICByZXR1cm4gZ2V0T3V0cHV0NERDb29yZHMoXG4gICAgICAgICAgb3V0U2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIG91dFRleFNoYXBlKTtcbiAgICBkZWZhdWx0OlxuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGAke291dFNoYXBlLmxlbmd0aH0tRCBvdXRwdXQgc2FtcGxpbmcgaXMgbm90IHlldCBzdXBwb3J0ZWRgKTtcbiAgfVxufVxuXG5jb25zdCBTQU1QTEVfMURfU05JUFBFVCA9IGBcbnZlYzIgVVZmcm9tMUQoZmxvYXQgdGV4TnVtUiwgZmxvYXQgdGV4TnVtQywgZmxvYXQgaW5kZXgpIHtcbiAgZmxvYXQgdGV4UiA9IGZsb29yKGluZGV4IC8gdGV4TnVtQyk7XG4gIGZsb2F0IHRleEMgPSBtb2QoaW5kZXgsIHRleE51bUMpO1xuICByZXR1cm4gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC8gdmVjMih0ZXhOdW1DLCB0ZXhOdW1SKTtcbn1cbmA7XG5cbmNvbnN0IFNBTVBMRV8yRF9TTklQUEVUID0gYFxudmVjMiBVVmZyb20yRChmbG9hdCB0ZXhOdW1SLCBmbG9hdCB0ZXhOdW1DLCBmbG9hdCBudW1DLCBmbG9hdCByb3csXG4gICAgZmxvYXQgY29sKSB7XG4gIGZsb2F0IGluZGV4ID0gZG90KHZlYzIocm93LCBjb2wpLCB2ZWMyKG51bUMsIDEuMCkpO1xuICBmbG9hdCB0ZXhSID0gZmxvb3IoaW5kZXggLyB0ZXhOdW1DKTtcbiAgZmxvYXQgdGV4QyA9IG1vZChpbmRleCwgdGV4TnVtQyk7XG4gIHJldHVybiAodmVjMih0ZXhDLCB0ZXhSKSArIGhhbGZDUikgLyB2ZWMyKHRleE51bUMsIHRleE51bVIpO1xufVxuYDtcblxuY29uc3QgU0FNUExFXzNEX1NOSVBQRVQgPSBgXG52ZWMyIFVWZnJvbTNEKGZsb2F0IHRleE51bVIsIGZsb2F0IHRleE51bUMsIGZsb2F0IHN0cmlkZTAsXG4gICAgZmxvYXQgc3RyaWRlMSwgZmxvYXQgcm93LCBmbG9hdCBjb2wsIGZsb2F0IGRlcHRoKSB7XG4gIGZsb2F0IGluZGV4ID0gZG90KHZlYzMocm93LCBjb2wsIGRlcHRoKSwgdmVjMyhzdHJpZGUwLCBzdHJpZGUxLCAxLjApKTtcbiAgZmxvYXQgdGV4UiA9IGZsb29yKGluZGV4IC8gdGV4TnVtQyk7XG4gIGZsb2F0IHRleEMgPSBtb2QoaW5kZXgsIHRleE51bUMpO1xuICByZXR1cm4gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC8gdmVjMih0ZXhOdW1DLCB0ZXhOdW1SKTtcbn1cbmA7XG5cbmNvbnN0IFNBTVBMRV80RF9TTklQUEVUID0gYFxudmVjMiBVVmZyb200RChmbG9hdCB0ZXhOdW1SLCBmbG9hdCB0ZXhOdW1DLCBmbG9hdCBzdHJpZGUwLFxuICAgIGZsb2F0IHN0cmlkZTEsIGZsb2F0IHN0cmlkZTIsIGZsb2F0IHJvdywgZmxvYXQgY29sLCBmbG9hdCBkZXB0aCxcbiAgICBmbG9hdCBkZXB0aDIpIHtcbiAgZmxvYXQgaW5kZXggPSBkb3QodmVjNChyb3csIGNvbCwgZGVwdGgsIGRlcHRoMiksXG4gICAgICAgICAgICAgICAgICAgIHZlYzQoc3RyaWRlMCwgc3RyaWRlMSwgc3RyaWRlMiwgMS4wKSk7XG4gIGZsb2F0IHRleFIgPSBmbG9vcihpbmRleCAvIHRleE51bUMpO1xuICBmbG9hdCB0ZXhDID0gbW9kKGluZGV4LCB0ZXhOdW1DKTtcbiAgcmV0dXJuICh2ZWMyKHRleEMsIHRleFIpICsgaGFsZkNSKSAvIHZlYzIodGV4TnVtQywgdGV4TnVtUik7XG59XG5gO1xuXG5jb25zdCBTSEFERVJfUFJFRklYID0gYFxuICBwcmVjaXNpb24gaGlnaHAgZmxvYXQ7XG4gIHZhcnlpbmcgdmVjMiByZXN1bHRVVjtcbiAgY29uc3QgdmVjMiBoYWxmQ1IgPSB2ZWMyKDAuNSwgMC41KTtcblxuICBmbG9hdCBzYW1wbGUoc2FtcGxlcjJEIHRleHR1cmUsIHZlYzIgdXYpIHtcbiAgICByZXR1cm4gdGV4dHVyZTJEKHRleHR1cmUsIHV2KS5yO1xuICB9XG5cbiAgdm9pZCBzZXRPdXRwdXQoZmxvYXQgdmFsKSB7XG4gICAgZ2xfRnJhZ0NvbG9yID0gdmVjNCh2YWwsIDAsIDAsIDApO1xuICB9XG5cbiAgYm9vbCBpc05hTihmbG9hdCB2YWwpIHtcbiAgICByZXR1cm4gdmFsID09IHZhbCA/IGZhbHNlIDogdHJ1ZTtcbiAgfVxuICAke1NBTVBMRV8xRF9TTklQUEVUfVxuICAke1NBTVBMRV8yRF9TTklQUEVUfVxuICAke1NBTVBMRV8zRF9TTklQUEVUfVxuICAke1NBTVBMRV80RF9TTklQUEVUfVxuYDtcblxuZnVuY3Rpb24gZ2V0T3V0cHV0MURDb29yZHMoXG4gICAgc2hhcGU6IFtudW1iZXJdLCB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSk6IHN0cmluZyB7XG4gIGlmICh0ZXhTaGFwZVswXSA9PT0gMSkge1xuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICAgIHJldHVybiBmbG9vcihnbF9GcmFnQ29vcmQueCk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICBpZiAodGV4U2hhcGVbMV0gPT09IDEpIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgICByZXR1cm4gZmxvb3IoZ2xfRnJhZ0Nvb3JkLnkpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgcmV0dXJuIGBcbiAgICBmbG9hdCBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICB2ZWMyIHJlc1RleFJDID0gZmxvb3IoZ2xfRnJhZ0Nvb3JkLnl4KTtcbiAgICAgIHJldHVybiBkb3QocmVzVGV4UkMsIHZlYzIoJHt0ZXhTaGFwZVsxXX0uMCwgMS4wKSk7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRPdXRwdXQzRENvb3JkcyhcbiAgICBzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSk6IHN0cmluZyB7XG4gIGNvbnN0IHN0cmlkZTAgPSBzaGFwZVsxXSAqIHNoYXBlWzJdO1xuICBjb25zdCBzdHJpZGUxID0gc2hhcGVbMl07XG4gIHJldHVybiBgXG4gICAgdmVjMyBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICB2ZWMyIHJlc1RleFJDID0gZmxvb3IoZ2xfRnJhZ0Nvb3JkLnl4KTtcbiAgICAgIGZsb2F0IGluZGV4ID0gZG90KHJlc1RleFJDLCB2ZWMyKCR7dGV4U2hhcGVbMV19LjAsIDEuMCkpO1xuICAgICAgZmxvYXQgciA9IGZsb29yKGluZGV4IC8gJHtzdHJpZGUwfS4wKTtcbiAgICAgIGluZGV4IC09IHIgKiAke3N0cmlkZTB9LjA7XG4gICAgICBmbG9hdCBjID0gZmxvb3IoaW5kZXggLyAke3N0cmlkZTF9LjApO1xuICAgICAgZmxvYXQgZCA9IG1vZChpbmRleCwgJHtzdHJpZGUxfS4wKTtcbiAgICAgIHJldHVybiB2ZWMzKHIsIGMsIGQpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0T3V0cHV0NERDb29yZHMoXG4gICAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogc3RyaW5nIHtcbiAgY29uc3Qgc3RyaWRlMiA9IHNoYXBlWzNdO1xuICBjb25zdCBzdHJpZGUxID0gc2hhcGVbMl0gKiBzdHJpZGUyO1xuICBjb25zdCBzdHJpZGUwID0gc2hhcGVbMV0gKiBzdHJpZGUxO1xuICByZXR1cm4gYFxuICAgIHZlYzQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgdmVjMiByZXNUZXhSQyA9IGZsb29yKGdsX0ZyYWdDb29yZC55eCk7XG4gICAgICBmbG9hdCBpbmRleCA9IGRvdChyZXNUZXhSQywgdmVjMigke3RleFNoYXBlWzFdfS4wLCAxLjApKTtcblxuICAgICAgZmxvYXQgciA9IGZsb29yKGluZGV4IC8gJHtzdHJpZGUwfS4wKTtcbiAgICAgIGluZGV4IC09IHIgKiAke3N0cmlkZTB9LjA7XG5cbiAgICAgIGZsb2F0IGMgPSBmbG9vcihpbmRleCAvICR7c3RyaWRlMX0uMCk7XG4gICAgICBpbmRleCAtPSBjICogJHtzdHJpZGUxfS4wO1xuXG4gICAgICBmbG9hdCBkID0gZmxvb3IoaW5kZXggLyAke3N0cmlkZTJ9LjApO1xuICAgICAgZmxvYXQgZDIgPSBtb2QoaW5kZXgsICR7c3RyaWRlMn0uMCk7XG5cbiAgICAgIHJldHVybiB2ZWM0KHIsIGMsIGQsIGQyKTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldE91dHB1dDJEQ29vcmRzKFxuICAgIHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLCB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSk6IHN0cmluZyB7XG4gIGlmICh1dGlsLmFycmF5c0VxdWFsKHNoYXBlLCB0ZXhTaGFwZSkpIHtcbiAgICByZXR1cm4gYFxuICAgICAgdmVjMiBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICAgIHJldHVybiBmbG9vcihnbF9GcmFnQ29vcmQueXgpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgcmV0dXJuIGBcbiAgICB2ZWMyIGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgIHZlYzIgcmVzVGV4UkMgPSBmbG9vcihnbF9GcmFnQ29vcmQueXgpO1xuICAgICAgZmxvYXQgaW5kZXggPSBkb3QocmVzVGV4UkMsIHZlYzIoJHt0ZXhTaGFwZVsxXX0uMCwgMS4wKSk7XG4gICAgICBmbG9hdCByID0gZmxvb3IoaW5kZXggLyAke3NoYXBlWzFdfS4wKTtcbiAgICAgIGZsb2F0IGMgPSBtb2QoaW5kZXgsICR7c2hhcGVbMV19LjApO1xuICAgICAgcmV0dXJuIHZlYzIociwgYyk7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRTYW1wbGVyU2NhbGFyKHRleE5hbWU6IHN0cmluZyk6IHN0cmluZyB7XG4gIGNvbnN0IGZ1bmNOYW1lID0gJ2dldCcgKyB0ZXhOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKTtcbiAgcmV0dXJuIGBcbiAgICBmbG9hdCAke2Z1bmNOYW1lfSgpIHtcbiAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgaGFsZkNSKTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldFNhbXBsZXIxRCh0ZXhOYW1lOiBzdHJpbmcsIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogc3RyaW5nIHtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCB0UiA9IHRleFNoYXBlWzBdO1xuICBjb25zdCB0QyA9IHRleFNoYXBlWzFdO1xuICBpZiAodGV4U2hhcGVbMF0gPT09IDEgJiYgdGV4U2hhcGVbMV0gPT09IDEpIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oZmxvYXQgaW5kZXgpIHtcbiAgICAgICAgcmV0dXJuIHNhbXBsZSgke3RleE5hbWV9LCBoYWxmQ1IpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgaWYgKHRleFNoYXBlWzFdID09PSAxKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IGluZGV4KSB7XG4gICAgICAgIHZlYzIgdXYgPSB2ZWMyKDAuNSwgKGluZGV4ICsgMC41KSAvICR7dFJ9LjApO1xuICAgICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG4gIGlmICh0ZXhTaGFwZVswXSA9PT0gMSkge1xuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShmbG9hdCBpbmRleCkge1xuICAgICAgICB2ZWMyIHV2ID0gdmVjMigoaW5kZXggKyAwLjUpIC8gJHt0Q30uMCwgMC41KTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZSgke3RleE5hbWV9LCB1dik7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICByZXR1cm4gYFxuICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IGluZGV4KSB7XG4gICAgICB2ZWMyIHV2ID0gVVZmcm9tMUQoJHt0Un0uMCwgJHt0Q30uMCwgaW5kZXgpO1xuICAgICAgcmV0dXJuIHNhbXBsZSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRTYW1wbGVyM0QoXG4gICAgdGV4TmFtZTogc3RyaW5nLCBzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogc3RyaW5nIHtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCB0UiA9IHRleFNoYXBlWzBdO1xuICBjb25zdCB0QyA9IHRleFNoYXBlWzFdO1xuICBjb25zdCBzdHJpZGUwID0gc2hhcGVbMV0gKiBzaGFwZVsyXTtcbiAgY29uc3Qgc3RyaWRlMSA9IHNoYXBlWzJdO1xuICBpZiAodEMgPT09IHN0cmlkZTApIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oZmxvYXQgcm93LCBmbG9hdCBjb2wsIGZsb2F0IGRlcHRoKSB7XG4gICAgICAgIGZsb2F0IHRleFIgPSByb3c7XG4gICAgICAgIGZsb2F0IHRleEMgPSBkb3QodmVjMihjb2wsIGRlcHRoKSwgdmVjMigke3N0cmlkZTF9LCAxLjApKTtcbiAgICAgICAgdmVjMiB1diA9ICh2ZWMyKHRleEMsIHRleFIpICsgaGFsZkNSKSAvIHZlYzIoJHt0Q30uMCwgJHt0Un0uMCk7XG4gICAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgcmV0dXJuIGBcbiAgICBmbG9hdCAke2Z1bmNOYW1lfShmbG9hdCByb3csIGZsb2F0IGNvbCwgZmxvYXQgZGVwdGgpIHtcbiAgICAgIHZlYzIgdXYgPSBVVmZyb20zRCgke3RSfS4wLCAke3RDfS4wLCAke3N0cmlkZTB9LjAsICR7c3RyaWRlMX0uMCwgcm93LFxuICAgICAgICBjb2wsIGRlcHRoKTtcbiAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0U2FtcGxlcjREKFxuICAgIHRleE5hbWU6IHN0cmluZywgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogc3RyaW5nIHtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCB0UiA9IHRleFNoYXBlWzBdO1xuICBjb25zdCB0QyA9IHRleFNoYXBlWzFdO1xuICBjb25zdCBzdHJpZGUyID0gc2hhcGVbM107XG4gIGNvbnN0IHN0cmlkZTEgPSBzaGFwZVsyXSAqIHN0cmlkZTI7XG4gIGNvbnN0IHN0cmlkZTAgPSBzaGFwZVsxXSAqIHN0cmlkZTE7XG5cbiAgaWYgKHRDID09PSBzdHJpZGUwKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IHJvdywgZmxvYXQgY29sLCBmbG9hdCBkZXB0aCwgZmxvYXQgZGVwdGgyKSB7XG4gICAgICAgIGZsb2F0IHRleFIgPSByb3c7XG4gICAgICAgIGZsb2F0IHRleEMgPSBkb3QodmVjMyhjb2wsIGRlcHRoLCBkZXB0aDIpLFxuICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzMoJHtzdHJpZGUxfS4wLCAke3N0cmlkZTJ9LjAsIDEuMCkpO1xuICAgICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC8gdmVjMigke3RDfS4wLCAke3RSfS4wKTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZSgke3RleE5hbWV9LCB1dik7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICByZXR1cm4gYFxuICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IHJvdywgZmxvYXQgY29sLCBmbG9hdCBkZXB0aCwgZmxvYXQgZGVwdGgyKSB7XG4gICAgICB2ZWMyIHV2ID0gVVZmcm9tNEQoJHt0Un0uMCwgJHt0Q30uMCwgJHtzdHJpZGUwfS4wLCAke3N0cmlkZTF9LjAsXG4gICAgICAgICAgJHtzdHJpZGUyfS4wLCByb3csIGNvbCwgZGVwdGgsIGRlcHRoMik7XG4gICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldFNhbXBsZXIyRChcbiAgICB0ZXhOYW1lOiBzdHJpbmcsIHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogc3RyaW5nIHtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCB0UiA9IHRleFNoYXBlWzBdO1xuICBjb25zdCB0QyA9IHRleFNoYXBlWzFdO1xuICBpZiAodXRpbC5hcnJheXNFcXVhbChzaGFwZSwgdGV4U2hhcGUpKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IHJvdywgZmxvYXQgY29sKSB7XG4gICAgICAgIHZlYzIgdXYgPSAodmVjMihjb2wsIHJvdykgKyBoYWxmQ1IpIC8gdmVjMigke3RDfS4wLCAke3RSfS4wKTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZSgke3RleE5hbWV9LCB1dik7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICBpZiAodEMgPT09IDEpIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oZmxvYXQgcm93LCBmbG9hdCBjb2wpIHtcbiAgICAgICAgZmxvYXQgaW5kZXggPSBkb3QodmVjMihyb3csIGNvbCksIHZlYzIoJHtzaGFwZVsxXX0uMCwgMS4wKSk7XG4gICAgICAgIHZlYzIgdXYgPSB2ZWMyKDAuNSwgKGluZGV4ICsgMC41KSAvICR7dFJ9LjApO1xuICAgICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG4gIGlmICh0UiA9PT0gMSkge1xuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShmbG9hdCByb3csIGZsb2F0IGNvbCkge1xuICAgICAgICBmbG9hdCBpbmRleCA9IGRvdCh2ZWMyKHJvdywgY29sKSwgdmVjMigke3NoYXBlWzFdfS4wLCAxLjApKTtcbiAgICAgICAgdmVjMiB1diA9IHZlYzIoKGluZGV4ICsgMC41KSAvICR7dEN9LjAsIDAuNSk7XG4gICAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgcmV0dXJuIGBcbiAgICBmbG9hdCAke2Z1bmNOYW1lfShmbG9hdCByb3csIGZsb2F0IGNvbCkge1xuICAgICAgdmVjMiB1diA9IFVWZnJvbTJEKCR7dFJ9LjAsICR7dEN9LjAsICR7c2hhcGVbMV19LjAsIHJvdywgY29sKTtcbiAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0U2FtcGxlckZsYXQodGV4TmFtZTogc3RyaW5nLCB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSk6IHN0cmluZyB7XG4gIGNvbnN0IGZ1bmNOYW1lID1cbiAgICAgICdnZXQnICsgdGV4TmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSArIHRleE5hbWUuc2xpY2UoMSkgKyAnRmxhdCc7XG4gIGNvbnN0IHROdW1SID0gdGV4U2hhcGVbMF07XG4gIGNvbnN0IHROdW1DID0gdGV4U2hhcGVbMV07XG4gIGlmICh0TnVtQyA9PT0gMSAmJiB0TnVtUiA9PT0gMSkge1xuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShmbG9hdCBpbmRleCkge1xuICAgICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIGhhbGZDUik7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICBpZiAodE51bUMgPT09IDEpIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oZmxvYXQgaW5kZXgpIHtcbiAgICAgICAgdmVjMiB1diA9IHZlYzIoMC41LCAoaW5kZXggKyAwLjUpIC8gJHt0TnVtUn0uMCk7XG4gICAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgaWYgKHROdW1SID09PSAxKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IGluZGV4KSB7XG4gICAgICAgIHZlYzIgdXYgPSB2ZWMyKChpbmRleCArIDAuNSkgLyAke3ROdW1DfS4wLCAwLjUpO1xuICAgICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG4gIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oZmxvYXQgaW5kZXgpIHtcbiAgICAgIGZsb2F0IHRleFIgPSBmbG9vcihpbmRleCAvICR7dE51bUN9LjApO1xuICAgICAgZmxvYXQgdGV4QyA9IG1vZChpbmRleCwgJHt0TnVtQ30uMCk7XG4gICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC8gdmVjMigke3ROdW1DfS4wLCAke3ROdW1SfS4wKTtcbiAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0U2FtcGxlckF0T3V0cHV0Q29vcmRzKFxuICAgIHRleE5hbWU6IHN0cmluZywgaW5UZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSxcbiAgICBvdXRUZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSwgYnJvYWRjYXN0OiBib29sZWFuKSB7XG4gIGNvbnN0IGZ1bmNOYW1lID0gJ2dldCcgKyB0ZXhOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKSArXG4gICAgICAnQXRPdXRDb29yZHMnO1xuICBpZiAodXRpbC5hcnJheXNFcXVhbChpblRleFNoYXBlLCBvdXRUZXhTaGFwZSkpIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oKSB7XG4gICAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgcmVzdWx0VVYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgY29uc3QgaW5TaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKGluVGV4U2hhcGUpO1xuICBjb25zdCBicm9hZGNhc3RTbmlwcGV0ID0gYnJvYWRjYXN0ID8gYGluZGV4ID0gbW9kKGluZGV4LCAke2luU2l6ZX0uMCk7YCA6ICcnO1xuXG4gIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oKSB7XG4gICAgICB2ZWMyIHJlc1RleFJDID0gZmxvb3IoZ2xfRnJhZ0Nvb3JkLnl4KTtcbiAgICAgIGZsb2F0IGluZGV4ID0gZG90KHJlc1RleFJDLCB2ZWMyKCR7b3V0VGV4U2hhcGVbMV19LjAsIDEuMCkpO1xuICAgICAgJHticm9hZGNhc3RTbmlwcGV0fVxuICAgICAgZmxvYXQgdGV4UiA9IGZsb29yKGluZGV4IC8gJHtpblRleFNoYXBlWzFdfS4wKTtcbiAgICAgIGZsb2F0IHRleEMgPSBtb2QoaW5kZXgsICR7aW5UZXhTaGFwZVsxXX0uMCk7XG4gICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC9cbiAgICAgICAgICAgICAgICAgdmVjMigke2luVGV4U2hhcGVbMV19LjAsICR7aW5UZXhTaGFwZVswXX0uMCk7XG4gICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICB9XG4gIGA7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRVbnBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KFxuICAgIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogW251bWJlciwgbnVtYmVyXSB7XG4gIHJldHVybiBbY29sdW1ucywgcm93c107XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRVbnBhY2tlZEFycmF5U2l6ZUZyb21NYXRyaXhTaXplKFxuICAgIG1hdHJpeFNpemU6IG51bWJlciwgY2hhbm5lbHNQZXJUZXh0dXJlOiBudW1iZXIpOiBudW1iZXIge1xuICByZXR1cm4gbWF0cml4U2l6ZSAqIGNoYW5uZWxzUGVyVGV4dHVyZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldENvbG9yTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQoXG4gICAgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgcmV0dXJuIFtjb2x1bW5zICogNCwgcm93c107XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRNYXRyaXhTaXplRnJvbVVucGFja2VkQXJyYXlTaXplKFxuICAgIHVucGFja2VkU2l6ZTogbnVtYmVyLCBjaGFubmVsc1BlclRleHR1cmU6IG51bWJlcik6IG51bWJlciB7XG4gIGlmICh1bnBhY2tlZFNpemUgJSBjaGFubmVsc1BlclRleHR1cmUgIT09IDApIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICd1bnBhY2tlZFNpemUgKCcgKyB1bnBhY2tlZFNpemUgKyAnKSBtdXN0IGJlIGEgbXVsdGlwbGUgb2YgJyArXG4gICAgICAgIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG4gIH1cbiAgcmV0dXJuIHVucGFja2VkU2l6ZSAvIGNoYW5uZWxzUGVyVGV4dHVyZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGVuY29kZU1hdHJpeFRvVW5wYWNrZWRBcnJheShcbiAgICBtYXRyaXg6IEZsb2F0MzJBcnJheSwgdW5wYWNrZWRBcnJheTogRmxvYXQzMkFycmF5LFxuICAgIGNoYW5uZWxzUGVyVGV4dHVyZTogbnVtYmVyKSB7XG4gIGNvbnN0IHJlcXVpcmVkU2l6ZSA9XG4gICAgICBnZXRVbnBhY2tlZEFycmF5U2l6ZUZyb21NYXRyaXhTaXplKG1hdHJpeC5sZW5ndGgsIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG4gIGlmICh1bnBhY2tlZEFycmF5Lmxlbmd0aCA8IHJlcXVpcmVkU2l6ZSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ3VucGFja2VkQXJyYXkgbGVuZ3RoICgnICsgdW5wYWNrZWRBcnJheS5sZW5ndGggK1xuICAgICAgICAnKSBtdXN0IGJlID49ICcgKyByZXF1aXJlZFNpemUpO1xuICB9XG4gIGxldCBkc3QgPSAwO1xuICBmb3IgKGxldCBzcmMgPSAwOyBzcmMgPCBtYXRyaXgubGVuZ3RoOyArK3NyYykge1xuICAgIHVucGFja2VkQXJyYXlbZHN0XSA9IG1hdHJpeFtzcmNdO1xuICAgIGRzdCArPSBjaGFubmVsc1BlclRleHR1cmU7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRlY29kZU1hdHJpeEZyb21VbnBhY2tlZEFycmF5KFxuICAgIHVucGFja2VkQXJyYXk6IEZsb2F0MzJBcnJheSwgbWF0cml4OiBGbG9hdDMyQXJyYXksXG4gICAgY2hhbm5lbHNQZXJUZXh0dXJlOiBudW1iZXIpIHtcbiAgY29uc3QgcmVxdWlyZWRTaXplID0gZ2V0TWF0cml4U2l6ZUZyb21VbnBhY2tlZEFycmF5U2l6ZShcbiAgICAgIHVucGFja2VkQXJyYXkubGVuZ3RoLCBjaGFubmVsc1BlclRleHR1cmUpO1xuICBpZiAobWF0cml4Lmxlbmd0aCA8IHJlcXVpcmVkU2l6ZSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ21hdHJpeCBsZW5ndGggKCcgKyBtYXRyaXgubGVuZ3RoICsgJykgbXVzdCBiZSA+PSAnICsgcmVxdWlyZWRTaXplKTtcbiAgfVxuICBsZXQgZHN0ID0gMDtcbiAgZm9yIChsZXQgc3JjID0gMDsgc3JjIDwgdW5wYWNrZWRBcnJheS5sZW5ndGg7IHNyYyArPSBjaGFubmVsc1BlclRleHR1cmUpIHtcbiAgICBtYXRyaXhbZHN0KytdID0gdW5wYWNrZWRBcnJheVtzcmNdO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChcbiAgICByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFtudW1iZXIsIG51bWJlcl0ge1xuICByZXR1cm4gW01hdGguY2VpbChjb2x1bW5zIC8gMiksIE1hdGguY2VpbChyb3dzIC8gMildO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0UGFja2VkUkdCQUFycmF5U2l6ZUZyb21NYXRyaXhTaGFwZShcbiAgICByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IG51bWJlciB7XG4gIGNvbnN0IFt3LCBoXSA9IGdldFBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KHJvd3MsIGNvbHVtbnMpO1xuICByZXR1cm4gdyAqIGggKiA0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZW5jb2RlTWF0cml4VG9QYWNrZWRSR0JBKFxuICAgIG1hdHJpeDogRmxvYXQzMkFycmF5LCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcixcbiAgICBwYWNrZWRSR0JBOiBGbG9hdDMyQXJyYXkpIHtcbiAgY29uc3QgcmVxdWlyZWRTaXplID0gZ2V0UGFja2VkUkdCQUFycmF5U2l6ZUZyb21NYXRyaXhTaGFwZShyb3dzLCBjb2x1bW5zKTtcbiAgaWYgKHBhY2tlZFJHQkEubGVuZ3RoIDwgcmVxdWlyZWRTaXplKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAncGFja2VkUkdCQSBsZW5ndGggKCcgKyBwYWNrZWRSR0JBLmxlbmd0aCArXG4gICAgICAgICcpIG11c3QgYmUgPj0gJyArIHJlcXVpcmVkU2l6ZSk7XG4gIH1cbiAgLypcbiAgICBVbnBhY2tlZCBtYXRyaXgsIHJvdy1tYWpvciBvcmRlciBpbiBGbG9hdDMyQXJyYXlbMTZdOiAgQSBCIEMgRFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBFIEYgRyBIXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIEkgSiBLIExcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgTSBOIE8gUFxuXG4gICAgUGFja2VkIG1hdHJpeCwgMngyIFJHQkEzMiB0ZXh0dXJlIChtZW1vcnkgdmlldyk6ICAgICAgIEFCRUYgQ0RHSCBJSk1OIEtMT1BcblxuICAgIFBhY2tlZCBtYXRyaXgsIDJ4MiBSR0JBMzIgdGV4dHVyZSAobWF0cml4IHZpZXcpOiAgICAgICBBQnxDRFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBFRnxHSFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAtLSstLVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBJSnxLTFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBNTnxPUFxuICAgKi9cbiAgY29uc3QgW3RleHR1cmVXaWR0aCwgdGV4dHVyZUhlaWdodF0gPVxuICAgICAgZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gIGNvbnN0IG9kZFdpZHRoID0gKGNvbHVtbnMgJSAyKSA9PT0gMTtcbiAgY29uc3Qgb2RkSGVpZ2h0ID0gKHJvd3MgJSAyKSA9PT0gMTtcbiAgY29uc3Qgd2lkdGhJbkZ1bGxCbG9ja3MgPSBNYXRoLmZsb29yKGNvbHVtbnMgLyAyKTtcbiAgY29uc3QgaGVpZ2h0SW5GdWxsQmxvY2tzID0gTWF0aC5mbG9vcihyb3dzIC8gMik7XG5cbiAgLy8gbG9vcCBvdmVyIGZ1bGwgMngyIGJsb2Nrc1xuICB7XG4gICAgY29uc3QgZHN0U3RyaWRlID0gKG9kZFdpZHRoID8gNCA6IDApO1xuICAgIGNvbnN0IG9uZVJvdyA9IGNvbHVtbnM7XG4gICAgbGV0IGRzdCA9IDA7XG4gICAgZm9yIChsZXQgYmxvY2tZID0gMDsgYmxvY2tZIDwgaGVpZ2h0SW5GdWxsQmxvY2tzOyArK2Jsb2NrWSkge1xuICAgICAgY29uc3QgbWF0cml4U3JjUm93ID0gKGJsb2NrWSAqIDIgKiBjb2x1bW5zKTtcbiAgICAgIGZvciAobGV0IGJsb2NrWCA9IDA7IGJsb2NrWCA8IHdpZHRoSW5GdWxsQmxvY2tzOyArK2Jsb2NrWCkge1xuICAgICAgICBjb25zdCBtYXRyaXhTcmNDb2wgPSBibG9ja1ggKiAyO1xuICAgICAgICBjb25zdCBzcmMgPSBtYXRyaXhTcmNSb3cgKyBtYXRyaXhTcmNDb2w7XG4gICAgICAgIHBhY2tlZFJHQkFbZHN0XSA9IG1hdHJpeFtzcmNdO1xuICAgICAgICBwYWNrZWRSR0JBW2RzdCArIDFdID0gbWF0cml4W3NyYyArIDFdO1xuICAgICAgICBwYWNrZWRSR0JBW2RzdCArIDJdID0gbWF0cml4W3NyYyArIG9uZVJvd107XG4gICAgICAgIHBhY2tlZFJHQkFbZHN0ICsgM10gPSBtYXRyaXhbc3JjICsgb25lUm93ICsgMV07XG4gICAgICAgIGRzdCArPSA0O1xuICAgICAgfVxuICAgICAgZHN0ICs9IGRzdFN0cmlkZTtcbiAgICB9XG4gIH1cblxuICAvLyBsb29wIGRvd24gZmluYWwgb2RkIGNvbHVtblxuICBpZiAob2RkV2lkdGgpIHtcbiAgICBsZXQgc3JjID0gY29sdW1ucyAtIDE7XG4gICAgbGV0IGRzdCA9ICh0ZXh0dXJlV2lkdGggLSAxKSAqIDQ7XG4gICAgY29uc3Qgc3JjU3RyaWRlID0gMiAqIGNvbHVtbnM7XG4gICAgY29uc3QgZHN0U3RyaWRlID0gdGV4dHVyZVdpZHRoICogNDtcbiAgICBmb3IgKGxldCBibG9ja1kgPSAwOyBibG9ja1kgPCBoZWlnaHRJbkZ1bGxCbG9ja3M7ICsrYmxvY2tZKSB7XG4gICAgICBwYWNrZWRSR0JBW2RzdF0gPSBtYXRyaXhbc3JjXTtcbiAgICAgIHBhY2tlZFJHQkFbZHN0ICsgMl0gPSBtYXRyaXhbc3JjICsgY29sdW1uc107XG4gICAgICBzcmMgKz0gc3JjU3RyaWRlO1xuICAgICAgZHN0ICs9IGRzdFN0cmlkZTtcbiAgICB9XG4gIH1cblxuICAvLyBsb29wIGFjcm9zcyBmaW5hbCByb3dcbiAgaWYgKG9kZEhlaWdodCkge1xuICAgIGxldCBzcmMgPSAocm93cyAtIDEpICogY29sdW1ucztcbiAgICBsZXQgZHN0ID0gKHRleHR1cmVIZWlnaHQgLSAxKSAqIHRleHR1cmVXaWR0aCAqIDQ7XG4gICAgZm9yIChsZXQgYmxvY2tYID0gMDsgYmxvY2tYIDwgd2lkdGhJbkZ1bGxCbG9ja3M7ICsrYmxvY2tYKSB7XG4gICAgICBwYWNrZWRSR0JBW2RzdCsrXSA9IG1hdHJpeFtzcmMrK107XG4gICAgICBwYWNrZWRSR0JBW2RzdCsrXSA9IG1hdHJpeFtzcmMrK107XG4gICAgICBkc3QgKz0gMjtcbiAgICB9XG4gIH1cblxuICAvLyBmaWxsIGluIGJvdHRvbS1yaWdodCB0ZXhlbFxuICBpZiAob2RkV2lkdGggJiYgb2RkSGVpZ2h0KSB7XG4gICAgcGFja2VkUkdCQVtwYWNrZWRSR0JBLmxlbmd0aCAtIDRdID0gbWF0cml4W21hdHJpeC5sZW5ndGggLSAxXTtcbiAgfVxuXG4gIHJldHVybiBwYWNrZWRSR0JBO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZGVjb2RlTWF0cml4RnJvbVBhY2tlZFJHQkEoXG4gICAgcGFja2VkUkdCQTogRmxvYXQzMkFycmF5LCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcixcbiAgICBtYXRyaXg6IEZsb2F0MzJBcnJheSk6IEZsb2F0MzJBcnJheSB7XG4gIGNvbnN0IHJlcXVpcmVkU2l6ZSA9IHJvd3MgKiBjb2x1bW5zO1xuICBpZiAocmVxdWlyZWRTaXplIDwgbWF0cml4Lmxlbmd0aCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ21hdHJpeCBsZW5ndGggKCcgKyBtYXRyaXgubGVuZ3RoICsgJykgbXVzdCBiZSA+PSAnICsgcmVxdWlyZWRTaXplKTtcbiAgfVxuICBjb25zdCBvZGRXaWR0aCA9IChjb2x1bW5zICUgMikgPT09IDE7XG4gIGNvbnN0IG9kZEhlaWdodCA9IChyb3dzICUgMikgPT09IDE7XG4gIGNvbnN0IHdpZHRoSW5GdWxsQmxvY2tzID0gTWF0aC5mbG9vcihjb2x1bW5zIC8gMik7XG4gIGNvbnN0IGhlaWdodEluRnVsbEJsb2NrcyA9IE1hdGguZmxvb3Iocm93cyAvIDIpO1xuICBjb25zdCBbdGV4dHVyZVdpZHRoLCB0ZXh0dXJlSGVpZ2h0XSA9XG4gICAgICBnZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcblxuICAvLyBsb29wIG92ZXIgZnVsbCAyeDIgYmxvY2tzXG4gIHtcbiAgICBjb25zdCBzcmNTdHJpZGUgPSBvZGRXaWR0aCA/IDQgOiAwO1xuICAgIGNvbnN0IGRzdFN0cmlkZSA9IGNvbHVtbnMgKyAob2RkV2lkdGggPyAxIDogMCk7XG4gICAgbGV0IHNyYyA9IDA7XG4gICAgbGV0IGRzdFJvdzEgPSAwO1xuICAgIGxldCBkc3RSb3cyID0gY29sdW1ucztcbiAgICBmb3IgKGxldCBibG9ja1kgPSAwOyBibG9ja1kgPCBoZWlnaHRJbkZ1bGxCbG9ja3M7ICsrYmxvY2tZKSB7XG4gICAgICBmb3IgKGxldCBibG9ja1ggPSAwOyBibG9ja1ggPCB3aWR0aEluRnVsbEJsb2NrczsgKytibG9ja1gpIHtcbiAgICAgICAgbWF0cml4W2RzdFJvdzErK10gPSBwYWNrZWRSR0JBW3NyYysrXTtcbiAgICAgICAgbWF0cml4W2RzdFJvdzErK10gPSBwYWNrZWRSR0JBW3NyYysrXTtcbiAgICAgICAgbWF0cml4W2RzdFJvdzIrK10gPSBwYWNrZWRSR0JBW3NyYysrXTtcbiAgICAgICAgbWF0cml4W2RzdFJvdzIrK10gPSBwYWNrZWRSR0JBW3NyYysrXTtcbiAgICAgIH1cbiAgICAgIHNyYyArPSBzcmNTdHJpZGU7XG4gICAgICBkc3RSb3cxICs9IGRzdFN0cmlkZTtcbiAgICAgIGRzdFJvdzIgKz0gZHN0U3RyaWRlO1xuICAgIH1cbiAgfVxuXG4gIC8vIGxvb3AgZG93biBmaW5hbCBjb2x1bW5cbiAgaWYgKG9kZFdpZHRoKSB7XG4gICAgbGV0IHNyYyA9ICh0ZXh0dXJlV2lkdGggLSAxKSAqIDQ7XG4gICAgbGV0IGRzdCA9IGNvbHVtbnMgLSAxO1xuICAgIGNvbnN0IHNyY1N0cmlkZSA9IHRleHR1cmVXaWR0aCAqIDQ7XG4gICAgY29uc3QgZHN0U3RyaWRlID0gMiAqIGNvbHVtbnM7XG4gICAgZm9yIChsZXQgYmxvY2tZID0gMDsgYmxvY2tZIDwgaGVpZ2h0SW5GdWxsQmxvY2tzOyArK2Jsb2NrWSkge1xuICAgICAgbWF0cml4W2RzdF0gPSBwYWNrZWRSR0JBW3NyY107XG4gICAgICBtYXRyaXhbZHN0ICsgY29sdW1uc10gPSBwYWNrZWRSR0JBW3NyYyArIDJdO1xuICAgICAgc3JjICs9IHNyY1N0cmlkZTtcbiAgICAgIGRzdCArPSBkc3RTdHJpZGU7XG4gICAgfVxuICB9XG5cbiAgLy8gbG9vcCBhY3Jvc3MgZmluYWwgcm93XG4gIGlmIChvZGRIZWlnaHQpIHtcbiAgICBsZXQgc3JjID0gKHRleHR1cmVIZWlnaHQgLSAxKSAqIHRleHR1cmVXaWR0aCAqIDQ7XG4gICAgbGV0IGRzdCA9IChyb3dzIC0gMSkgKiBjb2x1bW5zO1xuICAgIGZvciAobGV0IGJsb2NrWCA9IDA7IGJsb2NrWCA8IHdpZHRoSW5GdWxsQmxvY2tzOyArK2Jsb2NrWCkge1xuICAgICAgbWF0cml4W2RzdCsrXSA9IHBhY2tlZFJHQkFbc3JjKytdO1xuICAgICAgbWF0cml4W2RzdCsrXSA9IHBhY2tlZFJHQkFbc3JjKytdO1xuICAgICAgc3JjICs9IDI7XG4gICAgfVxuICB9XG5cbiAgLy8gZmlsbCBpbiBib3R0b20tcmlnaHQgY2VsbFxuICBpZiAob2RkV2lkdGggJiYgb2RkSGVpZ2h0KSB7XG4gICAgbWF0cml4W21hdHJpeC5sZW5ndGggLSAxXSA9IHBhY2tlZFJHQkFbcGFja2VkUkdCQS5sZW5ndGggLSA0XTtcbiAgfVxuXG4gIHJldHVybiBtYXRyaXg7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuL2dwZ3B1X2NvbnRleHQnO1xuXG5leHBvcnQgY2xhc3MgVGV4dHVyZU1hbmFnZXIge1xuICBwcml2YXRlIG51bVVzZWRUZXh0dXJlcyA9IDA7XG4gIHByaXZhdGUgbnVtRnJlZVRleHR1cmVzID0gMDtcbiAgcHJpdmF0ZSBmcmVlVGV4dHVyZXM6IHtbc2hhcGU6IHN0cmluZ106IFdlYkdMVGV4dHVyZVtdfSA9IHt9O1xuICBwcml2YXRlIGxvZ0VuYWJsZWQgPSBmYWxzZTtcbiAgcHJpdmF0ZSB1c2VkVGV4dHVyZUNvdW50OiB7W3NoYXBlOiBzdHJpbmddOiBudW1iZXJ9ID0ge307XG5cbiAgY29uc3RydWN0b3IocHJpdmF0ZSBncGdwdTogR1BHUFVDb250ZXh0KSB7fVxuXG4gIGFjcXVpcmVUZXh0dXJlKHNoYXBlUkM6IFtudW1iZXIsIG51bWJlcl0pOiBXZWJHTFRleHR1cmUge1xuICAgIGNvbnN0IHNoYXBlS2V5ID0gZ2V0S2V5RnJvbVRleHR1cmVTaGFwZShzaGFwZVJDKTtcbiAgICBpZiAoIShzaGFwZUtleSBpbiB0aGlzLmZyZWVUZXh0dXJlcykpIHtcbiAgICAgIHRoaXMuZnJlZVRleHR1cmVzW3NoYXBlS2V5XSA9IFtdO1xuICAgIH1cbiAgICBpZiAoIShzaGFwZUtleSBpbiB0aGlzLnVzZWRUZXh0dXJlQ291bnQpKSB7XG4gICAgICB0aGlzLnVzZWRUZXh0dXJlQ291bnRbc2hhcGVLZXldID0gMDtcbiAgICB9XG4gICAgdGhpcy51c2VkVGV4dHVyZUNvdW50W3NoYXBlS2V5XSsrO1xuXG4gICAgaWYgKHRoaXMuZnJlZVRleHR1cmVzW3NoYXBlS2V5XS5sZW5ndGggPiAwKSB7XG4gICAgICB0aGlzLm51bUZyZWVUZXh0dXJlcy0tO1xuICAgICAgdGhpcy5udW1Vc2VkVGV4dHVyZXMrKztcbiAgICAgIHRoaXMubG9nKCk7XG4gICAgICByZXR1cm4gdGhpcy5mcmVlVGV4dHVyZXNbc2hhcGVLZXldLnNoaWZ0KCk7XG4gICAgfVxuICAgIHRoaXMubnVtVXNlZFRleHR1cmVzKys7XG4gICAgdGhpcy5sb2coKTtcblxuICAgIHJldHVybiB0aGlzLmdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUoc2hhcGVSQ1swXSwgc2hhcGVSQ1sxXSk7XG4gIH1cblxuICByZWxlYXNlVGV4dHVyZSh0ZXh0dXJlOiBXZWJHTFRleHR1cmUsIHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogdm9pZCB7XG4gICAgY29uc3Qgc2hhcGVLZXkgPSBnZXRLZXlGcm9tVGV4dHVyZVNoYXBlKHNoYXBlKTtcbiAgICBpZiAoIShzaGFwZUtleSBpbiB0aGlzLmZyZWVUZXh0dXJlcykpIHtcbiAgICAgIHRoaXMuZnJlZVRleHR1cmVzW3NoYXBlS2V5XSA9IFtdO1xuICAgIH1cbiAgICB0aGlzLmZyZWVUZXh0dXJlc1tzaGFwZUtleV0ucHVzaCh0ZXh0dXJlKTtcbiAgICB0aGlzLm51bUZyZWVUZXh0dXJlcysrO1xuICAgIHRoaXMubnVtVXNlZFRleHR1cmVzLS07XG4gICAgdGhpcy51c2VkVGV4dHVyZUNvdW50W3NoYXBlS2V5XS0tO1xuICAgIHRoaXMubG9nKCk7XG4gIH1cblxuICBwcml2YXRlIGxvZygpIHtcbiAgICBpZiAoIXRoaXMubG9nRW5hYmxlZCkge1xuICAgICAgcmV0dXJuO1xuICAgIH1cbiAgICBjb25zdCB0b3RhbCA9IHRoaXMubnVtRnJlZVRleHR1cmVzICsgdGhpcy5udW1Vc2VkVGV4dHVyZXM7XG4gICAgY29uc29sZS5sb2coXG4gICAgICAgICdGcmVlL1VzZWQnLCB0aGlzLm51bUZyZWVUZXh0dXJlcyArICcgLyAnICsgdGhpcy5udW1Vc2VkVGV4dHVyZXMsXG4gICAgICAgIGAoJHt0b3RhbH0pYCk7XG4gIH1cblxuICBnZXROdW1Vc2VkVGV4dHVyZXMoKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5udW1Vc2VkVGV4dHVyZXM7XG4gIH1cblxuICBnZXROdW1GcmVlVGV4dHVyZXMoKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5udW1GcmVlVGV4dHVyZXM7XG4gIH1cblxuICBkaXNwb3NlKCkge1xuICAgIGZvciAoY29uc3Qgc2hhcGUgaW4gdGhpcy5mcmVlVGV4dHVyZXMpIHtcbiAgICAgIGlmICh0aGlzLmZyZWVUZXh0dXJlcy5oYXNPd25Qcm9wZXJ0eShzaGFwZSkpIHtcbiAgICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0aGlzLmZyZWVUZXh0dXJlc1tzaGFwZV0ubGVuZ3RoOyBpKyspIHtcbiAgICAgICAgICB0aGlzLmdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUodGhpcy5mcmVlVGV4dHVyZXNbc2hhcGVdW2ldKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxufVxuXG5mdW5jdGlvbiBnZXRLZXlGcm9tVGV4dHVyZVNoYXBlKHNoYXBlUm93c0NvbDogW251bWJlciwgbnVtYmVyXSk6IHN0cmluZyB7XG4gIHJldHVybiBzaGFwZVJvd3NDb2xbMF0gKyAnXycgKyBzaGFwZVJvd3NDb2xbMV07XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmxldCBVU0VfV0VCR0wyX1dIRU5fQVZBSUxBQkxFID0gdHJ1ZTtcbmxldCBXRUJHTDJfRU5BQkxFRDogYm9vbGVhbnx1bmRlZmluZWQgPSBudWxsO1xubGV0IE1BWF9URVhUVVJFX1NJWkU6IG51bWJlciA9IG51bGw7XG5cbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vLi4vdXRpbCc7XG5cbmV4cG9ydCBpbnRlcmZhY2UgV2ViR0xDb250ZXh0QXR0cmlidXRlcyB7XG4gIGFscGhhPzogYm9vbGVhbjtcbiAgYW50aWFsaWFzPzogYm9vbGVhbjtcbiAgcHJlbXVsdGlwbGllZEFscGhhPzogYm9vbGVhbjtcbiAgcHJlc2VydmVEcmF3aW5nQnVmZmVyPzogYm9vbGVhbjtcbiAgZGVwdGg/OiBib29sZWFuO1xuICBzdGVuY2lsPzogYm9vbGVhbjtcbiAgZmFpbElmTWFqb3JQZXJmb3JtYW5jZUNhdmVhdD86IGJvb2xlYW47XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgV2ViR0xMb3NlQ29udGV4dEV4dGVuc2lvbiB7IGxvc2VDb250ZXh0KCk6IHZvaWQ7IH1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVdlYkdMUmVuZGVyaW5nQ29udGV4dChhdHRyaWJ1dGVzOiBXZWJHTENvbnRleHRBdHRyaWJ1dGVzKTpcbiAgICBXZWJHTFJlbmRlcmluZ0NvbnRleHQge1xuICBjb25zdCBjYW52YXMgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdjYW52YXMnKTtcbiAgY2FudmFzLndpZHRoID0gMTtcbiAgY2FudmFzLmhlaWdodCA9IDE7XG4gIHJldHVybiBjcmVhdGVXZWJHTFJlbmRlcmluZ0NvbnRleHRGcm9tQ2FudmFzKGNhbnZhcywgYXR0cmlidXRlcyk7XG59XG5cbi8qKlxuICogRm9yY2UgdGhlIGxpYnJhcnkgdG8gcHJlZmVyIFdlYkdMIDEuMCBpbnN0ZWFkIG9mIFdlYkdMIDIuMCBldmVuIHdoZW4gV2ViR0xcbiAqIDIuMCBpcyBhdmFpbGFibGUuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBwcmVmZXJXZWJHTDEoKSB7XG4gIFVTRV9XRUJHTDJfV0hFTl9BVkFJTEFCTEUgPSBmYWxzZTtcbiAgV0VCR0wyX0VOQUJMRUQgPSBudWxsO1xufVxuXG4vKipcbiAqIFByZWZlciBXZWJHTCAyLjAgdG8gV2ViR0wgMS4wLiBUaGlzIGlzIHRoZSBkZWZhdWx0IGNvbmZpZ3VyYXRpb24uXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBwcmVmZXJXZWJHTDIoKSB7XG4gIFVTRV9XRUJHTDJfV0hFTl9BVkFJTEFCTEUgPSB0cnVlO1xuICBXRUJHTDJfRU5BQkxFRCA9IG51bGw7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc1dlYkdMMkVuYWJsZWQoKSB7XG4gIGlmICghVVNFX1dFQkdMMl9XSEVOX0FWQUlMQUJMRSkge1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuXG4gIGlmIChXRUJHTDJfRU5BQkxFRCA9PSBudWxsKSB7XG4gICAgY29uc3QgdGVtcENhbnZhcyA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2NhbnZhcycpO1xuICAgIGNvbnN0IGdsID0gdGVtcENhbnZhcy5nZXRDb250ZXh0KCd3ZWJnbDInKTtcbiAgICBpZiAoZ2wgIT0gbnVsbCkge1xuICAgICAgV0VCR0wyX0VOQUJMRUQgPSB0cnVlO1xuXG4gICAgICBjb25zdCBsb3NlQ29udGV4dEV4dGVuc2lvbiA9XG4gICAgICAgICAgZ2V0RXh0ZW5zaW9uT3JUaHJvdyhcbiAgICAgICAgICAgICAgZ2wgYXMgV2ViR0xSZW5kZXJpbmdDb250ZXh0LCAnV0VCR0xfbG9zZV9jb250ZXh0JykgYXNcbiAgICAgICAgICBXZWJHTExvc2VDb250ZXh0RXh0ZW5zaW9uO1xuICAgICAgbG9zZUNvbnRleHRFeHRlbnNpb24ubG9zZUNvbnRleHQoKTtcbiAgICB9IGVsc2Uge1xuICAgICAgV0VCR0wyX0VOQUJMRUQgPSBmYWxzZTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIFdFQkdMMl9FTkFCTEVEO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlV2ViR0xSZW5kZXJpbmdDb250ZXh0RnJvbUNhbnZhcyhcbiAgICBjYW52YXM6IEhUTUxDYW52YXNFbGVtZW50LFxuICAgIGF0dHJpYnV0ZXM6IFdlYkdMQ29udGV4dEF0dHJpYnV0ZXMpOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQge1xuICBsZXQgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dDtcbiAgaWYgKGlzV2ViR0wyRW5hYmxlZCgpKSB7XG4gICAgZ2wgPSBjYW52YXMuZ2V0Q29udGV4dCgnd2ViZ2wyJywgYXR0cmlidXRlcykgYXMgV2ViR0xSZW5kZXJpbmdDb250ZXh0O1xuICB9IGVsc2Uge1xuICAgIGdsID0gKGNhbnZhcy5nZXRDb250ZXh0KCd3ZWJnbCcsIGF0dHJpYnV0ZXMpIHx8XG4gICAgICAgICAgY2FudmFzLmdldENvbnRleHQoJ2V4cGVyaW1lbnRhbC13ZWJnbCcsIGF0dHJpYnV0ZXMpKSBhc1xuICAgICAgICBXZWJHTFJlbmRlcmluZ0NvbnRleHQ7XG4gIH1cblxuICBpZiAoZ2wgPT0gbnVsbCkge1xuICAgIHRocm93IG5ldyBFcnJvcignVGhpcyBicm93c2VyIGRvZXMgbm90IHN1cHBvcnQgV2ViR0wuJyk7XG4gIH1cbiAgcmV0dXJuIGdsO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY2FsbEFuZENoZWNrPFQ+KGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGZ1bmM6ICgpID0+IFQpOiBUIHtcbiAgY29uc3QgcmV0dXJuVmFsdWUgPSBmdW5jKCk7XG4gIGNoZWNrV2ViR0xFcnJvcihnbCk7XG4gIHJldHVybiByZXR1cm5WYWx1ZTtcbn1cblxubGV0IHdlYkdMRGVidWdFcnJvckNoZWNraW5nRW5hYmxlZCA9IGZhbHNlO1xuXG5leHBvcnQgZnVuY3Rpb24gZW5hYmxlRGVidWdXZWJHTEVycm9yQ2hlY2tpbmcoZW5hYmxlZDogYm9vbGVhbikge1xuICB3ZWJHTERlYnVnRXJyb3JDaGVja2luZ0VuYWJsZWQgPSBlbmFibGVkO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY2hlY2tXZWJHTEVycm9yKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpIHtcbiAgaWYgKHdlYkdMRGVidWdFcnJvckNoZWNraW5nRW5hYmxlZCkge1xuICAgIGNvbnN0IGVycm9yID0gZ2wuZ2V0RXJyb3IoKTtcbiAgICBpZiAoZXJyb3IgIT09IGdsLk5PX0VSUk9SKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ1dlYkdMIEVycm9yOiAnICsgZ2V0V2ViR0xFcnJvck1lc3NhZ2UoZ2wsIGVycm9yKSk7XG4gICAgfVxuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRXZWJHTEVycm9yTWVzc2FnZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBzdGF0dXM6IG51bWJlcik6IHN0cmluZyB7XG4gIHN3aXRjaCAoc3RhdHVzKSB7XG4gICAgY2FzZSBnbC5OT19FUlJPUjpcbiAgICAgIHJldHVybiAnTk9fRVJST1InO1xuICAgIGNhc2UgZ2wuSU5WQUxJRF9FTlVNOlxuICAgICAgcmV0dXJuICdJTlZBTElEX0VOVU0nO1xuICAgIGNhc2UgZ2wuSU5WQUxJRF9WQUxVRTpcbiAgICAgIHJldHVybiAnSU5WQUxJRF9WQUxVRSc7XG4gICAgY2FzZSBnbC5JTlZBTElEX09QRVJBVElPTjpcbiAgICAgIHJldHVybiAnSU5WQUxJRF9PUEVSQVRJT04nO1xuICAgIGNhc2UgZ2wuSU5WQUxJRF9GUkFNRUJVRkZFUl9PUEVSQVRJT046XG4gICAgICByZXR1cm4gJ0lOVkFMSURfRlJBTUVCVUZGRVJfT1BFUkFUSU9OJztcbiAgICBjYXNlIGdsLk9VVF9PRl9NRU1PUlk6XG4gICAgICByZXR1cm4gJ09VVF9PRl9NRU1PUlknO1xuICAgIGNhc2UgZ2wuQ09OVEVYVF9MT1NUX1dFQkdMOlxuICAgICAgcmV0dXJuICdDT05URVhUX0xPU1RfV0VCR0wnO1xuICAgIGRlZmF1bHQ6XG4gICAgICByZXR1cm4gJ1Vua25vd24gZXJyb3IgY29kZSAnICsgc3RhdHVzO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRFeHRlbnNpb25PclRocm93KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGV4dGVuc2lvbk5hbWU6IHN0cmluZyk6IHt9IHtcbiAgcmV0dXJuIHRocm93SWZOdWxsPHt9PihcbiAgICAgIGdsLCAoKSA9PiBnbC5nZXRFeHRlbnNpb24oZXh0ZW5zaW9uTmFtZSksXG4gICAgICAnRXh0ZW5zaW9uIFwiJyArIGV4dGVuc2lvbk5hbWUgKyAnXCIgbm90IHN1cHBvcnRlZCBvbiB0aGlzIGJyb3dzZXIuJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVWZXJ0ZXhTaGFkZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdmVydGV4U2hhZGVyU291cmNlOiBzdHJpbmcpOiBXZWJHTFNoYWRlciB7XG4gIGNvbnN0IHZlcnRleFNoYWRlcjogV2ViR0xTaGFkZXIgPSB0aHJvd0lmTnVsbDxXZWJHTFNoYWRlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlU2hhZGVyKGdsLlZFUlRFWF9TSEFERVIpLFxuICAgICAgJ1VuYWJsZSB0byBjcmVhdGUgdmVydGV4IFdlYkdMU2hhZGVyLicpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnNoYWRlclNvdXJjZSh2ZXJ0ZXhTaGFkZXIsIHZlcnRleFNoYWRlclNvdXJjZSkpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmNvbXBpbGVTaGFkZXIodmVydGV4U2hhZGVyKSk7XG4gIGlmIChnbC5nZXRTaGFkZXJQYXJhbWV0ZXIodmVydGV4U2hhZGVyLCBnbC5DT01QSUxFX1NUQVRVUykgPT09IGZhbHNlKSB7XG4gICAgY29uc29sZS5sb2coZ2wuZ2V0U2hhZGVySW5mb0xvZyh2ZXJ0ZXhTaGFkZXIpKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0ZhaWxlZCB0byBjb21waWxlIHZlcnRleCBzaGFkZXIuJyk7XG4gIH1cbiAgcmV0dXJuIHZlcnRleFNoYWRlcjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZUZyYWdtZW50U2hhZGVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGZyYWdtZW50U2hhZGVyU291cmNlOiBzdHJpbmcpOiBXZWJHTFNoYWRlciB7XG4gIGNvbnN0IGZyYWdtZW50U2hhZGVyOiBXZWJHTFNoYWRlciA9IHRocm93SWZOdWxsPFdlYkdMU2hhZGVyPihcbiAgICAgIGdsLCAoKSA9PiBnbC5jcmVhdGVTaGFkZXIoZ2wuRlJBR01FTlRfU0hBREVSKSxcbiAgICAgICdVbmFibGUgdG8gY3JlYXRlIGZyYWdtZW50IFdlYkdMU2hhZGVyLicpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnNoYWRlclNvdXJjZShmcmFnbWVudFNoYWRlciwgZnJhZ21lbnRTaGFkZXJTb3VyY2UpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5jb21waWxlU2hhZGVyKGZyYWdtZW50U2hhZGVyKSk7XG4gIGlmIChnbC5nZXRTaGFkZXJQYXJhbWV0ZXIoZnJhZ21lbnRTaGFkZXIsIGdsLkNPTVBJTEVfU1RBVFVTKSA9PT0gZmFsc2UpIHtcbiAgICBjb25zb2xlLmxvZyhnbC5nZXRTaGFkZXJJbmZvTG9nKGZyYWdtZW50U2hhZGVyKSk7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gY29tcGlsZSBmcmFnbWVudCBzaGFkZXIuJyk7XG4gIH1cbiAgcmV0dXJuIGZyYWdtZW50U2hhZGVyO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlUHJvZ3JhbShnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogV2ViR0xQcm9ncmFtIHtcbiAgcmV0dXJuIHRocm93SWZOdWxsPFdlYkdMUHJvZ3JhbT4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlUHJvZ3JhbSgpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTFByb2dyYW0uJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBsaW5rUHJvZ3JhbShnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0pIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5saW5rUHJvZ3JhbShwcm9ncmFtKSk7XG4gIGlmIChnbC5nZXRQcm9ncmFtUGFyYW1ldGVyKHByb2dyYW0sIGdsLkxJTktfU1RBVFVTKSA9PT0gZmFsc2UpIHtcbiAgICBjb25zb2xlLmxvZyhnbC5nZXRQcm9ncmFtSW5mb0xvZyhwcm9ncmFtKSk7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdGYWlsZWQgdG8gbGluayB2ZXJ0ZXggYW5kIGZyYWdtZW50IHNoYWRlcnMuJyk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHZhbGlkYXRlUHJvZ3JhbShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0pIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC52YWxpZGF0ZVByb2dyYW0ocHJvZ3JhbSkpO1xuICBpZiAoZ2wuZ2V0UHJvZ3JhbVBhcmFtZXRlcihwcm9ncmFtLCBnbC5WQUxJREFURV9TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGNvbnNvbGUubG9nKGdsLmdldFByb2dyYW1JbmZvTG9nKHByb2dyYW0pKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ1NoYWRlciBwcm9ncmFtIHZhbGlkYXRpb24gZmFpbGVkLicpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVTdGF0aWNWZXJ0ZXhCdWZmZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZGF0YTogRmxvYXQzMkFycmF5KTogV2ViR0xCdWZmZXIge1xuICBjb25zdCBidWZmZXI6IFdlYkdMQnVmZmVyID0gdGhyb3dJZk51bGw8V2ViR0xCdWZmZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZUJ1ZmZlcigpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTEJ1ZmZlcicpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuQVJSQVlfQlVGRkVSLCBidWZmZXIpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5idWZmZXJEYXRhKGdsLkFSUkFZX0JVRkZFUiwgZGF0YSwgZ2wuU1RBVElDX0RSQVcpKTtcbiAgcmV0dXJuIGJ1ZmZlcjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVN0YXRpY0luZGV4QnVmZmVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGRhdGE6IFVpbnQxNkFycmF5KTogV2ViR0xCdWZmZXIge1xuICBjb25zdCBidWZmZXI6IFdlYkdMQnVmZmVyID0gdGhyb3dJZk51bGw8V2ViR0xCdWZmZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZUJ1ZmZlcigpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTEJ1ZmZlcicpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuRUxFTUVOVF9BUlJBWV9CVUZGRVIsIGJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soXG4gICAgICBnbCwgKCkgPT4gZ2wuYnVmZmVyRGF0YShnbC5FTEVNRU5UX0FSUkFZX0JVRkZFUiwgZGF0YSwgZ2wuU1RBVElDX0RSQVcpKTtcbiAgcmV0dXJuIGJ1ZmZlcjtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHF1ZXJ5TWF4VGV4dHVyZVNpemUoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IG51bWJlciB7XG4gIGlmIChNQVhfVEVYVFVSRV9TSVpFICE9IG51bGwpIHtcbiAgICByZXR1cm4gTUFYX1RFWFRVUkVfU0laRTtcbiAgfVxuICBNQVhfVEVYVFVSRV9TSVpFID1cbiAgICAgIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZ2V0UGFyYW1ldGVyKGdsLk1BWF9URVhUVVJFX1NJWkUpKTtcbiAgcmV0dXJuIE1BWF9URVhUVVJFX1NJWkU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRDaGFubmVsc1BlclRleHR1cmUoKTogbnVtYmVyIHtcbiAgaWYgKGlzV2ViR0wyRW5hYmxlZCgpKSB7XG4gICAgcmV0dXJuIDE7XG4gIH1cbiAgcmV0dXJuIDQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVUZXh0dXJlKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpOiBXZWJHTFRleHR1cmUge1xuICByZXR1cm4gdGhyb3dJZk51bGw8V2ViR0xUZXh0dXJlPihcbiAgICAgIGdsLCAoKSA9PiBnbC5jcmVhdGVUZXh0dXJlKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMVGV4dHVyZS4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHZhbGlkYXRlVGV4dHVyZVNpemUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgd2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIpIHtcbiAgY29uc3QgbWF4VGV4dHVyZVNpemU6IG51bWJlciA9IHF1ZXJ5TWF4VGV4dHVyZVNpemUoZ2wpO1xuICBpZiAoKHdpZHRoIDw9IDApIHx8IChoZWlnaHQgPD0gMCkpIHtcbiAgICBjb25zdCByZXF1ZXN0ZWQgPSAnWycgKyB3aWR0aCArICd4JyArIGhlaWdodCArICddJztcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ1JlcXVlc3RlZCB0ZXh0dXJlIHNpemUgJyArIHJlcXVlc3RlZCArICcgaXMgaW52YWxpZC4nKTtcbiAgfVxuICBpZiAoKHdpZHRoID4gbWF4VGV4dHVyZVNpemUpIHx8IChoZWlnaHQgPiBtYXhUZXh0dXJlU2l6ZSkpIHtcbiAgICBjb25zdCByZXF1ZXN0ZWQgPSAnWycgKyB3aWR0aCArICd4JyArIGhlaWdodCArICddJztcbiAgICBjb25zdCBtYXggPSAnWycgKyBtYXhUZXh0dXJlU2l6ZSArICd4JyArIG1heFRleHR1cmVTaXplICsgJ10nO1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ1JlcXVlc3RlZCB0ZXh0dXJlIHNpemUgJyArIHJlcXVlc3RlZCArXG4gICAgICAgICcgZ3JlYXRlciB0aGFuIFdlYkdMIG1heGltdW0gb24gdGhpcyBicm93c2VyIC8gR1BVICcgKyBtYXggKyAnLicpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVGcmFtZWJ1ZmZlcihnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogV2ViR0xGcmFtZWJ1ZmZlciB7XG4gIHJldHVybiB0aHJvd0lmTnVsbDxXZWJHTEZyYW1lYnVmZmVyPihcbiAgICAgIGdsLCAoKSA9PiBnbC5jcmVhdGVGcmFtZWJ1ZmZlcigpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTEZyYW1lYnVmZmVyLicpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZFZlcnRleEJ1ZmZlclRvUHJvZ3JhbUF0dHJpYnV0ZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0sIGF0dHJpYnV0ZTogc3RyaW5nLFxuICAgIGJ1ZmZlcjogV2ViR0xCdWZmZXIsIGFycmF5RW50cmllc1Blckl0ZW06IG51bWJlciwgaXRlbVN0cmlkZUluQnl0ZXM6IG51bWJlcixcbiAgICBpdGVtT2Zmc2V0SW5CeXRlczogbnVtYmVyKSB7XG4gIGNvbnN0IGxvYyA9IGdsLmdldEF0dHJpYkxvY2F0aW9uKHByb2dyYW0sIGF0dHJpYnV0ZSk7XG4gIGlmIChsb2MgPT09IC0xKSB7XG4gICAgY29uc3QgZXJyb3IgPSBuZXcgRXJyb3IoXG4gICAgICAgICdVbmFibGUgdG8gZ2V0IGF0dHJpYnV0ZSBcIicgKyBhdHRyaWJ1dGUgKyAnXCIgb24gV2ViR0xQcm9ncmFtLicpO1xuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAoZXJyb3IgYXMgYW55KS5uYW1lZFZlcnRleEF0dHJpYnV0ZU5vdEZvdW5kID0gYXR0cmlidXRlO1xuICAgIHRocm93IGVycm9yO1xuICB9XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5BUlJBWV9CVUZGRVIsIGJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soXG4gICAgICBnbCxcbiAgICAgICgpID0+IGdsLnZlcnRleEF0dHJpYlBvaW50ZXIoXG4gICAgICAgICAgbG9jLCBhcnJheUVudHJpZXNQZXJJdGVtLCBnbC5GTE9BVCwgZmFsc2UsIGl0ZW1TdHJpZGVJbkJ5dGVzLFxuICAgICAgICAgIGl0ZW1PZmZzZXRJbkJ5dGVzKSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZW5hYmxlVmVydGV4QXR0cmliQXJyYXkobG9jKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBiaW5kVGV4dHVyZVVuaXQoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gIHZhbGlkYXRlVGV4dHVyZVVuaXQoZ2wsIHRleHR1cmVVbml0KTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5hY3RpdmVUZXh0dXJlKGdsLlRFWFRVUkUwICsgdGV4dHVyZVVuaXQpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0ZXh0dXJlKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1bmJpbmRUZXh0dXJlVW5pdChcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gIHZhbGlkYXRlVGV4dHVyZVVuaXQoZ2wsIHRleHR1cmVVbml0KTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5hY3RpdmVUZXh0dXJlKGdsLlRFWFRVUkUwICsgdGV4dHVyZVVuaXQpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRQcm9ncmFtVW5pZm9ybUxvY2F0aW9uT3JUaHJvdyhcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBwcm9ncmFtOiBXZWJHTFByb2dyYW0sXG4gICAgdW5pZm9ybU5hbWU6IHN0cmluZyk6IFdlYkdMVW5pZm9ybUxvY2F0aW9uIHtcbiAgcmV0dXJuIHRocm93SWZOdWxsPFdlYkdMVW5pZm9ybUxvY2F0aW9uPihcbiAgICAgIGdsLCAoKSA9PiBnbC5nZXRVbmlmb3JtTG9jYXRpb24ocHJvZ3JhbSwgdW5pZm9ybU5hbWUpLFxuICAgICAgJ3VuaWZvcm0gXCInICsgdW5pZm9ybU5hbWUgKyAnXCIgbm90IHByZXNlbnQgaW4gcHJvZ3JhbS4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRUZXh0dXJlVG9Qcm9ncmFtVW5pZm9ybVNhbXBsZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcHJvZ3JhbTogV2ViR0xQcm9ncmFtLCB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsXG4gICAgdW5pZm9ybVNhbXBsZXJOYW1lOiBzdHJpbmcsIHRleHR1cmVVbml0OiBudW1iZXIpIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBiaW5kVGV4dHVyZVVuaXQoZ2wsIHRleHR1cmUsIHRleHR1cmVVbml0KSk7XG4gIGNvbnN0IHNhbXBsZXJMb2NhdGlvbiA9XG4gICAgICBnZXRQcm9ncmFtVW5pZm9ybUxvY2F0aW9uT3JUaHJvdyhnbCwgcHJvZ3JhbSwgdW5pZm9ybVNhbXBsZXJOYW1lKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC51bmlmb3JtMWkoc2FtcGxlckxvY2F0aW9uLCB0ZXh0dXJlVW5pdCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZENhbnZhc1RvRnJhbWVidWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCkge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRGcmFtZWJ1ZmZlcihnbC5GUkFNRUJVRkZFUiwgbnVsbCkpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnZpZXdwb3J0KDAsIDAsIGdsLmNhbnZhcy53aWR0aCwgZ2wuY2FudmFzLmhlaWdodCkpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnNjaXNzb3IoMCwgMCwgZ2wuY2FudmFzLndpZHRoLCBnbC5jYW52YXMuaGVpZ2h0KSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBiaW5kQ29sb3JUZXh0dXJlVG9GcmFtZWJ1ZmZlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsXG4gICAgZnJhbWVidWZmZXI6IFdlYkdMRnJhbWVidWZmZXIpIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIGZyYW1lYnVmZmVyKSk7XG4gIGNhbGxBbmRDaGVjayhcbiAgICAgIGdsLFxuICAgICAgKCkgPT4gZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoXG4gICAgICAgICAgZ2wuRlJBTUVCVUZGRVIsIGdsLkNPTE9SX0FUVEFDSE1FTlQwLCBnbC5URVhUVVJFXzJELCB0ZXh0dXJlLCAwKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1bmJpbmRDb2xvclRleHR1cmVGcm9tRnJhbWVidWZmZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZnJhbWVidWZmZXI6IFdlYkdMRnJhbWVidWZmZXIpIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIGZyYW1lYnVmZmVyKSk7XG4gIGNhbGxBbmRDaGVjayhcbiAgICAgIGdsLFxuICAgICAgKCkgPT4gZ2wuZnJhbWVidWZmZXJUZXh0dXJlMkQoXG4gICAgICAgICAgZ2wuRlJBTUVCVUZGRVIsIGdsLkNPTE9SX0FUVEFDSE1FTlQwLCBnbC5URVhUVVJFXzJELCBudWxsLCAwKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB2YWxpZGF0ZUZyYW1lYnVmZmVyKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpIHtcbiAgY29uc3Qgc3RhdHVzID0gZ2wuY2hlY2tGcmFtZWJ1ZmZlclN0YXR1cyhnbC5GUkFNRUJVRkZFUik7XG4gIGlmIChzdGF0dXMgIT09IGdsLkZSQU1FQlVGRkVSX0NPTVBMRVRFKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAnRXJyb3IgYmluZGluZyBmcmFtZWJ1ZmZlcjogJyArIGdldEZyYW1lYnVmZmVyRXJyb3JNZXNzYWdlKGdsLCBzdGF0dXMpKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0RnJhbWVidWZmZXJFcnJvck1lc3NhZ2UoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgc3RhdHVzOiBudW1iZXIpOiBzdHJpbmcge1xuICBzd2l0Y2ggKHN0YXR1cykge1xuICAgIGNhc2UgZ2wuRlJBTUVCVUZGRVJfSU5DT01QTEVURV9BVFRBQ0hNRU5UOlxuICAgICAgcmV0dXJuICdGUkFNRUJVRkZFUl9JTkNPTVBMRVRFX0FUVEFDSE1FTlQnO1xuICAgIGNhc2UgZ2wuRlJBTUVCVUZGRVJfSU5DT01QTEVURV9NSVNTSU5HX0FUVEFDSE1FTlQ6XG4gICAgICByZXR1cm4gJ0ZSQU1FQlVGRkVSX0lOQ09NUExFVEVfTUlTU0lOR19BVFRBQ0hNRU5UJztcbiAgICBjYXNlIGdsLkZSQU1FQlVGRkVSX0lOQ09NUExFVEVfRElNRU5TSU9OUzpcbiAgICAgIHJldHVybiAnRlJBTUVCVUZGRVJfSU5DT01QTEVURV9ESU1FTlNJT05TJztcbiAgICBjYXNlIGdsLkZSQU1FQlVGRkVSX1VOU1VQUE9SVEVEOlxuICAgICAgcmV0dXJuICdGUkFNRUJVRkZFUl9VTlNVUFBPUlRFRCc7XG4gICAgZGVmYXVsdDpcbiAgICAgIHJldHVybiAndW5rbm93biBlcnJvciAnICsgc3RhdHVzO1xuICB9XG59XG5cbmZ1bmN0aW9uIHRocm93SWZOdWxsPFQ+KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHJldHVyblRPck51bGw6ICgpID0+IFQgfCBudWxsLFxuICAgIGZhaWx1cmVNZXNzYWdlOiBzdHJpbmcpOiBUIHtcbiAgY29uc3QgdE9yTnVsbDogVHxudWxsID0gY2FsbEFuZENoZWNrKGdsLCAoKSA9PiByZXR1cm5UT3JOdWxsKCkpO1xuICBpZiAodE9yTnVsbCA9PSBudWxsKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKGZhaWx1cmVNZXNzYWdlKTtcbiAgfVxuICByZXR1cm4gdE9yTnVsbCBhcyBUO1xufVxuXG5mdW5jdGlvbiB2YWxpZGF0ZVRleHR1cmVVbml0KGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmVVbml0OiBudW1iZXIpIHtcbiAgY29uc3QgbWF4VGV4dHVyZVVuaXQgPSBnbC5NQVhfQ09NQklORURfVEVYVFVSRV9JTUFHRV9VTklUUyAtIDE7XG4gIGNvbnN0IGdsVGV4dHVyZVVuaXQgPSB0ZXh0dXJlVW5pdCArIGdsLlRFWFRVUkUwO1xuICBpZiAoZ2xUZXh0dXJlVW5pdCA8IGdsLlRFWFRVUkUwIHx8IGdsVGV4dHVyZVVuaXQgPiBtYXhUZXh0dXJlVW5pdCkge1xuICAgIGNvbnN0IHRleHR1cmVVbml0UmFuZ2UgPSAnW2dsLlRFWFRVUkUwLCBnbC5URVhUVVJFJyArIG1heFRleHR1cmVVbml0ICsgJ10nO1xuICAgIHRocm93IG5ldyBFcnJvcigndGV4dHVyZVVuaXQgbXVzdCBiZSBpbiAnICsgdGV4dHVyZVVuaXRSYW5nZSArICcuJyk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFRleHR1cmVTaGFwZUZyb21Mb2dpY2FsU2hhcGUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgbG9nU2hhcGU6IG51bWJlcltdLFxuICAgIHByZWZlcnJlZFRleFNoYXBlPzogW251bWJlciwgbnVtYmVyXSk6IFtudW1iZXIsIG51bWJlcl0ge1xuICBjb25zdCBtYXhUZXhTaXplID0gcXVlcnlNYXhUZXh0dXJlU2l6ZShnbCk7XG4gIGNvbnN0IHNpemUgPSB1dGlsLnNpemVGcm9tU2hhcGUobG9nU2hhcGUpO1xuICBpZiAocHJlZmVycmVkVGV4U2hhcGUgIT0gbnVsbCkge1xuICAgIGNvbnN0IHNpemVQcmVmZXJyZWQgPSB1dGlsLnNpemVGcm9tU2hhcGUocHJlZmVycmVkVGV4U2hhcGUpO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBzaXplID09PSBzaXplUHJlZmVycmVkLFxuICAgICAgICBgU2l6ZSBvZiBzaGFwZSAoJHtzaXplfSkgbXVzdCBtYXRjaCBzaXplIG9mIGAgK1xuICAgICAgICAgICAgYHByZWZlcnJlZFNoYXBlICgke3NpemVQcmVmZXJyZWR9KWApO1xuICAgIGlmIChwcmVmZXJyZWRUZXhTaGFwZVswXSA8PSBtYXhUZXhTaXplICYmXG4gICAgICAgIHByZWZlcnJlZFRleFNoYXBlWzFdIDw9IG1heFRleFNpemUpIHtcbiAgICAgIHJldHVybiBwcmVmZXJyZWRUZXhTaGFwZTtcbiAgICB9XG4gIH1cblxuICBpZiAobG9nU2hhcGUubGVuZ3RoIDw9IDEgJiYgc2l6ZSA8PSBtYXhUZXhTaXplKSB7XG4gICAgcmV0dXJuIFtzaXplLCAxXTtcbiAgfSBlbHNlIGlmIChcbiAgICAgIGxvZ1NoYXBlLmxlbmd0aCA9PT0gMiAmJiBsb2dTaGFwZVswXSA8PSBtYXhUZXhTaXplICYmXG4gICAgICBsb2dTaGFwZVsxXSA8PSBtYXhUZXhTaXplKSB7XG4gICAgcmV0dXJuIGxvZ1NoYXBlIGFzIFtudW1iZXIsIG51bWJlcl07XG4gIH0gZWxzZSBpZiAoXG4gICAgICBsb2dTaGFwZS5sZW5ndGggPT09IDMgJiYgbG9nU2hhcGVbMF0gPD0gbWF4VGV4U2l6ZSAmJlxuICAgICAgbG9nU2hhcGVbMV0gKiBsb2dTaGFwZVsyXSA8PSBtYXhUZXhTaXplKSB7XG4gICAgcmV0dXJuIFtsb2dTaGFwZVswXSwgbG9nU2hhcGVbMV0gKiBsb2dTaGFwZVsyXV07XG4gIH0gZWxzZSBpZiAoXG4gICAgICBsb2dTaGFwZS5sZW5ndGggPT09IDQgJiYgbG9nU2hhcGVbMF0gPD0gbWF4VGV4U2l6ZSAmJlxuICAgICAgbG9nU2hhcGVbMV0gKiBsb2dTaGFwZVsyXSAqIGxvZ1NoYXBlWzNdIDw9IG1heFRleFNpemUpIHtcbiAgICByZXR1cm4gW2xvZ1NoYXBlWzBdLCBsb2dTaGFwZVsxXSAqIGxvZ1NoYXBlWzJdICogbG9nU2hhcGVbM11dO1xuICB9IGVsc2Uge1xuICAgIHJldHVybiB1dGlsLnNpemVUb1NxdWFyaXNoU2hhcGUoc2l6ZSk7XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuZXhwb3J0IGZ1bmN0aW9uIGV4cGVjdEFycmF5c0Nsb3NlKFxuICAgIGFjdHVhbDogRmxvYXQzMkFycmF5LCBleHBlY3RlZDogRmxvYXQzMkFycmF5LCBlcHNpbG9uOiBudW1iZXIpIHtcbiAgaWYgKGFjdHVhbC5sZW5ndGggIT09IGV4cGVjdGVkLmxlbmd0aCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ01hdHJpY2VzIGhhdmUgZGlmZmVyZW50IGxlbmd0aHMgKCcgKyBhY3R1YWwubGVuZ3RoICsgJyB2cyAnICtcbiAgICAgICAgZXhwZWN0ZWQubGVuZ3RoICsgJykuJyk7XG4gIH1cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBleHBlY3RlZC5sZW5ndGg7ICsraSkge1xuICAgIGNvbnN0IGEgPSBhY3R1YWxbaV07XG4gICAgY29uc3QgZSA9IGV4cGVjdGVkW2ldO1xuICAgIGlmIChpc05hTihhKSAmJiBpc05hTihlKSkge1xuICAgICAgY29udGludWU7XG4gICAgfVxuICAgIGlmIChpc05hTihhKSB8fCBpc05hTihlKSB8fCBNYXRoLmFicyhhIC0gZSkgPiBlcHNpbG9uKSB7XG4gICAgICBjb25zdCBhY3R1YWxTdHIgPSAnYWN0dWFsWycgKyBpICsgJ10gPT09ICcgKyBhO1xuICAgICAgY29uc3QgZXhwZWN0ZWRTdHIgPSAnZXhwZWN0ZWRbJyArIGkgKyAnXSA9PT0gJyArIGU7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0FycmF5cyBkaWZmZXI6ICcgKyBhY3R1YWxTdHIgKyAnLCAnICsgZXhwZWN0ZWRTdHIpO1xuICAgIH1cbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gcmFuZG9tQXJyYXlJblJhbmdlKFxuICAgIG46IG51bWJlciwgbWluVmFsdWU6IG51bWJlciwgbWF4VmFsdWU6IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gIGNvbnN0IHYgPSBuZXcgRmxvYXQzMkFycmF5KG4pO1xuICBjb25zdCByYW5nZSA9IG1heFZhbHVlIC0gbWluVmFsdWU7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbjsgKytpKSB7XG4gICAgdltpXSA9IChNYXRoLnJhbmRvbSgpICogcmFuZ2UpICsgbWluVmFsdWU7XG4gIH1cbiAgcmV0dXJuIHY7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlSWRlbnRpdHkobjogbnVtYmVyKTogRmxvYXQzMkFycmF5IHtcbiAgY29uc3QgaSA9IG5ldyBGbG9hdDMyQXJyYXkobiAqIG4pO1xuICBmb3IgKGxldCBqID0gMDsgaiA8IG47ICsraikge1xuICAgIGlbKGogKiBuKSArIGpdID0gMTtcbiAgfVxuICByZXR1cm4gaTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHNldFZhbHVlKFxuICAgIG06IEZsb2F0MzJBcnJheSwgbU51bVJvd3M6IG51bWJlciwgbU51bUNvbHM6IG51bWJlciwgdjogbnVtYmVyLCByb3c6IG51bWJlcixcbiAgICBjb2x1bW46IG51bWJlcikge1xuICBpZiAocm93ID49IG1OdW1Sb3dzKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdyb3cgKCcgKyByb3cgKyAnKSBtdXN0IGJlIGluIFswICcgKyBtTnVtUm93cyArICddLicpO1xuICB9XG4gIGlmIChjb2x1bW4gPj0gbU51bUNvbHMpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ2NvbHVtbiAoJyArIGNvbHVtbiArICcpIG11c3QgYmUgaW4gWzAgJyArIG1OdW1Db2xzICsgJ10uJyk7XG4gIH1cbiAgbVsocm93ICogbU51bUNvbHMpICsgY29sdW1uXSA9IHY7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcHVNdWx0aXBseU1hdHJpeChcbiAgICBhOiBGbG9hdDMyQXJyYXksIGFSb3c6IG51bWJlciwgYUNvbDogbnVtYmVyLCBiOiBGbG9hdDMyQXJyYXksIGJSb3c6IG51bWJlcixcbiAgICBiQ29sOiBudW1iZXIpIHtcbiAgY29uc3QgcmVzdWx0ID0gbmV3IEZsb2F0MzJBcnJheShhUm93ICogYkNvbCk7XG4gIGZvciAobGV0IHIgPSAwOyByIDwgYVJvdzsgKytyKSB7XG4gICAgZm9yIChsZXQgYyA9IDA7IGMgPCBiQ29sOyArK2MpIHtcbiAgICAgIGxldCBkID0gMDtcbiAgICAgIGZvciAobGV0IGsgPSAwOyBrIDwgYUNvbDsgKytrKSB7XG4gICAgICAgIGQgKz0gYVsociAqIGFDb2wpICsga10gKiBiWyhrICogYkNvbCkgKyBjXTtcbiAgICAgIH1cbiAgICAgIHJlc3VsdFsociAqIGJDb2wpICsgY10gPSBkO1xuICAgIH1cbiAgfVxuICByZXR1cm4gcmVzdWx0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3B1RG90UHJvZHVjdChhOiBGbG9hdDMyQXJyYXksIGI6IEZsb2F0MzJBcnJheSk6IG51bWJlciB7XG4gIGlmIChhLmxlbmd0aCAhPT0gYi5sZW5ndGgpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ2NwdURvdFByb2R1Y3Q6IGluY29tcGF0aWJsZSB2ZWN0b3JzLicpO1xuICB9XG4gIGxldCBkID0gMDtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBhLmxlbmd0aDsgKytpKSB7XG4gICAgZCArPSBhW2ldICogYltpXTtcbiAgfVxuICByZXR1cm4gZDtcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuZXhwb3J0IHR5cGUgVmVjdG9yID0gbnVtYmVyW10gfCBGbG9hdDY0QXJyYXkgfCBGbG9hdDMyQXJyYXkgfCBJbnQzMkFycmF5IHxcbiAgICBJbnQ4QXJyYXkgfCBJbnQxNkFycmF5O1xuXG4vKiogU2h1ZmZsZXMgdGhlIGFycmF5IHVzaW5nIEZpc2hlci1ZYXRlcyBhbGdvcml0aG0uICovXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG5leHBvcnQgZnVuY3Rpb24gc2h1ZmZsZShhcnJheTogYW55W118VWludDMyQXJyYXl8SW50MzJBcnJheXxcbiAgICAgICAgICAgICAgICAgICAgICAgIEZsb2F0MzJBcnJheSk6IHZvaWQge1xuICBsZXQgY291bnRlciA9IGFycmF5Lmxlbmd0aDtcbiAgbGV0IHRlbXAgPSAwO1xuICBsZXQgaW5kZXggPSAwO1xuICAvLyBXaGlsZSB0aGVyZSBhcmUgZWxlbWVudHMgaW4gdGhlIGFycmF5XG4gIHdoaWxlIChjb3VudGVyID4gMCkge1xuICAgIC8vIFBpY2sgYSByYW5kb20gaW5kZXhcbiAgICBpbmRleCA9IChNYXRoLnJhbmRvbSgpICogY291bnRlcikgfCAwO1xuICAgIC8vIERlY3JlYXNlIGNvdW50ZXIgYnkgMVxuICAgIGNvdW50ZXItLTtcbiAgICAvLyBBbmQgc3dhcCB0aGUgbGFzdCBlbGVtZW50IHdpdGggaXRcbiAgICB0ZW1wID0gYXJyYXlbY291bnRlcl07XG4gICAgYXJyYXlbY291bnRlcl0gPSBhcnJheVtpbmRleF07XG4gICAgYXJyYXlbaW5kZXhdID0gdGVtcDtcbiAgfVxufVxuXG4vKiogQ2xhbXBzIGEgdmFsdWUgdG8gYSBzcGVjaWZpZWQgcmFuZ2UuICovXG5leHBvcnQgZnVuY3Rpb24gY2xhbXAobWluOiBudW1iZXIsIHg6IG51bWJlciwgbWF4OiBudW1iZXIpOiBudW1iZXIge1xuICByZXR1cm4gTWF0aC5tYXgobWluLCBNYXRoLm1pbih4LCBtYXgpKTtcbn1cblxuLyoqIFJldHVybnMgYSBzYW1wbGUgZnJvbSBhIHVuaWZvcm0gW2EsIGJdIGRpc3RyaWJ1dGlvbi4gKi9cbmV4cG9ydCBmdW5jdGlvbiByYW5kVW5pZm9ybShhOiBudW1iZXIsIGI6IG51bWJlcikge1xuICByZXR1cm4gTWF0aC5yYW5kb20oKSAqIChiIC0gYSkgKyBhO1xufVxuXG4vKipcbiAqIFNhbXBsZXMgZnJvbSBhIGdhdXNzaWFuIGRpc3RyaWJ1dGlvbi5cbiAqXG4gKiBAcGFyYW0gbWVhbiBUaGUgbWVhbi4gRGVmYXVsdCBpcyAwLlxuICogQHBhcmFtIHN0ZERldiBUaGUgc3RhbmRhcmQgZGV2aWF0aW9uLiBEZWZhdWx0IGlzIDEuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiByYW5kR2F1c3MobWVhbiA9IDAsIHN0ZERldiA9IDEsIHRydW5jYXRlZCA9IGZhbHNlKTogbnVtYmVyIHtcbiAgbGV0IHYxOiBudW1iZXIsIHYyOiBudW1iZXIsIHM6IG51bWJlcjtcbiAgZG8ge1xuICAgIHYxID0gMiAqIE1hdGgucmFuZG9tKCkgLSAxO1xuICAgIHYyID0gMiAqIE1hdGgucmFuZG9tKCkgLSAxO1xuICAgIHMgPSB2MSAqIHYxICsgdjIgKiB2MjtcbiAgfSB3aGlsZSAocyA+IDEpO1xuXG4gIGNvbnN0IHJlc3VsdCA9IE1hdGguc3FydCgtMiAqIE1hdGgubG9nKHMpIC8gcykgKiB2MTtcbiAgaWYgKHRydW5jYXRlZCAmJiByZXN1bHQgPiAyKSB7XG4gICAgcmV0dXJuIHJhbmRHYXVzcyhtZWFuLCBzdGREZXYsIHRydWUpO1xuICB9XG4gIHJldHVybiBtZWFuICsgc3RkRGV2ICogcmVzdWx0O1xufVxuXG4vKiogUmV0dXJucyBzcXVhcmVkIGV1Y2xlZGlhbiBkaXN0YW5jZSBiZXR3ZWVuIHR3byB2ZWN0b3JzLiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGRpc3RTcXVhcmVkKGE6IFZlY3RvciwgYjogVmVjdG9yKTogbnVtYmVyIHtcbiAgbGV0IHJlc3VsdCA9IDA7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgYS5sZW5ndGg7IGkrKykge1xuICAgIGNvbnN0IGRpZmYgPSBhW2ldIC0gYltpXTtcbiAgICByZXN1bHQgKz0gZGlmZiAqIGRpZmY7XG4gIH1cbiAgcmV0dXJuIHJlc3VsdDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGFzc2VydChleHByOiBib29sZWFuLCBtc2c6IHN0cmluZykge1xuICBpZiAoIWV4cHIpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IobXNnKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gYXNzZXJ0U2hhcGVzTWF0Y2goXG4gICAgc2hhcGVBOiBudW1iZXJbXSwgc2hhcGVCOiBudW1iZXJbXSwgZXJyb3JNZXNzYWdlUHJlZml4ID0gJycpOiB2b2lkIHtcbiAgYXNzZXJ0KFxuICAgICAgYXJyYXlzRXF1YWwoc2hhcGVBLCBzaGFwZUIpLFxuICAgICAgZXJyb3JNZXNzYWdlUHJlZml4ICsgYFNoYXBlcyAke3NoYXBlQX0gYW5kICR7c2hhcGVCfSBtdXN0IG1hdGNoYCk7XG59XG5cbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbmV4cG9ydCBmdW5jdGlvbiBmbGF0dGVuKGFycjogYW55W10sIHJldD86IG51bWJlcltdKTogbnVtYmVyW10ge1xuICByZXQgPSAocmV0ID09PSB1bmRlZmluZWQgPyBbXSA6IHJldCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgYXJyLmxlbmd0aDsgKytpKSB7XG4gICAgaWYgKEFycmF5LmlzQXJyYXkoYXJyW2ldKSkge1xuICAgICAgZmxhdHRlbihhcnJbaV0sIHJldCk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldC5wdXNoKGFycltpXSk7XG4gICAgfVxuICB9XG4gIHJldHVybiByZXQ7XG59XG5cbmV4cG9ydCB0eXBlIEFycmF5RGF0YSA9IG51bWJlcnxudW1iZXJbXXxudW1iZXJbXVtdfG51bWJlcltdW11bXXxudW1iZXJbXVtdW11bXTtcblxuZXhwb3J0IGZ1bmN0aW9uIGluZmVyU2hhcGUoYXJyOiBBcnJheURhdGEpOiBudW1iZXJbXSB7XG4gIGNvbnN0IHNoYXBlOiBudW1iZXJbXSA9IFtdO1xuICB3aGlsZSAoYXJyIGluc3RhbmNlb2YgQXJyYXkpIHtcbiAgICBzaGFwZS5wdXNoKGFyci5sZW5ndGgpO1xuICAgIGFyciA9IGFyclswXTtcbiAgfVxuICByZXR1cm4gc2hhcGU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBzaXplRnJvbVNoYXBlKHNoYXBlOiBudW1iZXJbXSk6IG51bWJlciB7XG4gIGlmIChzaGFwZS5sZW5ndGggPT09IDApIHtcbiAgICAvLyBTY2FsYXIuXG4gICAgcmV0dXJuIDE7XG4gIH1cbiAgbGV0IHNpemUgPSBzaGFwZVswXTtcbiAgZm9yIChsZXQgaSA9IDE7IGkgPCBzaGFwZS5sZW5ndGg7IGkrKykge1xuICAgIHNpemUgKj0gc2hhcGVbaV07XG4gIH1cbiAgcmV0dXJuIHNpemU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc1NjYWxhclNoYXBlKHNoYXBlOiBudW1iZXJbXSk6IGJvb2xlYW4ge1xuICByZXR1cm4gc2hhcGUubGVuZ3RoID09PSAwO1xufVxuXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG5leHBvcnQgZnVuY3Rpb24gYXJyYXlzRXF1YWwobjE6IGFueVtdfEZsb2F0MzJBcnJheSwgbjI6IGFueVtdfEZsb2F0MzJBcnJheSkge1xuICBpZiAobjEubGVuZ3RoICE9PSBuMi5sZW5ndGgpIHtcbiAgICByZXR1cm4gZmFsc2U7XG4gIH1cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBuMS5sZW5ndGg7IGkrKykge1xuICAgIGlmIChuMVtpXSAhPT0gbjJbaV0pIHtcbiAgICAgIHJldHVybiBmYWxzZTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIHRydWU7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc0ludChhOiBudW1iZXIpOiBib29sZWFuIHtcbiAgcmV0dXJuIGEgJSAxID09PSAwO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdGFuaCh4OiBudW1iZXIpOiBudW1iZXIge1xuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gIGlmICgoTWF0aCBhcyBhbnkpLnRhbmggIT0gbnVsbCkge1xuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICByZXR1cm4gKE1hdGggYXMgYW55KS50YW5oKHgpO1xuICB9XG4gIGlmICh4ID09PSBJbmZpbml0eSkge1xuICAgIHJldHVybiAxO1xuICB9IGVsc2UgaWYgKHggPT09IC1JbmZpbml0eSkge1xuICAgIHJldHVybiAtMTtcbiAgfSBlbHNlIHtcbiAgICBjb25zdCBlMnggPSBNYXRoLmV4cCgyICogeCk7XG4gICAgcmV0dXJuIChlMnggLSAxKSAvIChlMnggKyAxKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gc2l6ZVRvU3F1YXJpc2hTaGFwZShzaXplOiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgZm9yIChsZXQgYSA9IE1hdGguZmxvb3IoTWF0aC5zcXJ0KHNpemUpKTsgYSA+IDE7IC0tYSkge1xuICAgIGlmIChzaXplICUgYSA9PT0gMCkge1xuICAgICAgcmV0dXJuIFthLCBzaXplIC8gYV07XG4gICAgfVxuICB9XG4gIHJldHVybiBbMSwgc2l6ZV07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVTaHVmZmxlZEluZGljZXMobjogbnVtYmVyKTogVWludDMyQXJyYXkge1xuICBjb25zdCBzaHVmZmxlZEluZGljZXMgPSBuZXcgVWludDMyQXJyYXkobik7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbjsgKytpKSB7XG4gICAgc2h1ZmZsZWRJbmRpY2VzW2ldID0gaTtcbiAgfVxuICBzaHVmZmxlKHNodWZmbGVkSW5kaWNlcyk7XG4gIHJldHVybiBzaHVmZmxlZEluZGljZXM7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBhc3NlcnRBbmRHZXRCcm9hZGNhc3RlZFNoYXBlKFxuICAgIHNoYXBlQTogbnVtYmVyW10sIHNoYXBlQjogbnVtYmVyW10pOiBudW1iZXJbXSB7XG4gIGNvbnN0IHJlc3VsdDogbnVtYmVyW10gPSBbXTtcbiAgbGV0IG5leHRBRGltTXVzdEJlT25lID0gZmFsc2U7XG4gIGxldCBuZXh0QkRpbU11c3RCZU9uZSA9IGZhbHNlO1xuICBjb25zdCBlcnJNc2cgPSBgT3BlcmFuZHMgY291bGQgbm90IGJlIGJyb2FkY2FzdCB0b2dldGhlciB3aXRoIHNoYXBlcyBgICtcbiAgICAgIGAke3NoYXBlQX0gYW5kICR7c2hhcGVCfS4gQ3VycmVudGx5LCB3ZSBvbmx5IHN1cHBvcnQgYSBgICtcbiAgICAgIGBzdHJpY3RlciB2ZXJzaW9uIG9mIGJyb2FkY2FzdGluZyB0aGFuIG51bXB5LmA7XG4gIGNvbnN0IGwgPSBNYXRoLm1heChzaGFwZUEubGVuZ3RoLCBzaGFwZUIubGVuZ3RoKTtcblxuICBzaGFwZUEgPSBzaGFwZUEuc2xpY2UoKS5yZXZlcnNlKCk7XG4gIHNoYXBlQiA9IHNoYXBlQi5zbGljZSgpLnJldmVyc2UoKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBsOyBpKyspIHtcbiAgICBjb25zdCBhID0gc2hhcGVBW2ldIHx8IDE7XG4gICAgY29uc3QgYiA9IHNoYXBlQltpXSB8fCAxO1xuICAgIGlmICgoYiA+IDEgJiYgbmV4dEJEaW1NdXN0QmVPbmUpIHx8IChhID4gMSAmJiBuZXh0QURpbU11c3RCZU9uZSkpIHtcbiAgICAgIHRocm93IEVycm9yKGVyck1zZyk7XG4gICAgfVxuICAgIGlmIChhID4gMSAmJiBiID09PSAxKSB7XG4gICAgICBuZXh0QkRpbU11c3RCZU9uZSA9IHRydWU7XG4gICAgfVxuICAgIGlmIChiID4gMSAmJiBhID09PSAxKSB7XG4gICAgICBuZXh0QURpbU11c3RCZU9uZSA9IHRydWU7XG4gICAgfVxuICAgIGlmIChhID4gMSAmJiBiID4gMSAmJiBhICE9PSBiKSB7XG4gICAgICB0aHJvdyBFcnJvcihlcnJNc2cpO1xuICAgIH1cbiAgICByZXN1bHQucHVzaChNYXRoLm1heChhLCBiKSk7XG4gIH1cbiAgcmV0dXJuIHJlc3VsdC5yZXZlcnNlKCk7XG59XG4iXX0=
