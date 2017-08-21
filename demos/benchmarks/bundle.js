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
    var inputDepth = 1;
    var inputShape = [size, size, inputDepth];
    var outputDepth = 1;
    var fieldSize = 11;
    var stride = 1;
    var zeroPad = conv_util.computeDefaultPad(inputShape, fieldSize, stride);
    var hasBias = true;
    var program = new conv_gpu_1.Conv2DProgram(inputShape, fieldSize, outputDepth, stride, zeroPad, hasBias);
    var outputShape = program.outputShape;
    var out = ndarray_1.Array3D.zeros(outputShape);
    var x = ndarray_1.Array3D.randUniform(inputShape, -1, 1);
    var wShape = conv_util.computeWeightsShape4D(1, outputDepth, fieldSize);
    var W = ndarray_1.Array4D.randUniform(wShape, -1, 1);
    var b = ndarray_1.Array1D.randUniform([outputDepth], -1, 1);
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

},{"../util":34,"./concat3d_util":14,"./copy2d_util":16,"./ndarray":19}],18:[function(require,module,exports){
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

},{"../math/conv_util":15,"../util":34,"./concat3d_util":14,"./copy2d_util":16,"./math":17,"./ndarray":19}],19:[function(require,module,exports){
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
var conv_util = require("../conv_util");
var Conv2DProgram = (function () {
    function Conv2DProgram(xShape, fieldSize, outputDepth, stride, pad, hasBias) {
        this.variableNames = ['x', 'W', 'bias'];
        this.outputShape = conv_util.computeOutputShape3D(xShape, fieldSize, outputDepth, stride, pad);
        var inputDepth = xShape[2];
        this.params = [fieldSize, stride, pad, hasBias];
        var biasSnippet = hasBias ? 'dotProd += getBias(d2);' : '';
        var xNumRows = xShape[0];
        var xNumCols = xShape[1];
        this.userCode = "\n      void main() {\n        vec3 coords = getOutputCoords();\n        float yR = coords.x;\n        float yC = coords.y;\n        float d2 = coords.z;\n\n        vec2 xRCCorner = vec2(yR, yC) * vec2(" + stride + ".0, " + stride + ".0) -\n            vec2(" + pad + ".0, " + pad + ".0);\n        float xRCorner = xRCCorner.x;\n        float xCCorner = xRCCorner.y;\n\n        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int iwR = 0; iwR < " + fieldSize + "; iwR++) {\n          float wR = float(iwR);\n          float xR = xRCorner + wR;\n\n          if (xR < 0.0 || xR >= " + xNumRows + ".0) {\n            continue;\n          }\n\n          for (int iwC = 0; iwC < " + fieldSize + "; iwC++) {\n            float wC = float(iwC);\n            float xC = xCCorner + wC;\n\n            if (xC < 0.0 || xC >= " + xNumCols + ".0) {\n              continue;\n            }\n\n            for (int id1 = 0; id1 < " + inputDepth + "; id1++) {\n              float d1 = float(id1);\n              float xValue = getX(xR, xC, d1);\n              float wValue = getW(wR, wC, d1, d2);\n              dotProd += xValue * wValue;\n            }\n          }\n        }\n        " + biasSnippet + "\n        setOutput(dotProd);\n      }\n    ";
    }
    return Conv2DProgram;
}());
exports.Conv2DProgram = Conv2DProgram;

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
        gl =
            (canvas.getContext('webgl', attributes) ||
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
//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJzb3VyY2VzIjpbIm5vZGVfbW9kdWxlcy9icm93c2VyLXBhY2svX3ByZWx1ZGUuanMiLCJkZW1vcy9iZW5jaG1hcmtzL2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvY29udl9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9jb252X3RyYW5zcG9zZV9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9sb2dzdW1leHBfY3B1X2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvbG9nc3VtZXhwX2dwdV9iZW5jaG1hcmsudHMiLCJkZW1vcy9iZW5jaG1hcmtzL21hdGgtYmVuY2htYXJrLXJ1bi1ncm91cHMudHMiLCJkZW1vcy9iZW5jaG1hcmtzL21hdGgtYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9tYXhfcG9vbF9ncHVfYmVuY2htYXJrLnRzIiwiZGVtb3MvYmVuY2htYXJrcy9tdWxtYXRfY3B1X2JlbmNobWFyay50cyIsImRlbW9zL2JlbmNobWFya3MvbXVsbWF0X2dwdV9iZW5jaG1hcmsudHMiLCJkZW1vcy9kZW1vLWZvb3Rlci50cyIsImRlbW9zL2RlbW8taGVhZGVyLnRzIiwiZGVtb3MvcG9seW1lci1zcGVjLnRzIiwic3JjL21hdGgvY29uY2F0M2RfdXRpbC50cyIsInNyYy9tYXRoL2NvbnZfdXRpbC50cyIsInNyYy9tYXRoL2NvcHkyZF91dGlsLnRzIiwic3JjL21hdGgvbWF0aC50cyIsInNyYy9tYXRoL21hdGhfY3B1LnRzIiwic3JjL21hdGgvbmRhcnJheS50cyIsInNyYy9tYXRoL3dlYmdsL2NvbnZfYmFja3Byb3BfZ3B1LnRzIiwic3JjL21hdGgvd2ViZ2wvY29udl9ncHUudHMiLCJzcmMvbWF0aC93ZWJnbC9ncGdwdV9jb250ZXh0LnRzIiwic3JjL21hdGgvd2ViZ2wvZ3BncHVfbWF0aC50cyIsInNyYy9tYXRoL3dlYmdsL2dwZ3B1X3V0aWwudHMiLCJzcmMvbWF0aC93ZWJnbC9sb2dzdW1leHBfZ3B1LnRzIiwic3JjL21hdGgvd2ViZ2wvbXVsbWF0X2dwdS50cyIsInNyYy9tYXRoL3dlYmdsL211bG1hdF9wYWNrZWRfZ3B1LnRzIiwic3JjL21hdGgvd2ViZ2wvcG9vbF9ncHUudHMiLCJzcmMvbWF0aC93ZWJnbC9zaGFkZXJfY29tcGlsZXIudHMiLCJzcmMvbWF0aC93ZWJnbC90ZXhfdXRpbC50cyIsInNyYy9tYXRoL3dlYmdsL3RleHR1cmVfbWFuYWdlci50cyIsInNyYy9tYXRoL3dlYmdsL3dlYmdsX3V0aWwudHMiLCJzcmMvdGVzdF91dGlsLnRzIiwic3JjL3V0aWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7OztBQzJCQTtJQUtFLHNCQUFZLElBQVksRUFBRSxhQUE0QjtRQUNwRCxJQUFJLENBQUMsSUFBSSxHQUFHLElBQUksQ0FBQztRQUNqQixJQUFJLENBQUMsYUFBYSxHQUFHLGFBQWEsQ0FBQztRQUNuQyxJQUFJLENBQUMsU0FBUyxHQUFHLEVBQUUsQ0FBQztJQUN0QixDQUFDO0lBQ0gsbUJBQUM7QUFBRCxDQVZBLEFBVUMsSUFBQTtBQVZZLG9DQUFZOzs7OztBQ1p6QixvREFBc0Q7QUFDdEQsa0RBQWdGO0FBQ2hGLDBEQUE0RDtBQUM1RCxvRUFBZ0U7QUFDaEUsNERBQThEO0FBQzlELHdFQUFvRTtBQUlwRSxJQUFNLE9BQU8sR0FBRyxFQUFFLENBQUM7QUFFTixRQUFBLGNBQWMsR0FBa0IsVUFBQyxJQUFZO0lBQ3hELElBQU0sS0FBSyxHQUFHLElBQUksNEJBQVksRUFBRSxDQUFDO0lBQ2pDLElBQU0sVUFBVSxHQUFHLElBQUksZ0NBQWMsQ0FBQyxLQUFLLENBQUMsQ0FBQztJQUM3Qyx1QkFBYSxDQUFDLEtBQUssRUFBRSxVQUFVLENBQUMsQ0FBQztJQUVqQyxJQUFNLFVBQVUsR0FBRyxDQUFDLENBQUM7SUFDckIsSUFBTSxVQUFVLEdBQTZCLENBQUMsSUFBSSxFQUFFLElBQUksRUFBRSxVQUFVLENBQUMsQ0FBQztJQUN0RSxJQUFNLFdBQVcsR0FBRyxDQUFDLENBQUM7SUFDdEIsSUFBTSxTQUFTLEdBQUcsRUFBRSxDQUFDO0lBQ3JCLElBQU0sTUFBTSxHQUFHLENBQUMsQ0FBQztJQUNqQixJQUFNLE9BQU8sR0FBRyxTQUFTLENBQUMsaUJBQWlCLENBQUMsVUFBVSxFQUFFLFNBQVMsRUFBRSxNQUFNLENBQUMsQ0FBQztJQUUzRSxJQUFNLE9BQU8sR0FBRyxJQUFJLENBQUM7SUFDckIsSUFBTSxPQUFPLEdBQUcsSUFBSSx3QkFBYSxDQUM3QixVQUFVLEVBQUUsU0FBUyxFQUFFLFdBQVcsRUFBRSxNQUFNLEVBQUUsT0FBTyxFQUFFLE9BQU8sQ0FBQyxDQUFDO0lBQ2xFLElBQU0sV0FBVyxHQUFHLE9BQU8sQ0FBQyxXQUF1QyxDQUFDO0lBQ3BFLElBQU0sR0FBRyxHQUFHLGlCQUFPLENBQUMsS0FBSyxDQUFDLFdBQVcsQ0FBQyxDQUFDO0lBQ3ZDLElBQU0sQ0FBQyxHQUFHLGlCQUFPLENBQUMsV0FBVyxDQUFDLFVBQVUsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNqRCxJQUFNLE1BQU0sR0FBRyxTQUFTLENBQUMscUJBQXFCLENBQUMsQ0FBQyxFQUFFLFdBQVcsRUFBRSxTQUFTLENBQUMsQ0FBQztJQUMxRSxJQUFNLENBQUMsR0FBRyxpQkFBTyxDQUFDLFdBQVcsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDN0MsSUFBTSxDQUFDLEdBQUcsaUJBQU8sQ0FBQyxXQUFXLENBQUMsQ0FBQyxXQUFXLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNwRCxJQUFNLE1BQU0sR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDekIsSUFBTSxNQUFNLEdBQUcsVUFBVSxDQUFDLGNBQWMsQ0FBQyxLQUFLLEVBQUUsT0FBTyxFQUFFLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQztJQUV0RSxJQUFNLEtBQUssR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDaEMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxPQUFPLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztRQUNqQyxVQUFVLENBQUMsVUFBVSxDQUFDLE1BQU0sRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUM7SUFDN0MsQ0FBQztJQUNELEdBQUcsQ0FBQyxTQUFTLEVBQUUsQ0FBQztJQUNoQixJQUFNLE9BQU8sR0FBRyxDQUFDLFdBQVcsQ0FBQyxHQUFHLEVBQUUsR0FBRyxLQUFLLENBQUMsR0FBRyxPQUFPLENBQUM7SUFFdEQsQ0FBQyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ1osQ0FBQyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ1osQ0FBQyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ1osR0FBRyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ2QsVUFBVSxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ3JCLEtBQUssQ0FBQyxhQUFhLENBQUMsTUFBTSxDQUFDLFlBQVksQ0FBQyxDQUFDO0lBQ3pDLEtBQUssQ0FBQyxPQUFPLEVBQUUsQ0FBQztJQUVoQixNQUFNLENBQUMsT0FBTyxDQUFDO0FBQ2pCLENBQUMsQ0FBQzs7Ozs7QUNuREYsb0RBQXNEO0FBQ3RELGtEQUF1RTtBQUN2RSw0RUFBOEU7QUFDOUUsb0VBQWdFO0FBQ2hFLDREQUE4RDtBQUM5RCx3RUFBb0U7QUFHcEUsSUFBTSxPQUFPLEdBQUcsRUFBRSxDQUFDO0FBRU4sUUFBQSxjQUFjLEdBQWtCLFVBQUMsSUFBWTtJQUN4RCxJQUFNLGNBQWMsR0FBRyxDQUFDLENBQUM7SUFDekIsSUFBTSxlQUFlLEdBQUcsQ0FBQyxDQUFDO0lBQzFCLElBQU0sTUFBTSxHQUE2QixDQUFDLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDekQsSUFBTSxTQUFTLEdBQUcsRUFBRSxDQUFDO0lBQ3JCLElBQU0sVUFBVSxHQUFHLENBQUMsQ0FBQztJQUNyQixJQUFNLE9BQU8sR0FBRyxDQUFDLENBQUM7SUFFbEIsSUFBTSxLQUFLLEdBQUcsSUFBSSw0QkFBWSxFQUFFLENBQUM7SUFDakMsSUFBTSxVQUFVLEdBQUcsSUFBSSxnQ0FBYyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQzdDLHVCQUFhLENBQUMsS0FBSyxFQUFFLFVBQVUsQ0FBQyxDQUFDO0lBQ2pDLEtBQUssQ0FBQyw4QkFBOEIsQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUUzQyxJQUFNLE9BQU8sR0FBRyxLQUFLLENBQUM7SUFDdEIsSUFBTSxPQUFPLEdBQUcsSUFBSSwwQ0FBc0IsQ0FDdEMsTUFBTSxFQUFFLFNBQVMsRUFBRSxjQUFjLEVBQUUsVUFBVSxFQUFFLE9BQU8sRUFBRSxPQUFPLENBQUMsQ0FBQztJQUNyRSxJQUFNLFdBQVcsR0FBRyxPQUFPLENBQUMsV0FBdUMsQ0FBQztJQUNwRSxJQUFNLEdBQUcsR0FBRyxpQkFBTyxDQUFDLEtBQUssQ0FBQyxXQUFXLENBQUMsQ0FBQztJQUN2QyxJQUFNLENBQUMsR0FBRyxpQkFBTyxDQUFDLFdBQVcsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDN0MsSUFBTSxNQUFNLEdBQUcsU0FBUyxDQUFDLHFCQUFxQixDQUMxQyxjQUFjLEVBQUUsZUFBZSxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBQ2hELElBQU0sQ0FBQyxHQUFHLGlCQUFPLENBQUMsV0FBVyxDQUFDLE1BQU0sRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUM3QyxJQUFNLE1BQU0sR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUN0QixJQUFNLE1BQU0sR0FBRyxVQUFVLENBQUMsY0FBYyxDQUFDLEtBQUssRUFBRSxPQUFPLEVBQUUsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDO0lBQ3RFLElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQ2pDLFVBQVUsQ0FBQyxVQUFVLENBQUMsTUFBTSxFQUFFLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQztJQUM3QyxDQUFDO0lBQ0QsR0FBRyxDQUFDLFNBQVMsRUFBRSxDQUFDO0lBQ2hCLElBQU0sT0FBTyxHQUFHLENBQUMsV0FBVyxDQUFDLEdBQUcsRUFBRSxHQUFHLEtBQUssQ0FBQyxHQUFHLE9BQU8sQ0FBQztJQUV0RCxVQUFVLENBQUMsT0FBTyxFQUFFLENBQUM7SUFDckIsS0FBSyxDQUFDLGFBQWEsQ0FBQyxNQUFNLENBQUMsWUFBWSxDQUFDLENBQUM7SUFDekMsS0FBSyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ2hCLE1BQU0sQ0FBQyxPQUFPLENBQUM7QUFDakIsQ0FBQyxDQUFDOzs7OztBQzdDRixvREFBdUQ7QUFDdkQsa0RBQXdEO0FBSXhELElBQU0sV0FBVyxHQUFHLEVBQUUsQ0FBQztBQUVWLFFBQUEsY0FBYyxHQUFrQixVQUFDLElBQVk7SUFDeEQsSUFBTSxJQUFJLEdBQUcsSUFBSSx5QkFBYyxFQUFFLENBQUM7SUFDbEMsSUFBTSxDQUFDLEdBQUcsaUJBQU8sQ0FBQyxXQUFXLENBQVUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDNUQsSUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsV0FBVyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7UUFDckMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUNwQixDQUFDO0lBQ0QsSUFBTSxHQUFHLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQzlCLE1BQU0sQ0FBQyxDQUFDLEdBQUcsR0FBRyxLQUFLLENBQUMsR0FBRyxXQUFXLENBQUM7QUFDckMsQ0FBQyxDQUFDOzs7OztBQ2hCRixrREFBc0U7QUFDdEUsb0VBQWdFO0FBQ2hFLDREQUE4RDtBQUM5RCxvRUFBb0U7QUFDcEUsd0VBQW9FO0FBSXBFLElBQU0sT0FBTyxHQUFHLENBQUMsQ0FBQztBQUVMLFFBQUEsY0FBYyxHQUFrQixVQUFDLElBQVk7SUFDeEQsSUFBTSxLQUFLLEdBQUcsSUFBSSw0QkFBWSxFQUFFLENBQUM7SUFDakMsSUFBTSxVQUFVLEdBQUcsSUFBSSxnQ0FBYyxDQUFDLEtBQUssQ0FBQyxDQUFDO0lBQzdDLHVCQUFhLENBQUMsS0FBSyxFQUFFLFVBQVUsQ0FBQyxDQUFDO0lBQ2pDLElBQU0sR0FBRyxHQUFHLElBQUksZ0JBQU0sQ0FBQyxFQUFDLE9BQU8sRUFBRSxVQUFVLENBQUMsY0FBYyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUMsQ0FBQyxDQUFDO0lBQ3JFLElBQU0sQ0FBQyxHQUFHLGlCQUFPLENBQUMsV0FBVyxDQUFDLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ25ELElBQU0sT0FBTyxHQUFHLElBQUksZ0NBQWdCLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDO0lBQzdDLElBQU0sTUFBTSxHQUFHLFVBQVUsQ0FBQyxjQUFjLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDO0lBRW5FLElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQ2pDLFVBQVUsQ0FBQyxVQUFVLENBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUM7SUFDMUMsQ0FBQztJQUNELEdBQUcsQ0FBQyxTQUFTLEVBQUUsQ0FBQztJQUNoQixJQUFNLE9BQU8sR0FBRyxDQUFDLFdBQVcsQ0FBQyxHQUFHLEVBQUUsR0FBRyxLQUFLLENBQUMsR0FBRyxPQUFPLENBQUM7SUFDdEQsQ0FBQyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ1osR0FBRyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ2QsVUFBVSxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ3JCLEtBQUssQ0FBQyxhQUFhLENBQUMsTUFBTSxDQUFDLFlBQVksQ0FBQyxDQUFDO0lBQ3pDLEtBQUssQ0FBQyxPQUFPLEVBQUUsQ0FBQztJQUVoQixNQUFNLENBQUMsT0FBTyxDQUFDO0FBQ2pCLENBQUMsQ0FBQzs7Ozs7QUNoQ0YseUNBQTREO0FBQzVELHlEQUEyRDtBQUMzRCw2RUFBK0U7QUFDL0UsbUVBQXFFO0FBQ3JFLG1FQUFxRTtBQUNyRSxpRUFBbUU7QUFDbkUsNkRBQStEO0FBQy9ELDZEQUErRDtBQUVsRCxRQUFBLG9CQUFvQixHQUF3QjtJQUN2RDtRQUNFLElBQUksRUFDQSxzQ0FBc0M7WUFDbEMsb0NBQW9DO1FBQzVDLEdBQUcsRUFBRSxDQUFDO1FBQ04sR0FBRyxFQUFFLElBQUk7UUFDVCxRQUFRLEVBQUUsRUFBRTtRQUNaLHdCQUF3QixFQUFFLFVBQUMsSUFBWSxJQUFLLE9BQUEsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLEVBQWpCLENBQWlCO1FBQzdELGFBQWEsRUFBRTtZQUNiLElBQUksd0JBQVksQ0FBQyxZQUFZLEVBQUUsb0JBQW9CLENBQUMsY0FBYyxDQUFDO1lBQ25FLElBQUksd0JBQVksQ0FBQyxZQUFZLEVBQUUsb0JBQW9CLENBQUMsY0FBYyxDQUFDO1NBQ3BFO0tBQ0Y7SUFDRDtRQUNFLElBQUksRUFBRSxvREFBb0Q7UUFDMUQsR0FBRyxFQUFFLENBQUM7UUFDTixHQUFHLEVBQUUsSUFBSTtRQUNULFFBQVEsRUFBRSxFQUFFO1FBQ1osd0JBQXdCLEVBQUUsVUFBQyxJQUFZLElBQUssT0FBQSxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsRUFBakIsQ0FBaUI7UUFDN0QsYUFBYSxFQUFFLENBQUMsSUFBSSx3QkFBWSxDQUM1Qix1QkFBdUIsRUFBRSxrQkFBa0IsQ0FBQyxjQUFjLENBQUMsQ0FBQztLQUNqRTtJQUNEO1FBQ0UsSUFBSSxFQUFFLGlFQUFpRTtRQUN2RSxHQUFHLEVBQUUsQ0FBQztRQUNOLEdBQUcsRUFBRSxJQUFJO1FBQ1QsUUFBUSxFQUFFLEVBQUU7UUFDWix3QkFBd0IsRUFBRSxVQUFDLElBQVksSUFBSyxPQUFBLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxFQUFqQixDQUFpQjtRQUM3RCxhQUFhLEVBQUUsQ0FBQyxJQUFJLHdCQUFZLENBQzVCLHVCQUF1QixFQUFFLDRCQUE0QixDQUFDLGNBQWMsQ0FBQyxDQUFDO0tBQzNFO0lBQ0Q7UUFDRSxJQUFJLEVBQUUsZ0JBQWdCO1FBQ3RCLEdBQUcsRUFBRSxDQUFDO1FBQ04sR0FBRyxFQUFFLElBQUk7UUFDVCxRQUFRLEVBQUUsRUFBRTtRQUNaLHdCQUF3QixFQUFFLFVBQUMsSUFBWSxJQUFLLE9BQUEsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLEVBQWpCLENBQWlCO1FBQzdELGFBQWEsRUFBRSxDQUFDLElBQUksd0JBQVksQ0FDNUIsdUJBQXVCLEVBQ3ZCLHNCQUFzQixDQUFDLHVCQUF1QixDQUFDLENBQUM7S0FDckQ7SUFDRDtRQUNFLElBQUksRUFBRSw0Q0FBNEM7UUFDbEQsR0FBRyxFQUFFLENBQUM7UUFDTixHQUFHLEVBQUUsSUFBSTtRQUNULFFBQVEsRUFBRSxFQUFFO1FBQ1osd0JBQXdCLEVBQUUsVUFBQyxJQUFZLElBQUssT0FBQSxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsRUFBakIsQ0FBaUI7UUFDN0QsYUFBYSxFQUFFO1lBQ2IsSUFBSSx3QkFBWSxDQUNaLGVBQWUsRUFBRSx1QkFBdUIsQ0FBQyxjQUFjLENBQUM7WUFDNUQsSUFBSSx3QkFBWSxDQUFDLGVBQWUsRUFBRSx1QkFBdUIsQ0FBQyxjQUFjLENBQUM7U0FDMUU7S0FDRjtDQUNGLENBQUM7Ozs7Ozs7Ozs7Ozs7OztBQy9ERiwwQkFBd0I7QUFDeEIsMEJBQXdCO0FBRXhCLGdEQUFtRTtBQUduRSx5RUFBaUU7QUFHdEQsUUFBQSxvQkFBb0IsR0FBaUMsNkJBQWMsQ0FDMUUsRUFBQyxFQUFFLEVBQUUsZ0JBQWdCLEVBQUUsVUFBVSxFQUFFLEVBQUMsc0JBQXNCLEVBQUUsS0FBSyxFQUFDLEVBQUMsQ0FBQyxDQUFDO0FBRXpFO0lBQW1DLGlDQUFvQjtJQUF2RDs7SUFtTUEsQ0FBQztJQTlMQyw2QkFBSyxHQUFMO1FBQUEsaUJBdUJDO1FBckJDLElBQU0sc0JBQXNCLEdBQWEsRUFBRSxDQUFDO1FBQzVDLElBQUksQ0FBQyxZQUFZLEdBQUcsRUFBRSxDQUFDO1FBQ3ZCLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsZ0RBQW9CLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7WUFDckQsc0JBQXNCLENBQUMsSUFBSSxDQUFDLGdEQUFvQixDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDO1lBQzFELElBQUksQ0FBQyxZQUFZLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ2hDLENBQUM7UUFDRCxJQUFJLENBQUMsc0JBQXNCLEdBQUcsc0JBQXNCLENBQUM7UUFHckQsVUFBVSxDQUFDO1lBQ1QsSUFBTSxVQUFVLEdBQUcsS0FBSSxDQUFDLGdCQUFnQixDQUFDLFdBQVcsQ0FBQyxDQUFDO1lBQ3RELElBQU0sV0FBVyxHQUFHLEtBQUksQ0FBQyxnQkFBZ0IsQ0FBQyxXQUFXLENBQUMsQ0FBQztvQ0FDOUMsQ0FBQztnQkFDUixVQUFVLENBQUMsQ0FBQyxDQUFDLENBQUMsZ0JBQWdCLENBQUMsT0FBTyxFQUFFO29CQUN0QyxLQUFJLENBQUMsaUJBQWlCLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQzVCLENBQUMsQ0FBQyxDQUFDO2dCQUNILFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxnQkFBZ0IsQ0FBQyxPQUFPLEVBQUU7b0JBQ3ZDLEtBQUksQ0FBQyxZQUFZLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDO2dCQUM5QixDQUFDLENBQUMsQ0FBQztZQUNMLENBQUM7WUFQRCxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLFVBQVUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFO3dCQUFqQyxDQUFDO2FBT1Q7UUFDSCxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFDUixDQUFDO0lBRU8seUNBQWlCLEdBQXpCLFVBQTBCLHNCQUE4QjtRQUN0RCxJQUFNLGlCQUFpQixHQUFHLGdEQUFvQixDQUFDLHNCQUFzQixDQUFDLENBQUM7UUFFdkUsSUFBTSxNQUFNLEdBQUcsSUFBSSxDQUFDLGdCQUFnQixDQUFDLFdBQVcsQ0FBQyxDQUFDLHNCQUFzQixDQUNuRCxDQUFDO1FBQ3RCLElBQU0sT0FBTyxHQUFHLE1BQU0sQ0FBQyxVQUFVLENBQUMsSUFBSSxDQUE2QixDQUFDO1FBRXBFLElBQU0sUUFBUSxHQUFvQixFQUFFLENBQUM7UUFDckMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxpQkFBaUIsQ0FBQyxhQUFhLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7WUFDaEUsSUFBTSxHQUFHLEdBQUcsSUFBSSxDQUFDLEtBQUssQ0FBQyxHQUFHLEdBQUcsQ0FBQyxHQUFHLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxNQUFNLENBQUMsQ0FBQztZQUN6RSxRQUFRLENBQUMsSUFBSSxDQUFDO2dCQUNaLElBQUksRUFBRSxpQkFBaUIsQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUMsU0FBUztnQkFDbEQsSUFBSSxFQUFFLEtBQUs7Z0JBQ1gsS0FBSyxFQUFFLGlCQUFpQixDQUFDLGFBQWEsQ0FBQyxDQUFDLENBQUMsQ0FBQyxJQUFJO2dCQUM5QyxXQUFXLEVBQUUsTUFBTSxHQUFHLEdBQUcsR0FBRyxjQUFjO2dCQUMxQyxlQUFlLEVBQUUsTUFBTSxHQUFHLEdBQUcsR0FBRyxjQUFjO2dCQUM5QyxXQUFXLEVBQUUsQ0FBQztnQkFDZCxjQUFjLEVBQUUsQ0FBQztnQkFDakIsV0FBVyxFQUFFLENBQUM7Z0JBQ2QsV0FBVyxFQUFFLENBQUM7YUFDZixDQUFDLENBQUM7UUFDTCxDQUFDO1FBRUQsSUFBTSxLQUFLLEdBQUcsSUFBSSxLQUFLLENBQUMsT0FBTyxFQUFFO1lBQy9CLElBQUksRUFBRSxNQUFNO1lBQ1osSUFBSSxFQUFFLEVBQUMsUUFBUSxVQUFBLEVBQUM7WUFDaEIsT0FBTyxFQUFFO2dCQUNQLFNBQVMsRUFBRSxFQUFDLFFBQVEsRUFBRSxDQUFDLEVBQUM7Z0JBQ3hCLFVBQVUsRUFBRSxLQUFLO2dCQUNqQixNQUFNLEVBQUU7b0JBQ04sS0FBSyxFQUFFLENBQUM7NEJBQ04sSUFBSSxFQUFFLFFBQVE7NEJBQ2QsUUFBUSxFQUFFLFFBQVE7NEJBQ2xCLEtBQUssRUFBRTtnQ0FDTCxHQUFHLEVBQUUsaUJBQWlCLENBQUMsR0FBRztnQ0FDMUIsR0FBRyxFQUFFLGlCQUFpQixDQUFDLEdBQUc7Z0NBQzFCLFFBQVEsRUFBRSxpQkFBaUIsQ0FBQyxRQUFRO2dDQUNwQyxRQUFRLEVBQUUsVUFBQyxLQUFhO29DQUN0QixNQUFNLENBQUMsaUJBQWlCLENBQUMsd0JBQXdCLElBQUksSUFBSTt3Q0FDckQsaUJBQWlCLENBQUMsd0JBQXdCLENBQUMsQ0FBQyxLQUFLLENBQUM7d0NBQ2xELENBQUMsS0FBSyxDQUFDO2dDQUNiLENBQUM7NkJBRUs7eUJBQ1QsQ0FBQztvQkFDRixLQUFLLEVBQUUsQ0FBQzs0QkFDTixLQUFLLEVBQUU7Z0NBQ0wsUUFBUSxFQUFFLFVBQUMsS0FBSyxFQUFFLEtBQUssRUFBRSxNQUFNO29DQUM3QixNQUFNLENBQUMsS0FBSyxHQUFHLElBQUksQ0FBQztnQ0FDdEIsQ0FBQzs2QkFDRjt5QkFDRixDQUFDO2lCQUNIO2dCQUNELFFBQVEsRUFBRSxFQUFDLElBQUksRUFBRSxPQUFPLEVBQUM7Z0JBQ3pCLEtBQUssRUFBRSxFQUFDLElBQUksRUFBRSxpQkFBaUIsQ0FBQyxJQUFJLEVBQUM7YUFDdEM7U0FDRixDQUFDLENBQUM7UUFDSCxNQUFNLENBQUMsS0FBSyxDQUFDLE9BQU8sR0FBRyxNQUFNLENBQUM7UUFFOUIsSUFBTSxVQUFVLEdBQ1osSUFBSSxDQUFDLGdCQUFnQixDQUFDLGNBQWMsQ0FBQyxDQUFDLHNCQUFzQixDQUNqRCxDQUFDO1FBQ2hCLFVBQVUsQ0FBQyxLQUFLLENBQUMsT0FBTyxHQUFHLE9BQU8sQ0FBQztRQUVuQyxJQUFNLGVBQWUsR0FDakIsSUFBSSxDQUFDLGdCQUFnQixDQUFDLG9CQUFvQixDQUFDLENBQUMsc0JBQXNCLENBQ3ZELENBQUM7UUFDaEIsZUFBZSxDQUFDLFNBQVMsR0FBRyxFQUFFLENBQUM7UUFDL0IsZUFBZSxDQUFDLEtBQUssQ0FBQyxPQUFPLEdBQUcsTUFBTSxDQUFDO1FBR3ZDLElBQU0sT0FBTyxHQUFHLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDekIsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxpQkFBaUIsQ0FBQyxhQUFhLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7WUFDaEUsT0FBTyxDQUFDLElBQUksQ0FBQyxpQkFBaUIsQ0FBQyxhQUFhLENBQUMsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUM7UUFDeEQsQ0FBQztRQUNELGVBQWUsQ0FBQyxXQUFXLENBQUMsSUFBSSxDQUFDLGtCQUFrQixDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7UUFFOUQsSUFBSSxDQUFDLGlCQUFpQixDQUNsQixLQUFLLEVBQUUsaUJBQWlCLEVBQUUsc0JBQXNCLEVBQ2hELGlCQUFpQixDQUFDLEdBQUcsQ0FBQyxDQUFDO0lBQzdCLENBQUM7SUFFTywwQ0FBa0IsR0FBMUIsVUFBMkIsTUFBZ0I7UUFDekMsSUFBTSxtQkFBbUIsR0FBRyxRQUFRLENBQUMsYUFBYSxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQzFELG1CQUFtQixDQUFDLFNBQVMsR0FBRyxnQ0FBZ0MsQ0FBQztRQUVqRSxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE1BQU0sQ0FBQyxNQUFNLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztZQUN2QyxJQUFNLG9CQUFvQixHQUFHLFFBQVEsQ0FBQyxhQUFhLENBQUMsS0FBSyxDQUFDLENBQUM7WUFDM0Qsb0JBQW9CLENBQUMsU0FBUyxHQUFHLGlDQUFpQyxDQUFDO1lBQ25FLG9CQUFvQixDQUFDLFNBQVMsR0FBRyxNQUFNLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDM0MsbUJBQW1CLENBQUMsV0FBVyxDQUFDLG9CQUFvQixDQUFDLENBQUM7UUFDeEQsQ0FBQztRQUNELE1BQU0sQ0FBQyxtQkFBbUIsQ0FBQztJQUM3QixDQUFDO0lBRU8seUNBQWlCLEdBQXpCLFVBQ0ksS0FBWSxFQUFFLGlCQUFvQyxFQUNsRCxzQkFBOEIsRUFBRSxJQUFZO1FBRmhELGlCQXFFQztRQWxFQyxJQUFNLGVBQWUsR0FDakIsSUFBSSxDQUFDLGdCQUFnQixDQUFDLG9CQUFvQixDQUFDLENBQUMsc0JBQXNCLENBQ3ZELENBQUM7UUFDaEIsRUFBRSxDQUFDLENBQUMsSUFBSSxHQUFHLGlCQUFpQixDQUFDLEdBQUc7WUFDNUIsSUFBSSxDQUFDLFlBQVksQ0FBQyxzQkFBc0IsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUM5QyxJQUFJLENBQUMsWUFBWSxDQUFDLHNCQUFzQixDQUFDLEdBQUcsS0FBSyxDQUFDO1lBRWxELGVBQWUsQ0FBQyxLQUFLLENBQUMsT0FBTyxHQUFHLEVBQUUsQ0FBQztZQUVuQyxJQUFNLE1BQU0sR0FDUixJQUFJLENBQUMsZ0JBQWdCLENBQUMsV0FBVyxDQUFDLENBQUMsc0JBQXNCLENBQ3hDLENBQUM7WUFDdEIsTUFBTSxDQUFDLEtBQUssQ0FBQyxPQUFPLEdBQUcsT0FBTyxDQUFDO1lBQy9CLEtBQUssQ0FBQyxNQUFNLEVBQUUsQ0FBQztZQUVmLElBQU0sVUFBVSxHQUNaLElBQUksQ0FBQyxnQkFBZ0IsQ0FBQyxjQUFjLENBQUMsQ0FBQyxzQkFBc0IsQ0FDakQsQ0FBQztZQUNoQixVQUFVLENBQUMsS0FBSyxDQUFDLE9BQU8sR0FBRyxNQUFNLENBQUM7WUFFbEMsTUFBTSxDQUFDO1FBQ1QsQ0FBQztRQUVELElBQU0sbUJBQW1CLEdBQUcsUUFBUSxDQUFDLGFBQWEsQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUMxRCxtQkFBbUIsQ0FBQyxTQUFTLEdBQUcsZ0NBQWdDLENBQUM7UUFFakUsSUFBTSxTQUFTLEdBQWEsQ0FBQyxFQUFFLEdBQUcsSUFBSSxDQUFDLENBQUM7UUFDeEMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxpQkFBaUIsQ0FBQyxhQUFhLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7WUFDaEUsSUFBTSxZQUFZLEdBQUcsaUJBQWlCLENBQUMsYUFBYSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3hELElBQU0sYUFBYSxHQUFHLFlBQVksQ0FBQyxhQUFhLENBQUM7WUFFakQsSUFBTSxJQUFJLEdBQUcsaUJBQWlCLENBQUMsd0JBQXdCLElBQUksSUFBSTtnQkFDM0QsaUJBQWlCLENBQUMsd0JBQXdCLENBQUMsSUFBSSxDQUFDO2dCQUNoRCxJQUFJLENBQUM7WUFFVCxJQUFJLFlBQVksU0FBUSxDQUFDO1lBQ3pCLElBQUksU0FBUyxTQUFRLENBQUM7WUFDdEIsSUFBSSxJQUFJLEdBQUcsQ0FBQyxDQUFDO1lBQ2IsSUFBSSxPQUFPLEdBQUcsSUFBSSxDQUFDO1lBRW5CLElBQUksQ0FBQztnQkFDSCxJQUFJLEdBQUcsYUFBYSxDQUFDLElBQUksQ0FBQyxDQUFDO2dCQUMzQixZQUFZLEdBQUcsSUFBSSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLENBQUM7Z0JBQ3RDLFNBQVMsR0FBRyxZQUFZLENBQUM7WUFDM0IsQ0FBQztZQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUM7Z0JBQ1gsT0FBTyxHQUFHLEtBQUssQ0FBQztnQkFDaEIsWUFBWSxHQUFHLE9BQU8sQ0FBQztnQkFDdkIsU0FBUyxHQUFHLENBQUMsQ0FBQyxPQUFPLENBQUM7WUFDeEIsQ0FBQztZQUVELEVBQUUsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUNkLEVBQUUsQ0FBQyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7b0JBQ1osWUFBWSxDQUFDLFNBQVMsQ0FBQyxJQUFJLENBQUMsRUFBQyxDQUFDLEVBQUUsSUFBSSxFQUFFLENBQUMsRUFBRSxJQUFJLEVBQUMsQ0FBQyxDQUFDO2dCQUNsRCxDQUFDO2dCQUNELFNBQVMsQ0FBQyxJQUFJLENBQUMsWUFBWSxDQUFDLENBQUM7WUFDL0IsQ0FBQztZQUNELE9BQU8sQ0FBQyxHQUFHLENBQUMsWUFBWSxDQUFDLElBQUksR0FBRyxHQUFHLEdBQUcsSUFBSSxHQUFHLEtBQUssR0FBRyxTQUFTLENBQUMsQ0FBQztRQUNsRSxDQUFDO1FBQ0QsZUFBZSxDQUFDLFdBQVcsQ0FBQyxJQUFJLENBQUMsa0JBQWtCLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQztRQUVoRSxJQUFJLElBQUksaUJBQWlCLENBQUMsUUFBUSxDQUFDO1FBRW5DLFVBQVUsQ0FDTixjQUFNLE9BQUEsS0FBSSxDQUFDLGlCQUFpQixDQUN4QixLQUFLLEVBQUUsaUJBQWlCLEVBQUUsc0JBQXNCLEVBQUUsSUFBSSxDQUFDLEVBRHJELENBQ3FELEVBQzNELEdBQUcsQ0FBQyxDQUFDO0lBQ1gsQ0FBQztJQUNILG9CQUFDO0FBQUQsQ0FuTUEsQUFtTUMsQ0FuTWtDLDRCQUFvQixHQW1NdEQ7QUFuTVksc0NBQWE7QUFvTTFCLFFBQVEsQ0FBQyxlQUFlLENBQUMsYUFBYSxDQUFDLFNBQVMsQ0FBQyxFQUFFLEVBQUUsYUFBYSxDQUFDLENBQUM7Ozs7O0FDaE5wRSxvREFBc0Q7QUFDdEQsa0RBQXVFO0FBQ3ZFLG9FQUFnRTtBQUNoRSw0REFBOEQ7QUFDOUQsMERBQTREO0FBQzVELHdFQUFvRTtBQUlwRSxJQUFNLE9BQU8sR0FBRyxFQUFFLENBQUM7QUFFTixRQUFBLHVCQUF1QixHQUFrQixVQUFDLElBQVk7SUFDakUsSUFBTSxTQUFTLEdBQUcsS0FBSyxDQUFDO0lBQ3hCLE1BQU0sQ0FBQyxXQUFXLENBQUMsSUFBSSxFQUFFLFNBQVMsQ0FBQyxDQUFDO0FBQ3RDLENBQUMsQ0FBQztBQUVXLFFBQUEsNkJBQTZCLEdBQWtCLFVBQUMsSUFBWTtJQUN2RSxJQUFNLFNBQVMsR0FBRyxJQUFJLENBQUM7SUFDdkIsTUFBTSxDQUFDLFdBQVcsQ0FBQyxJQUFJLEVBQUUsU0FBUyxDQUFDLENBQUM7QUFDdEMsQ0FBQyxDQUFDO0FBRUYscUJBQXFCLElBQVksRUFBRSxTQUFrQjtJQUNuRCxJQUFNLEtBQUssR0FBRyxJQUFJLDRCQUFZLEVBQUUsQ0FBQztJQUNqQyxJQUFNLFVBQVUsR0FBRyxJQUFJLGdDQUFjLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDN0MsdUJBQWEsQ0FBQyxLQUFLLEVBQUUsVUFBVSxDQUFDLENBQUM7SUFFakMsSUFBTSxXQUFXLEdBQUcsQ0FBQyxDQUFDO0lBQ3RCLElBQU0sTUFBTSxHQUE2QixDQUFDLElBQUksRUFBRSxJQUFJLEVBQUUsV0FBVyxDQUFDLENBQUM7SUFDbkUsSUFBTSxTQUFTLEdBQUcsRUFBRSxDQUFDO0lBQ3JCLElBQU0sTUFBTSxHQUFHLENBQUMsQ0FBQztJQUNqQixJQUFNLE9BQU8sR0FBRyxTQUFTLENBQUMsaUJBQWlCLENBQUMsTUFBTSxFQUFFLFNBQVMsRUFBRSxNQUFNLENBQUMsQ0FBQztJQUV2RSxJQUFNLE9BQU8sR0FDVCxJQUFJLHdCQUFhLENBQUMsTUFBTSxFQUFFLFNBQVMsRUFBRSxNQUFNLEVBQUUsT0FBTyxFQUFFLEtBQUssRUFBRSxTQUFTLENBQUMsQ0FBQztJQUM1RSxJQUFNLEdBQUcsR0FBRyxpQkFBTyxDQUFDLEtBQUssQ0FBQyxPQUFPLENBQUMsV0FBVyxDQUFDLENBQUM7SUFDL0MsSUFBTSxDQUFDLEdBQUcsaUJBQU8sQ0FBQyxXQUFXLENBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzdDLElBQU0sTUFBTSxHQUFHLFVBQVUsQ0FBQyxjQUFjLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxDQUFDO0lBRW5FLElBQU0sS0FBSyxHQUFHLFdBQVcsQ0FBQyxHQUFHLEVBQUUsQ0FBQztJQUNoQyxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLE9BQU8sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1FBQ2pDLFVBQVUsQ0FBQyxVQUFVLENBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLENBQUM7SUFDMUMsQ0FBQztJQUNELEdBQUcsQ0FBQyxTQUFTLEVBQUUsQ0FBQztJQUNoQixJQUFNLE9BQU8sR0FBRyxDQUFDLFdBQVcsQ0FBQyxHQUFHLEVBQUUsR0FBRyxLQUFLLENBQUMsR0FBRyxPQUFPLENBQUM7SUFFdEQsQ0FBQyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ1osR0FBRyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ2QsVUFBVSxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBQ3JCLEtBQUssQ0FBQyxhQUFhLENBQUMsTUFBTSxDQUFDLFlBQVksQ0FBQyxDQUFDO0lBQ3pDLEtBQUssQ0FBQyxPQUFPLEVBQUUsQ0FBQztJQUVoQixNQUFNLENBQUMsT0FBTyxDQUFDO0FBQ2pCLENBQUM7Ozs7O0FDcERELG9EQUF1RDtBQUN2RCxrREFBd0Q7QUFJeEQsSUFBTSxpQkFBaUIsR0FBRyxDQUFDLENBQUM7QUFFZixRQUFBLGNBQWMsR0FBa0IsVUFBQyxJQUFZO0lBQ3hELEVBQUUsQ0FBQyxDQUFDLElBQUksR0FBRyxHQUFHLENBQUMsQ0FBQyxDQUFDO1FBQ2YsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ1osQ0FBQztJQUNELElBQU0sSUFBSSxHQUFHLElBQUkseUJBQWMsRUFBRSxDQUFDO0lBQ2xDLElBQU0sQ0FBQyxHQUFHLGlCQUFPLENBQUMsV0FBVyxDQUFVLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzVELElBQU0sQ0FBQyxHQUFHLGlCQUFPLENBQUMsV0FBVyxDQUFVLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzVELElBQU0sSUFBSSxHQUFHLENBQUMsSUFBSSxHQUFHLEdBQUcsQ0FBQyxHQUFHLGlCQUFpQixHQUFHLENBQUMsQ0FBQztJQUNsRCxJQUFNLEtBQUssR0FBRyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDaEMsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztRQUM5QixJQUFJLENBQUMsTUFBTSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNwQixDQUFDO0lBQ0QsSUFBTSxHQUFHLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQzlCLE1BQU0sQ0FBQyxDQUFDLEdBQUcsR0FBRyxLQUFLLENBQUMsR0FBRyxJQUFJLENBQUM7QUFDOUIsQ0FBQyxDQUFDOzs7OztBQ3JCRiw0Q0FBc0Q7QUFDdEQsa0RBQStDO0FBQy9DLG9FQUFnRTtBQUNoRSw4REFBOEQ7QUFDOUQsNERBQThEO0FBQzlELDBFQUE0RTtBQUM1RSwrQ0FBaUQ7QUFJakQsSUFBTSxPQUFPLEdBQUcsRUFBRSxDQUFDO0FBRU4sUUFBQSxjQUFjLEdBQWtCLFVBQUMsSUFBWTtJQUN4RCxJQUFNLEtBQUssR0FBRyxJQUFJLDRCQUFZLEVBQUUsQ0FBQztJQUNqQyxJQUFNLFFBQVEsR0FBRyxLQUFLLENBQUMsbUJBQW1CLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQ3ZELElBQU0sUUFBUSxHQUFHLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDdkQsSUFBTSxhQUFhLEdBQUcsS0FBSyxDQUFDLG1CQUFtQixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUU1RCxJQUFNLElBQUksR0FBRyxJQUFJLGlCQUFPLENBQ3BCLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxFQUFFLEVBQUMsT0FBTyxFQUFFLFFBQVEsRUFBRSxjQUFjLEVBQUUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUMsQ0FBQyxDQUFDO0lBQ3JFLElBQU0sSUFBSSxHQUFHLElBQUksaUJBQU8sQ0FDcEIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUUsRUFBQyxPQUFPLEVBQUUsUUFBUSxFQUFFLGNBQWMsRUFBRSxDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsRUFBQyxDQUFDLENBQUM7SUFDckUsSUFBTSxNQUFNLEdBQUcsSUFBSSxpQkFBTyxDQUN0QixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsRUFBRSxFQUFDLE9BQU8sRUFBRSxhQUFhLEVBQUUsY0FBYyxFQUFFLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxFQUFDLENBQUMsQ0FBQztJQUMxRSxJQUFNLE9BQU8sR0FBRyxJQUFJLDBCQUFhLENBQUMsSUFBSSxDQUFDLEtBQUssRUFBRSxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUM7SUFDMUQsSUFBTSxNQUFNLEdBQ1IsVUFBVSxDQUFDLGNBQWMsQ0FBQyxLQUFLLEVBQUUsT0FBTyxFQUFFLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDO0lBQ3BFLElBQU0sQ0FBQyxHQUFHLFNBQVMsQ0FBQyxrQkFBa0IsQ0FBQyxJQUFJLEdBQUcsSUFBSSxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzNELElBQU0sQ0FBQyxHQUFHLFNBQVMsQ0FBQyxrQkFBa0IsQ0FBQyxJQUFJLEdBQUcsSUFBSSxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzNELEtBQUssQ0FBQyxxQkFBcUIsQ0FBQyxRQUFRLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNyRCxLQUFLLENBQUMscUJBQXFCLENBQUMsUUFBUSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFFckQsSUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7UUFDakMsVUFBVSxDQUFDLFVBQVUsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFDdEQsQ0FBQztJQUNELEtBQUssQ0FBQyx5QkFBeUIsQ0FBQyxhQUFhLEVBQUUsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQzNELElBQU0sT0FBTyxHQUFHLENBQUMsV0FBVyxDQUFDLEdBQUcsRUFBRSxHQUFHLEtBQUssQ0FBQyxHQUFHLE9BQU8sQ0FBQztJQUV0RCxLQUFLLENBQUMsbUJBQW1CLENBQUMsUUFBUSxDQUFDLENBQUM7SUFDcEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLFFBQVEsQ0FBQyxDQUFDO0lBQ3BDLEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxhQUFhLENBQUMsQ0FBQztJQUN6QyxLQUFLLENBQUMsYUFBYSxDQUFDLE1BQU0sQ0FBQyxZQUFZLENBQUMsQ0FBQztJQUN6QyxLQUFLLENBQUMsT0FBTyxFQUFFLENBQUM7SUFFaEIsTUFBTSxDQUFDLE9BQU8sQ0FBQztBQUNqQixDQUFDLENBQUM7QUFFVyxRQUFBLHFCQUFxQixHQUFrQixVQUFDLElBQVk7SUFDL0QsSUFBTSxLQUFLLEdBQUcsSUFBSSw0QkFBWSxFQUFFLENBQUM7SUFDakMsSUFBTSxPQUFPLEdBQ1QsS0FBSyxDQUFDLGFBQWEsQ0FBQyxpQkFBaUIsQ0FBQyx1QkFBdUIsQ0FDekQsSUFBSSxFQUFFLHdCQUFpQixDQUFDLE9BQU8sRUFBRSx3QkFBaUIsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDO0lBRXJFLElBQU0sUUFBUSxHQUFHLEtBQUssQ0FBQyx5QkFBeUIsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDN0QsSUFBTSxRQUFRLEdBQUcsS0FBSyxDQUFDLHlCQUF5QixDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUM3RCxJQUFNLGFBQWEsR0FBRyxLQUFLLENBQUMseUJBQXlCLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBRWxFLElBQU0sQ0FBQyxHQUFHLFNBQVMsQ0FBQyxrQkFBa0IsQ0FBQyxJQUFJLEdBQUcsSUFBSSxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzNELElBQU0sQ0FBQyxHQUFHLFNBQVMsQ0FBQyxrQkFBa0IsQ0FBQyxJQUFJLEdBQUcsSUFBSSxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQzNELEtBQUssQ0FBQywyQkFBMkIsQ0FBQyxRQUFRLEVBQUUsSUFBSSxFQUFFLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQztJQUMzRCxLQUFLLENBQUMsMkJBQTJCLENBQUMsUUFBUSxFQUFFLElBQUksRUFBRSxJQUFJLEVBQUUsQ0FBQyxDQUFDLENBQUM7SUFFM0QsSUFBTSxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsRUFBRSxDQUFDO0lBQ2hDLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsT0FBTyxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7UUFDakMsaUJBQWlCLENBQUMsb0JBQW9CLENBQ2xDLEtBQUssRUFBRSxPQUFPLEVBQUUsUUFBUSxFQUFFLFFBQVEsRUFBRSxhQUFhLEVBQUUsQ0FBQyxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQztJQUN2RSxDQUFDO0lBRUQsS0FBSyxDQUFDLCtCQUErQixDQUFDLGFBQWEsRUFBRSxJQUFJLEVBQUUsSUFBSSxDQUFDLENBQUM7SUFDakUsSUFBTSxPQUFPLEdBQUcsQ0FBQyxXQUFXLENBQUMsR0FBRyxFQUFFLEdBQUcsS0FBSyxDQUFDLEdBQUcsT0FBTyxDQUFDO0lBRXRELEtBQUssQ0FBQyxtQkFBbUIsQ0FBQyxRQUFRLENBQUMsQ0FBQztJQUNwQyxLQUFLLENBQUMsbUJBQW1CLENBQUMsUUFBUSxDQUFDLENBQUM7SUFDcEMsS0FBSyxDQUFDLG1CQUFtQixDQUFDLGFBQWEsQ0FBQyxDQUFDO0lBQ3pDLEtBQUssQ0FBQyxhQUFhLENBQUMsT0FBTyxDQUFDLENBQUM7SUFDN0IsS0FBSyxDQUFDLE9BQU8sRUFBRSxDQUFDO0lBRWhCLE1BQU0sQ0FBQyxPQUFPLENBQUM7QUFDakIsQ0FBQyxDQUFDOzs7QUNoRkYsT0FBTyxDQUFDLEVBQUMsRUFBRSxFQUFFLGFBQWEsRUFBQyxDQUFDLENBQUM7OztBQ0E3QixPQUFPLENBQUMsRUFBQyxFQUFFLEVBQUUsYUFBYSxFQUFDLENBQUMsQ0FBQzs7Ozs7QUM0QzdCLHdCQUErQixJQUFVO0lBRXZDLE1BQU0sQ0FBQyxPQUFPLENBQUMsS0FBSyxDQUFDLElBQVcsQ0FBaUMsQ0FBQztBQUNwRSxDQUFDO0FBSEQsd0NBR0M7Ozs7O0FDOUNELDhCQUFnQztBQUVoQyxtQ0FDSSxPQUFpQixFQUFFLE9BQWlCLEVBQUUsSUFBWSxFQUNsRCxrQkFBdUI7SUFBdkIsbUNBQUEsRUFBQSx1QkFBdUI7SUFDekIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsTUFBTSxLQUFLLENBQUMsRUFDcEIsa0JBQWtCLEdBQUcsd0NBQXdDLENBQUMsQ0FBQztJQUNuRSxJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUNwQixrQkFBa0IsR0FBRyx3Q0FBd0MsQ0FBQyxDQUFDO0lBRW5FLElBQUksQ0FBQyxNQUFNLENBQ1AsSUFBSSxJQUFJLENBQUMsSUFBSSxJQUFJLEdBQUcsQ0FBQyxFQUFFLDRDQUE0QyxDQUFDLENBQUM7SUFFekUsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxFQUFFLEVBQUUsQ0FBQztRQUMzQixJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxLQUFLLElBQUksQ0FBQyxJQUFJLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxLQUFLLE9BQU8sQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUMzQyxrQkFBa0I7YUFDZCxZQUFVLE9BQU8sMEJBQXFCLE9BQU8sYUFBVSxDQUFBO1lBQ3ZELHdCQUF3QixDQUFDLENBQUM7SUFDcEMsQ0FBQztBQUNILENBQUM7QUFwQkQsOERBb0JDO0FBRUQsb0NBQ0ksT0FBaUIsRUFBRSxPQUFpQixFQUNwQyxJQUFZO0lBQ2QsSUFBSSxDQUFDLE1BQU0sQ0FBQyxPQUFPLENBQUMsTUFBTSxLQUFLLENBQUMsRUFBRSx3Q0FBd0MsQ0FBQyxDQUFDO0lBQzVFLElBQUksQ0FBQyxNQUFNLENBQUMsT0FBTyxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQUUsdUNBQXVDLENBQUMsQ0FBQztJQUUzRSxJQUFNLFdBQVcsR0FBRyxPQUFPLENBQUMsS0FBSyxFQUFFLENBQUM7SUFDcEMsV0FBVyxDQUFDLElBQUksQ0FBQyxJQUFJLE9BQU8sQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUNuQyxNQUFNLENBQUMsV0FBdUMsQ0FBQztBQUNqRCxDQUFDO0FBVEQsZ0VBU0M7Ozs7O0FDakNELDhCQUFnQztBQUVoQyw4QkFDSSxxQkFBK0MsRUFBRSxTQUFpQixFQUNsRSxLQUFhLEVBQUUsTUFBYyxFQUFFLE9BQWdCO0lBQ2pELEVBQUUsQ0FBQyxDQUFDLE9BQU8sSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDO1FBQ3BCLE9BQU8sR0FBRyxpQkFBaUIsQ0FBQyxxQkFBcUIsRUFBRSxTQUFTLEVBQUUsTUFBTSxDQUFDLENBQUM7SUFDeEUsQ0FBQztJQUNELElBQU0sU0FBUyxHQUFHLHFCQUFxQixDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzNDLElBQU0sU0FBUyxHQUFHLHFCQUFxQixDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzNDLElBQU0sVUFBVSxHQUFHLENBQUMsU0FBUyxHQUFHLFNBQVMsR0FBRyxDQUFDLEdBQUcsT0FBTyxDQUFDLEdBQUcsTUFBTSxHQUFHLENBQUMsQ0FBQztJQUN0RSxJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLEVBQ3RCLDJCQUF5QixVQUFVLHNDQUFtQztRQUNsRSxtQ0FBbUMsQ0FBQyxDQUFDO0lBRTdDLElBQU0sVUFBVSxHQUFHLENBQUMsU0FBUyxHQUFHLFNBQVMsR0FBRyxDQUFDLEdBQUcsT0FBTyxDQUFDLEdBQUcsTUFBTSxHQUFHLENBQUMsQ0FBQztJQUN0RSxJQUFJLENBQUMsTUFBTSxDQUNQLElBQUksQ0FBQyxLQUFLLENBQUMsVUFBVSxDQUFDLEVBQ3RCLDhCQUE0QixVQUFVLGtDQUErQjtRQUNqRSx1Q0FBdUMsQ0FBQyxDQUFDO0lBRWpELE1BQU0sQ0FBQyxDQUFDLFVBQVUsRUFBRSxVQUFVLEVBQUUsS0FBSyxDQUFDLENBQUM7QUFDekMsQ0FBQztBQXJCRCxvREFxQkM7QUFFRCwyQkFDSSxVQUFvQyxFQUFFLFNBQWlCLEVBQ3ZELE1BQWM7SUFDaEIsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxVQUFVLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLEdBQUcsTUFBTSxHQUFHLFNBQVMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxDQUFDO0FBQzdFLENBQUM7QUFKRCw4Q0FJQztBQUVELCtCQUNJLGdCQUEwQztJQUM1QyxNQUFNLENBQUMsQ0FBQyxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsRUFBRSxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsR0FBRyxnQkFBZ0IsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0FBQzFFLENBQUM7QUFIRCxzREFHQztBQUVELCtCQUNJLFVBQWtCLEVBQUUsV0FBbUIsRUFDdkMsS0FBYTtJQUNmLE1BQU0sQ0FBQyxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsVUFBVSxFQUFFLFdBQVcsQ0FBQyxDQUFDO0FBQ2pELENBQUM7QUFKRCxzREFJQztBQUVELDBCQUNJLEVBQW9CLEVBQUUsVUFBa0I7SUFDMUMsSUFBTSxXQUFXLEdBQUcsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEdBQUcsVUFBVSxHQUFHLENBQUMsQ0FBQztJQUNqRCxJQUFNLFdBQVcsR0FBRyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsR0FBRyxVQUFVLEdBQUcsQ0FBQyxDQUFDO0lBQ2pELE1BQU0sQ0FBQyxDQUFDLFdBQVcsRUFBRSxXQUFXLENBQUMsQ0FBQztBQUNwQyxDQUFDO0FBTEQsNENBS0M7Ozs7O0FDL0NELHdCQUNJLFVBQTRCLEVBQUUsUUFBMEI7SUFDMUQsSUFBTSxPQUFPLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBQyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM5QyxJQUFNLE9BQU8sR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzFDLEVBQUUsQ0FBQyxDQUFDLE9BQU8sS0FBSyxPQUFPLENBQUMsQ0FBQyxDQUFDO1FBQ3hCLElBQU0sTUFBTSxHQUFHLEdBQUcsR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsR0FBRyxHQUFHLENBQUM7UUFDaEUsSUFBTSxNQUFNLEdBQUcsR0FBRyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsR0FBRyxJQUFJLEdBQUcsUUFBUSxDQUFDLENBQUMsQ0FBQyxHQUFHLEdBQUcsQ0FBQztRQUM1RCxNQUFNLElBQUksS0FBSyxDQUNYLG9EQUFvRCxHQUFHLE1BQU07WUFDN0QsU0FBUyxHQUFHLE9BQU8sR0FBRyxlQUFlLEdBQUcsTUFBTSxHQUFHLFNBQVMsR0FBRyxPQUFPLENBQUMsQ0FBQztJQUM1RSxDQUFDO0FBQ0gsQ0FBQztBQVhELHdDQVdDOzs7OztBQ1hELDhCQUFnQztBQUNoQywrQ0FBaUQ7QUFDakQsMkNBQTZDO0FBRTdDLHFDQUE4RTtBQVM5RTtJQWFFLHFCQUFvQixRQUFpQjtRQUFqQixhQUFRLEdBQVIsUUFBUSxDQUFTO1FBWjdCLGtCQUFhLEdBQWdCLEVBQUUsQ0FBQztRQUdoQyxtQkFBYyxHQUFnQixFQUFFLENBQUM7UUFDakMsOEJBQXlCLEdBQWMsRUFBRSxDQUFDO1FBRTFDLGNBQVMsR0FBRyxLQUFLLENBQUM7SUFNYyxDQUFDO0lBVXpDLDJCQUFLLEdBQUwsVUFDSSxPQUV5RDtRQUg3RCxpQkFhQztRQVRDLElBQUksQ0FBQyxVQUFVLEVBQUUsQ0FBQztRQUVsQixJQUFNLE1BQU0sR0FBRyxVQUFvQixPQUFVLElBQVEsT0FBQSxLQUFJLENBQUMsSUFBSSxDQUFDLE9BQU8sQ0FBQyxFQUFsQixDQUFrQixDQUFDO1FBQ3hFLElBQU0sT0FBTyxHQUFHLFVBQW9CLE9BQVUsSUFBUSxPQUFBLEtBQUksQ0FBQyxLQUFLLENBQUMsT0FBTyxDQUFDLEVBQW5CLENBQW1CLENBQUM7UUFDMUUsSUFBTSxNQUFNLEdBQUcsT0FBTyxDQUFDLE1BQU0sRUFBRSxPQUFPLENBQUMsQ0FBQztRQUV4QyxJQUFJLENBQUMsUUFBUSxDQUFDLE1BQU0sQ0FBQyxDQUFDO1FBRXRCLE1BQU0sQ0FBQyxNQUFNLENBQUM7SUFDaEIsQ0FBQztJQU9ELHFDQUFlLEdBQWY7UUFDRSxJQUFJLENBQUMsU0FBUyxHQUFHLElBQUksQ0FBQztRQUN0QixPQUFPLENBQUMsSUFBSSxDQUNSLDJEQUEyRDtZQUMzRCw2Q0FBNkM7WUFDN0MseUNBQXlDLENBQUMsQ0FBQztJQUNqRCxDQUFDO0lBTUQsZ0NBQVUsR0FBVjtRQUNFLElBQU0sUUFBUSxHQUFjLEVBQUUsQ0FBQztRQUMvQixJQUFJLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQztRQUNsQyxJQUFJLENBQUMsV0FBVyxHQUFHLFFBQVEsQ0FBQztRQUU1QixJQUFNLGlCQUFpQixHQUFjLEVBQUUsQ0FBQztRQUN4QyxJQUFJLENBQUMsY0FBYyxDQUFDLElBQUksQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDO1FBQzVDLElBQUksQ0FBQyx5QkFBeUIsR0FBRyxpQkFBaUIsQ0FBQztJQUNyRCxDQUFDO0lBTUQsOEJBQVEsR0FBUixVQUFTLE1BQW1CO1FBQTVCLGlCQXFDQztRQXBDQyxJQUFJLFlBQVksR0FBRyxJQUFJLENBQUMseUJBQXlCLENBQUM7UUFDbEQsRUFBRSxDQUFDLENBQUMsTUFBTSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDbkIsWUFBWSxHQUFHLFlBQVksQ0FBQyxNQUFNLENBQUMsTUFBNkIsQ0FBQyxDQUFDO1FBQ3BFLENBQUM7UUFFRCxHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLElBQUksQ0FBQyxXQUFXLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7WUFDakQsSUFBTSxPQUFPLEdBQUcsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUNwQyxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsbUJBQW1CLENBQUMsT0FBTyxFQUFFLFlBQVksQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDcEQsUUFBUSxDQUFDO1lBQ1gsQ0FBQztZQUNELE9BQU8sQ0FBQyxPQUFPLEVBQUUsQ0FBQztRQUNwQixDQUFDO1FBR0QsSUFBSSxDQUFDLGFBQWEsQ0FBQyxHQUFHLEVBQUUsQ0FBQztRQUN6QixJQUFJLENBQUMsV0FBVyxHQUFHLElBQUksQ0FBQyxhQUFhLENBQUMsTUFBTSxLQUFLLENBQUM7WUFDOUMsSUFBSztZQUNMLElBQUksQ0FBQyxhQUFhLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLENBQUM7UUFHdEQsRUFBRSxDQUFDLENBQUMsTUFBTSxZQUFZLGlCQUFPO1lBQ3pCLENBQUMsSUFBSSxDQUFDLG1CQUFtQixDQUFDLE1BQU0sRUFBRSxJQUFJLENBQUMseUJBQXlCLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDdEUsSUFBSSxDQUFDLEtBQUssQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUNyQixDQUFDO1FBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxPQUFPLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ2pDLE1BQU0sQ0FBQyxPQUFPLENBQUMsVUFBQSxDQUFDO2dCQUNkLEVBQUUsQ0FBQyxDQUFDLENBQUMsWUFBWSxpQkFBTztvQkFDcEIsQ0FBQyxLQUFJLENBQUMsbUJBQW1CLENBQUMsQ0FBQyxFQUFFLEtBQUksQ0FBQyx5QkFBeUIsQ0FBQyxDQUFDLENBQUMsQ0FBQztvQkFDakUsS0FBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDaEIsQ0FBQztZQUNILENBQUMsQ0FBQyxDQUFDO1FBQ0wsQ0FBQztRQUVELElBQUksQ0FBQyxjQUFjLENBQUMsR0FBRyxFQUFFLENBQUM7UUFDMUIsSUFBSSxDQUFDLHlCQUF5QixHQUFHLElBQUksQ0FBQyxjQUFjLENBQUMsTUFBTSxLQUFLLENBQUM7WUFDN0QsSUFBSztZQUNMLElBQUksQ0FBQyxjQUFjLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxNQUFNLEdBQUcsQ0FBQyxDQUFDLENBQUM7SUFDMUQsQ0FBQztJQUVPLHlDQUFtQixHQUEzQixVQUE0QixPQUFnQixFQUFFLFdBQXNCO1FBQ2xFLEdBQUcsQ0FBQyxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsRUFBRSxDQUFDLEdBQUcsV0FBVyxDQUFDLE1BQU0sRUFBRSxDQUFDLEVBQUUsRUFBRSxDQUFDO1lBQzVDLEVBQUUsQ0FBQyxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxPQUFPLEVBQUUsS0FBSyxPQUFPLENBQUMsT0FBTyxFQUFFLENBQUMsQ0FBQyxDQUFDO2dCQUNuRCxNQUFNLENBQUMsSUFBSSxDQUFDO1lBQ2QsQ0FBQztRQUNILENBQUM7UUFDRCxNQUFNLENBQUMsS0FBSyxDQUFDO0lBQ2YsQ0FBQztJQU1ELDBCQUFJLEdBQUosVUFBd0IsTUFBUztRQUMvQixFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsV0FBVyxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDN0IsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFFBQVEsQ0FBQyxDQUFDLENBQUM7Z0JBQ2xCLE1BQU0sSUFBSSxLQUFLLENBQ1gsK0NBQStDO29CQUMvQyxzQ0FBc0M7b0JBQ3RDLHdEQUF3RDtvQkFDeEQsUUFBUSxDQUFDLENBQUM7WUFDaEIsQ0FBQztZQUNELE1BQU0sQ0FBQyxNQUFNLENBQUM7UUFDaEIsQ0FBQztRQUNELElBQUksQ0FBQyx5QkFBeUIsQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUM7UUFDNUMsTUFBTSxDQUFDLE1BQU0sQ0FBQztJQUNoQixDQUFDO0lBRU8saUNBQVcsR0FBbkIsVUFBb0IsR0FBWTtRQUM5QixJQUFNLElBQUksR0FBRyxHQUFHLENBQUMsU0FBUyxFQUFFLENBQUM7UUFDN0IsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxJQUFJLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7WUFDckMsRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztnQkFDbkIsTUFBTSxLQUFLLENBQUMsb0RBQW9ELENBQUMsQ0FBQztZQUNwRSxDQUFDO1FBQ0gsQ0FBQztJQUNILENBQUM7SUFPRCwyQkFBSyxHQUFMLFVBQXlCLE1BQVM7UUFDaEMsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUM7WUFDbkIsSUFBSSxDQUFDLFdBQVcsQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUMzQixDQUFDO1FBQ0QsRUFBRSxDQUFDLENBQUMsSUFBSSxDQUFDLFdBQVcsSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDO1lBQzdCLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxDQUFDO2dCQUNsQixNQUFNLElBQUksS0FBSyxDQUNYLCtDQUErQztvQkFDL0Msc0NBQXNDO29CQUN0Qyx3REFBd0Q7b0JBQ3hELFFBQVEsQ0FBQyxDQUFDO1lBQ2hCLENBQUM7WUFDRCxNQUFNLENBQUMsTUFBTSxDQUFDO1FBQ2hCLENBQUM7UUFDRCxJQUFJLENBQUMsV0FBVyxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUM5QixNQUFNLENBQUMsTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFhRCw0QkFBTSxHQUFOLFVBQ0ksQ0FBVSxFQUFFLENBQVUsRUFBRSxZQUF3QyxFQUNoRSxZQUF3QztRQURoQiw2QkFBQSxFQUFBLGVBQWUsaUJBQWlCLENBQUMsT0FBTztRQUNoRSw2QkFBQSxFQUFBLGVBQWUsaUJBQWlCLENBQUMsT0FBTztRQUMxQyxJQUFNLFdBQVcsR0FDYixDQUFDLFlBQVksS0FBSyxpQkFBaUIsQ0FBQyxPQUFPLENBQUMsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUM7UUFDM0UsSUFBTSxXQUFXLEdBQ2IsQ0FBQyxZQUFZLEtBQUssaUJBQWlCLENBQUMsT0FBTyxDQUFDLEdBQUcsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO1FBRTNFLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLElBQUksQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQzVCLHVEQUFxRCxDQUFDLENBQUMsSUFBTTthQUN6RCxTQUFPLENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFFMUIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxXQUFXLEtBQUssV0FBVyxFQUMzQixvQ0FBa0MsV0FBVyxZQUFTO2FBQy9DLFdBQVcsa0NBQTZCLENBQUMsQ0FBQyxLQUFLLFVBQU8sQ0FBQTthQUN0RCxDQUFDLENBQUMsS0FBSywwQkFBcUIsaUJBQWlCLENBQUMsWUFBWSxDQUFHLENBQUE7YUFDaEUsVUFBUSxpQkFBaUIsQ0FBQyxZQUFZLENBQUMsaUJBQWMsQ0FBQSxDQUFDLENBQUM7UUFFL0QsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLFlBQVksRUFBRSxZQUFZLENBQUMsQ0FBQyxDQUFDO0lBQzNFLENBQUM7SUFVRCx1Q0FBaUIsR0FBakIsVUFBa0IsQ0FBVSxFQUFFLE1BQWU7UUFDM0MsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixrRUFBa0U7YUFDOUQsVUFBUSxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzNCLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2pCLG1FQUFtRTthQUMvRCxVQUFRLE1BQU0sQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDaEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQzFCLDZEQUEyRCxDQUFDLENBQUMsSUFBSSxPQUFJO1lBQ2pFLDZEQUE2RDthQUM3RCxVQUFRLE1BQU0sQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFFaEMsTUFBTSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLElBQUksQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDO0lBQ3ZELENBQUM7SUFPRCx1Q0FBaUIsR0FBakIsVUFBa0IsTUFBZSxFQUFFLENBQVU7UUFDM0MsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixnRUFBZ0U7YUFDNUQsVUFBUSxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzNCLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2pCLG9FQUFvRTthQUNoRSxVQUFRLE1BQU0sQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDaEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLE1BQU0sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQzFCLDREQUEwRCxDQUFDLENBQUMsSUFBSSxNQUFHO1lBQy9ELDZEQUE2RDthQUM3RCxXQUFTLE1BQU0sQ0FBQyxLQUFLLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFFbEMsTUFBTSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsTUFBTSxFQUFFLENBQUMsQ0FBQyxJQUFJLENBQUMsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLElBQUksRUFBRSxDQUFDO0lBQ3ZELENBQUM7SUFPRCxnQ0FBVSxHQUFWLFVBQVcsRUFBVyxFQUFFLEVBQVc7UUFDakMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDOUIsNERBQTREO2FBQ3JELEVBQUUsQ0FBQyxJQUFJLGFBQVEsRUFBRSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUN0QyxJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxJQUFJLEtBQUssRUFBRSxDQUFDLElBQUksRUFDbkIsMENBQXdDLEVBQUUsQ0FBQyxJQUFJLFlBQVM7YUFDakQsRUFBRSxDQUFDLElBQUksa0JBQWUsQ0FBQSxDQUFDLENBQUM7UUFDbkMsTUFBTSxDQUFDLElBQUksQ0FBQyxNQUFNLENBQUMsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLElBQUksQ0FBQyxFQUFFLEVBQUUsQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDLElBQUksRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDLFFBQVEsRUFBRSxDQUFDO0lBQzFFLENBQUM7SUFPRCxrQ0FBWSxHQUFaLFVBQWEsRUFBVyxFQUFFLEVBQVc7UUFDbkMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxFQUFFLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDOUIsOERBQThEO2FBQ3ZELEVBQUUsQ0FBQyxJQUFJLGFBQVEsRUFBRSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUV0QyxNQUFNLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxJQUFJLEVBQUUsQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsRUFBRSxDQUFDLElBQUksQ0FBQyxDQUFDLENBQUM7SUFDL0QsQ0FBQztJQVVELDJCQUFLLEdBQUwsVUFBeUIsT0FBVTtRQUNqQyxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDakQsQ0FBQztJQU1ELDZCQUFPLEdBQVAsVUFDSSxPQUFXLEVBQUUsUUFBa0I7UUFDakMsT0FBTyxDQUFDLElBQUksQ0FDUixzREFBc0Q7WUFDdEQsZ0NBQWdDLENBQUMsQ0FBQztRQUN0QyxNQUFNLENBQUMsT0FBTyxDQUFDLE9BQU8sQ0FBQyxRQUFRLENBQUMsQ0FBQztJQUNuQyxDQUFDO0lBVUQsNkJBQU8sR0FBUCxVQUFRLEtBQWMsRUFBRSxLQUF1QixFQUFFLElBQXNCO1FBRXJFLElBQUksQ0FBQyxNQUFNLENBQ1AsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLElBQUksQ0FBQyxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQztZQUNoQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEdBQUcsSUFBSSxDQUFDLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQ3hDLGdEQUE4QyxLQUFLLGVBQVk7YUFDeEQsSUFBSSx1Q0FBa0MsS0FBSyxDQUFDLEtBQUssTUFBRyxDQUFBLENBQUMsQ0FBQztRQUNqRSxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsZUFBZSxDQUFDLEtBQUssRUFBRSxLQUFLLEVBQUUsSUFBSSxDQUFDLENBQUMsQ0FBQztJQUM5RCxDQUFDO0lBZUQsNEJBQU0sR0FBTixVQUNJLE1BQWUsRUFBRSxXQUE2QixFQUM5QyxVQUE0QixFQUFFLElBQWEsRUFBRSxTQUEyQixFQUN4RSxRQUEwQjtRQUM1QixJQUFJLENBQUMsTUFBTSxDQUNQLFdBQVcsQ0FBQyxDQUFDLENBQUMsR0FBRyxVQUFVLENBQUMsQ0FBQyxDQUFDLElBQUksTUFBTSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7WUFDN0MsV0FBVyxDQUFDLENBQUMsQ0FBQyxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUMsSUFBSSxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUNyRCxzREFBb0QsV0FBVyxNQUFHO2FBQzlELHFCQUFtQixVQUFVLG1DQUFnQyxDQUFBO2FBQzdELGNBQVksTUFBTSxDQUFDLEtBQUssTUFBRyxDQUFBLENBQUMsQ0FBQztRQUNyQyxJQUFJLENBQUMsTUFBTSxDQUNQLFNBQVMsQ0FBQyxDQUFDLENBQUMsR0FBRyxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksSUFBSSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUM7WUFDdkMsU0FBUyxDQUFDLENBQUMsQ0FBQyxHQUFHLFFBQVEsQ0FBQyxDQUFDLENBQUMsSUFBSSxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUMvQyxvREFBa0QsU0FBUyxNQUFHO2FBQzFELHFCQUFtQixRQUFRLG9DQUFpQyxDQUFBO2FBQzVELFdBQVMsSUFBSSxDQUFDLEtBQUssTUFBRyxDQUFBLENBQUMsQ0FBQztRQUNoQyxXQUFXLENBQUMsY0FBYyxDQUFDLFVBQVUsRUFBRSxRQUFRLENBQUMsQ0FBQztRQUVqRCxNQUFNLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FDdEIsTUFBTSxFQUFFLFdBQVcsRUFBRSxVQUFVLEVBQUUsSUFBSSxFQUFFLFNBQVMsRUFBRSxRQUFRLENBQUMsQ0FBQztJQUNsRSxDQUFDO0lBb0NELDhCQUFRLEdBQVIsVUFBUyxRQUFpQixFQUFFLFFBQWlCLEVBQUUsSUFBWTtRQUN6RCxhQUFhLENBQUMseUJBQXlCLENBQ25DLFFBQVEsQ0FBQyxLQUFLLEVBQUUsUUFBUSxDQUFDLEtBQUssRUFBRSxJQUFJLEVBQUUscUJBQXFCLENBQUMsQ0FBQztRQUNqRSxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsZ0JBQWdCLENBQUMsUUFBUSxFQUFFLFFBQVEsRUFBRSxJQUFJLENBQUMsQ0FBQyxDQUFDO0lBQ3JFLENBQUM7SUFZRCwrQkFBUyxHQUFULFVBQVUsT0FBZ0I7UUFDeEIsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDckQsQ0FBQztJQU9ELHlCQUFHLEdBQUgsVUFBSSxPQUFnQjtRQUNsQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQU9ELDRCQUFNLEdBQU4sVUFBTyxPQUFnQjtRQUNyQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDbEQsQ0FBQztJQU9ELDRCQUFNLEdBQU4sVUFBTyxPQUFnQjtRQUNyQixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLE9BQU8sQ0FBQyxDQUFDLENBQUM7SUFDbEQsQ0FBQztJQVFELGtDQUFZLEdBQVosVUFBYSxFQUFXLEVBQUUsRUFBVztRQUNuQyxJQUFJLENBQUMsaUJBQWlCLENBQUMsRUFBRSxDQUFDLEtBQUssRUFBRSxFQUFFLENBQUMsS0FBSyxFQUFFLHlCQUF5QixDQUFDLENBQUM7UUFDdEUsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLG9CQUFvQixDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDO0lBQ3ZELENBQUM7SUFRRCwwQkFBSSxHQUFKLFVBQUssT0FBZ0IsRUFBRSxDQUFTO1FBQzlCLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxJQUFJLE9BQU8sQ0FBQyxJQUFJLEVBQ2pCLDZCQUEyQixDQUFDLHVDQUFvQzthQUM1RCx3QkFBc0IsT0FBTyxDQUFDLEtBQUssTUFBRyxDQUFBLENBQUMsQ0FBQztRQUNoRCxJQUFNLE1BQU0sR0FBRyxJQUFJLENBQUMsWUFBWSxDQUFDLE9BQU8sRUFBRSxDQUFDLENBQUMsQ0FBQztRQUM3QyxJQUFJLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxNQUFNLENBQUMsQ0FBQztRQUMxQixJQUFJLENBQUMsS0FBSyxDQUFDLE1BQU0sQ0FBQyxPQUFPLENBQUMsQ0FBQztRQUMzQixNQUFNLENBQUMsTUFBTSxDQUFDO0lBQ2hCLENBQUM7SUFRRCx5QkFBRyxHQUFILFVBQUksT0FBZ0I7UUFDbEIsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDO0lBQy9DLENBQUM7SUFPRCx5QkFBRyxHQUFILFVBQUksT0FBZ0I7UUFDbEIsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxPQUFPLENBQUMsQ0FBQyxDQUFDO0lBQy9DLENBQUM7SUFPRCw2QkFBTyxHQUFQLFVBQVEsQ0FBVTtRQUFsQixpQkFRQztRQVBDLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDO1lBR2hCLElBQU0sR0FBRyxHQUFHLEtBQUksQ0FBQyxTQUFTLENBQUMsQ0FBQyxDQUFDLENBQUM7WUFDOUIsSUFBTSxTQUFTLEdBQUcsS0FBSSxDQUFDLGdCQUFnQixDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsQ0FBQztZQUNoRCxNQUFNLENBQUMsS0FBSSxDQUFDLEdBQUcsQ0FBQyxTQUFTLENBQUMsQ0FBQztRQUM3QixDQUFDLENBQUMsQ0FBQztJQUNMLENBQUM7SUFXRCwrQkFBUyxHQUFULFVBQTZCLENBQUksRUFBRSxNQUFnQjtRQUNqRCxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssTUFBTSxDQUFDLE1BQU0sRUFDeEIsK0NBQTZDLENBQUMsQ0FBQyxLQUFLLE1BQUc7YUFDbkQscUNBQW1DLE1BQU0sTUFBRyxDQUFBLENBQUMsQ0FBQztRQUN0RCxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsaUJBQWlCLENBQUMsQ0FBQyxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUM7SUFDdkQsQ0FBQztJQVNELHFDQUFlLEdBQWYsVUFBbUMsQ0FBUyxFQUFFLENBQUk7UUFDaEQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixtRUFBbUU7YUFDL0QsVUFBUSxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzNCLE1BQU0sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQU0sQ0FBQztJQUM3QixDQUFDO0lBT0Qsc0NBQWdCLEdBQWhCLFVBQW9DLENBQVMsRUFBRSxDQUFJO1FBQ2pELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osb0VBQW9FO2FBQ2hFLFVBQVEsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMzQixNQUFNLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFNLENBQUM7SUFDN0IsQ0FBQztJQU9ELHNDQUFnQixHQUFoQixVQUFvQyxDQUFJLEVBQUUsQ0FBUztRQUNqRCxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLGlFQUFpRTthQUM3RCxjQUFZLENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDL0IsTUFBTSxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBTSxDQUFDO0lBQzdCLENBQUM7SUFNRCx5QkFBRyxHQUFILFVBQXVCLENBQUk7UUFDekIsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLFdBQVcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ3pDLENBQUM7SUFVRCx5QkFBRyxHQUFILFVBQUksQ0FBVSxFQUFFLENBQVU7UUFDeEIsSUFBSSxDQUFDLDRCQUE0QixDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ3BELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDNUMsQ0FBQztJQVVELCtCQUFTLEdBQVQsVUFBNkIsQ0FBSSxFQUFFLENBQUk7UUFDckMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSxzQkFBc0IsQ0FBQyxDQUFDO1FBQ2pFLE1BQU0sQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQU0sQ0FBQztJQUM3QixDQUFDO0lBU0QseUJBQUcsR0FBSCxVQUFJLENBQVUsRUFBRSxDQUFVO1FBQ3hCLElBQUksQ0FBQyw0QkFBNEIsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQztRQUNwRCxNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsV0FBVyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQzVDLENBQUM7SUFVRCwrQkFBUyxHQUFULFVBQTZCLENBQUksRUFBRSxDQUFJO1FBQ3JDLElBQUksQ0FBQyxpQkFBaUIsQ0FBQyxDQUFDLENBQUMsS0FBSyxFQUFFLENBQUMsQ0FBQyxLQUFLLEVBQUUsc0JBQXNCLENBQUMsQ0FBQztRQUNqRSxNQUFNLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFNLENBQUM7SUFDN0IsQ0FBQztJQVNELDhCQUFRLEdBQVIsVUFBUyxDQUFVLEVBQUUsQ0FBVTtRQUM3QixJQUFJLENBQUMsNEJBQTRCLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUM7UUFDcEQsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGdCQUFnQixDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBQ2pELENBQUM7SUFNRCxvQ0FBYyxHQUFkLFVBQWtDLENBQUksRUFBRSxDQUFJO1FBQzFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsY0FBYyxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsQ0FBQztJQUNuQyxDQUFDO0lBU0Qsb0NBQWMsR0FBZCxVQUFrQyxDQUFJLEVBQUUsQ0FBSTtRQUMxQyxJQUFJLENBQUMsaUJBQWlCLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsS0FBSyxFQUFFLDJCQUEyQixDQUFDLENBQUM7UUFDdEUsTUFBTSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBTSxDQUFDO0lBQ2xDLENBQUM7SUFTRCw0QkFBTSxHQUFOLFVBQU8sQ0FBVSxFQUFFLENBQVU7UUFDM0IsSUFBSSxDQUFDLDRCQUE0QixDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDO1FBQ3BELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxjQUFjLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDL0MsQ0FBQztJQVVELGtDQUFZLEdBQVosVUFBZ0MsQ0FBSSxFQUFFLENBQUk7UUFDeEMsSUFBSSxDQUFDLGlCQUFpQixDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLEtBQUssRUFBRSx5QkFBeUIsQ0FBQyxDQUFDO1FBQ3BFLE1BQU0sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQU0sQ0FBQztJQUNoQyxDQUFDO0lBUUQsMENBQW9CLEdBQXBCLFVBQXdDLENBQVMsRUFBRSxDQUFJO1FBQ3JELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osb0VBQW9FO2FBQ2hFLHlCQUF1QixDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzFDLE1BQU0sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQU0sQ0FBQztJQUNoQyxDQUFDO0lBUUQsMENBQW9CLEdBQXBCLFVBQXdDLENBQUksRUFBRSxDQUFTO1FBQ3JELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osaUVBQWlFO2FBQzdELDZCQUEyQixDQUFDLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzlDLE1BQU0sQ0FBQyxJQUFJLENBQUMsTUFBTSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQU0sQ0FBQztJQUNoQyxDQUFDO0lBTUQseUJBQUcsR0FBSCxVQUF1QixPQUFVO1FBQy9CLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUMvQyxDQUFDO0lBT0QseUJBQUcsR0FBSCxVQUF1QixPQUFVO1FBQy9CLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUMvQyxDQUFDO0lBT0QsMEJBQUksR0FBSixVQUF3QixPQUFVO1FBQ2hDLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNoRCxDQUFDO0lBT0QsNkJBQU8sR0FBUCxVQUEyQixPQUFVO1FBQ25DLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNuRCxDQUFDO0lBT0QsMEJBQUksR0FBSixVQUF3QixPQUFVO1FBQ2hDLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNoRCxDQUFDO0lBT0QseUJBQUcsR0FBSCxVQUF1QixPQUFVO1FBQy9CLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxXQUFXLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUMvQyxDQUFDO0lBUUQsMEJBQUksR0FBSixVQUF3QixPQUFVO1FBQ2hDLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxZQUFZLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUNoRCxDQUFDO0lBVUQsb0NBQWMsR0FBZCxVQUFrQyxFQUFVLEVBQUUsQ0FBSSxFQUFFLEVBQVUsRUFBRSxDQUFJO1FBQ2xFLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2IsK0RBQStEO2FBQzNELFdBQVMsRUFBRSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUM3QixJQUFJLENBQUMsTUFBTSxDQUNQLEVBQUUsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNiLGtFQUFrRTthQUM5RCxxQkFBbUIsRUFBRSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUN2QyxJQUFJLENBQUMsaUJBQWlCLENBQUMsQ0FBQyxDQUFDLEtBQUssRUFBRSxDQUFDLENBQUMsS0FBSyxFQUFFLDJCQUEyQixDQUFDLENBQUM7UUFFdEUsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLHNCQUFzQixDQUFDLEVBQUUsRUFBRSxDQUFDLEVBQUUsRUFBRSxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUM7SUFDL0QsQ0FBQztJQVVELHNDQUFnQixHQUFoQixVQUFvQyxDQUFTLEVBQUUsQ0FBSTtRQUNqRCxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLG9FQUFvRTthQUNoRSxjQUFZLENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDL0IsTUFBTSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBTSxDQUFDO0lBQ2xDLENBQUM7SUFLRCw2Q0FBdUIsR0FBdkIsVUFBd0IsQ0FBVSxFQUFFLENBQVU7UUFDNUMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWiwyREFBMkQ7YUFDdkQsMEJBQXdCLENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDM0MsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWiw0REFBNEQ7YUFDeEQsMEJBQXdCLENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDM0MsTUFBTSxDQUFDLElBQUksQ0FBQyxRQUFRLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBWSxDQUFDO0lBQ3hDLENBQUM7SUFnQkQsNEJBQU0sR0FBTixVQUNJLENBQVUsRUFBRSxPQUFnQixFQUFFLE1BQW9CLEVBQUUsTUFBYyxFQUNsRSxPQUFlO1FBQ2pCLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1oscURBQW1ELENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQyxDQUFDO1FBQ2xFLElBQUksQ0FBQyxNQUFNLENBQ1AsT0FBTyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2xCLHdEQUF3RDthQUNqRCxPQUFPLENBQUMsSUFBSSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzVCLEVBQUUsQ0FBQyxDQUFDLE1BQU0sSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDO1lBQ25CLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2pCLHVEQUF1RDtpQkFDaEQsTUFBTSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUM3QixDQUFDO1FBRUQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQy9CLHNDQUFvQyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxtQkFBZ0I7YUFDMUQsNkJBQTJCLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFHeEQsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGNBQWMsQ0FBQyxDQUFDLEVBQUUsT0FBTyxFQUFFLE1BQU0sRUFBRSxNQUFNLEVBQUUsT0FBTyxDQUFDLENBQUMsQ0FBQztJQUM5RSxDQUFDO0lBY0Qsb0NBQWMsR0FBZCxVQUNJLENBQVUsRUFBRSxFQUFXLEVBQUUsT0FBZ0IsRUFBRSxNQUFjLEVBQ3pELEdBQVc7UUFDYixJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLDJEQUEyRDthQUNwRCxDQUFDLENBQUMsS0FBSyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQ3ZCLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2IsNERBQTREO2FBQ3JELEVBQUUsQ0FBQyxLQUFLLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDeEIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDbEIsaUVBQWlFO2FBQzFELE9BQU8sQ0FBQyxLQUFLLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDN0IsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQy9CLHlDQUF1QyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxZQUFTO2FBQ3RELG9DQUFrQyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQy9ELElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsS0FBSyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUNoQywyQ0FBeUMsRUFBRSxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsWUFBUzthQUN6RCxxQ0FBbUMsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsT0FBSSxDQUFBLENBQUMsQ0FBQztRQUVqRSxJQUFNLGNBQWMsR0FDaEIsSUFBSSxDQUFDLHNCQUFzQixDQUFDLENBQUMsRUFBRSxFQUFFLEVBQUUsT0FBTyxFQUFFLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQztRQUU3RCxJQUFJLENBQUMsS0FBSyxDQUFDLGNBQWMsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUM5QixJQUFJLENBQUMsS0FBSyxDQUFDLGNBQWMsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUM5QixJQUFJLENBQUMsS0FBSyxDQUFDLGNBQWMsQ0FBQyxFQUFFLENBQUMsQ0FBQztRQUU5QixNQUFNLENBQUMsY0FBYyxDQUFDO0lBQ3hCLENBQUM7SUFnQkQscUNBQWUsR0FBZixVQUNJLENBQVUsRUFBRSxPQUFnQixFQUFFLE1BQW9CLEVBQUUsTUFBYyxFQUNsRSxHQUFXO1FBQ2IsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWiwyREFBMkQ7YUFDcEQsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUN0QixJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNsQiw0REFBNEQ7YUFDeEQsVUFBUSxPQUFPLENBQUMsSUFBTSxDQUFBLENBQUMsQ0FBQztRQUNoQyxFQUFFLENBQUMsQ0FBQyxNQUFNLElBQUksSUFBSSxDQUFDLENBQUMsQ0FBQztZQUNuQixJQUFJLENBQUMsTUFBTSxDQUNQLE1BQU0sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNqQix1RkFDWSxNQUFNLENBQUMsSUFBSSxNQUFHLENBQUMsQ0FBQztRQUNsQyxDQUFDO1FBQ0QsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQy9CLCtDQUE2QyxDQUFDLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxZQUFTO2FBQzVELG1DQUFpQyxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBRTlELE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUNiLElBQUksQ0FBQyx1QkFBdUIsQ0FBQyxDQUFDLEVBQUUsT0FBTyxFQUFFLE1BQU0sRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUNyRSxDQUFDO0lBYUQsNkJBQU8sR0FBUCxVQUFRLENBQVUsRUFBRSxLQUFhLEVBQUUsTUFBYyxFQUFFLEdBQVc7UUFDNUQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixrREFBa0QsR0FBRyxDQUFDLENBQUMsSUFBSSxHQUFHLEdBQUcsQ0FBQyxDQUFDO1FBQ3ZFLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLEtBQUssRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUNqRSxDQUFDO0lBYUQscUNBQWUsR0FBZixVQUNJLEVBQVcsRUFBRSxDQUFVLEVBQUUsS0FBYSxFQUFFLE1BQWMsRUFDdEQsR0FBVztRQUNiLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2IsMkRBQTJEO2FBQ3BELEVBQUUsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDdkIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWiwwREFBMEQ7YUFDbkQsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUV0QixNQUFNLENBQUMsSUFBSSxDQUFDLEtBQUssQ0FBQyxJQUFJLENBQUMsdUJBQXVCLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRSxLQUFLLEVBQUUsTUFBTSxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUM7SUFDN0UsQ0FBQztJQWFELDZCQUFPLEdBQVAsVUFBUSxDQUFVLEVBQUUsS0FBYSxFQUFFLE1BQWMsRUFBRSxHQUFXO1FBQzVELElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1oscURBQW1ELENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQyxDQUFDO1FBQ2xFLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUFDLElBQUksQ0FBQyxlQUFlLENBQUMsQ0FBQyxFQUFFLEtBQUssRUFBRSxNQUFNLEVBQUUsR0FBRyxDQUFDLENBQUMsQ0FBQztJQUNqRSxDQUFDO0lBWUQsNkJBQU8sR0FBUCxVQUFRLENBQVUsRUFBRSxLQUFhLEVBQUUsTUFBYyxFQUFFLEdBQVc7UUFDNUQsSUFBSSxDQUFDLE1BQU0sQ0FDUCxDQUFDLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDWixxREFBbUQsQ0FBQyxDQUFDLElBQUksTUFBRyxDQUFDLENBQUM7UUFDbEUsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLGVBQWUsQ0FBQyxDQUFDLEVBQUUsS0FBSyxFQUFFLE1BQU0sRUFBRSxHQUFHLENBQUMsQ0FBQyxDQUFDO0lBQ2pFLENBQUM7SUFjRCxzQ0FBZ0IsR0FBaEIsVUFDSSxDQUFVLEVBQUUsVUFBNEIsRUFBRSxZQUFvQjtRQUFwQiw2QkFBQSxFQUFBLG9CQUFvQjtRQUNoRSxJQUFJLENBQUMsTUFBTSxDQUNQLENBQUMsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUNaLDhEQUE0RCxDQUFDLENBQUMsSUFBSSxNQUFHLENBQUMsQ0FBQztRQUMzRSxJQUFJLENBQUMsTUFBTSxDQUNQLFVBQVUsQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUN2Qiw4REFBOEQ7YUFDdkQsVUFBVSxNQUFHLENBQUEsQ0FBQyxDQUFDO1FBQzFCLE1BQU0sQ0FBQyxJQUFJLENBQUMsS0FBSyxDQUNiLElBQUksQ0FBQyx3QkFBd0IsQ0FBQyxDQUFDLEVBQUUsVUFBVSxFQUFFLFlBQVksQ0FBQyxDQUFDLENBQUM7SUFDbEUsQ0FBQztJQWdCRCwwQ0FBb0IsR0FBcEIsVUFDSSxDQUFVLEVBQUUsSUFBcUIsRUFBRSxRQUF5QixFQUM1RCxlQUFzQixFQUFFLEtBQXVCLEVBQy9DLE1BQXdCO1FBRHhCLGdDQUFBLEVBQUEsc0JBQXNCO1FBRXhCLElBQUksQ0FBQyxNQUFNLENBQ1AsQ0FBQyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ1osK0RBQStEO2FBQ3hELENBQUMsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDdEIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxJQUFJLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDbEMsbUVBQW1FO2FBQy9ELGNBQVksSUFBSSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUNsQyxJQUFJLENBQUMsTUFBTSxDQUNQLFFBQVEsQ0FBQyxJQUFJLEtBQUssQ0FBQyxJQUFJLFFBQVEsQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUMxQyxtRUFBbUU7YUFDL0Qsa0JBQWdCLFFBQVEsQ0FBQyxJQUFJLE1BQUcsQ0FBQSxDQUFDLENBQUM7UUFDMUMsRUFBRSxDQUFDLENBQUMsS0FBSyxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDbEIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxLQUFLLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDcEMsZ0VBQWdFO2lCQUM1RCxrQkFBZ0IsS0FBTSxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMxQyxDQUFDO1FBQ0QsRUFBRSxDQUFDLENBQUMsTUFBTSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUM7WUFDbkIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDdEMsaUVBQWlFO2lCQUM3RCxrQkFBZ0IsTUFBTyxDQUFDLElBQUksTUFBRyxDQUFBLENBQUMsQ0FBQztRQUMzQyxDQUFDO1FBRUQsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsSUFBSSxDQUFDLDRCQUE0QixDQUMvQyxDQUFDLEVBQUUsSUFBSSxFQUFFLFFBQVEsRUFBRSxlQUFlLEVBQUUsS0FBSyxFQUFFLE1BQU0sQ0FBQyxDQUFDLENBQUM7SUFDMUQsQ0FBQztJQXFCRCxrQ0FBWSxHQUFaLFVBQ0ksU0FBcUIsRUFBRSxJQUFhLEVBQUUsQ0FBWSxFQUNsRCxDQUFZO1FBQ2QsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsRUFDbkIsdURBQXFELElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLE9BQUk7WUFDbEUsNENBQTRDLENBQUMsQ0FBQztRQUN0RCxJQUFNLEdBQUcsR0FBRyxJQUFJLENBQUMsS0FBSyxDQUFDO1lBQ3JCLElBQUksS0FBSyxHQUFHLElBQUksQ0FBQztZQUNqQixJQUFNLFNBQVMsR0FBRyxFQUFFLENBQUM7WUFDckIsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLEdBQUcsQ0FBQyxFQUFFLENBQUMsR0FBRyxTQUFTLENBQUMsTUFBTSxFQUFFLENBQUMsRUFBRSxFQUFFLENBQUM7Z0JBQzFDLElBQU0sTUFBTSxHQUFHLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUMvQyxTQUFTLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUMxQixTQUFTLENBQUMsSUFBSSxDQUFDLE1BQU0sQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO2dCQUMxQixLQUFLLEdBQUcsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3BCLENBQUM7WUFFRCxNQUFNLENBQUMsU0FBUyxDQUFDO1FBQ25CLENBQUMsQ0FBQyxDQUFDO1FBQ0gsSUFBTSxJQUFJLEdBQWMsRUFBRSxDQUFDO1FBQzNCLElBQU0sSUFBSSxHQUFjLEVBQUUsQ0FBQztRQUMzQixHQUFHLENBQUMsQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLEdBQUcsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDO1lBQ3ZDLElBQUksQ0FBQyxJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBWSxDQUFDLENBQUM7WUFDN0IsSUFBSSxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBWSxDQUFDLENBQUM7UUFDbkMsQ0FBQztRQUNELE1BQU0sQ0FBQyxDQUFDLElBQUksRUFBRSxJQUFJLENBQUMsQ0FBQztJQUN0QixDQUFDO0lBY0QsbUNBQWEsR0FBYixVQUNJLFVBQWtCLEVBQUUsVUFBbUIsRUFBRSxRQUFpQixFQUFFLElBQWEsRUFDekUsQ0FBVSxFQUFFLENBQVU7UUFGMUIsaUJBdUNDO1FBcENDLElBQU0sR0FBRyxHQUFHLElBQUksQ0FBQyxLQUFLLENBQUM7WUFDckIsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsRUFDbkIsb0RBQW9EO2lCQUM3QyxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxpREFBOEMsQ0FBQSxDQUFDLENBQUM7WUFJeEUsSUFBTSxNQUFNLEdBQUcsSUFBSSxDQUFDLElBQUksQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLElBQUksQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUM5QyxJQUFNLEdBQUcsR0FBRyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3JDLElBQU0sVUFBVSxHQUFHLEtBQUksQ0FBQyxRQUFRLENBQUMsTUFBTSxFQUFFLEdBQUcsRUFBRSxDQUFDLENBQUMsQ0FBQztZQUNqRCxJQUFNLFVBQVUsR0FBRyxVQUFVLENBQUMsSUFBSSxDQUFDLENBQUMsRUFBRSxJQUFJLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztZQUVsRSxJQUFNLFFBQVEsR0FBRyxLQUFJLENBQUMsTUFBTSxDQUFDLFVBQVUsRUFBRSxVQUFVLENBQUMsQ0FBQztZQUNyRCxJQUFNLEdBQUcsR0FBRyxLQUFJLENBQUMsR0FBRyxDQUFDLFFBQVEsRUFBRSxRQUFRLENBQVksQ0FBQztZQUdwRCxJQUFNLENBQUMsR0FBRyxLQUFJLENBQUMsT0FBTyxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsR0FBRyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3RFLElBQU0sQ0FBQyxHQUFHLEtBQUksQ0FBQyxPQUFPLENBQ2xCLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxHQUFHLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxD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S0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUM7Z0JBQ3ZDLElBQU0sUUFBUSxHQUFHLEVBQUUsR0FBRyxHQUFHLENBQUM7Z0JBQzFCLElBQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQyxFQUFFLElBQUksQ0FBQyxJQUFJLENBQUMsUUFBUSxHQUFHLFVBQVUsQ0FBQyxDQUFDLENBQUM7Z0JBQzVELElBQU0sS0FBSyxHQUFHLElBQUksQ0FBQyxHQUFHLENBQUMsS0FBSyxFQUFFLENBQUMsS0FBSyxHQUFHLFFBQVEsQ0FBQyxHQUFHLFVBQVUsQ0FBQyxDQUFDO2dCQUUvRCxHQUFHLENBQUMsQ0FBQyxJQUFJLEVBQUUsR0FBRyxDQUFDLEVBQUUsRUFBRSxHQUFHLENBQUMsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQztvQkFDdkMsSUFBTSxRQUFRLEdBQUcsRUFBRSxHQUFHLEdBQUcsQ0FBQztvQkFDMUIsSUFBTSxLQUFLLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxDQUFDLEVBQUUsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRLEdBQUcsVUFBVSxDQUFDLENBQUMsQ0FBQztvQkFDNUQsSUFBTSxLQUFLLEdBQUcsSUFBSSxDQUFDLEdBQUcsQ0FBQyxLQUFLLEVBQUUsQ0FBQyxLQUFLLEdBQUcsUUFBUSxDQUFDLEdBQUcsVUFBVSxDQUFDLENBQUM7b0JBRS9ELElBQUksT0FBTyxHQUFHLENBQUMsQ0FBQztvQkFDaEIsR0FBRyxDQUFDLENBQUMsSUFBSSxFQUFFLEdBQUcsS0FBSyxFQUFFLEVBQUUsR0FBRyxLQUFLLEVBQUUsRUFBRSxFQUFFLEVBQUUsQ0FBQzt3QkFDdEMsSUFBTSxFQUFFLEdBQUcsRUFBRSxHQUFHLFVBQVUsR0FBRyxRQUFRLENBQUM7d0JBRXRDLEdBQUcsQ0FBQyxDQUFDLElBQUksRUFBRSxHQUFHLEtBQUssRUFBRSxFQUFFLEdBQUcsS0FBSyxFQUFFLEVBQUUsRUFBRSxFQUFFLENBQUM7NEJB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LEdBQUcsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsQ0FBQztJQUM5QixDQUFDO0lBQ0QsTUFBTSxDQUFDLE1BQU0sQ0FBQyxPQUFPLEVBQUUsQ0FBQztBQUMxQixDQUFDO0FBOUJELG9FQThCQyIsImZpbGUiOiJnZW5lcmF0ZWQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlc0NvbnRlbnQiOlsiKGZ1bmN0aW9uIGUodCxuLHIpe2Z1bmN0aW9uIHMobyx1KXtpZighbltvXSl7aWYoIXRbb10pe3ZhciBhPXR5cGVvZiByZXF1aXJlPT1cImZ1bmN0aW9uXCImJnJlcXVpcmU7aWYoIXUmJmEpcmV0dXJuIGEobywhMCk7aWYoaSlyZXR1cm4gaShvLCEwKTt2YXIgZj1uZXcgRXJyb3IoXCJDYW5ub3QgZmluZCBtb2R1bGUgJ1wiK28rXCInXCIpO3Rocm93IGYuY29kZT1cIk1PRFVMRV9OT1RfRk9VTkRcIixmfXZhciBsPW5bb109e2V4cG9ydHM6e319O3Rbb11bMF0uY2FsbChsLmV4cG9ydHMsZnVuY3Rpb24oZSl7dmFyIG49dFtvXVsxXVtlXTtyZXR1cm4gcyhuP246ZSl9LGwsbC5leHBvcnRzLGUsdCxuLHIpfXJldHVybiBuW29dLmV4cG9ydHN9dmFyIGk9dHlwZW9mIHJlcXVpcmU9PVwiZnVuY3Rpb25cIiYmcmVxdWlyZTtmb3IodmFyIG89MDtvPHIubGVuZ3RoO28rKylzKHJbb10pO3JldHVybiBzfSkiLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmV4cG9ydCBpbnRlcmZhY2UgQmVuY2htYXJrUnVuR3JvdXAge1xuICBuYW1lOiBzdHJpbmc7XG4gIC8vIE1pbiBhbmQgbWF4IHN0ZXBzIHRvIHJ1biB0aGUgYmVuY2htYXJrIHRlc3Qgb3Zlci5cbiAgbWluOiBudW1iZXI7XG4gIG1heDogbnVtYmVyO1xuICAvLyBUaGUgc2l6ZSBvZiB0aGUgc3RlcCB0byB0YWtlIGJldHdlZW4gYmVuY2htYXJrIHJ1bnMuXG4gIHN0ZXBTaXplOiBudW1iZXI7XG4gIC8vIEEgdHJhbnNmb3JtYXRpb24gb2Ygc3RlcCB0byB0aGUgc2l6ZSBwYXNzZWQgdG8gdGhlIGJlbmNobWFyayB0ZXN0LlxuICBzdGVwVG9TaXplVHJhbnNmb3JtYXRpb24/OiAoc3RlcDogbnVtYmVyKSA9PiBudW1iZXI7XG4gIGJlbmNobWFya1J1bnM6IEJlbmNobWFya1J1bltdO1xufVxuXG5leHBvcnQgY2xhc3MgQmVuY2htYXJrUnVuIHtcbiAgbmFtZTogc3RyaW5nO1xuICBiZW5jaG1hcmtUZXN0OiBCZW5jaG1hcmtUZXN0O1xuXG4gIGNoYXJ0RGF0YTogQ2hhcnREYXRhW107XG4gIGNvbnN0cnVjdG9yKG5hbWU6IHN0cmluZywgYmVuY2htYXJrVGVzdDogQmVuY2htYXJrVGVzdCkge1xuICAgIHRoaXMubmFtZSA9IG5hbWU7XG4gICAgdGhpcy5iZW5jaG1hcmtUZXN0ID0gYmVuY2htYXJrVGVzdDtcbiAgICB0aGlzLmNoYXJ0RGF0YSA9IFtdO1xuICB9XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgQmVuY2htYXJrVGVzdCB7IChzaXplOiBudW1iZXIpOiBudW1iZXI7IH1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uLy4uL3NyYy9tYXRoL2NvbnZfdXRpbCc7XG5pbXBvcnQge0FycmF5MUQsIEFycmF5M0QsIEFycmF5NEQsIGluaXRpYWxpemVHUFV9IGZyb20gJy4uLy4uL3NyYy9tYXRoL25kYXJyYXknO1xuaW1wb3J0IHtDb252MkRQcm9ncmFtfSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9jb252X2dwdSc7XG5pbXBvcnQge0dQR1BVQ29udGV4dH0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvZ3BncHVfY29udGV4dCc7XG5pbXBvcnQgKiBhcyBncGdwdV9tYXRoIGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X21hdGgnO1xuaW1wb3J0IHtUZXh0dXJlTWFuYWdlcn0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvdGV4dHVyZV9tYW5hZ2VyJztcblxuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QX1JVTlMgPSA0MDtcblxuZXhwb3J0IGNvbnN0IEJFTkNITUFSS19URVNUOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBjb25zdCBncGdwdSA9IG5ldyBHUEdQVUNvbnRleHQoKTtcbiAgY29uc3QgdGV4TWFuYWdlciA9IG5ldyBUZXh0dXJlTWFuYWdlcihncGdwdSk7XG4gIGluaXRpYWxpemVHUFUoZ3BncHUsIHRleE1hbmFnZXIpO1xuXG4gIGNvbnN0IGlucHV0RGVwdGggPSAxO1xuICBjb25zdCBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbc2l6ZSwgc2l6ZSwgaW5wdXREZXB0aF07XG4gIGNvbnN0IG91dHB1dERlcHRoID0gMTtcbiAgY29uc3QgZmllbGRTaXplID0gMTE7XG4gIGNvbnN0IHN0cmlkZSA9IDE7XG4gIGNvbnN0IHplcm9QYWQgPSBjb252X3V0aWwuY29tcHV0ZURlZmF1bHRQYWQoaW5wdXRTaGFwZSwgZmllbGRTaXplLCBzdHJpZGUpO1xuXG4gIGNvbnN0IGhhc0JpYXMgPSB0cnVlO1xuICBjb25zdCBwcm9ncmFtID0gbmV3IENvbnYyRFByb2dyYW0oXG4gICAgICBpbnB1dFNoYXBlLCBmaWVsZFNpemUsIG91dHB1dERlcHRoLCBzdHJpZGUsIHplcm9QYWQsIGhhc0JpYXMpO1xuICBjb25zdCBvdXRwdXRTaGFwZSA9IHByb2dyYW0ub3V0cHV0U2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICBjb25zdCBvdXQgPSBBcnJheTNELnplcm9zKG91dHB1dFNoYXBlKTtcbiAgY29uc3QgeCA9IEFycmF5M0QucmFuZFVuaWZvcm0oaW5wdXRTaGFwZSwgLTEsIDEpO1xuICBjb25zdCB3U2hhcGUgPSBjb252X3V0aWwuY29tcHV0ZVdlaWdodHNTaGFwZTREKDEsIG91dHB1dERlcHRoLCBmaWVsZFNpemUpO1xuICBjb25zdCBXID0gQXJyYXk0RC5yYW5kVW5pZm9ybSh3U2hhcGUsIC0xLCAxKTtcbiAgY29uc3QgYiA9IEFycmF5MUQucmFuZFVuaWZvcm0oW291dHB1dERlcHRoXSwgLTEsIDEpO1xuICBjb25zdCBpbnB1dHMgPSBbeCwgVywgYl07XG4gIGNvbnN0IGJpbmFyeSA9IGdwZ3B1X21hdGguY29tcGlsZVByb2dyYW0oZ3BncHUsIHByb2dyYW0sIGlucHV0cywgb3V0KTtcblxuICBjb25zdCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IE9QX1JVTlM7IGkrKykge1xuICAgIGdwZ3B1X21hdGgucnVuUHJvZ3JhbShiaW5hcnksIGlucHV0cywgb3V0KTtcbiAgfVxuICBvdXQuZ2V0VmFsdWVzKCk7XG4gIGNvbnN0IGF2Z1RpbWUgPSAocGVyZm9ybWFuY2Uubm93KCkgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIHguZGlzcG9zZSgpO1xuICBXLmRpc3Bvc2UoKTtcbiAgYi5kaXNwb3NlKCk7XG4gIG91dC5kaXNwb3NlKCk7XG4gIHRleE1hbmFnZXIuZGlzcG9zZSgpO1xuICBncGdwdS5kZWxldGVQcm9ncmFtKGJpbmFyeS53ZWJHTFByb2dyYW0pO1xuICBncGdwdS5kaXNwb3NlKCk7XG5cbiAgcmV0dXJuIGF2Z1RpbWU7XG59O1xuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyBjb252X3V0aWwgZnJvbSAnLi4vLi4vc3JjL21hdGgvY29udl91dGlsJztcbmltcG9ydCB7QXJyYXkzRCwgQXJyYXk0RCwgaW5pdGlhbGl6ZUdQVX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvbmRhcnJheSc7XG5pbXBvcnQge0NvbnYyRFRyYW5zcG9zZVByb2dyYW19IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2NvbnZfYmFja3Byb3BfZ3B1JztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9ncGdwdV9jb250ZXh0JztcbmltcG9ydCAqIGFzIGdwZ3B1X21hdGggZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvZ3BncHVfbWF0aCc7XG5pbXBvcnQge1RleHR1cmVNYW5hZ2VyfSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC90ZXh0dXJlX21hbmFnZXInO1xuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QX1JVTlMgPSA0MDtcblxuZXhwb3J0IGNvbnN0IEJFTkNITUFSS19URVNUOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IDE7XG4gIGNvbnN0IG9yaWdPdXRwdXREZXB0aCA9IDI7XG4gIGNvbnN0IHhTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdID0gW3NpemUsIHNpemUsIDFdO1xuICBjb25zdCBmaWVsZFNpemUgPSAxMTtcbiAgY29uc3Qgb3JpZ1N0cmlkZSA9IDE7XG4gIGNvbnN0IG9yaWdQYWQgPSAxO1xuXG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCB0ZXhNYW5hZ2VyID0gbmV3IFRleHR1cmVNYW5hZ2VyKGdwZ3B1KTtcbiAgaW5pdGlhbGl6ZUdQVShncGdwdSwgdGV4TWFuYWdlcik7XG4gIGdwZ3B1LmVuYWJsZUF1dG9tYXRpY0RlYnVnVmFsaWRhdGlvbih0cnVlKTtcblxuICBjb25zdCBoYXNCaWFzID0gZmFsc2U7XG4gIGNvbnN0IHByb2dyYW0gPSBuZXcgQ29udjJEVHJhbnNwb3NlUHJvZ3JhbShcbiAgICAgIHhTaGFwZSwgZmllbGRTaXplLCBvcmlnSW5wdXREZXB0aCwgb3JpZ1N0cmlkZSwgb3JpZ1BhZCwgaGFzQmlhcyk7XG4gIGNvbnN0IG91dHB1dFNoYXBlID0gcHJvZ3JhbS5vdXRwdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG4gIGNvbnN0IG91dCA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICBjb25zdCB4ID0gQXJyYXkzRC5yYW5kVW5pZm9ybSh4U2hhcGUsIC0xLCAxKTtcbiAgY29uc3Qgd1NoYXBlID0gY29udl91dGlsLmNvbXB1dGVXZWlnaHRzU2hhcGU0RChcbiAgICAgIG9yaWdJbnB1dERlcHRoLCBvcmlnT3V0cHV0RGVwdGgsIGZpZWxkU2l6ZSk7XG4gIGNvbnN0IFcgPSBBcnJheTRELnJhbmRVbmlmb3JtKHdTaGFwZSwgLTEsIDEpO1xuICBjb25zdCBpbnB1dHMgPSBbeCwgV107XG4gIGNvbnN0IGJpbmFyeSA9IGdwZ3B1X21hdGguY29tcGlsZVByb2dyYW0oZ3BncHUsIHByb2dyYW0sIGlucHV0cywgb3V0KTtcbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUF9SVU5TOyBpKyspIHtcbiAgICBncGdwdV9tYXRoLnJ1blByb2dyYW0oYmluYXJ5LCBpbnB1dHMsIG91dCk7XG4gIH1cbiAgb3V0LmdldFZhbHVlcygpO1xuICBjb25zdCBhdmdUaW1lID0gKHBlcmZvcm1hbmNlLm5vdygpIC0gc3RhcnQpIC8gT1BfUlVOUztcblxuICB0ZXhNYW5hZ2VyLmRpc3Bvc2UoKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShiaW5hcnkud2ViR0xQcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuICByZXR1cm4gYXZnVGltZTtcbn07XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7TkRBcnJheU1hdGhDUFV9IGZyb20gJy4uLy4uL3NyYy9tYXRoL21hdGhfY3B1JztcbmltcG9ydCB7QXJyYXkyRCwgTkRBcnJheX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvbmRhcnJheSc7XG5cbmltcG9ydCB7QmVuY2htYXJrVGVzdH0gZnJvbSAnLi9iZW5jaG1hcmsnO1xuXG5jb25zdCBPUFNfUEVSX1JVTiA9IDEwO1xuXG5leHBvcnQgY29uc3QgQkVOQ0hNQVJLX1RFU1Q6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IG1hdGggPSBuZXcgTkRBcnJheU1hdGhDUFUoKTtcbiAgY29uc3QgYSA9IE5EQXJyYXkucmFuZFVuaWZvcm08QXJyYXkyRD4oW3NpemUsIHNpemVdLCAtMSwgMSk7XG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BTX1BFUl9SVU47IGkrKykge1xuICAgIG1hdGgubG9nU3VtRXhwKGEpO1xuICB9XG4gIGNvbnN0IGVuZCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICByZXR1cm4gKGVuZCAtIHN0YXJ0KSAvIE9QU19QRVJfUlVOO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtBcnJheTJELCBpbml0aWFsaXplR1BVLCBTY2FsYXJ9IGZyb20gJy4uLy4uL3NyYy9tYXRoL25kYXJyYXknO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgZ3BncHVfbWF0aCBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9ncGdwdV9tYXRoJztcbmltcG9ydCB7TG9nU3VtRXhwUHJvZ3JhbX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvbG9nc3VtZXhwX2dwdSc7XG5pbXBvcnQge1RleHR1cmVNYW5hZ2VyfSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC90ZXh0dXJlX21hbmFnZXInO1xuXG5pbXBvcnQge0JlbmNobWFya1Rlc3R9IGZyb20gJy4vYmVuY2htYXJrJztcblxuY29uc3QgT1BfUlVOUyA9IDI7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgY29uc3QgZ3BncHUgPSBuZXcgR1BHUFVDb250ZXh0KCk7XG4gIGNvbnN0IHRleE1hbmFnZXIgPSBuZXcgVGV4dHVyZU1hbmFnZXIoZ3BncHUpO1xuICBpbml0aWFsaXplR1BVKGdwZ3B1LCB0ZXhNYW5hZ2VyKTtcbiAgY29uc3Qgb3V0ID0gbmV3IFNjYWxhcih7dGV4dHVyZTogdGV4TWFuYWdlci5hY3F1aXJlVGV4dHVyZShbMSwgMV0pfSk7XG4gIGNvbnN0IGEgPSBBcnJheTJELnJhbmRVbmlmb3JtKFtzaXplLCBzaXplXSwgLTEsIDEpO1xuICBjb25zdCBwcm9ncmFtID0gbmV3IExvZ1N1bUV4cFByb2dyYW0oYS5zaXplKTtcbiAgY29uc3QgYmluYXJ5ID0gZ3BncHVfbWF0aC5jb21waWxlUHJvZ3JhbShncGdwdSwgcHJvZ3JhbSwgW2FdLCBvdXQpO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgZ3BncHVfbWF0aC5ydW5Qcm9ncmFtKGJpbmFyeSwgW2FdLCBvdXQpO1xuICB9XG4gIG91dC5nZXRWYWx1ZXMoKTtcbiAgY29uc3QgYXZnVGltZSA9IChwZXJmb3JtYW5jZS5ub3coKSAtIHN0YXJ0KSAvIE9QX1JVTlM7XG4gIGEuZGlzcG9zZSgpO1xuICBvdXQuZGlzcG9zZSgpO1xuICB0ZXhNYW5hZ2VyLmRpc3Bvc2UoKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShiaW5hcnkud2ViR0xQcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0IHtCZW5jaG1hcmtSdW4sIEJlbmNobWFya1J1bkdyb3VwfSBmcm9tICcuL2JlbmNobWFyayc7XG5pbXBvcnQgKiBhcyBjb252X2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9jb252X2dwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgY29udl90cmFuc3Bvc2VfZ3B1X2JlbmNobWFyayBmcm9tICcuL2NvbnZfdHJhbnNwb3NlX2dwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgbG9nc3VtZXhwX2NwdV9iZW5jaG1hcmsgZnJvbSAnLi9sb2dzdW1leHBfY3B1X2JlbmNobWFyayc7XG5pbXBvcnQgKiBhcyBsb2dzdW1leHBfZ3B1X2JlbmNobWFyayBmcm9tICcuL2xvZ3N1bWV4cF9ncHVfYmVuY2htYXJrJztcbmltcG9ydCAqIGFzIG1heF9wb29sX2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9tYXhfcG9vbF9ncHVfYmVuY2htYXJrJztcbmltcG9ydCAqIGFzIG11bG1hdF9jcHVfYmVuY2htYXJrIGZyb20gJy4vbXVsbWF0X2NwdV9iZW5jaG1hcmsnO1xuaW1wb3J0ICogYXMgbXVsbWF0X2dwdV9iZW5jaG1hcmsgZnJvbSAnLi9tdWxtYXRfZ3B1X2JlbmNobWFyayc7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfUlVOX0dST1VQUzogQmVuY2htYXJrUnVuR3JvdXBbXSA9IFtcbiAge1xuICAgIG5hbWU6XG4gICAgICAgICdNYXRyaXggTXVsdGlwbGljYXRpb24gKENQVSB2cyBHUFUpOiAnICtcbiAgICAgICAgICAgICdtYXRtdWwoW3NpemUsIHNpemVdLCBbc2l6ZSwgc2l6ZV0pJyxcbiAgICBtaW46IDAsXG4gICAgbWF4OiAxMDI0LFxuICAgIHN0ZXBTaXplOiA2NCxcbiAgICBzdGVwVG9TaXplVHJhbnNmb3JtYXRpb246IChzdGVwOiBudW1iZXIpID0+IE1hdGgubWF4KDEsIHN0ZXApLFxuICAgIGJlbmNobWFya1J1bnM6IFtcbiAgICAgIG5ldyBCZW5jaG1hcmtSdW4oJ211bG1hdF9ncHUnLCBtdWxtYXRfZ3B1X2JlbmNobWFyay5CRU5DSE1BUktfVEVTVCksXG4gICAgICBuZXcgQmVuY2htYXJrUnVuKCdtdWxtYXRfY3B1JywgbXVsbWF0X2NwdV9iZW5jaG1hcmsuQkVOQ0hNQVJLX1RFU1QpXG4gICAgXSxcbiAgfSxcbiAge1xuICAgIG5hbWU6ICdDb252b2x1dGlvbiAoR1BVKTogY29udiBvdmVyIGltYWdlIFtzaXplLCBzaXplLCAxXScsXG4gICAgbWluOiAwLFxuICAgIG1heDogMTAyNCxcbiAgICBzdGVwU2l6ZTogNjQsXG4gICAgc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uOiAoc3RlcDogbnVtYmVyKSA9PiBNYXRoLm1heCgxLCBzdGVwKSxcbiAgICBiZW5jaG1hcmtSdW5zOiBbbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgJ2QxPTEsIGQyPTEsIGY9MTEsIHM9MScsIGNvbnZfZ3B1X2JlbmNobWFyay5CRU5DSE1BUktfVEVTVCldLFxuICB9LFxuICB7XG4gICAgbmFtZTogJ0NvbnZvbHV0aW9uIFRyYW5zcG9zZWQgKEdQVSk6IGRlY29udiBvdmVyIGltYWdlIFtzaXplLCBzaXplLCAxXScsXG4gICAgbWluOiAwLFxuICAgIG1heDogMTAyNCxcbiAgICBzdGVwU2l6ZTogNjQsXG4gICAgc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uOiAoc3RlcDogbnVtYmVyKSA9PiBNYXRoLm1heCgxLCBzdGVwKSxcbiAgICBiZW5jaG1hcmtSdW5zOiBbbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgJ2QxPTEsIGQyPTEsIGY9MTEsIHM9MScsIGNvbnZfdHJhbnNwb3NlX2dwdV9iZW5jaG1hcmsuQkVOQ0hNQVJLX1RFU1QpXSxcbiAgfSxcbiAge1xuICAgIG5hbWU6ICdNYXggcG9vbCAoR1BVKScsXG4gICAgbWluOiAwLFxuICAgIG1heDogMTAyNCxcbiAgICBzdGVwU2l6ZTogNjQsXG4gICAgc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uOiAoc3RlcDogbnVtYmVyKSA9PiBNYXRoLm1heCgxLCBzdGVwKSxcbiAgICBiZW5jaG1hcmtSdW5zOiBbbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgJ2QxPTEsIGQyPTEsIGY9MTEsIHM9MScsXG4gICAgICAgIG1heF9wb29sX2dwdV9iZW5jaG1hcmsuTUFYX1BPT0xfQkVOQ0hNQVJLX1RFU1QpXSxcbiAgfSxcbiAge1xuICAgIG5hbWU6ICdMb2dTdW1FeHAgKENQVSB2cyBHUFUpOiBpbnB1dCBbc2l6ZSwgc2l6ZV0nLFxuICAgIG1pbjogMCxcbiAgICBtYXg6IDEwMjQsXG4gICAgc3RlcFNpemU6IDY0LFxuICAgIHN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbjogKHN0ZXA6IG51bWJlcikgPT4gTWF0aC5tYXgoMSwgc3RlcCksXG4gICAgYmVuY2htYXJrUnVuczogW1xuICAgICAgbmV3IEJlbmNobWFya1J1bihcbiAgICAgICAgICAnbG9nc3VtZXhwX2dwdScsIGxvZ3N1bWV4cF9ncHVfYmVuY2htYXJrLkJFTkNITUFSS19URVNUKSxcbiAgICAgIG5ldyBCZW5jaG1hcmtSdW4oJ2xvZ3N1bWV4cF9jcHUnLCBsb2dzdW1leHBfY3B1X2JlbmNobWFyay5CRU5DSE1BUktfVEVTVClcbiAgICBdLFxuICB9XG5dO1xuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgJy4uL2RlbW8taGVhZGVyJztcbmltcG9ydCAnLi4vZGVtby1mb290ZXInO1xuXG5pbXBvcnQge1BvbHltZXJFbGVtZW50LCBQb2x5bWVySFRNTEVsZW1lbnR9IGZyb20gJy4uL3BvbHltZXItc3BlYyc7XG5pbXBvcnQge0JlbmNobWFya1J1bkdyb3VwfSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmltcG9ydCB7QkVOQ0hNQVJLX1JVTl9HUk9VUFN9IGZyb20gJy4vbWF0aC1iZW5jaG1hcmstcnVuLWdyb3Vwcyc7XG5cbi8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTp2YXJpYWJsZS1uYW1lXG5leHBvcnQgbGV0IE1hdGhCZW5jaG1hcmtQb2x5bWVyOiBuZXcgKCkgPT4gUG9seW1lckhUTUxFbGVtZW50ID0gUG9seW1lckVsZW1lbnQoXG4gICAge2lzOiAnbWF0aC1iZW5jaG1hcmsnLCBwcm9wZXJ0aWVzOiB7YmVuY2htYXJrUnVuR3JvdXBOYW1lczogQXJyYXl9fSk7XG5cbmV4cG9ydCBjbGFzcyBNYXRoQmVuY2htYXJrIGV4dGVuZHMgTWF0aEJlbmNobWFya1BvbHltZXIge1xuICAvLyBQb2x5bWVyIHByb3BlcnRpZXMuXG4gIHByaXZhdGUgYmVuY2htYXJrUnVuR3JvdXBOYW1lczogc3RyaW5nW107XG4gIHByaXZhdGUgc3RvcE1lc3NhZ2VzOiBib29sZWFuW107XG5cbiAgcmVhZHkoKSB7XG4gICAgLy8gU2V0IHVwIHRoZSBiZW5jaG1hcmtzIFVJLlxuICAgIGNvbnN0IGJlbmNobWFya1J1bkdyb3VwTmFtZXM6IHN0cmluZ1tdID0gW107XG4gICAgdGhpcy5zdG9wTWVzc2FnZXMgPSBbXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IEJFTkNITUFSS19SVU5fR1JPVVBTLmxlbmd0aDsgaSsrKSB7XG4gICAgICBiZW5jaG1hcmtSdW5Hcm91cE5hbWVzLnB1c2goQkVOQ0hNQVJLX1JVTl9HUk9VUFNbaV0ubmFtZSk7XG4gICAgICB0aGlzLnN0b3BNZXNzYWdlcy5wdXNoKGZhbHNlKTtcbiAgICB9XG4gICAgdGhpcy5iZW5jaG1hcmtSdW5Hcm91cE5hbWVzID0gYmVuY2htYXJrUnVuR3JvdXBOYW1lcztcblxuICAgIC8vIEluIGEgc2V0VGltZW91dCB0byBsZXQgdGhlIFVJIHVwZGF0ZSBiZWZvcmUgd2UgYWRkIGV2ZW50IGxpc3RlbmVycy5cbiAgICBzZXRUaW1lb3V0KCgpID0+IHtcbiAgICAgIGNvbnN0IHJ1bkJ1dHRvbnMgPSB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tdGVzdCcpO1xuICAgICAgY29uc3Qgc3RvcEJ1dHRvbnMgPSB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tc3RvcCcpO1xuICAgICAgZm9yIChsZXQgaSA9IDA7IGkgPCBydW5CdXR0b25zLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIHJ1bkJ1dHRvbnNbaV0uYWRkRXZlbnRMaXN0ZW5lcignY2xpY2snLCAoKSA9PiB7XG4gICAgICAgICAgdGhpcy5ydW5CZW5jaG1hcmtHcm91cChpKTtcbiAgICAgICAgfSk7XG4gICAgICAgIHN0b3BCdXR0b25zW2ldLmFkZEV2ZW50TGlzdGVuZXIoJ2NsaWNrJywgKCkgPT4ge1xuICAgICAgICAgIHRoaXMuc3RvcE1lc3NhZ2VzW2ldID0gdHJ1ZTtcbiAgICAgICAgfSk7XG4gICAgICB9XG4gICAgfSwgMCk7XG4gIH1cblxuICBwcml2YXRlIHJ1bkJlbmNobWFya0dyb3VwKGJlbmNobWFya1J1bkdyb3VwSW5kZXg6IG51bWJlcikge1xuICAgIGNvbnN0IGJlbmNobWFya1J1bkdyb3VwID0gQkVOQ0hNQVJLX1JVTl9HUk9VUFNbYmVuY2htYXJrUnVuR3JvdXBJbmRleF07XG5cbiAgICBjb25zdCBjYW52YXMgPSB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tcGxvdCcpW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdIGFzXG4gICAgICAgIEhUTUxDYW52YXNFbGVtZW50O1xuICAgIGNvbnN0IGNvbnRleHQgPSBjYW52YXMuZ2V0Q29udGV4dCgnMmQnKSBhcyBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQ7XG5cbiAgICBjb25zdCBkYXRhc2V0czogQ2hhcnREYXRhU2V0c1tdID0gW107XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBiZW5jaG1hcmtSdW5Hcm91cC5iZW5jaG1hcmtSdW5zLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBodWUgPSBNYXRoLmZsb29yKDM2MCAqIGkgLyBiZW5jaG1hcmtSdW5Hcm91cC5iZW5jaG1hcmtSdW5zLmxlbmd0aCk7XG4gICAgICBkYXRhc2V0cy5wdXNoKHtcbiAgICAgICAgZGF0YTogYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVuc1tpXS5jaGFydERhdGEsXG4gICAgICAgIGZpbGw6IGZhbHNlLFxuICAgICAgICBsYWJlbDogYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVuc1tpXS5uYW1lLFxuICAgICAgICBib3JkZXJDb2xvcjogJ2hzbCgnICsgaHVlICsgJywgMTAwJSwgNDAlKScsXG4gICAgICAgIGJhY2tncm91bmRDb2xvcjogJ2hzbCgnICsgaHVlICsgJywgMTAwJSwgNzAlKScsXG4gICAgICAgIHBvaW50UmFkaXVzOiAwLFxuICAgICAgICBwb2ludEhpdFJhZGl1czogNSxcbiAgICAgICAgYm9yZGVyV2lkdGg6IDEsXG4gICAgICAgIGxpbmVUZW5zaW9uOiAwXG4gICAgICB9KTtcbiAgICB9XG5cbiAgICBjb25zdCBjaGFydCA9IG5ldyBDaGFydChjb250ZXh0LCB7XG4gICAgICB0eXBlOiAnbGluZScsXG4gICAgICBkYXRhOiB7ZGF0YXNldHN9LFxuICAgICAgb3B0aW9uczoge1xuICAgICAgICBhbmltYXRpb246IHtkdXJhdGlvbjogMH0sXG4gICAgICAgIHJlc3BvbnNpdmU6IGZhbHNlLFxuICAgICAgICBzY2FsZXM6IHtcbiAgICAgICAgICB4QXhlczogW3tcbiAgICAgICAgICAgIHR5cGU6ICdsaW5lYXInLFxuICAgICAgICAgICAgcG9zaXRpb246ICdib3R0b20nLFxuICAgICAgICAgICAgdGlja3M6IHtcbiAgICAgICAgICAgICAgbWluOiBiZW5jaG1hcmtSdW5Hcm91cC5taW4sXG4gICAgICAgICAgICAgIG1heDogYmVuY2htYXJrUnVuR3JvdXAubWF4LFxuICAgICAgICAgICAgICBzdGVwU2l6ZTogYmVuY2htYXJrUnVuR3JvdXAuc3RlcFNpemUsXG4gICAgICAgICAgICAgIGNhbGxiYWNrOiAobGFiZWw6IHN0cmluZykgPT4ge1xuICAgICAgICAgICAgICAgIHJldHVybiBiZW5jaG1hcmtSdW5Hcm91cC5zdGVwVG9TaXplVHJhbnNmb3JtYXRpb24gIT0gbnVsbCA/XG4gICAgICAgICAgICAgICAgICAgIGJlbmNobWFya1J1bkdyb3VwLnN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbigrbGFiZWwpIDpcbiAgICAgICAgICAgICAgICAgICAgK2xhYmVsO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgICAgIH0gYXMgYW55ICAvLyBOb3RlOiB0aGUgdHlwaW5ncyBmb3IgdGhpcyBhcmUgaW5jb3JyZWN0LCBjYXN0IGFzIGFueS5cbiAgICAgICAgICB9XSxcbiAgICAgICAgICB5QXhlczogW3tcbiAgICAgICAgICAgIHRpY2tzOiB7XG4gICAgICAgICAgICAgIGNhbGxiYWNrOiAobGFiZWwsIGluZGV4LCBsYWJlbHMpID0+IHtcbiAgICAgICAgICAgICAgICByZXR1cm4gbGFiZWwgKyAnbXMnO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9LFxuICAgICAgICAgIH1dXG4gICAgICAgIH0sXG4gICAgICAgIHRvb2x0aXBzOiB7bW9kZTogJ2xhYmVsJ30sXG4gICAgICAgIHRpdGxlOiB7dGV4dDogYmVuY2htYXJrUnVuR3JvdXAubmFtZX1cbiAgICAgIH1cbiAgICB9KTtcbiAgICBjYW52YXMuc3R5bGUuZGlzcGxheSA9ICdub25lJztcblxuICAgIGNvbnN0IHJ1bk1lc3NhZ2UgPVxuICAgICAgICB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tbWVzc2FnZScpW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdIGFzXG4gICAgICAgIEhUTUxFbGVtZW50O1xuICAgIHJ1bk1lc3NhZ2Uuc3R5bGUuZGlzcGxheSA9ICdibG9jayc7XG5cbiAgICBjb25zdCBydW5OdW1iZXJzVGFibGUgPVxuICAgICAgICB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tbnVtYmVycy10YWJsZScpW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdIGFzXG4gICAgICAgIEhUTUxFbGVtZW50O1xuICAgIHJ1bk51bWJlcnNUYWJsZS5pbm5lckhUTUwgPSAnJztcbiAgICBydW5OdW1iZXJzVGFibGUuc3R5bGUuZGlzcGxheSA9ICdub25lJztcblxuICAgIC8vIFNldCB1cCB0aGUgaGVhZGVyIGZvciB0aGUgdGFibGUuXG4gICAgY29uc3QgaGVhZGVycyA9IFsnc2l6ZSddO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgYmVuY2htYXJrUnVuR3JvdXAuYmVuY2htYXJrUnVucy5sZW5ndGg7IGkrKykge1xuICAgICAgaGVhZGVycy5wdXNoKGJlbmNobWFya1J1bkdyb3VwLmJlbmNobWFya1J1bnNbaV0ubmFtZSk7XG4gICAgfVxuICAgIHJ1bk51bWJlcnNUYWJsZS5hcHBlbmRDaGlsZCh0aGlzLmJ1aWxkUnVuTnVtYmVyc1JvdyhoZWFkZXJzKSk7XG5cbiAgICB0aGlzLnJ1bkJlbmNobWFya1N0ZXBzKFxuICAgICAgICBjaGFydCwgYmVuY2htYXJrUnVuR3JvdXAsIGJlbmNobWFya1J1bkdyb3VwSW5kZXgsXG4gICAgICAgIGJlbmNobWFya1J1bkdyb3VwLm1pbik7XG4gIH1cblxuICBwcml2YXRlIGJ1aWxkUnVuTnVtYmVyc1Jvdyh2YWx1ZXM6IHN0cmluZ1tdKSB7XG4gICAgY29uc3QgcnVuTnVtYmVyUm93RWxlbWVudCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpO1xuICAgIHJ1bk51bWJlclJvd0VsZW1lbnQuY2xhc3NOYW1lID0gJ3J1bi1udW1iZXJzLXJvdyBtYXRoLWJlbmNobWFyayc7XG5cbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgcnVuTnVtYmVyQ2VsbEVsZW1lbnQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdkaXYnKTtcbiAgICAgIHJ1bk51bWJlckNlbGxFbGVtZW50LmNsYXNzTmFtZSA9ICdydW4tbnVtYmVycy1jZWxsIG1hdGgtYmVuY2htYXJrJztcbiAgICAgIHJ1bk51bWJlckNlbGxFbGVtZW50LmlubmVyVGV4dCA9IHZhbHVlc1tpXTtcbiAgICAgIHJ1bk51bWJlclJvd0VsZW1lbnQuYXBwZW5kQ2hpbGQocnVuTnVtYmVyQ2VsbEVsZW1lbnQpO1xuICAgIH1cbiAgICByZXR1cm4gcnVuTnVtYmVyUm93RWxlbWVudDtcbiAgfVxuXG4gIHByaXZhdGUgcnVuQmVuY2htYXJrU3RlcHMoXG4gICAgICBjaGFydDogQ2hhcnQsIGJlbmNobWFya1J1bkdyb3VwOiBCZW5jaG1hcmtSdW5Hcm91cCxcbiAgICAgIGJlbmNobWFya1J1bkdyb3VwSW5kZXg6IG51bWJlciwgc3RlcDogbnVtYmVyKSB7XG4gICAgY29uc3QgcnVuTnVtYmVyc1RhYmxlID1cbiAgICAgICAgdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLW51bWJlcnMtdGFibGUnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICBIVE1MRWxlbWVudDtcbiAgICBpZiAoc3RlcCA+IGJlbmNobWFya1J1bkdyb3VwLm1heCB8fFxuICAgICAgICB0aGlzLnN0b3BNZXNzYWdlc1tiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSkge1xuICAgICAgdGhpcy5zdG9wTWVzc2FnZXNbYmVuY2htYXJrUnVuR3JvdXBJbmRleF0gPSBmYWxzZTtcblxuICAgICAgcnVuTnVtYmVyc1RhYmxlLnN0eWxlLmRpc3BsYXkgPSAnJztcblxuICAgICAgY29uc3QgY2FudmFzID1cbiAgICAgICAgICB0aGlzLnF1ZXJ5U2VsZWN0b3JBbGwoJy5ydW4tcGxvdCcpW2JlbmNobWFya1J1bkdyb3VwSW5kZXhdIGFzXG4gICAgICAgICAgSFRNTENhbnZhc0VsZW1lbnQ7XG4gICAgICBjYW52YXMuc3R5bGUuZGlzcGxheSA9ICdibG9jayc7XG4gICAgICBjaGFydC51cGRhdGUoKTtcblxuICAgICAgY29uc3QgcnVuTWVzc2FnZSA9XG4gICAgICAgICAgdGhpcy5xdWVyeVNlbGVjdG9yQWxsKCcucnVuLW1lc3NhZ2UnKVtiZW5jaG1hcmtSdW5Hcm91cEluZGV4XSBhc1xuICAgICAgICAgIEhUTUxFbGVtZW50O1xuICAgICAgcnVuTWVzc2FnZS5zdHlsZS5kaXNwbGF5ID0gJ25vbmUnO1xuXG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3QgcnVuTnVtYmVyUm93RWxlbWVudCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2RpdicpO1xuICAgIHJ1bk51bWJlclJvd0VsZW1lbnQuY2xhc3NOYW1lID0gJ3J1bi1udW1iZXJzLXJvdyBtYXRoLWJlbmNobWFyayc7XG5cbiAgICBjb25zdCByb3dWYWx1ZXM6IHN0cmluZ1tdID0gWycnICsgc3RlcF07XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBiZW5jaG1hcmtSdW5Hcm91cC5iZW5jaG1hcmtSdW5zLmxlbmd0aDsgaSsrKSB7XG4gICAgICBjb25zdCBiZW5jaG1hcmtSdW4gPSBiZW5jaG1hcmtSdW5Hcm91cC5iZW5jaG1hcmtSdW5zW2ldO1xuICAgICAgY29uc3QgYmVuY2htYXJrVGVzdCA9IGJlbmNobWFya1J1bi5iZW5jaG1hcmtUZXN0O1xuXG4gICAgICBjb25zdCBzaXplID0gYmVuY2htYXJrUnVuR3JvdXAuc3RlcFRvU2l6ZVRyYW5zZm9ybWF0aW9uICE9IG51bGwgP1xuICAgICAgICAgIGJlbmNobWFya1J1bkdyb3VwLnN0ZXBUb1NpemVUcmFuc2Zvcm1hdGlvbihzdGVwKSA6XG4gICAgICAgICAgc3RlcDtcblxuICAgICAgbGV0IHJlc3VsdFN0cmluZzogc3RyaW5nO1xuICAgICAgbGV0IGxvZ1N0cmluZzogc3RyaW5nO1xuICAgICAgbGV0IHRpbWUgPSAwO1xuICAgICAgbGV0IHN1Y2Nlc3MgPSB0cnVlO1xuXG4gICAgICB0cnkge1xuICAgICAgICB0aW1lID0gYmVuY2htYXJrVGVzdChzaXplKTtcbiAgICAgICAgcmVzdWx0U3RyaW5nID0gdGltZS50b0ZpeGVkKDMpICsgJ21zJztcbiAgICAgICAgbG9nU3RyaW5nID0gcmVzdWx0U3RyaW5nO1xuICAgICAgfSBjYXRjaCAoZSkge1xuICAgICAgICBzdWNjZXNzID0gZmFsc2U7XG4gICAgICAgIHJlc3VsdFN0cmluZyA9ICdFcnJvcic7XG4gICAgICAgIGxvZ1N0cmluZyA9IGUubWVzc2FnZTtcbiAgICAgIH1cblxuICAgICAgaWYgKHRpbWUgPj0gMCkge1xuICAgICAgICBpZiAoc3VjY2Vzcykge1xuICAgICAgICAgIGJlbmNobWFya1J1bi5jaGFydERhdGEucHVzaCh7eDogc3RlcCwgeTogdGltZX0pO1xuICAgICAgICB9XG4gICAgICAgIHJvd1ZhbHVlcy5wdXNoKHJlc3VsdFN0cmluZyk7XG4gICAgICB9XG4gICAgICBjb25zb2xlLmxvZyhiZW5jaG1hcmtSdW4ubmFtZSArICdbJyArIHN0ZXAgKyAnXTogJyArIGxvZ1N0cmluZyk7XG4gICAgfVxuICAgIHJ1bk51bWJlcnNUYWJsZS5hcHBlbmRDaGlsZCh0aGlzLmJ1aWxkUnVuTnVtYmVyc1Jvdyhyb3dWYWx1ZXMpKTtcblxuICAgIHN0ZXAgKz0gYmVuY2htYXJrUnVuR3JvdXAuc3RlcFNpemU7XG4gICAgLy8gQWxsb3cgdGhlIFVJIHRvIHVwZGF0ZS5cbiAgICBzZXRUaW1lb3V0KFxuICAgICAgICAoKSA9PiB0aGlzLnJ1bkJlbmNobWFya1N0ZXBzKFxuICAgICAgICAgICAgY2hhcnQsIGJlbmNobWFya1J1bkdyb3VwLCBiZW5jaG1hcmtSdW5Hcm91cEluZGV4LCBzdGVwKSxcbiAgICAgICAgMTAwKTtcbiAgfVxufVxuZG9jdW1lbnQucmVnaXN0ZXJFbGVtZW50KE1hdGhCZW5jaG1hcmsucHJvdG90eXBlLmlzLCBNYXRoQmVuY2htYXJrKTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uLy4uL3NyYy9tYXRoL2NvbnZfdXRpbCc7XG5pbXBvcnQge0FycmF5M0QsIGluaXRpYWxpemVHUFUsIE5EQXJyYXl9IGZyb20gJy4uLy4uL3NyYy9tYXRoL25kYXJyYXknO1xuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgZ3BncHVfbWF0aCBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9ncGdwdV9tYXRoJztcbmltcG9ydCB7UG9vbDJEUHJvZ3JhbX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvcG9vbF9ncHUnO1xuaW1wb3J0IHtUZXh0dXJlTWFuYWdlcn0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvdGV4dHVyZV9tYW5hZ2VyJztcblxuaW1wb3J0IHtCZW5jaG1hcmtUZXN0fSBmcm9tICcuL2JlbmNobWFyayc7XG5cbmNvbnN0IE9QX1JVTlMgPSA0MDtcblxuZXhwb3J0IGNvbnN0IE1BWF9QT09MX0JFTkNITUFSS19URVNUOiBCZW5jaG1hcmtUZXN0ID0gKHNpemU6IG51bWJlcikgPT4ge1xuICBjb25zdCBwb3NpdGlvbnMgPSBmYWxzZTtcbiAgcmV0dXJuIHRlc3RNYXhQb29sKHNpemUsIHBvc2l0aW9ucyk7XG59O1xuXG5leHBvcnQgY29uc3QgTUFYX1BPT0xfUE9TTlNfQkVOQ0hNQVJLX1RFU1Q6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IHBvc2l0aW9ucyA9IHRydWU7XG4gIHJldHVybiB0ZXN0TWF4UG9vbChzaXplLCBwb3NpdGlvbnMpO1xufTtcblxuZnVuY3Rpb24gdGVzdE1heFBvb2woc2l6ZTogbnVtYmVyLCBwb3NpdGlvbnM6IGJvb2xlYW4pOiBudW1iZXIge1xuICBjb25zdCBncGdwdSA9IG5ldyBHUEdQVUNvbnRleHQoKTtcbiAgY29uc3QgdGV4TWFuYWdlciA9IG5ldyBUZXh0dXJlTWFuYWdlcihncGdwdSk7XG4gIGluaXRpYWxpemVHUFUoZ3BncHUsIHRleE1hbmFnZXIpO1xuXG4gIGNvbnN0IG91dHB1dERlcHRoID0gMTtcbiAgY29uc3QgeFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0gPSBbc2l6ZSwgc2l6ZSwgb3V0cHV0RGVwdGhdO1xuICBjb25zdCBmaWVsZFNpemUgPSAxMTtcbiAgY29uc3Qgc3RyaWRlID0gMTtcbiAgY29uc3QgemVyb1BhZCA9IGNvbnZfdXRpbC5jb21wdXRlRGVmYXVsdFBhZCh4U2hhcGUsIGZpZWxkU2l6ZSwgc3RyaWRlKTtcblxuICBjb25zdCBwcm9ncmFtID1cbiAgICAgIG5ldyBQb29sMkRQcm9ncmFtKHhTaGFwZSwgZmllbGRTaXplLCBzdHJpZGUsIHplcm9QYWQsICdtYXgnLCBwb3NpdGlvbnMpO1xuICBjb25zdCByZXMgPSBOREFycmF5Lnplcm9zKHByb2dyYW0ub3V0cHV0U2hhcGUpO1xuICBjb25zdCB4ID0gQXJyYXkzRC5yYW5kVW5pZm9ybSh4U2hhcGUsIC0xLCAxKTtcbiAgY29uc3QgYmluYXJ5ID0gZ3BncHVfbWF0aC5jb21waWxlUHJvZ3JhbShncGdwdSwgcHJvZ3JhbSwgW3hdLCByZXMpO1xuXG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgT1BfUlVOUzsgaSsrKSB7XG4gICAgZ3BncHVfbWF0aC5ydW5Qcm9ncmFtKGJpbmFyeSwgW3hdLCByZXMpO1xuICB9XG4gIHJlcy5nZXRWYWx1ZXMoKTtcbiAgY29uc3QgYXZnVGltZSA9IChwZXJmb3JtYW5jZS5ub3coKSAtIHN0YXJ0KSAvIE9QX1JVTlM7XG5cbiAgeC5kaXNwb3NlKCk7XG4gIHJlcy5kaXNwb3NlKCk7XG4gIHRleE1hbmFnZXIuZGlzcG9zZSgpO1xuICBncGdwdS5kZWxldGVQcm9ncmFtKGJpbmFyeS53ZWJHTFByb2dyYW0pO1xuICBncGdwdS5kaXNwb3NlKCk7XG5cbiAgcmV0dXJuIGF2Z1RpbWU7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7TkRBcnJheU1hdGhDUFV9IGZyb20gJy4uLy4uL3NyYy9tYXRoL21hdGhfY3B1JztcbmltcG9ydCB7QXJyYXkyRCwgTkRBcnJheX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvbmRhcnJheSc7XG5cbmltcG9ydCB7QmVuY2htYXJrVGVzdH0gZnJvbSAnLi9iZW5jaG1hcmsnO1xuXG5jb25zdCBPUFNfUEVSX1NNQUxMX1JVTiA9IDE7XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVDogQmVuY2htYXJrVGVzdCA9IChzaXplOiBudW1iZXIpID0+IHtcbiAgaWYgKHNpemUgPiA1MTIpIHtcbiAgICByZXR1cm4gLTE7XG4gIH1cbiAgY29uc3QgbWF0aCA9IG5ldyBOREFycmF5TWF0aENQVSgpO1xuICBjb25zdCBhID0gTkRBcnJheS5yYW5kVW5pZm9ybTxBcnJheTJEPihbc2l6ZSwgc2l6ZV0sIC0xLCAxKTtcbiAgY29uc3QgYiA9IE5EQXJyYXkucmFuZFVuaWZvcm08QXJyYXkyRD4oW3NpemUsIHNpemVdLCAtMSwgMSk7XG4gIGNvbnN0IHJ1bnMgPSAoc2l6ZSA8IDE5MikgPyBPUFNfUEVSX1NNQUxMX1JVTiA6IDE7XG4gIGNvbnN0IHN0YXJ0ID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgcnVuczsgaSsrKSB7XG4gICAgbWF0aC5tYXRNdWwoYSwgYik7XG4gIH1cbiAgY29uc3QgZW5kID0gcGVyZm9ybWFuY2Uubm93KCk7XG4gIHJldHVybiAoZW5kIC0gc3RhcnQpIC8gcnVucztcbn07XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7TWF0cml4T3JpZW50YXRpb259IGZyb20gJy4uLy4uL3NyYy9tYXRoL21hdGgnO1xuaW1wb3J0IHtBcnJheTJEfSBmcm9tICcuLi8uLi9zcmMvbWF0aC9uZGFycmF5JztcbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuLi8uLi9zcmMvbWF0aC93ZWJnbC9ncGdwdV9jb250ZXh0JztcbmltcG9ydCB7TWF0TXVsUHJvZ3JhbX0gZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvbXVsbWF0X2dwdSc7XG5pbXBvcnQgKiBhcyBncGdwdV9tYXRoIGZyb20gJy4uLy4uL3NyYy9tYXRoL3dlYmdsL2dwZ3B1X21hdGgnO1xuaW1wb3J0ICogYXMgbXVsbWF0X3BhY2tlZF9ncHUgZnJvbSAnLi4vLi4vc3JjL21hdGgvd2ViZ2wvbXVsbWF0X3BhY2tlZF9ncHUnO1xuaW1wb3J0ICogYXMgdGVzdF91dGlsIGZyb20gJy4uLy4uL3NyYy90ZXN0X3V0aWwnO1xuXG5pbXBvcnQge0JlbmNobWFya1Rlc3R9IGZyb20gJy4vYmVuY2htYXJrJztcblxuY29uc3QgT1BfUlVOUyA9IDQwO1xuXG5leHBvcnQgY29uc3QgQkVOQ0hNQVJLX1RFU1Q6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCBhVGV4dHVyZSA9IGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUoc2l6ZSwgc2l6ZSk7XG4gIGNvbnN0IGJUZXh0dXJlID0gZ3BncHUuY3JlYXRlTWF0cml4VGV4dHVyZShzaXplLCBzaXplKTtcbiAgY29uc3QgcmVzdWx0VGV4dHVyZSA9IGdwZ3B1LmNyZWF0ZU1hdHJpeFRleHR1cmUoc2l6ZSwgc2l6ZSk7XG5cbiAgY29uc3QgYUFyciA9IG5ldyBBcnJheTJEKFxuICAgICAgW3NpemUsIHNpemVdLCB7dGV4dHVyZTogYVRleHR1cmUsIHRleHR1cmVTaGFwZVJDOiBbc2l6ZSwgc2l6ZV19KTtcbiAgY29uc3QgYkFyciA9IG5ldyBBcnJheTJEKFxuICAgICAgW3NpemUsIHNpemVdLCB7dGV4dHVyZTogYlRleHR1cmUsIHRleHR1cmVTaGFwZVJDOiBbc2l6ZSwgc2l6ZV19KTtcbiAgY29uc3QgcmVzQXJyID0gbmV3IEFycmF5MkQoXG4gICAgICBbc2l6ZSwgc2l6ZV0sIHt0ZXh0dXJlOiByZXN1bHRUZXh0dXJlLCB0ZXh0dXJlU2hhcGVSQzogW3NpemUsIHNpemVdfSk7XG4gIGNvbnN0IHByb2dyYW0gPSBuZXcgTWF0TXVsUHJvZ3JhbShhQXJyLnNoYXBlLCBiQXJyLnNoYXBlKTtcbiAgY29uc3QgYmluYXJ5ID1cbiAgICAgIGdwZ3B1X21hdGguY29tcGlsZVByb2dyYW0oZ3BncHUsIHByb2dyYW0sIFthQXJyLCBiQXJyXSwgcmVzQXJyKTtcbiAgY29uc3QgYSA9IHRlc3RfdXRpbC5yYW5kb21BcnJheUluUmFuZ2Uoc2l6ZSAqIHNpemUsIC0xLCAxKTtcbiAgY29uc3QgYiA9IHRlc3RfdXRpbC5yYW5kb21BcnJheUluUmFuZ2Uoc2l6ZSAqIHNpemUsIC0xLCAxKTtcbiAgZ3BncHUudXBsb2FkTWF0cml4VG9UZXh0dXJlKGFUZXh0dXJlLCBzaXplLCBzaXplLCBhKTtcbiAgZ3BncHUudXBsb2FkTWF0cml4VG9UZXh0dXJlKGJUZXh0dXJlLCBzaXplLCBzaXplLCBiKTtcblxuICBjb25zdCBzdGFydCA9IHBlcmZvcm1hbmNlLm5vdygpO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IE9QX1JVTlM7IGkrKykge1xuICAgIGdwZ3B1X21hdGgucnVuUHJvZ3JhbShiaW5hcnksIFthQXJyLCBiQXJyXSwgcmVzQXJyKTtcbiAgfVxuICBncGdwdS5kb3dubG9hZE1hdHJpeEZyb21UZXh0dXJlKHJlc3VsdFRleHR1cmUsIHNpemUsIHNpemUpO1xuICBjb25zdCBhdmdUaW1lID0gKHBlcmZvcm1hbmNlLm5vdygpIC0gc3RhcnQpIC8gT1BfUlVOUztcblxuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGFUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShiVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUocmVzdWx0VGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZVByb2dyYW0oYmluYXJ5LndlYkdMUHJvZ3JhbSk7XG4gIGdwZ3B1LmRpc3Bvc2UoKTtcblxuICByZXR1cm4gYXZnVGltZTtcbn07XG5cbmV4cG9ydCBjb25zdCBCRU5DSE1BUktfVEVTVF9QQUNLRUQ6IEJlbmNobWFya1Rlc3QgPSAoc2l6ZTogbnVtYmVyKSA9PiB7XG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCBwcm9ncmFtOiBXZWJHTFByb2dyYW0gPVxuICAgICAgZ3BncHUuY3JlYXRlUHJvZ3JhbShtdWxtYXRfcGFja2VkX2dwdS5nZXRGcmFnbWVudFNoYWRlclNvdXJjZShcbiAgICAgICAgICBzaXplLCBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSLCBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSk7XG5cbiAgY29uc3QgYVRleHR1cmUgPSBncGdwdS5jcmVhdGVQYWNrZWRNYXRyaXhUZXh0dXJlKHNpemUsIHNpemUpO1xuICBjb25zdCBiVGV4dHVyZSA9IGdwZ3B1LmNyZWF0ZVBhY2tlZE1hdHJpeFRleHR1cmUoc2l6ZSwgc2l6ZSk7XG4gIGNvbnN0IHJlc3VsdFRleHR1cmUgPSBncGdwdS5jcmVhdGVQYWNrZWRNYXRyaXhUZXh0dXJlKHNpemUsIHNpemUpO1xuXG4gIGNvbnN0IGEgPSB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKHNpemUgKiBzaXplLCAtMSwgMSk7XG4gIGNvbnN0IGIgPSB0ZXN0X3V0aWwucmFuZG9tQXJyYXlJblJhbmdlKHNpemUgKiBzaXplLCAtMSwgMSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShhVGV4dHVyZSwgc2l6ZSwgc2l6ZSwgYSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShiVGV4dHVyZSwgc2l6ZSwgc2l6ZSwgYik7XG5cbiAgY29uc3Qgc3RhcnQgPSBwZXJmb3JtYW5jZS5ub3coKTtcbiAgZm9yIChsZXQgaSA9IDA7IGkgPCBPUF9SVU5TOyBpKyspIHtcbiAgICBtdWxtYXRfcGFja2VkX2dwdS5tdWx0aXBseU1hdHJpeFBhY2tlZChcbiAgICAgICAgZ3BncHUsIHByb2dyYW0sIGFUZXh0dXJlLCBiVGV4dHVyZSwgcmVzdWx0VGV4dHVyZSwgW3NpemUsIHNpemVdKTtcbiAgfVxuXG4gIGdwZ3B1LmRvd25sb2FkTWF0cml4RnJvbVBhY2tlZFRleHR1cmUocmVzdWx0VGV4dHVyZSwgc2l6ZSwgc2l6ZSk7XG4gIGNvbnN0IGF2Z1RpbWUgPSAocGVyZm9ybWFuY2Uubm93KCkgLSBzdGFydCkgLyBPUF9SVU5TO1xuXG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoYVRleHR1cmUpO1xuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKGJUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShyZXN1bHRUZXh0dXJlKTtcbiAgZ3BncHUuZGVsZXRlUHJvZ3JhbShwcm9ncmFtKTtcbiAgZ3BncHUuZGlzcG9zZSgpO1xuXG4gIHJldHVybiBhdmdUaW1lO1xufTtcbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblBvbHltZXIoe2lzOiAnZGVtby1mb290ZXInfSk7XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5Qb2x5bWVyKHtpczogJ2RlbW8taGVhZGVyJ30pO1xuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG4vKipcbiAqIEBmaWxlb3ZlcnZpZXdcbiAqXG4gKiBEZWZpbmVzIGFuIGludGVyZmFjZSBmb3IgY3JlYXRpbmcgUG9seW1lciBlbGVtZW50cyBpbiBUeXBlc2NyaXB0IHdpdGggdGhlXG4gKiBjb3JyZWN0IHR5cGluZ3MuIEEgUG9seW1lciBlbGVtZW50IHNob3VsZCBiZSBkZWZpbmVkIGxpa2UgdGhpczpcbiAqXG4gKiBgYGBcbiAqIGxldCBNeUVsZW1lbnRQb2x5bWVyID0gUG9seW1lckVsZW1lbnQoe1xuICogICBpczogJ215LXBvbHltZXItZWxlbWVudCcsXG4gKiAgIHByb3BlcnRpZXM6IHtcbiAqICAgICBmb286IHN0cmluZyxcbiAqICAgICBiYXI6IEFycmF5XG4gKiAgIH1cbiAqIH0pO1xuICpcbiAqIGNsYXNzIE15RWxlbWVudCBleHRlbmRzIE15RWxlbWVudFBvbHltZXIge1xuICogICBmb286IHN0cmluZztcbiAqICAgYmFyOiBudW1iZXJbXTtcbiAqXG4gKiAgIHJlYWR5KCkge1xuICogICAgIGNvbnNvbGUubG9nKCdNeUVsZW1lbnQgaW5pdGlhbGl6ZWQhJyk7XG4gKiAgIH1cbiAqIH1cbiAqXG4gKiBkb2N1bWVudC5yZWdpc3RlckVsZW1lbnQoTXlFbGVtZW50LnByb3RvdHlwZS5pcywgTXlFbGVtZW50KTtcbiAqIGBgYFxuICovXG5cbmV4cG9ydCB0eXBlIFNwZWMgPSB7XG4gIGlzOiBzdHJpbmc7IHByb3BlcnRpZXM6IHtcbiAgICBba2V5OiBzdHJpbmddOiAoRnVuY3Rpb258e1xuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgdHlwZTogRnVuY3Rpb24sIHZhbHVlPzogYW55O1xuICAgICAgcmVmbGVjdFRvQXR0cmlidXRlPzogYm9vbGVhbjtcbiAgICAgIHJlYWRvbmx5PzogYm9vbGVhbjtcbiAgICAgIG5vdGlmeT86IGJvb2xlYW47XG4gICAgICBjb21wdXRlZD86IHN0cmluZztcbiAgICAgIG9ic2VydmVyPzogc3RyaW5nO1xuICAgIH0pXG4gIH07XG4gIG9ic2VydmVycz86IHN0cmluZ1tdO1xufTtcblxuZXhwb3J0IGZ1bmN0aW9uIFBvbHltZXJFbGVtZW50KHNwZWM6IFNwZWMpIHtcbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICByZXR1cm4gUG9seW1lci5DbGFzcyhzcGVjIGFzIGFueSkgYXMge25ldyAoKTogUG9seW1lckhUTUxFbGVtZW50fTtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBQb2x5bWVySFRNTEVsZW1lbnQgZXh0ZW5kcyBIVE1MRWxlbWVudCwgcG9seW1lci5CYXNlIHt9XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBhc3NlcnRDb25jYXQzRFNoYXBlc01hdGNoKFxuICAgIHgxU2hhcGU6IG51bWJlcltdLCB4MlNoYXBlOiBudW1iZXJbXSwgYXhpczogbnVtYmVyLFxuICAgIGVycm9yTWVzc2FnZVByZWZpeCA9ICcnKSB7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgeDFTaGFwZS5sZW5ndGggPT09IDMsXG4gICAgICBlcnJvck1lc3NhZ2VQcmVmaXggKyAnQ29uY2F0M0QgeDEgc2hhcGUgc2hvdWxkIGJlIG9mIHJhbmsgMy4nKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICB4MlNoYXBlLmxlbmd0aCA9PT0gMyxcbiAgICAgIGVycm9yTWVzc2FnZVByZWZpeCArICdDb25jYXQzRCB4MiBzaGFwZSBzaG91bGQgYmUgb2YgcmFuayAzLicpO1xuXG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgYXhpcyA+PSAwICYmIGF4aXMgPCAzLCAnQXhpcyBmb3IgY29uY2F0M0QgbXVzdCBiZSBiZXR3ZWVuIDAgYW5kIDIuJyk7XG5cbiAgZm9yIChsZXQgaSA9IDA7IGkgPCAzOyBpKyspIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgKGkgPT09IGF4aXMpIHx8ICh4MVNoYXBlW2ldID09PSB4MlNoYXBlW2ldKSxcbiAgICAgICAgZXJyb3JNZXNzYWdlUHJlZml4ICtcbiAgICAgICAgICAgIGBTaGFwZSAoJHt4MVNoYXBlfSkgZG9lcyBub3QgbWF0Y2ggKCR7eDJTaGFwZX0pIGFsb25nIGAgK1xuICAgICAgICAgICAgYG5vbi1jb25jYXRlbmF0ZWQgYXhpcy5gKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZUNvbmNhdDNET3V0cHV0U2hhcGUoXG4gICAgeDFTaGFwZTogbnVtYmVyW10sIHgyU2hhcGU6IG51bWJlcltdLFxuICAgIGF4aXM6IG51bWJlcik6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSB7XG4gIHV0aWwuYXNzZXJ0KHgxU2hhcGUubGVuZ3RoID09PSAzLCAnQ29uY2F0M0QgeDEgc2hhcGUgc2hvdWxkIGJlIG9mIHJhbmsgMy4nKTtcbiAgdXRpbC5hc3NlcnQoeDJTaGFwZS5sZW5ndGggPT09IDMsICdDb25jYXQzRCB4MnNoYXBlIHNob3VsZCBiZSBvZiByYW5rIDMuJyk7XG5cbiAgY29uc3Qgb3V0cHV0U2hhcGUgPSB4MVNoYXBlLnNsaWNlKCk7XG4gIG91dHB1dFNoYXBlW2F4aXNdICs9IHgyU2hhcGVbYXhpc107XG4gIHJldHVybiBvdXRwdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl07XG59IiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgaW5wdXRTaGFwZVJvd0NvbERlcHRoOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZpZWxkU2l6ZTogbnVtYmVyLFxuICAgIGRlcHRoOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCB6ZXJvUGFkPzogbnVtYmVyKTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdIHtcbiAgaWYgKHplcm9QYWQgPT0gbnVsbCkge1xuICAgIHplcm9QYWQgPSBjb21wdXRlRGVmYXVsdFBhZChpbnB1dFNoYXBlUm93Q29sRGVwdGgsIGZpZWxkU2l6ZSwgc3RyaWRlKTtcbiAgfVxuICBjb25zdCBpbnB1dFJvd3MgPSBpbnB1dFNoYXBlUm93Q29sRGVwdGhbMF07XG4gIGNvbnN0IGlucHV0Q29scyA9IGlucHV0U2hhcGVSb3dDb2xEZXB0aFsxXTtcbiAgY29uc3Qgb3V0cHV0Um93cyA9IChpbnB1dFJvd3MgLSBmaWVsZFNpemUgKyAyICogemVyb1BhZCkgLyBzdHJpZGUgKyAxO1xuICB1dGlsLmFzc2VydChcbiAgICAgIHV0aWwuaXNJbnQob3V0cHV0Um93cyksXG4gICAgICBgVGhlIG91dHB1dCAjIG9mIHJvd3MgKCR7b3V0cHV0Um93c30pIG11c3QgYmUgYW4gaW50ZWdlci4gQ2hhbmdlIHRoZSBgICtcbiAgICAgICAgICBgc3RyaWRlIGFuZC9vciB6ZXJvIHBhZCBwYXJhbWV0ZXJzYCk7XG5cbiAgY29uc3Qgb3V0cHV0Q29scyA9IChpbnB1dENvbHMgLSBmaWVsZFNpemUgKyAyICogemVyb1BhZCkgLyBzdHJpZGUgKyAxO1xuICB1dGlsLmFzc2VydChcbiAgICAgIHV0aWwuaXNJbnQob3V0cHV0Q29scyksXG4gICAgICBgVGhlIG91dHB1dCAjIG9mIGNvbHVtbnMgKCR7b3V0cHV0Q29sc30pIG11c3QgYmUgYW4gaW50ZWdlci4gQ2hhbmdlIGAgK1xuICAgICAgICAgIGB0aGUgc3RyaWRlIGFuZC9vciB6ZXJvIHBhZCBwYXJhbWV0ZXJzYCk7XG5cbiAgcmV0dXJuIFtvdXRwdXRSb3dzLCBvdXRwdXRDb2xzLCBkZXB0aF07XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlRGVmYXVsdFBhZChcbiAgICBpbnB1dFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZpZWxkU2l6ZTogbnVtYmVyLFxuICAgIHN0cmlkZTogbnVtYmVyKTogbnVtYmVyIHtcbiAgcmV0dXJuIE1hdGguZmxvb3IoKGlucHV0U2hhcGVbMF0gKiAoc3RyaWRlIC0gMSkgLSBzdHJpZGUgKyBmaWVsZFNpemUpIC8gMik7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjb21wdXRlVGV4U2hhcGVGcm9tM0QoXG4gICAgc2hhcGVSb3dDb2xEZXB0aDogW251bWJlciwgbnVtYmVyLCBudW1iZXJdKTogW251bWJlciwgbnVtYmVyXSB7XG4gIHJldHVybiBbc2hhcGVSb3dDb2xEZXB0aFswXSwgc2hhcGVSb3dDb2xEZXB0aFsxXSAqIHNoYXBlUm93Q29sRGVwdGhbMl1dO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZVdlaWdodHNTaGFwZTREKFxuICAgIGlucHV0RGVwdGg6IG51bWJlciwgb3V0cHV0RGVwdGg6IG51bWJlcixcbiAgICBmU2l6ZTogbnVtYmVyKTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0ge1xuICByZXR1cm4gW2ZTaXplLCBmU2l6ZSwgaW5wdXREZXB0aCwgb3V0cHV0RGVwdGhdO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY29tcHV0ZURpbGF0ZWRSQyhcbiAgICByYzogW251bWJlciwgbnVtYmVyXSwgb3JpZ1N0cmlkZTogbnVtYmVyKTogW251bWJlciwgbnVtYmVyXSB7XG4gIGNvbnN0IHJvd3NEaWxhdGVkID0gKHJjWzBdIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcbiAgY29uc3QgY29sc0RpbGF0ZWQgPSAocmNbMV0gLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuICByZXR1cm4gW3Jvd3NEaWxhdGVkLCBjb2xzRGlsYXRlZF07XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmV4cG9ydCBmdW5jdGlvbiB2YWxpZGF0ZVNoYXBlcyhcbiAgICBzb3VyY2VTaXplOiBbbnVtYmVyLCBudW1iZXJdLCBkZXN0U2l6ZTogW251bWJlciwgbnVtYmVyXSkge1xuICBjb25zdCBzcmNBcmVhID0gc291cmNlU2l6ZVswXSAqIHNvdXJjZVNpemVbMV07XG4gIGNvbnN0IGRzdEFyZWEgPSBkZXN0U2l6ZVswXSAqIGRlc3RTaXplWzFdO1xuICBpZiAoc3JjQXJlYSAhPT0gZHN0QXJlYSkge1xuICAgIGNvbnN0IHNyY1N0ciA9ICdbJyArIHNvdXJjZVNpemVbMF0gKyAnLCAnICsgc291cmNlU2l6ZVsxXSArICddJztcbiAgICBjb25zdCBkc3RTdHIgPSAnWycgKyBkZXN0U2l6ZVswXSArICcsICcgKyBkZXN0U2l6ZVsxXSArICddJztcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdjb3B5MkQgc2hhcGVzIGhhdmUgZGlmZmVyZW50IGFyZWFzOlxcbiAgc291cmNlU2l6ZSAnICsgc3JjU3RyICtcbiAgICAgICAgJywgYXJlYSAnICsgc3JjQXJlYSArICdcXG4gIGRlc3RTaXplICcgKyBkc3RTdHIgKyAnLCBhcmVhICcgKyBkc3RBcmVhKTtcbiAgfVxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uL3V0aWwnO1xuaW1wb3J0ICogYXMgY29uY2F0M2RfdXRpbCBmcm9tICcuL2NvbmNhdDNkX3V0aWwnO1xuaW1wb3J0ICogYXMgY29weTJkX3V0aWwgZnJvbSAnLi9jb3B5MmRfdXRpbCc7XG5cbmltcG9ydCB7QXJyYXkxRCwgQXJyYXkyRCwgQXJyYXkzRCwgQXJyYXk0RCwgTkRBcnJheSwgU2NhbGFyfSBmcm9tICcuL25kYXJyYXknO1xuXG5leHBvcnQgdHlwZSBTY29wZVJlc3VsdCA9IE5EQXJyYXlbXXxOREFycmF5fHZvaWQ7XG5cbmV4cG9ydCBpbnRlcmZhY2UgTFNUTUNlbGwge1xuICAoZGF0YTogQXJyYXkyRCwgYzogQXJyYXkyRCwgaDogQXJyYXkyRCk6IFtBcnJheTJELCBBcnJheTJEXTtcbn1cblxuXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgTkRBcnJheU1hdGgge1xuICBwcml2YXRlIG5kYXJyYXlTY29wZXM6IE5EQXJyYXlbXVtdID0gW107XG4gIHByaXZhdGUgYWN0aXZlU2NvcGU6IE5EQXJyYXlbXTtcblxuICBwcml2YXRlIG5kYXJyYXlzVG9LZWVwOiBOREFycmF5W11bXSA9IFtdO1xuICBwcml2YXRlIGFjdGl2ZVNjb3BlTkRBcnJheXNUb0tlZXA6IE5EQXJyYXlbXSA9IFtdO1xuXG4gIHByaXZhdGUgZGVidWdNb2RlID0gZmFsc2U7XG5cbiAgLyoqXG4gICAqIEBwYXJhbSBzYWZlTW9kZSBJbiBzYWZlIG1vZGUsIHlvdSBtdXN0IHVzZSBtYXRoIG9wZXJhdGlvbnMgaW5zaWRlXG4gICAqICAgICBhIG1hdGguc2NvcGUoKSB3aGljaCB3aWxsIGF1dG9tYXRpY2FsbHkgY2xlYW4gdXAgaW50ZXJtZWRpYXRlIE5EQXJyYXlzLlxuICAgKi9cbiAgY29uc3RydWN0b3IocHJpdmF0ZSBzYWZlTW9kZTogYm9vbGVhbikge31cblxuICAvKipcbiAgICogQ3JlYXRlIGEgbmV3IG1hdGggc2NvcGUuIFB1dCBjaGFpbmVkIG1hdGggb3BlcmF0aW9ucyBpbnNpZGUgYSBzY29wZVxuICAgKiBmdW5jdGlvbiBjbG9zdXJlIHNvIHRoYXQgdGhlIGxpYnJhcnkgYXV0b21hdGljYWxseSBjbGVhbnMgdXAgTkRBcnJheXNcbiAgICogZnJvbSBpbnRlcm1lZGlhdGUgbWF0aCBvcGVyYXRpb25zLiBZb3UgbXVzdCBjcmVhdGUgYSBzY29wZSBpbiBzYWZlIG1vZGVcbiAgICogdG8gY2FsbCBtYXRoIG9wZXJhdGlvbnMuIElmIGEgcmVzdWx0IGlzIHJldHVybmVkIGZyb20gdGhlIHNjb3BlLCBpdCB3aWxsXG4gICAqIGFsc28gYmUgdHJhY2tlZCwgd2hpY2ggbWVhbnMgdGhlcmUgbXVzdCBiZSB5ZXQgYW5vdGhlciB3cmFwcGluZyBzY29wZS5cbiAgICogQHBhcmFtIHNjb3BlRm4gVGhlIGZ1bmN0aW9uIHRvIGV4ZWN1dGUgd2l0aCBjaGFpbmVkIG1hdGggb3BlcmF0aW9ucy5cbiAgICovXG4gIHNjb3BlPFQgZXh0ZW5kcyBTY29wZVJlc3VsdD4oXG4gICAgICBzY29wZUZuOlxuICAgICAgICAgIChrZWVwOiA8VDEgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUMSkgPT4gVDEsXG4gICAgICAgICAgIHRyYWNrOiA8VDIgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUMikgPT4gVDIpID0+IFQpIHtcbiAgICB0aGlzLnN0YXJ0U2NvcGUoKTtcblxuICAgIGNvbnN0IGtlZXBGbiA9IDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQgPT4gdGhpcy5rZWVwKG5kYXJyYXkpO1xuICAgIGNvbnN0IHRyYWNrRm4gPSA8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUID0+IHRoaXMudHJhY2sobmRhcnJheSk7XG4gICAgY29uc3QgcmVzdWx0ID0gc2NvcGVGbihrZWVwRm4sIHRyYWNrRm4pO1xuXG4gICAgdGhpcy5lbmRTY29wZShyZXN1bHQpO1xuXG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG5cbiAgLyoqXG4gICAqIEluIGRlYnVnIG1vZGUsIHRoZSBvdXRwdXQgb2YgZXZlcnkgbWF0aCBjYWxsIHdpbGwgYmUgZG93bmxvYWRlZCB0byB0aGUgQ1BVXG4gICAqIGFuZCBjaGVja2VkIGZvciBOYU5zLiBUaGlzIHNpZ25pZmljYW50bHkgaW1wYWN0cyBwZXJmb3JtYW5jZS5cbiAgICovXG4gIGVuYWJsZURlYnVnTW9kZSgpIHtcbiAgICB0aGlzLmRlYnVnTW9kZSA9IHRydWU7XG4gICAgY29uc29sZS53YXJuKFxuICAgICAgICAnRGVidWdnaW5nIG1vZGUgaXMgT04uIFRoZSBvdXRwdXQgb2YgZXZlcnkgbWF0aCBjYWxsIHdpbGwgJyArXG4gICAgICAgICdiZSBkb3dubG9hZGVkIHRvIENQVSBhbmQgY2hlY2tlZCBmb3IgTmFOcy4gJyArXG4gICAgICAgICdUaGlzIHNpZ25pZmljYW50bHkgaW1wYWN0cyBwZXJmb3JtYW5jZS4nKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBTdGFydCBhIHNjb3BlLiBVc2UgdGhpcyB3aXRoIGVuZFNjb3BlKCkgdG8gYWNoaWV2ZSB0aGUgc2FtZSBmdW5jdGlvbmFsaXR5XG4gICAqIGFzIHNjb3BlKCkgd2l0aG91dCB0aGUgbmVlZCBmb3IgYSBmdW5jdGlvbiBjbG9zdXJlLlxuICAgKi9cbiAgc3RhcnRTY29wZSgpIHtcbiAgICBjb25zdCBuZXdTY29wZTogTkRBcnJheVtdID0gW107XG4gICAgdGhpcy5uZGFycmF5U2NvcGVzLnB1c2gobmV3U2NvcGUpO1xuICAgIHRoaXMuYWN0aXZlU2NvcGUgPSBuZXdTY29wZTtcblxuICAgIGNvbnN0IG5ld05EQXJyYXlzVG9LZWVwOiBOREFycmF5W10gPSBbXTtcbiAgICB0aGlzLm5kYXJyYXlzVG9LZWVwLnB1c2gobmV3TkRBcnJheXNUb0tlZXApO1xuICAgIHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcCA9IG5ld05EQXJyYXlzVG9LZWVwO1xuICB9XG5cbiAgLyoqXG4gICAqIEVuZCBhIHNjb3BlLiBVc2UgdGhpcyB3aXRoIHN0YXJ0U2NvcGUoKSB0byBhY2hpZXZlIHRoZSBzYW1lIGZ1bmN0aW9uYWxpdHlcbiAgICogYXMgc2NvcGUoKSB3aXRob3V0IHRoZSBuZWVkIGZvciBhIGZ1bmN0aW9uIGNsb3N1cmUuXG4gICAqL1xuICBlbmRTY29wZShyZXN1bHQ6IFNjb3BlUmVzdWx0KSB7XG4gICAgbGV0IGFycmF5c1RvS2VlcCA9IHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcDtcbiAgICBpZiAocmVzdWx0ICE9IG51bGwpIHtcbiAgICAgIGFycmF5c1RvS2VlcCA9IGFycmF5c1RvS2VlcC5jb25jYXQocmVzdWx0IGFzIE5EQXJyYXkgfCBOREFycmF5W10pO1xuICAgIH1cbiAgICAvLyBEaXNwb3NlIHRoZSBjdXJyZW50IHNjb3BlLlxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5hY3RpdmVTY29wZS5sZW5ndGg7IGkrKykge1xuICAgICAgY29uc3QgbmRhcnJheSA9IHRoaXMuYWN0aXZlU2NvcGVbaV07XG4gICAgICBpZiAodGhpcy5pc05EQXJyYXlEYXRhSW5MaXN0KG5kYXJyYXksIGFycmF5c1RvS2VlcCkpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBuZGFycmF5LmRpc3Bvc2UoKTtcbiAgICB9XG5cbiAgICAvLyBQb3AgdGhlIGN1cnJlbnQgc2NvcGUuXG4gICAgdGhpcy5uZGFycmF5U2NvcGVzLnBvcCgpO1xuICAgIHRoaXMuYWN0aXZlU2NvcGUgPSB0aGlzLm5kYXJyYXlTY29wZXMubGVuZ3RoID09PSAwID9cbiAgICAgICAgbnVsbCEgOlxuICAgICAgICB0aGlzLm5kYXJyYXlTY29wZXNbdGhpcy5uZGFycmF5U2NvcGVzLmxlbmd0aCAtIDFdO1xuXG4gICAgLy8gVHJhY2sgdGhlIGN1cnJlbnQgcmVzdWx0IGluIHRoZSBwYXJlbnQgc2NvcGUuXG4gICAgaWYgKHJlc3VsdCBpbnN0YW5jZW9mIE5EQXJyYXkgJiZcbiAgICAgICAgIXRoaXMuaXNOREFycmF5RGF0YUluTGlzdChyZXN1bHQsIHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcCkpIHtcbiAgICAgIHRoaXMudHJhY2socmVzdWx0KTtcbiAgICB9IGVsc2UgaWYgKEFycmF5LmlzQXJyYXkocmVzdWx0KSkge1xuICAgICAgcmVzdWx0LmZvckVhY2gociA9PiB7XG4gICAgICAgIGlmIChyIGluc3RhbmNlb2YgTkRBcnJheSAmJlxuICAgICAgICAgICAgIXRoaXMuaXNOREFycmF5RGF0YUluTGlzdChyLCB0aGlzLmFjdGl2ZVNjb3BlTkRBcnJheXNUb0tlZXApKSB7XG4gICAgICAgICAgdGhpcy50cmFjayhyKTtcbiAgICAgICAgfVxuICAgICAgfSk7XG4gICAgfVxuXG4gICAgdGhpcy5uZGFycmF5c1RvS2VlcC5wb3AoKTtcbiAgICB0aGlzLmFjdGl2ZVNjb3BlTkRBcnJheXNUb0tlZXAgPSB0aGlzLm5kYXJyYXlzVG9LZWVwLmxlbmd0aCA9PT0gMCA/XG4gICAgICAgIG51bGwhIDpcbiAgICAgICAgdGhpcy5uZGFycmF5c1RvS2VlcFt0aGlzLm5kYXJyYXlzVG9LZWVwLmxlbmd0aCAtIDFdO1xuICB9XG5cbiAgcHJpdmF0ZSBpc05EQXJyYXlEYXRhSW5MaXN0KG5kYXJyYXk6IE5EQXJyYXksIG5kYXJyYXlMaXN0OiBOREFycmF5W10pIHtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG5kYXJyYXlMaXN0Lmxlbmd0aDsgaSsrKSB7XG4gICAgICBpZiAobmRhcnJheUxpc3RbaV0uZ2V0RGF0YSgpID09PSBuZGFycmF5LmdldERhdGEoKSkge1xuICAgICAgICByZXR1cm4gdHJ1ZTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG5cbiAgLyoqXG4gICAqIEtlZXBzIGFuIE5EQXJyYXkgaW4gdGhlIGN1cnJlbnQgc2NvcGUgZnJvbSBiZWluZyBkaXNwb3NlZCBhdXRvbWF0aWNhbGx5LlxuICAgKiBAcGFyYW0gcmVzdWx0IFRoZSBOREFycmF5IHRvIGtlZXAgZnJvbSBiZWluZyBkaXNwb3NlZC5cbiAgICovXG4gIGtlZXA8VCBleHRlbmRzIE5EQXJyYXk+KHJlc3VsdDogVCk6IFQge1xuICAgIGlmICh0aGlzLmFjdGl2ZVNjb3BlID09IG51bGwpIHtcbiAgICAgIGlmICh0aGlzLnNhZmVNb2RlKSB7XG4gICAgICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgICAgICdZb3UgYXJlIHVzaW5nIG1hdGggaW4gc2FmZSBtb2RlLiBFbmNsb3NlIGFsbCAnICtcbiAgICAgICAgICAgICdtYXRoLm1ldGhvZCgpIGNhbGxzIGluc2lkZSBhIHNjb3BlOiAnICtcbiAgICAgICAgICAgICdtYXRoLnNjb3BlKCgpID0+IHttYXRoLm1ldGhvZCgpOy4uLn0pIHRvIGF2b2lkIG1lbW9yeSAnICtcbiAgICAgICAgICAgICdsZWFrcy4nKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiByZXN1bHQ7XG4gICAgfVxuICAgIHRoaXMuYWN0aXZlU2NvcGVOREFycmF5c1RvS2VlcC5wdXNoKHJlc3VsdCk7XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHByaXZhdGUgY2hlY2tGb3JOYU4oYXJyOiBOREFycmF5KTogdm9pZCB7XG4gICAgY29uc3QgdmFscyA9IGFyci5nZXRWYWx1ZXMoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHMubGVuZ3RoOyBpKyspIHtcbiAgICAgIGlmIChpc05hTih2YWxzW2ldKSkge1xuICAgICAgICB0aHJvdyBFcnJvcignVGhlIHJlc3VsdCBOREFycmF5IG9mIHRoZSBsYXN0IG1hdGggY2FsbCBoYXMgTmFOcy4nKTtcbiAgICAgIH1cbiAgICB9XG4gIH1cblxuICAvKipcbiAgICogVHJhY2tzIGFuIE5EQXJyYXkgaW4gdGhlIGN1cnJlbnQgc2NvcGUgdG8gYmUgYXV0b21hdGljYWxseSBjbGVhbmVkIHVwIHdoZW5cbiAgICogdGhlIGN1cnJlbnQgc2NvcGUgZW5kcywgYW5kIHJldHVybnMgdGhlIHZhbHVlLlxuICAgKiBAcGFyYW0gcmVzdWx0IFRoZSBOREFycmF5IHRvIHRyYWNrIGluIHRoZSBjdXJyZW50IHNjb3BlLlxuICAgKi9cbiAgdHJhY2s8VCBleHRlbmRzIE5EQXJyYXk+KHJlc3VsdDogVCk6IFQge1xuICAgIGlmICh0aGlzLmRlYnVnTW9kZSkge1xuICAgICAgdGhpcy5jaGVja0Zvck5hTihyZXN1bHQpO1xuICAgIH1cbiAgICBpZiAodGhpcy5hY3RpdmVTY29wZSA9PSBudWxsKSB7XG4gICAgICBpZiAodGhpcy5zYWZlTW9kZSkge1xuICAgICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgICAnWW91IGFyZSB1c2luZyBtYXRoIGluIHNhZmUgbW9kZS4gRW5jbG9zZSBhbGwgJyArXG4gICAgICAgICAgICAnbWF0aC5tZXRob2QoKSBjYWxscyBpbnNpZGUgYSBzY29wZTogJyArXG4gICAgICAgICAgICAnbWF0aC5zY29wZSgoKSA9PiB7bWF0aC5tZXRob2QoKTsuLi59KSB0byBhdm9pZCBtZW1vcnkgJyArXG4gICAgICAgICAgICAnbGVha3MuJyk7XG4gICAgICB9XG4gICAgICByZXR1cm4gcmVzdWx0O1xuICAgIH1cbiAgICB0aGlzLmFjdGl2ZVNjb3BlLnB1c2gocmVzdWx0KTtcbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBkb3QgcHJvZHVjdCBvZiB0d28gbWF0cmljZXMsIEEgKiBCLiBUaGVzZSBtdXN0IGJlIG1hdHJpY2VzLFxuICAgKiB1c2UgbWF0cml4VGltZXNWZWN0b3IgYW5kIHZlY3RvclRpbWVzTWF0cml4LCBkb3RQcm9kdWN0LCBhbmQgb3V0ZXJQcm9kdWN0XG4gICAqIGluIG90aGVyIGNhc2VzLlxuICAgKiBAcGFyYW0gYSBGaXJzdCBtYXRyaXggaW4gZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKiBAcGFyYW0gYiBTZWNvbmQgbWF0cml4IGluIGRvdCBwcm9kdWN0IG9wZXJhdGlvbi5cbiAgICogQHBhcmFtIGFPcmllbnRhdGlvbiBUaGUgTWF0cml4T3JpZW50YXRpb24gb2YgQS4gSWYgdXNpbmcgVFJBTlNQT1NFRCwgd2lsbFxuICAgKiBjb21wdXRlIEFeVCAqIEIuXG4gICAqIEBwYXJhbSBiT3JpZW50YXRpb24gVGhlIE1hdHJpeE9yaWVudGF0aW9uIG9mIEIuIElmIHVzaW5nIFRSQU5TUE9TRUQsIHdpbGxcbiAgICogY29tcHV0ZSBBICogQl5ULlxuICAgKi9cbiAgbWF0TXVsKFxuICAgICAgYTogQXJyYXkyRCwgYjogQXJyYXkyRCwgYU9yaWVudGF0aW9uID0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUixcbiAgICAgIGJPcmllbnRhdGlvbiA9IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpOiBBcnJheTJEIHtcbiAgICBjb25zdCBpbm5lclNoYXBlQSA9XG4gICAgICAgIChhT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID8gYS5zaGFwZVsxXSA6IGEuc2hhcGVbMF07XG4gICAgY29uc3QgaW5uZXJTaGFwZUIgPVxuICAgICAgICAoYk9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IGIuc2hhcGVbMF0gOiBiLnNoYXBlWzFdO1xuXG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGEucmFuayA9PT0gMiAmJiBiLnJhbmsgPT09IDIsXG4gICAgICAgIGBFcnJvciBpbiBtYXRNdWw6IGlucHV0cyBtdXN0IGJlIHJhbmsgMiwgZ290IHJhbmtzICR7YS5yYW5rfWAgK1xuICAgICAgICAgICAgYGFuZCAke2IucmFua30uYCk7XG5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgaW5uZXJTaGFwZUEgPT09IGlubmVyU2hhcGVCLFxuICAgICAgICBgRXJyb3IgaW4gbWF0TXVsOiBpbm5lciBzaGFwZXMgKCR7aW5uZXJTaGFwZUF9KSBhbmQgKGAgK1xuICAgICAgICAgICAgYCR7aW5uZXJTaGFwZUJ9KSBvZiBOREFycmF5cyB3aXRoIHNoYXBlcyAke2Euc2hhcGV9IGFuZCBgICtcbiAgICAgICAgICAgIGAke2Iuc2hhcGV9IGFuZCBvcmllbnRhdGlvbnMgJHtNYXRyaXhPcmllbnRhdGlvblthT3JpZW50YXRpb25dfWAgK1xuICAgICAgICAgICAgYCBhbmQgJHtNYXRyaXhPcmllbnRhdGlvbltiT3JpZW50YXRpb25dfSBtdXN0IG1hdGNoLmApO1xuXG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5tYXRNdWxJbnRlcm5hbChhLCBiLCBhT3JpZW50YXRpb24sIGJPcmllbnRhdGlvbikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBtYXRNdWxJbnRlcm5hbChcbiAgICAgIGE6IEFycmF5MkQsIGI6IEFycmF5MkQsIGFPcmllbnRhdGlvbjogTWF0cml4T3JpZW50YXRpb24sXG4gICAgICBiT3JpZW50YXRpb246IE1hdHJpeE9yaWVudGF0aW9uKTogQXJyYXkyRDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGRvdCBwcm9kdWN0IG9mIGEgdmVjdG9yIGFuZCBhIG1hdHJpeCwgdiAqIEIuXG4gICAqIEBwYXJhbSB2IFRoZSB2ZWN0b3IgaW4gZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKiBAcGFyYW0gbWF0cml4IFRoZSBtYXRyaXggaW4gZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKi9cbiAgdmVjdG9yVGltZXNNYXRyaXgodjogQXJyYXkxRCwgbWF0cml4OiBBcnJheTJEKTogQXJyYXkxRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHYucmFuayA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIHZlY3RvclRpbWVzTWF0cml4OiBmaXJzdCBpbnB1dCBtdXN0IGJlIHJhbmsgMSwgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGByYW5rICR7di5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgbWF0cml4LnJhbmsgPT09IDIsXG4gICAgICAgIGBFcnJvciBpbiB2ZWN0b3JUaW1lc01hdHJpeDogc2Vjb25kIGlucHV0IG11c3QgYmUgcmFuayAyLCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYHJhbmsgJHttYXRyaXgucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHYuc2l6ZSA9PT0gbWF0cml4LnNoYXBlWzBdLFxuICAgICAgICBgRXJyb3IgaW4gdmVjdG9yVGltZXNNYXRyaXg6IHNpemUgb2YgZmlyc3QgcmFuayAxIGlucHV0ICgke3Yuc2l6ZX0pIGAgK1xuICAgICAgICAgICAgYG11c3QgbWF0Y2ggaW5uZXIgZGltZW5zaW9uIG9mIHNlY29uZCByYW5rIDIgaW5wdXQsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke21hdHJpeC5yYW5rfS5gKTtcblxuICAgIHJldHVybiB0aGlzLm1hdE11bCh2LmFzMkQoMSwgdi5zaXplKSwgbWF0cml4KS5hczFEKCk7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGRvdCBwcm9kdWN0IG9mIGEgbWF0cml4IGFuZCB2ZWN0b3IsIEEgKiB2LlxuICAgKiBAcGFyYW0gbWF0cml4IFRoZSBtYXRyaXggaW4gZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKiBAcGFyYW0gdiBUaGUgdmVjdG9yIGluIGRvdCBwcm9kdWN0IG9wZXJhdGlvbi5cbiAgICovXG4gIG1hdHJpeFRpbWVzVmVjdG9yKG1hdHJpeDogQXJyYXkyRCwgdjogQXJyYXkxRCk6IEFycmF5MUQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB2LnJhbmsgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiB2ZWN0b3JUaW1lc01hdHJpeDogc2Vjb25kIGlucHV0IG11c3QgcmFuayAxLCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYHJhbmsgJHt2LnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBtYXRyaXgucmFuayA9PT0gMixcbiAgICAgICAgYEVycm9yIGluIHZlY3RvclRpbWVzTWF0cml4OiBmaXJzdCBpbnB1dCBtdXN0IGJlIGEgcmFuayAyLCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYHJhbmsgJHttYXRyaXgucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHYuc2l6ZSA9PT0gbWF0cml4LnNoYXBlWzFdLFxuICAgICAgICBgRXJyb3IgaW4gdmVjdG9yVGltZXNNYXRyaXg6IHNpemUgb2YgZmlyc3QgcmFuayAxIGlucHV0ICR7di5zaXplfSBgICtcbiAgICAgICAgICAgIGBtdXN0IG1hdGNoIGlubmVyIGRpbWVuc2lvbiBvZiBzZWNvbmQgcmFuayAyIGlucHV0LCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYHNoYXBlICR7bWF0cml4LnNoYXBlfS5gKTtcblxuICAgIHJldHVybiB0aGlzLm1hdE11bChtYXRyaXgsIHYuYXMyRCh2LnNpemUsIDEpKS5hczFEKCk7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGRvdCBwcm9kdWN0IG9mIHR3byB2ZWN0b3JzLCB2MSAqIHYyLlxuICAgKiBAcGFyYW0gdjEgVGhlIGZpcnN0IHZlY3RvciBpbiB0aGUgZG90IHByb2R1Y3Qgb3BlcmF0aW9uLlxuICAgKiBAcGFyYW0gdjIgVGhlIHNlY29uZCB2ZWN0b3IgaW4gdGhlIGRvdCBwcm9kdWN0IG9wZXJhdGlvbi5cbiAgICovXG4gIGRvdFByb2R1Y3QodjE6IEFycmF5MUQsIHYyOiBBcnJheTFEKTogU2NhbGFyIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgdjEucmFuayA9PT0gMSAmJiB2Mi5yYW5rID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gZG90UHJvZHVjdDogaW5wdXRzIG11c3QgYmUgcmFuayAxLCBidXQgZ290IHJhbmtzIGAgK1xuICAgICAgICAgICAgYCR7djEucmFua30gYW5kICR7djIucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHYxLnNpemUgPT09IHYyLnNpemUsXG4gICAgICAgIGBFcnJvciBpbiBkb3RQcm9kdWN0OiBzaXplIG9mIGlucHV0cyAoJHt2MS5zaXplfSkgYW5kIChgICtcbiAgICAgICAgICAgIGAke3YyLnNpemV9KSBtdXN0IG1hdGNoLmApO1xuICAgIHJldHVybiB0aGlzLm1hdE11bCh2MS5hczJEKDEsIHYxLnNpemUpLCB2Mi5hczJEKHYyLnNpemUsIDEpKS5hc1NjYWxhcigpO1xuICB9XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBvdXRlciBwcm9kdWN0IG9mIHR3byB2ZWN0b3JzLCB2MSBhbmQgdjIuXG4gICAqIEBwYXJhbSB2MSBUaGUgZmlyc3QgdmVjdG9yIGluIHRoZSBvdXRlciBwcm9kdWN0IG9wZXJhdGlvbi5cbiAgICogQHBhcmFtIHYyIFRoZSBzZWNvbmQgdmVjdG9yIGluIHRoZSBkb3QgcHJvZHVjdCBvcGVyYXRpb24uXG4gICAqL1xuICBvdXRlclByb2R1Y3QodjE6IEFycmF5MUQsIHYyOiBBcnJheTFEKTogQXJyYXkyRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHYxLnJhbmsgPT09IDEgJiYgdjIucmFuayA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIG91dGVyUHJvZHVjdDogaW5wdXRzIG11c3QgYmUgcmFuayAxLCBidXQgZ290IHJhbmtzIGAgK1xuICAgICAgICAgICAgYCR7djEucmFua30gYW5kICR7djIucmFua30uYCk7XG5cbiAgICByZXR1cm4gdGhpcy5tYXRNdWwodjEuYXMyRCh2MS5zaXplLCAxKSwgdjIuYXMyRCgxLCB2Mi5zaXplKSk7XG4gIH1cblxuICAvLy8vLy8vLy8vLy8vLy9cbiAgLy8gU2hhcGUgb3BzIC8vXG4gIC8vLy8vLy8vLy8vLy8vL1xuXG4gIC8qKlxuICAgKiBDbG9uZXMgYW4gTkRBcnJheSBvZiBhbnkgc2hhcGUuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBOREFycmF5IHRvIGNsb25lLlxuICAgKi9cbiAgY2xvbmU8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmNsb25lSW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBjbG9uZUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVDtcblxuICAvKipcbiAgICogQGRlcHJlY2F0ZWQgUGxlYXNlIGNhbGwgcmVzaGFwZSgpIGRpcmVjdGx5IG9uIHRoZSBuZGFycmF5IG9iamVjdC5cbiAgICovXG4gIHJlc2hhcGU8VDEgZXh0ZW5kcyBOREFycmF5LCBUMiBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgbmRhcnJheTogVDEsIG5ld1NoYXBlOiBudW1iZXJbXSk6IFQyIHtcbiAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICdtYXRoLnJlc2hhcGUoKSBpcyBkZXByZWNhdGVkLiBQbGVhc2UgY2FsbCByZXNoYXBlKCkgJyArXG4gICAgICAgICdkaXJlY3RseSBvbiB0aGUgbmRhcnJheSBvYmplY3QnKTtcbiAgICByZXR1cm4gbmRhcnJheS5yZXNoYXBlKG5ld1NoYXBlKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBFeHRyYWN0cyBhIHNsaWNlIGZyb20gYSBtYXRyaXguIFRoZSBvcGVyYXRpb24gZXh0cmFjZXMgYSBzbGljZSBmcm9tIGlucHV0XG4gICAqIHRoYXQgc3RhcnRzIGF0IGNvb3JkaW5hdGVzIGBiZWdpbmAgYW5kIGlzIG9mIHNpemUgYHNpemVgLlxuICAgKiBAcGFyYW0gaW5wdXQgVGhlIGlucHV0IG1hdHJpeCB0byBzbGljZSBmcm9tLlxuICAgKiBAcGFyYW0gYmVnaW4gVGhlIDJEIGNvb3JkaW5hdGVzIGluIHRoZSBpbnB1dCBtYXRyaXggdG8gc3RhcnQgdGhlIHNsaWNlXG4gICAqIGZyb20uXG4gICAqIEBwYXJhbSBzaXplIFRoZSBzaWNlIG9mIHRoZSAyRCB3aW5kb3cgdG8gc2xpY2UuXG4gICAqL1xuICBzbGljZTJEKGlucHV0OiBBcnJheTJELCBiZWdpbjogW251bWJlciwgbnVtYmVyXSwgc2l6ZTogW251bWJlciwgbnVtYmVyXSk6XG4gICAgICBBcnJheTJEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYmVnaW5bMF0gKyBzaXplWzBdIDw9IGlucHV0LnNoYXBlWzBdICYmXG4gICAgICAgICAgICBiZWdpblsxXSArIHNpemVbMV0gPD0gaW5wdXQuc2hhcGVbMV0sXG4gICAgICAgIGBFcnJvciBpbiBzbGljZTJEOiByZXF1ZXN0ZWQgc3RhcnQgcG9zaXRpb24gJHtiZWdpbn0gYW5kIHNpemUgYCArXG4gICAgICAgICAgICBgJHtzaXplfSB3b3VsZCBvdmVyZmxvdyBpbnB1dCBvZiBzaGFwZSAke2lucHV0LnNoYXBlfS5gKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnNsaWNlMkRJbnRlcm5hbChpbnB1dCwgYmVnaW4sIHNpemUpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc2xpY2UyREludGVybmFsKFxuICAgICAgaW5wdXQ6IEFycmF5MkQsIGJlZ2luOiBbbnVtYmVyLCBudW1iZXJdLCBzaXplOiBbbnVtYmVyLCBudW1iZXJdKTogQXJyYXkyRDtcblxuICAvKipcbiAgICogQ29waWVzIGEgd2luZG93IGZyb20gdGhlIGBzb3VyY2VgIG1hdHJpeCBzdGFydGluZyBhdCBgc291cmNlQmVnaW5gIGFuZCBpc1xuICAgKiBvZiBzaXplIGBzb3VyY2VTaXplYCB0byBhIHdpbmRvdyBpbiB0aGUgYGRlc3RgIG1hdHJpeCBzdGFydGluZyBhdFxuICAgKiBgZGVzdEJlZ2luYCBhbmQgaXMgb2Ygc2l6ZSBgZGVzdFNpemVgL1xuICAgKiBAcGFyYW0gc291cmNlIFRoZSBzb3VyY2UgbWF0cml4IHRvIGNvcHkgZnJvbS5cbiAgICogQHBhcmFtIHNvdXJjZUJlZ2luIFRoZSBjb29yZGluYXRlcyB0byBzdGFydCB0aGUgY29weSBmcm9tLlxuICAgKiBAcGFyYW0gc291cmNlU2l6ZSBUaGUgc2l6ZSBvZiB0aGUgY29weSB3aW5kb3cuXG4gICAqIEBwYXJhbSBkZXN0IFRoZSBkZXN0aW5hdGlvbiBtYXRyaXggdG8gY29weSB0by5cbiAgICogQHBhcmFtIGRlc3RCZWdpbiBUaGUgY29vcmRpbmF0ZXMgaW4gYGRlc3RgIHRvIGNvcHkgdG8uXG4gICAqIEBwYXJhbSBkZXN0U2l6ZSBUaGUgc2l6ZSBvZiB0aGUgZGVzdGluYXRpb24gd2luZG93LlxuICAgKi9cbiAgY29weTJEKFxuICAgICAgc291cmNlOiBBcnJheTJELCBzb3VyY2VCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHNvdXJjZVNpemU6IFtudW1iZXIsIG51bWJlcl0sIGRlc3Q6IEFycmF5MkQsIGRlc3RCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIGRlc3RTaXplOiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHNvdXJjZUJlZ2luWzBdICsgc291cmNlU2l6ZVswXSA8PSBzb3VyY2Uuc2hhcGVbMF0gJiZcbiAgICAgICAgICAgIHNvdXJjZUJlZ2luWzFdICsgc291cmNlU2l6ZVsxXSA8PSBzb3VyY2Uuc2hhcGVbMV0sXG4gICAgICAgIGBFcnJvciBpbiBjb3B5MkQ6IHJlcXVlc3RlZCBzb3VyY2Ugc3RhcnQgcG9zaXRpb24gJHtzb3VyY2VCZWdpbn0gYCArXG4gICAgICAgICAgICBgYW5kIHNvdXJjZSBzaXplICR7c291cmNlU2l6ZX0gd291bGQgb3ZlcmZsb3cgc291cmNlIE5EQXJyYXlgICtcbiAgICAgICAgICAgIGBvZiBzaGFwZSAke3NvdXJjZS5zaGFwZX0uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGRlc3RCZWdpblswXSArIGRlc3RTaXplWzBdIDw9IGRlc3Quc2hhcGVbMF0gJiZcbiAgICAgICAgICAgIGRlc3RCZWdpblsxXSArIGRlc3RTaXplWzFdIDw9IGRlc3Quc2hhcGVbMV0sXG4gICAgICAgIGBFcnJvciBpbiBjb3B5MkQ6IHJlcXVlc3RlZCBkZXN0IHN0YXJ0IHBvc2l0aW9uICR7ZGVzdEJlZ2lufSBgICtcbiAgICAgICAgICAgIGBhbmQgc291cmNlIHNpemUgJHtkZXN0U2l6ZX0gd291bGQgb3ZlcmZsb3cgZGVzdCBOREFycmF5IG9mYCArXG4gICAgICAgICAgICBgc2hhcGUgJHtkZXN0LnNoYXBlfS5gKTtcbiAgICBjb3B5MmRfdXRpbC52YWxpZGF0ZVNoYXBlcyhzb3VyY2VTaXplLCBkZXN0U2l6ZSk7XG5cbiAgICByZXR1cm4gdGhpcy5jb3B5MkRJbnRlcm5hbChcbiAgICAgICAgc291cmNlLCBzb3VyY2VCZWdpbiwgc291cmNlU2l6ZSwgZGVzdCwgZGVzdEJlZ2luLCBkZXN0U2l6ZSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvcHkyREludGVybmFsKFxuICAgICAgc291cmNlOiBBcnJheTJELCBzb3VyY2VCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHNvdXJjZVNpemU6IFtudW1iZXIsIG51bWJlcl0sIGRlc3Q6IEFycmF5MkQsIGRlc3RCZWdpbjogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIGRlc3RTaXplOiBbbnVtYmVyLCBudW1iZXJdKTogdm9pZDtcblxuICAvKipcbiAgICogQ29uY2F0ZW5hdGVzIHR3byAzRCBuZGFycmF5cyBhbG9uZyBhIGdpdmVuIGF4aXMuXG4gICAqXG4gICAqIEZvciBleGFtcGxlLCBpZjpcbiAgICogQTogc2hhcGUoMiwgMSwgMykgPSB8IHIxLCBnMSwgYjEgfFxuICAgKiAgICAgICAgICAgICAgICAgICAgIHwgcjIsIGcyLCBiMiB8XG4gICAqXG4gICAqIEI6IHNoYXBlKDIsIDEsIDMpID0gfCByMywgZzMsIGIzIHxcbiAgICogICAgICAgICAgICAgICAgICAgICB8IHI0LCBnNCwgYjQgfFxuICAgKlxuICAgKiBDID0gY29uY2F0M0QoQSwgQiwgYXhpcylcbiAgICpcbiAgICogaWYgYXhpcyA9IDA6XG4gICAqIEM6IHNoYXBlKDQsIDEsIDMpID0gfCByMSwgZzEsIGIxIHxcbiAgICogICAgICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIgfFxuICAgKiAgICAgICAgICAgICAgICAgICAgIHwgcjMsIGczLCBiMyB8XG4gICAqICAgICAgICAgICAgICAgICAgICAgfCByNCwgZzQsIGI0IHxcbiAgICpcbiAgICogaWYgYXhpcyA9IDE6XG4gICAqIEM6IHNoYXBlKDIsIDIsIDMpID0gfCByMSwgZzEsIGIxLCByMywgZzMsIGIzIHxcbiAgICogICAgICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIsIHI0LCBnNCwgYjQgfFxuICAgKlxuICAgKiBpZiBheGlzID0gMjpcbiAgICogQyA9IHNoYXBlKDIsIDEsIDYpID0gfCByMSwgZzEsIGIxLCByMywgZzMsIGIzIHxcbiAgICogICAgICAgICAgICAgICAgICAgICAgfCByMiwgZzIsIGIyLCByNCwgZzQsIGI0IHxcbiAgICpcbiAgICogQHBhcmFtIG5kYXJyYXkxIFRoZSBmaXJzdCBhcnJheSB0byBjb25jYXQuXG4gICAqIEBwYXJhbSBuZGFycmF5MiBUaGUgc2Vjb25kIGFycmF5IHRvIGNvbmF0LlxuICAgKiBAcGFyYW0gYXhpcyBUaGUgYXhpcyB0byBjb25jYXRlIGFsb25nLlxuICAgKi9cbiAgY29uY2F0M0QobmRhcnJheTE6IEFycmF5M0QsIG5kYXJyYXkyOiBBcnJheTNELCBheGlzOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICBjb25jYXQzZF91dGlsLmFzc2VydENvbmNhdDNEU2hhcGVzTWF0Y2goXG4gICAgICAgIG5kYXJyYXkxLnNoYXBlLCBuZGFycmF5Mi5zaGFwZSwgYXhpcywgJ0Vycm9yIGluIGNvbmNhdDNkOiAnKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmNvbmNhdDNESW50ZXJuYWwobmRhcnJheTEsIG5kYXJyYXkyLCBheGlzKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGNvbmNhdDNESW50ZXJuYWwoXG4gICAgICBuZGFycmF5MTogQXJyYXkzRCwgbmRhcnJheTI6IEFycmF5M0QsIGF4aXM6IG51bWJlcik6IEFycmF5M0Q7XG5cbiAgLy8vLy8vLy8vLy8vLy8vLy8vL1xuICAvLyBSZWR1Y3Rpb24gb3BzIC8vXG4gIC8vLy8vLy8vLy8vLy8vLy8vLy9cblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIHRoZSBsb2coc3VtKGUgXiB4KSkgZm9yIGVhY2ggeCBpbiB0aGUgaW5wdXQgbmRhcnJheS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkgdG8gY29tcHV0ZSB0aGUgbG9nU3VtRXhwIG92ZXIuXG4gICAqL1xuICBsb2dTdW1FeHAobmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5sb2dTdW1FeHBJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGxvZ1N1bUV4cEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXI7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBzdW0gb2YgYWxsIHRoZSBlbnRyaWVzIGluIHRoZSBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheSB0byBjb21wdXRlIHRoZSBzdW0gb3Zlci5cbiAgICovXG4gIHN1bShuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnN1bUludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc3VtSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGZsYXR0ZW5lZCBpbmRleCBvZiB0aGUgbWluaW11bSBlbGVtZW50IGluIHRoZSBuZGFycmF5LlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIGFyZ01pbihuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmFyZ01pbkludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgYXJnTWluSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGZsYXR0ZW5lZCBpbmRleCBvZiB0aGUgbWF4aW11bSBlbGVtZW50IGluIHRoZSBuZGFycmF5LlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIGFyZ01heChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmFyZ01heEludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgYXJnTWF4SW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogUmV0dXJucyBhIDEgaWYgdGhlIGFyZ01heCBvZiB4MSBhbmQgeDIgYXJlIHRoZSBzYW1lLCBvdGhlcndpc2UgMC5cbiAgICogQHBhcmFtIHgxIFRoZSBmaXJzdCBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0geDIgVGhlIHNlY29uZCBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgYXJnTWF4RXF1YWxzKHgxOiBOREFycmF5LCB4MjogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaCh4MS5zaGFwZSwgeDIuc2hhcGUsICdFcnJvciBpbiBhcmdNYXhFcXVhbHM6ICcpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuYXJnTWF4RXF1YWxzSW50ZXJuYWwoeDEsIHgyKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGFyZ01heEVxdWFsc0ludGVybmFsKHgxOiBOREFycmF5LCB4MjogTkRBcnJheSk6IFNjYWxhcjtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIHRvcCBLIHZhbHVlcyBhbmQgZmxhdHRlbmVkIGluZGljZXMuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0gayBIb3cgbWFueSB0b3AgdmFsdWVzIHRvIGNvbXB1dGUuXG4gICAqL1xuICB0b3BLKG5kYXJyYXk6IE5EQXJyYXksIGs6IG51bWJlcik6IHt2YWx1ZXM6IEFycmF5MUQsIGluZGljZXM6IEFycmF5MUR9IHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgayA8PSBuZGFycmF5LnNpemUsXG4gICAgICAgIGBFcnJvciBpbiB0b3BLOiBrIHZhbHVlICgke2t9KSBtdXN0IGJlIGxlc3MgdGhhbiBzaXplIG9mIGlucHV0IGAgK1xuICAgICAgICAgICAgYG5kYXJyYXksIGdvdCBzaGFwZSAke25kYXJyYXkuc2hhcGV9LmApO1xuICAgIGNvbnN0IHJlc3VsdCA9IHRoaXMudG9wS0ludGVybmFsKG5kYXJyYXksIGspO1xuICAgIHRoaXMudHJhY2socmVzdWx0LnZhbHVlcyk7XG4gICAgdGhpcy50cmFjayhyZXN1bHQuaW5kaWNlcyk7XG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgdG9wS0ludGVybmFsKG5kYXJyYXk6IE5EQXJyYXksIGs6IG51bWJlcik6XG4gICAgICB7dmFsdWVzOiBBcnJheTFELCBpbmRpY2VzOiBBcnJheTFEfTtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIG1pbmltdW0gdmFsdWUgZnJvbSB0aGUgaW5wdXQuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgbWluKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMubWluSW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBtaW5JbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgbWF4aW11bSB2YWx1ZSBmcm9tIHRoZSBpbnB1dC5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICBtYXgobmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5tYXhJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG1heEludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXI7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBzb2Z0bWF4IG5vcm1hbGl6ZWQgdmVjdG9yIGZyb20gdGhlIGlucHV0IHZlY3Rvci5cbiAgICogQHBhcmFtIHggVGhlIGlucHV0IHZlY3Rvci5cbiAgICovXG4gIHNvZnRtYXgoeDogQXJyYXkxRCk6IEFycmF5MUQge1xuICAgIHJldHVybiB0aGlzLnNjb3BlKCgpID0+IHtcbiAgICAgIC8vIERvIGl0IGluIGxvZyBzcGFjZSBmb3IgbnVtZXJpY2FsIHN0YWJpbGl0eS5cbiAgICAgIC8vIGV4cChYIC0gbG9nU3VtRXhwKFgpKVxuICAgICAgY29uc3QgbHNlID0gdGhpcy5sb2dTdW1FeHAoeCk7XG4gICAgICBjb25zdCBsb2dSZXN1bHQgPSB0aGlzLmFycmF5TWludXNTY2FsYXIoeCwgbHNlKTtcbiAgICAgIHJldHVybiB0aGlzLmV4cChsb2dSZXN1bHQpO1xuICAgIH0pO1xuICB9XG5cbiAgLy8vLy8vLy8vLy8vLy8vLy8vLy8vL1xuICAvLyBFbGVtZW50LXdpc2Ugb3BzIC8vXG4gIC8vLy8vLy8vLy8vLy8vLy8vLy8vLy9cblxuICAvKipcbiAgICogU3dpdGNoZXMgZGltZW5zaW9ucyBvZiB0aGUgaW5wdXQgTkRBcnJheS5cbiAgICogQHBhcmFtIGEgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqIEBwYXJhbSBuZXdEaW0gVGhlIG5ldyBpbmRpY2VzIHRoYXQgZGVmaW5lIHdoaWNoIHNoYXBlcyB2YWx1ZXMgdG8gc3dpdGNoLlxuICAgKi9cbiAgc3dpdGNoRGltPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBuZXdEaW06IG51bWJlcltdKTogVCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGEucmFuayA9PT0gbmV3RGltLmxlbmd0aCxcbiAgICAgICAgYEVycm9yIGluIHN3aXRjaERpbTogbGVuZ3RoIG9mIGlucHV0IHNoYXBlICR7YS5zaGFwZX0gYCArXG4gICAgICAgICAgICBgbXVzdCBtYXRjaCBzaXplIG9mIG5ld0RpbSBhcnJheSAke25ld0RpbX0uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zd2l0Y2hEaW1JbnRlcm5hbChhLCBuZXdEaW0pKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc3dpdGNoRGltSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KFxuICAgICAgYTogVCwgbmV3RGltOiBudW1iZXJbXSk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgc2NhbGFyIHBsdXMgTkRBcnJheSwgYyArIEEuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgYyBpbiBjICsgQS5cbiAgICogQHBhcmFtIGEgVGhlIE5EQXJyYXkgQSBpbiBjICsgQS5cbiAgICovXG4gIHNjYWxhclBsdXNBcnJheTxUIGV4dGVuZHMgTkRBcnJheT4oYzogU2NhbGFyLCBhOiBUKTogVCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGMuc2l6ZSA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIHNjYWxhclBsdXNBcnJheTogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMuYWRkKGMsIGEpIGFzIFQ7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgYSBzY2FsYXIgbWludXMgTkRBcnJheSwgYyAtIEEuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgYyBpbiBjIC0gQS5cbiAgICogQHBhcmFtIGEgVGhlIE5EQXJyYXkgQSBpbiBjIC0gQS5cbiAgICovXG4gIHNjYWxhck1pbnVzQXJyYXk8VCBleHRlbmRzIE5EQXJyYXk+KGM6IFNjYWxhciwgYTogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBzY2FsYXJNaW51c0FycmF5OiBmaXJzdCBhcmd1bWVudCBtdXN0IGJlIHJhbmsgMCwgYnV0IGdvdCBgICtcbiAgICAgICAgICAgIGByYW5rICR7Yy5yYW5rfS5gKTtcbiAgICByZXR1cm4gdGhpcy5zdWIoYywgYSkgYXMgVDtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBBIC0gYy4gQSBpcyBOREFycmF5LCBjIGlzIFNjYWxhci5cbiAgICogQHBhcmFtIGEgVGhlIE5EQXJyYXkgQSBpbiBBIC0gYy5cbiAgICogQHBhcmFtIGMgVGhlIFNjYWxhciBjIGluIEEgLSBjLlxuICAgKi9cbiAgYXJyYXlNaW51c1NjYWxhcjxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYzogU2NhbGFyKTogVCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGMuc2l6ZSA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIGFycmF5TWludXNTY2FsYXI6IHNlY29uZCBhcmd1bWVudCBtdXN0IGJlIHJhbmsgMCwgYnV0IGAgK1xuICAgICAgICAgICAgYGdvdCByYW5rICR7Yy5yYW5rfS5gKTtcbiAgICByZXR1cm4gdGhpcy5zdWIoYSwgYykgYXMgVDtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21wdXRlcyAtMSAqIEEgZWxlbWVudC13aXNlLlxuICAgKiBAcGFyYW0gYSBUaGUgaW5wdXQgYXJyYXkuXG4gICAqL1xuICBuZWc8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLm5lZ0ludGVybmFsKGEpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgbmVnSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBBZGRzIHR3byBOREFycmF5cyBlbGVtZW50LXdpc2UsIEEgKyBCLiBTdXBwb3J0cyBicm9hZGNhc3RpbmcuXG4gICAqIEZvciBhIHN0cmljdGVyIHZlcnNpb24gd2l0aG91dCBicm9hZGNhc3RpbmcgdXNlIG1hdGguYWRkU3RyaWN0KCkuXG4gICAqXG4gICAqIEBwYXJhbSBhIFRoZSBmaXJzdCBOREFycmF5IHRvIGFkZCBlbGVtZW50LXdpc2UuXG4gICAqIEBwYXJhbSBiIFRoZSBzZWNvbmQgTkRBcnJheSB0byBhZGQgZWxlbWVudC13aXNlLlxuICAgKi9cbiAgYWRkKGE6IE5EQXJyYXksIGI6IE5EQXJyYXkpOiBOREFycmF5IHtcbiAgICB1dGlsLmFzc2VydEFuZEdldEJyb2FkY2FzdGVkU2hhcGUoYS5zaGFwZSwgYi5zaGFwZSk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5hZGRJbnRlcm5hbChhLCBiKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGFkZEludGVybmFsKGE6IE5EQXJyYXksIGI6IE5EQXJyYXkpOiBOREFycmF5O1xuXG4gIC8qKlxuICAgKiBBZGRzIHR3byBOREFycmF5cyBlbGVtZW50LXdpc2UsIEEgKyBCLiBJbnB1dHMgbXVzdFxuICAgKiBiZSB0aGUgc2FtZSBzaGFwZS4gRm9yIGJyb2FkY2FzdGluZyBzdXBwb3J0LCB1c2UgbWF0aC5hZGQoKSBpbnN0ZWFkLlxuICAgKlxuICAgKiBAcGFyYW0gYSBUaGUgZmlyc3QgTkRBcnJheSB0byBtdWx0aXBseSBlbGVtZW50LXdpc2UuXG4gICAqIEBwYXJhbSBiIFRoZSBzZWNvbmQgTkRBcnJheSB0byBtdWx0aXBseSBlbGVtZW50LXdpc2UuXG4gICAqL1xuICBhZGRTdHJpY3Q8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydFNoYXBlc01hdGNoKGEuc2hhcGUsIGIuc2hhcGUsICdFcnJvciBpbiBhZGRTdHJpY3Q6ICcpO1xuICAgIHJldHVybiB0aGlzLmFkZChhLCBiKSBhcyBUO1xuICB9XG5cbiAgLyoqXG4gICAqIFN1YnRyYWN0cyB0d28gTkRBcnJheXMgZWxlbWVudC13aXNlLCBBIC0gQi4gU3VwcG9ydHMgYnJvYWRjYXN0aW5nLlxuICAgKiBGb3IgYSBzdHJpY3RlciB2ZXJzaW9uIHdpdGhvdXQgYnJvYWRjYXN0aW5nIHVzZSBtYXRoLnN1YlN0cmljdCgpLlxuICAgKlxuICAgKiBAcGFyYW0gYSBUaGUgZmlyc3QgTkRBcnJheSB0byBzdWJ0cmFjdCBlbGVtZW50LXdpc2UuXG4gICAqIEBwYXJhbSBiIFRoZSBzZWNvbmQgTkRBcnJheSB0byBzdWJ0cmFjdCBlbGVtZW50LXdpc2UuXG4gICAqL1xuICBzdWIoYTogTkRBcnJheSwgYjogTkRBcnJheSk6IE5EQXJyYXkge1xuICAgIHV0aWwuYXNzZXJ0QW5kR2V0QnJvYWRjYXN0ZWRTaGFwZShhLnNoYXBlLCBiLnNoYXBlKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnN1YkludGVybmFsKGEsIGIpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc3ViSW50ZXJuYWwoYTogTkRBcnJheSwgYjogTkRBcnJheSk6IE5EQXJyYXk7XG5cbiAgLyoqXG4gICAqIFN1YnRyYWN0cyB0d28gTkRBcnJheXMgZWxlbWVudC13aXNlLCBBIC0gQi4gSW5wdXRzIG11c3RcbiAgICogYmUgdGhlIHNhbWUgc2hhcGUuIEZvciBicm9hZGNhc3Rpbmcgc3VwcG9ydCwgdXNlIG1hdGguc3ViKCkgaW5zdGVhZC5cbiAgICpcbiAgICogQHBhcmFtIGEgVGhlIGZpcnN0IE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKiBAcGFyYW0gYiBUaGUgc2Vjb25kIE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKi9cbiAgc3ViU3RyaWN0PFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChhLnNoYXBlLCBiLnNoYXBlLCAnRXJyb3IgaW4gc3ViU3RyaWN0OiAnKTtcbiAgICByZXR1cm4gdGhpcy5zdWIoYSwgYikgYXMgVDtcbiAgfVxuXG4gIC8qKlxuICAgKiBNdWx0aXBsaWVzIHR3byBOREFycmF5cyBlbGVtZW50LXdpc2UsIEEgKiBCLiBTdXBwb3J0cyBicm9hZGNhc3RpbmcuXG4gICAqIEZvciBhIHN0cmljdGVyIHZlcnNpb24gd2l0aG91dCBicm9hZGNhc3RpbmcgdXNlIG1hdGgubXVsdGlwbHlTdHJpY3QoKS5cbiAgICpcbiAgICogQHBhcmFtIGEgVGhlIGZpcnN0IE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKiBAcGFyYW0gYiBUaGUgc2Vjb25kIE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKi9cbiAgbXVsdGlwbHkoYTogTkRBcnJheSwgYjogTkRBcnJheSk6IE5EQXJyYXkge1xuICAgIHV0aWwuYXNzZXJ0QW5kR2V0QnJvYWRjYXN0ZWRTaGFwZShhLnNoYXBlLCBiLnNoYXBlKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLm11bHRpcGx5SW50ZXJuYWwoYSwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBtdWx0aXBseUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVDtcblxuICAvKipcbiAgICogQGRlcHJlY2F0ZWQgVXNlIG1hdGgubXVsdGlwbHlTdHJpY3QoKSBpbnN0ZWFkLlxuICAgKi9cbiAgZWxlbWVudFdpc2VNdWw8VCBleHRlbmRzIE5EQXJyYXk+KGE6IFQsIGI6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy5tdWx0aXBseVN0cmljdChhLCBiKTtcbiAgfVxuXG4gIC8qKlxuICAgKiBNdWx0aXBsaWVzIHR3byBOREFycmF5cyBlbGVtZW50LXdpc2UsIEEgKiBCLiBJbnB1dHMgbXVzdFxuICAgKiBiZSB0aGUgc2FtZSBzaGFwZS4gRm9yIGJyb2FkY2FzdGluZyBzdXBwb3J0LCB1c2UgbWF0aC5tdWx0aXBseSgpIGluc3RlYWQuXG4gICAqXG4gICAqIEBwYXJhbSBhIFRoZSBmaXJzdCBOREFycmF5IHRvIG11bHRpcGx5IGVsZW1lbnQtd2lzZS5cbiAgICogQHBhcmFtIGIgVGhlIHNlY29uZCBOREFycmF5IHRvIG11bHRpcGx5IGVsZW1lbnQtd2lzZS5cbiAgICovXG4gIG11bHRpcGx5U3RyaWN0PFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChhLnNoYXBlLCBiLnNoYXBlLCAnRXJyb3IgaW4gbXVsdGlwbHlTdHJpY3Q6ICcpO1xuICAgIHJldHVybiB0aGlzLm11bHRpcGx5KGEsIGIpIGFzIFQ7XG4gIH1cblxuICAvKipcbiAgICogRGl2aWRlcyB0d28gTkRBcnJheXMgZWxlbWVudC13aXNlLCBBIC8gQi4gU3VwcG9ydHMgYnJvYWRjYXN0aW5nLlxuICAgKiBGb3IgYSBzdHJpY3RlciB2ZXJzaW9uIHdpdGhvdXQgYnJvYWRjYXN0aW5nIHVzZSBtYXRoLmRpdmlkZVN0cmljdCgpLlxuICAgKlxuICAgKiBAcGFyYW0gYSBUaGUgZmlyc3QgTkRBcnJheSB0byBkaXZpZGUgZWxlbWVudC13aXNlLlxuICAgKiBAcGFyYW0gYiBUaGUgc2Vjb25kIE5EQXJyYXkgdG8gZGl2aWRlIGVsZW1lbnQtd2lzZS5cbiAgICovXG4gIGRpdmlkZShhOiBOREFycmF5LCBiOiBOREFycmF5KTogTkRBcnJheSB7XG4gICAgdXRpbC5hc3NlcnRBbmRHZXRCcm9hZGNhc3RlZFNoYXBlKGEuc2hhcGUsIGIuc2hhcGUpO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuZGl2aWRlSW50ZXJuYWwoYSwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBkaXZpZGVJbnRlcm5hbChhOiBOREFycmF5LCBiOiBOREFycmF5KTogTkRBcnJheTtcblxuICAvKipcbiAgICogRGl2aWRlcyB0d28gTkRBcnJheXMgZWxlbWVudC13aXNlLCBBIC8gQi4gSW5wdXRzIG11c3RcbiAgICogYmUgdGhlIHNhbWUgc2hhcGUuIEZvciBicm9hZGNhc3Rpbmcgc3VwcG9ydCwgdXNlIG1hdGguZGl2aWRlKCkgaW5zdGVhZC5cbiAgICpcbiAgICogQHBhcmFtIGEgVGhlIGZpcnN0IE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKiBAcGFyYW0gYiBUaGUgc2Vjb25kIE5EQXJyYXkgdG8gbXVsdGlwbHkgZWxlbWVudC13aXNlLlxuICAgKi9cbiAgZGl2aWRlU3RyaWN0PFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChhLnNoYXBlLCBiLnNoYXBlLCAnRXJyb3IgaW4gZGl2aWRlU3RyaWN0OiAnKTtcbiAgICByZXR1cm4gdGhpcy5kaXZpZGUoYSwgYikgYXMgVDtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBhIHNjYWxhciBkaXZpZGVkIGJ5IGFuIE5EQXJyYXksIGJyb2FkY2FzdGVkIG92ZXIgdGhlIE5EQXJyYXksIGMgL1xuICAgKiBBLlxuICAgKiBAcGFyYW0gYyBUaGUgc2NhbGFyIHZhbHVlIGluIGMgLyBBLlxuICAgKiBAcGFyYW0gYSBUaGUgTkRBcnJheSB2YWx1ZSBpbiBjIC8gQS5cbiAgICovXG4gIHNjYWxhckRpdmlkZWRCeUFycmF5PFQgZXh0ZW5kcyBOREFycmF5PihjOiBTY2FsYXIsIGE6IFQpOiBUIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYy5zaXplID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gc2NhbGFyRGl2aWRlZEJ5QXJyYXk6IGZpcnN0IGFyZ3VtZW50IG11c3QgYmUgcmFuayAwLCBidXQgYCArXG4gICAgICAgICAgICBgZ290IE5EQXJyYXkgb2YgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMuZGl2aWRlKGMsIGEpIGFzIFQ7XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgYW4gTkRBcnJheSBkaXZpZGVkIGJ5IGEgc2NhbGFyLCBicm9hZGNhc3RlZCBvdmVyIHRoZSBOREFycmF5LCBBIC9cbiAgICogYy5cbiAgICogQHBhcmFtIGEgVGhlIE5EQXJyYXkgdmFsdWUgaW4gQSAvIGMuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgdmFsdWUgaW4gQSAvIGMuXG4gICAqL1xuICBhcnJheURpdmlkZWRCeVNjYWxhcjxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYzogU2NhbGFyKTogVCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGMuc2l6ZSA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIGFycmF5RGl2aWRlZEJ5U2NhbGFyOiBzZWNvbmQgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGAgK1xuICAgICAgICAgICAgYGJ1dCBnb3QgTkRBcnJheSBvZiByYW5rICR7Yy5yYW5rfS5gKTtcbiAgICByZXR1cm4gdGhpcy5kaXZpZGUoYSwgYykgYXMgVDtcbiAgfVxuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBleHBvbmVudGlhbCBvZiB0aGUgaW5wdXQgTkRBcnJheSBlbGVtZW50LXdpc2UuIHkgPSBlIF4geFxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIGV4cDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuZXhwSW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBleHBJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIG5hdHVyYWwgbG9nYXJpdGhtIG9mIHRoZSBpbnB1dCBOREFycmF5IGVsZW1lbnQtd2lzZS4geSA9IGxuKHgpXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgbG9nPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5sb2dJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IGxvZ0ludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgcmVjdGlmaWVkIGxpbmVhciBlbGVtZW50LXdpc2UsIG1heCh4LCAwKS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICByZWx1PFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5yZWx1SW50ZXJuYWwobmRhcnJheSkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCByZWx1SW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBzaWdtb2lkIGVsZW1lbnQtd2lzZSwgeSA9IDEgLyAoMSArIGV4cCgteCkpLlxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIHNpZ21vaWQ8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnNpZ21vaWRJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHNpZ21vaWRJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGh5cGVyYm9saWMgdGFuZ2VudCBvZiB0aGUgaW5wdXQgTkRBcnJheSBlbGVtZW50LXdpc2UuXG4gICAqIEBwYXJhbSBuZGFycmF5IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKi9cbiAgdGFuaDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMudGFuaEludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgdGFuaEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgc2luIG9mIHRoZSBpbnB1dCBOREFycmF5IGVsZW1lbnQtd2lzZSwgeSA9IHNpbih4KS5cbiAgICogQHBhcmFtIG5kYXJyYXkgVGhlIGlucHV0IE5EQXJyYXkuXG4gICAqL1xuICBzaW48VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnNpbkludGVybmFsKG5kYXJyYXkpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3Qgc2luSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBzdGVwIG9mIHRoZSBpbnB1dCBOREFycmF5IGVsZW1lbnQtd2lzZSwgeSA9IDEgaWYgeCA+IDAgfCAwIGlmIHggPD1cbiAgICogMFxuICAgKiBAcGFyYW0gbmRhcnJheSBUaGUgaW5wdXQgTkRBcnJheS5cbiAgICovXG4gIHN0ZXA8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLnN0ZXBJbnRlcm5hbChuZGFycmF5KSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IHN0ZXBJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQ7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIGEgc2NhbGVkIGFycmF5IGFkZCBvcGVyYXRpb24sIGMxICogQSArIGMyICogQi5cbiAgICogQHBhcmFtIGMxIFRoZSBmaXJzdCBzY2FsYXIgaW4gdGhlIHNjYWxlZCBhcnJheSBhZGQgY29tcHV0YXRpb24uXG4gICAqIEBwYXJhbSBhIFRoZSBmaXJzdCBOREFycmF5IGluIHRoZSBzY2FsZWQgYXJyYXkgYWRkIGNvbXB1dGF0aW9uLlxuICAgKiBAcGFyYW0gYzIgVGhlIHNlY29uZCBzY2FsYXIgaW4gdGhlIHNjYWxlZCBhcnJheSBhZGQgY29tcHV0YXRpb24uXG4gICAqIEBwYXJhbSBjYiBUaGUgc2Vjb25kIE5EQXJyYXkgaW4gdGhlIHNjYWxlZCBhcnJheSBhZGQgY29tcHV0YXRpb24uXG4gICAqL1xuICBzY2FsZWRBcnJheUFkZDxUIGV4dGVuZHMgTkRBcnJheT4oYzE6IFNjYWxhciwgYTogVCwgYzI6IFNjYWxhciwgYjogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjMS5zaXplID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gc2NhbGVkQXJyYXlBZGQ6IGZpcnN0IGFyZ3VtZW50IG11c3QgcmFuayAwLCBidXQgZ290IGAgK1xuICAgICAgICAgICAgYCByYW5rICR7YzEucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGMyLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBzY2FsZWRBcnJheUFkZDogdGhpcmQgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgTkRBcnJheSBvZiByYW5rICR7YzIucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChhLnNoYXBlLCBiLnNoYXBlLCAnRXJyb3IgaW4gc2NhbGVkQXJyYXlBZGQ6ICcpO1xuXG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5zY2FsZWRBcnJheUFkZEludGVybmFsKGMxLCBhLCBjMiwgYikpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBzY2FsZWRBcnJheUFkZEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIGMxOiBTY2FsYXIsIGE6IFQsIGMyOiBTY2FsYXIsIGI6IFQpOiBUO1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyBhIHNjYWxhciB0aW1lcyBhcnJheSBvcGVyYXRpb24gYnJvYWRjYXN0ZWQgb3ZlciB0aGUgTkRBcnJheSwgYyAqXG4gICAqIEEuXG4gICAqIEBwYXJhbSBjIFRoZSBzY2FsYXIgaW4gdGhlIG9wZXJhdGlvbi5cbiAgICogQHBhcmFtIEEgdGhlIE5EQXJyYXkgaW4gdGhlIG9wZXJhdGlvbiB0aGF0IHdpbGwgYmUgYnJvYWRjYXN0ZWQgb3Zlci5cbiAgICovXG4gIHNjYWxhclRpbWVzQXJyYXk8VCBleHRlbmRzIE5EQXJyYXk+KGM6IFNjYWxhciwgYTogVCk6IFQge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBjLnNpemUgPT09IDEsXG4gICAgICAgIGBFcnJvciBpbiBhcnJheURpdmlkZWRCeVNjYWxhcjogZmlyc3QgYXJndW1lbnQgbXVzdCBiZSByYW5rIDAsIGJ1dCBgICtcbiAgICAgICAgICAgIGBnb3QgcmFuayAke2MucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMubXVsdGlwbHkoYywgYSkgYXMgVDtcbiAgfVxuXG4gIC8qKlxuICAgKiBAZGVwcmVjYXRlZCBVc2UgbWF0aC5tdWx0aXBseSgpIGluc3RlYWQuXG4gICAqL1xuICBlbGVtZW50V2lzZU11bEJyb2FkY2FzdChhOiBBcnJheTJELCBiOiBBcnJheTJEKTogQXJyYXkyRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGEucmFuayA9PT0gMixcbiAgICAgICAgYEVycm9yIGluIGVsZW1lbnRXaXNlTXVsQnJvYWRjYXN0OiBmaXJzdCBhcmd1bWVudCBtdXN0IGJlIGAgK1xuICAgICAgICAgICAgYHJhbmsgMiwgYnV0IGdvdCByYW5rICR7YS5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgYi5yYW5rID09PSAyLFxuICAgICAgICBgRXJyb3IgaW4gZWxlbWVudFdpc2VNdWxCcm9hZGNhc3Q6IHNlY29uZCBhcmd1bWVudCBtdXN0IGJlIGAgK1xuICAgICAgICAgICAgYHJhbmsgMiwgYnV0IGdvdCByYW5rICR7Yi5yYW5rfS5gKTtcbiAgICByZXR1cm4gdGhpcy5tdWx0aXBseShhLCBiKSBhcyBBcnJheTJEO1xuICB9XG5cbiAgLy8vLy8vLy8vLy8vLy8vLy8vLy8vXG4gIC8vIENvbnZvbHV0aW9uIG9wcyAvL1xuICAvLy8vLy8vLy8vLy8vLy8vLy8vLy9cblxuICAvKipcbiAgICogQ29tcHV0ZXMgYSAyRCBjb252b2x1dGlvbiBvdmVyIHRoZSBpbnB1dCB4LlxuICAgKiBAcGFyYW0geCBUaGUgaW5wdXQgaW1hZ2UsIG11c3QgYmUgcmFuayAzLCBvZiBzaGFwZSBbcm93cywgY29scywgZGVwdGgxXS5cbiAgICogQHBhcmFtIHdlaWdodHMgVGhlIHdlaWdodHMgTkRBcnJheSwgbXVzdCBiZSByYW5rIDQsIG9mIHNoYXBlIFtmLCBmLCBkZXB0aDEsXG4gICAqIGRlcHRoMl0uXG4gICAqIEBwYXJhbSBiaWFzZXMgT3B0aW9uYWwgYmlhc2VzIE5EQXJyYXksIG11c3QgYmUgcmFuayAxIG9mIHNoYXBlIFtkZXB0aDJdLlxuICAgKiBAcGFyYW0gc3RyaWRlIFRoZSBzdHJpZGUgb2YgdGhlIGNvbnZvbHV0aW9uLlxuICAgKiBAcGFyYW0gemVyb1BhZCBUaGUgemVybyBwYWRkaW5nIG9mIGVhY2ggc2lkZSBvZiB0aGUgaW5wdXQgTkRBcnJheS4gV2lsbCBwYWRcbiAgICogZXF1YWxseSBvbiBhbGwgc2lkZXMuXG4gICAqL1xuICBjb252MmQoXG4gICAgICB4OiBBcnJheTNELCB3ZWlnaHRzOiBBcnJheTRELCBiaWFzZXM6IEFycmF5MUR8bnVsbCwgc3RyaWRlOiBudW1iZXIsXG4gICAgICB6ZXJvUGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkOiB4IG11c3QgYmUgcmFuayAzLCBidXQgZ290IHJhbmsgJHt4LnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB3ZWlnaHRzLnJhbmsgPT09IDQsXG4gICAgICAgIGBFcnJvciBpbiBjb252MmQ6IHdlaWdodHMgbXVzdCBiZSByYW5rIDQsIGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICAgIGAke3dlaWdodHMucmFua30uYCk7XG4gICAgaWYgKGJpYXNlcyAhPSBudWxsKSB7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICBiaWFzZXMucmFuayA9PT0gMSxcbiAgICAgICAgICBgRXJyb3IgaW4gY29udjJkOiBiaWFzZXMgbXVzdCBiZSByYW5rIDEsIGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICAgICAgYCR7Ymlhc2VzLnJhbmt9LmApO1xuICAgIH1cblxuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnNoYXBlWzJdID09PSB3ZWlnaHRzLnNoYXBlWzJdLFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkOiBkZXB0aCBvZiBpbnB1dCAoJHt4LnNoYXBlWzJdfSkgbXVzdCBtYXRjaCAgYCArXG4gICAgICAgICAgICBgaW5wdXQgZGVwdGggZm9yIHdlaWdodHMgJHt3ZWlnaHRzLnNoYXBlWzJdfS5gKTtcblxuXG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5jb252MmRJbnRlcm5hbCh4LCB3ZWlnaHRzLCBiaWFzZXMsIHN0cmlkZSwgemVyb1BhZCkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBjb252MmRJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIHdlaWdodHM6IEFycmF5NEQsIGJpYXNlczogQXJyYXkxRHxudWxsLCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHplcm9QYWQ6IG51bWJlcik6IEFycmF5M0Q7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSBiYWNrcHJvcCBvZiBhIDJEIGNvbnZvbHV0aW9uLlxuICAgKiBAcGFyYW0geCBUaGUgaW5wdXQgaW1hZ2UsIG11c3QgYmUgcmFuayAzLCBvZiBzaGFwZSBbeHJvd3MsIHhjb2xzLCBkZXB0aDFdLlxuICAgKiBAcGFyYW0gZHkgVGhlIGR5IGltYWdlLCBtdXN0IGJlIHJhbmsgMywgb2Ygc2hhcGUgW3lyb3dzLCB5Y29scywgZGVwdGgyXS5cbiAgICogQHBhcmFtIHdlaWdodHMgVGhlIHdlaWdodHMgTkRBcnJheSwgbXVzdCBiZSByYW5rIDQsIG9mIHNoYXBlIFtmLCBmLCBkZXB0aDEsXG4gICAqIGRlcHRoMl0uXG4gICAqIEBwYXJhbSBzdHJpZGUgVGhlIHN0cmlkZSBvZiB0aGUgb3JpZ2luYWwgY29udm9sdXRpb24uXG4gICAqIEBwYXJhbSBwYWQgVGhlIHBhZGRpbmcgb2YgdGhlIG9yaWdpbmFsIGNvbnZvbHV0aW9uLlxuICAgKi9cbiAgY29udjJkQmFja1Byb3AoXG4gICAgICB4OiBBcnJheTNELCBkeTogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgc3RyaWRlOiBudW1iZXIsXG4gICAgICBwYWQ6IG51bWJlcik6IHtkeDogQXJyYXkzRCwgZHc6IEFycmF5NEQsIGRiOiBBcnJheTFEfSB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZEJhY2tQcm9wOiB4IG11c3QgYmUgcmFuayAzLCBidXQgZ290IHNoYXBlIGAgK1xuICAgICAgICAgICAgYCR7eC5zaGFwZX0uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGR5LnJhbmsgPT09IDMsXG4gICAgICAgIGBFcnJvciBpbiBjb252MmRCYWNrUHJvcDogZHkgbXVzdCBiZSByYW5rIDMsIGJ1dCBnb3Qgc2hhcGUgYCArXG4gICAgICAgICAgICBgJHtkeS5zaGFwZX0uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHdlaWdodHMucmFuayA9PT0gNCxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZEJhY2tQcm9wOiB3ZWlnaHRzIG11c3QgYmUgcmFuayA0LCBidXQgZ290IHNoYXBlIGAgK1xuICAgICAgICAgICAgYCR7d2VpZ2h0cy5zaGFwZX0uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHguc2hhcGVbMl0gPT09IHdlaWdodHMuc2hhcGVbMl0sXG4gICAgICAgIGBFcnJvciBpbiBjb252MmRCYWNrUHJvcDogZGVwdGggb2YgeCAke3guc2hhcGVbMl19KSBtdXN0IGAgK1xuICAgICAgICAgICAgYG1hdGNoIGlucHV0IGRlcHRoIGZvciB3ZWlnaHRzICgke3dlaWdodHMuc2hhcGVbMl19LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBkeS5zaGFwZVsyXSA9PT0gd2VpZ2h0cy5zaGFwZVszXSxcbiAgICAgICAgYEVycm9yIGluIGNvbnYyZEJhY2tQcm9wOiBkZXB0aCBvZiBkeSAoJHtkeS5zaGFwZVsyXX0pIG11c3QgYCArXG4gICAgICAgICAgICBgbWF0Y2ggb3V0cHV0IGRlcHRoIGZvciB3ZWlnaHRzICgke3dlaWdodHMuc2hhcGVbM119KS5gKTtcblxuICAgIGNvbnN0IGJhY2twcm9wUmVzdWx0ID1cbiAgICAgICAgdGhpcy5jb252MmRCYWNrUHJvcEludGVybmFsKHgsIGR5LCB3ZWlnaHRzLCBzdHJpZGUsIHBhZCk7XG5cbiAgICB0aGlzLnRyYWNrKGJhY2twcm9wUmVzdWx0LmRiKTtcbiAgICB0aGlzLnRyYWNrKGJhY2twcm9wUmVzdWx0LmR3KTtcbiAgICB0aGlzLnRyYWNrKGJhY2twcm9wUmVzdWx0LmR4KTtcblxuICAgIHJldHVybiBiYWNrcHJvcFJlc3VsdDtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgY29udjJkQmFja1Byb3BJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIGR5OiBBcnJheTNELCB3ZWlnaHRzOiBBcnJheTRELCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHBhZDogbnVtYmVyKToge2R4OiBBcnJheTNELCBkdzogQXJyYXk0RCwgZGI6IEFycmF5MUR9O1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgdHJhbnNwb3NlZCAyRCBjb252b2x1dGlvbiBvZiBhbiBpbWFnZSwgYWxzbyBrbm93biBhcyBhXG4gICAqIGRlY29udm9sdXRpb24uXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgbXVzdCBiZSByYW5rIDMsIG9mIHNoYXBlIFt4cm93cywgeGNvbHMsIGRlcHRoMV0uXG4gICAqIEBwYXJhbSB3ZWlnaHRzIFRoZSB3ZWlnaHRzIE5EQXJyYXksIG11c3QgYmUgcmFuayA0LCBvZiBzaGFwZSBbZiwgZiwgZGVwdGgxLFxuICAgKiBkZXB0aDJdLlxuICAgKiBAcGFyYW0gYmlhc2VzIE9wdGlvbmFsIGJpYXNlcyBOREFycmF5LCBtdXN0IGJlIHJhbmsgMSBvZiBzaGFwZSBbZGVwdGgyXS5cbiAgICogQHBhcmFtIHN0cmlkZSBUaGUgc3RyaWRlIG9mIHRoZSBjb252b2x1dGlvbi5cbiAgICogQHBhcmFtIHBhZCBUaGUgcGFkZGluZyBvZiBlYWNoIHNpZGUgb2YgdGhlIGlucHV0IE5EQXJyYXkuIFdpbGwgcGFkIGVxdWFsbHlcbiAgICogb24gYWxsIHNpZGVzLlxuICAgKi9cbiAgY29udjJkVHJhbnNwb3NlKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhc2VzOiBBcnJheTFEfG51bGwsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgeC5yYW5rID09PSAzLFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkVHJhbnNwb3NlOiB4IG11c3QgYmUgcmFuayAzLCBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgICBgJHt4LnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB3ZWlnaHRzLnJhbmsgPT09IDQsXG4gICAgICAgIGBFcnJvciBpbiBjb252MmRUcmFuc3Bvc2U6IHdlaWdodHMgbXVzdCBiZSByYW5rIDQsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgICBgcmFuayAke3dlaWdodHMucmFua31gKTtcbiAgICBpZiAoYmlhc2VzICE9IG51bGwpIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIGJpYXNlcy5yYW5rID09PSAxLFxuICAgICAgICAgIGBFcnJvciBpbiBjb252MmRUcmFuc3Bvc2U6IGJpYXNlcyBtdXN0IGJlIHJhbmsgMSwgYnV0IGdvdCAnICtcbiAgICAgICAgICAgICAgJ3JhbmsgJHtiaWFzZXMucmFua30uYCk7XG4gICAgfVxuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnNoYXBlWzJdID09PSB3ZWlnaHRzLnNoYXBlWzNdLFxuICAgICAgICBgRXJyb3IgaW4gY29udjJkVHJhbnNwb3NlOiBkZXB0aCBvZiBpbnB1dCAoJHt4LnNoYXBlWzJdfSkgbXVzdCBgICtcbiAgICAgICAgICAgIGBtYXRjaCBpbnB1dCBkZXB0aCBmb3Igd2VpZ2h0cyAke3dlaWdodHMuc2hhcGVbM119LmApO1xuXG4gICAgcmV0dXJuIHRoaXMudHJhY2soXG4gICAgICAgIHRoaXMuY29udjJkVHJhbnNwb3NlSW50ZXJuYWwoeCwgd2VpZ2h0cywgYmlhc2VzLCBzdHJpZGUsIHBhZCkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBjb252MmRUcmFuc3Bvc2VJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIHdlaWdodHM6IEFycmF5NEQsIGJpYXNlczogQXJyYXkxRHxudWxsLCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHBhZDogbnVtYmVyKTogQXJyYXkzRDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIDJEIG1heCBwb29saW5nIG9mIGFuIGltYWdlLlxuICAgKiBAcGFyYW0geCBUaGUgaW5wdXQgaW1hZ2UsIG11c3QgYmUgcmFuayAzLlxuICAgKiBAcGFyYW0gZlNpemUgVGhlIGZpZWxkIHNpemUgb2YgdGhlIG1heCBwb29sLlxuICAgKiBAcGFyYW0gc3RyaWRlIFRoZSBzdHJpZGUgb2YgdGhlIG1heCBwb29sLlxuICAgKiBAcGFyYW0gcGFkIFRoZSBwYWRkaW5nIG9mIGVhY2ggc2lkZSBvZiB0aGUgaW5wdXQgTkRBcnJheS4gV2lsbCBwYWQgZXF1YWxseVxuICAgKiBvbiBhbGwgc2lkZXMuXG4gICAqL1xuICBtYXhQb29sKHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnJhbmsgPT09IDMsXG4gICAgICAgICdFcnJvciBpbiBtYXhQb29sOiB4IG11c3QgYmUgcmFuayAzIGJ1dCBnb3QgcmFuayAnICsgeC5yYW5rICsgJy4nKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLm1heFBvb2xJbnRlcm5hbCh4LCBmU2l6ZSwgc3RyaWRlLCBwYWQpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgbWF4UG9vbEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIGJhY2twcm9wIG9mIGEgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBkeSBUaGUgZHkgZXJyb3IuXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgbXVzdCBiZSByYW5rIDMuXG4gICAqIEBwYXJhbSBmU2l6ZSBUaGUgZmllbGQgc2l6ZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBzdHJpZGUgVGhlIHN0cmlkZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBwYWQgVGhlIHBhZGRpbmcgb2YgZWFjaCBzaWRlIG9mIHRoZSBpbnB1dCBOREFycmF5LiBXaWxsIHBhZCBlcXVhbGx5XG4gICAqIG9uIGFsbCBzaWRlcy5cbiAgICovXG4gIG1heFBvb2xCYWNrcHJvcChcbiAgICAgIGR5OiBBcnJheTNELCB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHBhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIGR5LnJhbmsgPT09IDMsXG4gICAgICAgIGBFcnJvciBpbiBtYXhQb29sQmFja3Byb3A6IGR5IG11c3QgYmUgcmFuayAzIGJ1dCBnb3QgcmFuayBgICtcbiAgICAgICAgICAgIGAke2R5LnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnJhbmsgPT09IDMsXG4gICAgICAgIGBFcnJvciBpbiBtYXhQb29sQmFja3Byb3A6IHggbXVzdCBiZSByYW5rIDMgYnV0IGdvdCByYW5rIGAgK1xuICAgICAgICAgICAgYCR7eC5yYW5rfS5gKTtcblxuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMubWF4UG9vbEJhY2twcm9wSW50ZXJuYWwoZHksIHgsIGZTaXplLCBzdHJpZGUsIHBhZCkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBtYXhQb29sQmFja3Byb3BJbnRlcm5hbChcbiAgICAgIGR5OiBBcnJheTNELCB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHBhZDogbnVtYmVyKTogQXJyYXkzRDtcblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIDJEIG1pbiBwb29saW5nIG9mIGFuIGltYWdlLlxuICAgKiBAcGFyYW0geCBUaGUgaW5wdXQgaW1hZ2UsIG11c3QgYmUgcmFuayAzLlxuICAgKiBAcGFyYW0gZlNpemUgVGhlIGZpZWxkIHNpemUgb2YgdGhlIG1heCBwb29sLlxuICAgKiBAcGFyYW0gc3RyaWRlIFRoZSBzdHJpZGUgb2YgdGhlIG1heCBwb29sLlxuICAgKiBAcGFyYW0gcGFkIFRoZSBwYWRkaW5nIG9mIGVhY2ggc2lkZSBvZiB0aGUgaW5wdXQgTkRBcnJheS4gV2lsbCBwYWQgZXF1YWxseVxuICAgKiBvbiBhbGwgc2lkZXMuXG4gICAqL1xuICBtaW5Qb29sKHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnJhbmsgPT09IDMsXG4gICAgICAgIGBFcnJvciBpbiBtaW5Qb29sOiB4IG11c3QgYmUgcmFuayAzIGJ1dCBnb3QgcmFuayAke3gucmFua30uYCk7XG4gICAgcmV0dXJuIHRoaXMudHJhY2sodGhpcy5taW5Qb29sSW50ZXJuYWwoeCwgZlNpemUsIHN0cmlkZSwgcGFkKSk7XG4gIH1cbiAgcHJvdGVjdGVkIGFic3RyYWN0IG1pblBvb2xJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcik6IEFycmF5M0Q7XG5cbiAgLyoqXG4gICAqIENvbXB1dGVzIHRoZSAyRCBhdmVyYWdlIHBvb2xpbmcgb2YgYW4gaW1hZ2UuXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBpbWFnZSwgbXVzdCBiZSByYW5rIDMuXG4gICAqIEBwYXJhbSBmU2l6ZSBUaGUgZmllbGQgc2l6ZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBzdHJpZGUgVGhlIHN0cmlkZSBvZiB0aGUgbWF4IHBvb2wuXG4gICAqIEBwYXJhbSBwYWQgVGhlIHBhZGRpbmcgb2YgZWFjaCBzaWRlIG9mIHRoZSBpbnB1dCBOREFycmF5LiBXaWxsIHBhZCBlcXVhbGx5XG4gICAqIG9uIGFsbCBzaWRlcy5cbiAgICovXG4gIGF2Z1Bvb2woeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIGF2Z1Bvb2w6IHggbXVzdCBiZSByYW5rIDMgYnV0IGdvdCByYW5rICR7eC5yYW5rfS5gKTtcbiAgICByZXR1cm4gdGhpcy50cmFjayh0aGlzLmF2Z1Bvb2xJbnRlcm5hbCh4LCBmU2l6ZSwgc3RyaWRlLCBwYWQpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgYXZnUG9vbEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRDtcblxuICAvKlxuICAgKiBCaWxpbmVhciByZXNpemUgYSAzRCBhcnJheSBwZXIgZWFjaCBjaGFubmVsIHRvIGEgbmV3IDJEIHNoYXBlLlxuICAgKiBAcGFyYW0geCBUaGUgaW5wdXQgQXJyYXkzRC5cbiAgICogQHBhcmFtIG5ld1NoYXBlMkQgVGhlIG5ldyBzaGFwZSB0byByZXNpemUgdGhlIEFycmF5M0QgdG8uIEVhY2ggY2hhbm5lbCBpc1xuICAgKiByZXNpemVkIGluZGl2aWR1YWxseS5cbiAgICogQHBhcmFtIGFsaWduQ29ybmVycyBBbiBvcHRpb25hbCBib29sLiBEZWZhdWx0cyB0byBGYWxzZS4gSWYgdHJ1ZSwgcmVzY2FsZVxuICAgKiBpbnB1dCBieSAobmV3X2hlaWdodCAtIDEpIC8gKGhlaWdodCAtIDEpLCB3aGljaCBleGFjdGx5IGFsaWducyB0aGUgNFxuICAgKiBjb3JuZXJzIG9mIGltYWdlcyBhbmQgcmVzaXplZCBpbWFnZXMuIElmIGZhbHNlLCByZXNjYWxlIGJ5IG5ld19oZWlnaHQgL1xuICAgKiBoZWlnaHQuIFRyZWF0IHNpbWlsYXJseSB0aGUgd2lkdGggZGltZW5zaW9uLlxuICAgKi9cbiAgcmVzaXplQmlsaW5lYXIzRChcbiAgICAgIHg6IEFycmF5M0QsIG5ld1NoYXBlMkQ6IFtudW1iZXIsIG51bWJlcl0sIGFsaWduQ29ybmVycyA9IGZhbHNlKTogQXJyYXkzRCB7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHgucmFuayA9PT0gMyxcbiAgICAgICAgYEVycm9yIGluIHJlc2l6ZUJpbGluZWFyM0Q6IHggbXVzdCBiZSByYW5rIDMgYnV0IGdvdCByYW5rICR7eC5yYW5rfS5gKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgbmV3U2hhcGUyRC5sZW5ndGggPT09IDIsXG4gICAgICAgIGBFcnJvciBpbiByZXNpemVCaWxpbmVhcjNEOiBuZXcgc2hhcGUgbXVzdCAyRCwgYnV0IGdvdCBzaGFwZSBgICtcbiAgICAgICAgICAgIGAke25ld1NoYXBlMkR9LmApO1xuICAgIHJldHVybiB0aGlzLnRyYWNrKFxuICAgICAgICB0aGlzLnJlc2l6ZUJpbGluZWFyM0RJbnRlcm5hbCh4LCBuZXdTaGFwZTJELCBhbGlnbkNvcm5lcnMpKTtcbiAgfVxuICBwcm90ZWN0ZWQgYWJzdHJhY3QgcmVzaXplQmlsaW5lYXIzREludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgbmV3U2hhcGUyRDogW251bWJlciwgbnVtYmVyXSwgYWxpZ25Db3JuZXJzOiBib29sZWFuKTogQXJyYXkzRDtcblxuICAvKipcbiAgICogQmF0Y2ggbm9ybWFsaXphdGlvbiAzRC4gTWVhbiwgdmFyaWFuY2UsIHNjYWxlLCBhbmQgb2Zmc2V0IGNhbiBiZSBvZiB0d29cbiAgICogc2hhcGVzOiAxKSBUaGUgc2FtZSBzaGFwZSBhcyB0aGUgaW5wdXQ6IGFuIEFycmF5M0QuIDIpIEluIHRoZSBjb21tb24gY2FzZSxcbiAgICogdGhlIGRlcHRoIGRpbWVuc2lvbiBpcyB0aGUgbGFzdCBkaW1lbnNpb24gb2YgeCwgc28gdGhlIHZhbHVlcyB3b3VsZCBiZSBhblxuICAgKiBBcnJheTFEIG9mIHNoYXBlIFtkZXB0aF0uXG4gICAqIEBwYXJhbSB4IFRoZSBpbnB1dCBOREFycmF5LlxuICAgKiBAcGFyYW0gbWVhbiBBIG1lYW4gTkRBcnJheS5cbiAgICogQHBhcmFtIHZhcmlhbmNlIEEgdmFyaWFuY2UgTkRBcnJheS5cbiAgICogQHBhcmFtIHZhcmlhbmNlRXBzaWxvbiBBIHNtYWxsIGZsb2F0IG51bWJlciB0byBhdm9pZCBkaXZpZGluZyBieSAwLlxuICAgKiBAcGFyYW0gc2NhbGUgQSBzY2FsZSBOREFycmF5LlxuICAgKiBAcGFyYW0gb2Zmc2V0IEFuIG9mZnNldCBOREFycmF5LlxuICAgKi9cbiAgYmF0Y2hOb3JtYWxpemF0aW9uM0QoXG4gICAgICB4OiBBcnJheTNELCBtZWFuOiBBcnJheTNEfEFycmF5MUQsIHZhcmlhbmNlOiBBcnJheTNEfEFycmF5MUQsXG4gICAgICB2YXJpYW5jZUVwc2lsb24gPSAuMDAxLCBzY2FsZT86IEFycmF5M0R8QXJyYXkxRCxcbiAgICAgIG9mZnNldD86IEFycmF5M0R8QXJyYXkxRCk6IEFycmF5M0Qge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB4LnJhbmsgPT09IDMsXG4gICAgICAgIGBFcnJvciBpbiBiYXRjaE5vcm1hbGl6YXRpb24zRDogeCBtdXN0IGJlIHJhbmsgMyBidXQgZ290IHJhbmsgYCArXG4gICAgICAgICAgICBgJHt4LnJhbmt9LmApO1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBtZWFuLnJhbmsgPT09IDMgfHwgbWVhbi5yYW5rID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gYmF0Y2hOb3JtYWxpemF0aW9uM0Q6IG1lYW4gbXVzdCBiZSByYW5rIDMgb3IgcmFuayAxIGJ1dCBgICtcbiAgICAgICAgICAgIGBnb3QgcmFuayAke21lYW4ucmFua30uYCk7XG4gICAgdXRpbC5hc3NlcnQoXG4gICAgICAgIHZhcmlhbmNlLnJhbmsgPT09IDMgfHwgdmFyaWFuY2UucmFuayA9PT0gMSxcbiAgICAgICAgYEVycm9yIGluIGJhdGNoTm9ybWFsaXphdGlvbjNEOiB2YXJpYW5jZSBtdXN0IGJlIHJhbmsgMyBvciByYW5rIDEgYCArXG4gICAgICAgICAgICBgYnV0IGdvdCByYW5rICR7dmFyaWFuY2UucmFua30uYCk7XG4gICAgaWYgKHNjYWxlICE9IG51bGwpIHtcbiAgICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICAgIHNjYWxlLnJhbmsgPT09IDMgfHwgc2NhbGUucmFuayA9PT0gMSxcbiAgICAgICAgICBgRXJyb3IgaW4gYmF0Y2hOb3JtYWxpemF0aW9uM0Q6IHNjYWxlIG11c3QgYmUgcmFuayAzIG9yIHJhbmsgMSBgICtcbiAgICAgICAgICAgICAgYGJ1dCBnb3QgcmFuayAke3NjYWxlIS5yYW5rfS5gKTtcbiAgICB9XG4gICAgaWYgKG9mZnNldCAhPSBudWxsKSB7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICBvZmZzZXQucmFuayA9PT0gMyB8fCBvZmZzZXQucmFuayA9PT0gMSxcbiAgICAgICAgICBgRXJyb3IgaW4gYmF0Y2hOb3JtYWxpemF0aW9uM0Q6IG9mZnNldCBtdXN0IGJlIHJhbmsgMyBvciByYW5rIDEgYCArXG4gICAgICAgICAgICAgIGBidXQgZ290IHJhbmsgJHtvZmZzZXQhLnJhbmt9LmApO1xuICAgIH1cblxuICAgIHJldHVybiB0aGlzLnRyYWNrKHRoaXMuYmF0Y2hOb3JtYWxpemF0aW9uM0RJbnRlcm5hbChcbiAgICAgICAgeCwgbWVhbiwgdmFyaWFuY2UsIHZhcmlhbmNlRXBzaWxvbiwgc2NhbGUsIG9mZnNldCkpO1xuICB9XG4gIHByb3RlY3RlZCBhYnN0cmFjdCBiYXRjaE5vcm1hbGl6YXRpb24zREludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgbWVhbjogQXJyYXkzRHxBcnJheTFELCB2YXJpYW5jZTogQXJyYXkzRHxBcnJheTFELFxuICAgICAgdmFyaWFuY2VFcHNpbG9uOiBudW1iZXIsIHNjYWxlPzogQXJyYXkzRHxBcnJheTFELFxuICAgICAgb2Zmc2V0PzogQXJyYXkzRHxBcnJheTFEKTogQXJyYXkzRDtcblxuICAvLy8vLy8vLy8vLy8vL1xuICAvLyBMU1RNIG9wcyAvL1xuICAvLy8vLy8vLy8vLy8vL1xuXG4gIC8qKlxuICAgKiBDb21wdXRlcyB0aGUgbmV4dCBzdGF0ZXMgYW5kIG91dHB1dHMgb2YgYSBzdGFjayBvZiBMU1RNQ2VsbHMuXG4gICAqIEVhY2ggY2VsbCBvdXRwdXQgaXMgdXNlZCBhcyBpbnB1dCB0byB0aGUgbmV4dCBjZWxsLlxuICAgKiBUaGlzIGlzIG9ubHkgdGhlIGZvcndhcmQgbW9kZS5cbiAgICogRGVyaXZlZCBmcm9tIHRmLmNvbnRyaWIucm4uTXVsdGlSTk5DZWxsLlxuICAgKiBAcGFyYW0gbHN0bUNlbGxzIEFycmF5IG9mIExTVE1DZWxsIGZ1bmN0aW9ucy5cbiAgICogQHBhcmFtIGRhdGEgVGhlIGlucHV0IHRvIHRoZSBjZWxsLlxuICAgKiBAcGFyYW0gYyBBcnJheSBvZiBwcmV2aW91cyBjZWxsIHN0YXRlcy5cbiAgICogQHBhcmFtIGggQXJyYXkgb2YgcHJldmlvdXMgY2VsbCBvdXRwdXRzLlxuICAgKiBAcmV0dXJuIFR1cGxlIFtuZXh0Q2VsbFN0YXRlcywgY2VsbE91dHB1dHNdXG4gICAqL1xuICBtdWx0aVJOTkNlbGwoXG4gICAgICBsc3RtQ2VsbHM6IExTVE1DZWxsW10sIGRhdGE6IEFycmF5MkQsIGM6IEFycmF5MkRbXSxcbiAgICAgIGg6IEFycmF5MkRbXSk6IFtBcnJheTJEW10sIEFycmF5MkRbXV0ge1xuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICBkYXRhLnNoYXBlWzBdID09PSAxLFxuICAgICAgICBgRXJyb3IgaW4gbXVsdGlSTk5DZWxsOiBmaXJzdCBkaW1lbnNpb24gb2YgZGF0YSBpcyAke2RhdGEuc2hhcGVbMF19LCBgICtcbiAgICAgICAgICAgIGBidXQgYmF0Y2ggc2l6ZXMgPiAxIGFyZSBub3QgeWV0IHN1cHBvcnRlZC5gKTtcbiAgICBjb25zdCByZXMgPSB0aGlzLnNjb3BlKCgpID0+IHtcbiAgICAgIGxldCBpbnB1dCA9IGRhdGE7XG4gICAgICBjb25zdCBuZXdTdGF0ZXMgPSBbXTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbHN0bUNlbGxzLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIGNvbnN0IG91dHB1dCA9IGxzdG1DZWxsc1tpXShpbnB1dCwgY1tpXSwgaFtpXSk7XG4gICAgICAgIG5ld1N0YXRlcy5wdXNoKG91dHB1dFswXSk7XG4gICAgICAgIG5ld1N0YXRlcy5wdXNoKG91dHB1dFsxXSk7XG4gICAgICAgIGlucHV0ID0gb3V0cHV0WzFdO1xuICAgICAgfVxuXG4gICAgICByZXR1cm4gbmV3U3RhdGVzO1xuICAgIH0pO1xuICAgIGNvbnN0IG5ld0M6IEFycmF5MkRbXSA9IFtdO1xuICAgIGNvbnN0IG5ld0g6IEFycmF5MkRbXSA9IFtdO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgcmVzLmxlbmd0aDsgaSArPSAyKSB7XG4gICAgICBuZXdDLnB1c2gocmVzW2ldIGFzIEFycmF5MkQpO1xuICAgICAgbmV3SC5wdXNoKHJlc1tpICsgMV0gYXMgQXJyYXkyRCk7XG4gICAgfVxuICAgIHJldHVybiBbbmV3QywgbmV3SF07XG4gIH1cblxuICAvKipcbiAgICogQ29tcHV0ZXMgdGhlIG5leHQgc3RhdGUgYW5kIG91dHB1dCBvZiBhIEJhc2ljTFNUTUNlbGwuXG4gICAqIFRoaXMgaXMgb25seSB0aGUgZm9yd2FyZCBtb2RlLlxuICAgKiBEZXJpdmVkIGZyb20gdGYuY29udHJpYi5ybm4uQmFzaWNMU1RNQ2VsbC5cbiAgICogQHBhcmFtIGZvcmdldEJpYXMgRm9yZ2V0IGJpYXMgZm9yIHRoZSBjZWxsLlxuICAgKiBAcGFyYW0gbHN0bUtlcm5lbCBUaGUgd2VpZ2h0cyBmb3IgdGhlIGNlbGwuXG4gICAqIEBwYXJhbSBsc3RtQmlhcyBUaGUgYmlhc2VzIGZvciB0aGUgY2VsbC5cbiAgICogQHBhcmFtIGRhdGEgVGhlIGlucHV0IHRvIHRoZSBjZWxsLlxuICAgKiBAcGFyYW0gYyBQcmV2aW91cyBjZWxsIHN0YXRlLlxuICAgKiBAcGFyYW0gaCBQcmV2aW91cyBjZWxsIG91dHB1dC5cbiAgICogQHJldHVybiBUdXBsZSBbbmV4dENlbGxTdGF0ZSwgY2VsbE91dHB1dF1cbiAgICovXG4gIGJhc2ljTFNUTUNlbGwoXG4gICAgICBmb3JnZXRCaWFzOiBTY2FsYXIsIGxzdG1LZXJuZWw6IEFycmF5MkQsIGxzdG1CaWFzOiBBcnJheTFELCBkYXRhOiBBcnJheTJELFxuICAgICAgYzogQXJyYXkyRCwgaDogQXJyYXkyRCk6IFtBcnJheTJELCBBcnJheTJEXSB7XG4gICAgY29uc3QgcmVzID0gdGhpcy5zY29wZSgoKSA9PiB7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICBkYXRhLnNoYXBlWzBdID09PSAxLFxuICAgICAgICAgIGBFcnJvciBpbiBtdWx0aVJOTkNlbGw6IGZpcnN0IGRpbWVuc2lvbiBvZiBkYXRhIGlzIGAgK1xuICAgICAgICAgICAgICBgJHtkYXRhLnNoYXBlWzBdfSwgYnV0IGJhdGNoIHNpemVzID4gMSBhcmUgbm90IHlldCBzdXBwb3J0ZWQuYCk7XG4gICAgICAvLyBjb25jYXQoaW5wdXRzLCBoLCAxKVxuICAgICAgLy8gVGhlcmUgaXMgbm8gY29uY2F0MWQsIHNvIHJlc2hhcGUgaW5wdXRzIGFuZCBoIHRvIDNkLCBjb25jYXQsIHRoZW5cbiAgICAgIC8vIHJlc2hhcGUgYmFjayB0byAyZC5cbiAgICAgIGNvbnN0IGRhdGEzRCA9IGRhdGEuYXMzRCgxLCAxLCBkYXRhLnNoYXBlWzFdKTtcbiAgICAgIGNvbnN0IGgzRCA9IGguYXMzRCgxLCAxLCBoLnNoYXBlWzFdKTtcbiAgICAgIGNvbnN0IGNvbWJpbmVkM0QgPSB0aGlzLmNvbmNhdDNEKGRhdGEzRCwgaDNELCAyKTtcbiAgICAgIGNvbnN0IGNvbWJpbmVkMkQgPSBjb21iaW5lZDNELmFzMkQoMSwgZGF0YS5zaGFwZVsxXSArIGguc2hhcGVbMV0pO1xuXG4gICAgICBjb25zdCB3ZWlnaHRlZCA9IHRoaXMubWF0TXVsKGNvbWJpbmVkMkQsIGxzdG1LZXJuZWwpO1xuICAgICAgY29uc3QgcmVzID0gdGhpcy5hZGQod2VpZ2h0ZWQsIGxzdG1CaWFzKSBhcyBBcnJheTJEO1xuXG4gICAgICAvLyBpID0gaW5wdXRfZ2F0ZSwgaiA9IG5ld19pbnB1dCwgZiA9IGZvcmdldF9nYXRlLCBvID0gb3V0cHV0X2dhdGVcbiAgICAgIGNvbnN0IGkgPSB0aGlzLnNsaWNlMkQocmVzLCBbMCwgMF0sIFtyZXMuc2hhcGVbMF0sIHJlcy5zaGFwZVsxXSAvIDRdKTtcbiAgICAgIGNvbnN0IGogPSB0aGlzLnNsaWNlMkQoXG4gICAgICAgICAgcmVzLCBbMCwgcmVzLnNoYXBlWzFdIC8gNCAqIDFdLCBbcmVzLnNoYXBlWzBdLCByZXMuc2hhcGVbMV0gLyA0XSk7XG4gICAgICBjb25zdCBmID0gdGhpcy5zbGljZTJEKFxuICAgICAgICAgIHJlcywgWzAsIHJlcy5zaGFwZVsxXSAvIDQgKiAyXSwgW3Jlcy5zaGFwZVswXSwgcmVzLnNoYXBlWzFdIC8gNF0pO1xuICAgICAgY29uc3QgbyA9IHRoaXMuc2xpY2UyRChcbiAgICAgICAgICByZXMsIFswLCByZXMuc2hhcGVbMV0gLyA0ICogM10sIFtyZXMuc2hhcGVbMF0sIHJlcy5zaGFwZVsxXSAvIDRdKTtcblxuICAgICAgY29uc3QgbmV3QyA9XG4gICAgICAgICAgdGhpcy5hZGQoXG4gICAgICAgICAgICAgIHRoaXMubXVsdGlwbHlTdHJpY3QoXG4gICAgICAgICAgICAgICAgICBjLCB0aGlzLnNpZ21vaWQodGhpcy5zY2FsYXJQbHVzQXJyYXkoZm9yZ2V0QmlhcywgZikpKSxcbiAgICAgICAgICAgICAgdGhpcy5tdWx0aXBseVN0cmljdCh0aGlzLnNpZ21vaWQoaSksIHRoaXMudGFuaChqKSkpIGFzIEFycmF5MkQ7XG4gICAgICBjb25zdCBuZXdIID1cbiAgICAgICAgICB0aGlzLm11bHRpcGx5U3RyaWN0KHRoaXMudGFuaChuZXdDKSwgdGhpcy5zaWdtb2lkKG8pKSBhcyBBcnJheTJEO1xuXG4gICAgICByZXR1cm4gW25ld0MsIG5ld0hdO1xuICAgIH0pO1xuICAgIHJldHVybiBbcmVzWzBdLCByZXNbMV1dO1xuICB9XG59XG5cbmV4cG9ydCBlbnVtIE1hdHJpeE9yaWVudGF0aW9uIHtcbiAgUkVHVUxBUixcbiAgVFJBTlNQT1NFRFxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyBjb252X3V0aWwgZnJvbSAnLi4vbWF0aC9jb252X3V0aWwnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0ICogYXMgY29uY2F0M2RfdXRpbCBmcm9tICcuL2NvbmNhdDNkX3V0aWwnO1xuaW1wb3J0ICogYXMgY29weTJEX3V0aWwgZnJvbSAnLi9jb3B5MmRfdXRpbCc7XG5pbXBvcnQge01hdHJpeE9yaWVudGF0aW9uLCBOREFycmF5TWF0aH0gZnJvbSAnLi9tYXRoJztcbmltcG9ydCB7QXJyYXkxRCwgQXJyYXkyRCwgQXJyYXkzRCwgQXJyYXk0RCwgTkRBcnJheSwgU2NhbGFyfSBmcm9tICcuL25kYXJyYXknO1xuXG5leHBvcnQgY2xhc3MgTkRBcnJheU1hdGhDUFUgZXh0ZW5kcyBOREFycmF5TWF0aCB7XG4gIGNvbnN0cnVjdG9yKHNhZmVNb2RlID0gZmFsc2UpIHtcbiAgICBzdXBlcihzYWZlTW9kZSk7XG4gIH1cblxuICBwcm90ZWN0ZWQgY2xvbmVJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4oXG4gICAgICAgIG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IG5ldyBGbG9hdDMyQXJyYXkobmRhcnJheS5nZXRWYWx1ZXMoKSl9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBzbGljZTJESW50ZXJuYWwoXG4gICAgICBpbnB1dDogQXJyYXkyRCwgYmVnaW5Sb3dDb2w6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBzaXplUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdKTogQXJyYXkyRCB7XG4gICAgY29uc3QgcmVzdWx0ID0gQXJyYXkyRC56ZXJvcyhzaXplUm93Q29sKTtcbiAgICB0aGlzLmNvcHkyREludGVybmFsKFxuICAgICAgICBpbnB1dCwgYmVnaW5Sb3dDb2wsIHNpemVSb3dDb2wsIHJlc3VsdCwgWzAsIDBdLCBzaXplUm93Q29sKTtcbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHJvdGVjdGVkIGNvcHkyREludGVybmFsKFxuICAgICAgc291cmNlOiBBcnJheTJELCBzb3VyY2VCZWdpblJvd0NvbDogW251bWJlciwgbnVtYmVyXSxcbiAgICAgIHNvdXJjZVNpemVSb3dDb2w6IFtudW1iZXIsIG51bWJlcl0sIGRlc3Q6IEFycmF5MkQsXG4gICAgICBkZXN0QmVnaW5Sb3dDb2w6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBkZXN0U2l6ZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSk6IHZvaWQge1xuICAgIGNvcHkyRF91dGlsLnZhbGlkYXRlU2hhcGVzKHNvdXJjZVNpemVSb3dDb2wsIGRlc3RTaXplUm93Q29sKTtcbiAgICBjb25zdCBzcmNWYWx1ZXMgPSBzb3VyY2UuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3QgZHN0VmFsdWVzID0gZGVzdC5nZXRWYWx1ZXMoKTtcbiAgICBjb25zdCBuID0gc291cmNlU2l6ZVJvd0NvbFswXSAqIHNvdXJjZVNpemVSb3dDb2xbMV07XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBuOyArK2kpIHtcbiAgICAgIGNvbnN0IHNyY1JvdyA9IHNvdXJjZUJlZ2luUm93Q29sWzBdICsgTWF0aC5mbG9vcihpIC8gc291cmNlU2l6ZVJvd0NvbFsxXSk7XG4gICAgICBjb25zdCBzcmNDb2wgPSBzb3VyY2VCZWdpblJvd0NvbFsxXSArIChpICUgc291cmNlU2l6ZVJvd0NvbFsxXSk7XG4gICAgICBjb25zdCBzcmNPZmYgPSBzcmNSb3cgKiBzb3VyY2Uuc2hhcGVbMV0gKyBzcmNDb2w7XG4gICAgICBjb25zdCBkc3RSb3cgPSBkZXN0QmVnaW5Sb3dDb2xbMF0gKyBNYXRoLmZsb29yKGkgLyBkZXN0U2l6ZVJvd0NvbFsxXSk7XG4gICAgICBjb25zdCBkc3RDb2wgPSBkZXN0QmVnaW5Sb3dDb2xbMV0gKyAoaSAlIGRlc3RTaXplUm93Q29sWzFdKTtcbiAgICAgIGNvbnN0IGRzdE9mZiA9IGRzdFJvdyAqIGRlc3Quc2hhcGVbMV0gKyBkc3RDb2w7XG4gICAgICBkc3RWYWx1ZXNbZHN0T2ZmXSA9IHNyY1ZhbHVlc1tzcmNPZmZdO1xuICAgIH1cbiAgfVxuXG4gIHByb3RlY3RlZCBjb25jYXQzREludGVybmFsKHgxOiBBcnJheTNELCB4MjogQXJyYXkzRCwgYXhpczogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgY29uc3Qgb3V0cHV0U2hhcGUgPVxuICAgICAgICBjb25jYXQzZF91dGlsLmNvbXB1dGVDb25jYXQzRE91dHB1dFNoYXBlKHgxLnNoYXBlLCB4Mi5zaGFwZSwgYXhpcyk7XG5cbiAgICBjb25zdCB2YWx1ZXMgPSBBcnJheTNELnplcm9zKG91dHB1dFNoYXBlKTtcblxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgb3V0cHV0U2hhcGVbMF07IGkrKykge1xuICAgICAgZm9yIChsZXQgaiA9IDA7IGogPCBvdXRwdXRTaGFwZVsxXTsgaisrKSB7XG4gICAgICAgIGZvciAobGV0IGsgPSAwOyBrIDwgb3V0cHV0U2hhcGVbMl07IGsrKykge1xuICAgICAgICAgIC8vIFNoYWRlciBiZWdpbnMuXG4gICAgICAgICAgY29uc3QgaW5kZXg6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSA9IFtpLCBqLCBrXTtcbiAgICAgICAgICBsZXQgdmFsdWU6IG51bWJlcjtcbiAgICAgICAgICBpZiAoaW5kZXhbYXhpc10gPCB4MS5zaGFwZVtheGlzXSkge1xuICAgICAgICAgICAgdmFsdWUgPSB4MS5nZXQoaSwgaiwgayk7XG4gICAgICAgICAgfSBlbHNlIHtcbiAgICAgICAgICAgIGluZGV4W2F4aXNdIC09IHgxLnNoYXBlW2F4aXNdO1xuICAgICAgICAgICAgY29uc3QgW2kyLCBqMiwgazJdID0gaW5kZXg7XG4gICAgICAgICAgICB2YWx1ZSA9IHgyLmdldChpMiwgajIsIGsyKTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICB2YWx1ZXMuc2V0KHZhbHVlLCBpLCBqLCBrKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiB2YWx1ZXM7XG4gIH1cblxuICBwcm90ZWN0ZWQgc2NhbGVkQXJyYXlBZGRJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oXG4gICAgICBjMTogU2NhbGFyLCBhOiBULCBjMjogU2NhbGFyLCBiOiBUKSB7XG4gICAgY29uc3QgbmV3U2hhcGUgPSB1dGlsLmFzc2VydEFuZEdldEJyb2FkY2FzdGVkU2hhcGUoYS5zaGFwZSwgYi5zaGFwZSk7XG4gICAgY29uc3QgbmV3VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheSh1dGlsLnNpemVGcm9tU2hhcGUobmV3U2hhcGUpKTtcblxuICAgIGNvbnN0IGFWYWx1ZXMgPSBhLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IGJWYWx1ZXMgPSBiLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IGMxVmFsID0gYzEuZ2V0KCk7XG4gICAgY29uc3QgYzJWYWwgPSBjMi5nZXQoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG5ld1ZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgbmV3VmFsdWVzW2ldID0gYzFWYWwgKiBhVmFsdWVzW2kgJSBhLnNpemVdICsgYzJWYWwgKiBiVmFsdWVzW2kgJSBiLnNpemVdO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5ld1NoYXBlLCB7dmFsdWVzOiBuZXdWYWx1ZXN9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBuZWdJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnNjYWxhclRpbWVzQXJyYXkoU2NhbGFyLk5FR19PTkUsIGEpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGFkZEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgcmV0dXJuIHRoaXMuc2NhbGVkQXJyYXlBZGRJbnRlcm5hbDxUPihTY2FsYXIuT05FLCBhLCBTY2FsYXIuT05FLCBiKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBzdWJJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQge1xuICAgIHJldHVybiB0aGlzLnNjYWxlZEFycmF5QWRkSW50ZXJuYWw8VD4oU2NhbGFyLk9ORSwgYSwgU2NhbGFyLk5FR19PTkUsIGIpO1xuICB9XG5cbiAgcHJvdGVjdGVkIG1hdE11bEludGVybmFsKFxuICAgICAgYTogQXJyYXkyRCwgYjogQXJyYXkyRCwgYU9yaWVudGF0aW9uID0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUixcbiAgICAgIGJPcmllbnRhdGlvbiA9IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpOiBBcnJheTJEIHtcbiAgICBjb25zdCBzaGFyZWREaW0gPVxuICAgICAgICAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IGEuc2hhcGVbMV0gOiBhLnNoYXBlWzBdO1xuXG4gICAgY29uc3QgbGVmdERpbSA9XG4gICAgICAgIChhT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID8gYS5zaGFwZVswXSA6IGEuc2hhcGVbMV07XG4gICAgY29uc3QgcmlnaHREaW0gPVxuICAgICAgICAoYk9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IGIuc2hhcGVbMV0gOiBiLnNoYXBlWzBdO1xuXG4gICAgY29uc3Qgbm9ybWFsR2V0dGVyID0gKG1hdHJpeDogQXJyYXkyRCwgaTogbnVtYmVyLCBqOiBudW1iZXIpID0+XG4gICAgICAgIG1hdHJpeC5nZXQoaSwgaik7XG4gICAgY29uc3QgdHJhbnNwb3NlZEdldHRlciA9IChtYXRyaXg6IEFycmF5MkQsIGk6IG51bWJlciwgajogbnVtYmVyKSA9PlxuICAgICAgICBtYXRyaXguZ2V0KGosIGkpO1xuXG4gICAgY29uc3QgYUdldHRlciA9IChhT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgICAgbm9ybWFsR2V0dGVyIDpcbiAgICAgICAgdHJhbnNwb3NlZEdldHRlcjtcbiAgICBjb25zdCBiR2V0dGVyID0gKGJPcmllbnRhdGlvbiA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgP1xuICAgICAgICBub3JtYWxHZXR0ZXIgOlxuICAgICAgICB0cmFuc3Bvc2VkR2V0dGVyO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobGVmdERpbSAqIHJpZ2h0RGltKTtcbiAgICBsZXQgaW5kZXggPSAwO1xuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsZWZ0RGltOyArK2kpIHtcbiAgICAgIGZvciAobGV0IGogPSAwOyBqIDwgcmlnaHREaW07ICsraikge1xuICAgICAgICBsZXQgc3VtID0gMDtcbiAgICAgICAgZm9yIChsZXQgayA9IDA7IGsgPCBzaGFyZWREaW07ICsraykge1xuICAgICAgICAgIC8vIFRPRE86IG9wdGltaXplIENQVSBtYXRtdWwuXG4gICAgICAgICAgc3VtICs9IGFHZXR0ZXIoYSwgaSwgaykgKiBiR2V0dGVyKGIsIGssIGopO1xuICAgICAgICB9XG4gICAgICAgIHZhbHVlc1tpbmRleCsrXSA9IHN1bTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIEFycmF5MkQubmV3KFtsZWZ0RGltLCByaWdodERpbV0sIHZhbHVlcyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgbXVsdGlwbHlJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4oYTogVCwgYjogVCk6IFQge1xuICAgIGNvbnN0IG5ld1NoYXBlID0gdXRpbC5hc3NlcnRBbmRHZXRCcm9hZGNhc3RlZFNoYXBlKGEuc2hhcGUsIGIuc2hhcGUpO1xuICAgIGNvbnN0IG5ld1ZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkodXRpbC5zaXplRnJvbVNoYXBlKG5ld1NoYXBlKSk7XG5cbiAgICBjb25zdCBhVmFsdWVzID0gYS5nZXRWYWx1ZXMoKTtcbiAgICBjb25zdCBiVmFsdWVzID0gYi5nZXRWYWx1ZXMoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IG5ld1ZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgbmV3VmFsdWVzW2ldID0gYVZhbHVlc1tpICUgYS5zaXplXSAqIGJWYWx1ZXNbaSAlIGIuc2l6ZV07XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8VD4obmV3U2hhcGUsIHt2YWx1ZXM6IG5ld1ZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIGRpdmlkZUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihhOiBULCBiOiBUKTogVCB7XG4gICAgY29uc3QgbmV3U2hhcGUgPSB1dGlsLmFzc2VydEFuZEdldEJyb2FkY2FzdGVkU2hhcGUoYS5zaGFwZSwgYi5zaGFwZSk7XG4gICAgY29uc3QgbmV3VmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheSh1dGlsLnNpemVGcm9tU2hhcGUobmV3U2hhcGUpKTtcblxuICAgIGNvbnN0IGFWYWx1ZXMgPSBhLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IGJWYWx1ZXMgPSBiLmdldFZhbHVlcygpO1xuXG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBuZXdWYWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIG5ld1ZhbHVlc1tpXSA9IGFWYWx1ZXNbaSAlIGEuc2l6ZV0gLyBiVmFsdWVzW2kgJSBiLnNpemVdO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5ld1NoYXBlLCB7dmFsdWVzOiBuZXdWYWx1ZXN9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBzdW1JbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICBsZXQgc3VtID0gMDtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBzdW0gKz0gdmFsdWVzW2ldO1xuICAgIH1cbiAgICByZXR1cm4gU2NhbGFyLm5ldyhzdW0pO1xuICB9XG5cbiAgcHJvdGVjdGVkIGFyZ01pbkludGVybmFsKG5kYXJyYXk6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGxldCBtaW4gPSBOdW1iZXIuTUFYX1ZBTFVFO1xuICAgIGxldCBtaW5JbmRleCA9IC0xO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgaWYgKGlzTmFOKHZhbHVlKSkge1xuICAgICAgICByZXR1cm4gU2NhbGFyLm5ldyhOYU4pO1xuICAgICAgfVxuICAgICAgaWYgKHZhbHVlIDwgbWluKSB7XG4gICAgICAgIG1pbiA9IHZhbHVlO1xuICAgICAgICBtaW5JbmRleCA9IGk7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBTY2FsYXIubmV3KG1pbkluZGV4KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBhcmdNYXhJbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICBsZXQgbWF4ID0gTnVtYmVyLk5FR0FUSVZFX0lORklOSVRZO1xuICAgIGxldCBtYXhJbmRleCA9IC0xO1xuICAgIGNvbnN0IHZhbHVlcyA9IG5kYXJyYXkuZ2V0VmFsdWVzKCk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgaWYgKGlzTmFOKHZhbHVlKSkge1xuICAgICAgICByZXR1cm4gU2NhbGFyLm5ldyhOYU4pO1xuICAgICAgfVxuICAgICAgaWYgKHZhbHVlID4gbWF4KSB7XG4gICAgICAgIG1heCA9IHZhbHVlO1xuICAgICAgICBtYXhJbmRleCA9IGk7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBTY2FsYXIubmV3KG1heEluZGV4KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBhcmdNYXhFcXVhbHNJbnRlcm5hbCh4MTogTkRBcnJheSwgeDI6IE5EQXJyYXkpOiBTY2FsYXIge1xuICAgIGNvbnN0IGFyZ01heDEgPSB0aGlzLmFyZ01heEludGVybmFsKHgxKS5nZXQoKTtcbiAgICBjb25zdCBhcmdNYXgyID0gdGhpcy5hcmdNYXhJbnRlcm5hbCh4MikuZ2V0KCk7XG4gICAgaWYgKGlzTmFOKGFyZ01heDEpIHx8IGlzTmFOKGFyZ01heDIpKSB7XG4gICAgICByZXR1cm4gU2NhbGFyLm5ldyhOYU4pO1xuICAgIH1cbiAgICByZXR1cm4gU2NhbGFyLm5ldygrKGFyZ01heDEgPT09IGFyZ01heDIpKTtcbiAgfVxuXG4gIHByb3RlY3RlZCB0b3BLSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSwgazogbnVtYmVyKTpcbiAgICAgIHt2YWx1ZXM6IEFycmF5MUQsIGluZGljZXM6IEFycmF5MUR9IHtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IHZhbHVlc0FuZEluZGljZXM6IEFycmF5PHt2YWx1ZTogbnVtYmVyLCBpbmRleDogbnVtYmVyfT4gPSBbXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7IGkrKykge1xuICAgICAgdmFsdWVzQW5kSW5kaWNlcy5wdXNoKHt2YWx1ZTogdmFsdWVzW2ldLCBpbmRleDogaX0pO1xuICAgIH1cbiAgICB2YWx1ZXNBbmRJbmRpY2VzLnNvcnQoKGEsIGIpID0+IHtcbiAgICAgIHJldHVybiBiLnZhbHVlIC0gYS52YWx1ZTtcbiAgICB9KTtcbiAgICBjb25zdCB0b3BrVmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShrKTtcbiAgICBjb25zdCB0b3BrSW5kaWNlcyA9IG5ldyBGbG9hdDMyQXJyYXkoayk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBrOyBpKyspIHtcbiAgICAgIHRvcGtWYWx1ZXNbaV0gPSB2YWx1ZXNBbmRJbmRpY2VzW2ldLnZhbHVlO1xuICAgICAgdG9wa0luZGljZXNbaV0gPSB2YWx1ZXNBbmRJbmRpY2VzW2ldLmluZGV4O1xuICAgIH1cbiAgICByZXR1cm4ge3ZhbHVlczogQXJyYXkxRC5uZXcodG9wa1ZhbHVlcyksIGluZGljZXM6IEFycmF5MUQubmV3KHRvcGtJbmRpY2VzKX07XG4gIH1cblxuICBwcm90ZWN0ZWQgbWluSW50ZXJuYWwobmRhcnJheTogTkRBcnJheSk6IFNjYWxhciB7XG4gICAgY29uc3QgdmFsdWVzID0gbmRhcnJheS5nZXRWYWx1ZXMoKTtcbiAgICBsZXQgbWluID0gdmFsdWVzWzBdO1xuICAgIGZvciAobGV0IGkgPSAxOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBjb25zdCB2YWx1ZSA9IHZhbHVlc1tpXTtcbiAgICAgIGlmIChpc05hTih2YWx1ZSkpIHtcbiAgICAgICAgcmV0dXJuIFNjYWxhci5uZXcoTmFOKTtcbiAgICAgIH1cbiAgICAgIGlmICh2YWx1ZSA8IG1pbikge1xuICAgICAgICBtaW4gPSB2YWx1ZTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIFNjYWxhci5uZXcobWluKTtcbiAgfVxuXG4gIHByb3RlY3RlZCBtYXhJbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGxldCBtYXggPSB2YWx1ZXNbMF07XG4gICAgZm9yIChsZXQgaSA9IDE7IGkgPCB2YWx1ZXMubGVuZ3RoOyArK2kpIHtcbiAgICAgIGNvbnN0IHZhbHVlID0gdmFsdWVzW2ldO1xuICAgICAgaWYgKGlzTmFOKHZhbHVlKSkge1xuICAgICAgICByZXR1cm4gU2NhbGFyLm5ldyhOYU4pO1xuICAgICAgfVxuICAgICAgaWYgKHZhbHVlID4gbWF4KSB7XG4gICAgICAgIG1heCA9IHZhbHVlO1xuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4gU2NhbGFyLm5ldyhtYXgpO1xuICB9XG5cbiAgcHJvdGVjdGVkIGV4cEludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgY29uc3QgdmFsdWVzID0gbmRhcnJheS5nZXRWYWx1ZXMoKTtcbiAgICBjb25zdCBuZXdWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHZhbHVlcy5sZW5ndGgpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBuZXdWYWx1ZXNbaV0gPSBNYXRoLmV4cCh2YWx1ZXNbaV0pO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IG5ld1ZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIGxvZ0ludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5PihuZGFycmF5OiBUKTogVCB7XG4gICAgY29uc3QgdmFsdWVzID0gbmRhcnJheS5nZXRWYWx1ZXMoKTtcbiAgICBjb25zdCBuZXdWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHZhbHVlcy5sZW5ndGgpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBjb25zdCB2YWx1ZSA9IHZhbHVlc1tpXTtcbiAgICAgIG5ld1ZhbHVlc1tpXSA9IE1hdGgubG9nKHZhbHVlKTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihuZGFycmF5LnNoYXBlLCB7dmFsdWVzOiBuZXdWYWx1ZXN9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBsb2dTdW1FeHBJbnRlcm5hbChuZGFycmF5OiBOREFycmF5KTogU2NhbGFyIHtcbiAgICBjb25zdCB4TWF4ID0gdGhpcy5tYXgobmRhcnJheSk7XG4gICAgY29uc3QgYSA9IHRoaXMuYXJyYXlNaW51c1NjYWxhcihuZGFycmF5LCB4TWF4KTtcbiAgICBjb25zdCBiID0gdGhpcy5leHAoYSk7XG4gICAgY29uc3QgYyA9IHRoaXMuc3VtKGIpO1xuICAgIGNvbnN0IGQgPSB0aGlzLmxvZyhjKTtcbiAgICBjb25zdCByZXN1bHQgPSB0aGlzLmFkZCh4TWF4LCBkKTtcblxuICAgIHhNYXguZGlzcG9zZSgpO1xuICAgIGEuZGlzcG9zZSgpO1xuICAgIGIuZGlzcG9zZSgpO1xuICAgIGMuZGlzcG9zZSgpO1xuICAgIGQuZGlzcG9zZSgpO1xuXG4gICAgcmV0dXJuIHJlc3VsdDtcbiAgfVxuXG4gIHByb3RlY3RlZCByZWx1SW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICBjb25zdCByZXN1bHRWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KG5kYXJyYXkuc2l6ZSk7XG4gICAgY29uc3QgdmFsdWVzID0gbmRhcnJheS5nZXRWYWx1ZXMoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgcmVzdWx0VmFsdWVzW2ldID0gTWF0aC5tYXgoMCwgdmFsdWVzW2ldKTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihuZGFycmF5LnNoYXBlLCB7dmFsdWVzOiByZXN1bHRWYWx1ZXN9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBzaWdtb2lkSW50ZXJuYWw8VCBleHRlbmRzIE5EQXJyYXk+KG5kYXJyYXk6IFQpOiBUIHtcbiAgICBjb25zdCByZXN1bHRWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KG5kYXJyYXkuc2l6ZSk7XG4gICAgY29uc3QgdmFsdWVzID0gbmRhcnJheS5nZXRWYWx1ZXMoKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHZhbHVlcy5sZW5ndGg7ICsraSkge1xuICAgICAgcmVzdWx0VmFsdWVzW2ldID0gMSAvICgxICsgTWF0aC5leHAoLXZhbHVlc1tpXSkpO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHRhbmhJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIGNvbnN0IHJlc3VsdFZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobmRhcnJheS5zaXplKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICByZXN1bHRWYWx1ZXNbaV0gPSB1dGlsLnRhbmgodmFsdWVzW2ldKTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihuZGFycmF5LnNoYXBlLCB7dmFsdWVzOiByZXN1bHRWYWx1ZXN9KTtcbiAgfVxuXG4gIHByb3RlY3RlZCBzaW5JbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIGNvbnN0IHJlc3VsdFZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobmRhcnJheS5zaXplKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICByZXN1bHRWYWx1ZXNbaV0gPSBNYXRoLnNpbih2YWx1ZXNbaV0pO1xuICAgIH1cbiAgICByZXR1cm4gTkRBcnJheS5tYWtlPFQ+KG5kYXJyYXkuc2hhcGUsIHt2YWx1ZXM6IHJlc3VsdFZhbHVlc30pO1xuICB9XG5cbiAgcHJvdGVjdGVkIHN0ZXBJbnRlcm5hbDxUIGV4dGVuZHMgTkRBcnJheT4obmRhcnJheTogVCk6IFQge1xuICAgIGNvbnN0IHJlc3VsdFZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkobmRhcnJheS5zaXplKTtcbiAgICBjb25zdCB2YWx1ZXMgPSBuZGFycmF5LmdldFZhbHVlcygpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdmFsdWVzLmxlbmd0aDsgKytpKSB7XG4gICAgICBjb25zdCB2YWx1ZSA9IHZhbHVlc1tpXTtcbiAgICAgIHJlc3VsdFZhbHVlc1tpXSA9IHZhbHVlID4gMCA/IDEgOiAodmFsdWUgPCAwID8gMCA6IHZhbHVlKTtcbiAgICB9XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihuZGFycmF5LnNoYXBlLCB7dmFsdWVzOiByZXN1bHRWYWx1ZXN9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBpbWFnZSBpcyBvZiBzaGFwZSBbciwgYywgZDFdLlxuICAgKiB3ZWlnaHRzIGlzIG9mIHNoYXBlIFtGLCBGLCBkMSwgZDJdLlxuICAgKi9cbiAgcHJvdGVjdGVkIGNvbnYyZEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgd2VpZ2h0czogQXJyYXk0RCwgYmlhc2VzOiBBcnJheTFEfG51bGwsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICBjb25zdCBbeFJvd3MsIHhDb2xzLCBpbnB1dERlcHRoXSA9IHguc2hhcGU7XG4gICAgY29uc3QgZmllbGRTaXplID0gd2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBvdXRwdXREZXB0aCA9IHdlaWdodHMuc2hhcGVbM107XG4gICAgY29uc3Qgb3V0cHV0U2hhcGUgPSBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgIFt4Um93cywgeENvbHMsIGlucHV0RGVwdGhdLCBmaWVsZFNpemUsIG91dHB1dERlcHRoLCBzdHJpZGUsIHBhZCk7XG4gICAgY29uc3QgeSA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICAgIGZvciAobGV0IGQyID0gMDsgZDIgPCBvdXRwdXREZXB0aDsgKytkMikge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IHkuc2hhcGVbMF07ICsreVIpIHtcbiAgICAgICAgY29uc3QgeFJDb3JuZXIgPSB5UiAqIHN0cmlkZSAtIHBhZDtcbiAgICAgICAgY29uc3QgeFJNaW4gPSBNYXRoLm1heCgwLCB4UkNvcm5lcik7XG4gICAgICAgIGNvbnN0IHhSTWF4ID0gTWF0aC5taW4oeFJvd3MsIGZpZWxkU2l6ZSArIHhSQ29ybmVyKTtcbiAgICAgICAgZm9yIChsZXQgeUMgPSAwOyB5QyA8IHkuc2hhcGVbMV07ICsreUMpIHtcbiAgICAgICAgICBjb25zdCB4Q0Nvcm5lciA9IHlDICogc3RyaWRlIC0gcGFkO1xuICAgICAgICAgIGNvbnN0IHhDTWluID0gTWF0aC5tYXgoMCwgeENDb3JuZXIpO1xuICAgICAgICAgIGNvbnN0IHhDTWF4ID0gTWF0aC5taW4oeENvbHMsIGZpZWxkU2l6ZSArIHhDQ29ybmVyKTtcbiAgICAgICAgICBsZXQgZG90UHJvZCA9IDA7XG4gICAgICAgICAgZm9yIChsZXQgeFIgPSB4Uk1pbjsgeFIgPCB4Uk1heDsgKyt4Uikge1xuICAgICAgICAgICAgY29uc3Qgd1IgPSB4UiAtIHhSQ29ybmVyO1xuICAgICAgICAgICAgZm9yIChsZXQgeEMgPSB4Q01pbjsgeEMgPCB4Q01heDsgKyt4Qykge1xuICAgICAgICAgICAgICBjb25zdCB3QyA9IHhDIC0geENDb3JuZXI7XG4gICAgICAgICAgICAgIGZvciAobGV0IGQxID0gMDsgZDEgPCBpbnB1dERlcHRoOyArK2QxKSB7XG4gICAgICAgICAgICAgICAgY29uc3QgcGl4ZWwgPSB4LmdldCh4UiwgeEMsIGQxKTtcbiAgICAgICAgICAgICAgICBjb25zdCB3ZWlnaHQgPSB3ZWlnaHRzLmdldCh3Uiwgd0MsIGQxLCBkMik7XG4gICAgICAgICAgICAgICAgZG90UHJvZCArPSBwaXhlbCAqIHdlaWdodDtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgIH1cbiAgICAgICAgICBjb25zdCBiaWFzID0gKGJpYXNlcyAhPSBudWxsKSA/IGJpYXNlcy5nZXQoZDIpIDogMDtcbiAgICAgICAgICB5LnNldChkb3RQcm9kICsgYmlhcywgeVIsIHlDLCBkMik7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIHk7XG4gIH1cblxuICBwcm90ZWN0ZWQgY29udjJkQmFja1Byb3BJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIGR5OiBBcnJheTNELCB3ZWlnaHRzOiBBcnJheTRELCBzdHJpZGU6IG51bWJlcixcbiAgICAgIHBhZDogbnVtYmVyKToge2R4OiBBcnJheTNELCBkdzogQXJyYXk0RCwgZGI6IEFycmF5MUR9IHtcbiAgICBjb25zdCBmU2l6ZSA9IHdlaWdodHMuc2hhcGVbMF07XG4gICAgY29uc3QgZHcgPSB0aGlzLmNvbnYyZERlcldlaWdodHMoeCwgZHksIGZTaXplLCBzdHJpZGUsIHBhZCk7XG4gICAgY29uc3QgZGIgPSB0aGlzLmNvbnYyZERlckJpYXMoZHkpO1xuICAgIGNvbnN0IGR4ID0gdGhpcy5jb252MmRUcmFuc3Bvc2VJbnRlcm5hbChkeSwgd2VpZ2h0cywgbnVsbCwgc3RyaWRlLCBwYWQpO1xuICAgIHJldHVybiB7ZHgsIGRiLCBkd307XG4gIH1cblxuICAvKipcbiAgICogaW1hZ2UgaXMgb2Ygc2hhcGUgW3IsIGMsIGQxXS5cbiAgICogd2VpZ2h0cyBpcyBvZiBzaGFwZSBbRiwgRiwgZDEsIGQyXS5cbiAgICovXG4gIHByb3RlY3RlZCBjb252MmRUcmFuc3Bvc2VJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIHdlaWdodHM6IEFycmF5NEQsIGJpYXNlczogQXJyYXkxRHxudWxsLCBvcmlnU3RyaWRlOiBudW1iZXIsXG4gICAgICBvcmlnUGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICBjb25zdCBmU2l6ZSA9IHdlaWdodHMuc2hhcGVbMF07XG4gICAgY29uc3QgcGFkID0gZlNpemUgLSAxIC0gb3JpZ1BhZDtcbiAgICBjb25zdCBvcmlnSW5wdXREZXB0aCA9IHdlaWdodHMuc2hhcGVbMl07XG4gICAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gd2VpZ2h0cy5zaGFwZVszXTtcbiAgICBjb25zdCB4Um93cyA9IHguc2hhcGVbMF07XG4gICAgY29uc3QgeENvbHMgPSB4LnNoYXBlWzFdO1xuXG4gICAgLy8gRGlsYXRlIHRoZSBpbnB1dC5cbiAgICBjb25zdCB4Um93c0RpbGF0ZWQgPSAoeFJvd3MgLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuICAgIGNvbnN0IHhDb2xzRGlsYXRlZCA9ICh4Q29scyAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG5cbiAgICBjb25zdCBvdXRwdXRTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgICAgW3hSb3dzRGlsYXRlZCwgeENvbHNEaWxhdGVkLCBvcmlnT3V0cHV0RGVwdGhdLCBmU2l6ZSwgb3JpZ0lucHV0RGVwdGgsIDEsXG4gICAgICAgIHBhZCk7XG4gICAgY29uc3QgeSA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICAgIGZvciAobGV0IGQyID0gMDsgZDIgPCBvcmlnSW5wdXREZXB0aDsgKytkMikge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IHkuc2hhcGVbMF07ICsreVIpIHtcbiAgICAgICAgY29uc3QgeFJDb3JuZXIgPSB5UiAtIHBhZDtcbiAgICAgICAgY29uc3QgeFJNaW4gPSBNYXRoLm1heCgwLCBNYXRoLmNlaWwoeFJDb3JuZXIgLyBvcmlnU3RyaWRlKSk7XG4gICAgICAgIGNvbnN0IHhSTWF4ID0gTWF0aC5taW4oeFJvd3MsIChmU2l6ZSArIHhSQ29ybmVyKSAvIG9yaWdTdHJpZGUpO1xuXG4gICAgICAgIGZvciAobGV0IHlDID0gMDsgeUMgPCB5LnNoYXBlWzFdOyArK3lDKSB7XG4gICAgICAgICAgY29uc3QgeENDb3JuZXIgPSB5QyAtIHBhZDtcbiAgICAgICAgICBjb25zdCB4Q01pbiA9IE1hdGgubWF4KDAsIE1hdGguY2VpbCh4Q0Nvcm5lciAvIG9yaWdTdHJpZGUpKTtcbiAgICAgICAgICBjb25zdCB4Q01heCA9IE1hdGgubWluKHhDb2xzLCAoZlNpemUgKyB4Q0Nvcm5lcikgLyBvcmlnU3RyaWRlKTtcblxuICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB4UiA9IHhSTWluOyB4UiA8IHhSTWF4OyArK3hSKSB7XG4gICAgICAgICAgICBjb25zdCB3UiA9IHhSICogb3JpZ1N0cmlkZSAtIHhSQ29ybmVyO1xuXG4gICAgICAgICAgICBmb3IgKGxldCB4QyA9IHhDTWluOyB4QyA8IHhDTWF4OyArK3hDKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHdDID0geEMgKiBvcmlnU3RyaWRlIC0geENDb3JuZXI7XG5cbiAgICAgICAgICAgICAgZm9yIChsZXQgZDEgPSAwOyBkMSA8IG9yaWdPdXRwdXREZXB0aDsgKytkMSkge1xuICAgICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geC5nZXQoeFIsIHhDLCBkMSk7XG4gICAgICAgICAgICAgICAgY29uc3Qgd2VpZ2h0ID1cbiAgICAgICAgICAgICAgICAgICAgd2VpZ2h0cy5nZXQoZlNpemUgLSAxIC0gd1IsIGZTaXplIC0gMSAtIHdDLCBkMiwgZDEpO1xuICAgICAgICAgICAgICAgIGRvdFByb2QgKz0gcGl4ZWwgKiB3ZWlnaHQ7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgY29uc3QgYmlhcyA9IGJpYXNlcyAhPSBudWxsID8gYmlhc2VzLmdldChkMikgOiAwO1xuICAgICAgICAgIHkuc2V0KGRvdFByb2QgKyBiaWFzLCB5UiwgeUMsIGQyKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfVxuXG4gIC8qKlxuICAgKiBpbWFnZSBpcyBvZiBzaGFwZSBbciwgYywgZDFdLlxuICAgKiB3ZWlnaHRzIGlzIG9mIHNoYXBlIFtGLCBGLCBkMSwgZDJdLlxuICAgKi9cbiAgcHJvdGVjdGVkIGNvbnYyZFRyYW5zcG9zZVNoYWRlckxpa2UoXG4gICAgICB4OiBBcnJheTNELCBvcmlnV2VpZ2h0czogQXJyYXk0RCwgb3JpZ1N0cmlkZTogbnVtYmVyLFxuICAgICAgb3JpZ1BhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgY29uc3QgZlNpemUgPSBvcmlnV2VpZ2h0cy5zaGFwZVswXTtcbiAgICBjb25zdCBwYWQgPSBmU2l6ZSAtIDEgLSBvcmlnUGFkO1xuICAgIGNvbnN0IG9yaWdJbnB1dERlcHRoID0gb3JpZ1dlaWdodHMuc2hhcGVbMl07XG4gICAgY29uc3Qgb3JpZ091dHB1dERlcHRoID0gb3JpZ1dlaWdodHMuc2hhcGVbM107XG4gICAgY29uc3QgeFJvd3MgPSB4LnNoYXBlWzBdO1xuICAgIGNvbnN0IHhDb2xzID0geC5zaGFwZVsxXTtcblxuICAgIC8vIERpbGF0ZSB0aGUgaW5wdXQuXG4gICAgY29uc3QgeFJvd3NEaWxhdGVkID0gKHhSb3dzIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcbiAgICBjb25zdCB4Q29sc0RpbGF0ZWQgPSAoeENvbHMgLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuXG4gICAgY29uc3Qgb3V0cHV0U2hhcGUgPSBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgIFt4Um93c0RpbGF0ZWQsIHhDb2xzRGlsYXRlZCwgb3JpZ091dHB1dERlcHRoXSwgZlNpemUsIG9yaWdJbnB1dERlcHRoLCAxLFxuICAgICAgICBwYWQpO1xuICAgIGNvbnN0IHkgPSBBcnJheTNELnplcm9zKG91dHB1dFNoYXBlKTtcblxuICAgIGZvciAobGV0IGQyID0gMDsgZDIgPCBvcmlnSW5wdXREZXB0aDsgKytkMikge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IHkuc2hhcGVbMF07ICsreVIpIHtcbiAgICAgICAgZm9yIChsZXQgeUMgPSAwOyB5QyA8IHkuc2hhcGVbMV07ICsreUMpIHtcbiAgICAgICAgICAvLyBTaGFkZXIgY29kZSBiZWdpbnMuXG4gICAgICAgICAgY29uc3QgeFJDb3JuZXIgPSB5UiAtIHBhZDtcbiAgICAgICAgICBjb25zdCB4Q0Nvcm5lciA9IHlDIC0gcGFkO1xuICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB3UiA9IDA7IHdSIDwgZlNpemU7ICsrd1IpIHtcbiAgICAgICAgICAgIGNvbnN0IHhSID0gKHhSQ29ybmVyICsgd1IpIC8gb3JpZ1N0cmlkZTtcbiAgICAgICAgICAgIGlmICh4UiA8IDAgfHwgeFIgPj0geFJvd3MgfHwgTWF0aC5mbG9vcih4UikgIT09IHhSKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuICAgICAgICAgICAgZm9yIChsZXQgd0MgPSAwOyB3QyA8IGZTaXplOyArK3dDKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHhDID0gKHhDQ29ybmVyICsgd0MpIC8gb3JpZ1N0cmlkZTtcbiAgICAgICAgICAgICAgaWYgKHhDIDwgMCB8fCB4QyA+PSB4Q29scyB8fCBNYXRoLmZsb29yKHhDKSAhPT0geEMpIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICBmb3IgKGxldCBkMSA9IDA7IGQxIDwgb3JpZ091dHB1dERlcHRoOyArK2QxKSB7XG4gICAgICAgICAgICAgICAgY29uc3QgcGl4ZWwgPSB4LmdldCh4UiwgeEMsIGQxKTtcbiAgICAgICAgICAgICAgICBjb25zdCB3ZWlnaHQgPVxuICAgICAgICAgICAgICAgICAgICBvcmlnV2VpZ2h0cy5nZXQoZlNpemUgLSAxIC0gd1IsIGZTaXplIC0gMSAtIHdDLCBkMiwgZDEpO1xuICAgICAgICAgICAgICAgIGRvdFByb2QgKz0gcGl4ZWwgKiB3ZWlnaHQ7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgeS5zZXQoZG90UHJvZCwgeVIsIHlDLCBkMik7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIHk7XG4gIH1cblxuICBjb252MmREZXJXZWlnaHRzKFxuICAgICAgeDogQXJyYXkzRCwgZFk6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLFxuICAgICAgemVyb1BhZDogbnVtYmVyKTogQXJyYXk0RCB7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IHguc2hhcGVbMl07XG4gICAgY29uc3Qgb3V0cHV0RGVwdGggPSBkWS5zaGFwZVsyXTtcbiAgICBjb25zdCB3ZWlnaHRzU2hhcGUgPVxuICAgICAgICBjb252X3V0aWwuY29tcHV0ZVdlaWdodHNTaGFwZTREKGlucHV0RGVwdGgsIG91dHB1dERlcHRoLCBmU2l6ZSk7XG4gICAgY29uc3QgZFcgPSBBcnJheTRELnplcm9zKHdlaWdodHNTaGFwZSk7XG5cbiAgICBjb25zdCB5TnVtUm93cyA9IGRZLnNoYXBlWzBdO1xuICAgIGNvbnN0IHlOdW1Db2xzID0gZFkuc2hhcGVbMV07XG4gICAgY29uc3QgeE51bVJvd3MgPSB4LnNoYXBlWzBdO1xuICAgIGNvbnN0IHhOdW1Db2xzID0geC5zaGFwZVsxXTtcblxuICAgIGZvciAobGV0IHdSID0gMDsgd1IgPCBmU2l6ZTsgKyt3Uikge1xuICAgICAgY29uc3QgeVJNaW4gPSBNYXRoLm1heCgwLCBNYXRoLmNlaWwoKHplcm9QYWQgLSB3UikgLyBzdHJpZGUpKTtcbiAgICAgIGNvbnN0IHlSTWF4ID0gTWF0aC5taW4oeU51bVJvd3MsICh4TnVtUm93cyArIHplcm9QYWQgLSB3UikgLyBzdHJpZGUpO1xuXG4gICAgICBmb3IgKGxldCB3QyA9IDA7IHdDIDwgZlNpemU7ICsrd0MpIHtcbiAgICAgICAgY29uc3QgeUNNaW4gPSBNYXRoLm1heCgwLCBNYXRoLmNlaWwoKHplcm9QYWQgLSB3QykgLyBzdHJpZGUpKTtcbiAgICAgICAgY29uc3QgeUNNYXggPSBNYXRoLm1pbih5TnVtQ29scywgKHhOdW1Db2xzICsgemVyb1BhZCAtIHdDKSAvIHN0cmlkZSk7XG5cbiAgICAgICAgZm9yIChsZXQgZDEgPSAwOyBkMSA8IGlucHV0RGVwdGg7ICsrZDEpIHtcbiAgICAgICAgICBmb3IgKGxldCBkMiA9IDA7IGQyIDwgb3V0cHV0RGVwdGg7ICsrZDIpIHtcbiAgICAgICAgICAgIC8vIE5lZWQgdG8gY29udm9sdmUuXG4gICAgICAgICAgICBsZXQgZG90UHJvZCA9IDA7XG4gICAgICAgICAgICBmb3IgKGxldCB5UiA9IHlSTWluOyB5UiA8IHlSTWF4OyArK3lSKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHhSID0gd1IgKyB5UiAqIHN0cmlkZSAtIHplcm9QYWQ7XG4gICAgICAgICAgICAgIGZvciAobGV0IHlDID0geUNNaW47IHlDIDwgeUNNYXg7ICsreUMpIHtcbiAgICAgICAgICAgICAgICBjb25zdCB4QyA9IHdDICsgeUMgKiBzdHJpZGUgLSB6ZXJvUGFkO1xuICAgICAgICAgICAgICAgIGRvdFByb2QgKz0geC5nZXQoeFIsIHhDLCBkMSkgKiBkWS5nZXQoeVIsIHlDLCBkMik7XG4gICAgICAgICAgICAgIH1cbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGRXLnNldChkb3RQcm9kLCB3Uiwgd0MsIGQxLCBkMik7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBkVztcbiAgfVxuXG4gIGNvbnYyZERlckJpYXMoZFk6IEFycmF5M0QpOiBBcnJheTFEIHtcbiAgICBjb25zdCBvdXRwdXREZXB0aCA9IGRZLnNoYXBlWzJdO1xuICAgIGNvbnN0IG51bVJvd3MgPSBkWS5zaGFwZVswXTtcbiAgICBjb25zdCBudW1Db2xzID0gZFkuc2hhcGVbMV07XG4gICAgY29uc3QgdmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShvdXRwdXREZXB0aCk7XG4gICAgZm9yIChsZXQgZDIgPSAwOyBkMiA8IG91dHB1dERlcHRoOyArK2QyKSB7XG4gICAgICBsZXQgc3VtID0gMDtcbiAgICAgIGZvciAobGV0IHIgPSAwOyByIDwgbnVtUm93czsgKytyKSB7XG4gICAgICAgIGZvciAobGV0IGMgPSAwOyBjIDwgbnVtQ29sczsgKytjKSB7XG4gICAgICAgICAgc3VtICs9IGRZLmdldChyLCBjLCBkMik7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIHZhbHVlc1tkMl0gPSBzdW07XG4gICAgfVxuICAgIHJldHVybiBBcnJheTFELm5ldyh2YWx1ZXMpO1xuICB9XG5cbiAgcHJvdGVjdGVkIHN3aXRjaERpbUludGVybmFsPFQgZXh0ZW5kcyBOREFycmF5Pih0OiBULCBuZXdEaW06IG51bWJlcltdKTogVCB7XG4gICAgY29uc3QgbmV3U2hhcGU6IG51bWJlcltdID0gbmV3IEFycmF5KHQucmFuayk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBuZXdTaGFwZS5sZW5ndGg7IGkrKykge1xuICAgICAgbmV3U2hhcGVbaV0gPSB0LnNoYXBlW25ld0RpbVtpXV07XG4gICAgfVxuICAgIGNvbnN0IHJlc3VsdFZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkodC5zaXplKTtcbiAgICBjb25zdCB2YWx1ZXMgPSB0LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IHJlc3VsdCA9IE5EQXJyYXkubWFrZTxUPihuZXdTaGFwZSwge3ZhbHVlczogcmVzdWx0VmFsdWVzfSk7XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCB0LnNpemU7ICsraSkge1xuICAgICAgY29uc3QgbG9jID0gdC5pbmRleFRvTG9jKGkpO1xuXG4gICAgICAvLyBQZXJtdXRlIGxvY2F0aW9uLlxuICAgICAgY29uc3QgbmV3TG9jOiBudW1iZXJbXSA9IG5ldyBBcnJheShsb2MubGVuZ3RoKTtcbiAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbmV3TG9jLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgIG5ld0xvY1tpXSA9IGxvY1tuZXdEaW1baV1dO1xuICAgICAgfVxuXG4gICAgICBjb25zdCBuZXdJbmRleCA9IHJlc3VsdC5sb2NUb0luZGV4KG5ld0xvYyk7XG4gICAgICByZXN1bHRWYWx1ZXNbbmV3SW5kZXhdID0gdmFsdWVzW2ldO1xuICAgIH1cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHJpdmF0ZSBwb29sKFxuICAgICAgeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyLFxuICAgICAgcG9vbFR5cGU6ICdtYXgnfCdtaW4nfCdhdmcnKSB7XG4gICAgY29uc3QgW3hSb3dzLCB4Q29scywgZGVwdGhdID0geC5zaGFwZTtcbiAgICBjb25zdCBvdXRwdXRTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgICAgW3hSb3dzLCB4Q29scywgZGVwdGhdLCBmU2l6ZSwgZGVwdGgsIHN0cmlkZSwgcGFkKTtcbiAgICBjb25zdCB5ID0gQXJyYXkzRC56ZXJvcyhvdXRwdXRTaGFwZSk7XG4gICAgZm9yIChsZXQgZCA9IDA7IGQgPCBkZXB0aDsgKytkKSB7XG4gICAgICBmb3IgKGxldCB5UiA9IDA7IHlSIDwgeS5zaGFwZVswXTsgKyt5Uikge1xuICAgICAgICBjb25zdCB4UkNvcm5lciA9IHlSICogc3RyaWRlIC0gcGFkO1xuICAgICAgICBjb25zdCB4Uk1pbiA9IE1hdGgubWF4KDAsIHhSQ29ybmVyKTtcbiAgICAgICAgY29uc3QgeFJNYXggPSBNYXRoLm1pbih4Um93cywgZlNpemUgKyB4UkNvcm5lcik7XG4gICAgICAgIGZvciAobGV0IHlDID0gMDsgeUMgPCB5LnNoYXBlWzFdOyArK3lDKSB7XG4gICAgICAgICAgY29uc3QgeENDb3JuZXIgPSB5QyAqIHN0cmlkZSAtIHBhZDtcbiAgICAgICAgICBjb25zdCB4Q01pbiA9IE1hdGgubWF4KDAsIHhDQ29ybmVyKTtcbiAgICAgICAgICBjb25zdCB4Q01heCA9IE1hdGgubWluKHhDb2xzLCBmU2l6ZSArIHhDQ29ybmVyKTtcblxuXG4gICAgICAgICAgbGV0IG1pbk1heFZhbHVlID1cbiAgICAgICAgICAgICAgKHBvb2xUeXBlID09PSAnbWF4JyA/IE51bWJlci5ORUdBVElWRV9JTkZJTklUWSA6XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBOdW1iZXIuUE9TSVRJVkVfSU5GSU5JVFkpO1xuICAgICAgICAgIGxldCBhdmdWYWx1ZSA9IDA7XG5cbiAgICAgICAgICBmb3IgKGxldCB4UiA9IHhSTWluOyB4UiA8IHhSTWF4OyArK3hSKSB7XG4gICAgICAgICAgICBmb3IgKGxldCB4QyA9IHhDTWluOyB4QyA8IHhDTWF4OyArK3hDKSB7XG4gICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geC5nZXQoeFIsIHhDLCBkKTtcbiAgICAgICAgICAgICAgaWYgKGlzTmFOKHBpeGVsKSkge1xuICAgICAgICAgICAgICAgIG1pbk1heFZhbHVlID0gTmFOO1xuICAgICAgICAgICAgICAgIGF2Z1ZhbHVlID0gTmFOO1xuICAgICAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIGlmICgocG9vbFR5cGUgPT09ICdtYXgnICYmIHBpeGVsID4gbWluTWF4VmFsdWUpIHx8XG4gICAgICAgICAgICAgICAgICAocG9vbFR5cGUgPT09ICdtaW4nICYmIHBpeGVsIDwgbWluTWF4VmFsdWUpKSB7XG4gICAgICAgICAgICAgICAgbWluTWF4VmFsdWUgPSBwaXhlbDtcbiAgICAgICAgICAgICAgfSBlbHNlIGlmIChwb29sVHlwZSA9PT0gJ2F2ZycpIHtcbiAgICAgICAgICAgICAgICBhdmdWYWx1ZSArPSBwaXhlbCAvIChmU2l6ZSAqIGZTaXplKTtcbiAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgfVxuICAgICAgICAgICAgaWYgKGlzTmFOKG1pbk1heFZhbHVlKSkge1xuICAgICAgICAgICAgICBicmVhaztcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgICAgeS5zZXQocG9vbFR5cGUgPT09ICdhdmcnID8gYXZnVmFsdWUgOiBtaW5NYXhWYWx1ZSwgeVIsIHlDLCBkKTtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgICByZXR1cm4geTtcbiAgfVxuXG4gIHByb3RlY3RlZCBtYXhQb29sSW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICByZXR1cm4gdGhpcy5wb29sKHgsIGZTaXplLCBzdHJpZGUsIHBhZCwgJ21heCcpO1xuICB9XG5cbiAgbWF4UG9vbFBvc2l0aW9ucyh4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBzdHJpZGU6IG51bWJlciwgcGFkOiBudW1iZXIpIHtcbiAgICBjb25zdCBbeFJvd3MsIHhDb2xzLCBkZXB0aF0gPSB4LnNoYXBlO1xuICAgIGNvbnN0IG91dHB1dFNoYXBlID1cbiAgICAgICAgY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKHguc2hhcGUsIGZTaXplLCBkZXB0aCwgc3RyaWRlLCBwYWQpO1xuICAgIGNvbnN0IG1heFBvc2l0aW9ucyA9IEFycmF5M0QuemVyb3Mob3V0cHV0U2hhcGUpO1xuICAgIGZvciAobGV0IGQgPSAwOyBkIDwgZGVwdGg7ICsrZCkge1xuICAgICAgZm9yIChsZXQgeVIgPSAwOyB5UiA8IG91dHB1dFNoYXBlWzBdOyArK3lSKSB7XG4gICAgICAgIGNvbnN0IHhSQ29ybmVyID0geVIgKiBzdHJpZGUgLSBwYWQ7XG4gICAgICAgIGNvbnN0IHhSTWluID0gTWF0aC5tYXgoMCwgeFJDb3JuZXIpO1xuICAgICAgICBjb25zdCB4Uk1heCA9IE1hdGgubWluKHhSb3dzLCBmU2l6ZSArIHhSQ29ybmVyKTtcbiAgICAgICAgZm9yIChsZXQgeUMgPSAwOyB5QyA8IG91dHB1dFNoYXBlWzFdOyArK3lDKSB7XG4gICAgICAgICAgY29uc3QgeENDb3JuZXIgPSB5QyAqIHN0cmlkZSAtIHBhZDtcbiAgICAgICAgICBjb25zdCB4Q01pbiA9IE1hdGgubWF4KDAsIHhDQ29ybmVyKTtcbiAgICAgICAgICBjb25zdCB4Q01heCA9IE1hdGgubWluKHhDb2xzLCBmU2l6ZSArIHhDQ29ybmVyKTtcbiAgICAgICAgICBsZXQgbWF4VmFsdWUgPSBOdW1iZXIuTkVHQVRJVkVfSU5GSU5JVFk7XG4gICAgICAgICAgbGV0IG1heFBvc2l0aW9uID0gLTE7XG4gICAgICAgICAgZm9yIChsZXQgeFIgPSB4Uk1pbjsgeFIgPCB4Uk1heDsgKyt4Uikge1xuICAgICAgICAgICAgY29uc3Qgd1IgPSB4UiAtIHhSQ29ybmVyO1xuICAgICAgICAgICAgZm9yIChsZXQgeEMgPSB4Q01pbjsgeEMgPCB4Q01heDsgKyt4Qykge1xuICAgICAgICAgICAgICBjb25zdCB3QyA9IHhDIC0geENDb3JuZXI7XG4gICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0geC5nZXQoeFIsIHhDLCBkKTtcbiAgICAgICAgICAgICAgaWYgKHBpeGVsID4gbWF4VmFsdWUpIHtcbiAgICAgICAgICAgICAgICBtYXhWYWx1ZSA9IHBpeGVsO1xuICAgICAgICAgICAgICAgIG1heFBvc2l0aW9uID0gd1IgKiBmU2l6ZSArIHdDO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIG1heFBvc2l0aW9ucy5zZXQobWF4UG9zaXRpb24sIHlSLCB5QywgZCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG1heFBvc2l0aW9ucztcbiAgfVxuXG4gIHByb3RlY3RlZCBtYXhQb29sQmFja3Byb3BJbnRlcm5hbChcbiAgICAgIGR5OiBBcnJheTNELCB4OiBBcnJheTNELCBmU2l6ZTogbnVtYmVyLCBvcmlnU3RyaWRlOiBudW1iZXIsXG4gICAgICBvcmlnUGFkOiBudW1iZXIpOiBBcnJheTNEIHtcbiAgICBjb25zdCBtYXhQb3NpdGlvbnMgPSB0aGlzLm1heFBvb2xQb3NpdGlvbnMoeCwgZlNpemUsIG9yaWdTdHJpZGUsIG9yaWdQYWQpO1xuICAgIGNvbnN0IHBhZCA9IGZTaXplIC0gMSAtIG9yaWdQYWQ7XG4gICAgY29uc3QgW2R5Um93cywgZHlDb2xzLCBkZXB0aF0gPSBkeS5zaGFwZTtcblxuICAgIC8vIERpbGF0ZSB0aGUgaW5wdXQuXG4gICAgY29uc3QgZHlSb3dzRGlsYXRlZCA9IChkeVJvd3MgLSAxKSAqIG9yaWdTdHJpZGUgKyAxO1xuICAgIGNvbnN0IGR4Q29sc0RpbGF0ZWQgPSAoZHlDb2xzIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcblxuICAgIGNvbnN0IG91dHB1dFNoYXBlID0gY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKFxuICAgICAgICBbZHlSb3dzRGlsYXRlZCwgZHhDb2xzRGlsYXRlZCwgZGVwdGhdLCBmU2l6ZSwgZGVwdGgsIDEsIHBhZCk7XG4gICAgY29uc3QgZHggPSBBcnJheTNELnplcm9zKG91dHB1dFNoYXBlKTtcblxuICAgIGZvciAobGV0IGQgPSAwOyBkIDwgZGVwdGg7ICsrZCkge1xuICAgICAgZm9yIChsZXQgZHhSID0gMDsgZHhSIDwgZHguc2hhcGVbMF07ICsrZHhSKSB7XG4gICAgICAgIGZvciAobGV0IGR4QyA9IDA7IGR4QyA8IGR4LnNoYXBlWzFdOyArK2R4Qykge1xuICAgICAgICAgIC8vIFNoYWRlciBjb2RlIGJlZ2lucy5cbiAgICAgICAgICBjb25zdCBkeVJDb3JuZXIgPSBkeFIgLSBwYWQ7XG4gICAgICAgICAgY29uc3QgZHlDQ29ybmVyID0gZHhDIC0gcGFkO1xuICAgICAgICAgIGxldCBkb3RQcm9kID0gMDtcbiAgICAgICAgICBmb3IgKGxldCB3UiA9IDA7IHdSIDwgZlNpemU7ICsrd1IpIHtcbiAgICAgICAgICAgIGNvbnN0IGR5UiA9IChkeVJDb3JuZXIgKyB3UikgLyBvcmlnU3RyaWRlO1xuICAgICAgICAgICAgaWYgKGR5UiA8IDAgfHwgZHlSID49IGR5Um93cyB8fCBNYXRoLmZsb29yKGR5UikgIT09IGR5Uikge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICAgIGZvciAobGV0IHdDID0gMDsgd0MgPCBmU2l6ZTsgKyt3Qykge1xuICAgICAgICAgICAgICBjb25zdCBkeUMgPSAoZHlDQ29ybmVyICsgd0MpIC8gb3JpZ1N0cmlkZTtcbiAgICAgICAgICAgICAgaWYgKGR5QyA8IDAgfHwgZHlDID49IGR5Q29scyB8fCBNYXRoLmZsb29yKGR5QykgIT09IGR5Qykge1xuICAgICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICAgIGNvbnN0IG1heFBvcyA9IGZTaXplICogZlNpemUgLSAxIC0gbWF4UG9zaXRpb25zLmdldChkeVIsIGR5QywgZCk7XG4gICAgICAgICAgICAgIGNvbnN0IGN1clBvcyA9IHdSICogZlNpemUgKyB3QztcblxuICAgICAgICAgICAgICBjb25zdCBtYXNrID0gbWF4UG9zID09PSBjdXJQb3MgPyAxIDogMDtcbiAgICAgICAgICAgICAgaWYgKG1hc2sgPT09IDApIHtcbiAgICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICAgIGNvbnN0IHBpeGVsID0gZHkuZ2V0KGR5UiwgZHlDLCBkKTtcbiAgICAgICAgICAgICAgZG90UHJvZCArPSBwaXhlbCAqIG1hc2s7XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICAgIGR4LnNldChkb3RQcm9kLCBkeFIsIGR4QywgZCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIGR4O1xuICB9XG5cbiAgcHJvdGVjdGVkIG1pblBvb2xJbnRlcm5hbChcbiAgICAgIHg6IEFycmF5M0QsIGZTaXplOiBudW1iZXIsIHN0cmlkZTogbnVtYmVyLCBwYWQ6IG51bWJlcik6IEFycmF5M0Qge1xuICAgIHJldHVybiB0aGlzLnBvb2woeCwgZlNpemUsIHN0cmlkZSwgcGFkLCAnbWluJyk7XG4gIH1cblxuICBwcm90ZWN0ZWQgYXZnUG9vbEludGVybmFsKFxuICAgICAgeDogQXJyYXkzRCwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgcmV0dXJuIHRoaXMucG9vbCh4LCBmU2l6ZSwgc3RyaWRlLCBwYWQsICdhdmcnKTtcbiAgfVxuXG4gIHByb3RlY3RlZCByZXNpemVCaWxpbmVhcjNESW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBuZXdTaGFwZTJEOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgICAgYWxpZ25Db3JuZXJzOiBib29sZWFuKTogQXJyYXkzRCB7XG4gICAgY29uc3Qgb3V0cHV0ID0gQXJyYXkzRC56ZXJvcyhbbmV3U2hhcGUyRFswXSwgbmV3U2hhcGUyRFsxXSwgeC5zaGFwZVsyXV0pO1xuXG4gICAgY29uc3QgZWZmZWN0aXZlSW5wdXRTaXplID1cbiAgICAgICAgYWxpZ25Db3JuZXJzID8gW3guc2hhcGVbMF0gLSAxLCB4LnNoYXBlWzFdIC0gMSwgeC5zaGFwZVsyXV0gOiB4LnNoYXBlO1xuICAgIGNvbnN0IGVmZmVjdGl2ZU91dHB1dFNpemUgPSBhbGlnbkNvcm5lcnMgP1xuICAgICAgICBbb3V0cHV0LnNoYXBlWzBdIC0gMSwgb3V0cHV0LnNoYXBlWzFdIC0gMSwgb3V0cHV0LnNoYXBlWzJdXSA6XG4gICAgICAgIG91dHB1dC5zaGFwZTtcbiAgICBmb3IgKGxldCByID0gMDsgciA8IG91dHB1dC5zaGFwZVswXTsgcisrKSB7XG4gICAgICBmb3IgKGxldCBjID0gMDsgYyA8IG91dHB1dC5zaGFwZVsxXTsgYysrKSB7XG4gICAgICAgIGZvciAobGV0IGQgPSAwOyBkIDwgb3V0cHV0LnNoYXBlWzJdOyBkKyspIHtcbiAgICAgICAgICAvLyBCZWdpbiBzaGFkZXIuXG5cbiAgICAgICAgICAvLyBDb21wdXRlIHRoZSBmcmFjdGlvbmFsIGluZGV4IG9mIHRoZSBzb3VyY2UuXG4gICAgICAgICAgY29uc3Qgc291cmNlRnJhY1JvdyA9XG4gICAgICAgICAgICAgIChlZmZlY3RpdmVJbnB1dFNpemVbMF0pICogciAvIChlZmZlY3RpdmVPdXRwdXRTaXplWzBdKTtcbiAgICAgICAgICBjb25zdCBzb3VyY2VGcmFjQ29sID1cbiAgICAgICAgICAgICAgKGVmZmVjdGl2ZUlucHV0U2l6ZVsxXSkgKiBjIC8gKGVmZmVjdGl2ZU91dHB1dFNpemVbMV0pO1xuXG4gICAgICAgICAgY29uc3Qgc291cmNlUm93Rmxvb3IgPSBNYXRoLmZsb29yKHNvdXJjZUZyYWNSb3cpO1xuICAgICAgICAgIGNvbnN0IHNvdXJjZVJvd0NlaWwgPVxuICAgICAgICAgICAgICBNYXRoLm1pbih4LnNoYXBlWzBdIC0gMSwgTWF0aC5jZWlsKHNvdXJjZUZyYWNSb3cpKTtcbiAgICAgICAgICBjb25zdCBzb3VyY2VDb2xGbG9vciA9IE1hdGguZmxvb3Ioc291cmNlRnJhY0NvbCk7XG4gICAgICAgICAgY29uc3Qgc291cmNlQ29sQ2VpbCA9XG4gICAgICAgICAgICAgIE1hdGgubWluKHguc2hhcGVbMV0gLSAxLCBNYXRoLmNlaWwoc291cmNlRnJhY0NvbCkpO1xuXG4gICAgICAgICAgY29uc3QgdG9wTGVmdCA9IHguZ2V0KHNvdXJjZVJvd0Zsb29yLCBzb3VyY2VDb2xGbG9vciwgZCk7XG4gICAgICAgICAgY29uc3QgYm90dG9tTGVmdCA9IHguZ2V0KHNvdXJjZVJvd0NlaWwsIHNvdXJjZUNvbEZsb29yLCBkKTtcbiAgICAgICAgICBjb25zdCB0b3BSaWdodCA9IHguZ2V0KHNvdXJjZVJvd0Zsb29yLCBzb3VyY2VDb2xDZWlsLCBkKTtcbiAgICAgICAgICBjb25zdCBib3R0b21SaWdodCA9IHguZ2V0KHNvdXJjZVJvd0NlaWwsIHNvdXJjZUNvbENlaWwsIGQpO1xuXG4gICAgICAgICAgY29uc3Qgcm93RnJhYyA9IHNvdXJjZUZyYWNSb3cgLSBzb3VyY2VSb3dGbG9vcjtcbiAgICAgICAgICBjb25zdCBjb2xGcmFjID0gc291cmNlRnJhY0NvbCAtIHNvdXJjZUNvbEZsb29yO1xuXG4gICAgICAgICAgY29uc3QgdG9wID0gdG9wTGVmdCArICh0b3BSaWdodCAtIHRvcExlZnQpICogY29sRnJhYztcbiAgICAgICAgICBjb25zdCBib3R0b20gPSBib3R0b21MZWZ0ICsgKGJvdHRvbVJpZ2h0IC0gYm90dG9tTGVmdCkgKiBjb2xGcmFjO1xuICAgICAgICAgIGNvbnN0IG5ld1ZhbHVlID0gdG9wICsgKGJvdHRvbSAtIHRvcCkgKiByb3dGcmFjO1xuXG4gICAgICAgICAgb3V0cHV0LnNldChuZXdWYWx1ZSwgciwgYywgZCk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gb3V0cHV0O1xuICB9XG5cbiAgcHJvdGVjdGVkIGJhdGNoTm9ybWFsaXphdGlvbjNESW50ZXJuYWwoXG4gICAgICB4OiBBcnJheTNELCBtZWFuOiBBcnJheTNEfEFycmF5MUQsIHZhcmlhbmNlOiBBcnJheTNEfEFycmF5MUQsXG4gICAgICB2YXJpYW5jZUVwc2lsb24gPSAuMDAxLCBzY2FsZT86IEFycmF5M0R8QXJyYXkxRCxcbiAgICAgIG9mZnNldD86IEFycmF5M0R8QXJyYXkxRCk6IEFycmF5M0Qge1xuICAgIGNvbnN0IHhWYWx1ZXMgPSB4LmdldFZhbHVlcygpO1xuICAgIGNvbnN0IG1lYW5WYWx1ZXMgPSBtZWFuLmdldFZhbHVlcygpO1xuICAgIGNvbnN0IHZhcmlhbmNlVmFsdWVzID0gdmFyaWFuY2UuZ2V0VmFsdWVzKCk7XG4gICAgY29uc3Qgc2NhbGVWYWx1ZXMgPSBzY2FsZSA/IHNjYWxlLmdldFZhbHVlcygpIDogbmV3IEZsb2F0MzJBcnJheShbMV0pO1xuICAgIGNvbnN0IG9mZnNldFZhbHVlcyA9IG9mZnNldCA/IG9mZnNldC5nZXRWYWx1ZXMoKSA6IG5ldyBGbG9hdDMyQXJyYXkoWzBdKTtcbiAgICBjb25zdCBvdXRWYWx1ZXMgPSBuZXcgRmxvYXQzMkFycmF5KHhWYWx1ZXMubGVuZ3RoKTtcblxuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgeFZhbHVlcy5sZW5ndGg7IGkrKykge1xuICAgICAgb3V0VmFsdWVzW2ldID0gb2Zmc2V0VmFsdWVzW2kgJSBvZmZzZXRWYWx1ZXMubGVuZ3RoXSArXG4gICAgICAgICAgKHhWYWx1ZXNbaV0gLSBtZWFuVmFsdWVzW2kgJSBtZWFuVmFsdWVzLmxlbmd0aF0pICpcbiAgICAgICAgICAgICAgc2NhbGVWYWx1ZXNbaSAlIHNjYWxlVmFsdWVzLmxlbmd0aF0gL1xuICAgICAgICAgICAgICBNYXRoLnNxcnQoXG4gICAgICAgICAgICAgICAgICB2YXJpYW5jZVZhbHVlc1tpICUgdmFyaWFuY2VWYWx1ZXMubGVuZ3RoXSArIHZhcmlhbmNlRXBzaWxvbik7XG4gICAgfVxuICAgIHJldHVybiBOREFycmF5Lm1ha2U8QXJyYXkzRD4oeC5zaGFwZSwge3ZhbHVlczogb3V0VmFsdWVzfSk7XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi91dGlsJztcblxuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vd2ViZ2wvZ3BncHVfY29udGV4dCc7XG5pbXBvcnQge1RleHR1cmVNYW5hZ2VyfSBmcm9tICcuL3dlYmdsL3RleHR1cmVfbWFuYWdlcic7XG5pbXBvcnQgKiBhcyB3ZWJnbF91dGlsIGZyb20gJy4vd2ViZ2wvd2ViZ2xfdXRpbCc7XG5cbi8vIFRoZXNlIGdsb2JhbCB2YXJpYWJsZXMgbmVlZCB0byBiZSBpbml0aWFsaXplZCB0byBudWxsIHNvIHRoYXQgY2xvc3VyZSBrbm93c1xuLy8gbm90IHRvIHNlYWwgdGhlbS5cbi8qKiBAaGlkZGVuICovXG5leHBvcnQgbGV0IEdQR1BVOiBHUEdQVUNvbnRleHQgPSBudWxsITtcbi8qKiBAaGlkZGVuICovXG5leHBvcnQgbGV0IFRFWFRVUkVfTUFOQUdFUjogVGV4dHVyZU1hbmFnZXIgPSBudWxsITtcblxuLyoqIEBoaWRkZW4gKi9cbmV4cG9ydCBpbnRlcmZhY2UgTkRBcnJheURhdGEge1xuICB2YWx1ZXM/OiBGbG9hdDMyQXJyYXk7XG4gIHRleHR1cmU/OiBXZWJHTFRleHR1cmU7XG4gIC8qKiBbcm93cywgY29sdW1uc10gc2hhcGUgb2YgdGhlIHRleHR1cmUuICovXG4gIHRleHR1cmVTaGFwZVJDPzogW251bWJlciwgbnVtYmVyXTtcbn1cblxuLyoqIEBoaWRkZW4gKi9cbmV4cG9ydCBmdW5jdGlvbiBpbml0aWFsaXplR1BVKFxuICAgIGdwZ3B1OiBHUEdQVUNvbnRleHQsIHRleHR1cmVNYW5hZ2VyOiBUZXh0dXJlTWFuYWdlcikge1xuICBHUEdQVSA9IGdwZ3B1O1xuICBURVhUVVJFX01BTkFHRVIgPSB0ZXh0dXJlTWFuYWdlcjtcbn1cblxuZnVuY3Rpb24gdGhyb3dJZkdQVU5vdEluaXRpYWxpemVkKCkge1xuICBpZiAoR1BHUFUgPT0gbnVsbCB8fCBURVhUVVJFX01BTkFHRVIgPT0gbnVsbCkge1xuICAgIHRocm93IG5ldyBFcnJvcignR1BVIG5vdCBpbnRpYWxpemVkLicpO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBOREFycmF5IHtcbiAgLyoqIFRoZSBzaGFwZSBvZiB0aGUgbmRhcnJheS4gKi9cbiAgc2hhcGU6IG51bWJlcltdO1xuICAvKiogTnVtYmVyIG9mIGVsZW1lbnRzIGluIHRoZSBuZGFycmF5LiAqL1xuICBzaXplOiBudW1iZXI7XG5cbiAgLyoqXG4gICAqIE51bWJlciBvZiBlbGVtZW50cyB0byBza2lwIGluIGVhY2ggZGltZW5zaW9uIHdoZW4gaW5kZXhpbmcuIFNlZVxuICAgKiBodHRwczovL2RvY3Muc2NpcHkub3JnL2RvYy9udW1weS9yZWZlcmVuY2UvZ2VuZXJhdGVkXG4gICAqICAgICAvbnVtcHkubmRhcnJheS5zdHJpZGVzLmh0bWxcbiAgICovXG4gIHByb3RlY3RlZCBzdHJpZGVzOiBudW1iZXJbXTtcblxuICBwcml2YXRlIGRhdGE6IE5EQXJyYXlEYXRhO1xuXG4gIHByb3RlY3RlZCBjb25zdHJ1Y3RvcihzaGFwZTogbnVtYmVyW10sIGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgLy8gU2FuaXR5IGNoZWNrcy5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgZGF0YS52YWx1ZXMgIT0gbnVsbCB8fCBkYXRhLnRleHR1cmUgIT0gbnVsbCxcbiAgICAgICAgJ0VpdGhlciBgdmFsdWVzYCBvciBgdGV4dHVyZWAgbXVzdCBiZSBkZWZpbmVkJyk7XG5cbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgZGF0YS50ZXh0dXJlID09IG51bGwgfHwgKGRhdGEudGV4dHVyZVNoYXBlUkMgIT0gbnVsbCksXG4gICAgICAgICdgdGV4dHVyZVNoYXBlYCBtdXN0IGJlIGRlZmluZWQgd2hlbiBgdGV4dHVyZWAgaXMgZGVmaW5lZCcpO1xuXG4gICAgdGhpcy5zaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKTtcblxuICAgIGlmIChkYXRhLnZhbHVlcyAhPSBudWxsKSB7XG4gICAgICB1dGlsLmFzc2VydChcbiAgICAgICAgICB0aGlzLnNpemUgPT09IGRhdGEudmFsdWVzLmxlbmd0aCxcbiAgICAgICAgICAnQ29uc3RydWN0aW5nIG5kYXJyYXkgb2Ygc2hhcGUgKCcgKyB0aGlzLnNpemUgKyAnKSBzaG91bGQgbWF0Y2ggdGhlJyArXG4gICAgICAgICAgICAgICcgbGVuZ3RoIG9mIHZhbHVlcyAoJyArIGRhdGEudmFsdWVzLmxlbmd0aCArICcpJyk7XG4gICAgfVxuXG4gICAgdGhpcy5zaGFwZSA9IHNoYXBlO1xuICAgIHRoaXMuZGF0YSA9IGRhdGE7XG4gICAgY29uc3QgZGltID0gdGhpcy5zaGFwZS5sZW5ndGg7XG5cbiAgICBpZiAoZGltIDwgMikge1xuICAgICAgdGhpcy5zdHJpZGVzID0gW107XG4gICAgfSBlbHNlIHtcbiAgICAgIC8vIExhc3QgZGltZW5zaW9uIGhhcyBpbXBsaWNpdCBzdHJpZGUgb2YgMSwgdGh1cyBoYXZpbmcgRC0xIChpbnN0ZWFkIG9mIEQpXG4gICAgICAvLyBzdHJpZGVzLlxuICAgICAgdGhpcy5zdHJpZGVzID0gbmV3IEFycmF5KGRpbSAtIDEpO1xuICAgICAgdGhpcy5zdHJpZGVzW2RpbSAtIDJdID0gdGhpcy5zaGFwZVtkaW0gLSAxXTtcbiAgICAgIGZvciAobGV0IGkgPSBkaW0gLSAzOyBpID49IDA7IC0taSkge1xuICAgICAgICB0aGlzLnN0cmlkZXNbaV0gPSB0aGlzLnN0cmlkZXNbaSArIDFdICogdGhpcy5zaGFwZVtpICsgMV07XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgLyoqIENyZWF0ZXMgYSBuZGFycmF5IG9mIHplcm9zIHdpdGggdGhlIHNwZWNpZmllZCBzaGFwZS4gKi9cbiAgc3RhdGljIHplcm9zKHNoYXBlOiBudW1iZXJbXSk6IE5EQXJyYXkge1xuICAgIGNvbnN0IHZhbHVlcyA9IG5ldyBGbG9hdDMyQXJyYXkodXRpbC5zaXplRnJvbVNoYXBlKHNoYXBlKSk7XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZShzaGFwZSwge3ZhbHVlc30pO1xuICB9XG5cbiAgLyoqIENyZWF0ZXMgYSBuZGFycmF5IG9mIHplcm9zIHdpdGggdGhlIHNhbWUgc2hhcGUgYXMgdGhlIHNwZWNpZmllZCBuZGFycmF5LlxuICAgKi9cbiAgc3RhdGljIHplcm9zTGlrZTxUIGV4dGVuZHMgTkRBcnJheT4oYW5vdGhlcjogVCk6IFQge1xuICAgIHJldHVybiBOREFycmF5Lnplcm9zKGFub3RoZXIuc2hhcGUpIGFzIFQ7XG4gIH1cblxuICAvKiogQ3JlYXRlcyBhIG5kYXJyYXkgd2l0aCB0aGUgc2FtZSB2YWx1ZXMvc2hhcGUgYXMgdGhlIHNwZWNpZmllZCBuZGFycmF5LiAqL1xuICBzdGF0aWMgbGlrZTxUIGV4dGVuZHMgTkRBcnJheT4oYW5vdGhlcjogVCk6IFQge1xuICAgIGNvbnN0IHZhbHVlcyA9IGFub3RoZXIuZ2V0VmFsdWVzKCk7XG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihhbm90aGVyLnNoYXBlLCB7dmFsdWVzOiBuZXcgRmxvYXQzMkFycmF5KHZhbHVlcyl9KTtcbiAgfVxuXG4gIC8qKlxuICAgKiBNYWtlcyBhIG5ldyBuZGFycmF5IHdpdGggdGhlIHByb3ZpZGVkIHNoYXBlIGFuZCB2YWx1ZXMuIFZhbHVlcyBzaG91bGQgYmUgaW5cbiAgICogYSBmbGF0IGFycmF5LlxuICAgKi9cbiAgc3RhdGljIG1ha2U8VCBleHRlbmRzIE5EQXJyYXk+KHNoYXBlOiBudW1iZXJbXSwgZGF0YTogTkRBcnJheURhdGEpOiBUIHtcbiAgICBzd2l0Y2ggKHNoYXBlLmxlbmd0aCkge1xuICAgICAgY2FzZSAwOlxuICAgICAgICByZXR1cm4gbmV3IFNjYWxhcihkYXRhKSBhcyBUO1xuICAgICAgY2FzZSAxOlxuICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICAgIHJldHVybiBuZXcgQXJyYXkxRChkYXRhKSBhcyBhbnk7XG4gICAgICBjYXNlIDI6XG4gICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgcmV0dXJuIG5ldyBBcnJheTJEKHNoYXBlIGFzIFtudW1iZXIsIG51bWJlcl0sIGRhdGEpIGFzIGFueTtcbiAgICAgIGNhc2UgMzpcbiAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgICByZXR1cm4gbmV3IEFycmF5M0Qoc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCBkYXRhKSBhcyBhbnk7XG4gICAgICBjYXNlIDQ6XG4gICAgICAgIHJldHVybiBuZXcgQXJyYXk0RChcbiAgICAgICAgICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICAgICAgICAgIHNoYXBlIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLCBkYXRhKSBhcyBhbnk7XG4gICAgICBkZWZhdWx0OlxuICAgICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICAgIHJldHVybiBuZXcgTkRBcnJheShzaGFwZSwgZGF0YSkgYXMgYW55O1xuICAgIH1cbiAgfVxuXG4gIC8qKiBSZXNoYXBlcyB0aGUgY3VycmVudCBuZGFycmF5IGludG8gdGhlIHByb3ZpZGVkIHNoYXBlLiAqL1xuICByZXNoYXBlPFQgZXh0ZW5kcyBOREFycmF5PihuZXdTaGFwZTogbnVtYmVyW10pOiBUIHtcbiAgICBpZiAodXRpbC5hcnJheXNFcXVhbCh0aGlzLnNoYXBlLCBuZXdTaGFwZSkpIHtcbiAgICAgIC8vIE5vLW9wLlxuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgcmV0dXJuIHRoaXMgYXMgYW55O1xuICAgIH1cblxuICAgIHV0aWwuYXNzZXJ0KFxuICAgICAgICB0aGlzLnNpemUgPT09IHV0aWwuc2l6ZUZyb21TaGFwZShuZXdTaGFwZSksXG4gICAgICAgICduZXcgc2hhcGUgYW5kIG9sZCBzaGFwZSBtdXN0IGhhdmUgdGhlIHNhbWUgbnVtYmVyIG9mIGVsZW1lbnRzLicpO1xuXG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihuZXdTaGFwZSwgdGhpcy5kYXRhKTtcbiAgfVxuXG4gIGFzU2NhbGFyKCk6IFNjYWxhciB7XG4gICAgdXRpbC5hc3NlcnQodGhpcy5zaXplID09PSAxLCAnVGhlIGFycmF5IG11c3QgaGF2ZSBvbmx5IDEgZWxlbWVudC4nKTtcbiAgICByZXR1cm4gdGhpcy5yZXNoYXBlPFNjYWxhcj4oW10pO1xuICB9XG5cbiAgYXMxRCgpOiBBcnJheTFEIHtcbiAgICByZXR1cm4gdGhpcy5yZXNoYXBlPEFycmF5MUQ+KFt0aGlzLnNpemVdKTtcbiAgfVxuXG4gIGFzMkQocm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBBcnJheTJEIHtcbiAgICByZXR1cm4gdGhpcy5yZXNoYXBlPEFycmF5MkQ+KFtyb3dzLCBjb2x1bW5zXSk7XG4gIH1cblxuICBhczNEKHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyLCBkZXB0aDogbnVtYmVyKTogQXJyYXkzRCB7XG4gICAgcmV0dXJuIHRoaXMucmVzaGFwZTxBcnJheTNEPihbcm93cywgY29sdW1ucywgZGVwdGhdKTtcbiAgfVxuXG4gIGFzNEQocm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIsIGRlcHRoOiBudW1iZXIsIGRlcHRoMjogbnVtYmVyKTogQXJyYXk0RCB7XG4gICAgcmV0dXJuIHRoaXMucmVzaGFwZTxBcnJheTREPihbcm93cywgY29sdW1ucywgZGVwdGgsIGRlcHRoMl0pO1xuICB9XG5cbiAgZ2V0IHJhbmsoKTogbnVtYmVyIHtcbiAgICByZXR1cm4gdGhpcy5zaGFwZS5sZW5ndGg7XG4gIH1cblxuICBnZXQoLi4ubG9jczogbnVtYmVyW10pIHtcbiAgICBsZXQgaW5kZXggPSBsb2NzW2xvY3MubGVuZ3RoIC0gMV07XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsb2NzLmxlbmd0aCAtIDE7ICsraSkge1xuICAgICAgaW5kZXggKz0gdGhpcy5zdHJpZGVzW2ldICogbG9jc1tpXTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuZ2V0VmFsdWVzKClbaW5kZXhdO1xuICB9XG5cbiAgYWRkKHZhbHVlOiBudW1iZXIsIC4uLmxvY3M6IG51bWJlcltdKSB7XG4gICAgdGhpcy5zZXQodGhpcy5nZXQoLi4ubG9jcykgKyB2YWx1ZSwgLi4ubG9jcyk7XG4gIH1cblxuICBzZXQodmFsdWU6IG51bWJlciwgLi4ubG9jczogbnVtYmVyW10pIHtcbiAgICBsZXQgaW5kZXggPSBsb2NzW2xvY3MubGVuZ3RoIC0gMV07XG4gICAgZm9yIChsZXQgaSA9IDA7IGkgPCBsb2NzLmxlbmd0aCAtIDE7ICsraSkge1xuICAgICAgaW5kZXggKz0gdGhpcy5zdHJpZGVzW2ldICogbG9jc1tpXTtcbiAgICB9XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVtpbmRleF0gPSB2YWx1ZTtcbiAgfVxuXG4gIGxvY1RvSW5kZXgobG9jczogbnVtYmVyW10pOiBudW1iZXIge1xuICAgIGxldCBpbmRleCA9IGxvY3NbbG9jcy5sZW5ndGggLSAxXTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IGxvY3MubGVuZ3RoIC0gMTsgKytpKSB7XG4gICAgICBpbmRleCArPSB0aGlzLnN0cmlkZXNbaV0gKiBsb2NzW2ldO1xuICAgIH1cbiAgICByZXR1cm4gaW5kZXg7XG4gIH1cblxuICBpbmRleFRvTG9jKGluZGV4OiBudW1iZXIpOiBudW1iZXJbXSB7XG4gICAgY29uc3QgbG9jczogbnVtYmVyW10gPSBuZXcgQXJyYXkodGhpcy5zaGFwZS5sZW5ndGgpO1xuICAgIGZvciAobGV0IGkgPSAwOyBpIDwgbG9jcy5sZW5ndGggLSAxOyArK2kpIHtcbiAgICAgIGxvY3NbaV0gPSBNYXRoLmZsb29yKGluZGV4IC8gdGhpcy5zdHJpZGVzW2ldKTtcbiAgICAgIGluZGV4IC09IGxvY3NbaV0gKiB0aGlzLnN0cmlkZXNbaV07XG4gICAgfVxuICAgIGxvY3NbbG9jcy5sZW5ndGggLSAxXSA9IGluZGV4O1xuICAgIHJldHVybiBsb2NzO1xuICB9XG5cbiAgZmlsbCh2YWx1ZTogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKS5maWxsKHZhbHVlKTtcbiAgfVxuXG4gIGdldERhdGEoKTogTkRBcnJheURhdGEge1xuICAgIHJldHVybiB0aGlzLmRhdGE7XG4gIH1cblxuICBnZXRWYWx1ZXMoKTogRmxvYXQzMkFycmF5IHtcbiAgICBpZiAodGhpcy5kYXRhLnZhbHVlcyA9PSBudWxsKSB7XG4gICAgICB0aHJvd0lmR1BVTm90SW5pdGlhbGl6ZWQoKTtcbiAgICAgIHRoaXMuZGF0YS52YWx1ZXMgPSBHUEdQVS5kb3dubG9hZE1hdHJpeEZyb21UZXh0dXJlKFxuICAgICAgICAgIHRoaXMuZGF0YS50ZXh0dXJlISwgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDIVswXSxcbiAgICAgICAgICB0aGlzLmRhdGEudGV4dHVyZVNoYXBlUkMhWzFdKTtcbiAgICAgIHRoaXMuZGlzcG9zZVRleHR1cmUoKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMuZGF0YS52YWx1ZXM7XG4gIH1cblxuICBwcml2YXRlIHVwbG9hZFRvR1BVKHByZWZlcnJlZFRleFNoYXBlPzogW251bWJlciwgbnVtYmVyXSkge1xuICAgIHRocm93SWZHUFVOb3RJbml0aWFsaXplZCgpO1xuICAgIHRoaXMuZGF0YS50ZXh0dXJlU2hhcGVSQyA9IHdlYmdsX3V0aWwuZ2V0VGV4dHVyZVNoYXBlRnJvbUxvZ2ljYWxTaGFwZShcbiAgICAgICAgR1BHUFUuZ2wsIHRoaXMuc2hhcGUsIHByZWZlcnJlZFRleFNoYXBlKTtcbiAgICB0aGlzLmRhdGEudGV4dHVyZSA9XG4gICAgICAgIFRFWFRVUkVfTUFOQUdFUi5hY3F1aXJlVGV4dHVyZSh0aGlzLmRhdGEudGV4dHVyZVNoYXBlUkMpO1xuXG4gICAgR1BHUFUudXBsb2FkTWF0cml4VG9UZXh0dXJlKFxuICAgICAgICB0aGlzLmRhdGEudGV4dHVyZSwgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDWzBdLFxuICAgICAgICB0aGlzLmRhdGEudGV4dHVyZVNoYXBlUkNbMV0sIHRoaXMuZGF0YS52YWx1ZXMhKTtcblxuICAgIHRoaXMuZGF0YS52YWx1ZXMgPSBudWxsITtcbiAgfVxuXG4gIGdldFRleHR1cmUocHJlZmVycmVkU2hhcGVSQz86IFtudW1iZXIsIG51bWJlcl0pOiBXZWJHTFRleHR1cmUge1xuICAgIGlmICh0aGlzLmRhdGEudGV4dHVyZSA9PSBudWxsKSB7XG4gICAgICB0aGlzLnVwbG9hZFRvR1BVKHByZWZlcnJlZFNoYXBlUkMpO1xuICAgIH1cbiAgICByZXR1cm4gdGhpcy5kYXRhLnRleHR1cmUhO1xuICB9XG5cbiAgZ2V0VGV4dHVyZVNoYXBlUkMocHJlZmVycmVkU2hhcGVSQz86IFtudW1iZXIsIG51bWJlcl0pOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgICBpZiAodGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDID09IG51bGwpIHtcbiAgICAgIHRoaXMudXBsb2FkVG9HUFUocHJlZmVycmVkU2hhcGVSQyk7XG4gICAgfVxuICAgIHJldHVybiB0aGlzLmRhdGEudGV4dHVyZVNoYXBlUkMhO1xuICB9XG5cbiAgZGlzcG9zZSgpOiB2b2lkIHtcbiAgICB0aGlzLmRhdGEudmFsdWVzID0gbnVsbCE7XG4gICAgdGhpcy5zaGFwZSA9IG51bGwhO1xuICAgIGlmICh0aGlzLmRhdGEudGV4dHVyZSAhPSBudWxsKSB7XG4gICAgICB0aGlzLmRpc3Bvc2VUZXh0dXJlKCk7XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSBkaXNwb3NlVGV4dHVyZSgpIHtcbiAgICB0aHJvd0lmR1BVTm90SW5pdGlhbGl6ZWQoKTtcbiAgICBURVhUVVJFX01BTkFHRVIucmVsZWFzZVRleHR1cmUoXG4gICAgICAgIHRoaXMuZGF0YS50ZXh0dXJlISwgdGhpcy5kYXRhLnRleHR1cmVTaGFwZVJDISk7XG4gICAgdGhpcy5kYXRhLnRleHR1cmUgPSBudWxsITtcbiAgICB0aGlzLmRhdGEudGV4dHVyZVNoYXBlUkMgPSBudWxsITtcbiAgfVxuXG4gIGluR1BVKCk6IGJvb2xlYW4ge1xuICAgIHJldHVybiB0aGlzLmRhdGEudGV4dHVyZSAhPSBudWxsO1xuICB9XG5cbiAgZXF1YWxzKHQ6IE5EQXJyYXkpOiBib29sZWFuIHtcbiAgICByZXR1cm4gdXRpbC5hcnJheXNFcXVhbCh0aGlzLnNoYXBlLCB0LnNoYXBlKSAmJlxuICAgICAgICB1dGlsLmFycmF5c0VxdWFsKHRoaXMuZ2V0VmFsdWVzKCksIHQuZ2V0VmFsdWVzKCkpO1xuICB9XG5cbiAgc3RhdGljIHJhbmQ8VCBleHRlbmRzIE5EQXJyYXk+KHNoYXBlOiBudW1iZXJbXSwgcmFuZEZ1bmN0aW9uOiAoKSA9PiBudW1iZXIpOlxuICAgICAgVCB7XG4gICAgY29uc3Qgc2l6ZSA9IHV0aWwuc2l6ZUZyb21TaGFwZShzaGFwZSk7XG4gICAgY29uc3QgdmFsdWVzID0gbmV3IEZsb2F0MzJBcnJheShzaXplKTtcbiAgICBmb3IgKGxldCBpID0gMDsgaSA8IHNpemU7IGkrKykge1xuICAgICAgdmFsdWVzW2ldID0gcmFuZEZ1bmN0aW9uKCk7XG4gICAgfVxuXG4gICAgcmV0dXJuIE5EQXJyYXkubWFrZTxUPihzaGFwZSwge3ZhbHVlc30pO1xuICB9XG5cbiAgc3RhdGljIHJhbmROb3JtYWw8VCBleHRlbmRzIE5EQXJyYXk+KHNoYXBlOiBudW1iZXJbXSwgbWVhbiA9IDAsIHN0ZERldiA9IDEpIHtcbiAgICByZXR1cm4gTkRBcnJheS5yYW5kPFQ+KHNoYXBlLCAoKSA9PiB1dGlsLnJhbmRHYXVzcyhtZWFuLCBzdGREZXYpKTtcbiAgfVxuXG4gIHN0YXRpYyByYW5kVHJ1bmNhdGVkTm9ybWFsPFQgZXh0ZW5kcyBOREFycmF5PihcbiAgICAgIHNoYXBlOiBudW1iZXJbXSwgbWVhbiA9IDAsIHN0ZERldiA9IDEpIHtcbiAgICByZXR1cm4gTkRBcnJheS5yYW5kPFQ+KHNoYXBlLCAoKSA9PiB1dGlsLnJhbmRHYXVzcyhtZWFuLCBzdGREZXYsIHRydWUpKTtcbiAgfVxuXG4gIHN0YXRpYyByYW5kVW5pZm9ybTxUIGV4dGVuZHMgTkRBcnJheT4oc2hhcGU6IG51bWJlcltdLCBhOiBudW1iZXIsIGI6IG51bWJlcikge1xuICAgIHJldHVybiBOREFycmF5LnJhbmQ8VD4oc2hhcGUsICgpID0+IHV0aWwucmFuZFVuaWZvcm0oYSwgYikpO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBTY2FsYXIgZXh0ZW5kcyBOREFycmF5IHtcbiAgY29uc3RydWN0b3IoZGF0YTogTkRBcnJheURhdGEpIHtcbiAgICBpZiAoZGF0YS50ZXh0dXJlICE9IG51bGwpIHtcbiAgICAgIGRhdGEudGV4dHVyZVNoYXBlUkMgPSBbMSwgMV07XG4gICAgfVxuICAgIHN1cGVyKFtdLCBkYXRhKTtcbiAgfVxuXG4gIHN0YXRpYyBuZXcodmFsdWU6IG51bWJlcikge1xuICAgIHJldHVybiBuZXcgU2NhbGFyKHt2YWx1ZXM6IG5ldyBGbG9hdDMyQXJyYXkoW3ZhbHVlXSl9KTtcbiAgfVxuXG4gIHN0YXRpYyBaRVJPID0gU2NhbGFyLm5ldygwKTtcbiAgc3RhdGljIE9ORSA9IFNjYWxhci5uZXcoMSk7XG4gIHN0YXRpYyBUV08gPSBTY2FsYXIubmV3KDIpO1xuICBzdGF0aWMgTkVHX09ORSA9IFNjYWxhci5uZXcoLTEpO1xuXG4gIGdldCgpOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLmdldFZhbHVlcygpWzBdO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpWzBdID0gdmFsdWU7XG4gIH1cblxuICBhZGQodmFsdWU6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClbMF0gKz0gdmFsdWU7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIEFycmF5MUQgZXh0ZW5kcyBOREFycmF5IHtcbiAgc2hhcGU6IFtudW1iZXJdO1xuXG4gIGNvbnN0cnVjdG9yKGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgY29uc3Qgc2hhcGUgPSAoZGF0YS52YWx1ZXMgIT0gbnVsbCkgP1xuICAgICAgICBbZGF0YS52YWx1ZXMubGVuZ3RoXSA6XG4gICAgICAgIFt1dGlsLnNpemVGcm9tU2hhcGUoZGF0YS50ZXh0dXJlU2hhcGVSQyEpXTtcbiAgICBzdXBlcihzaGFwZSwgZGF0YSk7XG4gIH1cblxuICBzdGF0aWMgbmV3KHZhbHVlczogRmxvYXQzMkFycmF5fG51bWJlcltdKSB7XG4gICAgaWYgKCEodmFsdWVzIGluc3RhbmNlb2YgRmxvYXQzMkFycmF5KSkge1xuICAgICAgY29uc3QgaW5mZXJyZWRTaGFwZSA9IHV0aWwuaW5mZXJTaGFwZSh2YWx1ZXMpO1xuICAgICAgdXRpbC5hc3NlcnQoXG4gICAgICAgICAgaW5mZXJyZWRTaGFwZS5sZW5ndGggPT09IDEsXG4gICAgICAgICAgYEVycm9yIGNvbnN0cnVjdGluZyBBcnJheTFELiBTaGFwZSBvZiB2YWx1ZXMgJHtpbmZlcnJlZFNoYXBlfSBpcyBgICtcbiAgICAgICAgICAgICAgYG5vdCAxIGRpbWVuc2lvbmFsLmApO1xuICAgIH1cbiAgICByZXR1cm4gbmV3IEFycmF5MUQoe3ZhbHVlczogdG9UeXBlZEFycmF5KHZhbHVlcyl9KTtcbiAgfVxuXG4gIGdldChpOiBudW1iZXIpOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLmdldFZhbHVlcygpW2ldO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIsIGk6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClbaV0gPSB2YWx1ZTtcbiAgfVxuXG4gIGFkZCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpW2ldICs9IHZhbHVlO1xuICB9XG5cbiAgbG9jVG9JbmRleChsb2M6IFtudW1iZXJdKTogbnVtYmVyIHtcbiAgICByZXR1cm4gbG9jWzBdO1xuICB9XG5cbiAgaW5kZXhUb0xvYyhpbmRleDogbnVtYmVyKTogW251bWJlcl0ge1xuICAgIHJldHVybiBbaW5kZXhdO1xuICB9XG5cbiAgc3RhdGljIHplcm9zKHNoYXBlOiBbbnVtYmVyXSk6IEFycmF5MUQge1xuICAgIHJldHVybiBOREFycmF5Lnplcm9zKHNoYXBlKSBhcyBBcnJheTFEO1xuICB9XG59XG5cbmV4cG9ydCBjbGFzcyBBcnJheTJEIGV4dGVuZHMgTkRBcnJheSB7XG4gIHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdO1xuXG4gIHByaXZhdGUgc3RyaWRlMDogbnVtYmVyO1xuXG4gIGNvbnN0cnVjdG9yKHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLCBkYXRhOiBOREFycmF5RGF0YSkge1xuICAgIHV0aWwuYXNzZXJ0KHNoYXBlLmxlbmd0aCA9PT0gMiwgJ1NoYXBlIHNob3VsZCBiZSBvZiBsZW5ndGggMicpO1xuICAgIHN1cGVyKHNoYXBlLCBkYXRhKTtcbiAgICB0aGlzLnN0cmlkZTAgPSB0aGlzLnN0cmlkZXNbMF07XG4gIH1cblxuICBzdGF0aWMgbmV3KFxuICAgICAgc2hhcGU6IFtudW1iZXIsIG51bWJlcl0sIHZhbHVlczogRmxvYXQzMkFycmF5fG51bWJlcltdfG51bWJlcltdW10pIHtcbiAgICBpZiAoISh2YWx1ZXMgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpKSB7XG4gICAgICBjb25zdCBpbmZlcnJlZFNoYXBlID0gdXRpbC5pbmZlclNoYXBlKHZhbHVlcyk7XG4gICAgICBpZiAoaW5mZXJyZWRTaGFwZS5sZW5ndGggPiAxKSB7XG4gICAgICAgIHV0aWwuYXNzZXJ0U2hhcGVzTWF0Y2goXG4gICAgICAgICAgICBzaGFwZSwgaW5mZXJyZWRTaGFwZSxcbiAgICAgICAgICAgIGBFcnJvciB3aGVuIGNvbnN0cnVjdGluZyBBcnJheTJELiBTaGFwZSBvZiB2YWx1ZXMgYCArXG4gICAgICAgICAgICAgICAgYCR7aW5mZXJyZWRTaGFwZX0gZG9lcyBub3QgbWF0Y2ggdGhlIHByb3ZpZGVkIHNoYXBlIGAgK1xuICAgICAgICAgICAgICAgIGAke3NoYXBlfS4gYCk7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBuZXcgQXJyYXkyRChzaGFwZSwge3ZhbHVlczogdG9UeXBlZEFycmF5KHZhbHVlcyl9KTtcbiAgfVxuXG4gIGdldChpOiBudW1iZXIsIGo6IG51bWJlcikge1xuICAgIHJldHVybiB0aGlzLmdldFZhbHVlcygpW3RoaXMuc3RyaWRlMCAqIGkgKyBqXTtcbiAgfVxuXG4gIHNldCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIsIGo6IG51bWJlcikge1xuICAgIHRoaXMuZ2V0VmFsdWVzKClbdGhpcy5zdHJpZGUwICogaSArIGpdID0gdmFsdWU7XG4gIH1cblxuICBhZGQodmFsdWU6IG51bWJlciwgaTogbnVtYmVyLCBqOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpW3RoaXMuc3RyaWRlMCAqIGkgKyBqXSArPSB2YWx1ZTtcbiAgfVxuXG4gIGxvY1RvSW5kZXgobG9jczogW251bWJlciwgbnVtYmVyXSk6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMuc3RyaWRlMCAqIGxvY3NbMF0gKyBsb2NzWzFdO1xuICB9XG5cbiAgaW5kZXhUb0xvYyhpbmRleDogbnVtYmVyKTogW251bWJlciwgbnVtYmVyXSB7XG4gICAgcmV0dXJuIFtNYXRoLmZsb29yKGluZGV4IC8gdGhpcy5zdHJpZGUwKSwgaW5kZXggJSB0aGlzLnN0cmlkZTBdO1xuICB9XG5cbiAgc3RhdGljIHplcm9zKHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogQXJyYXkyRCB7XG4gICAgcmV0dXJuIE5EQXJyYXkuemVyb3Moc2hhcGUpIGFzIEFycmF5MkQ7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIEFycmF5M0QgZXh0ZW5kcyBOREFycmF5IHtcbiAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXTtcbiAgcHJpdmF0ZSBzdHJpZGUwOiBudW1iZXI7XG4gIHByaXZhdGUgc3RyaWRlMTogbnVtYmVyO1xuXG4gIGNvbnN0cnVjdG9yKHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGRhdGE6IE5EQXJyYXlEYXRhKSB7XG4gICAgdXRpbC5hc3NlcnQoc2hhcGUubGVuZ3RoID09PSAzLCAnU2hhcGUgc2hvdWxkIGJlIG9mIGxlbmd0aCAzJyk7XG4gICAgc3VwZXIoc2hhcGUsIGRhdGEpO1xuICAgIHRoaXMuc3RyaWRlMCA9IHRoaXMuc3RyaWRlc1swXTtcbiAgICB0aGlzLnN0cmlkZTEgPSB0aGlzLnN0cmlkZXNbMV07XG4gIH1cblxuICBzdGF0aWMgbmV3KFxuICAgICAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICAgIHZhbHVlczogRmxvYXQzMkFycmF5fG51bWJlcltdfG51bWJlcltdW11bXSkge1xuICAgIGlmICghKHZhbHVlcyBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkpIHtcbiAgICAgIGNvbnN0IGluZmVycmVkU2hhcGUgPSB1dGlsLmluZmVyU2hhcGUodmFsdWVzKTtcbiAgICAgIGlmIChpbmZlcnJlZFNoYXBlLmxlbmd0aCA+IDEpIHtcbiAgICAgICAgdXRpbC5hc3NlcnRTaGFwZXNNYXRjaChcbiAgICAgICAgICAgIHNoYXBlLCBpbmZlcnJlZFNoYXBlLFxuICAgICAgICAgICAgYEVycm9yIHdoZW4gY29uc3RydWN0aW5nIEFycmF5M0QuIFNoYXBlIG9mIHZhbHVlcyBgICtcbiAgICAgICAgICAgICAgICBgJHtpbmZlcnJlZFNoYXBlfSBkb2VzIG5vdCBtYXRjaCB0aGUgcHJvdmlkZWQgc2hhcGUgYCArXG4gICAgICAgICAgICAgICAgYCR7c2hhcGV9LiBgKTtcbiAgICAgIH1cbiAgICB9XG4gICAgcmV0dXJuIG5ldyBBcnJheTNEKHNoYXBlLCB7dmFsdWVzOiB0b1R5cGVkQXJyYXkodmFsdWVzKX0pO1xuICB9XG5cbiAgZ2V0KGk6IG51bWJlciwgajogbnVtYmVyLCBrOiBudW1iZXIpIHtcbiAgICByZXR1cm4gdGhpcy5nZXRWYWx1ZXMoKVt0aGlzLnN0cmlkZTAgKiBpICsgdGhpcy5zdHJpZGUxICogaiArIGtdO1xuICB9XG5cbiAgc2V0KHZhbHVlOiBudW1iZXIsIGk6IG51bWJlciwgajogbnVtYmVyLCBrOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpW3RoaXMuc3RyaWRlMCAqIGkgKyB0aGlzLnN0cmlkZTEgKiBqICsga10gPSB2YWx1ZTtcbiAgfVxuXG4gIGFkZCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIsIGo6IG51bWJlciwgazogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVt0aGlzLnN0cmlkZTAgKiBpICsgdGhpcy5zdHJpZGUxICogaiArIGtdICs9IHZhbHVlO1xuICB9XG5cbiAgbG9jVG9JbmRleChsb2NzOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLnN0cmlkZTAgKiBsb2NzWzBdICsgdGhpcy5zdHJpZGUxICogbG9jc1sxXSArIGxvY3NbMl07XG4gIH1cblxuICBpbmRleFRvTG9jKGluZGV4OiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0ge1xuICAgIGNvbnN0IGkgPSBNYXRoLmZsb29yKGluZGV4IC8gdGhpcy5zdHJpZGUwKTtcbiAgICBpbmRleCAtPSBpICogdGhpcy5zdHJpZGUwO1xuICAgIHJldHVybiBbaSwgTWF0aC5mbG9vcihpbmRleCAvIHRoaXMuc3RyaWRlMSksIGluZGV4ICUgdGhpcy5zdHJpZGUxXTtcbiAgfVxuXG4gIHN0YXRpYyB6ZXJvcyhzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdKTogQXJyYXkzRCB7XG4gICAgcmV0dXJuIE5EQXJyYXkuemVyb3Moc2hhcGUpIGFzIEFycmF5M0Q7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIEFycmF5NEQgZXh0ZW5kcyBOREFycmF5IHtcbiAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdO1xuICBwcml2YXRlIHN0cmlkZTA6IG51bWJlcjtcbiAgcHJpdmF0ZSBzdHJpZGUxOiBudW1iZXI7XG4gIHByaXZhdGUgc3RyaWRlMjogbnVtYmVyO1xuXG4gIGNvbnN0cnVjdG9yKHNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZGF0YTogTkRBcnJheURhdGEpIHtcbiAgICB1dGlsLmFzc2VydChzaGFwZS5sZW5ndGggPT09IDQsICdTaGFwZSBzaG91bGQgYmUgb2YgbGVuZ3RoIDQnKTtcbiAgICBzdXBlcihzaGFwZSwgZGF0YSk7XG4gICAgdGhpcy5zdHJpZGUwID0gdGhpcy5zdHJpZGVzWzBdO1xuICAgIHRoaXMuc3RyaWRlMSA9IHRoaXMuc3RyaWRlc1sxXTtcbiAgICB0aGlzLnN0cmlkZTIgPSB0aGlzLnN0cmlkZXNbMl07XG4gIH1cblxuICBzdGF0aWMgbmV3KFxuICAgICAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgICAgdmFsdWVzOiBGbG9hdDMyQXJyYXl8bnVtYmVyW118bnVtYmVyW11bXVtdW10pIHtcbiAgICBpZiAoISh2YWx1ZXMgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkpKSB7XG4gICAgICBjb25zdCBpbmZlcnJlZFNoYXBlID0gdXRpbC5pbmZlclNoYXBlKHZhbHVlcyk7XG4gICAgICBpZiAoaW5mZXJyZWRTaGFwZS5sZW5ndGggPiAxKSB7XG4gICAgICAgIHV0aWwuYXNzZXJ0U2hhcGVzTWF0Y2goXG4gICAgICAgICAgICBzaGFwZSwgaW5mZXJyZWRTaGFwZSxcbiAgICAgICAgICAgIGBFcnJvciB3aGVuIGNvbnN0cnVjdGluZyBBcnJheTRELiBTaGFwZSBvZiB2YWx1ZXMgYCArXG4gICAgICAgICAgICAgICAgYCR7aW5mZXJyZWRTaGFwZX0gZG9lcyBub3QgbWF0Y2ggdGhlIHByb3ZpZGVkIHNoYXBlIGAgK1xuICAgICAgICAgICAgICAgIGAke3NoYXBlfS4gYCk7XG4gICAgICB9XG4gICAgfVxuICAgIHJldHVybiBuZXcgQXJyYXk0RChzaGFwZSwge3ZhbHVlczogdG9UeXBlZEFycmF5KHZhbHVlcyl9KTtcbiAgfVxuXG4gIGdldChpOiBudW1iZXIsIGo6IG51bWJlciwgazogbnVtYmVyLCBsOiBudW1iZXIpIHtcbiAgICByZXR1cm4gdGhpcy5nZXRWYWx1ZXMoKVxuICAgICAgICBbdGhpcy5zdHJpZGUwICogaSArIHRoaXMuc3RyaWRlMSAqIGogKyB0aGlzLnN0cmlkZTIgKiBrICsgbF07XG4gIH1cblxuICBzZXQodmFsdWU6IG51bWJlciwgaTogbnVtYmVyLCBqOiBudW1iZXIsIGs6IG51bWJlciwgbDogbnVtYmVyKSB7XG4gICAgdGhpcy5nZXRWYWx1ZXMoKVxuICAgICAgICBbdGhpcy5zdHJpZGUwICogaSArIHRoaXMuc3RyaWRlMSAqIGogKyB0aGlzLnN0cmlkZTIgKiBrICsgbF0gPSB2YWx1ZTtcbiAgfVxuXG4gIGFkZCh2YWx1ZTogbnVtYmVyLCBpOiBudW1iZXIsIGo6IG51bWJlciwgazogbnVtYmVyLCBsOiBudW1iZXIpIHtcbiAgICB0aGlzLmdldFZhbHVlcygpXG4gICAgICAgIFt0aGlzLnN0cmlkZTAgKiBpICsgdGhpcy5zdHJpZGUxICogaiArIHRoaXMuc3RyaWRlMiAqIGsgKyBsXSArPSB2YWx1ZTtcbiAgfVxuXG4gIGxvY1RvSW5kZXgobG9jczogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pOiBudW1iZXIge1xuICAgIHJldHVybiB0aGlzLnN0cmlkZTAgKiBsb2NzWzBdICsgdGhpcy5zdHJpZGUxICogbG9jc1sxXSArXG4gICAgICAgIHRoaXMuc3RyaWRlMiAqIGxvY3NbMl0gKyBsb2NzWzNdO1xuICB9XG5cbiAgaW5kZXhUb0xvYyhpbmRleDogbnVtYmVyKTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0ge1xuICAgIGNvbnN0IGkgPSBNYXRoLmZsb29yKGluZGV4IC8gdGhpcy5zdHJpZGUwKTtcbiAgICBpbmRleCAtPSBpICogdGhpcy5zdHJpZGUwO1xuICAgIGNvbnN0IGogPSBNYXRoLmZsb29yKGluZGV4IC8gdGhpcy5zdHJpZGUxKTtcbiAgICBpbmRleCAtPSBqICogdGhpcy5zdHJpZGUxO1xuICAgIHJldHVybiBbaSwgaiwgTWF0aC5mbG9vcihpbmRleCAvIHRoaXMuc3RyaWRlMiksIGluZGV4ICUgdGhpcy5zdHJpZGUyXTtcbiAgfVxuXG4gIHN0YXRpYyB6ZXJvcyhzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0pOiBBcnJheTREIHtcbiAgICByZXR1cm4gTkRBcnJheS56ZXJvcyhzaGFwZSkgYXMgQXJyYXk0RDtcbiAgfVxufVxuXG50eXBlIEFycmF5RGF0YSA9IEZsb2F0MzJBcnJheXxudW1iZXJbXXxudW1iZXJbXVtdfG51bWJlcltdW11bXXxudW1iZXJbXVtdW11bXTtcblxuZnVuY3Rpb24gdG9UeXBlZEFycmF5KGE6IEFycmF5RGF0YSk6IEZsb2F0MzJBcnJheSB7XG4gIHJldHVybiAoYSBpbnN0YW5jZW9mIEZsb2F0MzJBcnJheSkgPyBhIDogbmV3IEZsb2F0MzJBcnJheSh1dGlsLmZsYXR0ZW4oYSkpO1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyBjb252X3V0aWwgZnJvbSAnLi4vY29udl91dGlsJztcbmltcG9ydCB7R1BHUFVQcm9ncmFtfSBmcm9tICcuL2dwZ3B1X21hdGgnO1xuXG5leHBvcnQgY2xhc3MgQ29udjJERGVyV2VpZ2h0c1Byb2dyYW0gaW1wbGVtZW50cyBHUEdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWyd4JywgJ2R5J107XG4gIHBhcmFtczogQXJyYXk8e30+O1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHVzZXJDb2RlOiBzdHJpbmc7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICB4U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZlNpemU6IG51bWJlciwgb3V0cHV0RGVwdGg6IG51bWJlcixcbiAgICAgIHN0cmlkZTogbnVtYmVyLCB6ZXJvUGFkOiBudW1iZXIpIHtcbiAgICBjb25zdCB5U2hhcGUgPSBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgIHhTaGFwZSwgZlNpemUsIG91dHB1dERlcHRoLCBzdHJpZGUsIHplcm9QYWQpO1xuICAgIGNvbnN0IHlOdW1Sb3dzID0geVNoYXBlWzBdO1xuICAgIGNvbnN0IHlOdW1Db2xzID0geVNoYXBlWzFdO1xuICAgIGNvbnN0IHhOdW1Sb3dzID0geFNoYXBlWzBdO1xuICAgIGNvbnN0IHhOdW1Db2xzID0geFNoYXBlWzFdO1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPVxuICAgICAgICBjb252X3V0aWwuY29tcHV0ZVdlaWdodHNTaGFwZTREKHhTaGFwZVsyXSwgb3V0cHV0RGVwdGgsIGZTaXplKTtcbiAgICB0aGlzLnBhcmFtcyA9IFtzdHJpZGUsIHplcm9QYWRdO1xuICAgIHRoaXMudXNlckNvZGUgPSBgXG4gICAgICB2b2lkIG1haW4oKSB7XG4gICAgICAgIHZlYzQgY29vcmRzID0gZ2V0T3V0cHV0Q29vcmRzKCk7XG4gICAgICAgIGZsb2F0IHdSID0gY29vcmRzLng7XG4gICAgICAgIGZsb2F0IHdDID0gY29vcmRzLnk7XG4gICAgICAgIGZsb2F0IGQxID0gY29vcmRzLno7XG4gICAgICAgIGZsb2F0IGQyID0gY29vcmRzLnc7XG5cbiAgICAgICAgLy8gQ29udm9sdmUgeCg/LCA/LCBkMSkgd2l0aCBkeSg6LCA6LCBkMikgdG8gZ2V0IGR3KHdSLCB3QywgZDEsIGQyKS5cbiAgICAgICAgLy8gPyA9IHRvIGJlIGRldGVybWluZWQuIDogPSBhY3Jvc3MgYWxsIHZhbHVlcyBpbiB0aGF0IGF4aXMuXG4gICAgICAgIGZsb2F0IGRvdFByb2QgPSAwLjA7XG4gICAgICAgIGZvciAoaW50IGl5UiA9IDA7IGl5UiA8ICR7eU51bVJvd3N9OyBpeVIrKykge1xuICAgICAgICAgIGZsb2F0IHlSID0gZmxvYXQoaXlSKTtcbiAgICAgICAgICBmbG9hdCB4UiA9IHdSICsgeVIgKiAke3N0cmlkZX0uMCAtICR7emVyb1BhZH0uMDtcblxuICAgICAgICAgIGlmICh4UiA8IDAuMCB8fCB4UiA+PSAke3hOdW1Sb3dzfS4wKSB7XG4gICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBmb3IgKGludCBpeUMgPSAwOyBpeUMgPCAke3lOdW1Db2xzfTsgaXlDKyspIHtcbiAgICAgICAgICAgIGZsb2F0IHlDID0gZmxvYXQoaXlDKTtcbiAgICAgICAgICAgIGZsb2F0IHhDID0gd0MgKyB5QyAqICR7c3RyaWRlfS4wIC0gJHt6ZXJvUGFkfS4wO1xuXG4gICAgICAgICAgICBpZiAoeEMgPCAwLjAgfHwgeEMgPj0gJHt4TnVtQ29sc30uMCkge1xuICAgICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgZmxvYXQgZHlWYWx1ZSA9IGdldER5KHlSLCB5QywgZDIpO1xuICAgICAgICAgICAgZmxvYXQgeFZhbHVlID0gZ2V0WCh4UiwgeEMsIGQxKTtcbiAgICAgICAgICAgIGRvdFByb2QgKz0gKHhWYWx1ZSAqIGR5VmFsdWUpO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBzZXRPdXRwdXQoZG90UHJvZCk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxufVxuXG5leHBvcnQgY2xhc3MgQ29udjJEVHJhbnNwb3NlUHJvZ3JhbSBpbXBsZW1lbnRzIEdQR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ3gnLCAnVycsICdiaWFzJ107XG4gIHBhcmFtczogQXJyYXk8e30+O1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHVzZXJDb2RlOiBzdHJpbmc7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICB4U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZlNpemU6IG51bWJlciwgb3JpZ0lucHV0RGVwdGg6IG51bWJlcixcbiAgICAgIG9yaWdTdHJpZGU6IG51bWJlciwgb3JpZ1BhZDogbnVtYmVyLCBoYXNCaWFzOiBib29sZWFuKSB7XG4gICAgY29uc3QgW3hSb3dzLCB4Q29scywgb3JpZ091dHB1dERlcHRoXSA9IHhTaGFwZTtcbiAgICBjb25zdCBiaWFzU25pcHBldCA9IGhhc0JpYXMgPyAnZG90UHJvZCArPSBnZXRCaWFzKGQyKTsnIDogJyc7XG5cbiAgICAvLyBGaWd1cmUgb3V0IHRoZSBvdXRwdXQgc2hhcGUgYnkgZGlsYXRpbmcgdGhlIGlucHV0LlxuICAgIGNvbnN0IHhSb3dzRGlsYXRlZCA9ICh4Um93cyAtIDEpICogb3JpZ1N0cmlkZSArIDE7XG4gICAgY29uc3QgeENvbHNEaWxhdGVkID0gKHhDb2xzIC0gMSkgKiBvcmlnU3RyaWRlICsgMTtcbiAgICBjb25zdCBwYWQgPSBmU2l6ZSAtIDEgLSBvcmlnUGFkO1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBjb252X3V0aWwuY29tcHV0ZU91dHB1dFNoYXBlM0QoXG4gICAgICAgIFt4Um93c0RpbGF0ZWQsIHhDb2xzRGlsYXRlZCwgb3JpZ091dHB1dERlcHRoXSwgZlNpemUsIG9yaWdJbnB1dERlcHRoLCAxLFxuICAgICAgICBwYWQpO1xuICAgIHRoaXMucGFyYW1zID0gW3BhZCwgZlNpemUsIG9yaWdTdHJpZGUsIGhhc0JpYXNdO1xuXG4gICAgdGhpcy51c2VyQ29kZSA9IGBcbiAgICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgdmVjMyBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgZmxvYXQgeVIgPSBjb29yZHMueDtcbiAgICAgICAgZmxvYXQgeUMgPSBjb29yZHMueTtcbiAgICAgICAgZmxvYXQgZDIgPSBjb29yZHMuejtcblxuICAgICAgICB2ZWMyIHhSQ0Nvcm5lciA9IHZlYzIoeVIsIHlDKSAtIHZlYzIoJHtwYWR9LjAsICR7cGFkfS4wKTtcbiAgICAgICAgZmxvYXQgeFJDb3JuZXIgPSB4UkNDb3JuZXIueDtcbiAgICAgICAgZmxvYXQgeENDb3JuZXIgPSB4UkNDb3JuZXIueTtcblxuICAgICAgICAvLyBDb252b2x2ZSB4KD8sID8sIGQxKSB3aXRoIHcoOiwgOiwgZDIsIGQxKSB0byBnZXQgeSh5UiwgeUMsIGQyKS5cbiAgICAgICAgLy8gPyA9IHRvIGJlIGRldGVybWluZWQuIDogPSBhY3Jvc3MgYWxsIHZhbHVlcyBpbiB0aGF0IGF4aXMuXG4gICAgICAgIGZsb2F0IGRvdFByb2QgPSAwLjA7XG4gICAgICAgIGZvciAoaW50IGl3UiA9IDA7IGl3UiA8ICR7ZlNpemV9OyBpd1IrKykge1xuICAgICAgICAgIGZsb2F0IHdSID0gZmxvYXQoaXdSKTtcbiAgICAgICAgICBmbG9hdCB4UiA9ICh4UkNvcm5lciArIHdSKSAvICR7b3JpZ1N0cmlkZX0uMDtcblxuICAgICAgICAgIGlmICh4UiA8IDAuMCB8fCB4UiA+PSAke3hSb3dzfS4wIHx8IGZyYWN0KHhSKSA+IDAuMCkge1xuICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgZmxvYXQgd1JQZXJtID0gJHtmU2l6ZX0uMCAtIDEuMCAtIHdSO1xuXG4gICAgICAgICAgZm9yIChpbnQgaXdDID0gMDsgaXdDIDwgJHtmU2l6ZX07IGl3QysrKSB7XG4gICAgICAgICAgICBmbG9hdCB3QyA9IGZsb2F0KGl3Qyk7XG4gICAgICAgICAgICBmbG9hdCB4QyA9ICh4Q0Nvcm5lciArIHdDKSAvICR7b3JpZ1N0cmlkZX0uMDtcblxuICAgICAgICAgICAgaWYgKHhDIDwgMC4wIHx8IHhDID49ICR7eENvbHN9LjAgfHwgZnJhY3QoeEMpID4gMC4wKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBmbG9hdCB3Q1Blcm0gPSAke2ZTaXplfS4wIC0gMS4wIC0gd0M7XG5cbiAgICAgICAgICAgIGZvciAoaW50IGlkMSA9IDA7IGlkMSA8ICR7b3JpZ091dHB1dERlcHRofTsgaWQxKyspIHtcbiAgICAgICAgICAgICAgZmxvYXQgZDEgPSBmbG9hdChpZDEpO1xuICAgICAgICAgICAgICBmbG9hdCB4VmFsdWUgPSBnZXRYKHhSLCB4QywgZDEpO1xuICAgICAgICAgICAgICBmbG9hdCB3VmFsdWUgPSBnZXRXKHdSUGVybSwgd0NQZXJtLCBkMiwgZDEpO1xuICAgICAgICAgICAgICBkb3RQcm9kICs9IHhWYWx1ZSAqIHdWYWx1ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgJHtiaWFzU25pcHBldH1cbiAgICAgICAgc2V0T3V0cHV0KGRvdFByb2QpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbn1cblxuZXhwb3J0IGNsYXNzIENvbnYyRERlckJpYXNQcm9ncmFtIGltcGxlbWVudHMgR1BHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lcyA9IFsnZHknXTtcbiAgcGFyYW1zOiBBcnJheTx7fT4gPSBbXTtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICB1c2VyQ29kZTogc3RyaW5nO1xuXG4gIGNvbnN0cnVjdG9yKHlTaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdKSB7XG4gICAgY29uc3QgW3lOdW1Sb3dzLCB5TnVtQ29scywgb3V0cHV0RGVwdGhdID0geVNoYXBlO1xuICAgIHRoaXMub3V0cHV0U2hhcGUgPSBbb3V0cHV0RGVwdGhdO1xuICAgIHRoaXMudXNlckNvZGUgPSBgXG4gICAgICB2b2lkIG1haW4oKSB7XG4gICAgICAgIGZsb2F0IGQyID0gZ2V0T3V0cHV0Q29vcmRzKCk7XG5cbiAgICAgICAgZmxvYXQgZGVyQmlhcyA9IDAuMDtcbiAgICAgICAgZm9yIChpbnQgaXlSID0gMDsgaXlSIDwgJHt5TnVtUm93c307IGl5UisrKSB7XG4gICAgICAgICAgZmxvYXQgeVIgPSBmbG9hdChpeVIpO1xuICAgICAgICAgIGZvciAoaW50IGl5QyA9IDA7IGl5QyA8ICR7eU51bUNvbHN9OyBpeUMrKykge1xuICAgICAgICAgICAgZmxvYXQgeUMgPSBmbG9hdChpeUMpO1xuICAgICAgICAgICAgZGVyQmlhcyArPSBnZXREeSh5UiwgeUMsIGQyKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgc2V0T3V0cHV0KGRlckJpYXMpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uL2NvbnZfdXRpbCc7XG5pbXBvcnQge0dQR1BVUHJvZ3JhbX0gZnJvbSAnLi9ncGdwdV9tYXRoJztcblxuZXhwb3J0IGNsYXNzIENvbnYyRFByb2dyYW0gaW1wbGVtZW50cyBHUEdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWyd4JywgJ1cnLCAnYmlhcyddO1xuICBwYXJhbXM6IEFycmF5PHt9PjtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICB1c2VyQ29kZTogc3RyaW5nO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgeFNoYXBlOiBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIGZpZWxkU2l6ZTogbnVtYmVyLCBvdXRwdXREZXB0aDogbnVtYmVyLFxuICAgICAgc3RyaWRlOiBudW1iZXIsIHBhZDogbnVtYmVyLCBoYXNCaWFzOiBib29sZWFuKSB7XG4gICAgdGhpcy5vdXRwdXRTaGFwZSA9IGNvbnZfdXRpbC5jb21wdXRlT3V0cHV0U2hhcGUzRChcbiAgICAgICAgeFNoYXBlLCBmaWVsZFNpemUsIG91dHB1dERlcHRoLCBzdHJpZGUsIHBhZCk7XG4gICAgY29uc3QgaW5wdXREZXB0aCA9IHhTaGFwZVsyXTtcbiAgICB0aGlzLnBhcmFtcyA9IFtmaWVsZFNpemUsIHN0cmlkZSwgcGFkLCBoYXNCaWFzXTtcbiAgICBjb25zdCBiaWFzU25pcHBldCA9IGhhc0JpYXMgPyAnZG90UHJvZCArPSBnZXRCaWFzKGQyKTsnIDogJyc7XG4gICAgY29uc3QgeE51bVJvd3MgPSB4U2hhcGVbMF07XG4gICAgY29uc3QgeE51bUNvbHMgPSB4U2hhcGVbMV07XG4gICAgdGhpcy51c2VyQ29kZSA9IGBcbiAgICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgdmVjMyBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgZmxvYXQgeVIgPSBjb29yZHMueDtcbiAgICAgICAgZmxvYXQgeUMgPSBjb29yZHMueTtcbiAgICAgICAgZmxvYXQgZDIgPSBjb29yZHMuejtcblxuICAgICAgICB2ZWMyIHhSQ0Nvcm5lciA9IHZlYzIoeVIsIHlDKSAqIHZlYzIoJHtzdHJpZGV9LjAsICR7c3RyaWRlfS4wKSAtXG4gICAgICAgICAgICB2ZWMyKCR7cGFkfS4wLCAke3BhZH0uMCk7XG4gICAgICAgIGZsb2F0IHhSQ29ybmVyID0geFJDQ29ybmVyLng7XG4gICAgICAgIGZsb2F0IHhDQ29ybmVyID0geFJDQ29ybmVyLnk7XG5cbiAgICAgICAgLy8gQ29udm9sdmUgeCg/LCA/LCBkMSkgd2l0aCB3KDosIDosIGQxLCBkMikgdG8gZ2V0IHkoeVIsIHlDLCBkMikuXG4gICAgICAgIC8vID8gPSB0byBiZSBkZXRlcm1pbmVkLiA6ID0gYWNyb3NzIGFsbCB2YWx1ZXMgaW4gdGhhdCBheGlzLlxuICAgICAgICBmbG9hdCBkb3RQcm9kID0gMC4wO1xuICAgICAgICBmb3IgKGludCBpd1IgPSAwOyBpd1IgPCAke2ZpZWxkU2l6ZX07IGl3UisrKSB7XG4gICAgICAgICAgZmxvYXQgd1IgPSBmbG9hdChpd1IpO1xuICAgICAgICAgIGZsb2F0IHhSID0geFJDb3JuZXIgKyB3UjtcblxuICAgICAgICAgIGlmICh4UiA8IDAuMCB8fCB4UiA+PSAke3hOdW1Sb3dzfS4wKSB7XG4gICAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgICB9XG5cbiAgICAgICAgICBmb3IgKGludCBpd0MgPSAwOyBpd0MgPCAke2ZpZWxkU2l6ZX07IGl3QysrKSB7XG4gICAgICAgICAgICBmbG9hdCB3QyA9IGZsb2F0KGl3Qyk7XG4gICAgICAgICAgICBmbG9hdCB4QyA9IHhDQ29ybmVyICsgd0M7XG5cbiAgICAgICAgICAgIGlmICh4QyA8IDAuMCB8fCB4QyA+PSAke3hOdW1Db2xzfS4wKSB7XG4gICAgICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICAgICAgfVxuXG4gICAgICAgICAgICBmb3IgKGludCBpZDEgPSAwOyBpZDEgPCAke2lucHV0RGVwdGh9OyBpZDErKykge1xuICAgICAgICAgICAgICBmbG9hdCBkMSA9IGZsb2F0KGlkMSk7XG4gICAgICAgICAgICAgIGZsb2F0IHhWYWx1ZSA9IGdldFgoeFIsIHhDLCBkMSk7XG4gICAgICAgICAgICAgIGZsb2F0IHdWYWx1ZSA9IGdldFcod1IsIHdDLCBkMSwgZDIpO1xuICAgICAgICAgICAgICBkb3RQcm9kICs9IHhWYWx1ZSAqIHdWYWx1ZTtcbiAgICAgICAgICAgIH1cbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgJHtiaWFzU25pcHBldH1cbiAgICAgICAgc2V0T3V0cHV0KGRvdFByb2QpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgZ3BncHVfdXRpbCBmcm9tICcuL2dwZ3B1X3V0aWwnO1xuaW1wb3J0ICogYXMgdGV4X3V0aWwgZnJvbSAnLi90ZXhfdXRpbCc7XG5pbXBvcnQgKiBhcyB3ZWJnbF91dGlsIGZyb20gJy4vd2ViZ2xfdXRpbCc7XG5cbmltcG9ydCB7V2ViR0xMb3NlQ29udGV4dEV4dGVuc2lvbn0gZnJvbSAnLi93ZWJnbF91dGlsJztcblxuZXhwb3J0IGNsYXNzIEdQR1BVQ29udGV4dCB7XG4gIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQ7XG4gIHRleHR1cmVGbG9hdEV4dGVuc2lvbjoge307XG4gIGNvbG9yQnVmZmVyRmxvYXRFeHRlbnNpb246IHt9O1xuICBsb3NlQ29udGV4dEV4dGVuc2lvbjogV2ViR0xMb3NlQ29udGV4dEV4dGVuc2lvbjtcbiAgdmVydGV4QnVmZmVyOiBXZWJHTEJ1ZmZlcjtcbiAgaW5kZXhCdWZmZXI6IFdlYkdMQnVmZmVyO1xuICBmcmFtZWJ1ZmZlcjogV2ViR0xGcmFtZWJ1ZmZlcjtcbiAgb3V0cHV0VGV4dHVyZTogV2ViR0xUZXh0dXJlfG51bGwgPSBudWxsO1xuICBwcm9ncmFtOiBXZWJHTFByb2dyYW18bnVsbCA9IG51bGw7XG4gIHByaXZhdGUgZGlzcG9zZWQgPSBmYWxzZTtcbiAgcHJpdmF0ZSBhdXRvRGVidWdWYWxpZGF0ZSA9IGZhbHNlO1xuXG4gIGNvbnN0cnVjdG9yKGdsPzogV2ViR0xSZW5kZXJpbmdDb250ZXh0KSB7XG4gICAgaWYgKGdsICE9IG51bGwpIHtcbiAgICAgIHRoaXMuZ2wgPSBnbDtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhpcy5nbCA9IGdwZ3B1X3V0aWwuY3JlYXRlV2ViR0xDb250ZXh0KCk7XG4gICAgfVxuXG4gICAgLy8gV2ViR0wgMi4wIGVuYWJsZXMgdGV4dHVyZSBmbG9hdHMgd2l0aG91dCBhbiBleHRlbnNpb24uXG4gICAgaWYgKCF3ZWJnbF91dGlsLmlzV2ViR0wyRW5hYmxlZCgpKSB7XG4gICAgICB0aGlzLnRleHR1cmVGbG9hdEV4dGVuc2lvbiA9XG4gICAgICAgICAgd2ViZ2xfdXRpbC5nZXRFeHRlbnNpb25PclRocm93KHRoaXMuZ2wsICdPRVNfdGV4dHVyZV9mbG9hdCcpO1xuICAgIH0gZWxzZSB7XG4gICAgICB0aGlzLmNvbG9yQnVmZmVyRmxvYXRFeHRlbnNpb24gPVxuICAgICAgICAgIHdlYmdsX3V0aWwuZ2V0RXh0ZW5zaW9uT3JUaHJvdyh0aGlzLmdsLCAnRVhUX2NvbG9yX2J1ZmZlcl9mbG9hdCcpO1xuICAgIH1cblxuICAgIHRoaXMubG9zZUNvbnRleHRFeHRlbnNpb24gPVxuICAgICAgICB3ZWJnbF91dGlsLmdldEV4dGVuc2lvbk9yVGhyb3codGhpcy5nbCwgJ1dFQkdMX2xvc2VfY29udGV4dCcpIGFzXG4gICAgICAgIFdlYkdMTG9zZUNvbnRleHRFeHRlbnNpb247XG4gICAgdGhpcy52ZXJ0ZXhCdWZmZXIgPSBncGdwdV91dGlsLmNyZWF0ZVZlcnRleEJ1ZmZlcih0aGlzLmdsKTtcbiAgICB0aGlzLmluZGV4QnVmZmVyID0gZ3BncHVfdXRpbC5jcmVhdGVJbmRleEJ1ZmZlcih0aGlzLmdsKTtcbiAgICB0aGlzLmZyYW1lYnVmZmVyID0gd2ViZ2xfdXRpbC5jcmVhdGVGcmFtZWJ1ZmZlcih0aGlzLmdsKTtcbiAgfVxuXG4gIHB1YmxpYyBkaXNwb3NlKCkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgaWYgKHRoaXMucHJvZ3JhbSAhPSBudWxsKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgJ0Rpc3Bvc2luZyBhIEdQR1BVQ29udGV4dCB0aGF0IHN0aWxsIGhhcyBhIGJvdW5kIFdlYkdMUHJvZ3JhbS4nICtcbiAgICAgICAgICAnIFRoaXMgaXMgcHJvYmFibHkgYSByZXNvdXJjZSBsZWFrLCBkZWxldGUgdGhlIHByb2dyYW0gd2l0aCAnICtcbiAgICAgICAgICAnR1BHUFVDb250ZXh0LmRlbGV0ZVByb2dyYW0gYmVmb3JlIGRpc3Bvc2luZy4nKTtcbiAgICB9XG4gICAgaWYgKHRoaXMub3V0cHV0VGV4dHVyZSAhPSBudWxsKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgJ0Rpc3Bvc2luZyBhIEdQR1BVQ29udGV4dCB0aGF0IHN0aWxsIGhhcyBhIGJvdW5kIG91dHB1dCBtYXRyaXggJyArXG4gICAgICAgICAgJ3RleHR1cmUuICBUaGlzIGlzIHByb2JhYmx5IGEgcmVzb3VyY2UgbGVhaywgZGVsZXRlIHRoZSBvdXRwdXQgJyArXG4gICAgICAgICAgJ21hdHJpeCB0ZXh0dXJlIHdpdGggR1BHUFVDb250ZXh0LmRlbGV0ZU1hdHJpeFRleHR1cmUgYmVmb3JlICcgK1xuICAgICAgICAgICdkaXNwb3NpbmcuJyk7XG4gICAgfVxuICAgIGNvbnN0IGdsID0gdGhpcy5nbDtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZmluaXNoKCkpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIG51bGwpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGVsZXRlRnJhbWVidWZmZXIodGhpcy5mcmFtZWJ1ZmZlcikpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kQnVmZmVyKGdsLkFSUkFZX0JVRkZFUiwgbnVsbCkpO1xuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kZWxldGVCdWZmZXIodGhpcy52ZXJ0ZXhCdWZmZXIpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgICAgZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuRUxFTUVOVF9BUlJBWV9CVUZGRVIsIG51bGwpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGVsZXRlQnVmZmVyKHRoaXMuaW5kZXhCdWZmZXIpKTtcbiAgICB0aGlzLmxvc2VDb250ZXh0RXh0ZW5zaW9uLmxvc2VDb250ZXh0KCk7XG4gICAgdGhpcy5kaXNwb3NlZCA9IHRydWU7XG4gIH1cblxuICBwdWJsaWMgZW5hYmxlQXV0b21hdGljRGVidWdWYWxpZGF0aW9uKGVuYWJsZWQ6IGJvb2xlYW4pIHtcbiAgICB0aGlzLmF1dG9EZWJ1Z1ZhbGlkYXRlID0gZW5hYmxlZDtcbiAgICB3ZWJnbF91dGlsLmVuYWJsZURlYnVnV2ViR0xFcnJvckNoZWNraW5nKGVuYWJsZWQpO1xuICB9XG5cbiAgcHVibGljIGNyZWF0ZU1hdHJpeFRleHR1cmUocm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBXZWJHTFRleHR1cmUge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwuY3JlYXRlTWF0cml4VGV4dHVyZSh0aGlzLmdsLCByb3dzLCBjb2x1bW5zKTtcbiAgfVxuXG4gIHB1YmxpYyB1cGxvYWRQaXhlbERhdGFUb1RleHR1cmUoXG4gICAgICB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsXG4gICAgICBwaXhlbHM6IEltYWdlRGF0YXxIVE1MSW1hZ2VFbGVtZW50fEhUTUxDYW52YXNFbGVtZW50fEhUTUxWaWRlb0VsZW1lbnQpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIGdwZ3B1X3V0aWwudXBsb2FkUGl4ZWxEYXRhVG9UZXh0dXJlKHRoaXMuZ2wsIHRleHR1cmUsIHBpeGVscyk7XG4gIH1cblxuICBwdWJsaWMgY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShyb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6XG4gICAgICBXZWJHTFRleHR1cmUge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgcmV0dXJuIGdwZ3B1X3V0aWwuY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZSh0aGlzLmdsLCByb3dzLCBjb2x1bW5zKTtcbiAgfVxuXG4gIHB1YmxpYyBkZWxldGVNYXRyaXhUZXh0dXJlKHRleHR1cmU6IFdlYkdMVGV4dHVyZSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgaWYgKHRoaXMub3V0cHV0VGV4dHVyZSA9PT0gdGV4dHVyZSkge1xuICAgICAgd2ViZ2xfdXRpbC51bmJpbmRDb2xvclRleHR1cmVGcm9tRnJhbWVidWZmZXIodGhpcy5nbCwgdGhpcy5mcmFtZWJ1ZmZlcik7XG4gICAgICB0aGlzLm91dHB1dFRleHR1cmUgPSBudWxsO1xuICAgIH1cbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayh0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLmRlbGV0ZVRleHR1cmUodGV4dHVyZSkpO1xuICB9XG5cbiAgcHVibGljIHVwbG9hZE1hdHJpeFRvVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIsXG4gICAgICBtYXRyaXg6IEZsb2F0MzJBcnJheSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgbnVtQ2hhbm5lbHMgPSAxO1xuICAgIHJldHVybiBncGdwdV91dGlsLnVwbG9hZE1hdHJpeFRvVGV4dHVyZShcbiAgICAgICAgdGhpcy5nbCwgdGV4dHVyZSwgcm93cywgY29sdW1ucywgbWF0cml4LCBudW1DaGFubmVscyk7XG4gIH1cblxuICBwdWJsaWMgdXBsb2FkTWF0cml4VG9QYWNrZWRUZXh0dXJlKFxuICAgICAgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcixcbiAgICAgIG1hdHJpeDogRmxvYXQzMkFycmF5KSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICByZXR1cm4gZ3BncHVfdXRpbC51cGxvYWRNYXRyaXhUb1BhY2tlZFRleHR1cmUoXG4gICAgICAgIHRoaXMuZ2wsIHRleHR1cmUsIHJvd3MsIGNvbHVtbnMsIG1hdHJpeCk7XG4gIH1cblxuICBwdWJsaWMgZG93bmxvYWRNYXRyaXhGcm9tVGV4dHVyZShcbiAgICAgIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBGbG9hdDMyQXJyYXkge1xuICAgIHJldHVybiB0aGlzLmRvd25sb2FkTWF0cml4RHJpdmVyKFxuICAgICAgICB0ZXh0dXJlLFxuICAgICAgICAoKSA9PlxuICAgICAgICAgICAgZ3BncHVfdXRpbC5kb3dubG9hZE1hdHJpeEZyb21PdXRwdXRUZXh0dXJlKHRoaXMuZ2wsIHJvd3MsIGNvbHVtbnMpKTtcbiAgfVxuXG4gIHB1YmxpYyBkb3dubG9hZE1hdHJpeEZyb21QYWNrZWRUZXh0dXJlKFxuICAgICAgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gICAgcmV0dXJuIHRoaXMuZG93bmxvYWRNYXRyaXhEcml2ZXIoXG4gICAgICAgIHRleHR1cmUsXG4gICAgICAgICgpID0+IGdwZ3B1X3V0aWwuZG93bmxvYWRNYXRyaXhGcm9tUGFja2VkT3V0cHV0VGV4dHVyZShcbiAgICAgICAgICAgIHRoaXMuZ2wsIHJvd3MsIGNvbHVtbnMpKTtcbiAgfVxuXG4gIHB1YmxpYyBjcmVhdGVQcm9ncmFtKGZyYWdtZW50U2hhZGVyU291cmNlOiBzdHJpbmcpOiBXZWJHTFByb2dyYW0ge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgZ2wgPSB0aGlzLmdsO1xuICAgIGNvbnN0IGZyYWdtZW50U2hhZGVyOiBXZWJHTFNoYWRlciA9XG4gICAgICAgIHdlYmdsX3V0aWwuY3JlYXRlRnJhZ21lbnRTaGFkZXIoZ2wsIGZyYWdtZW50U2hhZGVyU291cmNlKTtcbiAgICBjb25zdCB2ZXJ0ZXhTaGFkZXI6IFdlYkdMU2hhZGVyID0gZ3BncHVfdXRpbC5jcmVhdGVWZXJ0ZXhTaGFkZXIoZ2wpO1xuICAgIGNvbnN0IHByb2dyYW06IFdlYkdMUHJvZ3JhbSA9IHdlYmdsX3V0aWwuY3JlYXRlUHJvZ3JhbShnbCk7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmF0dGFjaFNoYWRlcihwcm9ncmFtLCB2ZXJ0ZXhTaGFkZXIpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYXR0YWNoU2hhZGVyKHByb2dyYW0sIGZyYWdtZW50U2hhZGVyKSk7XG4gICAgd2ViZ2xfdXRpbC5saW5rUHJvZ3JhbShnbCwgcHJvZ3JhbSk7XG4gICAgaWYgKHRoaXMuYXV0b0RlYnVnVmFsaWRhdGUpIHtcbiAgICAgIHdlYmdsX3V0aWwudmFsaWRhdGVQcm9ncmFtKGdsLCBwcm9ncmFtKTtcbiAgICB9XG5cbiAgICByZXR1cm4gcHJvZ3JhbTtcbiAgfVxuXG4gIHB1YmxpYyBkZWxldGVQcm9ncmFtKHByb2dyYW06IFdlYkdMUHJvZ3JhbSkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgaWYgKHByb2dyYW0gPT09IHRoaXMucHJvZ3JhbSkge1xuICAgICAgdGhpcy5wcm9ncmFtID0gbnVsbDtcbiAgICB9XG4gICAgaWYgKHByb2dyYW0gIT0gbnVsbCkge1xuICAgICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2sodGhpcy5nbCwgKCkgPT4gdGhpcy5nbC5kZWxldGVQcm9ncmFtKHByb2dyYW0pKTtcbiAgICB9XG4gIH1cblxuICBwdWJsaWMgc2V0UHJvZ3JhbShwcm9ncmFtOiBXZWJHTFByb2dyYW18bnVsbCkge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgdGhpcy5wcm9ncmFtID0gcHJvZ3JhbTtcbiAgICBpZiAoKHRoaXMucHJvZ3JhbSAhPSBudWxsKSAmJiB0aGlzLmF1dG9EZWJ1Z1ZhbGlkYXRlKSB7XG4gICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlUHJvZ3JhbSh0aGlzLmdsLCB0aGlzLnByb2dyYW0pO1xuICAgIH1cbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayh0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLnVzZVByb2dyYW0ocHJvZ3JhbSkpO1xuICB9XG5cbiAgcHVibGljIGdldFVuaWZvcm1Mb2NhdGlvbih1bmlmb3JtTmFtZTogc3RyaW5nKTogV2ViR0xVbmlmb3JtTG9jYXRpb24ge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgdGhpcy50aHJvd0lmTm9Qcm9ncmFtKCk7XG4gICAgcmV0dXJuIHdlYmdsX3V0aWwuZ2V0UHJvZ3JhbVVuaWZvcm1Mb2NhdGlvbk9yVGhyb3coXG4gICAgICAgIHRoaXMuZ2wsIHRoaXMucHJvZ3JhbSEsIHVuaWZvcm1OYW1lKTtcbiAgfVxuXG4gIHB1YmxpYyBzZXRJbnB1dE1hdHJpeFRleHR1cmUoXG4gICAgICBpbnB1dE1hdHJpeFRleHR1cmU6IFdlYkdMVGV4dHVyZSwgdW5pZm9ybU5hbWU6IHN0cmluZyxcbiAgICAgIHRleHR1cmVVbml0OiBudW1iZXIpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHRoaXMudGhyb3dJZk5vUHJvZ3JhbSgpO1xuICAgIHdlYmdsX3V0aWwuYmluZFRleHR1cmVUb1Byb2dyYW1Vbmlmb3JtU2FtcGxlcihcbiAgICAgICAgdGhpcy5nbCwgdGhpcy5wcm9ncmFtISwgaW5wdXRNYXRyaXhUZXh0dXJlLCB1bmlmb3JtTmFtZSwgdGV4dHVyZVVuaXQpO1xuICB9XG5cbiAgcHVibGljIHNldE91dHB1dE1hdHJpeFRleHR1cmUoXG4gICAgICBvdXRwdXRNYXRyaXhUZXh0dXJlOiBXZWJHTFRleHR1cmUsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKSB7XG4gICAgdGhpcy5zZXRPdXRwdXRNYXRyaXhUZXh0dXJlRHJpdmVyKG91dHB1dE1hdHJpeFRleHR1cmUsIGNvbHVtbnMsIHJvd3MpO1xuICB9XG5cbiAgcHVibGljIHNldE91dHB1dFBhY2tlZE1hdHJpeFRleHR1cmUoXG4gICAgICBvdXRwdXRQYWNrZWRNYXRyaXhUZXh0dXJlOiBXZWJHTFRleHR1cmUsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICBjb25zdCBbd2lkdGgsIGhlaWdodF0gPVxuICAgICAgICB0ZXhfdXRpbC5nZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcbiAgICB0aGlzLnNldE91dHB1dE1hdHJpeFRleHR1cmVEcml2ZXIob3V0cHV0UGFja2VkTWF0cml4VGV4dHVyZSwgd2lkdGgsIGhlaWdodCk7XG4gIH1cblxuICBwdWJsaWMgc2V0T3V0cHV0TWF0cml4V3JpdGVSZWdpb24oXG4gICAgICBzdGFydFJvdzogbnVtYmVyLCBudW1Sb3dzOiBudW1iZXIsIHN0YXJ0Q29sdW1uOiBudW1iZXIsXG4gICAgICBudW1Db2x1bW5zOiBudW1iZXIpIHtcbiAgICB0aGlzLnNldE91dHB1dE1hdHJpeFdyaXRlUmVnaW9uRHJpdmVyKFxuICAgICAgICBzdGFydENvbHVtbiwgc3RhcnRSb3csIG51bUNvbHVtbnMsIG51bVJvd3MpO1xuICB9XG5cbiAgcHVibGljIHNldE91dHB1dFBhY2tlZE1hdHJpeFdyaXRlUmVnaW9uKFxuICAgICAgc3RhcnRSb3c6IG51bWJlciwgbnVtUm93czogbnVtYmVyLCBzdGFydENvbHVtbjogbnVtYmVyLFxuICAgICAgbnVtQ29sdW1uczogbnVtYmVyKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdzZXRPdXRwdXRQYWNrZWRNYXRyaXhXcml0ZVJlZ2lvbiBub3QgaW1wbGVtZW50ZWQuJyk7XG4gIH1cblxuICBwdWJsaWMgZGVidWdWYWxpZGF0ZSgpIHtcbiAgICBpZiAodGhpcy5wcm9ncmFtICE9IG51bGwpIHtcbiAgICAgIHdlYmdsX3V0aWwudmFsaWRhdGVQcm9ncmFtKHRoaXMuZ2wsIHRoaXMucHJvZ3JhbSk7XG4gICAgfVxuICAgIHdlYmdsX3V0aWwudmFsaWRhdGVGcmFtZWJ1ZmZlcih0aGlzLmdsKTtcbiAgfVxuXG4gIHB1YmxpYyBleGVjdXRlUHJvZ3JhbSgpIHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHRoaXMudGhyb3dJZk5vUHJvZ3JhbSgpO1xuICAgIGNvbnN0IGdsID0gdGhpcy5nbDtcbiAgICBncGdwdV91dGlsLmJpbmRWZXJ0ZXhQcm9ncmFtQXR0cmlidXRlU3RyZWFtcyhcbiAgICAgICAgZ2wsIHRoaXMucHJvZ3JhbSEsIHRoaXMudmVydGV4QnVmZmVyKTtcbiAgICBpZiAodGhpcy5hdXRvRGVidWdWYWxpZGF0ZSkge1xuICAgICAgdGhpcy5kZWJ1Z1ZhbGlkYXRlKCk7XG4gICAgfVxuICAgIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgICBnbCwgKCkgPT4gZ2wuZHJhd0VsZW1lbnRzKGdsLlRSSUFOR0xFUywgNiwgZ2wuVU5TSUdORURfU0hPUlQsIDApKTtcbiAgfVxuXG4gIHB1YmxpYyBibG9ja1VudGlsQWxsUHJvZ3JhbXNDb21wbGV0ZWQoKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayh0aGlzLmdsLCAoKSA9PiB0aGlzLmdsLmZpbmlzaCgpKTtcbiAgfVxuXG4gIHByaXZhdGUgZG93bmxvYWRNYXRyaXhEcml2ZXIoXG4gICAgICB0ZXh0dXJlOiBXZWJHTFRleHR1cmUsXG4gICAgICBkb3dubG9hZEFuZERlY29kZTogKCkgPT4gRmxvYXQzMkFycmF5KTogRmxvYXQzMkFycmF5IHtcbiAgICB0aGlzLnRocm93SWZEaXNwb3NlZCgpO1xuICAgIHdlYmdsX3V0aWwuYmluZENvbG9yVGV4dHVyZVRvRnJhbWVidWZmZXIoXG4gICAgICAgIHRoaXMuZ2wsIHRleHR1cmUsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgIGNvbnN0IHJlc3VsdCA9IGRvd25sb2FkQW5kRGVjb2RlKCk7XG4gICAgaWYgKHRoaXMub3V0cHV0VGV4dHVyZSAhPSBudWxsKSB7XG4gICAgICB3ZWJnbF91dGlsLmJpbmRDb2xvclRleHR1cmVUb0ZyYW1lYnVmZmVyKFxuICAgICAgICAgIHRoaXMuZ2wsIHRoaXMub3V0cHV0VGV4dHVyZSwgdGhpcy5mcmFtZWJ1ZmZlcik7XG4gICAgICBpZiAodGhpcy5hdXRvRGVidWdWYWxpZGF0ZSkge1xuICAgICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlRnJhbWVidWZmZXIodGhpcy5nbCk7XG4gICAgICB9XG4gICAgfSBlbHNlIHtcbiAgICAgIHdlYmdsX3V0aWwudW5iaW5kQ29sb3JUZXh0dXJlRnJvbUZyYW1lYnVmZmVyKHRoaXMuZ2wsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgIH1cbiAgICByZXR1cm4gcmVzdWx0O1xuICB9XG5cbiAgcHJpdmF0ZSBzZXRPdXRwdXRNYXRyaXhUZXh0dXJlRHJpdmVyKFxuICAgICAgb3V0cHV0TWF0cml4VGV4dHVyZU1heWJlUGFja2VkOiBXZWJHTFRleHR1cmUsIHdpZHRoOiBudW1iZXIsXG4gICAgICBoZWlnaHQ6IG51bWJlcikge1xuICAgIHRoaXMudGhyb3dJZkRpc3Bvc2VkKCk7XG4gICAgY29uc3QgZ2wgPSB0aGlzLmdsO1xuICAgIHdlYmdsX3V0aWwuYmluZENvbG9yVGV4dHVyZVRvRnJhbWVidWZmZXIoXG4gICAgICAgIGdsLCBvdXRwdXRNYXRyaXhUZXh0dXJlTWF5YmVQYWNrZWQsIHRoaXMuZnJhbWVidWZmZXIpO1xuICAgIGlmICh0aGlzLmF1dG9EZWJ1Z1ZhbGlkYXRlKSB7XG4gICAgICB3ZWJnbF91dGlsLnZhbGlkYXRlRnJhbWVidWZmZXIoZ2wpO1xuICAgIH1cbiAgICB0aGlzLm91dHB1dFRleHR1cmUgPSBvdXRwdXRNYXRyaXhUZXh0dXJlTWF5YmVQYWNrZWQ7XG4gICAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnZpZXdwb3J0KDAsIDAsIHdpZHRoLCBoZWlnaHQpKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuc2Npc3NvcigwLCAwLCB3aWR0aCwgaGVpZ2h0KSk7XG4gIH1cblxuICBwcml2YXRlIHNldE91dHB1dE1hdHJpeFdyaXRlUmVnaW9uRHJpdmVyKFxuICAgICAgeDogbnVtYmVyLCB5OiBudW1iZXIsIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyKSB7XG4gICAgdGhpcy50aHJvd0lmRGlzcG9zZWQoKTtcbiAgICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgICAgdGhpcy5nbCwgKCkgPT4gdGhpcy5nbC5zY2lzc29yKHgsIHksIHdpZHRoLCBoZWlnaHQpKTtcbiAgfVxuXG4gIHByaXZhdGUgdGhyb3dJZkRpc3Bvc2VkKCkge1xuICAgIGlmICh0aGlzLmRpc3Bvc2VkKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0F0dGVtcHRlZCB0byB1c2UgZGlzcG9zZWQgR1BHUFVDb250ZXh0LicpO1xuICAgIH1cbiAgfVxuXG4gIHByaXZhdGUgdGhyb3dJZk5vUHJvZ3JhbSgpIHtcbiAgICBpZiAodGhpcy5wcm9ncmFtID09IG51bGwpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignTm8gR1BVIHByb2dyYW0gaXMgY3VycmVudGx5IHNldC4nKTtcbiAgICB9XG4gIH1cbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi8uLi91dGlsJztcbmltcG9ydCB7TkRBcnJheX0gZnJvbSAnLi4vbmRhcnJheSc7XG5cbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuL2dwZ3B1X2NvbnRleHQnO1xuaW1wb3J0ICogYXMgc2hhZGVyX2NvbXBpbGVyIGZyb20gJy4vc2hhZGVyX2NvbXBpbGVyJztcbmltcG9ydCB7U2hhcGVJbmZvfSBmcm9tICcuL3NoYWRlcl9jb21waWxlcic7XG5cbmV4cG9ydCBpbnRlcmZhY2UgR1BHUFVQcm9ncmFtIHtcbiAgdmFyaWFibGVOYW1lczogc3RyaW5nW107XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXTtcbiAgcGFyYW1zOiBBcnJheTx7fT47XG4gIHVzZXJDb2RlOiBzdHJpbmc7XG4gIHN1cHBvcnRzQnJvYWRjYXN0aW5nPzogYm9vbGVhbjtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBHUEdQVUJpbmFyeSB7XG4gIHdlYkdMUHJvZ3JhbTogV2ViR0xQcm9ncmFtO1xuICBwcm9ncmFtOiBHUEdQVVByb2dyYW07XG4gIGdwZ3B1OiBHUEdQVUNvbnRleHQ7XG4gIHNvdXJjZTogc3RyaW5nO1xuICBpblNoYXBlSW5mb3M6IFNoYXBlSW5mb1tdO1xuICBvdXRTaGFwZUluZm86IFNoYXBlSW5mbztcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNvbXBpbGVQcm9ncmFtPFQgZXh0ZW5kcyBOREFycmF5LCBLIGV4dGVuZHMgTkRBcnJheT4oXG4gICAgZ3BncHU6IEdQR1BVQ29udGV4dCwgcHJvZ3JhbTogR1BHUFVQcm9ncmFtLCBpbnB1dHM6IFRbXSxcbiAgICBvdXRwdXQ6IEspOiBHUEdQVUJpbmFyeSB7XG4gIGNvbnN0IHVzZXJDb2RlID0gcHJvZ3JhbS51c2VyQ29kZTtcbiAgY29uc3QgaW5wdXRJbmZvcyA9IGlucHV0cy5tYXAoKGlucHV0LCBpKSA9PiB7XG4gICAgY29uc3Qgc2hhcGVJbmZvID0ge1xuICAgICAgbG9naWNhbFNoYXBlOiBpbnB1dC5zaGFwZSxcbiAgICAgIHRleFNoYXBlOiBpbnB1dC5nZXRUZXh0dXJlU2hhcGVSQygpXG4gICAgfTtcbiAgICByZXR1cm4ge25hbWU6IHByb2dyYW0udmFyaWFibGVOYW1lc1tpXSwgc2hhcGVJbmZvfTtcbiAgfSk7XG4gIGNvbnN0IGluU2hhcGVJbmZvcyA9IGlucHV0SW5mb3MubWFwKHggPT4geC5zaGFwZUluZm8pO1xuICBjb25zdCBvdXRTaGFwZUluZm8gPSB7XG4gICAgbG9naWNhbFNoYXBlOiBvdXRwdXQuc2hhcGUsXG4gICAgdGV4U2hhcGU6IG91dHB1dC5nZXRUZXh0dXJlU2hhcGVSQygpXG4gIH07XG4gIGNvbnN0IHNvdXJjZSA9IHNoYWRlcl9jb21waWxlci5tYWtlU2hhZGVyKFxuICAgICAgaW5wdXRJbmZvcywgb3V0U2hhcGVJbmZvLCB1c2VyQ29kZSxcbiAgICAgIHByb2dyYW0uc3VwcG9ydHNCcm9hZGNhc3RpbmcgPT09IHRydWUpO1xuICByZXR1cm4ge1xuICAgIHByb2dyYW0sXG4gICAgc291cmNlLFxuICAgIHdlYkdMUHJvZ3JhbTogZ3BncHUuY3JlYXRlUHJvZ3JhbShzb3VyY2UpLCBncGdwdSwgaW5TaGFwZUluZm9zLCBvdXRTaGFwZUluZm9cbiAgfTtcbn1cblxuZnVuY3Rpb24gdmFsaWRhdGVCaW5hcnlBbmRQcm9ncmFtKHNoYXBlSW5mb3M6IFNoYXBlSW5mb1tdLCBpbnB1dHM6IE5EQXJyYXlbXSkge1xuICBpZiAoc2hhcGVJbmZvcy5sZW5ndGggIT09IGlucHV0cy5sZW5ndGgpIHtcbiAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgYEJpbmFyeSB3YXMgY29tcGlsZWQgd2l0aCAke3NoYXBlSW5mb3MubGVuZ3RofSBpbnB1dHMsIGJ1dCBgICtcbiAgICAgICAgYHdhcyBleGVjdXRlZCB3aXRoICR7aW5wdXRzLmxlbmd0aH0gaW5wdXRzYCk7XG4gIH1cblxuICBzaGFwZUluZm9zLmZvckVhY2goKHMsIGkpID0+IHtcbiAgICBjb25zdCBzaGFwZUEgPSBzLmxvZ2ljYWxTaGFwZTtcbiAgICBjb25zdCB0ZXhTaGFwZUEgPSBzLnRleFNoYXBlO1xuICAgIGNvbnN0IHNoYXBlQiA9IGlucHV0c1tpXS5zaGFwZTtcbiAgICBjb25zdCB0ZXhTaGFwZUIgPSBpbnB1dHNbaV0uZ2V0VGV4dHVyZVNoYXBlUkMoKTtcblxuICAgIGlmICghdXRpbC5hcnJheXNFcXVhbChzaGFwZUEsIHNoYXBlQikpIHtcbiAgICAgIHRocm93IEVycm9yKFxuICAgICAgICAgIGBCaW5hcnkgd2FzIGNvbXBpbGVkIHdpdGggZGlmZmVyZW50IHNoYXBlcyB0aGFuIGAgK1xuICAgICAgICAgIGB0aGUgY3VycmVudCBhcmdzLiBTaGFwZXMgJHtzaGFwZUF9IGFuZCAke3NoYXBlQn0gbXVzdCBtYXRjaGApO1xuICAgIH1cbiAgICBpZiAoIXV0aWwuYXJyYXlzRXF1YWwodGV4U2hhcGVBLCB0ZXhTaGFwZUIpKSB7XG4gICAgICB0aHJvdyBFcnJvcihcbiAgICAgICAgICBgQmluYXJ5IHdhcyBjb21waWxlZCB3aXRoIGRpZmZlcmVudCB0ZXh0dXJlIHNoYXBlcyB0aGFuIHRoZWAgK1xuICAgICAgICAgIGAgY3VycmVudCBhcmdzLiBTaGFwZSAke3RleFNoYXBlQX0gYW5kICR7dGV4U2hhcGVCfSBtdXN0IG1hdGNoYCk7XG4gICAgfVxuICB9KTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHJ1blByb2dyYW08VCBleHRlbmRzIE5EQXJyYXksIEsgZXh0ZW5kcyBOREFycmF5PihcbiAgICBiaW5hcnk6IEdQR1BVQmluYXJ5LCBpbnB1dHM6IFRbXSwgb3V0cHV0OiBLLFxuICAgIGN1c3RvbVNldHVwPzogKGdwZ3B1OiBHUEdQVUNvbnRleHQpID0+IHZvaWQpOiB2b2lkIHtcbiAgdmFsaWRhdGVCaW5hcnlBbmRQcm9ncmFtKGJpbmFyeS5pblNoYXBlSW5mb3MsIGlucHV0cyk7XG4gIHZhbGlkYXRlQmluYXJ5QW5kUHJvZ3JhbShbYmluYXJ5Lm91dFNoYXBlSW5mb10sIFtvdXRwdXRdKTtcblxuICBjb25zdCBvdXRUZXggPSBvdXRwdXQuZ2V0VGV4dHVyZSgpO1xuICBjb25zdCBvdXRUZXhTaGFwZSA9IG91dHB1dC5nZXRUZXh0dXJlU2hhcGVSQygpO1xuICBjb25zdCBncGdwdSA9IGJpbmFyeS5ncGdwdTtcbiAgZ3BncHUuc2V0T3V0cHV0TWF0cml4VGV4dHVyZShvdXRUZXgsIG91dFRleFNoYXBlWzBdLCBvdXRUZXhTaGFwZVsxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0oYmluYXJ5LndlYkdMUHJvZ3JhbSk7XG4gIGlucHV0cy5mb3JFYWNoKChpbnB1dCwgaSkgPT4ge1xuICAgIGNvbnN0IHRleCA9IGlucHV0LmdldFRleHR1cmUoKTtcbiAgICBncGdwdS5zZXRJbnB1dE1hdHJpeFRleHR1cmUodGV4LCBiaW5hcnkucHJvZ3JhbS52YXJpYWJsZU5hbWVzW2ldLCBpKTtcbiAgfSk7XG4gIGlmIChjdXN0b21TZXR1cCAhPSBudWxsKSB7XG4gICAgY3VzdG9tU2V0dXAoZ3BncHUpO1xuICB9XG4gIGdwZ3B1LmV4ZWN1dGVQcm9ncmFtKCk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlU2hhZGVyS2V5KFxuICAgIHByb2dyYW06IEdQR1BVUHJvZ3JhbSwgaW5wdXRzOiBOREFycmF5W10sIG91dHB1dDogTkRBcnJheSk6IHN0cmluZyB7XG4gIGNvbnN0IHBhcmFtcyA9IHByb2dyYW0ucGFyYW1zO1xuICBjb25zdCBrZXlTdGFydCA9XG4gICAgICBpbnB1dHMuY29uY2F0KG91dHB1dCkubWFwKHggPT4geC5zaGFwZSArICdfJyArIHguZ2V0VGV4dHVyZVNoYXBlUkMoKSk7XG4gIGNvbnN0IGtleUVuZCA9IHBhcmFtcy5tYXAocCA9PiBwLnRvU3RyaW5nKCkpO1xuICBsZXQga2V5ID0gW3Byb2dyYW0uY29uc3RydWN0b3IubmFtZV07XG4gIGtleS5wdXNoKChwcm9ncmFtLnN1cHBvcnRzQnJvYWRjYXN0aW5nID09PSB0cnVlKS50b1N0cmluZygpKTtcbiAga2V5ID0ga2V5LmNvbmNhdChrZXlTdGFydCwga2V5RW5kKTtcbiAgcmV0dXJuIGtleS5qb2luKCdfJyk7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCAqIGFzIHRleF91dGlsIGZyb20gJy4vdGV4X3V0aWwnO1xuaW1wb3J0ICogYXMgd2ViZ2xfdXRpbCBmcm9tICcuL3dlYmdsX3V0aWwnO1xuXG5leHBvcnQgZnVuY3Rpb24gZ2V0V2ViR0xDb250ZXh0QXR0cmlidXRlcygpOiBXZWJHTENvbnRleHRBdHRyaWJ1dGVzIHtcbiAgcmV0dXJuIHtcbiAgICBhbHBoYTogZmFsc2UsXG4gICAgYW50aWFsaWFzOiBmYWxzZSxcbiAgICBwcmVtdWx0aXBsaWVkQWxwaGE6IGZhbHNlLFxuICAgIHByZXNlcnZlRHJhd2luZ0J1ZmZlcjogZmFsc2UsXG4gICAgZGVwdGg6IGZhbHNlLFxuICAgIHN0ZW5jaWw6IGZhbHNlLFxuICAgIGZhaWxJZk1ham9yUGVyZm9ybWFuY2VDYXZlYXQ6IHRydWVcbiAgfTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVdlYkdMQ29udGV4dChjYW52YXM/OiBIVE1MQ2FudmFzRWxlbWVudCkge1xuICBjb25zdCBhdHRyaWJ1dGVzID0gZ2V0V2ViR0xDb250ZXh0QXR0cmlidXRlcygpO1xuICBsZXQgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dDtcbiAgaWYgKGNhbnZhcyAhPSBudWxsKSB7XG4gICAgZ2wgPSB3ZWJnbF91dGlsLmNyZWF0ZVdlYkdMUmVuZGVyaW5nQ29udGV4dEZyb21DYW52YXMoY2FudmFzLCBhdHRyaWJ1dGVzKTtcbiAgfSBlbHNlIHtcbiAgICBnbCA9IHdlYmdsX3V0aWwuY3JlYXRlV2ViR0xSZW5kZXJpbmdDb250ZXh0KGF0dHJpYnV0ZXMpO1xuICB9XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLkRFUFRIX1RFU1QpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmRpc2FibGUoZ2wuU1RFTkNJTF9URVNUKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLkJMRU5EKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLkRJVEhFUikpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZGlzYWJsZShnbC5QT0xZR09OX09GRlNFVF9GSUxMKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5kaXNhYmxlKGdsLlNBTVBMRV9DT1ZFUkFHRSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZW5hYmxlKGdsLlNDSVNTT1JfVEVTVCkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuZW5hYmxlKGdsLkNVTExfRkFDRSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuY3VsbEZhY2UoZ2wuQkFDSykpO1xuICByZXR1cm4gZ2w7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVWZXJ0ZXhTaGFkZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMU2hhZGVyIHtcbiAgY29uc3QgdmVydGV4U2hhZGVyU291cmNlID0gYFxuICAgIHByZWNpc2lvbiBoaWdocCBmbG9hdDtcbiAgICBhdHRyaWJ1dGUgdmVjMyBjbGlwU3BhY2VQb3M7XG4gICAgYXR0cmlidXRlIHZlYzIgdXY7XG4gICAgdmFyeWluZyB2ZWMyIHJlc3VsdFVWO1xuXG4gICAgdm9pZCBtYWluKCkge1xuICAgICAgZ2xfUG9zaXRpb24gPSB2ZWM0KGNsaXBTcGFjZVBvcywgMSk7XG4gICAgICByZXN1bHRVViA9IHV2O1xuICAgIH1gO1xuICByZXR1cm4gd2ViZ2xfdXRpbC5jcmVhdGVWZXJ0ZXhTaGFkZXIoZ2wsIHZlcnRleFNoYWRlclNvdXJjZSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVWZXJ0ZXhCdWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMQnVmZmVyIHtcbiAgLy8gW3ggeSB6IHUgdl0gKiBbdXBwZXItbGVmdCwgbG93ZXItbGVmdCwgdXBwZXItcmlnaHQsIGxvd2VyLXJpZ2h0XVxuICBjb25zdCB2ZXJ0ZXhBcnJheSA9IG5ldyBGbG9hdDMyQXJyYXkoXG4gICAgICBbLTEsIDEsIDAsIDAsIDEsIC0xLCAtMSwgMCwgMCwgMCwgMSwgMSwgMCwgMSwgMSwgMSwgLTEsIDAsIDEsIDBdKTtcbiAgcmV0dXJuIHdlYmdsX3V0aWwuY3JlYXRlU3RhdGljVmVydGV4QnVmZmVyKGdsLCB2ZXJ0ZXhBcnJheSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVJbmRleEJ1ZmZlcihnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogV2ViR0xCdWZmZXIge1xuICAvLyBPcGVuR0wgKGFuZCBXZWJHTCkgaGF2ZSBcIkNDVyA9PSBmcm9udFwiIHdpbmRpbmdcbiAgY29uc3QgdHJpYW5nbGVWZXJ0ZXhJbmRpY2VzID0gbmV3IFVpbnQxNkFycmF5KFswLCAxLCAyLCAyLCAxLCAzXSk7XG4gIHJldHVybiB3ZWJnbF91dGlsLmNyZWF0ZVN0YXRpY0luZGV4QnVmZmVyKGdsLCB0cmlhbmdsZVZlcnRleEluZGljZXMpO1xufVxuXG5mdW5jdGlvbiBnZXRUZXh0dXJlSW50ZXJuYWxGb3JtYXQoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgbnVtQ2hhbm5lbHM6IG51bWJlcik6IG51bWJlciB7XG4gIGlmICh3ZWJnbF91dGlsLmlzV2ViR0wyRW5hYmxlZCgpKSB7XG4gICAgaWYgKG51bUNoYW5uZWxzID09PSA0KSB7XG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgICByZXR1cm4gKGdsIGFzIGFueSkuUkdCQTMyRjtcbiAgICB9XG4gICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgIHJldHVybiAoZ2wgYXMgYW55KS5SMzJGO1xuICB9XG4gIHJldHVybiBnbC5SR0JBO1xufVxuXG5mdW5jdGlvbiBnZXRUZXh0dXJlRm9ybWF0KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIG51bUNoYW5uZWxzOiBudW1iZXIpOiBudW1iZXIge1xuICBpZiAod2ViZ2xfdXRpbC5pc1dlYkdMMkVuYWJsZWQoKSAmJiBudW1DaGFubmVscyA9PT0gMSkge1xuICAgIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTpuby1hbnlcbiAgICByZXR1cm4gKGdsIGFzIGFueSkuUkVEO1xuICB9XG4gIHJldHVybiBnbC5SR0JBO1xufVxuXG5mdW5jdGlvbiBjcmVhdGVBbmRDb25maWd1cmVUZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLFxuICAgIG51bUNoYW5uZWxzOiBudW1iZXIpOiBXZWJHTFRleHR1cmUge1xuICB3ZWJnbF91dGlsLnZhbGlkYXRlVGV4dHVyZVNpemUoZ2wsIHdpZHRoLCBoZWlnaHQpO1xuICBjb25zdCB0ZXh0dXJlID0gd2ViZ2xfdXRpbC5jcmVhdGVUZXh0dXJlKGdsKTtcblxuICBjb25zdCB0ZXgyZCA9IGdsLlRFWFRVUkVfMkQ7XG4gIGNvbnN0IGludGVybmFsRm9ybWF0ID0gZ2V0VGV4dHVyZUludGVybmFsRm9ybWF0KGdsLCBudW1DaGFubmVscyk7XG4gIGNvbnN0IGZvcm1hdCA9IGdldFRleHR1cmVGb3JtYXQoZ2wsIG51bUNoYW5uZWxzKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRUZXh0dXJlKHRleDJkLCB0ZXh0dXJlKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnRleFBhcmFtZXRlcmkodGV4MmQsIGdsLlRFWFRVUkVfV1JBUF9TLCBnbC5DTEFNUF9UT19FREdFKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnRleFBhcmFtZXRlcmkodGV4MmQsIGdsLlRFWFRVUkVfV1JBUF9ULCBnbC5DTEFNUF9UT19FREdFKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnRleFBhcmFtZXRlcmkodGV4MmQsIGdsLlRFWFRVUkVfTUlOX0ZJTFRFUiwgZ2wuTkVBUkVTVCkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC50ZXhQYXJhbWV0ZXJpKHRleDJkLCBnbC5URVhUVVJFX01BR19GSUxURVIsIGdsLk5FQVJFU1QpKTtcbiAgd2ViZ2xfdXRpbC5jYWxsQW5kQ2hlY2soXG4gICAgICBnbCxcbiAgICAgICgpID0+IGdsLnRleEltYWdlMkQoXG4gICAgICAgICAgdGV4MmQsIDAsIGludGVybmFsRm9ybWF0LCB3aWR0aCwgaGVpZ2h0LCAwLCBmb3JtYXQsIGdsLkZMT0FULCBudWxsKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKSk7XG4gIHJldHVybiB0ZXh0dXJlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlTWF0cml4VGV4dHVyZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFdlYkdMVGV4dHVyZSB7XG4gIGNvbnN0IFt3aWR0aCwgaGVpZ2h0XSA9XG4gICAgICB0ZXhfdXRpbC5nZXRVbnBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KHJvd3MsIGNvbHVtbnMpO1xuICBjb25zdCBudW1DaGFubmVscyA9IDE7XG4gIHJldHVybiBjcmVhdGVBbmRDb25maWd1cmVUZXh0dXJlKGdsLCB3aWR0aCwgaGVpZ2h0LCBudW1DaGFubmVscyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVDb2xvck1hdHJpeFRleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBXZWJHTFRleHR1cmUge1xuICBjb25zdCBbd2lkdGgsIGhlaWdodF0gPVxuICAgICAgdGV4X3V0aWwuZ2V0Q29sb3JNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcbiAgY29uc3QgbnVtQ2hhbm5lbHMgPSA0O1xuICByZXR1cm4gY3JlYXRlQW5kQ29uZmlndXJlVGV4dHVyZShnbCwgd2lkdGgsIGhlaWdodCwgbnVtQ2hhbm5lbHMpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFdlYkdMVGV4dHVyZSB7XG4gIGNvbnN0IFt3aWR0aCwgaGVpZ2h0XSA9XG4gICAgICB0ZXhfdXRpbC5nZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcbiAgY29uc3QgbnVtQ2hhbm5lbHMgPSA0O1xuICByZXR1cm4gY3JlYXRlQW5kQ29uZmlndXJlVGV4dHVyZShnbCwgd2lkdGgsIGhlaWdodCwgbnVtQ2hhbm5lbHMpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZFZlcnRleFByb2dyYW1BdHRyaWJ1dGVTdHJlYW1zKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSxcbiAgICB2ZXJ0ZXhCdWZmZXI6IFdlYkdMQnVmZmVyKSB7XG4gIGNvbnN0IHBvc09mZnNldCA9IDA7ICAgICAgICAgICAgICAgLy8geCBpcyB0aGUgZmlyc3QgYnVmZmVyIGVsZW1lbnRcbiAgY29uc3QgdXZPZmZzZXQgPSAzICogNDsgICAgICAgICAgICAvLyB1diBjb21lcyBhZnRlciBbeCB5IHpdXG4gIGNvbnN0IHN0cmlkZSA9ICgzICogNCkgKyAoMiAqIDQpOyAgLy8geHl6ICsgdXYsIGVhY2ggZW50cnkgaXMgNC1ieXRlIGZsb2F0LlxuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC5iaW5kQnVmZmVyKGdsLkFSUkFZX0JVRkZFUiwgdmVydGV4QnVmZmVyKSk7XG4gIHdlYmdsX3V0aWwuYmluZFZlcnRleEJ1ZmZlclRvUHJvZ3JhbUF0dHJpYnV0ZShcbiAgICAgIGdsLCBwcm9ncmFtLCAnY2xpcFNwYWNlUG9zJywgdmVydGV4QnVmZmVyLCAzLCBzdHJpZGUsIHBvc09mZnNldCk7XG4gIHRyeSB7XG4gICAgd2ViZ2xfdXRpbC5iaW5kVmVydGV4QnVmZmVyVG9Qcm9ncmFtQXR0cmlidXRlKFxuICAgICAgICBnbCwgcHJvZ3JhbSwgJ3V2JywgdmVydGV4QnVmZmVyLCAyLCBzdHJpZGUsIHV2T2Zmc2V0KTtcbiAgfSBjYXRjaCAoZSkge1xuICAgIC8vIFByb2dyYW1zIHdpdGggMXgxIG91dHB1dCB0ZXh0dXJlcyBkb24ndCB1c2UgdGhlIHV2IGF0dHJpYnV0ZS5cbiAgICAvLyBUaGlzIGNhbiBjYXVzZSB0aGUgc2hhZGVyIGxpbmtlciB0byBkZWFkLXN0cmlwIGl0LCBzbyB3ZSBzaG91bGRuJ3RcbiAgICAvLyBjb21wbGFpbiBvciBmYWlsIGlmIGl0J3Mgbm90IHByZXNlbnQuXG4gICAgaWYgKCFlLmhhc093blByb3BlcnR5KCduYW1lZFZlcnRleEF0dHJpYnV0ZU5vdEZvdW5kJykpIHtcbiAgICAgIHRocm93IGU7XG4gICAgfVxuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1cGxvYWRQaXhlbERhdGFUb1RleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgIHBpeGVsczogSW1hZ2VEYXRhfEhUTUxJbWFnZUVsZW1lbnR8SFRNTENhbnZhc0VsZW1lbnR8SFRNTFZpZGVvRWxlbWVudCkge1xuICBjb25zdCBudW1DaGFubmVscyA9IDQ7XG4gIGNvbnN0IGludGVybmFsRm9ybWF0ID0gZ2V0VGV4dHVyZUludGVybmFsRm9ybWF0KGdsLCBudW1DaGFubmVscyk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCB0ZXh0dXJlKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsXG4gICAgICAoKSA9PiBnbC50ZXhJbWFnZTJEKFxuICAgICAgICAgIGdsLlRFWFRVUkVfMkQsIDAsIGludGVybmFsRm9ybWF0LCBnbC5SR0JBLCBnbC5GTE9BVCwgcGl4ZWxzKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKSk7XG59XG5cbmZ1bmN0aW9uIHVwbG9hZERhdGFUb1RleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCB3aWR0aDogbnVtYmVyLFxuICAgIGhlaWdodDogbnVtYmVyLCBkYXRhOiBGbG9hdDMyQXJyYXksIG51bUNoYW5uZWxzOiBudW1iZXIpIHtcbiAgY29uc3QgdGV4dHVyZUZvcm1hdCA9IGdldFRleHR1cmVGb3JtYXQoZ2wsIG51bUNoYW5uZWxzKTtcblxuICB3ZWJnbF91dGlsLnZhbGlkYXRlVGV4dHVyZVNpemUoZ2wsIHdpZHRoLCBoZWlnaHQpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSkpO1xuICB3ZWJnbF91dGlsLmNhbGxBbmRDaGVjayhcbiAgICAgIGdsLFxuICAgICAgKCkgPT4gZ2wudGV4U3ViSW1hZ2UyRChcbiAgICAgICAgICBnbC5URVhUVVJFXzJELCAwLCAwLCAwLCB3aWR0aCwgaGVpZ2h0LCB0ZXh0dXJlRm9ybWF0LCBnbC5GTE9BVCxcbiAgICAgICAgICBkYXRhKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kVGV4dHVyZShnbC5URVhUVVJFXzJELCBudWxsKSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB1cGxvYWRNYXRyaXhUb1RleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLCByb3dzOiBudW1iZXIsXG4gICAgY29sdW1uczogbnVtYmVyLCBtYXRyaXg6IEZsb2F0MzJBcnJheSwgbnVtQ2hhbm5lbHM6IG51bWJlcikge1xuICBjb25zdCBbdywgaF0gPVxuICAgICAgdGV4X3V0aWwuZ2V0VW5wYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcblxuICBjb25zdCBjaGFubmVsc1BlclRleHR1cmUgPVxuICAgICAgbnVtQ2hhbm5lbHMgPT09IDEgPyB3ZWJnbF91dGlsLmdldENoYW5uZWxzUGVyVGV4dHVyZSgpIDogbnVtQ2hhbm5lbHM7XG4gIGNvbnN0IHVucGFja2VkQXJyYXkgPVxuICAgICAgbmV3IEZsb2F0MzJBcnJheSh0ZXhfdXRpbC5nZXRVbnBhY2tlZEFycmF5U2l6ZUZyb21NYXRyaXhTaXplKFxuICAgICAgICAgIG1hdHJpeC5sZW5ndGgsIGNoYW5uZWxzUGVyVGV4dHVyZSkpO1xuICB0ZXhfdXRpbC5lbmNvZGVNYXRyaXhUb1VucGFja2VkQXJyYXkoXG4gICAgICBtYXRyaXgsIHVucGFja2VkQXJyYXksIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG5cbiAgdXBsb2FkRGF0YVRvVGV4dHVyZShnbCwgdGV4dHVyZSwgdywgaCwgdW5wYWNrZWRBcnJheSwgbnVtQ2hhbm5lbHMpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdXBsb2FkTWF0cml4VG9QYWNrZWRUZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgcm93czogbnVtYmVyLFxuICAgIGNvbHVtbnM6IG51bWJlciwgbWF0cml4OiBGbG9hdDMyQXJyYXkpIHtcbiAgY29uc3QgW3csIGhdID0gdGV4X3V0aWwuZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gIGNvbnN0IHBhY2tlZFJHQkEgPSBuZXcgRmxvYXQzMkFycmF5KFxuICAgICAgdGV4X3V0aWwuZ2V0UGFja2VkUkdCQUFycmF5U2l6ZUZyb21NYXRyaXhTaGFwZShyb3dzLCBjb2x1bW5zKSk7XG4gIHRleF91dGlsLmVuY29kZU1hdHJpeFRvUGFja2VkUkdCQShtYXRyaXgsIHJvd3MsIGNvbHVtbnMsIHBhY2tlZFJHQkEpO1xuICBjb25zdCBudW1DaGFubmVscyA9IDQ7XG4gIHVwbG9hZERhdGFUb1RleHR1cmUoZ2wsIHRleHR1cmUsIHcsIGgsIHBhY2tlZFJHQkEsIG51bUNoYW5uZWxzKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRvd25sb2FkTWF0cml4RnJvbU91dHB1dFRleHR1cmUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBGbG9hdDMyQXJyYXkge1xuICBjb25zdCBbdywgaF0gPVxuICAgICAgdGV4X3V0aWwuZ2V0VW5wYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcblxuICBjb25zdCBjaGFubmVsc1BlclRleHR1cmUgPSA0O1xuICBjb25zdCB1bnBhY2tlZEFycmF5ID1cbiAgICAgIG5ldyBGbG9hdDMyQXJyYXkodGV4X3V0aWwuZ2V0VW5wYWNrZWRBcnJheVNpemVGcm9tTWF0cml4U2l6ZShcbiAgICAgICAgICByb3dzICogY29sdW1ucywgY2hhbm5lbHNQZXJUZXh0dXJlKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnJlYWRQaXhlbHMoMCwgMCwgdywgaCwgZ2wuUkdCQSwgZ2wuRkxPQVQsIHVucGFja2VkQXJyYXkpKTtcblxuICBjb25zdCBtYXRyaXggPSBuZXcgRmxvYXQzMkFycmF5KHJvd3MgKiBjb2x1bW5zKTtcbiAgdGV4X3V0aWwuZGVjb2RlTWF0cml4RnJvbVVucGFja2VkQXJyYXkoXG4gICAgICB1bnBhY2tlZEFycmF5LCBtYXRyaXgsIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG4gIHJldHVybiBtYXRyaXg7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBkb3dubG9hZE1hdHJpeEZyb21QYWNrZWRPdXRwdXRUZXh0dXJlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogRmxvYXQzMkFycmF5IHtcbiAgY29uc3QgW3csIGhdID0gdGV4X3V0aWwuZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gIGNvbnN0IHBhY2tlZFJHQkEgPSBuZXcgRmxvYXQzMkFycmF5KFxuICAgICAgdGV4X3V0aWwuZ2V0UGFja2VkUkdCQUFycmF5U2l6ZUZyb21NYXRyaXhTaGFwZShyb3dzLCBjb2x1bW5zKSk7XG4gIHdlYmdsX3V0aWwuY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsICgpID0+IGdsLnJlYWRQaXhlbHMoMCwgMCwgdywgaCwgZ2wuUkdCQSwgZ2wuRkxPQVQsIHBhY2tlZFJHQkEpKTtcbiAgY29uc3QgbWF0cml4ID0gbmV3IEZsb2F0MzJBcnJheShyb3dzICogY29sdW1ucyk7XG4gIHJldHVybiB0ZXhfdXRpbC5kZWNvZGVNYXRyaXhGcm9tUGFja2VkUkdCQShwYWNrZWRSR0JBLCByb3dzLCBjb2x1bW5zLCBtYXRyaXgpO1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge0dQR1BVUHJvZ3JhbX0gZnJvbSAnLi9ncGdwdV9tYXRoJztcblxuZXhwb3J0IGNsYXNzIExvZ1N1bUV4cFByb2dyYW0gaW1wbGVtZW50cyBHUEdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWydBJ107XG4gIHBhcmFtczogQXJyYXk8e30+ID0gW107XG4gIG91dHB1dFNoYXBlOiBudW1iZXJbXSA9IFtdO1xuICB1c2VyQ29kZTogc3RyaW5nO1xuXG4gIGNvbnN0cnVjdG9yKGFTaXplOiBudW1iZXIpIHtcbiAgICB0aGlzLnVzZXJDb2RlID0gYFxuICAgICAgdm9pZCBtYWluKCkge1xuICAgICAgICBmbG9hdCBhTWF4ID0gZ2V0QUZsYXQoMC4wKTtcbiAgICAgICAgZm9yIChpbnQgaSA9IDA7IGkgPCAke2FTaXplfTsgaSsrKSB7XG4gICAgICAgICAgYU1heCA9IG1heChhTWF4LCBnZXRBRmxhdChmbG9hdChpKSkpO1xuICAgICAgICB9XG5cbiAgICAgICAgZmxvYXQgZXhwU3VtID0gMC4wO1xuICAgICAgICBmb3IgKGludCBpID0gMDsgaSA8ICR7YVNpemV9OyBpKyspIHtcbiAgICAgICAgICBleHBTdW0gKz0gZXhwKGdldEFGbGF0KGZsb2F0KGkpKSAtIGFNYXgpO1xuICAgICAgICB9XG5cbiAgICAgICAgc2V0T3V0cHV0KGFNYXggKyBsb2coZXhwU3VtKSk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge01hdHJpeE9yaWVudGF0aW9ufSBmcm9tICcuLi9tYXRoJztcbmltcG9ydCB7R1BHUFVQcm9ncmFtfSBmcm9tICcuL2dwZ3B1X21hdGgnO1xuXG5leHBvcnQgY2xhc3MgTWF0TXVsUHJvZ3JhbSBpbXBsZW1lbnRzIEdQR1BVUHJvZ3JhbSB7XG4gIHZhcmlhYmxlTmFtZXMgPSBbJ21hdHJpeEEnLCAnbWF0cml4QiddO1xuICBwYXJhbXM6IEFycmF5PHt9PjtcbiAgb3V0cHV0U2hhcGU6IG51bWJlcltdO1xuICB1c2VyQ29kZTogc3RyaW5nO1xuXG4gIGNvbnN0cnVjdG9yKFxuICAgICAgYVNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLCBiU2hhcGU6IFtudW1iZXIsIG51bWJlcl0sXG4gICAgICBhT3JpZW50ID0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUixcbiAgICAgIGJPcmllbnQgPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSB7XG4gICAgdGhpcy5wYXJhbXMgPSBbYU9yaWVudCwgYk9yaWVudF07XG5cbiAgICBjb25zdCBvdXRlclNoYXBlQSA9XG4gICAgICAgIChhT3JpZW50ID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IGFTaGFwZVswXSA6IGFTaGFwZVsxXTtcbiAgICBjb25zdCBvdXRlclNoYXBlQiA9XG4gICAgICAgIChiT3JpZW50ID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IGJTaGFwZVsxXSA6IGJTaGFwZVswXTtcbiAgICB0aGlzLm91dHB1dFNoYXBlID0gW291dGVyU2hhcGVBLCBvdXRlclNoYXBlQl07XG5cbiAgICBjb25zdCBzaGFyZWREaW0gPVxuICAgICAgICAoYU9yaWVudCA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUiA/IGFTaGFwZVsxXSA6IGFTaGFwZVswXSk7XG4gICAgY29uc3QgYVNuaXBwZXQgPVxuICAgICAgICAoYU9yaWVudCA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyAnYVJvdywgaScgOiAnaSwgYVJvdyc7XG4gICAgY29uc3QgYlNuaXBwZXQgPVxuICAgICAgICAoYk9yaWVudCA9PT0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUikgPyAnaSwgYkNvbCcgOiAnYkNvbCwgaSc7XG5cbiAgICB0aGlzLnVzZXJDb2RlID0gYFxuICAgICAgY29uc3QgaW50IHNoYXJlZERpbSA9ICR7c2hhcmVkRGltfTtcblxuICAgICAgZmxvYXQgZG90QVJvd0JDb2woZmxvYXQgYVJvdywgZmxvYXQgYkNvbCkge1xuICAgICAgICBmbG9hdCByZXN1bHQgPSAwLjA7XG4gICAgICAgIGZvciAoaW50IGlpID0gMDsgaWkgPCBzaGFyZWREaW07IGlpKyspIHtcbiAgICAgICAgICBmbG9hdCBpID0gZmxvYXQoaWkpO1xuICAgICAgICAgIGZsb2F0IGEgPSBnZXRNYXRyaXhBKCR7YVNuaXBwZXR9KTtcbiAgICAgICAgICBmbG9hdCBiID0gZ2V0TWF0cml4Qigke2JTbmlwcGV0fSk7XG4gICAgICAgICAgcmVzdWx0ICs9IChhICogYik7XG4gICAgICAgIH1cbiAgICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICAgIH1cblxuICAgICAgdm9pZCBtYWluKCkge1xuICAgICAgICB2ZWMyIHJlc1JDID0gZ2V0T3V0cHV0Q29vcmRzKCk7XG4gICAgICAgIHNldE91dHB1dChkb3RBUm93QkNvbChyZXNSQy54LCByZXNSQy55KSk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQge01hdHJpeE9yaWVudGF0aW9ufSBmcm9tICcuLi9tYXRoJztcblxuaW1wb3J0IHtHUEdQVUNvbnRleHR9IGZyb20gJy4vZ3BncHVfY29udGV4dCc7XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRGcmFnbWVudFNoYWRlclNvdXJjZShcbiAgICBzaGFyZWREaW1lbnNpb246IG51bWJlciwgYU9yaWVudGF0aW9uOiBNYXRyaXhPcmllbnRhdGlvbixcbiAgICBiT3JpZW50YXRpb246IE1hdHJpeE9yaWVudGF0aW9uKTogc3RyaW5nIHtcbiAgLypcbiAgICAgIEEgPSBbMCAxICAgQiA9IFswIDEgIG91dCA9IFtBMCpCMCtBMSpCMiBBMCpCMStBMSpCM1xuICAgICAgICAgICAyIDNdICAgICAgIDIgM10gICAgICAgIEEyKkIwK0ExKkIyIEEyKkIxK0F3KkIzXVxuICAgICAgb3V0LjAgPSBBMCAqIEIwICsgQTEgKiBCMlxuICAgICAgb3V0LjEgPSBBMCAqIEIxICsgQTEgKiBCM1xuICAgICAgb3V0LjIgPSBBMiAqIEIwICsgQTMgKiBCMlxuICAgICAgb3V0LjMgPSBBMiAqIEIxICsgQTMgKiBCM1xuXG4gICAgICBBKkIgICAgID0gQS54eHp6ICogQi54eXh5ICsgQS55eXd3ICogQi56d3p3XG4gICAgICBBXnQqQiAgID0gQS54eHl5ICogQi54eXh5ICsgQS56end3ICogQi56d3p3XG4gICAgICBBKkJedCAgID0gQS54eHp6ICogQi54enh6ICsgQS55eXd3ICogQi55d3l3XG4gICAgICBBXnQqQl50ID0gQS54eHl5ICogQi54enh6ICsgQS56end3ICogQi55d3l3XG4gICAqL1xuICBjb25zdCBzaGFyZWREaW1lbnNpb25QYWNrZWQgPSBNYXRoLmNlaWwoc2hhcmVkRGltZW5zaW9uIC8gMik7XG4gIGNvbnN0IGFTYW1wbGUgPSAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICAnY2VudGVyLCByZXN1bHRVVi50JyA6XG4gICAgICAncmVzdWx0VVYudCwgY2VudGVyJztcbiAgY29uc3QgYlNhbXBsZSA9IChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgICdyZXN1bHRVVi5zLCBjZW50ZXInIDpcbiAgICAgICdjZW50ZXIsIHJlc3VsdFVWLnMnO1xuICBjb25zdCBhU3dpenpsZTogW3N0cmluZywgc3RyaW5nXSA9XG4gICAgICAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/IFsnYS54eHp6JywgJ2EueXl3dyddIDpcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgWydhLnh4eXknLCAnYS56end3J107XG4gIGNvbnN0IGJTd2l6emxlOiBbc3RyaW5nLCBzdHJpbmddID1cbiAgICAgIChiT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID8gWydiLnh5eHknLCAnYi56d3p3J10gOlxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBbJ2IueHp4eicsICdiLnl3eXcnXTtcbiAgcmV0dXJuIGBcbiAgICBwcmVjaXNpb24gaGlnaHAgZmxvYXQ7XG4gICAgdW5pZm9ybSBzYW1wbGVyMkQgbWF0cml4QTtcbiAgICB1bmlmb3JtIHNhbXBsZXIyRCBtYXRyaXhCO1xuICAgIHZhcnlpbmcgdmVjMiByZXN1bHRVVjtcblxuICAgIGNvbnN0IGZsb2F0IHNoYXJlZERpbWVuc2lvbiA9ICR7c2hhcmVkRGltZW5zaW9uUGFja2VkfS4wO1xuXG4gICAgdmVjNCBkb3QyeDJBUm93QkNvbCgpIHtcbiAgICAgIHZlYzQgcmVzdWx0ID0gdmVjNCgwLCAwLCAwLCAwKTtcbiAgICAgIGZvciAoaW50IGlpID0gMDsgaWkgPCAke3NoYXJlZERpbWVuc2lvblBhY2tlZH07IGlpKyspIHtcbiAgICAgICAgZmxvYXQgaSA9IGZsb2F0KGlpKTtcbiAgICAgICAgZmxvYXQgY2VudGVyID0gKGkgKyAwLjUpIC8gc2hhcmVkRGltZW5zaW9uO1xuICAgICAgICB2ZWM0IGEgPSB0ZXh0dXJlMkQobWF0cml4QSwgdmVjMigke2FTYW1wbGV9KSk7XG4gICAgICAgIHZlYzQgYiA9IHRleHR1cmUyRChtYXRyaXhCLCB2ZWMyKCR7YlNhbXBsZX0pKTtcbiAgICAgICAgcmVzdWx0ICs9XG4gICAgICAgICAgKCR7YVN3aXp6bGVbMF19ICogJHtiU3dpenpsZVswXX0pICsgKCR7YVN3aXp6bGVbMV19ICogJHtiU3dpenpsZVsxXX0pO1xuICAgICAgfVxuICAgICAgcmV0dXJuIHJlc3VsdDtcbiAgICB9XG5cbiAgICB2b2lkIG1haW4oKSB7XG4gICAgICBnbF9GcmFnQ29sb3IgPSBkb3QyeDJBUm93QkNvbCgpO1xuICAgIH1gO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbXVsdGlwbHlNYXRyaXhQYWNrZWQoXG4gICAgZ3BncHU6IEdQR1BVQ29udGV4dCwgbXVsdGlwbHlQcm9ncmFtOiBXZWJHTFByb2dyYW0sIGE6IFdlYkdMVGV4dHVyZSxcbiAgICBiOiBXZWJHTFRleHR1cmUsIHJlc3VsdDogV2ViR0xUZXh0dXJlLFxuICAgIHJlc3VsdFNoYXBlUm93Q29sOiBbbnVtYmVyLCBudW1iZXJdKSB7XG4gIGdwZ3B1LnNldE91dHB1dFBhY2tlZE1hdHJpeFRleHR1cmUoXG4gICAgICByZXN1bHQsIHJlc3VsdFNoYXBlUm93Q29sWzBdLCByZXN1bHRTaGFwZVJvd0NvbFsxXSk7XG4gIGdwZ3B1LnNldFByb2dyYW0obXVsdGlwbHlQcm9ncmFtKTtcbiAgZ3BncHUuc2V0SW5wdXRNYXRyaXhUZXh0dXJlKGEsICdtYXRyaXhBJywgMCk7XG4gIGdwZ3B1LnNldElucHV0TWF0cml4VGV4dHVyZShiLCAnbWF0cml4QicsIDEpO1xuICBncGdwdS5leGVjdXRlUHJvZ3JhbSgpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdXBsb2FkTXVsdGlwbHlNYXRyaXhQYWNrZWREb3dubG9hZChcbiAgICBhOiBGbG9hdDMyQXJyYXksIGFTaGFwZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSwgYjogRmxvYXQzMkFycmF5LFxuICAgIGJTaGFwZVJvd0NvbDogW251bWJlciwgbnVtYmVyXSwgYU9yaWVudGF0aW9uID0gTWF0cml4T3JpZW50YXRpb24uUkVHVUxBUixcbiAgICBiT3JpZW50YXRpb24gPSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKTogRmxvYXQzMkFycmF5IHtcbiAgY29uc3QgcmVzdWx0TnVtUm93cyA9IChhT3JpZW50YXRpb24gPT09IE1hdHJpeE9yaWVudGF0aW9uLlJFR1VMQVIpID9cbiAgICAgIGFTaGFwZVJvd0NvbFswXSA6XG4gICAgICBhU2hhcGVSb3dDb2xbMV07XG4gIGNvbnN0IHJlc3VsdE51bUNvbHMgPSAoYk9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICBiU2hhcGVSb3dDb2xbMV0gOlxuICAgICAgYlNoYXBlUm93Q29sWzBdO1xuICBjb25zdCBzaGFyZWREaW1lbnNpb24gPSAoYU9yaWVudGF0aW9uID09PSBNYXRyaXhPcmllbnRhdGlvbi5SRUdVTEFSKSA/XG4gICAgICBhU2hhcGVSb3dDb2xbMV0gOlxuICAgICAgYVNoYXBlUm93Q29sWzBdO1xuXG4gIGNvbnN0IGdwZ3B1ID0gbmV3IEdQR1BVQ29udGV4dCgpO1xuICBjb25zdCBwcm9ncmFtOiBXZWJHTFByb2dyYW0gPSBncGdwdS5jcmVhdGVQcm9ncmFtKFxuICAgICAgZ2V0RnJhZ21lbnRTaGFkZXJTb3VyY2Uoc2hhcmVkRGltZW5zaW9uLCBhT3JpZW50YXRpb24sIGJPcmllbnRhdGlvbikpO1xuXG4gIGNvbnN0IGFUZXh0dXJlOiBXZWJHTFRleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShhU2hhcGVSb3dDb2xbMF0sIGFTaGFwZVJvd0NvbFsxXSk7XG4gIGNvbnN0IGJUZXh0dXJlOiBXZWJHTFRleHR1cmUgPVxuICAgICAgZ3BncHUuY3JlYXRlUGFja2VkTWF0cml4VGV4dHVyZShiU2hhcGVSb3dDb2xbMF0sIGJTaGFwZVJvd0NvbFsxXSk7XG4gIGNvbnN0IHJlc3VsdFRleHR1cmU6IFdlYkdMVGV4dHVyZSA9XG4gICAgICBncGdwdS5jcmVhdGVQYWNrZWRNYXRyaXhUZXh0dXJlKHJlc3VsdE51bVJvd3MsIHJlc3VsdE51bUNvbHMpO1xuXG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShcbiAgICAgIGFUZXh0dXJlLCBhU2hhcGVSb3dDb2xbMF0sIGFTaGFwZVJvd0NvbFsxXSwgYSk7XG4gIGdwZ3B1LnVwbG9hZE1hdHJpeFRvUGFja2VkVGV4dHVyZShcbiAgICAgIGJUZXh0dXJlLCBiU2hhcGVSb3dDb2xbMF0sIGJTaGFwZVJvd0NvbFsxXSwgYik7XG5cbiAgbXVsdGlwbHlNYXRyaXhQYWNrZWQoXG4gICAgICBncGdwdSwgcHJvZ3JhbSwgYVRleHR1cmUsIGJUZXh0dXJlLCByZXN1bHRUZXh0dXJlLFxuICAgICAgW3Jlc3VsdE51bVJvd3MsIHJlc3VsdE51bUNvbHNdKTtcblxuICBjb25zdCByZXN1bHQgPSBncGdwdS5kb3dubG9hZE1hdHJpeEZyb21QYWNrZWRUZXh0dXJlKFxuICAgICAgcmVzdWx0VGV4dHVyZSwgcmVzdWx0TnVtUm93cywgcmVzdWx0TnVtQ29scyk7XG5cbiAgZ3BncHUuZGVsZXRlTWF0cml4VGV4dHVyZShhVGV4dHVyZSk7XG4gIGdwZ3B1LmRlbGV0ZU1hdHJpeFRleHR1cmUoYlRleHR1cmUpO1xuICBncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKHJlc3VsdFRleHR1cmUpO1xuICBncGdwdS5kZWxldGVQcm9ncmFtKHByb2dyYW0pO1xuICBncGdwdS5kaXNwb3NlKCk7XG5cbiAgcmV0dXJuIHJlc3VsdDtcbn1cbiIsIi8qIENvcHlyaWdodCAyMDE3IEdvb2dsZSBJbmMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG5cbkxpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG55b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG5Zb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcblxuICAgIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuXG5Vbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG5kaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG5XSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cblNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbmxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09Ki9cblxuaW1wb3J0ICogYXMgY29udl91dGlsIGZyb20gJy4uL2NvbnZfdXRpbCc7XG5pbXBvcnQge0dQR1BVUHJvZ3JhbX0gZnJvbSAnLi9ncGdwdV9tYXRoJztcblxuZXhwb3J0IGNsYXNzIFBvb2wyRFByb2dyYW0gaW1wbGVtZW50cyBHUEdQVVByb2dyYW0ge1xuICB2YXJpYWJsZU5hbWVzID0gWyd4J107XG4gIHBhcmFtczogQXJyYXk8e30+O1xuICBvdXRwdXRTaGFwZTogbnVtYmVyW107XG4gIHVzZXJDb2RlOiBzdHJpbmc7XG5cbiAgY29uc3RydWN0b3IoXG4gICAgICB4U2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgZlNpemU6IG51bWJlciwgc3RyaWRlOiBudW1iZXIsXG4gICAgICBwYWQ6IG51bWJlciwgcG9vbFR5cGU6ICdtYXgnfCdtaW4nfCdhdmcnLCBjb21wdXRlUG9zaXRpb25zOiBib29sZWFuKSB7XG4gICAgaWYgKHBvb2xUeXBlID09PSAnYXZnJyAmJiBjb21wdXRlUG9zaXRpb25zKSB7XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoJ0Nhbm5vdCBjb21wdXRlIHBvc2l0aW9ucyBmb3IgYXZlcmFnZSBwb29sLicpO1xuICAgIH1cblxuICAgIGxldCByZXR1cm5WYWx1ZSA9ICdtaW5NYXhWYWx1ZSc7XG4gICAgaWYgKGNvbXB1dGVQb3NpdGlvbnMpIHtcbiAgICAgIHJldHVyblZhbHVlID0gJ21pbk1heFBvc2l0aW9uJztcbiAgICB9IGVsc2UgaWYgKHBvb2xUeXBlID09PSAnYXZnJykge1xuICAgICAgcmV0dXJuVmFsdWUgPSBgYXZnVmFsdWUgLyAke2ZTaXplICogZlNpemV9LjBgO1xuICAgIH1cbiAgICBjb25zdCB4Um93c0xpbWl0ID0geFNoYXBlWzBdIC0gMC41O1xuICAgIGNvbnN0IHhDb2xzTGltaXQgPSB4U2hhcGVbMV0gLSAwLjU7XG4gICAgdGhpcy5wYXJhbXMgPSBbc3RyaWRlLCBwYWQsIGZTaXplLCBjb21wdXRlUG9zaXRpb25zXTtcbiAgICB0aGlzLm91dHB1dFNoYXBlID1cbiAgICAgICAgY29udl91dGlsLmNvbXB1dGVPdXRwdXRTaGFwZTNEKHhTaGFwZSwgZlNpemUsIHhTaGFwZVsyXSwgc3RyaWRlLCBwYWQpO1xuXG4gICAgdGhpcy51c2VyQ29kZSA9IGBcbiAgICAgIHZvaWQgbWFpbigpIHtcbiAgICAgICAgdmVjMyBjb29yZHMgPSBnZXRPdXRwdXRDb29yZHMoKTtcbiAgICAgICAgZmxvYXQgeVIgPSBjb29yZHMueDtcbiAgICAgICAgZmxvYXQgeUMgPSBjb29yZHMueTtcbiAgICAgICAgZmxvYXQgZCA9IGNvb3Jkcy56O1xuXG4gICAgICAgIHZlYzIgeFJDQ29ybmVyID0gdmVjMih5UiwgeUMpICogdmVjMigke3N0cmlkZX0uMCwgJHtzdHJpZGV9LjApIC1cbiAgICAgICAgICAgIHZlYzIoJHtwYWR9LjAsICR7cGFkfS4wKTtcbiAgICAgICAgZmxvYXQgeFJDb3JuZXIgPSB4UkNDb3JuZXIueDtcbiAgICAgICAgZmxvYXQgeENDb3JuZXIgPSB4UkNDb3JuZXIueTtcblxuICAgICAgICAvLyBtYXgvbWluIHgoPywgPywgZCkgdG8gZ2V0IHkoeVIsIHlDLCBkKS5cbiAgICAgICAgLy8gPyA9IHRvIGJlIGRldGVybWluZWRcbiAgICAgICAgZmxvYXQgbWluTWF4VmFsdWUgPSAwLjA7XG4gICAgICAgIGZsb2F0IG1pbk1heFZhbHVlRm91bmQgPSAwLjA7XG4gICAgICAgIGZsb2F0IG1pbk1heFBvc2l0aW9uID0gMC4wO1xuICAgICAgICBmbG9hdCBhdmdWYWx1ZSA9IDAuMDtcblxuICAgICAgICBmb3IgKGludCBpd1IgPSAwOyBpd1IgPCAke2ZTaXplfTsgaXdSKyspIHtcbiAgICAgICAgICBmbG9hdCB3UiA9IGZsb2F0KGl3Uik7XG4gICAgICAgICAgZmxvYXQgeFIgPSB4UkNvcm5lciArIHdSO1xuXG4gICAgICAgICAgaWYgKHhSIDwgMC4wIHx8IHhSID4gJHt4Um93c0xpbWl0fSkge1xuICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgfVxuXG4gICAgICAgICAgZm9yIChpbnQgaXdDID0gMDsgaXdDIDwgJHtmU2l6ZX07IGl3QysrKSB7XG4gICAgICAgICAgICBmbG9hdCB3QyA9IGZsb2F0KGl3Qyk7XG4gICAgICAgICAgICBmbG9hdCB4QyA9IHhDQ29ybmVyICsgd0M7XG5cbiAgICAgICAgICAgIGlmICh4QyA8IDAuMCB8fCB4QyA+ICR7eENvbHNMaW1pdH0pIHtcbiAgICAgICAgICAgICAgY29udGludWU7XG4gICAgICAgICAgICB9XG5cbiAgICAgICAgICAgIGZsb2F0IHZhbHVlID0gZ2V0WCh4UiwgeEMsIGQpO1xuXG4gICAgICAgICAgICBpZiAoaXNOYU4odmFsdWUpKSB7XG4gICAgICAgICAgICAgIHNldE91dHB1dCh2YWx1ZSk7XG4gICAgICAgICAgICAgIHJldHVybjtcbiAgICAgICAgICAgIH1cblxuICAgICAgICAgICAgaWYgKCR7cG9vbFR5cGUgPT09ICdhdmcnfSkge1xuICAgICAgICAgICAgICBhdmdWYWx1ZSArPSB2YWx1ZTtcbiAgICAgICAgICAgIH0gZWxzZSB7XG4gICAgICAgICAgICAgIC8vIElmIGEgbWluIC8gbWF4IHZhbHVlIGhhcyBhbHJlYWR5IGJlZW4gZm91bmQsIHVzZSBpdC4gSWYgbm90LFxuICAgICAgICAgICAgICAvLyB1c2UgdGhlIGN1cnJlbnQgdmFsdWUuXG4gICAgICAgICAgICAgIGZsb2F0IGN1cnJNaW5NYXhWYWx1ZSA9IG1peChcbiAgICAgICAgICAgICAgICAgIHZhbHVlLCBtaW5NYXhWYWx1ZSwgbWluTWF4VmFsdWVGb3VuZCk7XG4gICAgICAgICAgICAgIGlmICh2YWx1ZSAke3Bvb2xUeXBlID09PSAnbWluJyA/ICc8PScgOiAnPj0nfSBjdXJyTWluTWF4VmFsdWUpIHtcbiAgICAgICAgICAgICAgICBtaW5NYXhWYWx1ZSA9IHZhbHVlO1xuICAgICAgICAgICAgICAgIG1pbk1heFZhbHVlRm91bmQgPSAxLjA7XG4gICAgICAgICAgICAgICAgaWYgKCR7Y29tcHV0ZVBvc2l0aW9uc30pIHtcbiAgICAgICAgICAgICAgICAgIG1pbk1heFBvc2l0aW9uID0gd1IgKiAke2ZTaXplfS4wICsgd0M7XG4gICAgICAgICAgICAgICAgfVxuICAgICAgICAgICAgICB9XG4gICAgICAgICAgICB9XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIHNldE91dHB1dCgke3JldHVyblZhbHVlfSk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uLy4uL3V0aWwnO1xuXG5leHBvcnQgdHlwZSBTaGFwZUluZm8gPSB7XG4gIGxvZ2ljYWxTaGFwZTogbnVtYmVyW10sXG4gIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdXG59O1xuXG5leHBvcnQgdHlwZSBJbnB1dEluZm8gPSB7XG4gIG5hbWU6IHN0cmluZyxcbiAgc2hhcGVJbmZvOiBTaGFwZUluZm9cbn07XG5cbmV4cG9ydCBmdW5jdGlvbiBtYWtlU2hhZGVyKFxuICAgIGlucHV0c0luZm86IElucHV0SW5mb1tdLCBvdXRwdXRTaGFwZTogU2hhcGVJbmZvLCB1c2VyQ29kZTogc3RyaW5nLFxuICAgIGJyb2FkY2FzdDogYm9vbGVhbik6IHN0cmluZyB7XG4gIGNvbnN0IGlucHV0UHJlZml4U25pcHBldCA9XG4gICAgICBpbnB1dHNJbmZvLm1hcCh4ID0+IGB1bmlmb3JtIHNhbXBsZXIyRCAke3gubmFtZX07YCkuam9pbignXFxuJyk7XG4gIGNvbnN0IGlucHV0U2FtcGxpbmdTbmlwcGV0ID1cbiAgICAgIGlucHV0c0luZm8ubWFwKHggPT4gZ2V0SW5wdXRTYW1wbGluZ1NuaXBwZXQoeCwgb3V0cHV0U2hhcGUsIGJyb2FkY2FzdCkpXG4gICAgICAgICAgLmpvaW4oJ1xcbicpO1xuICBjb25zdCBvdXRUZXhTaGFwZSA9IG91dHB1dFNoYXBlLnRleFNoYXBlO1xuICBjb25zdCBvdXRwdXRTYW1wbGluZ1NuaXBwZXQgPVxuICAgICAgZ2V0T3V0cHV0U2FtcGxpbmdTbmlwcGV0KG91dHB1dFNoYXBlLmxvZ2ljYWxTaGFwZSwgb3V0VGV4U2hhcGUpO1xuICBjb25zdCBzb3VyY2UgPSBbXG4gICAgU0hBREVSX1BSRUZJWCwgaW5wdXRQcmVmaXhTbmlwcGV0LCBpbnB1dFNhbXBsaW5nU25pcHBldCxcbiAgICBvdXRwdXRTYW1wbGluZ1NuaXBwZXQsIHVzZXJDb2RlXG4gIF0uam9pbignXFxuJyk7XG4gIHJldHVybiBzb3VyY2U7XG59XG5cbmZ1bmN0aW9uIGdldElucHV0U2FtcGxpbmdTbmlwcGV0KFxuICAgIGluSW5mbzogSW5wdXRJbmZvLCBvdXRTaGFwZUluZm86IFNoYXBlSW5mbywgYnJvYWRjYXN0OiBib29sZWFuKSB7XG4gIGNvbnN0IHNoYXBlID0gaW5JbmZvLnNoYXBlSW5mby5sb2dpY2FsU2hhcGU7XG4gIGNvbnN0IHRleFNoYXBlID0gaW5JbmZvLnNoYXBlSW5mby50ZXhTaGFwZTtcbiAgY29uc3Qgb3V0VGV4U2hhcGUgPSBvdXRTaGFwZUluZm8udGV4U2hhcGU7XG5cbiAgbGV0IHJlcyA9ICcnO1xuICBzd2l0Y2ggKHNoYXBlLmxlbmd0aCkge1xuICAgIGNhc2UgMDpcbiAgICAgIHJlcyArPSBnZXRTYW1wbGVyU2NhbGFyKGluSW5mby5uYW1lKTtcbiAgICAgIGJyZWFrO1xuICAgIGNhc2UgMTpcbiAgICAgIHJlcyArPSBnZXRTYW1wbGVyMUQoaW5JbmZvLm5hbWUsIHRleFNoYXBlKTtcbiAgICAgIGJyZWFrO1xuICAgIGNhc2UgMjpcbiAgICAgIHJlcyArPSBnZXRTYW1wbGVyMkQoaW5JbmZvLm5hbWUsIHNoYXBlIGFzIFtudW1iZXIsIG51bWJlcl0sIHRleFNoYXBlKTtcbiAgICAgIGJyZWFrO1xuICAgIGNhc2UgMzpcbiAgICAgIHJlcyArPSBnZXRTYW1wbGVyM0QoXG4gICAgICAgICAgaW5JbmZvLm5hbWUsIHNoYXBlIGFzIFtudW1iZXIsIG51bWJlciwgbnVtYmVyXSwgdGV4U2hhcGUpO1xuICAgICAgYnJlYWs7XG4gICAgY2FzZSA0OlxuICAgICAgcmVzICs9IGdldFNhbXBsZXI0RChcbiAgICAgICAgICBpbkluZm8ubmFtZSwgc2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIHRleFNoYXBlKTtcbiAgICAgIGJyZWFrO1xuICAgIGRlZmF1bHQ6XG4gICAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICAgYCR7c2hhcGUubGVuZ3RofS1EIGlucHV0IHNhbXBsaW5nYCArXG4gICAgICAgICAgYCBpcyBub3QgeWV0IHN1cHBvcnRlZGApO1xuICB9XG4gIC8vIElmIGlucHV0IGFuZCBvdXRwdXQgaGF2ZSBtYXRjaGluZyBsb2dpY2FsIHNoYXBlcywgYWRkXG4gIC8vIGdldFRleE5hbWVBdE91dENvb3JkKCkgbWV0aG9kIHRoYXQgc2FtcGxlcyB0aGUgaW5wdXQgdGV4dHVyZSB1c2luZyB0aGVcbiAgLy8gb3V0cHV0IGNvb3JkaW5hdGVzLlxuICBpZiAoYnJvYWRjYXN0IHx8XG4gICAgICB1dGlsLmFycmF5c0VxdWFsKFxuICAgICAgICAgIGluSW5mby5zaGFwZUluZm8ubG9naWNhbFNoYXBlLCBvdXRTaGFwZUluZm8ubG9naWNhbFNoYXBlKSkge1xuICAgIHJlcyArPVxuICAgICAgICBnZXRTYW1wbGVyQXRPdXRwdXRDb29yZHMoaW5JbmZvLm5hbWUsIHRleFNoYXBlLCBvdXRUZXhTaGFwZSwgYnJvYWRjYXN0KTtcbiAgfVxuICByZXMgKz0gZ2V0U2FtcGxlckZsYXQoaW5JbmZvLm5hbWUsIHRleFNoYXBlKTtcbiAgcmV0dXJuIHJlcztcbn1cblxuZnVuY3Rpb24gZ2V0T3V0cHV0U2FtcGxpbmdTbmlwcGV0KFxuICAgIG91dFNoYXBlOiBudW1iZXJbXSwgb3V0VGV4U2hhcGU6IFtudW1iZXIsIG51bWJlcl0pOiBzdHJpbmcge1xuICBzd2l0Y2ggKG91dFNoYXBlLmxlbmd0aCkge1xuICAgIGNhc2UgMDpcbiAgICAgIC8vIERvZXNuJ3QgbWFrZSBzZW5zZSB0byBjYWxsIGdldE91dHB1dENvb3JkcygpIHdoZW4gb3V0cHV0IGlzIHNjYWxhci5cbiAgICAgIHJldHVybiAnJztcbiAgICBjYXNlIDE6XG4gICAgICByZXR1cm4gZ2V0T3V0cHV0MURDb29yZHMob3V0U2hhcGUgYXMgW251bWJlcl0sIG91dFRleFNoYXBlKTtcbiAgICBjYXNlIDI6XG4gICAgICByZXR1cm4gZ2V0T3V0cHV0MkRDb29yZHMob3V0U2hhcGUgYXMgW251bWJlciwgbnVtYmVyXSwgb3V0VGV4U2hhcGUpO1xuICAgIGNhc2UgMzpcbiAgICAgIHJldHVybiBnZXRPdXRwdXQzRENvb3JkcyhcbiAgICAgICAgICBvdXRTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIG91dFRleFNoYXBlKTtcbiAgICBjYXNlIDQ6XG4gICAgICByZXR1cm4gZ2V0T3V0cHV0NERDb29yZHMoXG4gICAgICAgICAgb3V0U2hhcGUgYXMgW251bWJlciwgbnVtYmVyLCBudW1iZXIsIG51bWJlcl0sIG91dFRleFNoYXBlKTtcbiAgICBkZWZhdWx0OlxuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGAke291dFNoYXBlLmxlbmd0aH0tRCBvdXRwdXQgc2FtcGxpbmcgaXMgbm90IHlldCBzdXBwb3J0ZWRgKTtcbiAgfVxufVxuXG5jb25zdCBTQU1QTEVfMURfU05JUFBFVCA9IGBcbnZlYzIgVVZmcm9tMUQoZmxvYXQgdGV4TnVtUiwgZmxvYXQgdGV4TnVtQywgZmxvYXQgaW5kZXgpIHtcbiAgZmxvYXQgdGV4UiA9IGZsb29yKGluZGV4IC8gdGV4TnVtQyk7XG4gIGZsb2F0IHRleEMgPSBtb2QoaW5kZXgsIHRleE51bUMpO1xuICByZXR1cm4gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC8gdmVjMih0ZXhOdW1DLCB0ZXhOdW1SKTtcbn1cbmA7XG5cbmNvbnN0IFNBTVBMRV8yRF9TTklQUEVUID0gYFxudmVjMiBVVmZyb20yRChmbG9hdCB0ZXhOdW1SLCBmbG9hdCB0ZXhOdW1DLCBmbG9hdCBudW1DLCBmbG9hdCByb3csXG4gICAgZmxvYXQgY29sKSB7XG4gIGZsb2F0IGluZGV4ID0gZG90KHZlYzIocm93LCBjb2wpLCB2ZWMyKG51bUMsIDEuMCkpO1xuICBmbG9hdCB0ZXhSID0gZmxvb3IoaW5kZXggLyB0ZXhOdW1DKTtcbiAgZmxvYXQgdGV4QyA9IG1vZChpbmRleCwgdGV4TnVtQyk7XG4gIHJldHVybiAodmVjMih0ZXhDLCB0ZXhSKSArIGhhbGZDUikgLyB2ZWMyKHRleE51bUMsIHRleE51bVIpO1xufVxuYDtcblxuY29uc3QgU0FNUExFXzNEX1NOSVBQRVQgPSBgXG52ZWMyIFVWZnJvbTNEKGZsb2F0IHRleE51bVIsIGZsb2F0IHRleE51bUMsIGZsb2F0IHN0cmlkZTAsXG4gICAgZmxvYXQgc3RyaWRlMSwgZmxvYXQgcm93LCBmbG9hdCBjb2wsIGZsb2F0IGRlcHRoKSB7XG4gIGZsb2F0IGluZGV4ID0gZG90KHZlYzMocm93LCBjb2wsIGRlcHRoKSwgdmVjMyhzdHJpZGUwLCBzdHJpZGUxLCAxLjApKTtcbiAgZmxvYXQgdGV4UiA9IGZsb29yKGluZGV4IC8gdGV4TnVtQyk7XG4gIGZsb2F0IHRleEMgPSBtb2QoaW5kZXgsIHRleE51bUMpO1xuICByZXR1cm4gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC8gdmVjMih0ZXhOdW1DLCB0ZXhOdW1SKTtcbn1cbmA7XG5cbmNvbnN0IFNBTVBMRV80RF9TTklQUEVUID0gYFxudmVjMiBVVmZyb200RChmbG9hdCB0ZXhOdW1SLCBmbG9hdCB0ZXhOdW1DLCBmbG9hdCBzdHJpZGUwLFxuICAgIGZsb2F0IHN0cmlkZTEsIGZsb2F0IHN0cmlkZTIsIGZsb2F0IHJvdywgZmxvYXQgY29sLCBmbG9hdCBkZXB0aCxcbiAgICBmbG9hdCBkZXB0aDIpIHtcbiAgZmxvYXQgaW5kZXggPSBkb3QodmVjNChyb3csIGNvbCwgZGVwdGgsIGRlcHRoMiksXG4gICAgICAgICAgICAgICAgICAgIHZlYzQoc3RyaWRlMCwgc3RyaWRlMSwgc3RyaWRlMiwgMS4wKSk7XG4gIGZsb2F0IHRleFIgPSBmbG9vcihpbmRleCAvIHRleE51bUMpO1xuICBmbG9hdCB0ZXhDID0gbW9kKGluZGV4LCB0ZXhOdW1DKTtcbiAgcmV0dXJuICh2ZWMyKHRleEMsIHRleFIpICsgaGFsZkNSKSAvIHZlYzIodGV4TnVtQywgdGV4TnVtUik7XG59XG5gO1xuXG5jb25zdCBTSEFERVJfUFJFRklYID0gYFxuICBwcmVjaXNpb24gaGlnaHAgZmxvYXQ7XG4gIHZhcnlpbmcgdmVjMiByZXN1bHRVVjtcbiAgY29uc3QgdmVjMiBoYWxmQ1IgPSB2ZWMyKDAuNSwgMC41KTtcblxuICBmbG9hdCBzYW1wbGUoc2FtcGxlcjJEIHRleHR1cmUsIHZlYzIgdXYpIHtcbiAgICByZXR1cm4gdGV4dHVyZTJEKHRleHR1cmUsIHV2KS5yO1xuICB9XG5cbiAgdm9pZCBzZXRPdXRwdXQoZmxvYXQgdmFsKSB7XG4gICAgZ2xfRnJhZ0NvbG9yID0gdmVjNCh2YWwsIDAsIDAsIDApO1xuICB9XG5cbiAgYm9vbCBpc05hTihmbG9hdCB2YWwpIHtcbiAgICByZXR1cm4gdmFsID09IHZhbCA/IGZhbHNlIDogdHJ1ZTtcbiAgfVxuICAke1NBTVBMRV8xRF9TTklQUEVUfVxuICAke1NBTVBMRV8yRF9TTklQUEVUfVxuICAke1NBTVBMRV8zRF9TTklQUEVUfVxuICAke1NBTVBMRV80RF9TTklQUEVUfVxuYDtcblxuZnVuY3Rpb24gZ2V0T3V0cHV0MURDb29yZHMoXG4gICAgc2hhcGU6IFtudW1iZXJdLCB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSk6IHN0cmluZyB7XG4gIGlmICh0ZXhTaGFwZVswXSA9PT0gMSkge1xuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICAgIHJldHVybiBmbG9vcihnbF9GcmFnQ29vcmQueCk7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICBpZiAodGV4U2hhcGVbMV0gPT09IDEpIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgICByZXR1cm4gZmxvb3IoZ2xfRnJhZ0Nvb3JkLnkpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgcmV0dXJuIGBcbiAgICBmbG9hdCBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICB2ZWMyIHJlc1RleFJDID0gZmxvb3IoZ2xfRnJhZ0Nvb3JkLnl4KTtcbiAgICAgIHJldHVybiBkb3QocmVzVGV4UkMsIHZlYzIoJHt0ZXhTaGFwZVsxXX0uMCwgMS4wKSk7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRPdXRwdXQzRENvb3JkcyhcbiAgICBzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLCB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSk6IHN0cmluZyB7XG4gIGNvbnN0IHN0cmlkZTAgPSBzaGFwZVsxXSAqIHNoYXBlWzJdO1xuICBjb25zdCBzdHJpZGUxID0gc2hhcGVbMl07XG4gIHJldHVybiBgXG4gICAgdmVjMyBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICB2ZWMyIHJlc1RleFJDID0gZmxvb3IoZ2xfRnJhZ0Nvb3JkLnl4KTtcbiAgICAgIGZsb2F0IGluZGV4ID0gZG90KHJlc1RleFJDLCB2ZWMyKCR7dGV4U2hhcGVbMV19LjAsIDEuMCkpO1xuICAgICAgZmxvYXQgciA9IGZsb29yKGluZGV4IC8gJHtzdHJpZGUwfS4wKTtcbiAgICAgIGluZGV4IC09IHIgKiAke3N0cmlkZTB9LjA7XG4gICAgICBmbG9hdCBjID0gZmxvb3IoaW5kZXggLyAke3N0cmlkZTF9LjApO1xuICAgICAgZmxvYXQgZCA9IG1vZChpbmRleCwgJHtzdHJpZGUxfS4wKTtcbiAgICAgIHJldHVybiB2ZWMzKHIsIGMsIGQpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0T3V0cHV0NERDb29yZHMoXG4gICAgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogc3RyaW5nIHtcbiAgY29uc3Qgc3RyaWRlMiA9IHNoYXBlWzNdO1xuICBjb25zdCBzdHJpZGUxID0gc2hhcGVbMl0gKiBzdHJpZGUyO1xuICBjb25zdCBzdHJpZGUwID0gc2hhcGVbMV0gKiBzdHJpZGUxO1xuICByZXR1cm4gYFxuICAgIHZlYzQgZ2V0T3V0cHV0Q29vcmRzKCkge1xuICAgICAgdmVjMiByZXNUZXhSQyA9IGZsb29yKGdsX0ZyYWdDb29yZC55eCk7XG4gICAgICBmbG9hdCBpbmRleCA9IGRvdChyZXNUZXhSQywgdmVjMigke3RleFNoYXBlWzFdfS4wLCAxLjApKTtcblxuICAgICAgZmxvYXQgciA9IGZsb29yKGluZGV4IC8gJHtzdHJpZGUwfS4wKTtcbiAgICAgIGluZGV4IC09IHIgKiAke3N0cmlkZTB9LjA7XG5cbiAgICAgIGZsb2F0IGMgPSBmbG9vcihpbmRleCAvICR7c3RyaWRlMX0uMCk7XG4gICAgICBpbmRleCAtPSBjICogJHtzdHJpZGUxfS4wO1xuXG4gICAgICBmbG9hdCBkID0gZmxvb3IoaW5kZXggLyAke3N0cmlkZTJ9LjApO1xuICAgICAgZmxvYXQgZDIgPSBtb2QoaW5kZXgsICR7c3RyaWRlMn0uMCk7XG5cbiAgICAgIHJldHVybiB2ZWM0KHIsIGMsIGQsIGQyKTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldE91dHB1dDJEQ29vcmRzKFxuICAgIHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLCB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSk6IHN0cmluZyB7XG4gIGlmICh1dGlsLmFycmF5c0VxdWFsKHNoYXBlLCB0ZXhTaGFwZSkpIHtcbiAgICByZXR1cm4gYFxuICAgICAgdmVjMiBnZXRPdXRwdXRDb29yZHMoKSB7XG4gICAgICAgIHJldHVybiBmbG9vcihnbF9GcmFnQ29vcmQueXgpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgcmV0dXJuIGBcbiAgICB2ZWMyIGdldE91dHB1dENvb3JkcygpIHtcbiAgICAgIHZlYzIgcmVzVGV4UkMgPSBmbG9vcihnbF9GcmFnQ29vcmQueXgpO1xuICAgICAgZmxvYXQgaW5kZXggPSBkb3QocmVzVGV4UkMsIHZlYzIoJHt0ZXhTaGFwZVsxXX0uMCwgMS4wKSk7XG4gICAgICBmbG9hdCByID0gZmxvb3IoaW5kZXggLyAke3NoYXBlWzFdfS4wKTtcbiAgICAgIGZsb2F0IGMgPSBtb2QoaW5kZXgsICR7c2hhcGVbMV19LjApO1xuICAgICAgcmV0dXJuIHZlYzIociwgYyk7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRTYW1wbGVyU2NhbGFyKHRleE5hbWU6IHN0cmluZyk6IHN0cmluZyB7XG4gIGNvbnN0IGZ1bmNOYW1lID0gJ2dldCcgKyB0ZXhOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKTtcbiAgcmV0dXJuIGBcbiAgICBmbG9hdCAke2Z1bmNOYW1lfSgpIHtcbiAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgaGFsZkNSKTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldFNhbXBsZXIxRCh0ZXhOYW1lOiBzdHJpbmcsIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogc3RyaW5nIHtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCB0UiA9IHRleFNoYXBlWzBdO1xuICBjb25zdCB0QyA9IHRleFNoYXBlWzFdO1xuICBpZiAodGV4U2hhcGVbMF0gPT09IDEgJiYgdGV4U2hhcGVbMV0gPT09IDEpIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oZmxvYXQgaW5kZXgpIHtcbiAgICAgICAgcmV0dXJuIHNhbXBsZSgke3RleE5hbWV9LCBoYWxmQ1IpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgaWYgKHRleFNoYXBlWzFdID09PSAxKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IGluZGV4KSB7XG4gICAgICAgIHZlYzIgdXYgPSB2ZWMyKDAuNSwgKGluZGV4ICsgMC41KSAvICR7dFJ9LjApO1xuICAgICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG4gIGlmICh0ZXhTaGFwZVswXSA9PT0gMSkge1xuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShmbG9hdCBpbmRleCkge1xuICAgICAgICB2ZWMyIHV2ID0gdmVjMigoaW5kZXggKyAwLjUpIC8gJHt0Q30uMCwgMC41KTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZSgke3RleE5hbWV9LCB1dik7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICByZXR1cm4gYFxuICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IGluZGV4KSB7XG4gICAgICB2ZWMyIHV2ID0gVVZmcm9tMUQoJHt0Un0uMCwgJHt0Q30uMCwgaW5kZXgpO1xuICAgICAgcmV0dXJuIHNhbXBsZSgke3RleE5hbWV9LCB1dik7XG4gICAgfVxuICBgO1xufVxuXG5mdW5jdGlvbiBnZXRTYW1wbGVyM0QoXG4gICAgdGV4TmFtZTogc3RyaW5nLCBzaGFwZTogW251bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogc3RyaW5nIHtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCB0UiA9IHRleFNoYXBlWzBdO1xuICBjb25zdCB0QyA9IHRleFNoYXBlWzFdO1xuICBjb25zdCBzdHJpZGUwID0gc2hhcGVbMV0gKiBzaGFwZVsyXTtcbiAgY29uc3Qgc3RyaWRlMSA9IHNoYXBlWzJdO1xuICBpZiAodEMgPT09IHN0cmlkZTApIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oZmxvYXQgcm93LCBmbG9hdCBjb2wsIGZsb2F0IGRlcHRoKSB7XG4gICAgICAgIGZsb2F0IHRleFIgPSByb3c7XG4gICAgICAgIGZsb2F0IHRleEMgPSBkb3QodmVjMihjb2wsIGRlcHRoKSwgdmVjMigke3N0cmlkZTF9LCAxLjApKTtcbiAgICAgICAgdmVjMiB1diA9ICh2ZWMyKHRleEMsIHRleFIpICsgaGFsZkNSKSAvIHZlYzIoJHt0Q30uMCwgJHt0Un0uMCk7XG4gICAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgcmV0dXJuIGBcbiAgICBmbG9hdCAke2Z1bmNOYW1lfShmbG9hdCByb3csIGZsb2F0IGNvbCwgZmxvYXQgZGVwdGgpIHtcbiAgICAgIHZlYzIgdXYgPSBVVmZyb20zRCgke3RSfS4wLCAke3RDfS4wLCAke3N0cmlkZTB9LjAsICR7c3RyaWRlMX0uMCwgcm93LFxuICAgICAgICBjb2wsIGRlcHRoKTtcbiAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0U2FtcGxlcjREKFxuICAgIHRleE5hbWU6IHN0cmluZywgc2hhcGU6IFtudW1iZXIsIG51bWJlciwgbnVtYmVyLCBudW1iZXJdLFxuICAgIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogc3RyaW5nIHtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCB0UiA9IHRleFNoYXBlWzBdO1xuICBjb25zdCB0QyA9IHRleFNoYXBlWzFdO1xuICBjb25zdCBzdHJpZGUyID0gc2hhcGVbM107XG4gIGNvbnN0IHN0cmlkZTEgPSBzaGFwZVsyXSAqIHN0cmlkZTI7XG4gIGNvbnN0IHN0cmlkZTAgPSBzaGFwZVsxXSAqIHN0cmlkZTE7XG5cbiAgaWYgKHRDID09PSBzdHJpZGUwKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IHJvdywgZmxvYXQgY29sLCBmbG9hdCBkZXB0aCwgZmxvYXQgZGVwdGgyKSB7XG4gICAgICAgIGZsb2F0IHRleFIgPSByb3c7XG4gICAgICAgIGZsb2F0IHRleEMgPSBkb3QodmVjMyhjb2wsIGRlcHRoLCBkZXB0aDIpLFxuICAgICAgICAgICAgICAgICAgICAgICAgIHZlYzMoJHtzdHJpZGUxfS4wLCAke3N0cmlkZTJ9LjAsIDEuMCkpO1xuICAgICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC8gdmVjMigke3RDfS4wLCAke3RSfS4wKTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZSgke3RleE5hbWV9LCB1dik7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICByZXR1cm4gYFxuICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IHJvdywgZmxvYXQgY29sLCBmbG9hdCBkZXB0aCwgZmxvYXQgZGVwdGgyKSB7XG4gICAgICB2ZWMyIHV2ID0gVVZmcm9tNEQoJHt0Un0uMCwgJHt0Q30uMCwgJHtzdHJpZGUwfS4wLCAke3N0cmlkZTF9LjAsXG4gICAgICAgICAgJHtzdHJpZGUyfS4wLCByb3csIGNvbCwgZGVwdGgsIGRlcHRoMik7XG4gICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICB9XG4gIGA7XG59XG5cbmZ1bmN0aW9uIGdldFNhbXBsZXIyRChcbiAgICB0ZXhOYW1lOiBzdHJpbmcsIHNoYXBlOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgIHRleFNoYXBlOiBbbnVtYmVyLCBudW1iZXJdKTogc3RyaW5nIHtcbiAgY29uc3QgZnVuY05hbWUgPSAnZ2V0JyArIHRleE5hbWUuY2hhckF0KDApLnRvVXBwZXJDYXNlKCkgKyB0ZXhOYW1lLnNsaWNlKDEpO1xuICBjb25zdCB0UiA9IHRleFNoYXBlWzBdO1xuICBjb25zdCB0QyA9IHRleFNoYXBlWzFdO1xuICBpZiAodXRpbC5hcnJheXNFcXVhbChzaGFwZSwgdGV4U2hhcGUpKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IHJvdywgZmxvYXQgY29sKSB7XG4gICAgICAgIHZlYzIgdXYgPSAodmVjMihjb2wsIHJvdykgKyBoYWxmQ1IpIC8gdmVjMigke3RDfS4wLCAke3RSfS4wKTtcbiAgICAgICAgcmV0dXJuIHNhbXBsZSgke3RleE5hbWV9LCB1dik7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICBpZiAodEMgPT09IDEpIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oZmxvYXQgcm93LCBmbG9hdCBjb2wpIHtcbiAgICAgICAgZmxvYXQgaW5kZXggPSBkb3QodmVjMihyb3csIGNvbCksIHZlYzIoJHtzaGFwZVsxXX0uMCwgMS4wKSk7XG4gICAgICAgIHZlYzIgdXYgPSB2ZWMyKDAuNSwgKGluZGV4ICsgMC41KSAvICR7dFJ9LjApO1xuICAgICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG4gIGlmICh0UiA9PT0gMSkge1xuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShmbG9hdCByb3csIGZsb2F0IGNvbCkge1xuICAgICAgICBmbG9hdCBpbmRleCA9IGRvdCh2ZWMyKHJvdywgY29sKSwgdmVjMigke3NoYXBlWzFdfS4wLCAxLjApKTtcbiAgICAgICAgdmVjMiB1diA9IHZlYzIoKGluZGV4ICsgMC41KSAvICR7dEN9LjAsIDAuNSk7XG4gICAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgcmV0dXJuIGBcbiAgICBmbG9hdCAke2Z1bmNOYW1lfShmbG9hdCByb3csIGZsb2F0IGNvbCkge1xuICAgICAgdmVjMiB1diA9IFVWZnJvbTJEKCR7dFJ9LjAsICR7dEN9LjAsICR7c2hhcGVbMV19LjAsIHJvdywgY29sKTtcbiAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0U2FtcGxlckZsYXQodGV4TmFtZTogc3RyaW5nLCB0ZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSk6IHN0cmluZyB7XG4gIGNvbnN0IGZ1bmNOYW1lID1cbiAgICAgICdnZXQnICsgdGV4TmFtZS5jaGFyQXQoMCkudG9VcHBlckNhc2UoKSArIHRleE5hbWUuc2xpY2UoMSkgKyAnRmxhdCc7XG4gIGNvbnN0IHROdW1SID0gdGV4U2hhcGVbMF07XG4gIGNvbnN0IHROdW1DID0gdGV4U2hhcGVbMV07XG4gIGlmICh0TnVtQyA9PT0gMSAmJiB0TnVtUiA9PT0gMSkge1xuICAgIHJldHVybiBgXG4gICAgICBmbG9hdCAke2Z1bmNOYW1lfShmbG9hdCBpbmRleCkge1xuICAgICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIGhhbGZDUik7XG4gICAgICB9XG4gICAgYDtcbiAgfVxuICBpZiAodE51bUMgPT09IDEpIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oZmxvYXQgaW5kZXgpIHtcbiAgICAgICAgdmVjMiB1diA9IHZlYzIoMC41LCAoaW5kZXggKyAwLjUpIC8gJHt0TnVtUn0uMCk7XG4gICAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgaWYgKHROdW1SID09PSAxKSB7XG4gICAgcmV0dXJuIGBcbiAgICAgIGZsb2F0ICR7ZnVuY05hbWV9KGZsb2F0IGluZGV4KSB7XG4gICAgICAgIHZlYzIgdXYgPSB2ZWMyKChpbmRleCArIDAuNSkgLyAke3ROdW1DfS4wLCAwLjUpO1xuICAgICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICAgIH1cbiAgICBgO1xuICB9XG4gIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oZmxvYXQgaW5kZXgpIHtcbiAgICAgIGZsb2F0IHRleFIgPSBmbG9vcihpbmRleCAvICR7dE51bUN9LjApO1xuICAgICAgZmxvYXQgdGV4QyA9IG1vZChpbmRleCwgJHt0TnVtQ30uMCk7XG4gICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC8gdmVjMigke3ROdW1DfS4wLCAke3ROdW1SfS4wKTtcbiAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgdXYpO1xuICAgIH1cbiAgYDtcbn1cblxuZnVuY3Rpb24gZ2V0U2FtcGxlckF0T3V0cHV0Q29vcmRzKFxuICAgIHRleE5hbWU6IHN0cmluZywgaW5UZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSxcbiAgICBvdXRUZXhTaGFwZTogW251bWJlciwgbnVtYmVyXSwgYnJvYWRjYXN0OiBib29sZWFuKSB7XG4gIGNvbnN0IGZ1bmNOYW1lID0gJ2dldCcgKyB0ZXhOYW1lLmNoYXJBdCgwKS50b1VwcGVyQ2FzZSgpICsgdGV4TmFtZS5zbGljZSgxKSArXG4gICAgICAnQXRPdXRDb29yZHMnO1xuICBpZiAodXRpbC5hcnJheXNFcXVhbChpblRleFNoYXBlLCBvdXRUZXhTaGFwZSkpIHtcbiAgICByZXR1cm4gYFxuICAgICAgZmxvYXQgJHtmdW5jTmFtZX0oKSB7XG4gICAgICAgIHJldHVybiBzYW1wbGUoJHt0ZXhOYW1lfSwgcmVzdWx0VVYpO1xuICAgICAgfVxuICAgIGA7XG4gIH1cbiAgY29uc3QgaW5TaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKGluVGV4U2hhcGUpO1xuICBjb25zdCBicm9hZGNhc3RTbmlwcGV0ID0gYnJvYWRjYXN0ID8gYGluZGV4ID0gbW9kKGluZGV4LCAke2luU2l6ZX0uMCk7YCA6ICcnO1xuXG4gIHJldHVybiBgXG4gICAgZmxvYXQgJHtmdW5jTmFtZX0oKSB7XG4gICAgICB2ZWMyIHJlc1RleFJDID0gZmxvb3IoZ2xfRnJhZ0Nvb3JkLnl4KTtcbiAgICAgIGZsb2F0IGluZGV4ID0gZG90KHJlc1RleFJDLCB2ZWMyKCR7b3V0VGV4U2hhcGVbMV19LjAsIDEuMCkpO1xuICAgICAgJHticm9hZGNhc3RTbmlwcGV0fVxuICAgICAgZmxvYXQgdGV4UiA9IGZsb29yKGluZGV4IC8gJHtpblRleFNoYXBlWzFdfS4wKTtcbiAgICAgIGZsb2F0IHRleEMgPSBtb2QoaW5kZXgsICR7aW5UZXhTaGFwZVsxXX0uMCk7XG4gICAgICB2ZWMyIHV2ID0gKHZlYzIodGV4QywgdGV4UikgKyBoYWxmQ1IpIC9cbiAgICAgICAgICAgICAgICAgdmVjMigke2luVGV4U2hhcGVbMV19LjAsICR7aW5UZXhTaGFwZVswXX0uMCk7XG4gICAgICByZXR1cm4gc2FtcGxlKCR7dGV4TmFtZX0sIHV2KTtcbiAgICB9XG4gIGA7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRVbnBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KFxuICAgIHJvd3M6IG51bWJlciwgY29sdW1uczogbnVtYmVyKTogW251bWJlciwgbnVtYmVyXSB7XG4gIHJldHVybiBbY29sdW1ucywgcm93c107XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRVbnBhY2tlZEFycmF5U2l6ZUZyb21NYXRyaXhTaXplKFxuICAgIG1hdHJpeFNpemU6IG51bWJlciwgY2hhbm5lbHNQZXJUZXh0dXJlOiBudW1iZXIpOiBudW1iZXIge1xuICByZXR1cm4gbWF0cml4U2l6ZSAqIGNoYW5uZWxzUGVyVGV4dHVyZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldENvbG9yTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQoXG4gICAgcm93czogbnVtYmVyLCBjb2x1bW5zOiBudW1iZXIpOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgcmV0dXJuIFtjb2x1bW5zICogNCwgcm93c107XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRNYXRyaXhTaXplRnJvbVVucGFja2VkQXJyYXlTaXplKFxuICAgIHVucGFja2VkU2l6ZTogbnVtYmVyLCBjaGFubmVsc1BlclRleHR1cmU6IG51bWJlcik6IG51bWJlciB7XG4gIGlmICh1bnBhY2tlZFNpemUgJSBjaGFubmVsc1BlclRleHR1cmUgIT09IDApIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICd1bnBhY2tlZFNpemUgKCcgKyB1bnBhY2tlZFNpemUgKyAnKSBtdXN0IGJlIGEgbXVsdGlwbGUgb2YgJyArXG4gICAgICAgIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG4gIH1cbiAgcmV0dXJuIHVucGFja2VkU2l6ZSAvIGNoYW5uZWxzUGVyVGV4dHVyZTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGVuY29kZU1hdHJpeFRvVW5wYWNrZWRBcnJheShcbiAgICBtYXRyaXg6IEZsb2F0MzJBcnJheSwgdW5wYWNrZWRBcnJheTogRmxvYXQzMkFycmF5LFxuICAgIGNoYW5uZWxzUGVyVGV4dHVyZTogbnVtYmVyKSB7XG4gIGNvbnN0IHJlcXVpcmVkU2l6ZSA9XG4gICAgICBnZXRVbnBhY2tlZEFycmF5U2l6ZUZyb21NYXRyaXhTaXplKG1hdHJpeC5sZW5ndGgsIGNoYW5uZWxzUGVyVGV4dHVyZSk7XG4gIGlmICh1bnBhY2tlZEFycmF5Lmxlbmd0aCA8IHJlcXVpcmVkU2l6ZSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ3VucGFja2VkQXJyYXkgbGVuZ3RoICgnICsgdW5wYWNrZWRBcnJheS5sZW5ndGggK1xuICAgICAgICAnKSBtdXN0IGJlID49ICcgKyByZXF1aXJlZFNpemUpO1xuICB9XG4gIGxldCBkc3QgPSAwO1xuICBmb3IgKGxldCBzcmMgPSAwOyBzcmMgPCBtYXRyaXgubGVuZ3RoOyArK3NyYykge1xuICAgIHVucGFja2VkQXJyYXlbZHN0XSA9IG1hdHJpeFtzcmNdO1xuICAgIGRzdCArPSBjaGFubmVsc1BlclRleHR1cmU7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGRlY29kZU1hdHJpeEZyb21VbnBhY2tlZEFycmF5KFxuICAgIHVucGFja2VkQXJyYXk6IEZsb2F0MzJBcnJheSwgbWF0cml4OiBGbG9hdDMyQXJyYXksXG4gICAgY2hhbm5lbHNQZXJUZXh0dXJlOiBudW1iZXIpIHtcbiAgY29uc3QgcmVxdWlyZWRTaXplID0gZ2V0TWF0cml4U2l6ZUZyb21VbnBhY2tlZEFycmF5U2l6ZShcbiAgICAgIHVucGFja2VkQXJyYXkubGVuZ3RoLCBjaGFubmVsc1BlclRleHR1cmUpO1xuICBpZiAobWF0cml4Lmxlbmd0aCA8IHJlcXVpcmVkU2l6ZSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ21hdHJpeCBsZW5ndGggKCcgKyBtYXRyaXgubGVuZ3RoICsgJykgbXVzdCBiZSA+PSAnICsgcmVxdWlyZWRTaXplKTtcbiAgfVxuICBsZXQgZHN0ID0gMDtcbiAgZm9yIChsZXQgc3JjID0gMDsgc3JjIDwgdW5wYWNrZWRBcnJheS5sZW5ndGg7IHNyYyArPSBjaGFubmVsc1BlclRleHR1cmUpIHtcbiAgICBtYXRyaXhbZHN0KytdID0gdW5wYWNrZWRBcnJheVtzcmNdO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChcbiAgICByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IFtudW1iZXIsIG51bWJlcl0ge1xuICByZXR1cm4gW01hdGguY2VpbChjb2x1bW5zIC8gMiksIE1hdGguY2VpbChyb3dzIC8gMildO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0UGFja2VkUkdCQUFycmF5U2l6ZUZyb21NYXRyaXhTaGFwZShcbiAgICByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcik6IG51bWJlciB7XG4gIGNvbnN0IFt3LCBoXSA9IGdldFBhY2tlZE1hdHJpeFRleHR1cmVTaGFwZVdpZHRoSGVpZ2h0KHJvd3MsIGNvbHVtbnMpO1xuICByZXR1cm4gdyAqIGggKiA0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZW5jb2RlTWF0cml4VG9QYWNrZWRSR0JBKFxuICAgIG1hdHJpeDogRmxvYXQzMkFycmF5LCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcixcbiAgICBwYWNrZWRSR0JBOiBGbG9hdDMyQXJyYXkpIHtcbiAgY29uc3QgcmVxdWlyZWRTaXplID0gZ2V0UGFja2VkUkdCQUFycmF5U2l6ZUZyb21NYXRyaXhTaGFwZShyb3dzLCBjb2x1bW5zKTtcbiAgaWYgKHBhY2tlZFJHQkEubGVuZ3RoIDwgcmVxdWlyZWRTaXplKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAncGFja2VkUkdCQSBsZW5ndGggKCcgKyBwYWNrZWRSR0JBLmxlbmd0aCArXG4gICAgICAgICcpIG11c3QgYmUgPj0gJyArIHJlcXVpcmVkU2l6ZSk7XG4gIH1cbiAgLypcbiAgICBVbnBhY2tlZCBtYXRyaXgsIHJvdy1tYWpvciBvcmRlciBpbiBGbG9hdDMyQXJyYXlbMTZdOiAgQSBCIEMgRFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBFIEYgRyBIXG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIEkgSiBLIExcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgTSBOIE8gUFxuXG4gICAgUGFja2VkIG1hdHJpeCwgMngyIFJHQkEzMiB0ZXh0dXJlIChtZW1vcnkgdmlldyk6ICAgICAgIEFCRUYgQ0RHSCBJSk1OIEtMT1BcblxuICAgIFBhY2tlZCBtYXRyaXgsIDJ4MiBSR0JBMzIgdGV4dHVyZSAobWF0cml4IHZpZXcpOiAgICAgICBBQnxDRFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBFRnxHSFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAtLSstLVxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBJSnxLTFxuICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBNTnxPUFxuICAgKi9cbiAgY29uc3QgW3RleHR1cmVXaWR0aCwgdGV4dHVyZUhlaWdodF0gPVxuICAgICAgZ2V0UGFja2VkTWF0cml4VGV4dHVyZVNoYXBlV2lkdGhIZWlnaHQocm93cywgY29sdW1ucyk7XG4gIGNvbnN0IG9kZFdpZHRoID0gKGNvbHVtbnMgJSAyKSA9PT0gMTtcbiAgY29uc3Qgb2RkSGVpZ2h0ID0gKHJvd3MgJSAyKSA9PT0gMTtcbiAgY29uc3Qgd2lkdGhJbkZ1bGxCbG9ja3MgPSBNYXRoLmZsb29yKGNvbHVtbnMgLyAyKTtcbiAgY29uc3QgaGVpZ2h0SW5GdWxsQmxvY2tzID0gTWF0aC5mbG9vcihyb3dzIC8gMik7XG5cbiAgLy8gbG9vcCBvdmVyIGZ1bGwgMngyIGJsb2Nrc1xuICB7XG4gICAgY29uc3QgZHN0U3RyaWRlID0gKG9kZFdpZHRoID8gNCA6IDApO1xuICAgIGNvbnN0IG9uZVJvdyA9IGNvbHVtbnM7XG4gICAgbGV0IGRzdCA9IDA7XG4gICAgZm9yIChsZXQgYmxvY2tZID0gMDsgYmxvY2tZIDwgaGVpZ2h0SW5GdWxsQmxvY2tzOyArK2Jsb2NrWSkge1xuICAgICAgY29uc3QgbWF0cml4U3JjUm93ID0gKGJsb2NrWSAqIDIgKiBjb2x1bW5zKTtcbiAgICAgIGZvciAobGV0IGJsb2NrWCA9IDA7IGJsb2NrWCA8IHdpZHRoSW5GdWxsQmxvY2tzOyArK2Jsb2NrWCkge1xuICAgICAgICBjb25zdCBtYXRyaXhTcmNDb2wgPSBibG9ja1ggKiAyO1xuICAgICAgICBjb25zdCBzcmMgPSBtYXRyaXhTcmNSb3cgKyBtYXRyaXhTcmNDb2w7XG4gICAgICAgIHBhY2tlZFJHQkFbZHN0XSA9IG1hdHJpeFtzcmNdO1xuICAgICAgICBwYWNrZWRSR0JBW2RzdCArIDFdID0gbWF0cml4W3NyYyArIDFdO1xuICAgICAgICBwYWNrZWRSR0JBW2RzdCArIDJdID0gbWF0cml4W3NyYyArIG9uZVJvd107XG4gICAgICAgIHBhY2tlZFJHQkFbZHN0ICsgM10gPSBtYXRyaXhbc3JjICsgb25lUm93ICsgMV07XG4gICAgICAgIGRzdCArPSA0O1xuICAgICAgfVxuICAgICAgZHN0ICs9IGRzdFN0cmlkZTtcbiAgICB9XG4gIH1cblxuICAvLyBsb29wIGRvd24gZmluYWwgb2RkIGNvbHVtblxuICBpZiAob2RkV2lkdGgpIHtcbiAgICBsZXQgc3JjID0gY29sdW1ucyAtIDE7XG4gICAgbGV0IGRzdCA9ICh0ZXh0dXJlV2lkdGggLSAxKSAqIDQ7XG4gICAgY29uc3Qgc3JjU3RyaWRlID0gMiAqIGNvbHVtbnM7XG4gICAgY29uc3QgZHN0U3RyaWRlID0gdGV4dHVyZVdpZHRoICogNDtcbiAgICBmb3IgKGxldCBibG9ja1kgPSAwOyBibG9ja1kgPCBoZWlnaHRJbkZ1bGxCbG9ja3M7ICsrYmxvY2tZKSB7XG4gICAgICBwYWNrZWRSR0JBW2RzdF0gPSBtYXRyaXhbc3JjXTtcbiAgICAgIHBhY2tlZFJHQkFbZHN0ICsgMl0gPSBtYXRyaXhbc3JjICsgY29sdW1uc107XG4gICAgICBzcmMgKz0gc3JjU3RyaWRlO1xuICAgICAgZHN0ICs9IGRzdFN0cmlkZTtcbiAgICB9XG4gIH1cblxuICAvLyBsb29wIGFjcm9zcyBmaW5hbCByb3dcbiAgaWYgKG9kZEhlaWdodCkge1xuICAgIGxldCBzcmMgPSAocm93cyAtIDEpICogY29sdW1ucztcbiAgICBsZXQgZHN0ID0gKHRleHR1cmVIZWlnaHQgLSAxKSAqIHRleHR1cmVXaWR0aCAqIDQ7XG4gICAgZm9yIChsZXQgYmxvY2tYID0gMDsgYmxvY2tYIDwgd2lkdGhJbkZ1bGxCbG9ja3M7ICsrYmxvY2tYKSB7XG4gICAgICBwYWNrZWRSR0JBW2RzdCsrXSA9IG1hdHJpeFtzcmMrK107XG4gICAgICBwYWNrZWRSR0JBW2RzdCsrXSA9IG1hdHJpeFtzcmMrK107XG4gICAgICBkc3QgKz0gMjtcbiAgICB9XG4gIH1cblxuICAvLyBmaWxsIGluIGJvdHRvbS1yaWdodCB0ZXhlbFxuICBpZiAob2RkV2lkdGggJiYgb2RkSGVpZ2h0KSB7XG4gICAgcGFja2VkUkdCQVtwYWNrZWRSR0JBLmxlbmd0aCAtIDRdID0gbWF0cml4W21hdHJpeC5sZW5ndGggLSAxXTtcbiAgfVxuXG4gIHJldHVybiBwYWNrZWRSR0JBO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZGVjb2RlTWF0cml4RnJvbVBhY2tlZFJHQkEoXG4gICAgcGFja2VkUkdCQTogRmxvYXQzMkFycmF5LCByb3dzOiBudW1iZXIsIGNvbHVtbnM6IG51bWJlcixcbiAgICBtYXRyaXg6IEZsb2F0MzJBcnJheSk6IEZsb2F0MzJBcnJheSB7XG4gIGNvbnN0IHJlcXVpcmVkU2l6ZSA9IHJvd3MgKiBjb2x1bW5zO1xuICBpZiAocmVxdWlyZWRTaXplIDwgbWF0cml4Lmxlbmd0aCkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ21hdHJpeCBsZW5ndGggKCcgKyBtYXRyaXgubGVuZ3RoICsgJykgbXVzdCBiZSA+PSAnICsgcmVxdWlyZWRTaXplKTtcbiAgfVxuICBjb25zdCBvZGRXaWR0aCA9IChjb2x1bW5zICUgMikgPT09IDE7XG4gIGNvbnN0IG9kZEhlaWdodCA9IChyb3dzICUgMikgPT09IDE7XG4gIGNvbnN0IHdpZHRoSW5GdWxsQmxvY2tzID0gTWF0aC5mbG9vcihjb2x1bW5zIC8gMik7XG4gIGNvbnN0IGhlaWdodEluRnVsbEJsb2NrcyA9IE1hdGguZmxvb3Iocm93cyAvIDIpO1xuICBjb25zdCBbdGV4dHVyZVdpZHRoLCB0ZXh0dXJlSGVpZ2h0XSA9XG4gICAgICBnZXRQYWNrZWRNYXRyaXhUZXh0dXJlU2hhcGVXaWR0aEhlaWdodChyb3dzLCBjb2x1bW5zKTtcblxuICAvLyBsb29wIG92ZXIgZnVsbCAyeDIgYmxvY2tzXG4gIHtcbiAgICBjb25zdCBzcmNTdHJpZGUgPSBvZGRXaWR0aCA/IDQgOiAwO1xuICAgIGNvbnN0IGRzdFN0cmlkZSA9IGNvbHVtbnMgKyAob2RkV2lkdGggPyAxIDogMCk7XG4gICAgbGV0IHNyYyA9IDA7XG4gICAgbGV0IGRzdFJvdzEgPSAwO1xuICAgIGxldCBkc3RSb3cyID0gY29sdW1ucztcbiAgICBmb3IgKGxldCBibG9ja1kgPSAwOyBibG9ja1kgPCBoZWlnaHRJbkZ1bGxCbG9ja3M7ICsrYmxvY2tZKSB7XG4gICAgICBmb3IgKGxldCBibG9ja1ggPSAwOyBibG9ja1ggPCB3aWR0aEluRnVsbEJsb2NrczsgKytibG9ja1gpIHtcbiAgICAgICAgbWF0cml4W2RzdFJvdzErK10gPSBwYWNrZWRSR0JBW3NyYysrXTtcbiAgICAgICAgbWF0cml4W2RzdFJvdzErK10gPSBwYWNrZWRSR0JBW3NyYysrXTtcbiAgICAgICAgbWF0cml4W2RzdFJvdzIrK10gPSBwYWNrZWRSR0JBW3NyYysrXTtcbiAgICAgICAgbWF0cml4W2RzdFJvdzIrK10gPSBwYWNrZWRSR0JBW3NyYysrXTtcbiAgICAgIH1cbiAgICAgIHNyYyArPSBzcmNTdHJpZGU7XG4gICAgICBkc3RSb3cxICs9IGRzdFN0cmlkZTtcbiAgICAgIGRzdFJvdzIgKz0gZHN0U3RyaWRlO1xuICAgIH1cbiAgfVxuXG4gIC8vIGxvb3AgZG93biBmaW5hbCBjb2x1bW5cbiAgaWYgKG9kZFdpZHRoKSB7XG4gICAgbGV0IHNyYyA9ICh0ZXh0dXJlV2lkdGggLSAxKSAqIDQ7XG4gICAgbGV0IGRzdCA9IGNvbHVtbnMgLSAxO1xuICAgIGNvbnN0IHNyY1N0cmlkZSA9IHRleHR1cmVXaWR0aCAqIDQ7XG4gICAgY29uc3QgZHN0U3RyaWRlID0gMiAqIGNvbHVtbnM7XG4gICAgZm9yIChsZXQgYmxvY2tZID0gMDsgYmxvY2tZIDwgaGVpZ2h0SW5GdWxsQmxvY2tzOyArK2Jsb2NrWSkge1xuICAgICAgbWF0cml4W2RzdF0gPSBwYWNrZWRSR0JBW3NyY107XG4gICAgICBtYXRyaXhbZHN0ICsgY29sdW1uc10gPSBwYWNrZWRSR0JBW3NyYyArIDJdO1xuICAgICAgc3JjICs9IHNyY1N0cmlkZTtcbiAgICAgIGRzdCArPSBkc3RTdHJpZGU7XG4gICAgfVxuICB9XG5cbiAgLy8gbG9vcCBhY3Jvc3MgZmluYWwgcm93XG4gIGlmIChvZGRIZWlnaHQpIHtcbiAgICBsZXQgc3JjID0gKHRleHR1cmVIZWlnaHQgLSAxKSAqIHRleHR1cmVXaWR0aCAqIDQ7XG4gICAgbGV0IGRzdCA9IChyb3dzIC0gMSkgKiBjb2x1bW5zO1xuICAgIGZvciAobGV0IGJsb2NrWCA9IDA7IGJsb2NrWCA8IHdpZHRoSW5GdWxsQmxvY2tzOyArK2Jsb2NrWCkge1xuICAgICAgbWF0cml4W2RzdCsrXSA9IHBhY2tlZFJHQkFbc3JjKytdO1xuICAgICAgbWF0cml4W2RzdCsrXSA9IHBhY2tlZFJHQkFbc3JjKytdO1xuICAgICAgc3JjICs9IDI7XG4gICAgfVxuICB9XG5cbiAgLy8gZmlsbCBpbiBib3R0b20tcmlnaHQgY2VsbFxuICBpZiAob2RkV2lkdGggJiYgb2RkSGVpZ2h0KSB7XG4gICAgbWF0cml4W21hdHJpeC5sZW5ndGggLSAxXSA9IHBhY2tlZFJHQkFbcGFja2VkUkdCQS5sZW5ndGggLSA0XTtcbiAgfVxuXG4gIHJldHVybiBtYXRyaXg7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmltcG9ydCB7R1BHUFVDb250ZXh0fSBmcm9tICcuL2dwZ3B1X2NvbnRleHQnO1xuXG5leHBvcnQgY2xhc3MgVGV4dHVyZU1hbmFnZXIge1xuICBwcml2YXRlIG51bVVzZWRUZXh0dXJlcyA9IDA7XG4gIHByaXZhdGUgbnVtRnJlZVRleHR1cmVzID0gMDtcbiAgcHJpdmF0ZSBmcmVlVGV4dHVyZXM6IHtbc2hhcGU6IHN0cmluZ106IFdlYkdMVGV4dHVyZVtdfSA9IHt9O1xuICBwcml2YXRlIGxvZ0VuYWJsZWQgPSBmYWxzZTtcbiAgcHJpdmF0ZSB1c2VkVGV4dHVyZUNvdW50OiB7W3NoYXBlOiBzdHJpbmddOiBudW1iZXJ9ID0ge307XG5cbiAgY29uc3RydWN0b3IocHJpdmF0ZSBncGdwdTogR1BHUFVDb250ZXh0KSB7fVxuXG4gIGFjcXVpcmVUZXh0dXJlKHNoYXBlUkM6IFtudW1iZXIsIG51bWJlcl0pOiBXZWJHTFRleHR1cmUge1xuICAgIGNvbnN0IHNoYXBlS2V5ID0gZ2V0S2V5RnJvbVRleHR1cmVTaGFwZShzaGFwZVJDKTtcbiAgICBpZiAoIShzaGFwZUtleSBpbiB0aGlzLmZyZWVUZXh0dXJlcykpIHtcbiAgICAgIHRoaXMuZnJlZVRleHR1cmVzW3NoYXBlS2V5XSA9IFtdO1xuICAgIH1cbiAgICBpZiAoIShzaGFwZUtleSBpbiB0aGlzLnVzZWRUZXh0dXJlQ291bnQpKSB7XG4gICAgICB0aGlzLnVzZWRUZXh0dXJlQ291bnRbc2hhcGVLZXldID0gMDtcbiAgICB9XG4gICAgdGhpcy51c2VkVGV4dHVyZUNvdW50W3NoYXBlS2V5XSsrO1xuXG4gICAgaWYgKHRoaXMuZnJlZVRleHR1cmVzW3NoYXBlS2V5XS5sZW5ndGggPiAwKSB7XG4gICAgICB0aGlzLm51bUZyZWVUZXh0dXJlcy0tO1xuICAgICAgdGhpcy5udW1Vc2VkVGV4dHVyZXMrKztcbiAgICAgIHRoaXMubG9nKCk7XG4gICAgICByZXR1cm4gdGhpcy5mcmVlVGV4dHVyZXNbc2hhcGVLZXldLnNoaWZ0KCkhO1xuICAgIH1cbiAgICB0aGlzLm51bVVzZWRUZXh0dXJlcysrO1xuICAgIHRoaXMubG9nKCk7XG5cbiAgICByZXR1cm4gdGhpcy5ncGdwdS5jcmVhdGVNYXRyaXhUZXh0dXJlKHNoYXBlUkNbMF0sIHNoYXBlUkNbMV0pO1xuICB9XG5cbiAgcmVsZWFzZVRleHR1cmUodGV4dHVyZTogV2ViR0xUZXh0dXJlLCBzaGFwZTogW251bWJlciwgbnVtYmVyXSk6IHZvaWQge1xuICAgIGNvbnN0IHNoYXBlS2V5ID0gZ2V0S2V5RnJvbVRleHR1cmVTaGFwZShzaGFwZSk7XG4gICAgaWYgKCEoc2hhcGVLZXkgaW4gdGhpcy5mcmVlVGV4dHVyZXMpKSB7XG4gICAgICB0aGlzLmZyZWVUZXh0dXJlc1tzaGFwZUtleV0gPSBbXTtcbiAgICB9XG4gICAgdGhpcy5mcmVlVGV4dHVyZXNbc2hhcGVLZXldLnB1c2godGV4dHVyZSk7XG4gICAgdGhpcy5udW1GcmVlVGV4dHVyZXMrKztcbiAgICB0aGlzLm51bVVzZWRUZXh0dXJlcy0tO1xuICAgIHRoaXMudXNlZFRleHR1cmVDb3VudFtzaGFwZUtleV0tLTtcbiAgICB0aGlzLmxvZygpO1xuICB9XG5cbiAgcHJpdmF0ZSBsb2coKSB7XG4gICAgaWYgKCF0aGlzLmxvZ0VuYWJsZWQpIHtcbiAgICAgIHJldHVybjtcbiAgICB9XG4gICAgY29uc3QgdG90YWwgPSB0aGlzLm51bUZyZWVUZXh0dXJlcyArIHRoaXMubnVtVXNlZFRleHR1cmVzO1xuICAgIGNvbnNvbGUubG9nKFxuICAgICAgICAnRnJlZS9Vc2VkJywgdGhpcy5udW1GcmVlVGV4dHVyZXMgKyAnIC8gJyArIHRoaXMubnVtVXNlZFRleHR1cmVzLFxuICAgICAgICBgKCR7dG90YWx9KWApO1xuICB9XG5cbiAgZ2V0TnVtVXNlZFRleHR1cmVzKCk6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMubnVtVXNlZFRleHR1cmVzO1xuICB9XG5cbiAgZ2V0TnVtRnJlZVRleHR1cmVzKCk6IG51bWJlciB7XG4gICAgcmV0dXJuIHRoaXMubnVtRnJlZVRleHR1cmVzO1xuICB9XG5cbiAgZGlzcG9zZSgpIHtcbiAgICBmb3IgKGNvbnN0IHNoYXBlIGluIHRoaXMuZnJlZVRleHR1cmVzKSB7XG4gICAgICBpZiAodGhpcy5mcmVlVGV4dHVyZXMuaGFzT3duUHJvcGVydHkoc2hhcGUpKSB7XG4gICAgICAgIGZvciAobGV0IGkgPSAwOyBpIDwgdGhpcy5mcmVlVGV4dHVyZXNbc2hhcGVdLmxlbmd0aDsgaSsrKSB7XG4gICAgICAgICAgdGhpcy5ncGdwdS5kZWxldGVNYXRyaXhUZXh0dXJlKHRoaXMuZnJlZVRleHR1cmVzW3NoYXBlXVtpXSk7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cbn1cblxuZnVuY3Rpb24gZ2V0S2V5RnJvbVRleHR1cmVTaGFwZShzaGFwZVJvd3NDb2w6IFtudW1iZXIsIG51bWJlcl0pOiBzdHJpbmcge1xuICByZXR1cm4gc2hhcGVSb3dzQ29sWzBdICsgJ18nICsgc2hhcGVSb3dzQ29sWzFdO1xufVxuIiwiLyogQ29weXJpZ2h0IDIwMTcgR29vZ2xlIEluYy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cblxuTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbnlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbllvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuXG4gICAgaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG5cblVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbmRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbldJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxubGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG49PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT0qL1xuXG5sZXQgVVNFX1dFQkdMMl9XSEVOX0FWQUlMQUJMRSA9IHRydWU7XG5sZXQgV0VCR0wyX0VOQUJMRUQ6IGJvb2xlYW58dW5kZWZpbmVkID0gbnVsbCE7XG5sZXQgTUFYX1RFWFRVUkVfU0laRTogbnVtYmVyID0gbnVsbCE7XG5cbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vLi4vdXRpbCc7XG5cbmV4cG9ydCBpbnRlcmZhY2UgV2ViR0xDb250ZXh0QXR0cmlidXRlcyB7XG4gIGFscGhhPzogYm9vbGVhbjtcbiAgYW50aWFsaWFzPzogYm9vbGVhbjtcbiAgcHJlbXVsdGlwbGllZEFscGhhPzogYm9vbGVhbjtcbiAgcHJlc2VydmVEcmF3aW5nQnVmZmVyPzogYm9vbGVhbjtcbiAgZGVwdGg/OiBib29sZWFuO1xuICBzdGVuY2lsPzogYm9vbGVhbjtcbiAgZmFpbElmTWFqb3JQZXJmb3JtYW5jZUNhdmVhdD86IGJvb2xlYW47XG59XG5cbmV4cG9ydCBpbnRlcmZhY2UgV2ViR0xMb3NlQ29udGV4dEV4dGVuc2lvbiB7IGxvc2VDb250ZXh0KCk6IHZvaWQ7IH1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVdlYkdMUmVuZGVyaW5nQ29udGV4dChhdHRyaWJ1dGVzOiBXZWJHTENvbnRleHRBdHRyaWJ1dGVzKTpcbiAgICBXZWJHTFJlbmRlcmluZ0NvbnRleHQge1xuICBjb25zdCBjYW52YXMgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCdjYW52YXMnKTtcbiAgY2FudmFzLndpZHRoID0gMTtcbiAgY2FudmFzLmhlaWdodCA9IDE7XG4gIHJldHVybiBjcmVhdGVXZWJHTFJlbmRlcmluZ0NvbnRleHRGcm9tQ2FudmFzKGNhbnZhcywgYXR0cmlidXRlcyk7XG59XG5cbi8qKlxuICogRm9yY2UgdGhlIGxpYnJhcnkgdG8gcHJlZmVyIFdlYkdMIDEuMCBpbnN0ZWFkIG9mIFdlYkdMIDIuMCBldmVuIHdoZW4gV2ViR0xcbiAqIDIuMCBpcyBhdmFpbGFibGUuXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBwcmVmZXJXZWJHTDEoKSB7XG4gIFVTRV9XRUJHTDJfV0hFTl9BVkFJTEFCTEUgPSBmYWxzZTtcbiAgV0VCR0wyX0VOQUJMRUQgPSBudWxsO1xufVxuXG4vKipcbiAqIFByZWZlciBXZWJHTCAyLjAgdG8gV2ViR0wgMS4wLiBUaGlzIGlzIHRoZSBkZWZhdWx0IGNvbmZpZ3VyYXRpb24uXG4gKi9cbmV4cG9ydCBmdW5jdGlvbiBwcmVmZXJXZWJHTDIoKSB7XG4gIFVTRV9XRUJHTDJfV0hFTl9BVkFJTEFCTEUgPSB0cnVlO1xuICBXRUJHTDJfRU5BQkxFRCA9IG51bGw7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBpc1dlYkdMMkVuYWJsZWQoKSB7XG4gIGlmICghVVNFX1dFQkdMMl9XSEVOX0FWQUlMQUJMRSkge1xuICAgIHJldHVybiBmYWxzZTtcbiAgfVxuXG4gIGlmIChXRUJHTDJfRU5BQkxFRCA9PSBudWxsKSB7XG4gICAgY29uc3QgdGVtcENhbnZhcyA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2NhbnZhcycpO1xuICAgIGNvbnN0IGdsID0gdGVtcENhbnZhcy5nZXRDb250ZXh0KCd3ZWJnbDInKTtcbiAgICBpZiAoZ2wgIT0gbnVsbCkge1xuICAgICAgV0VCR0wyX0VOQUJMRUQgPSB0cnVlO1xuXG4gICAgICBjb25zdCBsb3NlQ29udGV4dEV4dGVuc2lvbiA9IGdldEV4dGVuc2lvbk9yVGhyb3coXG4gICAgICAgICAgZ2wgYXMgV2ViR0xSZW5kZXJpbmdDb250ZXh0LFxuICAgICAgICAgICdXRUJHTF9sb3NlX2NvbnRleHQnKSBhcyBXZWJHTExvc2VDb250ZXh0RXh0ZW5zaW9uO1xuICAgICAgbG9zZUNvbnRleHRFeHRlbnNpb24ubG9zZUNvbnRleHQoKTtcbiAgICB9IGVsc2Uge1xuICAgICAgV0VCR0wyX0VOQUJMRUQgPSBmYWxzZTtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIFdFQkdMMl9FTkFCTEVEO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlV2ViR0xSZW5kZXJpbmdDb250ZXh0RnJvbUNhbnZhcyhcbiAgICBjYW52YXM6IEhUTUxDYW52YXNFbGVtZW50LFxuICAgIGF0dHJpYnV0ZXM6IFdlYkdMQ29udGV4dEF0dHJpYnV0ZXMpOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQge1xuICBsZXQgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dDtcbiAgaWYgKGlzV2ViR0wyRW5hYmxlZCgpKSB7XG4gICAgZ2wgPSBjYW52YXMuZ2V0Q29udGV4dCgnd2ViZ2wyJywgYXR0cmlidXRlcykgYXMgV2ViR0xSZW5kZXJpbmdDb250ZXh0O1xuICB9IGVsc2Uge1xuICAgIGdsID1cbiAgICAgICAgKGNhbnZhcy5nZXRDb250ZXh0KCd3ZWJnbCcsIGF0dHJpYnV0ZXMpIHx8XG4gICAgICAgICBjYW52YXMuZ2V0Q29udGV4dChcbiAgICAgICAgICAgICAnZXhwZXJpbWVudGFsLXdlYmdsJywgYXR0cmlidXRlcykpIGFzIFdlYkdMUmVuZGVyaW5nQ29udGV4dDtcbiAgfVxuXG4gIGlmIChnbCA9PSBudWxsKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdUaGlzIGJyb3dzZXIgZG9lcyBub3Qgc3VwcG9ydCBXZWJHTC4nKTtcbiAgfVxuICByZXR1cm4gZ2w7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjYWxsQW5kQ2hlY2s8VD4oZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZnVuYzogKCkgPT4gVCk6IFQge1xuICBjb25zdCByZXR1cm5WYWx1ZSA9IGZ1bmMoKTtcbiAgY2hlY2tXZWJHTEVycm9yKGdsKTtcbiAgcmV0dXJuIHJldHVyblZhbHVlO1xufVxuXG5sZXQgd2ViR0xEZWJ1Z0Vycm9yQ2hlY2tpbmdFbmFibGVkID0gZmFsc2U7XG5cbmV4cG9ydCBmdW5jdGlvbiBlbmFibGVEZWJ1Z1dlYkdMRXJyb3JDaGVja2luZyhlbmFibGVkOiBib29sZWFuKSB7XG4gIHdlYkdMRGVidWdFcnJvckNoZWNraW5nRW5hYmxlZCA9IGVuYWJsZWQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjaGVja1dlYkdMRXJyb3IoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCkge1xuICBpZiAod2ViR0xEZWJ1Z0Vycm9yQ2hlY2tpbmdFbmFibGVkKSB7XG4gICAgY29uc3QgZXJyb3IgPSBnbC5nZXRFcnJvcigpO1xuICAgIGlmIChlcnJvciAhPT0gZ2wuTk9fRVJST1IpIHtcbiAgICAgIHRocm93IG5ldyBFcnJvcignV2ViR0wgRXJyb3I6ICcgKyBnZXRXZWJHTEVycm9yTWVzc2FnZShnbCwgZXJyb3IpKTtcbiAgICB9XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFdlYkdMRXJyb3JNZXNzYWdlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHN0YXR1czogbnVtYmVyKTogc3RyaW5nIHtcbiAgc3dpdGNoIChzdGF0dXMpIHtcbiAgICBjYXNlIGdsLk5PX0VSUk9SOlxuICAgICAgcmV0dXJuICdOT19FUlJPUic7XG4gICAgY2FzZSBnbC5JTlZBTElEX0VOVU06XG4gICAgICByZXR1cm4gJ0lOVkFMSURfRU5VTSc7XG4gICAgY2FzZSBnbC5JTlZBTElEX1ZBTFVFOlxuICAgICAgcmV0dXJuICdJTlZBTElEX1ZBTFVFJztcbiAgICBjYXNlIGdsLklOVkFMSURfT1BFUkFUSU9OOlxuICAgICAgcmV0dXJuICdJTlZBTElEX09QRVJBVElPTic7XG4gICAgY2FzZSBnbC5JTlZBTElEX0ZSQU1FQlVGRkVSX09QRVJBVElPTjpcbiAgICAgIHJldHVybiAnSU5WQUxJRF9GUkFNRUJVRkZFUl9PUEVSQVRJT04nO1xuICAgIGNhc2UgZ2wuT1VUX09GX01FTU9SWTpcbiAgICAgIHJldHVybiAnT1VUX09GX01FTU9SWSc7XG4gICAgY2FzZSBnbC5DT05URVhUX0xPU1RfV0VCR0w6XG4gICAgICByZXR1cm4gJ0NPTlRFWFRfTE9TVF9XRUJHTCc7XG4gICAgZGVmYXVsdDpcbiAgICAgIHJldHVybiAnVW5rbm93biBlcnJvciBjb2RlICcgKyBzdGF0dXM7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldEV4dGVuc2lvbk9yVGhyb3coXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZXh0ZW5zaW9uTmFtZTogc3RyaW5nKToge30ge1xuICByZXR1cm4gdGhyb3dJZk51bGw8e30+KFxuICAgICAgZ2wsICgpID0+IGdsLmdldEV4dGVuc2lvbihleHRlbnNpb25OYW1lKSxcbiAgICAgICdFeHRlbnNpb24gXCInICsgZXh0ZW5zaW9uTmFtZSArICdcIiBub3Qgc3VwcG9ydGVkIG9uIHRoaXMgYnJvd3Nlci4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVZlcnRleFNoYWRlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB2ZXJ0ZXhTaGFkZXJTb3VyY2U6IHN0cmluZyk6IFdlYkdMU2hhZGVyIHtcbiAgY29uc3QgdmVydGV4U2hhZGVyOiBXZWJHTFNoYWRlciA9IHRocm93SWZOdWxsPFdlYkdMU2hhZGVyPihcbiAgICAgIGdsLCAoKSA9PiBnbC5jcmVhdGVTaGFkZXIoZ2wuVkVSVEVYX1NIQURFUiksXG4gICAgICAnVW5hYmxlIHRvIGNyZWF0ZSB2ZXJ0ZXggV2ViR0xTaGFkZXIuJyk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuc2hhZGVyU291cmNlKHZlcnRleFNoYWRlciwgdmVydGV4U2hhZGVyU291cmNlKSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuY29tcGlsZVNoYWRlcih2ZXJ0ZXhTaGFkZXIpKTtcbiAgaWYgKGdsLmdldFNoYWRlclBhcmFtZXRlcih2ZXJ0ZXhTaGFkZXIsIGdsLkNPTVBJTEVfU1RBVFVTKSA9PT0gZmFsc2UpIHtcbiAgICBjb25zb2xlLmxvZyhnbC5nZXRTaGFkZXJJbmZvTG9nKHZlcnRleFNoYWRlcikpO1xuICAgIHRocm93IG5ldyBFcnJvcignRmFpbGVkIHRvIGNvbXBpbGUgdmVydGV4IHNoYWRlci4nKTtcbiAgfVxuICByZXR1cm4gdmVydGV4U2hhZGVyO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlRnJhZ21lbnRTaGFkZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZnJhZ21lbnRTaGFkZXJTb3VyY2U6IHN0cmluZyk6IFdlYkdMU2hhZGVyIHtcbiAgY29uc3QgZnJhZ21lbnRTaGFkZXI6IFdlYkdMU2hhZGVyID0gdGhyb3dJZk51bGw8V2ViR0xTaGFkZXI+KFxuICAgICAgZ2wsICgpID0+IGdsLmNyZWF0ZVNoYWRlcihnbC5GUkFHTUVOVF9TSEFERVIpLFxuICAgICAgJ1VuYWJsZSB0byBjcmVhdGUgZnJhZ21lbnQgV2ViR0xTaGFkZXIuJyk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuc2hhZGVyU291cmNlKGZyYWdtZW50U2hhZGVyLCBmcmFnbWVudFNoYWRlclNvdXJjZSkpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmNvbXBpbGVTaGFkZXIoZnJhZ21lbnRTaGFkZXIpKTtcbiAgaWYgKGdsLmdldFNoYWRlclBhcmFtZXRlcihmcmFnbWVudFNoYWRlciwgZ2wuQ09NUElMRV9TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGNvbnNvbGUubG9nKGdsLmdldFNoYWRlckluZm9Mb2coZnJhZ21lbnRTaGFkZXIpKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0ZhaWxlZCB0byBjb21waWxlIGZyYWdtZW50IHNoYWRlci4nKTtcbiAgfVxuICByZXR1cm4gZnJhZ21lbnRTaGFkZXI7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBjcmVhdGVQcm9ncmFtKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpOiBXZWJHTFByb2dyYW0ge1xuICByZXR1cm4gdGhyb3dJZk51bGw8V2ViR0xQcm9ncmFtPihcbiAgICAgIGdsLCAoKSA9PiBnbC5jcmVhdGVQcm9ncmFtKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMUHJvZ3JhbS4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGxpbmtQcm9ncmFtKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSkge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmxpbmtQcm9ncmFtKHByb2dyYW0pKTtcbiAgaWYgKGdsLmdldFByb2dyYW1QYXJhbWV0ZXIocHJvZ3JhbSwgZ2wuTElOS19TVEFUVVMpID09PSBmYWxzZSkge1xuICAgIGNvbnNvbGUubG9nKGdsLmdldFByb2dyYW1JbmZvTG9nKHByb2dyYW0pKTtcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ0ZhaWxlZCB0byBsaW5rIHZlcnRleCBhbmQgZnJhZ21lbnQgc2hhZGVycy4nKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gdmFsaWRhdGVQcm9ncmFtKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSkge1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLnZhbGlkYXRlUHJvZ3JhbShwcm9ncmFtKSk7XG4gIGlmIChnbC5nZXRQcm9ncmFtUGFyYW1ldGVyKHByb2dyYW0sIGdsLlZBTElEQVRFX1NUQVRVUykgPT09IGZhbHNlKSB7XG4gICAgY29uc29sZS5sb2coZ2wuZ2V0UHJvZ3JhbUluZm9Mb2cocHJvZ3JhbSkpO1xuICAgIHRocm93IG5ldyBFcnJvcignU2hhZGVyIHByb2dyYW0gdmFsaWRhdGlvbiBmYWlsZWQuJyk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNyZWF0ZVN0YXRpY1ZlcnRleEJ1ZmZlcihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCBkYXRhOiBGbG9hdDMyQXJyYXkpOiBXZWJHTEJ1ZmZlciB7XG4gIGNvbnN0IGJ1ZmZlcjogV2ViR0xCdWZmZXIgPSB0aHJvd0lmTnVsbDxXZWJHTEJ1ZmZlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlQnVmZmVyKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMQnVmZmVyJyk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5BUlJBWV9CVUZGRVIsIGJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJ1ZmZlckRhdGEoZ2wuQVJSQVlfQlVGRkVSLCBkYXRhLCBnbC5TVEFUSUNfRFJBVykpO1xuICByZXR1cm4gYnVmZmVyO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlU3RhdGljSW5kZXhCdWZmZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgZGF0YTogVWludDE2QXJyYXkpOiBXZWJHTEJ1ZmZlciB7XG4gIGNvbnN0IGJ1ZmZlcjogV2ViR0xCdWZmZXIgPSB0aHJvd0lmTnVsbDxXZWJHTEJ1ZmZlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlQnVmZmVyKCksICdVbmFibGUgdG8gY3JlYXRlIFdlYkdMQnVmZmVyJyk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEJ1ZmZlcihnbC5FTEVNRU5UX0FSUkFZX0JVRkZFUiwgYnVmZmVyKSk7XG4gIGNhbGxBbmRDaGVjayhcbiAgICAgIGdsLCAoKSA9PiBnbC5idWZmZXJEYXRhKGdsLkVMRU1FTlRfQVJSQVlfQlVGRkVSLCBkYXRhLCBnbC5TVEFUSUNfRFJBVykpO1xuICByZXR1cm4gYnVmZmVyO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gcXVlcnlNYXhUZXh0dXJlU2l6ZShnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogbnVtYmVyIHtcbiAgaWYgKE1BWF9URVhUVVJFX1NJWkUgIT0gbnVsbCkge1xuICAgIHJldHVybiBNQVhfVEVYVFVSRV9TSVpFO1xuICB9XG4gIE1BWF9URVhUVVJFX1NJWkUgPVxuICAgICAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbCEuZ2V0UGFyYW1ldGVyKGdsIS5NQVhfVEVYVFVSRV9TSVpFKSk7XG4gIHJldHVybiBNQVhfVEVYVFVSRV9TSVpFO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0Q2hhbm5lbHNQZXJUZXh0dXJlKCk6IG51bWJlciB7XG4gIGlmIChpc1dlYkdMMkVuYWJsZWQoKSkge1xuICAgIHJldHVybiAxO1xuICB9XG4gIHJldHVybiA0O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlVGV4dHVyZShnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KTogV2ViR0xUZXh0dXJlIHtcbiAgcmV0dXJuIHRocm93SWZOdWxsPFdlYkdMVGV4dHVyZT4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlVGV4dHVyZSgpLCAnVW5hYmxlIHRvIGNyZWF0ZSBXZWJHTFRleHR1cmUuJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiB2YWxpZGF0ZVRleHR1cmVTaXplKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyKSB7XG4gIGNvbnN0IG1heFRleHR1cmVTaXplOiBudW1iZXIgPSBxdWVyeU1heFRleHR1cmVTaXplKGdsKTtcbiAgaWYgKCh3aWR0aCA8PSAwKSB8fCAoaGVpZ2h0IDw9IDApKSB7XG4gICAgY29uc3QgcmVxdWVzdGVkID0gJ1snICsgd2lkdGggKyAneCcgKyBoZWlnaHQgKyAnXSc7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdSZXF1ZXN0ZWQgdGV4dHVyZSBzaXplICcgKyByZXF1ZXN0ZWQgKyAnIGlzIGludmFsaWQuJyk7XG4gIH1cbiAgaWYgKCh3aWR0aCA+IG1heFRleHR1cmVTaXplKSB8fCAoaGVpZ2h0ID4gbWF4VGV4dHVyZVNpemUpKSB7XG4gICAgY29uc3QgcmVxdWVzdGVkID0gJ1snICsgd2lkdGggKyAneCcgKyBoZWlnaHQgKyAnXSc7XG4gICAgY29uc3QgbWF4ID0gJ1snICsgbWF4VGV4dHVyZVNpemUgKyAneCcgKyBtYXhUZXh0dXJlU2l6ZSArICddJztcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdSZXF1ZXN0ZWQgdGV4dHVyZSBzaXplICcgKyByZXF1ZXN0ZWQgK1xuICAgICAgICAnIGdyZWF0ZXIgdGhhbiBXZWJHTCBtYXhpbXVtIG9uIHRoaXMgYnJvd3NlciAvIEdQVSAnICsgbWF4ICsgJy4nKTtcbiAgfVxufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlRnJhbWVidWZmZXIoZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCk6IFdlYkdMRnJhbWVidWZmZXIge1xuICByZXR1cm4gdGhyb3dJZk51bGw8V2ViR0xGcmFtZWJ1ZmZlcj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuY3JlYXRlRnJhbWVidWZmZXIoKSwgJ1VuYWJsZSB0byBjcmVhdGUgV2ViR0xGcmFtZWJ1ZmZlci4nKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRWZXJ0ZXhCdWZmZXJUb1Byb2dyYW1BdHRyaWJ1dGUoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcHJvZ3JhbTogV2ViR0xQcm9ncmFtLCBhdHRyaWJ1dGU6IHN0cmluZyxcbiAgICBidWZmZXI6IFdlYkdMQnVmZmVyLCBhcnJheUVudHJpZXNQZXJJdGVtOiBudW1iZXIsIGl0ZW1TdHJpZGVJbkJ5dGVzOiBudW1iZXIsXG4gICAgaXRlbU9mZnNldEluQnl0ZXM6IG51bWJlcikge1xuICBjb25zdCBsb2MgPSBnbC5nZXRBdHRyaWJMb2NhdGlvbihwcm9ncmFtLCBhdHRyaWJ1dGUpO1xuICBpZiAobG9jID09PSAtMSkge1xuICAgIGNvbnN0IGVycm9yID0gbmV3IEVycm9yKFxuICAgICAgICAnVW5hYmxlIHRvIGdldCBhdHRyaWJ1dGUgXCInICsgYXR0cmlidXRlICsgJ1wiIG9uIFdlYkdMUHJvZ3JhbS4nKTtcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgKGVycm9yIGFzIGFueSkubmFtZWRWZXJ0ZXhBdHRyaWJ1dGVOb3RGb3VuZCA9IGF0dHJpYnV0ZTtcbiAgICB0aHJvdyBlcnJvcjtcbiAgfVxuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmJpbmRCdWZmZXIoZ2wuQVJSQVlfQlVGRkVSLCBidWZmZXIpKTtcbiAgY2FsbEFuZENoZWNrKFxuICAgICAgZ2wsXG4gICAgICAoKSA9PiBnbC52ZXJ0ZXhBdHRyaWJQb2ludGVyKFxuICAgICAgICAgIGxvYywgYXJyYXlFbnRyaWVzUGVySXRlbSwgZ2wuRkxPQVQsIGZhbHNlLCBpdGVtU3RyaWRlSW5CeXRlcyxcbiAgICAgICAgICBpdGVtT2Zmc2V0SW5CeXRlcykpO1xuICBjYWxsQW5kQ2hlY2soZ2wsICgpID0+IGdsLmVuYWJsZVZlcnRleEF0dHJpYkFycmF5KGxvYykpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZFRleHR1cmVVbml0KFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHRleHR1cmU6IFdlYkdMVGV4dHVyZSwgdGV4dHVyZVVuaXQ6IG51bWJlcikge1xuICB2YWxpZGF0ZVRleHR1cmVVbml0KGdsLCB0ZXh0dXJlVW5pdCk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYWN0aXZlVGV4dHVyZShnbC5URVhUVVJFMCArIHRleHR1cmVVbml0KSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdW5iaW5kVGV4dHVyZVVuaXQoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZVVuaXQ6IG51bWJlcikge1xuICB2YWxpZGF0ZVRleHR1cmVVbml0KGdsLCB0ZXh0dXJlVW5pdCk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYWN0aXZlVGV4dHVyZShnbC5URVhUVVJFMCArIHRleHR1cmVVbml0KSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZFRleHR1cmUoZ2wuVEVYVFVSRV8yRCwgbnVsbCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZ2V0UHJvZ3JhbVVuaWZvcm1Mb2NhdGlvbk9yVGhyb3coXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgcHJvZ3JhbTogV2ViR0xQcm9ncmFtLFxuICAgIHVuaWZvcm1OYW1lOiBzdHJpbmcpOiBXZWJHTFVuaWZvcm1Mb2NhdGlvbiB7XG4gIHJldHVybiB0aHJvd0lmTnVsbDxXZWJHTFVuaWZvcm1Mb2NhdGlvbj4oXG4gICAgICBnbCwgKCkgPT4gZ2wuZ2V0VW5pZm9ybUxvY2F0aW9uKHByb2dyYW0sIHVuaWZvcm1OYW1lKSxcbiAgICAgICd1bmlmb3JtIFwiJyArIHVuaWZvcm1OYW1lICsgJ1wiIG5vdCBwcmVzZW50IGluIHByb2dyYW0uJyk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBiaW5kVGV4dHVyZVRvUHJvZ3JhbVVuaWZvcm1TYW1wbGVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHByb2dyYW06IFdlYkdMUHJvZ3JhbSwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgIHVuaWZvcm1TYW1wbGVyTmFtZTogc3RyaW5nLCB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gYmluZFRleHR1cmVVbml0KGdsLCB0ZXh0dXJlLCB0ZXh0dXJlVW5pdCkpO1xuICBjb25zdCBzYW1wbGVyTG9jYXRpb24gPVxuICAgICAgZ2V0UHJvZ3JhbVVuaWZvcm1Mb2NhdGlvbk9yVGhyb3coZ2wsIHByb2dyYW0sIHVuaWZvcm1TYW1wbGVyTmFtZSk7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wudW5pZm9ybTFpKHNhbXBsZXJMb2NhdGlvbiwgdGV4dHVyZVVuaXQpKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGJpbmRDYW52YXNUb0ZyYW1lYnVmZmVyKGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQpIHtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5iaW5kRnJhbWVidWZmZXIoZ2wuRlJBTUVCVUZGRVIsIG51bGwpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC52aWV3cG9ydCgwLCAwLCBnbC5jYW52YXMud2lkdGgsIGdsLmNhbnZhcy5oZWlnaHQpKTtcbiAgY2FsbEFuZENoZWNrKGdsLCAoKSA9PiBnbC5zY2lzc29yKDAsIDAsIGdsLmNhbnZhcy53aWR0aCwgZ2wuY2FudmFzLmhlaWdodCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYmluZENvbG9yVGV4dHVyZVRvRnJhbWVidWZmZXIoXG4gICAgZ2w6IFdlYkdMUmVuZGVyaW5nQ29udGV4dCwgdGV4dHVyZTogV2ViR0xUZXh0dXJlLFxuICAgIGZyYW1lYnVmZmVyOiBXZWJHTEZyYW1lYnVmZmVyKSB7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBmcmFtZWJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soXG4gICAgICBnbCxcbiAgICAgICgpID0+IGdsLmZyYW1lYnVmZmVyVGV4dHVyZTJEKFxuICAgICAgICAgIGdsLkZSQU1FQlVGRkVSLCBnbC5DT0xPUl9BVFRBQ0hNRU5UMCwgZ2wuVEVYVFVSRV8yRCwgdGV4dHVyZSwgMCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdW5iaW5kQ29sb3JUZXh0dXJlRnJvbUZyYW1lYnVmZmVyKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGZyYW1lYnVmZmVyOiBXZWJHTEZyYW1lYnVmZmVyKSB7XG4gIGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gZ2wuYmluZEZyYW1lYnVmZmVyKGdsLkZSQU1FQlVGRkVSLCBmcmFtZWJ1ZmZlcikpO1xuICBjYWxsQW5kQ2hlY2soXG4gICAgICBnbCxcbiAgICAgICgpID0+IGdsLmZyYW1lYnVmZmVyVGV4dHVyZTJEKFxuICAgICAgICAgIGdsLkZSQU1FQlVGRkVSLCBnbC5DT0xPUl9BVFRBQ0hNRU5UMCwgZ2wuVEVYVFVSRV8yRCwgbnVsbCwgMCkpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gdmFsaWRhdGVGcmFtZWJ1ZmZlcihnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0KSB7XG4gIGNvbnN0IHN0YXR1cyA9IGdsLmNoZWNrRnJhbWVidWZmZXJTdGF0dXMoZ2wuRlJBTUVCVUZGRVIpO1xuICBpZiAoc3RhdHVzICE9PSBnbC5GUkFNRUJVRkZFUl9DT01QTEVURSkge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgJ0Vycm9yIGJpbmRpbmcgZnJhbWVidWZmZXI6ICcgKyBnZXRGcmFtZWJ1ZmZlckVycm9yTWVzc2FnZShnbCwgc3RhdHVzKSk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldEZyYW1lYnVmZmVyRXJyb3JNZXNzYWdlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIHN0YXR1czogbnVtYmVyKTogc3RyaW5nIHtcbiAgc3dpdGNoIChzdGF0dXMpIHtcbiAgICBjYXNlIGdsLkZSQU1FQlVGRkVSX0lOQ09NUExFVEVfQVRUQUNITUVOVDpcbiAgICAgIHJldHVybiAnRlJBTUVCVUZGRVJfSU5DT01QTEVURV9BVFRBQ0hNRU5UJztcbiAgICBjYXNlIGdsLkZSQU1FQlVGRkVSX0lOQ09NUExFVEVfTUlTU0lOR19BVFRBQ0hNRU5UOlxuICAgICAgcmV0dXJuICdGUkFNRUJVRkZFUl9JTkNPTVBMRVRFX01JU1NJTkdfQVRUQUNITUVOVCc7XG4gICAgY2FzZSBnbC5GUkFNRUJVRkZFUl9JTkNPTVBMRVRFX0RJTUVOU0lPTlM6XG4gICAgICByZXR1cm4gJ0ZSQU1FQlVGRkVSX0lOQ09NUExFVEVfRElNRU5TSU9OUyc7XG4gICAgY2FzZSBnbC5GUkFNRUJVRkZFUl9VTlNVUFBPUlRFRDpcbiAgICAgIHJldHVybiAnRlJBTUVCVUZGRVJfVU5TVVBQT1JURUQnO1xuICAgIGRlZmF1bHQ6XG4gICAgICByZXR1cm4gJ3Vua25vd24gZXJyb3IgJyArIHN0YXR1cztcbiAgfVxufVxuXG5mdW5jdGlvbiB0aHJvd0lmTnVsbDxUPihcbiAgICBnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCByZXR1cm5UT3JOdWxsOiAoKSA9PiBUIHwgbnVsbCxcbiAgICBmYWlsdXJlTWVzc2FnZTogc3RyaW5nKTogVCB7XG4gIGNvbnN0IHRPck51bGw6IFR8bnVsbCA9IGNhbGxBbmRDaGVjayhnbCwgKCkgPT4gcmV0dXJuVE9yTnVsbCgpKTtcbiAgaWYgKHRPck51bGwgPT0gbnVsbCkge1xuICAgIHRocm93IG5ldyBFcnJvcihmYWlsdXJlTWVzc2FnZSk7XG4gIH1cbiAgcmV0dXJuIHRPck51bGwgYXMgVDtcbn1cblxuZnVuY3Rpb24gdmFsaWRhdGVUZXh0dXJlVW5pdChnbDogV2ViR0xSZW5kZXJpbmdDb250ZXh0LCB0ZXh0dXJlVW5pdDogbnVtYmVyKSB7XG4gIGNvbnN0IG1heFRleHR1cmVVbml0ID0gZ2wuTUFYX0NPTUJJTkVEX1RFWFRVUkVfSU1BR0VfVU5JVFMgLSAxO1xuICBjb25zdCBnbFRleHR1cmVVbml0ID0gdGV4dHVyZVVuaXQgKyBnbC5URVhUVVJFMDtcbiAgaWYgKGdsVGV4dHVyZVVuaXQgPCBnbC5URVhUVVJFMCB8fCBnbFRleHR1cmVVbml0ID4gbWF4VGV4dHVyZVVuaXQpIHtcbiAgICBjb25zdCB0ZXh0dXJlVW5pdFJhbmdlID0gJ1tnbC5URVhUVVJFMCwgZ2wuVEVYVFVSRScgKyBtYXhUZXh0dXJlVW5pdCArICddJztcbiAgICB0aHJvdyBuZXcgRXJyb3IoJ3RleHR1cmVVbml0IG11c3QgYmUgaW4gJyArIHRleHR1cmVVbml0UmFuZ2UgKyAnLicpO1xuICB9XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBnZXRUZXh0dXJlU2hhcGVGcm9tTG9naWNhbFNoYXBlKFxuICAgIGdsOiBXZWJHTFJlbmRlcmluZ0NvbnRleHQsIGxvZ1NoYXBlOiBudW1iZXJbXSxcbiAgICBwcmVmZXJyZWRUZXhTaGFwZT86IFtudW1iZXIsIG51bWJlcl0pOiBbbnVtYmVyLCBudW1iZXJdIHtcbiAgY29uc3QgbWF4VGV4U2l6ZSA9IHF1ZXJ5TWF4VGV4dHVyZVNpemUoZ2wpO1xuICBjb25zdCBzaXplID0gdXRpbC5zaXplRnJvbVNoYXBlKGxvZ1NoYXBlKTtcbiAgaWYgKHByZWZlcnJlZFRleFNoYXBlICE9IG51bGwpIHtcbiAgICBjb25zdCBzaXplUHJlZmVycmVkID0gdXRpbC5zaXplRnJvbVNoYXBlKHByZWZlcnJlZFRleFNoYXBlKTtcbiAgICB1dGlsLmFzc2VydChcbiAgICAgICAgc2l6ZSA9PT0gc2l6ZVByZWZlcnJlZCxcbiAgICAgICAgYFNpemUgb2Ygc2hhcGUgKCR7c2l6ZX0pIG11c3QgbWF0Y2ggc2l6ZSBvZiBgICtcbiAgICAgICAgICAgIGBwcmVmZXJyZWRTaGFwZSAoJHtzaXplUHJlZmVycmVkfSlgKTtcbiAgICBpZiAocHJlZmVycmVkVGV4U2hhcGVbMF0gPD0gbWF4VGV4U2l6ZSAmJlxuICAgICAgICBwcmVmZXJyZWRUZXhTaGFwZVsxXSA8PSBtYXhUZXhTaXplKSB7XG4gICAgICByZXR1cm4gcHJlZmVycmVkVGV4U2hhcGU7XG4gICAgfVxuICB9XG5cbiAgaWYgKGxvZ1NoYXBlLmxlbmd0aCA8PSAxICYmIHNpemUgPD0gbWF4VGV4U2l6ZSkge1xuICAgIHJldHVybiBbc2l6ZSwgMV07XG4gIH0gZWxzZSBpZiAoXG4gICAgICBsb2dTaGFwZS5sZW5ndGggPT09IDIgJiYgbG9nU2hhcGVbMF0gPD0gbWF4VGV4U2l6ZSAmJlxuICAgICAgbG9nU2hhcGVbMV0gPD0gbWF4VGV4U2l6ZSkge1xuICAgIHJldHVybiBsb2dTaGFwZSBhcyBbbnVtYmVyLCBudW1iZXJdO1xuICB9IGVsc2UgaWYgKFxuICAgICAgbG9nU2hhcGUubGVuZ3RoID09PSAzICYmIGxvZ1NoYXBlWzBdIDw9IG1heFRleFNpemUgJiZcbiAgICAgIGxvZ1NoYXBlWzFdICogbG9nU2hhcGVbMl0gPD0gbWF4VGV4U2l6ZSkge1xuICAgIHJldHVybiBbbG9nU2hhcGVbMF0sIGxvZ1NoYXBlWzFdICogbG9nU2hhcGVbMl1dO1xuICB9IGVsc2UgaWYgKFxuICAgICAgbG9nU2hhcGUubGVuZ3RoID09PSA0ICYmIGxvZ1NoYXBlWzBdIDw9IG1heFRleFNpemUgJiZcbiAgICAgIGxvZ1NoYXBlWzFdICogbG9nU2hhcGVbMl0gKiBsb2dTaGFwZVszXSA8PSBtYXhUZXhTaXplKSB7XG4gICAgcmV0dXJuIFtsb2dTaGFwZVswXSwgbG9nU2hhcGVbMV0gKiBsb2dTaGFwZVsyXSAqIGxvZ1NoYXBlWzNdXTtcbiAgfSBlbHNlIHtcbiAgICByZXR1cm4gdXRpbC5zaXplVG9TcXVhcmlzaFNoYXBlKHNpemUpO1xuICB9XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmV4cG9ydCBmdW5jdGlvbiBleHBlY3RBcnJheXNDbG9zZShcbiAgICBhY3R1YWw6IEZsb2F0MzJBcnJheSwgZXhwZWN0ZWQ6IEZsb2F0MzJBcnJheSwgZXBzaWxvbjogbnVtYmVyKSB7XG4gIGlmIChhY3R1YWwubGVuZ3RoICE9PSBleHBlY3RlZC5sZW5ndGgpIHtcbiAgICB0aHJvdyBuZXcgRXJyb3IoXG4gICAgICAgICdNYXRyaWNlcyBoYXZlIGRpZmZlcmVudCBsZW5ndGhzICgnICsgYWN0dWFsLmxlbmd0aCArICcgdnMgJyArXG4gICAgICAgIGV4cGVjdGVkLmxlbmd0aCArICcpLicpO1xuICB9XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgZXhwZWN0ZWQubGVuZ3RoOyArK2kpIHtcbiAgICBjb25zdCBhID0gYWN0dWFsW2ldO1xuICAgIGNvbnN0IGUgPSBleHBlY3RlZFtpXTtcbiAgICBpZiAoaXNOYU4oYSkgJiYgaXNOYU4oZSkpIHtcbiAgICAgIGNvbnRpbnVlO1xuICAgIH1cbiAgICBpZiAoaXNOYU4oYSkgfHwgaXNOYU4oZSkgfHwgTWF0aC5hYnMoYSAtIGUpID4gZXBzaWxvbikge1xuICAgICAgY29uc3QgYWN0dWFsU3RyID0gJ2FjdHVhbFsnICsgaSArICddID09PSAnICsgYTtcbiAgICAgIGNvbnN0IGV4cGVjdGVkU3RyID0gJ2V4cGVjdGVkWycgKyBpICsgJ10gPT09ICcgKyBlO1xuICAgICAgdGhyb3cgbmV3IEVycm9yKCdBcnJheXMgZGlmZmVyOiAnICsgYWN0dWFsU3RyICsgJywgJyArIGV4cGVjdGVkU3RyKTtcbiAgICB9XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHJhbmRvbUFycmF5SW5SYW5nZShcbiAgICBuOiBudW1iZXIsIG1pblZhbHVlOiBudW1iZXIsIG1heFZhbHVlOiBudW1iZXIpOiBGbG9hdDMyQXJyYXkge1xuICBjb25zdCB2ID0gbmV3IEZsb2F0MzJBcnJheShuKTtcbiAgY29uc3QgcmFuZ2UgPSBtYXhWYWx1ZSAtIG1pblZhbHVlO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IG47ICsraSkge1xuICAgIHZbaV0gPSAoTWF0aC5yYW5kb20oKSAqIHJhbmdlKSArIG1pblZhbHVlO1xuICB9XG4gIHJldHVybiB2O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gbWFrZUlkZW50aXR5KG46IG51bWJlcik6IEZsb2F0MzJBcnJheSB7XG4gIGNvbnN0IGkgPSBuZXcgRmxvYXQzMkFycmF5KG4gKiBuKTtcbiAgZm9yIChsZXQgaiA9IDA7IGogPCBuOyArK2opIHtcbiAgICBpWyhqICogbikgKyBqXSA9IDE7XG4gIH1cbiAgcmV0dXJuIGk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBzZXRWYWx1ZShcbiAgICBtOiBGbG9hdDMyQXJyYXksIG1OdW1Sb3dzOiBudW1iZXIsIG1OdW1Db2xzOiBudW1iZXIsIHY6IG51bWJlciwgcm93OiBudW1iZXIsXG4gICAgY29sdW1uOiBudW1iZXIpIHtcbiAgaWYgKHJvdyA+PSBtTnVtUm93cykge1xuICAgIHRocm93IG5ldyBFcnJvcigncm93ICgnICsgcm93ICsgJykgbXVzdCBiZSBpbiBbMCAnICsgbU51bVJvd3MgKyAnXS4nKTtcbiAgfVxuICBpZiAoY29sdW1uID49IG1OdW1Db2xzKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdjb2x1bW4gKCcgKyBjb2x1bW4gKyAnKSBtdXN0IGJlIGluIFswICcgKyBtTnVtQ29scyArICddLicpO1xuICB9XG4gIG1bKHJvdyAqIG1OdW1Db2xzKSArIGNvbHVtbl0gPSB2O1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3B1TXVsdGlwbHlNYXRyaXgoXG4gICAgYTogRmxvYXQzMkFycmF5LCBhUm93OiBudW1iZXIsIGFDb2w6IG51bWJlciwgYjogRmxvYXQzMkFycmF5LCBiUm93OiBudW1iZXIsXG4gICAgYkNvbDogbnVtYmVyKSB7XG4gIGNvbnN0IHJlc3VsdCA9IG5ldyBGbG9hdDMyQXJyYXkoYVJvdyAqIGJDb2wpO1xuICBmb3IgKGxldCByID0gMDsgciA8IGFSb3c7ICsrcikge1xuICAgIGZvciAobGV0IGMgPSAwOyBjIDwgYkNvbDsgKytjKSB7XG4gICAgICBsZXQgZCA9IDA7XG4gICAgICBmb3IgKGxldCBrID0gMDsgayA8IGFDb2w7ICsraykge1xuICAgICAgICBkICs9IGFbKHIgKiBhQ29sKSArIGtdICogYlsoayAqIGJDb2wpICsgY107XG4gICAgICB9XG4gICAgICByZXN1bHRbKHIgKiBiQ29sKSArIGNdID0gZDtcbiAgICB9XG4gIH1cbiAgcmV0dXJuIHJlc3VsdDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGNwdURvdFByb2R1Y3QoYTogRmxvYXQzMkFycmF5LCBiOiBGbG9hdDMyQXJyYXkpOiBudW1iZXIge1xuICBpZiAoYS5sZW5ndGggIT09IGIubGVuZ3RoKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKCdjcHVEb3RQcm9kdWN0OiBpbmNvbXBhdGlibGUgdmVjdG9ycy4nKTtcbiAgfVxuICBsZXQgZCA9IDA7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgYS5sZW5ndGg7ICsraSkge1xuICAgIGQgKz0gYVtpXSAqIGJbaV07XG4gIH1cbiAgcmV0dXJuIGQ7XG59XG4iLCIvKiBDb3B5cmlnaHQgMjAxNyBHb29nbGUgSW5jLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuXG5MaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xueW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG5cbiAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcblxuVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG5TZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG5saW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbj09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PSovXG5cbmV4cG9ydCB0eXBlIFZlY3RvciA9IG51bWJlcltdIHwgRmxvYXQ2NEFycmF5IHwgRmxvYXQzMkFycmF5IHwgSW50MzJBcnJheSB8XG4gICAgSW50OEFycmF5IHwgSW50MTZBcnJheTtcblxuLyoqIFNodWZmbGVzIHRoZSBhcnJheSB1c2luZyBGaXNoZXItWWF0ZXMgYWxnb3JpdGhtLiAqL1xuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuZXhwb3J0IGZ1bmN0aW9uIHNodWZmbGUoYXJyYXk6IGFueVtdfFVpbnQzMkFycmF5fEludDMyQXJyYXl8XG4gICAgICAgICAgICAgICAgICAgICAgICBGbG9hdDMyQXJyYXkpOiB2b2lkIHtcbiAgbGV0IGNvdW50ZXIgPSBhcnJheS5sZW5ndGg7XG4gIGxldCB0ZW1wID0gMDtcbiAgbGV0IGluZGV4ID0gMDtcbiAgLy8gV2hpbGUgdGhlcmUgYXJlIGVsZW1lbnRzIGluIHRoZSBhcnJheVxuICB3aGlsZSAoY291bnRlciA+IDApIHtcbiAgICAvLyBQaWNrIGEgcmFuZG9tIGluZGV4XG4gICAgaW5kZXggPSAoTWF0aC5yYW5kb20oKSAqIGNvdW50ZXIpIHwgMDtcbiAgICAvLyBEZWNyZWFzZSBjb3VudGVyIGJ5IDFcbiAgICBjb3VudGVyLS07XG4gICAgLy8gQW5kIHN3YXAgdGhlIGxhc3QgZWxlbWVudCB3aXRoIGl0XG4gICAgdGVtcCA9IGFycmF5W2NvdW50ZXJdO1xuICAgIGFycmF5W2NvdW50ZXJdID0gYXJyYXlbaW5kZXhdO1xuICAgIGFycmF5W2luZGV4XSA9IHRlbXA7XG4gIH1cbn1cblxuLyoqIENsYW1wcyBhIHZhbHVlIHRvIGEgc3BlY2lmaWVkIHJhbmdlLiAqL1xuZXhwb3J0IGZ1bmN0aW9uIGNsYW1wKG1pbjogbnVtYmVyLCB4OiBudW1iZXIsIG1heDogbnVtYmVyKTogbnVtYmVyIHtcbiAgcmV0dXJuIE1hdGgubWF4KG1pbiwgTWF0aC5taW4oeCwgbWF4KSk7XG59XG5cbi8qKiBSZXR1cm5zIGEgc2FtcGxlIGZyb20gYSB1bmlmb3JtIFthLCBiXSBkaXN0cmlidXRpb24uICovXG5leHBvcnQgZnVuY3Rpb24gcmFuZFVuaWZvcm0oYTogbnVtYmVyLCBiOiBudW1iZXIpIHtcbiAgcmV0dXJuIE1hdGgucmFuZG9tKCkgKiAoYiAtIGEpICsgYTtcbn1cblxuLyoqXG4gKiBTYW1wbGVzIGZyb20gYSBnYXVzc2lhbiBkaXN0cmlidXRpb24uXG4gKlxuICogQHBhcmFtIG1lYW4gVGhlIG1lYW4uIERlZmF1bHQgaXMgMC5cbiAqIEBwYXJhbSBzdGREZXYgVGhlIHN0YW5kYXJkIGRldmlhdGlvbi4gRGVmYXVsdCBpcyAxLlxuICovXG5leHBvcnQgZnVuY3Rpb24gcmFuZEdhdXNzKG1lYW4gPSAwLCBzdGREZXYgPSAxLCB0cnVuY2F0ZWQgPSBmYWxzZSk6IG51bWJlciB7XG4gIGxldCB2MTogbnVtYmVyLCB2MjogbnVtYmVyLCBzOiBudW1iZXI7XG4gIGRvIHtcbiAgICB2MSA9IDIgKiBNYXRoLnJhbmRvbSgpIC0gMTtcbiAgICB2MiA9IDIgKiBNYXRoLnJhbmRvbSgpIC0gMTtcbiAgICBzID0gdjEgKiB2MSArIHYyICogdjI7XG4gIH0gd2hpbGUgKHMgPiAxKTtcblxuICBjb25zdCByZXN1bHQgPSBNYXRoLnNxcnQoLTIgKiBNYXRoLmxvZyhzKSAvIHMpICogdjE7XG4gIGlmICh0cnVuY2F0ZWQgJiYgcmVzdWx0ID4gMikge1xuICAgIHJldHVybiByYW5kR2F1c3MobWVhbiwgc3RkRGV2LCB0cnVlKTtcbiAgfVxuICByZXR1cm4gbWVhbiArIHN0ZERldiAqIHJlc3VsdDtcbn1cblxuLyoqIFJldHVybnMgc3F1YXJlZCBldWNsZWRpYW4gZGlzdGFuY2UgYmV0d2VlbiB0d28gdmVjdG9ycy4gKi9cbmV4cG9ydCBmdW5jdGlvbiBkaXN0U3F1YXJlZChhOiBWZWN0b3IsIGI6IFZlY3Rvcik6IG51bWJlciB7XG4gIGxldCByZXN1bHQgPSAwO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IGEubGVuZ3RoOyBpKyspIHtcbiAgICBjb25zdCBkaWZmID0gYVtpXSAtIGJbaV07XG4gICAgcmVzdWx0ICs9IGRpZmYgKiBkaWZmO1xuICB9XG4gIHJldHVybiByZXN1bHQ7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBhc3NlcnQoZXhwcjogYm9vbGVhbiwgbXNnOiBzdHJpbmcpIHtcbiAgaWYgKCFleHByKSB7XG4gICAgdGhyb3cgbmV3IEVycm9yKG1zZyk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGFzc2VydFNoYXBlc01hdGNoKFxuICAgIHNoYXBlQTogbnVtYmVyW10sIHNoYXBlQjogbnVtYmVyW10sIGVycm9yTWVzc2FnZVByZWZpeCA9ICcnKTogdm9pZCB7XG4gIGFzc2VydChcbiAgICAgIGFycmF5c0VxdWFsKHNoYXBlQSwgc2hhcGVCKSxcbiAgICAgIGVycm9yTWVzc2FnZVByZWZpeCArIGBTaGFwZXMgJHtzaGFwZUF9IGFuZCAke3NoYXBlQn0gbXVzdCBtYXRjaGApO1xufVxuXG4vLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG5leHBvcnQgZnVuY3Rpb24gZmxhdHRlbihhcnI6IGFueVtdLCByZXQ/OiBudW1iZXJbXSk6IG51bWJlcltdIHtcbiAgcmV0ID0gKHJldCA9PT0gdW5kZWZpbmVkID8gW10gOiByZXQpO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IGFyci5sZW5ndGg7ICsraSkge1xuICAgIGlmIChBcnJheS5pc0FycmF5KGFycltpXSkpIHtcbiAgICAgIGZsYXR0ZW4oYXJyW2ldLCByZXQpO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXQucHVzaChhcnJbaV0pO1xuICAgIH1cbiAgfVxuICByZXR1cm4gcmV0O1xufVxuXG5leHBvcnQgdHlwZSBBcnJheURhdGEgPSBudW1iZXJ8bnVtYmVyW118bnVtYmVyW11bXXxudW1iZXJbXVtdW118bnVtYmVyW11bXVtdW107XG5cbmV4cG9ydCBmdW5jdGlvbiBpbmZlclNoYXBlKGFycjogQXJyYXlEYXRhKTogbnVtYmVyW10ge1xuICBjb25zdCBzaGFwZTogbnVtYmVyW10gPSBbXTtcbiAgd2hpbGUgKGFyciBpbnN0YW5jZW9mIEFycmF5KSB7XG4gICAgc2hhcGUucHVzaChhcnIubGVuZ3RoKTtcbiAgICBhcnIgPSBhcnJbMF07XG4gIH1cbiAgcmV0dXJuIHNoYXBlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gc2l6ZUZyb21TaGFwZShzaGFwZTogbnVtYmVyW10pOiBudW1iZXIge1xuICBpZiAoc2hhcGUubGVuZ3RoID09PSAwKSB7XG4gICAgLy8gU2NhbGFyLlxuICAgIHJldHVybiAxO1xuICB9XG4gIGxldCBzaXplID0gc2hhcGVbMF07XG4gIGZvciAobGV0IGkgPSAxOyBpIDwgc2hhcGUubGVuZ3RoOyBpKyspIHtcbiAgICBzaXplICo9IHNoYXBlW2ldO1xuICB9XG4gIHJldHVybiBzaXplO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaXNTY2FsYXJTaGFwZShzaGFwZTogbnVtYmVyW10pOiBib29sZWFuIHtcbiAgcmV0dXJuIHNoYXBlLmxlbmd0aCA9PT0gMDtcbn1cblxuLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuZXhwb3J0IGZ1bmN0aW9uIGFycmF5c0VxdWFsKG4xOiBhbnlbXXxGbG9hdDMyQXJyYXksIG4yOiBhbnlbXXxGbG9hdDMyQXJyYXkpIHtcbiAgaWYgKG4xLmxlbmd0aCAhPT0gbjIubGVuZ3RoKSB7XG4gICAgcmV0dXJuIGZhbHNlO1xuICB9XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbjEubGVuZ3RoOyBpKyspIHtcbiAgICBpZiAobjFbaV0gIT09IG4yW2ldKSB7XG4gICAgICByZXR1cm4gZmFsc2U7XG4gICAgfVxuICB9XG4gIHJldHVybiB0cnVlO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gaXNJbnQoYTogbnVtYmVyKTogYm9vbGVhbiB7XG4gIHJldHVybiBhICUgMSA9PT0gMDtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHRhbmgoeDogbnVtYmVyKTogbnVtYmVyIHtcbiAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICBpZiAoKE1hdGggYXMgYW55KS50YW5oICE9IG51bGwpIHtcbiAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tYW55XG4gICAgcmV0dXJuIChNYXRoIGFzIGFueSkudGFuaCh4KTtcbiAgfVxuICBpZiAoeCA9PT0gSW5maW5pdHkpIHtcbiAgICByZXR1cm4gMTtcbiAgfSBlbHNlIGlmICh4ID09PSAtSW5maW5pdHkpIHtcbiAgICByZXR1cm4gLTE7XG4gIH0gZWxzZSB7XG4gICAgY29uc3QgZTJ4ID0gTWF0aC5leHAoMiAqIHgpO1xuICAgIHJldHVybiAoZTJ4IC0gMSkgLyAoZTJ4ICsgMSk7XG4gIH1cbn1cblxuZXhwb3J0IGZ1bmN0aW9uIHNpemVUb1NxdWFyaXNoU2hhcGUoc2l6ZTogbnVtYmVyKTogW251bWJlciwgbnVtYmVyXSB7XG4gIGZvciAobGV0IGEgPSBNYXRoLmZsb29yKE1hdGguc3FydChzaXplKSk7IGEgPiAxOyAtLWEpIHtcbiAgICBpZiAoc2l6ZSAlIGEgPT09IDApIHtcbiAgICAgIHJldHVybiBbYSwgc2l6ZSAvIGFdO1xuICAgIH1cbiAgfVxuICByZXR1cm4gWzEsIHNpemVdO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gY3JlYXRlU2h1ZmZsZWRJbmRpY2VzKG46IG51bWJlcik6IFVpbnQzMkFycmF5IHtcbiAgY29uc3Qgc2h1ZmZsZWRJbmRpY2VzID0gbmV3IFVpbnQzMkFycmF5KG4pO1xuICBmb3IgKGxldCBpID0gMDsgaSA8IG47ICsraSkge1xuICAgIHNodWZmbGVkSW5kaWNlc1tpXSA9IGk7XG4gIH1cbiAgc2h1ZmZsZShzaHVmZmxlZEluZGljZXMpO1xuICByZXR1cm4gc2h1ZmZsZWRJbmRpY2VzO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gYXNzZXJ0QW5kR2V0QnJvYWRjYXN0ZWRTaGFwZShcbiAgICBzaGFwZUE6IG51bWJlcltdLCBzaGFwZUI6IG51bWJlcltdKTogbnVtYmVyW10ge1xuICBjb25zdCByZXN1bHQ6IG51bWJlcltdID0gW107XG4gIGxldCBuZXh0QURpbU11c3RCZU9uZSA9IGZhbHNlO1xuICBsZXQgbmV4dEJEaW1NdXN0QmVPbmUgPSBmYWxzZTtcbiAgY29uc3QgZXJyTXNnID0gYE9wZXJhbmRzIGNvdWxkIG5vdCBiZSBicm9hZGNhc3QgdG9nZXRoZXIgd2l0aCBzaGFwZXMgYCArXG4gICAgICBgJHtzaGFwZUF9IGFuZCAke3NoYXBlQn0uIEN1cnJlbnRseSwgd2Ugb25seSBzdXBwb3J0IGEgYCArXG4gICAgICBgc3RyaWN0ZXIgdmVyc2lvbiBvZiBicm9hZGNhc3RpbmcgdGhhbiBudW1weS5gO1xuICBjb25zdCBsID0gTWF0aC5tYXgoc2hhcGVBLmxlbmd0aCwgc2hhcGVCLmxlbmd0aCk7XG5cbiAgc2hhcGVBID0gc2hhcGVBLnNsaWNlKCkucmV2ZXJzZSgpO1xuICBzaGFwZUIgPSBzaGFwZUIuc2xpY2UoKS5yZXZlcnNlKCk7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbDsgaSsrKSB7XG4gICAgY29uc3QgYSA9IHNoYXBlQVtpXSB8fCAxO1xuICAgIGNvbnN0IGIgPSBzaGFwZUJbaV0gfHwgMTtcbiAgICBpZiAoKGIgPiAxICYmIG5leHRCRGltTXVzdEJlT25lKSB8fCAoYSA+IDEgJiYgbmV4dEFEaW1NdXN0QmVPbmUpKSB7XG4gICAgICB0aHJvdyBFcnJvcihlcnJNc2cpO1xuICAgIH1cbiAgICBpZiAoYSA+IDEgJiYgYiA9PT0gMSkge1xuICAgICAgbmV4dEJEaW1NdXN0QmVPbmUgPSB0cnVlO1xuICAgIH1cbiAgICBpZiAoYiA+IDEgJiYgYSA9PT0gMSkge1xuICAgICAgbmV4dEFEaW1NdXN0QmVPbmUgPSB0cnVlO1xuICAgIH1cbiAgICBpZiAoYSA+IDEgJiYgYiA+IDEgJiYgYSAhPT0gYikge1xuICAgICAgdGhyb3cgRXJyb3IoZXJyTXNnKTtcbiAgICB9XG4gICAgcmVzdWx0LnB1c2goTWF0aC5tYXgoYSwgYikpO1xuICB9XG4gIHJldHVybiByZXN1bHQucmV2ZXJzZSgpO1xufVxuIl19
