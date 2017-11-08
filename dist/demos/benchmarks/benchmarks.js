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
exports.MathBenchmarkPolymer = polymer_spec_1.PolymerElement({
    is: 'math-benchmark',
    properties: { benchmarks: Array, benchmarkRunGroupNames: Array }
});
function getDisplayParams(params) {
    if (params == null) {
        return '';
    }
    var kvParams = params;
    var out = [];
    var keys = Object.keys(kvParams);
    if (keys.length === 0) {
        return '';
    }
    for (var i = 0; i < keys.length; i++) {
        out.push(keys[i] + ': ' + kvParams[keys[i]]);
    }
    return '{' + out.join(', ') + '}';
}
var MathBenchmark = (function (_super) {
    __extends(MathBenchmark, _super);
    function MathBenchmark() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    MathBenchmark.prototype.ready = function () {
        var _this = this;
        var groups = math_benchmark_run_groups_1.getRunGroups();
        var benchmarkRunGroupNames = [];
        var benchmarks = [];
        this.stopMessages = [];
        for (var i = 0; i < groups.length; i++) {
            benchmarkRunGroupNames.push(groups[i].name + ': ' + getDisplayParams(groups[i].params));
            benchmarks.push(groups[i]);
            this.stopMessages.push(false);
        }
        this.benchmarkRunGroupNames = benchmarkRunGroupNames;
        this.benchmarks = benchmarks;
        setTimeout(function () {
            var runButtons = _this.querySelectorAll('.run-test');
            var stopButtons = _this.querySelectorAll('.run-stop');
            var _loop_1 = function (i) {
                runButtons[i].addEventListener('click', function () {
                    _this.runBenchmarkGroup(groups, i);
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
    MathBenchmark.prototype.runBenchmarkGroup = function (groups, benchmarkRunGroupIndex) {
        var benchmarkRunGroup = groups[benchmarkRunGroupIndex];
        var canvas = this.querySelectorAll('.run-plot')[benchmarkRunGroupIndex];
        canvas.width = 400;
        canvas.height = 300;
        var context = canvas.getContext('2d');
        var datasets = [];
        for (var i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
            benchmarkRunGroup.benchmarkRuns[i].clearChartData();
            var hue = Math.floor(360 * i / benchmarkRunGroup.benchmarkRuns.length);
            datasets.push({
                data: benchmarkRunGroup.benchmarkRuns[i].chartData,
                fill: false,
                label: benchmarkRunGroup.benchmarkRuns[i].name,
                borderColor: "hsl(" + hue + ", 100%, 40%)",
                backgroundColor: "hsl(" + hue + ", 100%, 70%)",
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
                                    return label + "ms";
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
        var runPromises = [];
        var rowValues = [step.toString()];
        for (var i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
            var benchmarkRun = benchmarkRunGroup.benchmarkRuns[i];
            var benchmarkTest = benchmarkRun.benchmarkTest;
            var size = benchmarkRunGroup.stepToSizeTransformation != null ?
                benchmarkRunGroup.stepToSizeTransformation(step) :
                step;
            runPromises.push(benchmarkTest.run(size, benchmarkRunGroup.selectedOption));
        }
        Promise.all(runPromises).then(function (results) {
            for (var i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
                var benchmarkRun = benchmarkRunGroup.benchmarkRuns[i];
                var size = benchmarkRunGroup.stepToSizeTransformation != null ?
                    benchmarkRunGroup.stepToSizeTransformation(step) :
                    step;
                var resultString = void 0;
                var logString = void 0;
                var time = 0;
                var success = true;
                try {
                    time = results[i];
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
                console.log(benchmarkRun.name + "[" + size + "]: " + logString);
            }
            runNumbersTable.appendChild(_this.buildRunNumbersRow(rowValues));
            step += benchmarkRunGroup.stepSize;
            setTimeout(function () { return _this.runBenchmarkSteps(chart, benchmarkRunGroup, benchmarkRunGroupIndex, step); }, 100);
        });
    };
    return MathBenchmark;
}(exports.MathBenchmarkPolymer));
exports.MathBenchmark = MathBenchmark;
document.registerElement(MathBenchmark.prototype.is, MathBenchmark);
//# sourceMappingURL=benchmarks.js.map