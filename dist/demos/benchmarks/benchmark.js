"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var BenchmarkRun = (function () {
    function BenchmarkRun(name, benchmarkTest) {
        this.name = name;
        this.benchmarkTest = benchmarkTest;
        this.chartData = [];
    }
    BenchmarkRun.prototype.clearChartData = function () {
        this.chartData = [];
    };
    return BenchmarkRun;
}());
exports.BenchmarkRun = BenchmarkRun;
var BenchmarkTest = (function () {
    function BenchmarkTest(params) {
        this.params = params;
    }
    return BenchmarkTest;
}());
exports.BenchmarkTest = BenchmarkTest;
//# sourceMappingURL=benchmark.js.map