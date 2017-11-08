"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var math_cpu_1 = require("./math_cpu");
var math_gpu_1 = require("./math_gpu");
var ndarray_1 = require("./ndarray");
function executeTests(mathFactory) {
    var math;
    beforeEach(function () {
        math = mathFactory();
        math.startScope();
    });
    afterEach(function () {
        math.endScope(null);
        math.dispose();
    });
    it('Depth 1 throws error', function () {
        var indices = ndarray_1.Array1D.new([0, 0, 0]);
        expect(function () { return math.oneHot(indices, 1); }).toThrowError();
    });
    it('Depth 2, diagonal', function () {
        var indices = ndarray_1.Array1D.new([0, 1]);
        var res = math.oneHot(indices, 2);
        var expected = new Float32Array([1, 0, 0, 1]);
        expect(res.shape).toEqual([2, 2]);
        test_util.expectArraysClose(res.getValues(), expected);
    });
    it('Depth 2, transposed diagonal', function () {
        var indices = ndarray_1.Array1D.new([1, 0]);
        var res = math.oneHot(indices, 2);
        var expected = new Float32Array([0, 1, 1, 0]);
        expect(res.shape).toEqual([2, 2]);
        test_util.expectArraysClose(res.getValues(), expected);
    });
    it('Depth 3, 4 events', function () {
        var indices = ndarray_1.Array1D.new([2, 1, 2, 0]);
        var res = math.oneHot(indices, 3);
        var expected = new Float32Array([0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0]);
        expect(res.shape).toEqual([4, 3]);
        test_util.expectArraysClose(res.getValues(), expected);
    });
    it('Depth 2 onValue=3, offValue=-2', function () {
        var indices = ndarray_1.Array1D.new([0, 1]);
        var res = math.oneHot(indices, 2, 3, -2);
        var expected = new Float32Array([3, -2, -2, 3]);
        expect(res.shape).toEqual([2, 2]);
        test_util.expectArraysClose(res.getValues(), expected);
    });
}
describe('mathCPU oneHot', function () {
    executeTests(function () { return new math_cpu_1.NDArrayMathCPU(); });
});
describe('mathGPU oneHot', function () {
    executeTests(function () { return new math_gpu_1.NDArrayMathGPU(); });
});
//# sourceMappingURL=onehot_test.js.map