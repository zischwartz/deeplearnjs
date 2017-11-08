"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
var tests = function (it) {
    it('regular test', function (math) {
        var y = math.softmax(ndarray_1.Array1D.new([2, 1, 3]));
        test_util.expectArraysClose(y.getValues(), new Float32Array([0.24472847, 0.09003057, 0.66524095]));
        test_util.expectNumbersClose(y.get(0) + y.get(1) + y.get(2), 1);
    });
    it('overflow', function (math) {
        var y = math.softmax(ndarray_1.Array1D.new([1000, 1000]));
        test_util.expectArraysClose(y.getValues(), new Float32Array([0.5, 0.5]));
    });
    it('underflow', function (math) {
        var y = math.softmax(ndarray_1.Array1D.new([-1000, -1000]));
        test_util.expectArraysClose(y.getValues(), new Float32Array([0.5, 0.5]));
    });
    it('Huge difference between probabilities', function (math) {
        var y = math.softmax(ndarray_1.Array1D.new([-1000, +1000]));
        test_util.expectArraysClose(y.getValues(), new Float32Array([0.0, 1]));
    });
    it('Propagates NaNs', function (math) {
        var a = ndarray_1.Array1D.new([2, 1, NaN]);
        var y = math.softmax(a);
        test_util.expectArraysClose(y.getValues(), new Float32Array([NaN, NaN, NaN]));
        a.dispose();
    });
    it('2D, dim=1', function (math) {
        var y = math.softmax(ndarray_1.Array2D.new([2, 3], [[2, 1, 3], [1, 3, 2]]), 1);
        var expected = [
            0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
        ];
        expect(y.rank).toBe(2);
        test_util.expectArraysClose(y.getValues(), new Float32Array(expected));
    });
    it('2D, implicit dim=1', function (math) {
        var y = math.softmax(ndarray_1.Array2D.new([2, 3], [[2, 1, 3], [1, 3, 2]]));
        var expected = [
            0.24472847, 0.09003057, 0.66524095, 0.09003057, 0.66524095, 0.24472847
        ];
        expect(y.rank).toBe(2);
        test_util.expectArraysClose(y.getValues(), new Float32Array(expected));
    });
    it('2D, dim=0 throws error', function (math) {
        var f = function () {
            math.softmax(ndarray_1.Array2D.new([2, 3], [[2, 1, 3], [1, 3, 2]]), 0);
        };
        expect(f).toThrowError();
    });
};
test_util.describeMathCPU('softmax', [tests]);
test_util.describeMathGPU('softmax', [tests], [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=softmax_test.js.map