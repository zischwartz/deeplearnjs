"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('x=3x3x1, f=2, s=1, no duplicate max value, test #1', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
            var dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
            var expected = new Float32Array([0, 0, 0, 0, 1, 2, 0, 3, 4]);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
        it('x=3x3x1, f=2, s=1, no duplicate max value, test #2', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([3, 3, 1], [9, 5, 6, 6, 8, 4, 9, 5, 10]);
            var dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
            var expected = new Float32Array([1, 0, 0, 0, 2, 0, 3, 0, 4]);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
        it('x=3x3x1, f=2, s=1 duplicate max value, test 1', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([3, 3, 1], [0, 0, 0, 0, 5, 0, 0, 0, 0]);
            var dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
            var expected = new Float32Array([0, 0, 0, 0, 10, 0, 0, 0, 0]);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
        it('x=3x3x1, f=2, s=1 duplicate max value, test 2', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([3, 3, 1], [1, 3, 2, 1, 2, 1, 1, 1, 5]);
            var dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
            var expected = new Float32Array([0, 3, 0, 0, 3, 0, 0, 0, 4]);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
        it('x=4x4x1, f=2, s=2, test #1', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            var dx = math.maxPoolBackprop(dy, x, 2, 2, 0);
            var expected = new Float32Array([0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 4]);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
        it('x=4x4x1, f=2, s=2, test #2', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([4, 4, 1], [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1]);
            var expected = new Float32Array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 0]);
            var dx = math.maxPoolBackprop(dy, x, 2, 2, 0);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
        it('x=5x5x1, f=3, s=2 no duplicate max value', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([5, 5, 1], [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
            ]);
            var dx = math.maxPoolBackprop(dy, x, 3, 2, 0);
            var expected = new Float32Array([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4
            ]);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
        it('x=5x5x1, f=3, s=2 duplicate max value', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var x = ndarray_1.Array3D.new([5, 5, 1], [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 24,
                13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 12
            ]);
            var dx = math.maxPoolBackprop(dy, x, 3, 2, 0);
            var expected = new Float32Array([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
        it('x=3x3x2, f=2, s=1, no duplicate max value', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 2], [1, 44, 2, 33, 3, 22, 4, 11]);
            var x = ndarray_1.Array3D.new([3, 3, 2], [1, 99, 2, 55, 3, 66, 4, 66, 5, 88, 6, 44, 7, 99, 8, 55, 9, 100]);
            var dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
            var expected = new Float32Array([0, 44, 0, 0, 0, 0, 0, 0, 1, 33, 2, 0, 0, 22, 3, 0, 4, 11]);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
        it('x=3x3x2, f=2, s=1, duplicate max value', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 2], [1, 44, 2, 33, 3, 22, 4, 11]);
            var x = ndarray_1.Array3D.new([3, 3, 2], [0, 1, 0, 3, 0, 2, 0, 1, 5, 2, 0, 1, 0, 1, 0, 1, 0, 5]);
            var dx = math.maxPoolBackprop(dy, x, 2, 1, 0);
            var expected = new Float32Array([0, 0, 0, 77, 0, 0, 0, 0, 10, 22, 0, 0, 0, 0, 0, 0, 0, 11]);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
        it('x=4x4x2, f=2, s=1', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
            var x = ndarray_1.Array3D.new([4, 4, 2], [
                0, 1, 1, 2, 2, 2, 3, 1, 4, 1, 5, 1, 6, 1, 7, 1,
                8, 1, 9, 1, 10, 1, 11, 1, 12, 1, 13, 2, 14, 2, 15, 1
            ]);
            var dx = math.maxPoolBackprop(dy, x, 2, 2, 0);
            var expected = new Float32Array([
                0, 0, 0, 11, 0, 22, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 33, 0, 44, 4, 0
            ]);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
        it('x=5x5x2, f=3, s=2 no duplicate max value', function (math) {
            var dy = ndarray_1.Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
            var x = ndarray_1.Array3D.new([5, 5, 2], [
                0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8,
                8, 9, 9, 10, 10, 11, 11, 12, 24, 13, 13, 14, 14, 15, 15, 16, 16,
                17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 12
            ]);
            var dx = math.maxPoolBackprop(dy, x, 3, 2, 0);
            var expected = new Float32Array([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 110, 0, 0, 2, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 0
            ]);
            test_util.expectArraysClose(dx.getValues(), expected);
            dy.dispose();
            x.dispose();
        });
    };
    test_util.describeMathCPU('maxPoolBackprop', [tests]);
    test_util.describeMathGPU('maxPoolBackprop', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=max_pool_backprop_test.js.map