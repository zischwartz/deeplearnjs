"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('1x1x1 in, 1x1 filter, 1 stride: [0] => [0]', function (math) {
            var a = ndarray_1.Array3D.new([1, 1, 1], [0]);
            var result = math.maxPool(a, 1, 1, 0);
            test_util.expectArraysClose(result.getValues(), new Float32Array([0]));
        });
        it('3x3x1 in, 2x2 filter, 1 stride', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
            var result = math.maxPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([5, 6, 9, 9]));
        });
        it('3x3x1 in, 2x2 filter, 1 stride, propagates NaNs', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 9]);
            var result = math.maxPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([5, 6, NaN, NaN]));
        });
        it('3x3x2 in, 2x2 filter, 1 stride', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 2], [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
            var result = math.maxPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 2]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([5, 99, 6, 88, 9, 66, 9, 55]));
        });
        it('4x4x1 in, 2x2 filter, 2 stride', function (math) {
            var a = ndarray_1.Array3D.new([4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            var result = math.maxPool(a, 2, 2, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([5, 7, 13, 15]));
        });
        it('2x2x1 in, 2x2 filter, 2 stride, pad=1', function (math) {
            var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var result = math.maxPool(a, 2, 2, 1);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 2, 3, 4]));
        });
        it('throws when x is not rank 3', function (math) {
            var a = ndarray_1.Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
            expect(function () { return math.maxPool(a, 2, 1, 0); }).toThrowError();
            a.dispose();
        });
    };
    test_util.describeMathCPU('maxPool', [tests]);
    test_util.describeMathGPU('maxPool', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('1x1x1 in, 1x1 filter, 1 stride: [0] => [0]', function (math) {
            var a = ndarray_1.Array3D.new([1, 1, 1], [0]);
            var result = math.minPool(a, 1, 1, 0);
            test_util.expectArraysClose(result.getValues(), new Float32Array([0]));
        });
        it('3x3x1 in, 2x2 filter, 1 stride', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
            var result = math.minPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 2, 4, 5]));
        });
        it('3x3x1 in, 2x2 filter, 1 stride, propagates NaNs', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 8]);
            var result = math.minPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 2, NaN, NaN]));
        });
        it('3x3x2 in, 2x2 filter, 1 stride', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 2], [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
            var result = math.minPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 2]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 55, 2, 44, 4, 22, 5, 11]));
        });
        it('4x4x1 in, 2x2 filter, 2 stride', function (math) {
            var a = ndarray_1.Array3D.new([4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            var result = math.minPool(a, 2, 2, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([0, 2, 8, 10]));
        });
        it('2x2x1 in, 2x2 filter, 2 stride, pad=1', function (math) {
            var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var result = math.minPool(a, 2, 2, 1);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 2, 3, 4]));
        });
    };
    test_util.describeMathCPU('minPool', [tests]);
    test_util.describeMathGPU('minPool', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('1x1x1 in, 1x1 filter, 1 stride: [0] => [0]', function (math) {
            var a = ndarray_1.Array3D.new([1, 1, 1], [0]);
            var result = math.avgPool(a, 1, 1, 0);
            test_util.expectArraysClose(result.getValues(), new Float32Array([0]));
        });
        it('3x3x1 in, 2x2 filter, 1 stride', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, 9, 8]);
            var result = math.avgPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([3, 4, 6.25, 7]));
        });
        it('3x3x1 in, 2x2 filter, 1 stride, propagates NaNs', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 1], [1, 2, 3, 4, 5, 6, 7, NaN, 8]);
            var result = math.avgPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([3, 4, NaN, NaN]));
        });
        it('3x3x2 in, 2x2 filter, 1 stride', function (math) {
            var a = ndarray_1.Array3D.new([3, 3, 2], [1, 99, 2, 88, 3, 77, 4, 66, 5, 55, 6, 44, 7, 33, 9, 22, 8, 11]);
            var result = math.avgPool(a, 2, 1, 0);
            expect(result.shape).toEqual([2, 2, 2]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([3, 77, 4, 66, 6.25, 44, 7, 33]));
        });
        it('4x4x1 in, 2x2 filter, 2 stride', function (math) {
            var a = ndarray_1.Array3D.new([4, 4, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
            var result = math.avgPool(a, 2, 2, 0);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([2.5, 4.5, 10.5, 12.5]));
        });
        it('2x2x1 in, 2x2 filter, 2 stride, pad=1', function (math) {
            var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
            var result = math.avgPool(a, 2, 2, 1);
            expect(result.shape).toEqual([2, 2, 1]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([0.25, 0.5, 0.75, 1]));
        });
    };
    test_util.describeMathCPU('avgPool', [tests]);
    test_util.describeMathGPU('avgPool', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=pool_test.js.map