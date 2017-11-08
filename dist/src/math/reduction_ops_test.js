"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('Array1D', function (math) {
            var a = ndarray_1.Array1D.new([3, -1, 0, 100, -7, 2]);
            test_util.expectNumbersClose(math.min(a).get(), -7);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([3, NaN, 2]);
            expect(math.min(a).get()).toEqual(NaN);
            a.dispose();
        });
        it('2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            test_util.expectNumbersClose(math.min(a).get(), -7);
        });
        it('2D axis=[0,1]', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            test_util.expectNumbersClose(math.min(a, [0, 1]).get(), -7);
        });
        it('2D, axis=0 throws error', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            expect(function () { return math.min(a, 0); }).toThrowError();
        });
        it('2D, axis=1 provided as a number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.min(a, 1);
            test_util.expectArraysClose(r.getValues(), new Float32Array([2, -7]));
        });
        it('2D, axis=[1]', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.min(a, [1]);
            test_util.expectArraysClose(r.getValues(), new Float32Array([2, -7]));
        });
    };
    test_util.describeMathCPU('min', [tests]);
    test_util.describeMathGPU('min', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('with one element dominating', function (math) {
            var a = ndarray_1.Array1D.new([3, -1, 0, 100, -7, 2]);
            var r = math.max(a);
            test_util.expectNumbersClose(r.get(), 100);
            a.dispose();
        });
        it('with all elements being the same', function (math) {
            var a = ndarray_1.Array1D.new([3, 3, 3]);
            var r = math.max(a);
            test_util.expectNumbersClose(r.get(), 3);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            expect(math.max(ndarray_1.Array1D.new([3, NaN, 2])).get()).toEqual(NaN);
        });
        it('2D', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            test_util.expectNumbersClose(math.max(a).get(), 100);
        });
        it('2D axis=[0,1]', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            test_util.expectNumbersClose(math.max(a, [0, 1]).get(), 100);
        });
        it('2D, axis=0 throws error', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            expect(function () { return math.max(a, 0); }).toThrowError();
        });
        it('2D, axis=1 provided as a number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.max(a, 1);
            test_util.expectArraysClose(r.getValues(), new Float32Array([5, 100]));
        });
        it('2D, axis=[1]', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.max(a, [1]);
            test_util.expectArraysClose(r.getValues(), new Float32Array([5, 100]));
        });
    };
    test_util.describeMathCPU('max', [tests]);
    test_util.describeMathGPU('max', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, 0, 3, 2]);
            var result = math.argMax(a);
            expect(result.dtype).toBe('int32');
            expect(result.get()).toBe(2);
            a.dispose();
        });
        it('one value', function (math) {
            var a = ndarray_1.Array1D.new([10]);
            var result = math.argMax(a);
            expect(result.dtype).toBe('int32');
            expect(result.get()).toBe(0);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([5, 0, 3, NaN, 3]);
            var res = math.argMax(a);
            expect(res.dtype).toBe('int32');
            test_util.assertIsNan(res.get(), res.dtype);
            a.dispose();
        });
        it('2D, no axis specified', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            expect(math.argMax(a).get()).toBe(3);
        });
        it('2D, axis=0 throws error', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            expect(function () { return math.argMax(a, 0); }).toThrowError();
        });
        it('2D, axis=1', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, 2]);
            var r = math.argMax(a, 1);
            expect(r.dtype).toBe('int32');
            expect(r.getValues()).toEqual(new Int32Array([2, 0]));
        });
    };
    test_util.describeMathCPU('argmax', [tests]);
    test_util.describeMathGPU('argmax', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Array1D', function (math) {
            var a = ndarray_1.Array1D.new([1, 0, 3, 2]);
            var result = math.argMin(a);
            expect(result.get()).toBe(1);
            a.dispose();
        });
        it('one value', function (math) {
            var a = ndarray_1.Array1D.new([10]);
            var result = math.argMin(a);
            expect(result.get()).toBe(0);
            a.dispose();
        });
        it('Arg min propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([5, 0, NaN, 7, 3]);
            var res = math.argMin(a);
            test_util.assertIsNan(res.get(), res.dtype);
            a.dispose();
        });
        it('2D, no axis specified', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            expect(math.argMin(a).get()).toBe(4);
        });
        it('2D, axis=0 throws error', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, -1, 0, 100, -7, 2]);
            expect(function () { return math.argMin(a, 0); }).toThrowError();
        });
        it('2D, axis=1', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [3, 2, 5, 100, -7, -8]);
            var r = math.argMin(a, 1);
            expect(r.getValues()).toEqual(new Int32Array([1, 2]));
        });
    };
    test_util.describeMathCPU('argmin', [tests]);
    test_util.describeMathGPU('argmin', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('equals', function (math) {
            var a = ndarray_1.Array1D.new([5, 0, 3, 7, 3]);
            var b = ndarray_1.Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
            var result = math.argMaxEquals(a, b);
            expect(result.get()).toBe(1);
        });
        it('not equals', function (math) {
            var a = ndarray_1.Array1D.new([5, 0, 3, 1, 3]);
            var b = ndarray_1.Array1D.new([-100.3, -20.0, -10.0, -5, 0]);
            var result = math.argMaxEquals(a, b);
            expect(result.get()).toBe(0);
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([0, 3, 1, 3]);
            var b = ndarray_1.Array1D.new([NaN, -20.0, -10.0, -5]);
            var result = math.argMaxEquals(a, b);
            test_util.assertIsNan(result.get(), result.dtype);
        });
        it('throws when given arrays of different shape', function (math) {
            var a = ndarray_1.Array1D.new([5, 0, 3, 7, 3, 10]);
            var b = ndarray_1.Array1D.new([-100.3, -20.0, -10.0, -5, -100]);
            expect(function () { return math.argMaxEquals(a, b); }).toThrowError();
        });
    };
    test_util.describeMathCPU('argMaxEquals', [tests]);
    test_util.describeMathGPU('argMaxEquals', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('0', function (math) {
            var a = ndarray_1.Scalar.new(0);
            var result = math.logSumExp(a);
            test_util.expectNumbersClose(result.get(), 0);
            a.dispose();
            result.dispose();
        });
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, -3]);
            var result = math.logSumExp(a);
            test_util.expectNumbersClose(result.get(), Math.log(Math.exp(1) + Math.exp(2) + Math.exp(-3)));
            a.dispose();
            result.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, NaN]);
            var result = math.logSumExp(a);
            expect(result.get()).toEqual(NaN);
            a.dispose();
        });
        it('axes=0 in 2D array throws error', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var f = function () { return math.logSumExp(a, [0]); };
            expect(f).toThrowError();
        });
        it('axes=1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.logSumExp(a, [1]);
            expect(res.shape).toEqual([3]);
            var expected = new Float32Array([
                Math.log(Math.exp(1) + Math.exp(2)),
                Math.log(Math.exp(3) + Math.exp(0)),
                Math.log(Math.exp(0) + Math.exp(1)),
            ]);
            test_util.expectArraysClose(res.getValues(), expected);
        });
        it('2D, axes=1 provided as a single digit', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
            var res = math.logSumExp(a, 1);
            expect(res.shape).toEqual([2]);
            var expected = new Float32Array([
                Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3)),
                Math.log(Math.exp(0) + Math.exp(0) + Math.exp(1))
            ]);
            test_util.expectArraysClose(res.getValues(), expected);
        });
        it('axes=0,1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.logSumExp(a, [0, 1]);
            expect(res.shape).toEqual([]);
            var expected = new Float32Array([Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3) + Math.exp(0) + Math.exp(0) +
                    Math.exp(1))]);
            test_util.expectArraysClose(res.getValues(), expected);
        });
    };
    test_util.describeMathCPU('logSumExp', [tests]);
    test_util.describeMathGPU('logSumExp', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var result = math.sum(a);
            test_util.expectNumbersClose(result.get(), 7);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, NaN, 0, 1]);
            expect(math.sum(a).get()).toEqual(NaN);
            a.dispose();
        });
        it('sum over dtype int32', function (math) {
            var a = ndarray_1.Array1D.new([1, 5, 7, 3], 'int32');
            var sum = math.sum(a);
            expect(sum.get()).toBe(16);
        });
        it('sum over dtype bool', function (math) {
            var a = ndarray_1.Array1D.new([true, false, false, true, true], 'bool');
            var sum = math.sum(a);
            expect(sum.get()).toBe(3);
        });
        it('sums all values in 2D array with keep dim', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, null, true);
            expect(res.shape).toEqual([1, 1]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([7]));
        });
        it('sums across axis=0 in 2D array throws error', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var f = function () { return math.sum(a, [0]); };
            expect(f).toThrowError();
        });
        it('sums across axis=1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, [1]);
            expect(res.shape).toEqual([3]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([3, 3, 1]));
        });
        it('2D, axis=1 provided as number', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, 1);
            expect(res.shape).toEqual([2]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([6, 1]));
        });
        it('sums across axis=0,1 in 2D array', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 0, 0, 1]);
            var res = math.sum(a, [0, 1]);
            expect(res.shape).toEqual([]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([7]));
        });
    };
    test_util.describeMathCPU('sum', [tests]);
    test_util.describeMathGPU('sum', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=reduction_ops_test.js.map