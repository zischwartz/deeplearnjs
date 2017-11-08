"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var util = require("../util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('elementWiseMul same-shaped ndarrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7]);
            var expected = new Float32Array([5, 6, -12, 28]);
            var result = math.elementWiseMul(a, b);
            expect(result.shape).toEqual([2, 2]);
            test_util.expectArraysClose(result.getValues(), expected);
            a.dispose();
            b.dispose();
        });
        it('elementWiseMul propagates NaNs', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 3, 4, 0]);
            var b = ndarray_1.Array2D.new([2, 2], [NaN, 3, NaN, 3]);
            var result = math.elementWiseMul(a, b).getValues();
            test_util.expectArraysClose(result, new Float32Array([NaN, 9, NaN, 0]));
            a.dispose();
            b.dispose();
        });
        it('elementWiseMul throws when passed ndarrays of different shapes', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7]);
            expect(function () { return math.elementWiseMul(a, b); }).toThrowError();
            expect(function () { return math.elementWiseMul(b, a); }).toThrowError();
            a.dispose();
            b.dispose();
        });
        it('multiply same-shaped ndarrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7]);
            var expected = new Float32Array([5, 6, -12, 28]);
            var result = math.multiply(a, b);
            expect(result.shape).toEqual([2, 2]);
            test_util.expectArraysClose(result.getValues(), expected);
            a.dispose();
            b.dispose();
        });
        it('multiply broadcasting ndarrays', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, 2, -3, -4]);
            var b = ndarray_1.Scalar.new(2);
            var expected = new Float32Array([2, 4, -6, -8]);
            var result = math.multiply(a, b);
            expect(result.shape).toEqual([2, 2]);
            test_util.expectArraysClose(result.getValues(), expected);
            a.dispose();
            b.dispose();
        });
        it('divide', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var c = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 2, 5]);
            var r = math.divide(a, c);
            test_util.expectArraysClose(r.getValues(), new Float32Array([1, 1, 1, 1, 2.5, 6 / 5]));
            a.dispose();
            c.dispose();
        });
        it('divide propagates NaNs', function (math) {
            var a = ndarray_1.Array2D.new([2, 1], [1, 2]);
            var c = ndarray_1.Array2D.new([2, 1], [3, NaN]);
            var r = math.divide(a, c).getValues();
            test_util.expectArraysClose(r, new Float32Array([1 / 3, NaN]));
            a.dispose();
            c.dispose();
        });
        it('div throws when passed ndarrays of different shapes', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, -3, -4, 5, 6]);
            var b = ndarray_1.Array2D.new([2, 2], [5, 3, 4, -7]);
            expect(function () { return math.divide(a, b); }).toThrowError();
            expect(function () { return math.divide(b, a); }).toThrowError();
            a.dispose();
            b.dispose();
        });
        it('scalar divided by array', function (math) {
            var c = ndarray_1.Scalar.new(2);
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var r = math.scalarDividedByArray(c, a);
            test_util.expectArraysClose(r.getValues(), new Float32Array([2 / 1, 2 / 2, 2 / 3, 2 / 4, 2 / 5, 2 / 6]));
            a.dispose();
            c.dispose();
        });
        it('scalar divided by array propagates NaNs', function (math) {
            var c = ndarray_1.Scalar.new(NaN);
            var a = ndarray_1.Array2D.new([1, 3], [1, 2, 3]);
            var r = math.scalarDividedByArray(c, a).getValues();
            expect(r).toEqual(new Float32Array([NaN, NaN, NaN]));
            a.dispose();
            c.dispose();
        });
        it('scalar divided by array throws when passed non scalar', function (math) {
            var c = ndarray_1.Array1D.new([1, 2, 3]);
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            expect(function () { return math.scalarDividedByArray(c, a); }).toThrowError();
            a.dispose();
            c.dispose();
        });
        it('array divided by scalar', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var c = ndarray_1.Scalar.new(2);
            var r = math.arrayDividedByScalar(a, c);
            test_util.expectArraysClose(r.getValues(), new Float32Array([1 / 2, 2 / 2, 3 / 2, 4 / 2, 5 / 2, 6 / 2]));
            a.dispose();
            c.dispose();
        });
        it('array divided by scalar propagates NaNs', function (math) {
            var a = ndarray_1.Array2D.new([1, 3], [1, 2, NaN]);
            var c = ndarray_1.Scalar.new(2);
            var r = math.arrayDividedByScalar(a, c).getValues();
            test_util.expectArraysClose(r, new Float32Array([1 / 2, 2 / 2, NaN]));
            a.dispose();
            c.dispose();
        });
        it('array divided by scalar throws when passed non scalar', function (math) {
            var c = ndarray_1.Array1D.new([1, 2, 3]);
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            expect(function () { return math.arrayDividedByScalar(a, c); }).toThrowError();
            a.dispose();
            c.dispose();
        });
        it('scalar times ndarray', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [2, -5, 1, 1, 4, 0]);
            var c = ndarray_1.Scalar.new(2);
            var expected = new Float32Array([4, -10, 2, 2, 8, 0]);
            var result = math.scalarTimesArray(c, a);
            expect(result.shape).toEqual([3, 2]);
            test_util.expectArraysClose(result.getValues(), expected);
            a.dispose();
            c.dispose();
        });
        it('scalar times ndarray throws when passed non-scalar', function (math) {
            var a = ndarray_1.Array2D.new([3, 2], [2, -5, 1, 1, 4, 0]);
            var c = ndarray_1.Array1D.new([1, 2, 3, 4]);
            expect(function () { return math.scalarTimesArray(c, a); }).toThrowError();
            a.dispose();
            c.dispose();
        });
    };
    test_util.describeMathCPU('element-wise mul/div', [tests]);
    test_util.describeMathGPU('element-wise mul/div', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('c + A', function (math) {
            var c = ndarray_1.Scalar.new(5);
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var result = math.scalarPlusArray(c, a);
            test_util.expectArraysClose(result.getValues(), new Float32Array([6, 7, 8]));
            a.dispose();
            c.dispose();
        });
        it('c + A propagates NaNs', function (math) {
            var c = ndarray_1.Scalar.new(NaN);
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var res = math.scalarPlusArray(c, a).getValues();
            expect(res).toEqual(new Float32Array([NaN, NaN, NaN]));
            a.dispose();
            c.dispose();
        });
        it('c + A throws when passed non scalar', function (math) {
            var c = ndarray_1.Array1D.new([1, 2, 3]);
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            expect(function () { return math.scalarPlusArray(c, a); }).toThrowError();
            a.dispose();
            c.dispose();
        });
        it('c - A', function (math) {
            var c = ndarray_1.Scalar.new(5);
            var a = ndarray_1.Array1D.new([7, 2, 3]);
            var result = math.scalarMinusArray(c, a);
            test_util.expectArraysClose(result.getValues(), new Float32Array([-2, 3, 2]));
            a.dispose();
            c.dispose();
        });
        it('c - A throws when passed non scalar', function (math) {
            var c = ndarray_1.Array1D.new([1, 2, 3]);
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            expect(function () { return math.scalarMinusArray(c, a); }).toThrowError();
            a.dispose();
            c.dispose();
        });
        it('A - c', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, -3]);
            var c = ndarray_1.Scalar.new(5);
            var result = math.arrayMinusScalar(a, c);
            test_util.expectArraysClose(result.getValues(), new Float32Array([-4, -3, -8]));
            a.dispose();
            c.dispose();
            result.dispose();
        });
        it('A - c propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 3]);
            var c = ndarray_1.Scalar.new(5);
            var res = math.arrayMinusScalar(a, c).getValues();
            test_util.expectArraysClose(res, new Float32Array([-4, NaN, -2]));
            a.dispose();
            c.dispose();
        });
        it('A - c throws when passed non scalar', function (math) {
            var c = ndarray_1.Array1D.new([1, 2, 3]);
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            expect(function () { return math.arrayMinusScalar(a, c); }).toThrowError();
            a.dispose();
            c.dispose();
        });
        it('A - B', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, 1]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            var result = math.subtract(a, b);
            var expected = new Float32Array([-2, 3, 2]);
            test_util.expectArraysClose(result.getValues(), expected);
            a.dispose();
            b.dispose();
        });
        it('A - B propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, 1]);
            var b = ndarray_1.Array1D.new([4, NaN, -1]);
            var res = math.subtract(a, b).getValues();
            test_util.expectArraysClose(res, new Float32Array([-2, NaN, 2]));
            a.dispose();
            b.dispose();
        });
        it('A - B throws when passed ndarrays with different shape', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, 1, 5]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            expect(function () { return math.subtract(a, b); }).toThrowError();
            expect(function () { return math.subtract(b, a); }).toThrowError();
            a.dispose();
            b.dispose();
        });
        it('2D-scalar broadcast', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var b = ndarray_1.Scalar.new(2);
            var res = math.subtract(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([-1, 0, 1, 2, 3, 4]));
        });
        it('scalar-1D broadcast', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Array1D.new([1, 2, 3, 4, 5, 6]);
            var res = math.subtract(a, b);
            expect(res.shape).toEqual([6]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([1, 0, -1, -2, -3, -4]));
        });
        it('2D-2D broadcast each with 1 dim', function (math) {
            var a = ndarray_1.Array2D.new([1, 3], [1, 2, 5]);
            var b = ndarray_1.Array2D.new([2, 1], [7, 3]);
            var res = math.subtract(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([-6, -5, -2, -2, -1, 2]));
        });
        it('2D-2D broadcast inner dim of b', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 5, 4, 5, 6]);
            var b = ndarray_1.Array2D.new([2, 1], [7, 3]);
            var res = math.subtract(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([-6, -5, -2, 1, 2, 3]));
        });
        it('3D-scalar', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6]);
            var b = ndarray_1.Scalar.new(-1);
            var res = math.subtract(a, b);
            expect(res.shape).toEqual([2, 3, 1]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([2, 3, 4, 5, 6, 7]));
        });
        it('A + B', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, 1]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            var result = math.add(a, b);
            var expected = new Float32Array([6, 7, 0]);
            test_util.expectArraysClose(result.getValues(), expected);
            a.dispose();
            b.dispose();
        });
        it('A + B propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, NaN]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            var res = math.add(a, b).getValues();
            test_util.expectArraysClose(res, new Float32Array([6, 7, NaN]));
            a.dispose();
            b.dispose();
        });
        it('A + B throws when passed ndarrays with different shape', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, 1, 5]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            expect(function () { return math.add(a, b); }).toThrowError();
            expect(function () { return math.add(b, a); }).toThrowError();
            a.dispose();
            b.dispose();
        });
        it('2D+scalar broadcast', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var b = ndarray_1.Scalar.new(2);
            var res = math.add(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([3, 4, 5, 6, 7, 8]));
        });
        it('scalar+1D broadcast', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Array1D.new([1, 2, 3, 4, 5, 6]);
            var res = math.add(a, b);
            expect(res.shape).toEqual([6]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([3, 4, 5, 6, 7, 8]));
        });
        it('2D+2D broadcast each with 1 dim', function (math) {
            var a = ndarray_1.Array2D.new([1, 3], [1, 2, 5]);
            var b = ndarray_1.Array2D.new([2, 1], [7, 3]);
            var res = math.add(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([8, 9, 12, 4, 5, 8]));
        });
        it('2D+2D broadcast inner dim of b', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 5, 4, 5, 6]);
            var b = ndarray_1.Array2D.new([2, 1], [7, 3]);
            var res = math.add(a, b);
            expect(res.shape).toEqual([2, 3]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([8, 9, 12, 7, 8, 9]));
        });
        it('3D+scalar', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6]);
            var b = ndarray_1.Scalar.new(-1);
            var res = math.add(a, b);
            expect(res.shape).toEqual([2, 3, 1]);
            test_util.expectArraysClose(res.getValues(), new Float32Array([0, 1, 2, 3, 4, 5]));
        });
    };
    test_util.describeMathCPU('element-wise add/sub', [tests]);
    test_util.describeMathGPU('element-wise add/sub', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('Scaled ndarray add', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
            var b = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var c1 = ndarray_1.Scalar.new(3);
            var c2 = ndarray_1.Scalar.new(2);
            var result = math.scaledArrayAdd(c1, a, c2, b);
            expect(result.shape).toEqual([2, 3]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([8, 16, 24, 32, 40, 48]));
            var wrongSizeMat = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            expect(function () { return math.scaledArrayAdd(c1, wrongSizeMat, c2, b); })
                .toThrowError();
            a.dispose();
            b.dispose();
            c1.dispose();
            c2.dispose();
        });
        it('throws when passed non-scalars', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
            var b = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
            var c1 = ndarray_1.Array1D.randNormal([10]);
            var c2 = ndarray_1.Scalar.new(2);
            expect(function () { return math.scaledArrayAdd(c1, a, c2, b); }).toThrowError();
            expect(function () { return math.scaledArrayAdd(c2, a, c1, b); }).toThrowError();
            a.dispose();
            b.dispose();
            c1.dispose();
            c2.dispose();
        });
        it('throws when NDArrays are different shape', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [2, 4, 6, 8, 10, 12]);
            var b = ndarray_1.Array2D.new([2, 4], [1, 2, 3, 4, 5, 6, 7, 8]);
            var c1 = ndarray_1.Scalar.new(3);
            var c2 = ndarray_1.Scalar.new(2);
            expect(function () { return math.scaledArrayAdd(c1, a, c2, b); }).toThrowError();
            a.dispose();
            b.dispose();
            c1.dispose();
            c2.dispose();
        });
    };
    test_util.describeMathCPU('scaledArrayAdd', [tests]);
    test_util.describeMathGPU('scaledArrayAdd', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([2, 5, NaN]);
            var b = ndarray_1.Array1D.new([4, 5, -1]);
            var res = math.equal(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.getValues()).toEqual(new Uint8Array([0, 1, util.NAN_BOOL]));
            a.dispose();
            b.dispose();
        });
        it('strict version throws when x and y are different shape', function (math) {
            var a = ndarray_1.Array1D.new([2]);
            var b = ndarray_1.Array1D.new([4, 2, -1]);
            expect(function () { return math.equalStrict(a, b); }).toThrowError();
            expect(function () { return math.equalStrict(b, a); }).toThrowError();
            a.dispose();
            b.dispose();
        });
        it('2D and scalar broadcast', function (math) {
            var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 2, 5, 6]);
            var b = ndarray_1.Scalar.new(2);
            var res = math.equal(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([2, 3]);
            expect(res.getValues()).toEqual(new Uint8Array([0, 1, 0, 1, 0, 0]));
        });
        it('scalar and 1D broadcast', function (math) {
            var a = ndarray_1.Scalar.new(2);
            var b = ndarray_1.Array1D.new([1, 2, 3, 4, 5, 2]);
            var res = math.equal(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([6]);
            expect(res.getValues()).toEqual(new Uint8Array([0, 1, 0, 0, 0, 1]));
        });
        it('2D and 2D broadcast each with 1 dim', function (math) {
            var a = ndarray_1.Array2D.new([1, 3], [1, 2, 5]);
            var b = ndarray_1.Array2D.new([2, 1], [5, 1]);
            var res = math.equal(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([2, 3]);
            expect(res.getValues()).toEqual(new Uint8Array([0, 0, 1, 1, 0, 0]));
        });
        it('3D and scalar', function (math) {
            var a = ndarray_1.Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, -1]);
            var b = ndarray_1.Scalar.new(-1);
            var res = math.equal(a, b);
            expect(res.dtype).toBe('bool');
            expect(res.shape).toEqual([2, 3, 1]);
            expect(res.getValues()).toEqual(new Uint8Array([0, 0, 0, 0, 0, 1]));
        });
    };
    test_util.describeMathCPU('equal', [tests]);
    test_util.describeMathGPU('equal', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=element_wise_arithmetic_test.js.map