"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var math_1 = require("./math");
var ndarray_1 = require("./ndarray");
var webgl_util = require("./webgl/webgl_util");
var commonTests = function (it) {
    it('A x B', function (math) {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = ndarray_1.Array2D.new([3, 2], [0, 1, -3, 2, 2, 1]);
        var c = math.matMul(a, b);
        expect(c.shape).toEqual([2, 2]);
        test_util.expectArraysClose(c.getValues(), new Float32Array([0, 8, -3, 20]));
        a.dispose();
        b.dispose();
    });
    it('A x B^t', function (math) {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = ndarray_1.Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);
        var c = math.matMul(a, b, math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.TRANSPOSED);
        var expected = new Float32Array([7, 10, 16, 31]);
        test_util.expectArraysClose(c.getValues(), expected);
        a.dispose();
        b.dispose();
    });
    it('A^t x B', function (math) {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = ndarray_1.Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);
        var c = math.matMul(a, b, math_1.MatrixOrientation.TRANSPOSED, math_1.MatrixOrientation.REGULAR);
        var expected = new Float32Array([17, 12, 2, 22, 15, 4, 27, 18, 6]);
        test_util.expectArraysClose(c.getValues(), expected);
        a.dispose();
        b.dispose();
    });
    it('A^t x B^t', function (math) {
        var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
        var b = ndarray_1.Array2D.new([2, 3], [1, 0, 2, 4, 3, 0]);
        var c = math.matMul(a, b, math_1.MatrixOrientation.TRANSPOSED, math_1.MatrixOrientation.TRANSPOSED);
        var expected = new Float32Array([11, 13, 14, 20]);
        test_util.expectArraysClose(c.getValues(), expected);
        a.dispose();
        b.dispose();
    });
    it('A x B^t shapes do not match', function (math) {
        var a = ndarray_1.Array2D.zeros([2, 3]);
        var b = ndarray_1.Array2D.zeros([3, 2]);
        var f = function () {
            math.matMul(a, b, math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.TRANSPOSED);
        };
        expect(f).toThrowError();
        a.dispose();
        b.dispose();
    });
    it('A^t x B shapes do not match', function (math) {
        var a = ndarray_1.Array2D.zeros([2, 3]);
        var b = ndarray_1.Array2D.zeros([3, 2]);
        var f = function () {
            math.matMul(a, b, math_1.MatrixOrientation.TRANSPOSED, math_1.MatrixOrientation.REGULAR);
        };
        expect(f).toThrowError();
        a.dispose();
        b.dispose();
    });
    it('A^t x B^t shapes do not match', function (math) {
        var a = ndarray_1.Array2D.zeros([3, 2]);
        var b = ndarray_1.Array2D.zeros([3, 2]);
        var f = function () {
            math.matMul(a, b, math_1.MatrixOrientation.TRANSPOSED, math_1.MatrixOrientation.TRANSPOSED);
        };
        expect(f).toThrowError();
        a.dispose();
        b.dispose();
    });
    it('matmul throws when inner dimensions dont match', function (math) {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = ndarray_1.Array2D.new([4, 2], [0, 1, -3, 2, 2, 1, 2, 2]);
        expect(function () { return math.matMul(a, b); }).toThrowError();
        a.dispose();
        b.dispose();
    });
    it('matmul throws when passed non matrices', function (math) {
        var a = ndarray_1.Array3D.new([2, 3, 2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        var b = ndarray_1.Array2D.new([4, 2], [0, 1, -3, 2, 2, 1, 2, 2]);
        expect(function () { return math.matMul(a, b); }).toThrowError();
        expect(function () { return math.matMul(b, a); }).toThrowError();
        a.dispose();
        b.dispose();
    });
    it('Vector times matrix', function (math) {
        var v = ndarray_1.Array1D.new([2, 3]);
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var result = math.vectorTimesMatrix(v, matrix);
        var expected = new Float32Array([11, 16]);
        test_util.expectArraysClose(result.getValues(), expected);
        v.dispose();
        matrix.dispose();
        result.dispose();
    });
    it('Vector times matrix with implicit reshape', function (math) {
        var v = ndarray_1.Array1D.new([2, 3]);
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var result = math.vectorTimesMatrix(v, matrix);
        var expected = new Float32Array([11, 16]);
        test_util.expectArraysClose(result.getValues(), expected);
        v.dispose();
        matrix.dispose();
    });
    it('Vector times matrix throws when not passed a vector', function (math) {
        var v = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        expect(function () { return math.vectorTimesMatrix(v, matrix); }).toThrowError();
        v.dispose();
        matrix.dispose();
    });
    it('Vector times matrix throws when not passed a matrix', function (math) {
        var v = ndarray_1.Array1D.new([2, 3]);
        var matrix = ndarray_1.Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
        expect(function () { return math.vectorTimesMatrix(v, matrix); }).toThrowError();
        v.dispose();
        matrix.dispose();
    });
    it('Matrix times vector', function (math) {
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var v = ndarray_1.Array1D.new([2, 3]);
        var result = math.matrixTimesVector(matrix, v);
        var expected = new Float32Array([8, 18]);
        test_util.expectArraysClose(result.getValues(), expected);
        matrix.dispose();
        v.dispose();
    });
    it('Matrix * vector propagates NaNs', function (math) {
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var v = ndarray_1.Array1D.new([2, NaN]);
        var result = math.matrixTimesVector(matrix, v);
        var expected = new Float32Array([NaN, NaN]);
        test_util.expectArraysClose(result.getValues(), expected);
        matrix.dispose();
        v.dispose();
    });
    it('matrix times vector throws when not passed a vector', function (math) {
        var v = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        expect(function () { return math.matrixTimesVector(matrix, v); }).toThrowError();
        v.dispose();
        matrix.dispose();
    });
    it('matrix times vector throws when not passed a matrix', function (math) {
        var v = ndarray_1.Array1D.new([2, 3]);
        var matrix = ndarray_1.Array3D.new([2, 2, 2], [1, 2, 3, 4, 5, 6, 7, 8]);
        expect(function () { return math.matrixTimesVector(matrix, v); }).toThrowError();
        v.dispose();
    });
    it('Dot product', function (math) {
        var v1 = ndarray_1.Array1D.new([2, 3]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        var result = math.dotProduct(v1, v2);
        test_util.expectNumbersClose(result.get(), 7);
        v1.dispose();
        v2.dispose();
        result.dispose();
    });
    it('Dot product propagates NaNs', function (math) {
        var v1 = ndarray_1.Array1D.new([2, NaN]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        var result = math.dotProduct(v1, v2);
        expect(result.get()).toEqual(NaN);
        v1.dispose();
        v2.dispose();
    });
    it('Dot product throws when vectors are different size', function (math) {
        var v1 = ndarray_1.Array1D.new([2, 3, 3]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        expect(function () { return math.dotProduct(v1, v2); }).toThrowError();
        expect(function () { return math.dotProduct(v2, v1); }).toThrowError();
        v1.dispose();
        v2.dispose();
    });
    it('Dot product throws when passed non vectors', function (math) {
        var v1 = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 3]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        expect(function () { return math.dotProduct(v1, v2); }).toThrowError();
        expect(function () { return math.dotProduct(v2, v1); }).toThrowError();
        v1.dispose();
        v2.dispose();
    });
    it('Outer product', function (math) {
        var v1 = ndarray_1.Array1D.new([2, 3]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        var result = math.outerProduct(v1, v2);
        var expected = new Float32Array([4, 2, 6, 3]);
        expect(result.shape).toEqual([2, 2]);
        test_util.expectArraysClose(result.getValues(), expected);
        v1.dispose();
        v2.dispose();
    });
};
var gpuTests = function (it) {
    it('with implicit texture reshaping on GPU', function (math) {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        expect(a.getTextureShapeRC([6, 1])).toEqual([6, 1]);
        var b = ndarray_1.Array2D.new([3, 2], [1, 3, 0, 1, 2, 0]);
        expect(b.getTextureShapeRC()).toEqual([3, 2]);
        var result = math.matMul(a, b);
        expect(result.shape).toEqual([2, 2]);
        expect(result.getTextureShapeRC()).toEqual([2, 2]);
        test_util.expectArraysClose(result.getValues(), new Float32Array([7, 5, 16, 17]));
        a.dispose();
        b.dispose();
    });
    it('Matrix times vector, larger than max texture size', function (math) {
        var maxTexSize = webgl_util.queryMaxTextureSize(math.getGPGPUContext().gl);
        var sharedDim = maxTexSize + 4;
        var matrix = ndarray_1.Array2D.zeros([1, sharedDim]);
        matrix.set(1, 0, sharedDim - 3);
        matrix.set(1, 0, sharedDim - 2);
        var v = ndarray_1.Array1D.zeros([sharedDim]);
        v.set(1, sharedDim - 3);
        v.set(1, sharedDim - 2);
        var result = math.matrixTimesVector(matrix, v);
        var expected = new Float32Array([2]);
        test_util.expectArraysClose(result.getValues(), expected);
        matrix.dispose();
        v.dispose();
    });
    it('Matrix times vector with implicit reshape', function (math) {
        var matrix = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var v = ndarray_1.Array1D.new([2, 3]);
        expect(v.getTextureShapeRC([1, 2])).toEqual([1, 2]);
        var result = math.matrixTimesVector(matrix, v);
        var expected = new Float32Array([8, 18]);
        test_util.expectArraysClose(result.getValues(), expected);
        matrix.dispose();
        v.dispose();
    });
    it('Dot product with implicit reshaping', function (math) {
        var v1 = ndarray_1.Array1D.new([2, 3]);
        expect(v1.getTextureShapeRC([2, 1])).toEqual([2, 1]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        expect(v2.getTextureShapeRC([1, 2])).toEqual([1, 2]);
        var result = math.dotProduct(v1, v2);
        expect(result.get()).toBeCloseTo(7);
        v1.dispose();
        v2.dispose();
    });
    it('Outer product with implicit reshape', function (math) {
        var v1 = ndarray_1.Array1D.new([2, 3]);
        expect(v1.getTextureShapeRC([1, 2])).toEqual([1, 2]);
        var v2 = ndarray_1.Array1D.new([2, 1]);
        expect(v2.getTextureShapeRC([2, 1])).toEqual([2, 1]);
        var result = math.outerProduct(v1, v2);
        var expected = new Float32Array([4, 2, 6, 3]);
        expect(result.shape).toEqual([2, 2]);
        test_util.expectArraysClose(result.getValues(), expected);
        v1.dispose();
        v2.dispose();
    });
};
test_util.describeMathCPU('matMul', [commonTests]);
test_util.describeMathGPU('matMul', [commonTests, gpuTests], [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=matmul_test.js.map