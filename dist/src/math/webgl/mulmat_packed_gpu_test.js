"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../../test_util");
var math_1 = require("../math");
var gpgpu_context_1 = require("./gpgpu_context");
var mulmat_packed_gpu = require("./mulmat_packed_gpu");
describe('mulmat_packed_gpu (1x1 * 1x1)', function () {
    it('returns a 1x1 matrix', function () {
        var a = new Float32Array([0]);
        var b = new Float32Array([0]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, 1], b, [1, 1]);
        expect(result.length).toEqual(1);
    });
    it('returns [0] when multiplying [0] by [0]', function () {
        var a = new Float32Array([0]);
        var b = new Float32Array([0]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, 1], b, [1, 1]);
        expect(result[0]).toEqual(0);
    });
    it('returns [1] when multiplying [1] by [1]', function () {
        var a = new Float32Array([1]);
        var b = new Float32Array([1]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, 1], b, [1, 1]);
        expect(result[0]).toEqual(1);
    });
    it('returns [-1] when multiplying [1] by [-1]', function () {
        var a = new Float32Array([1]);
        var b = new Float32Array([-1]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, 1], b, [1, 1]);
        expect(result[0]).toEqual(-1);
    });
    it('returns [4.08] when multiplying [1.2] by [3.4]', function () {
        var a = new Float32Array([1.2]);
        var b = new Float32Array([3.4]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, 1], b, [1, 1]);
        expect(result[0]).toBeCloseTo(4.08);
    });
    it('returns [356000] when multiplying [356] by [1000]', function () {
        var a = new Float32Array([356]);
        var b = new Float32Array([1000]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, 1], b, [1, 1]);
        expect(result[0]).toEqual(356000);
    });
    it('returns [-31415926] when multiplying [-3.1415926] by [10000000]', function () {
        var a = new Float32Array([-3.1415926]);
        var b = new Float32Array([10000000]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, 1], b, [1, 1]);
        expect(result[0]).toEqual(-31415926);
    });
});
describe('mulmat_packed_gpu (dot product)', function () {
    it('returns a 1x1 matrix', function () {
        var a = new Float32Array(5);
        var b = new Float32Array(5);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, a.length], b, [b.length, 1]);
        expect(result.length).toEqual(1);
    });
    it('returns zero when one vector is all zeroes', function () {
        var a = new Float32Array(5);
        var b = new Float32Array([1, 2, 3, 4, 5]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, a.length], b, [b.length, 1]);
        expect(result[0]).toEqual(0);
    });
    it('returns the sum of b when a is all ones', function () {
        var a = new Float32Array([1, 1, 1, 1, 1]);
        var b = new Float32Array([0, 1, 2, 3, 100]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, a.length], b, [b.length, 1]);
        expect(result[0]).toEqual(106);
    });
    it('computes the dot product of a and b', function () {
        var a = new Float32Array([10, 20, 30, 40, 50]);
        var b = new Float32Array([0.5, 1.1, 12.4, 32.5, -123.98]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, a.length], b, [b.length, 1]);
        var expected = test_util.cpuDotProduct(a, b);
        expect(result[0]).toBeCloseTo(expected);
    });
    it('computes a dot product on very large vectors', function () {
        var a = test_util.randomArrayInRange(2048, -1, 1);
        var b = test_util.randomArrayInRange(2048, -1, 1);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, a.length], b, [b.length, 1]);
        var expected = test_util.cpuDotProduct(a, b);
        expect(result[0]).toBeCloseTo(expected);
    });
});
function cpuMul2x2(a, b) {
    if (a.length !== 4 || b.length !== 4) {
        throw new Error('a and b must have 4 elements.');
    }
    var result = new Float32Array(4);
    result[0] = (a[0] * b[0]) + (a[1] * b[2]);
    result[1] = (a[0] * b[1]) + (a[1] * b[3]);
    result[2] = (a[2] * b[0]) + (a[3] * b[2]);
    result[3] = (a[2] * b[1]) + (a[3] * b[3]);
    return result;
}
describe('mulmat_packed_gpu (2x2 * 2x2)', function () {
    it('returns a 2x2 matrix', function () {
        var a = new Float32Array([0, 0, 0, 0]);
        var b = new Float32Array([0, 0, 0, 0]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 2], b, [2, 2]);
        expect(result.length).toEqual(4);
    });
    it('returns the identity when multiplying two identity matrices', function () {
        var a = test_util.makeIdentity(2);
        var b = test_util.makeIdentity(2);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 2], b, [2, 2]);
        expect(result).toEqual(cpuMul2x2(a, b));
    });
    it('returns [0] when A is [0]', function () {
        var a = new Float32Array([0, 0, 0, 0]);
        var b = new Float32Array([1, 2, 3, 4]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 2], b, [2, 2]);
        expect(result).toEqual(a);
    });
    it('returns [0] when B is [0]', function () {
        var a = new Float32Array([1, 2, 3, 4]);
        var b = new Float32Array(4);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 2], b, [2, 2]);
        expect(result).toEqual(b);
    });
    it('returns B when A is identity', function () {
        var a = test_util.makeIdentity(2);
        var b = new Float32Array([11, -22, 33.333, -44.44444]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 2], b, [2, 2]);
        expect(result).toEqual(b);
    });
    it('returns A when B is identity', function () {
        var a = new Float32Array([11, -22, 33.333, -44.44444]);
        var b = test_util.makeIdentity(2);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 2], b, [2, 2]);
        expect(result).toEqual(a);
    });
    it('returns the product of A and B when non-identity', function () {
        var a = new Float32Array([10000.02, -1.2, 3.14159, -2345.1234]);
        var b = new Float32Array([-23.45, 0.01234, 100, 2.5]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 2], b, [2, 2]);
        expect(result).toEqual(cpuMul2x2(a, b));
    });
});
describe('mulmat_packed_gpu (different shapes)', function () {
    it('returns a 4x1 when multiplying a 4x4 with a 4x1', function () {
        var a = new Float32Array(16);
        var b = new Float32Array(4);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [4, 4], b, [4, 1]);
        expect(result.length).toEqual(4);
    });
    it('returns B (4x1) when A (4x4) is I', function () {
        var a = test_util.makeIdentity(4);
        var b = new Float32Array([1, 2, 3, 4]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [4, 4], b, [4, 1]);
        expect(result).toEqual(b);
    });
    it('4x2 * 2x2', function () {
        var a = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
        var b = new Float32Array([9, 10, 11, 12]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [4, 2], b, [2, 2]);
        var expected = test_util.cpuMultiplyMatrix(a, 4, 2, b, 2, 2);
        test_util.expectArraysClose(result, expected);
    });
    it('multiplies a 4x1 by a non-identity 4x4', function () {
        var a = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        var b = new Float32Array([1, 2, 3, 4]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [4, 4], b, [4, 1]);
        expect(result).toEqual(test_util.cpuMultiplyMatrix(a, 4, 4, b, 4, 1));
    });
    it('returns a 2x3 when multiplying a 2x4 by a 4x3', function () {
        var a = new Float32Array(8);
        var b = new Float32Array(12);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 4], b, [4, 3]);
        expect(result.length).toEqual(6);
    });
    it('multiplies A (2x4) by B(4x3)', function () {
        var a = new Float32Array([0.1, 3.2, -4.5, 11.78, -0.234, -2.999, 7, 9]);
        var b = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 4], b, [4, 3]);
        var expected = test_util.cpuMultiplyMatrix(a, 2, 4, b, 4, 3);
        test_util.expectArraysClose(result, expected);
    });
});
describe('mulmat_packed_gpu (large matrices)', function () {
    it('returns 128x128 when multiplying 2 128x128s', function () {
        var a = new Float32Array(128 * 128);
        var b = new Float32Array(128 * 128);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [128, 128], b, [128, 128]);
        expect(result.length).toEqual(128 * 128);
    });
    it('multiplies 2 128x128s', function () {
        var a = test_util.randomArrayInRange(128 * 128, -1, 1);
        var b = test_util.randomArrayInRange(128 * 128, -1, 1);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [128, 128], b, [128, 128]);
        var expected = test_util.cpuMultiplyMatrix(a, 128, 128, b, 128, 128);
        test_util.expectArraysClose(result, expected);
    });
});
describe('mulmat_packed_gpu (multiple matrices)', function () {
    it('4x2 * 2x12 * 12x1 === 4x1', function () {
        var aData = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]);
        var bData = new Float32Array([
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ]);
        var cData = new Float32Array([
            -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0, -1.1, -1.2
        ]);
        var gpgpu = new gpgpu_context_1.GPGPUContext();
        var axbProgram = gpgpu.createProgram(mulmat_packed_gpu.getFragmentShaderSource(2, math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.REGULAR));
        var abxcProgram = gpgpu.createProgram(mulmat_packed_gpu.getFragmentShaderSource(12, math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.REGULAR));
        var a = gpgpu.createPackedMatrixTexture(4, 2);
        var b = gpgpu.createPackedMatrixTexture(2, 12);
        var ab = gpgpu.createPackedMatrixTexture(4, 12);
        var c = gpgpu.createPackedMatrixTexture(12, 1);
        var r = gpgpu.createPackedMatrixTexture(4, 1);
        gpgpu.uploadMatrixToPackedTexture(a, 4, 2, aData);
        gpgpu.uploadMatrixToPackedTexture(b, 2, 12, bData);
        gpgpu.uploadMatrixToPackedTexture(c, 12, 1, cData);
        mulmat_packed_gpu.multiplyMatrixPacked(gpgpu, axbProgram, a, b, ab, [4, 12]);
        mulmat_packed_gpu.multiplyMatrixPacked(gpgpu, abxcProgram, ab, c, r, [4, 1]);
        var result = gpgpu.downloadMatrixFromPackedTexture(r, 4, 1);
        var expected = test_util.cpuMultiplyMatrix(test_util.cpuMultiplyMatrix(aData, 4, 2, bData, 2, 12), 4, 12, cData, 12, 1);
        test_util.expectArraysClose(result, expected);
        gpgpu.deleteMatrixTexture(a);
        gpgpu.deleteMatrixTexture(b);
        gpgpu.deleteMatrixTexture(ab);
        gpgpu.deleteMatrixTexture(c);
        gpgpu.deleteMatrixTexture(r);
        gpgpu.deleteProgram(axbProgram);
        gpgpu.deleteProgram(abxcProgram);
        gpgpu.dispose();
    });
});
describe('mulmat_packed_gpu A * B^t', function () {
    it('1x1 * 1x1', function () {
        var a = new Float32Array([2]);
        var b = new Float32Array([3]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [1, 1], b, [1, 1], math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.TRANSPOSED);
        expect(result[0]).toEqual(6);
    });
    it('2x2 * 2x2', function () {
        var a = new Float32Array([1, 2, 3, 4]);
        var b = new Float32Array([5, 6, 7, 8]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 2], b, [2, 2], math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.TRANSPOSED);
        var bt = new Float32Array([b[0], b[2], b[1], b[3]]);
        var expected = test_util.cpuMultiplyMatrix(a, 2, 2, bt, 2, 2);
        test_util.expectArraysClose(result, expected);
    });
    it('2x4 * 4x2', function () {
        var a = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
        var b = new Float32Array([9, 10, 11, 12, 13, 14, 15, 16]);
        var result = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 4], b, [2, 4], math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.TRANSPOSED);
        var bt = new Float32Array([b[0], b[4], b[1], b[5], b[2], b[6], b[3], b[7]]);
        var expected = test_util.cpuMultiplyMatrix(a, 2, 4, bt, 4, 2);
        test_util.expectArraysClose(result, expected);
    });
});
describe('mulmat_packed_gpu (transposed versions)', function () {
    it('A * B^t', function () {
        var a = new Float32Array([1, 2, 3, 4, 5, 6]);
        var b = new Float32Array([1, 0, 2, 4, 3, 0]);
        var c = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 3], b, [2, 3], math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.TRANSPOSED);
        var expected = new Float32Array([7, 10, 16, 31]);
        expect(c).toEqual(expected);
    });
    it('A^t * B', function () {
        var a = new Float32Array([1, 2, 3, 4, 5, 6]);
        var b = new Float32Array([1, 0, 2, 4, 3, 0]);
        var c = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [2, 3], b, [2, 3], math_1.MatrixOrientation.TRANSPOSED, math_1.MatrixOrientation.REGULAR);
        var expected = new Float32Array([17, 12, 2, 22, 15, 4, 27, 18, 6]);
        expect(c).toEqual(expected);
    });
    it('A^t * B^t', function () {
        var a = new Float32Array([1, 2, 3, 4, 5, 6]);
        var b = new Float32Array([1, 0, 2, 4, 3, 0]);
        var c = mulmat_packed_gpu.uploadMultiplyMatrixPackedDownload(a, [3, 2], b, [2, 3], math_1.MatrixOrientation.TRANSPOSED, math_1.MatrixOrientation.TRANSPOSED);
        var expected = new Float32Array([11, 13, 14, 20]);
        expect(c).toEqual(expected);
    });
});
//# sourceMappingURL=mulmat_packed_gpu_test.js.map