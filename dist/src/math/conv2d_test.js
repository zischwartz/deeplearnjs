"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('input=2x2x1,d2=1,f=1,s=1,p=0', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 1;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array4D.new([fSize, fSize, inputDepth, outputDepth], [2]);
            var bias = ndarray_1.Array1D.new([-1]);
            var result = math.conv2d(x, w, bias, stride, pad);
            var expected = new Float32Array([1, 3, 5, 7]);
            test_util.expectArraysClose(result.getValues(), expected);
            x.dispose();
            w.dispose();
            bias.dispose();
        });
        it('input=2x2x1,d2=1,f=2,s=1,p=0', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
            var bias = ndarray_1.Array1D.new([-1]);
            var result = math.conv2d(x, w, bias, stride, pad);
            var expected = new Float32Array([19]);
            test_util.expectArraysClose(result.getValues(), expected);
            x.dispose();
            w.dispose();
            bias.dispose();
        });
        it('throws when x is not rank 3', function (math) {
            var inputDepth = 1;
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
            var w = ndarray_1.Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
            var bias = ndarray_1.Array1D.new([-1]);
            expect(function () { return math.conv2d(x, w, bias, stride, pad); }).toThrowError();
            x.dispose();
            w.dispose();
            bias.dispose();
        });
        it('throws when weights is not rank 4', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array3D.new([2, 2, 1], [3, 1, 5, 0]);
            var bias = ndarray_1.Array1D.new([-1]);
            expect(function () { return math.conv2d(x, w, bias, stride, pad); }).toThrowError();
            x.dispose();
            w.dispose();
            bias.dispose();
        });
        it('throws when biases is not rank 1', function (math) {
            var inputDepth = 1;
            var inputShape = [2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array4D.new([fSize, fSize, inputDepth, outputDepth], [3, 1, 5, 0]);
            var bias = ndarray_1.Array2D.new([2, 2], [2, 2, 2, 2]);
            expect(function () { return math.conv2d(x, w, bias, stride, pad); }).toThrowError();
            x.dispose();
            w.dispose();
            bias.dispose();
        });
        it('throws when x depth does not match weight depth', function (math) {
            var inputDepth = 1;
            var wrongInputDepth = 5;
            var inputShape = [2, 2, inputDepth];
            var outputDepth = 1;
            var fSize = 2;
            var pad = 0;
            var stride = 1;
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4]);
            var w = ndarray_1.Array4D.randNormal([fSize, fSize, wrongInputDepth, outputDepth]);
            var bias = ndarray_1.Array1D.new([-1]);
            expect(function () { return math.conv2d(x, w, bias, stride, pad); }).toThrowError();
            x.dispose();
            w.dispose();
            bias.dispose();
        });
    };
    test_util.describeMathCPU('conv2d', [tests]);
    test_util.describeMathGPU('conv2d', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=conv2d_test.js.map