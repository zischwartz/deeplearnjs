"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('input=3x3x1,d2=1,f=2,s=1,p=0', function (math) {
            var inputDepth = 1;
            var outputDepth = 1;
            var inputShape = [3, 3, inputDepth];
            var fSize = 2;
            var stride = 1;
            var pad = 0;
            var weightsShape = [fSize, fSize, inputDepth, outputDepth];
            var x = ndarray_1.Array3D.new(inputShape, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
            var dy = ndarray_1.Array3D.new([2, 2, 1], [3, 1, 2, 0]);
            var result = math.conv2dDerFilter(x, dy, weightsShape, stride, pad);
            var expected = new Float32Array([13, 19, 31, 37]);
            expect(result.shape).toEqual(weightsShape);
            test_util.expectArraysClose(result.getValues(), expected, 1e-1);
            x.dispose();
            dy.dispose();
        });
    };
    test_util.describeMathCPU('conv2dDerWeights', [tests]);
    test_util.describeMathGPU('conv2dDerWeights', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it(' dy=2x2x2', function (math) {
            var outputDepth = 2;
            var dyShape = [2, 2, outputDepth];
            var dy = ndarray_1.Array3D.new(dyShape, [1, 2, 3, 4, 5, 6, 7, 8]);
            var result = math.conv2dDerBias(dy);
            var expected = new Float32Array([16, 20]);
            expect(result.shape).toEqual([outputDepth]);
            test_util.expectArraysClose(result.getValues(), expected);
            dy.dispose();
        });
    };
    test_util.describeMathCPU('conv2dDerBias', [tests]);
    test_util.describeMathGPU('conv2dDerBias', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=conv2d_der_test.js.map