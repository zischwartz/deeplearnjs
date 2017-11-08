"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var epsilon_1 = 1e-1;
    var tests = function (it) {
        it('simple batchnorm, no offset or scale, 2x1x2', function (math) {
            var x = ndarray_1.Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
            var mean = ndarray_1.Array1D.new([1, 2]);
            var variance = ndarray_1.Array1D.new([2, 3]);
            var varianceEpsilon = .001;
            var result = math.batchNormalization3D(x, mean, variance, varianceEpsilon, undefined, undefined);
            test_util.expectArraysClose(result.getValues(), new Float32Array([
                (x.get(0, 0, 0) - mean.get(0)) * 1 /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
                (x.get(0, 0, 1) - mean.get(1)) * 1 /
                    Math.sqrt(variance.get(1) + varianceEpsilon),
                (x.get(1, 0, 0) - mean.get(0)) * 1 /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
                (x.get(1, 0, 1) - mean.get(1)) * 1 /
                    Math.sqrt(variance.get(1) + varianceEpsilon)
            ]), epsilon_1);
            x.dispose();
            mean.dispose();
            variance.dispose();
        });
        it('simple batchnorm, no offset, 2x1x2', function (math) {
            var x = ndarray_1.Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
            var mean = ndarray_1.Array1D.new([1, 2]);
            var variance = ndarray_1.Array1D.new([2, 3]);
            var scale = ndarray_1.Array1D.new([4, 5]);
            var varianceEpsilon = .001;
            var result = math.batchNormalization3D(x, mean, variance, varianceEpsilon, scale, undefined);
            test_util.expectArraysClose(result.getValues(), new Float32Array([
                (x.get(0, 0, 0) - mean.get(0)) * scale.get(0) /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
                (x.get(0, 0, 1) - mean.get(1)) * scale.get(1) /
                    Math.sqrt(variance.get(1) + varianceEpsilon),
                (x.get(1, 0, 0) - mean.get(0)) * scale.get(0) /
                    Math.sqrt(variance.get(0) + varianceEpsilon),
                (x.get(1, 0, 1) - mean.get(1)) * scale.get(1) /
                    Math.sqrt(variance.get(1) + varianceEpsilon)
            ]), epsilon_1);
            x.dispose();
            mean.dispose();
            variance.dispose();
            scale.dispose();
        });
        it('simple batchnorm, no scale, 2x1x2', function (math) {
            var x = ndarray_1.Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
            var mean = ndarray_1.Array1D.new([1, 2]);
            var variance = ndarray_1.Array1D.new([2, 3]);
            var offset = ndarray_1.Array1D.new([4, 5]);
            var varianceEpsilon = .001;
            var result = math.batchNormalization3D(x, mean, variance, varianceEpsilon, undefined, offset);
            test_util.expectArraysClose(result.getValues(), new Float32Array([
                offset.get(0) +
                    (x.get(0, 0, 0) - mean.get(0)) * 1 /
                        Math.sqrt(variance.get(0) + varianceEpsilon),
                offset.get(1) +
                    (x.get(0, 0, 1) - mean.get(1)) * 1 /
                        Math.sqrt(variance.get(1) + varianceEpsilon),
                offset.get(0) +
                    (x.get(1, 0, 0) - mean.get(0)) * 1 /
                        Math.sqrt(variance.get(0) + varianceEpsilon),
                offset.get(1) +
                    (x.get(1, 0, 1) - mean.get(1)) * 1 /
                        Math.sqrt(variance.get(1) + varianceEpsilon)
            ]), epsilon_1);
            x.dispose();
            mean.dispose();
            variance.dispose();
            offset.dispose();
        });
        it('simple batchnorm, 2x1x2', function (math) {
            var x = ndarray_1.Array3D.new([2, 1, 2], new Float32Array([2, 100, 4, 400]));
            var mean = ndarray_1.Array1D.new([1, 2]);
            var variance = ndarray_1.Array1D.new([2, 3]);
            var offset = ndarray_1.Array1D.new([3, 4]);
            var scale = ndarray_1.Array1D.new([4, 5]);
            var varianceEpsilon = .001;
            var result = math.batchNormalization3D(x, mean, variance, varianceEpsilon, scale, offset);
            test_util.expectArraysClose(result.getValues(), new Float32Array([
                offset.get(0) +
                    (x.get(0, 0, 0) - mean.get(0)) * scale.get(0) /
                        Math.sqrt(variance.get(0) + varianceEpsilon),
                offset.get(1) +
                    (x.get(0, 0, 1) - mean.get(1)) * scale.get(1) /
                        Math.sqrt(variance.get(1) + varianceEpsilon),
                offset.get(0) +
                    (x.get(1, 0, 0) - mean.get(0)) * scale.get(0) /
                        Math.sqrt(variance.get(0) + varianceEpsilon),
                offset.get(1) +
                    (x.get(1, 0, 1) - mean.get(1)) * scale.get(1) /
                        Math.sqrt(variance.get(1) + varianceEpsilon)
            ]), epsilon_1);
            x.dispose();
            mean.dispose();
            variance.dispose();
            scale.dispose();
            offset.dispose();
        });
        it('batchnorm matches tensorflow, 2x3x3', function (math) {
            var x = ndarray_1.Array3D.new([2, 3, 3], new Float32Array([
                0.49955603, 0.04158615, -1.09440524, 2.03854165, -0.61578344,
                2.87533573, 1.18105987, 0.807462, 1.87888837, 2.26563962,
                -0.37040935, 1.35848753, -0.75347094, 0.15683117, 0.91925946,
                0.34121279, 0.92717143, 1.89683965
            ]));
            var mean = ndarray_1.Array1D.new([0.39745062, -0.48062894, 0.4847822]);
            var variance = ndarray_1.Array1D.new([0.32375343, 0.67117643, 1.08334653]);
            var offset = ndarray_1.Array1D.new([0.69398749, -1.29056387, 0.9429723]);
            var scale = ndarray_1.Array1D.new([-0.5607271, 0.9878457, 0.25181573]);
            var varianceEpsilon = .001;
            var result = math.batchNormalization3D(x, mean, variance, varianceEpsilon, scale, offset);
            test_util.expectArraysClose(result.getValues(), new Float32Array([
                0.59352049, -0.66135202, 0.5610874, -0.92077015, -1.45341019,
                1.52106473, -0.07704776, 0.26144429, 1.28010017, -1.14422404,
                -1.15776136, 1.15425493, 1.82644104, -0.52249442, 1.04803919,
                0.74932291, 0.40568101, 1.2844412
            ]));
            x.dispose();
            mean.dispose();
            variance.dispose();
            scale.dispose();
            offset.dispose();
        });
    };
    test_util.describeMathCPU('batchNormalization3D', [tests]);
    test_util.describeMathGPU('batchNormalization3D', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=batchnorm_test.js.map