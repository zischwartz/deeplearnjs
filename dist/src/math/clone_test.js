"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
var commonTests = function (it) {
    it('returns a ndarray with the same shape and value', function (math) {
        var a = ndarray_1.Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        var aPrime = math.clone(a);
        expect(aPrime.shape).toEqual(a.shape);
        test_util.expectArraysClose(aPrime.getValues(), a.getValues());
        a.dispose();
    });
};
var gpuTests = function (it) {
    it('returns a ndarray with a different texture handle', function (math) {
        var a = ndarray_1.Array2D.new([3, 3], [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        var aPrime = math.clone(a);
        expect(a.inGPU()).toEqual(true);
        expect(aPrime.inGPU()).toEqual(true);
        expect(aPrime.getTexture()).not.toBe(a.getTexture());
        a.dispose();
    });
};
test_util.describeMathCPU('clone', [commonTests]);
test_util.describeMathGPU('clone', [commonTests, gpuTests], [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
]);
//# sourceMappingURL=clone_test.js.map