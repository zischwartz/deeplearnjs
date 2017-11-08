"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('2D (no change)', function (math) {
            var t = ndarray_1.Array2D.new([2, 4], [1, 11, 2, 22, 3, 33, 4, 44]);
            var t2 = math.transpose(t, [0, 1]);
            expect(t2.shape).toEqual(t.shape);
            test_util.expectArraysClose(t2.getValues(), t.getValues());
            t.dispose();
        });
        it('2D (transpose)', function (math) {
            var t = ndarray_1.Array2D.new([2, 4], [1, 11, 2, 22, 3, 33, 4, 44]);
            var t2 = math.transpose(t, [1, 0]);
            expect(t2.shape).toEqual([4, 2]);
            var expected = new Float32Array([1, 3, 11, 33, 2, 4, 22, 44]);
            test_util.expectArraysClose(t2.getValues(), expected);
            t.dispose();
        });
        it('3D [r, c, d] => [d, r, c]', function (math) {
            var t = ndarray_1.Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
            var t2 = math.transpose(t, [2, 0, 1]);
            expect(t2.shape).toEqual([2, 2, 2]);
            var expected = new Float32Array([1, 2, 3, 4, 11, 22, 33, 44]);
            test_util.expectArraysClose(t2.getValues(), expected);
            t.dispose();
        });
        it('3D [r, c, d] => [d, c, r]', function (math) {
            var t = ndarray_1.Array3D.new([2, 2, 2], [1, 11, 2, 22, 3, 33, 4, 44]);
            var t2 = math.transpose(t, [2, 1, 0]);
            expect(t2.shape).toEqual([2, 2, 2]);
            var expected = new Float32Array([1, 3, 2, 4, 11, 33, 22, 44]);
            test_util.expectArraysClose(t2.getValues(), expected);
            t.dispose();
        });
    };
    test_util.describeMathCPU('transpose', [tests]);
    test_util.describeMathGPU('transpose', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=transpose_test.js.map