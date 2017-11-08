"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("./conv_util");
describe('conv_util computeConvInfo', function () {
    it('1x1 conv over 1x1 array with same pad', function () {
        var inShape = [1, 1, 1];
        var convInfo = conv_util.computeConvInfo(inShape, 1, 1, 1, 1, 1, 'same');
        expect(convInfo.outShape).toEqual([1, 1, 1]);
    });
    it('2x2 conv over 3x3 array with same pad', function () {
        var inShape = [3, 3, 1];
        var convInfo = conv_util.computeConvInfo(inShape, 2, 2, 1, 1, 1, 'same');
        expect(convInfo.outShape).toEqual([3, 3, 1]);
        expect(convInfo.padInfo.left).toBe(0);
        expect(convInfo.padInfo.right).toBe(1);
        expect(convInfo.padInfo.top).toBe(0);
        expect(convInfo.padInfo.bottom).toBe(1);
    });
    it('2x2 conv over 3x3 array with same pad', function () {
        var inShape = [3, 3, 1];
        var convInfo = conv_util.computeConvInfo(inShape, 2, 2, 1, 1, 1, 'same');
        expect(convInfo.outShape).toEqual([3, 3, 1]);
    });
    it('2x2 conv over 3x3 array with valid pad', function () {
        var inShape = [3, 3, 1];
        var convInfo = conv_util.computeConvInfo(inShape, 2, 2, 1, 1, 1, 'valid');
        expect(convInfo.outShape).toEqual([2, 2, 1]);
    });
    it('2x2 conv over 3x3 array with valid pad with stride 2', function () {
        var inShape = [3, 3, 1];
        var convInfo = conv_util.computeConvInfo(inShape, 2, 2, 1, 2, 2, 'valid');
        expect(convInfo.outShape).toEqual([1, 1, 1]);
    });
    it('2x2 conv over 3x3 array with valid pad with stride 2', function () {
        var inShape = [3, 3, 1];
        var convInfo = conv_util.computeConvInfo(inShape, 2, 2, 1, 2, 2, 'valid');
        expect(convInfo.outShape).toEqual([1, 1, 1]);
    });
    it('2x1 conv over 3x3 array with valid pad with stride 1', function () {
        var inShape = [3, 3, 1];
        var convInfo = conv_util.computeConvInfo(inShape, 2, 1, 1, 1, 1, 'valid');
        expect(convInfo.outShape).toEqual([2, 3, 1]);
    });
    it('2x1 conv over 3x3 array with valid pad with strides h=2, w=1', function () {
        var inShape = [3, 3, 1];
        var convInfo = conv_util.computeConvInfo(inShape, 2, 1, 1, 2, 1, 'valid');
        expect(convInfo.outShape).toEqual([1, 3, 1]);
    });
    it('1x2 conv over 3x3 array with valid pad with stride 1', function () {
        var inShape = [3, 3, 1];
        var convInfo = conv_util.computeConvInfo(inShape, 1, 2, 1, 1, 1, 'valid');
        expect(convInfo.outShape).toEqual([3, 2, 1]);
    });
});
//# sourceMappingURL=conv_util_test.js.map