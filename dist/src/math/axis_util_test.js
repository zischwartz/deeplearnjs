"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var axis_util = require("./axis_util");
describe('axis_util combineLocations', function () {
    it('rank 4, reduce last 2 dims', function () {
        var loc = axis_util.combineLocations([4, 1], [3, 7], [2, 3]);
        expect(loc).toEqual([4, 1, 3, 7]);
    });
    it('rank 4, reduce first two dims', function () {
        var loc = axis_util.combineLocations([4, 1], [3, 7], [0, 1]);
        expect(loc).toEqual([3, 7, 4, 1]);
    });
    it('rank 4, reduce 1st and 3rd dims', function () {
        var loc = axis_util.combineLocations([4, 1], [3, 7], [0, 2]);
        expect(loc).toEqual([3, 4, 7, 1]);
    });
    it('rank 4, reduce 1st and 4th dims', function () {
        var loc = axis_util.combineLocations([4, 1], [3, 7], [0, 3]);
        expect(loc).toEqual([3, 4, 1, 7]);
    });
    it('rank 3, reduce all dims', function () {
        var loc = axis_util.combineLocations([], [3, 7, 1], [0, 1, 2]);
        expect(loc).toEqual([3, 7, 1]);
    });
    it('rank 2, reduce last dim', function () {
        var loc = axis_util.combineLocations([3], [5], [1]);
        expect(loc).toEqual([3, 5]);
    });
    it('rank 2, reduce first dim', function () {
        var loc = axis_util.combineLocations([3], [5], [0]);
        expect(loc).toEqual([5, 3]);
    });
});
describe('axis_util computeOutAndReduceShapes', function () {
    it('rank 4, reduce all dims', function () {
        var _a = axis_util.computeOutAndReduceShapes([3, 7, 2, 4], [0, 1, 2, 3]), out = _a[0], red = _a[1];
        expect(out).toEqual([]);
        expect(red).toEqual([3, 7, 2, 4]);
    });
    it('rank 4, reduce last 2 dims', function () {
        var _a = axis_util.computeOutAndReduceShapes([3, 7, 2, 4], [2, 3]), out = _a[0], red = _a[1];
        expect(out).toEqual([3, 7]);
        expect(red).toEqual([2, 4]);
    });
    it('rank 4, reduce first 2 dims', function () {
        var _a = axis_util.computeOutAndReduceShapes([3, 7, 2, 4], [0, 1]), out = _a[0], red = _a[1];
        expect(out).toEqual([2, 4]);
        expect(red).toEqual([3, 7]);
    });
    it('rank 4, reduce last 3 dims', function () {
        var _a = axis_util.computeOutAndReduceShapes([3, 7, 2, 4], [1, 2, 3]), out = _a[0], red = _a[1];
        expect(out).toEqual([3]);
        expect(red).toEqual([7, 2, 4]);
    });
    it('rank 4, reduce 1st and 3rd dims', function () {
        var _a = axis_util.computeOutAndReduceShapes([3, 7, 2, 4], [0, 2]), out = _a[0], red = _a[1];
        expect(out).toEqual([7, 4]);
        expect(red).toEqual([3, 2]);
    });
    it('rank 3, reduce all dims', function () {
        var _a = axis_util.computeOutAndReduceShapes([3, 7, 2], [0, 1, 2]), out = _a[0], red = _a[1];
        expect(out).toEqual([]);
        expect(red).toEqual([3, 7, 2]);
    });
});
describe('axis_util axesAreInnerMostDims', function () {
    it('rank 4, reduce last dim', function () {
        var res = axis_util.axesAreInnerMostDims([3], 4);
        expect(res).toBe(true);
    });
    it('rank 4, reduce last 2 dims', function () {
        var res = axis_util.axesAreInnerMostDims([2, 3], 4);
        expect(res).toBe(true);
    });
    it('rank 4, reduce last 3 dims', function () {
        var res = axis_util.axesAreInnerMostDims([1, 2, 3], 4);
        expect(res).toBe(true);
    });
    it('rank 4, reduce all dims', function () {
        var res = axis_util.axesAreInnerMostDims([0, 1, 2, 3], 4);
        expect(res).toBe(true);
    });
    it('rank 4, reduce all but 2nd', function () {
        var res = axis_util.axesAreInnerMostDims([0, 2, 3], 4);
        expect(res).toBe(false);
    });
    it('rank 4, reduce all but 3rd', function () {
        var res = axis_util.axesAreInnerMostDims([0, 1, 3], 4);
        expect(res).toBe(false);
    });
    it('rank 4, reduce all but last', function () {
        var res = axis_util.axesAreInnerMostDims([0, 1, 2], 4);
        expect(res).toBe(false);
    });
});
describe('axis_util expandShapeToKeepDim', function () {
    it('2d -> 1d axis=0', function () {
        var shape = axis_util.expandShapeToKeepDim([2], [0]);
        expect(shape).toEqual([1, 2]);
    });
    it('2d -> 1d axis=1', function () {
        var shape = axis_util.expandShapeToKeepDim([4], [1]);
        expect(shape).toEqual([4, 1]);
    });
    it('3d -> 1d axis=1,2', function () {
        var shape = axis_util.expandShapeToKeepDim([7], [1, 2]);
        expect(shape).toEqual([7, 1, 1]);
    });
    it('3d -> 2d axis=1', function () {
        var shape = axis_util.expandShapeToKeepDim([7, 3], [1]);
        expect(shape).toEqual([7, 1, 3]);
    });
});
//# sourceMappingURL=axis_util_test.js.map