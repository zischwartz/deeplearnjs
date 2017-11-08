"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var concat_util = require("./concat_util");
describe('concat_util.assertConcatShapesMatch rank=3D', function () {
    it('Non-3D tensor x1', function () {
        var assertFn = function () {
            concat_util.assertParams([1], [1, 2, 3], 1);
        };
        expect(assertFn).toThrow();
    });
    it('Non-3D tensor x2', function () {
        var assertFn = function () {
            concat_util.assertParams([1, 2, 3], [2, 3], 1);
        };
        expect(assertFn).toThrow();
    });
    it('axis out of bound', function () {
        var assertFn = function () {
            concat_util.assertParams([1, 2, 3], [1, 2, 3], 4);
        };
        expect(assertFn).toThrow();
    });
    it('non-axis shape mismatch', function () {
        var assertFn = function () {
            concat_util.assertParams([2, 3, 3], [2, 2, 4], 2);
        };
        expect(assertFn).toThrow();
    });
    it('shapes line up', function () {
        var assertFn = function () {
            concat_util.assertParams([2, 3, 3], [2, 3, 4], 2);
        };
        expect(assertFn).not.toThrow();
    });
});
describe('concat_util.computeConcatOutputShape', function () {
    it('compute output shape, axis=0', function () {
        expect(concat_util.computeOutShape([2, 2, 3], [1, 2, 3], 0)).toEqual([
            3, 2, 3
        ]);
    });
});
//# sourceMappingURL=concat_util_test.js.map