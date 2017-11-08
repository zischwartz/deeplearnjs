"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("./util");
describe('Util', function () {
    it('Flatten arrays', function () {
        expect(util.flatten([[1, 2, 3], [4, 5, 6]])).toEqual([1, 2, 3, 4, 5, 6]);
        expect(util.flatten([[[1, 2], [3, 4], [5, 6], [7, 8]]])).toEqual([
            1, 2, 3, 4, 5, 6, 7, 8
        ]);
        expect(util.flatten([1, 2, 3, 4, 5, 6])).toEqual([1, 2, 3, 4, 5, 6]);
    });
    it('Correctly gets size from shape', function () {
        expect(util.sizeFromShape([1, 2, 3, 4])).toEqual(24);
    });
    it('Correctly identifies scalars', function () {
        expect(util.isScalarShape([])).toBe(true);
        expect(util.isScalarShape([1, 2])).toBe(false);
        expect(util.isScalarShape([1])).toBe(false);
    });
    it('Number arrays equal', function () {
        expect(util.arraysEqual([1, 2, 3, 6], [1, 2, 3, 6])).toBe(true);
        expect(util.arraysEqual([1, 2], [1, 2, 3])).toBe(false);
        expect(util.arraysEqual([1, 2, 5], [1, 2])).toBe(false);
    });
    it('Is integer', function () {
        expect(util.isInt(0.5)).toBe(false);
        expect(util.isInt(1)).toBe(true);
    });
    it('Size to squarish shape (perfect square)', function () {
        expect(util.sizeToSquarishShape(9)).toEqual([3, 3]);
    });
    it('Size to squarish shape (prime number)', function () {
        expect(util.sizeToSquarishShape(11)).toEqual([1, 11]);
    });
    it('Size to squarish shape (almost square)', function () {
        expect(util.sizeToSquarishShape(35)).toEqual([5, 7]);
    });
    it('Size of 1 to squarish shape', function () {
        expect(util.sizeToSquarishShape(1)).toEqual([1, 1]);
    });
    it('infer shape single number', function () {
        expect(util.inferShape(4)).toEqual([]);
    });
    it('infer shape 1d array', function () {
        expect(util.inferShape([1, 2, 5])).toEqual([3]);
    });
    it('infer shape 2d array', function () {
        expect(util.inferShape([[1, 2, 5], [5, 4, 1]])).toEqual([2, 3]);
    });
    it('infer shape 3d array', function () {
        var a = [[[1, 2], [2, 3], [5, 6]], [[5, 6], [4, 5], [1, 2]]];
        expect(util.inferShape(a)).toEqual([2, 3, 2]);
    });
    it('infer shape 4d array', function () {
        var a = [
            [[[1], [2]], [[2], [3]], [[5], [6]]],
            [[[5], [6]], [[4], [5]], [[1], [2]]]
        ];
        expect(util.inferShape(a)).toEqual([2, 3, 2, 1]);
    });
});
describe('util.repeatedTry', function () {
    it('resolves', function (doneFn) {
        var counter = 0;
        var checkFn = function () {
            counter++;
            if (counter === 2) {
                return true;
            }
            return false;
        };
        util.repeatedTry(checkFn).then(doneFn).catch(function () {
            throw new Error('Rejected backoff.');
        });
    });
    it('rejects', function (doneFn) {
        var checkFn = function () { return false; };
        util.repeatedTry(checkFn, function () { return 0; }, 5)
            .then(function () {
            throw new Error('Backoff resolved');
        })
            .catch(doneFn);
    });
});
describe('util.getQueryParams', function () {
    it('basic', function () {
        expect(util.getQueryParams('?a=1&b=hi&f=animal'))
            .toEqual({ 'a': '1', 'b': 'hi', 'f': 'animal' });
    });
});
describe('util.inferFromImplicitShape', function () {
    it('empty shape', function () {
        var result = util.inferFromImplicitShape([], 0);
        expect(result).toEqual([]);
    });
    it('[2, 3, 4] -> [2, 3, 4]', function () {
        var result = util.inferFromImplicitShape([2, 3, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, -1, 4] -> [2, 3, 4], size=24', function () {
        var result = util.inferFromImplicitShape([2, -1, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[-1, 3, 4] -> [2, 3, 4], size=24', function () {
        var result = util.inferFromImplicitShape([-1, 3, 4], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, 3, -1] -> [2, 3, 4], size=24', function () {
        var result = util.inferFromImplicitShape([2, 3, -1], 24);
        expect(result).toEqual([2, 3, 4]);
    });
    it('[2, -1, -1] throws error', function () {
        expect(function () { return util.inferFromImplicitShape([2, -1, -1], 24); }).toThrowError();
    });
    it('[2, 3, -1] size=13 throws error', function () {
        expect(function () { return util.inferFromImplicitShape([2, 3, -1], 13); }).toThrowError();
    });
    it('[2, 3, 4] size=25 (should be 24) throws error', function () {
        expect(function () { return util.inferFromImplicitShape([2, 3, 4], 25); }).toThrowError();
    });
});
describe('util.randGauss', function () {
    it('standard normal', function () {
        var a = util.randGauss();
        expect(a != null);
    });
    it('truncated standard normal', function () {
        var numSamples = 1000;
        for (var i = 0; i < numSamples; ++i) {
            var sample = util.randGauss(0, 1, true);
            expect(Math.abs(sample) <= 2);
        }
    });
    it('truncated normal, mu = 3, std=4', function () {
        var numSamples = 1000;
        var mean = 3;
        var stdDev = 4;
        for (var i = 0; i < numSamples; ++i) {
            var sample = util.randGauss(mean, stdDev, true);
            var normalizedSample = (sample - mean) / stdDev;
            expect(Math.abs(normalizedSample) <= 2);
        }
    });
});
describe('util.getNaN', function () {
    it('float32', function () {
        expect(isNaN(util.getNaN('float32'))).toBe(true);
    });
    it('int32', function () {
        expect(util.getNaN('int32')).toBe(util.NAN_INT32);
    });
    it('bool', function () {
        expect(util.getNaN('bool')).toBe(util.NAN_BOOL);
    });
    it('unknown type throws error', function () {
        expect(function () { return util.getNaN('hello'); }).toThrowError();
    });
});
describe('util.isValNaN', function () {
    it('NaN for float32', function () {
        expect(util.isValNaN(NaN, 'float32')).toBe(true);
    });
    it('2 for float32', function () {
        expect(util.isValNaN(3, 'float32')).toBe(false);
    });
    it('255 for float32', function () {
        expect(util.isValNaN(255, 'float32')).toBe(false);
    });
    it('255 for int32', function () {
        expect(util.isValNaN(255, 'int32')).toBe(false);
    });
    it('NAN_INT32 for int32', function () {
        expect(util.isValNaN(util.NAN_INT32, 'int32')).toBe(true);
    });
    it('NAN_INT32 for bool', function () {
        expect(util.isValNaN(util.NAN_INT32, 'bool')).toBe(false);
    });
    it('NAN_BOOL for bool', function () {
        expect(util.isValNaN(util.NAN_BOOL, 'bool')).toBe(true);
    });
    it('2 for bool', function () {
        expect(util.isValNaN(2, 'bool')).toBe(false);
    });
    it('throws error for unknown dtype', function () {
        expect(function () { return util.isValNaN(0, 'hello'); }).toThrowError();
    });
});
//# sourceMappingURL=util_test.js.map