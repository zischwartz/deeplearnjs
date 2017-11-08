"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var cost_functions_1 = require("./cost_functions");
var math_cpu_1 = require("./math_cpu");
var ndarray_1 = require("./ndarray");
describe('Cost functions', function () {
    var math;
    beforeEach(function () {
        math = new math_cpu_1.NDArrayMathCPU();
    });
    it('Square cost', function () {
        var y = ndarray_1.Array1D.new([1, 3, -2]);
        var target = ndarray_1.Array1D.new([0, 3, -1.5]);
        var square = new cost_functions_1.SquareCostFunc();
        var cost = square.cost(math, y, target);
        expect(cost.get(0)).toBeCloseTo(1 / 2);
        expect(cost.get(1)).toBeCloseTo(0 / 2);
        expect(cost.get(2)).toBeCloseTo(0.25 / 2);
    });
    it('Square derivative', function () {
        var y = ndarray_1.Array1D.new([1, 3, -2]);
        var target = ndarray_1.Array1D.new([0, 3, -1.5]);
        var square = new cost_functions_1.SquareCostFunc();
        var dy = square.der(math, y, target);
        expect(dy.get(0)).toBeCloseTo(1);
        expect(dy.get(1)).toBeCloseTo(0);
        expect(dy.get(2)).toBeCloseTo(-0.5);
    });
});
//# sourceMappingURL=cost_functions_test.js.map