"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
var activation_functions_1 = require("./activation_functions");
var math_cpu_1 = require("./math_cpu");
var ndarray_1 = require("./ndarray");
describe('Activation functions', function () {
    var math;
    beforeEach(function () {
        math = new math_cpu_1.NDArrayMathCPU();
    });
    it('Tanh output', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2, 100, -100, 0]);
        var tanH = new activation_functions_1.TanHFunc();
        var y = tanH.output(math, x);
        expect(y.get(0)).toBeCloseTo(util.tanh(x.get(0)));
        expect(y.get(1)).toBeCloseTo(util.tanh(x.get(1)));
        expect(y.get(2)).toBeCloseTo(util.tanh(x.get(2)));
        expect(y.get(3)).toBeCloseTo(1);
        expect(y.get(4)).toBeCloseTo(-1);
        expect(y.get(5)).toBeCloseTo(0);
    });
    it('Tanh derivative', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2, 100, -100, 0]);
        var tanH = new activation_functions_1.TanHFunc();
        var y = tanH.output(math, x);
        var dx = tanH.der(math, x, y);
        expect(dx.get(0)).toBeCloseTo(1 - Math.pow(y.get(0), 2));
        expect(dx.get(1)).toBeCloseTo(1 - Math.pow(y.get(1), 2));
        expect(dx.get(2)).toBeCloseTo(1 - Math.pow(y.get(2), 2));
    });
    it('ReLU output', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2]);
        var relu = new activation_functions_1.ReLUFunc();
        var y = relu.output(math, x);
        expect(y.get(0)).toBeCloseTo(1);
        expect(y.get(1)).toBeCloseTo(3);
        expect(y.get(2)).toBeCloseTo(0);
    });
    it('ReLU derivative', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2]);
        var relu = new activation_functions_1.ReLUFunc();
        var y = relu.output(math, x);
        var dx = relu.der(math, x, y);
        expect(dx.get(0)).toBeCloseTo(1);
        expect(dx.get(1)).toBeCloseTo(1);
        expect(dx.get(2)).toBeCloseTo(0);
    });
    it('Sigmoid output', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2, 100, -100, 0]);
        var sigmoid = new activation_functions_1.SigmoidFunc();
        var y = sigmoid.output(math, x);
        expect(y.get(0)).toBeCloseTo(1 / (1 + Math.exp(-1)));
        expect(y.get(1)).toBeCloseTo(1 / (1 + Math.exp(-3)));
        expect(y.get(2)).toBeCloseTo(1 / (1 + Math.exp(2)));
        expect(y.get(3)).toBeCloseTo(1);
        expect(y.get(4)).toBeCloseTo(0);
        expect(y.get(5)).toBeCloseTo(0.5);
    });
    it('Sigmoid derivative', function () {
        var x = ndarray_1.Array1D.new([1, 3, -2, 100, -100, 0]);
        var sigmoid = new activation_functions_1.SigmoidFunc();
        var y = sigmoid.output(math, x);
        var dx = sigmoid.der(math, x, y);
        expect(dx.get(0)).toBeCloseTo(y.get(0) * (1 - y.get(0)));
        expect(dx.get(1)).toBeCloseTo(y.get(1) * (1 - y.get(1)));
        expect(dx.get(2)).toBeCloseTo(y.get(2) * (1 - y.get(2)));
    });
});
//# sourceMappingURL=activation_functions_test.js.map