"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_cpu_1 = require("../../math/math_cpu");
var ndarray_1 = require("../../math/ndarray");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var divide_1 = require("./divide");
describe('divide operation', function () {
    var math;
    var x1Tensor;
    var x2Tensor;
    var yTensor;
    var divideOp;
    var activations;
    var gradients;
    beforeEach(function () {
        math = new math_cpu_1.NDArrayMathCPU();
        activations = new tensor_array_map_1.TensorArrayMap();
        gradients = new tensor_array_map_1.SummedTensorArrayMap(math);
    });
    afterEach(function () {
        activations.disposeArray(x1Tensor);
        activations.disposeArray(x2Tensor);
        activations.disposeArray(yTensor);
        gradients.disposeArray(x1Tensor);
        gradients.disposeArray(x2Tensor);
        gradients.disposeArray(yTensor);
    });
    it('element wise divide', function () {
        var x1 = ndarray_1.Array1D.new([1, 2, 3]);
        var x2 = ndarray_1.Array1D.new([2, 4, 6]);
        x1Tensor = new graph_1.Tensor(x1.shape);
        x2Tensor = new graph_1.Tensor(x2.shape);
        yTensor = new graph_1.Tensor(x2.shape);
        activations.set(x1Tensor, x1);
        activations.set(x2Tensor, x2);
        divideOp = new divide_1.Divide(x1Tensor, x2Tensor, yTensor);
        divideOp.feedForward(math, activations);
        var y = activations.get(yTensor);
        expect(y.get(0)).toBeCloseTo(1 / 2);
        expect(y.get(1)).toBeCloseTo(2 / 4);
        expect(y.get(2)).toBeCloseTo(3 / 6);
        var dy = ndarray_1.Array1D.new([3, 4, 5]);
        gradients.add(yTensor, dy);
        divideOp.backProp(math, activations, gradients);
        var dx1 = gradients.get(x1Tensor);
        expect(dx1.get(0)).toBeCloseTo(dy.get(0) / x2.get(0));
        expect(dx1.get(1)).toBeCloseTo(dy.get(1) / x2.get(1));
        expect(dx1.get(2)).toBeCloseTo(dy.get(2) / x2.get(2));
        var dx2 = gradients.get(x2Tensor);
        expect(dx2.get(0))
            .toBeCloseTo(-1 * x1.get(0) * dy.get(0) * Math.pow(x2.get(0), -2));
        expect(dx2.get(1))
            .toBeCloseTo(-1 * x1.get(1) * dy.get(1) * Math.pow(x2.get(1), -2));
        expect(dx2.get(2))
            .toBeCloseTo(-1 * x1.get(2) * dy.get(2) * Math.pow(x2.get(2), -2));
    });
    it('scalar divided by ndarray', function () {
        var x1 = ndarray_1.Scalar.new(2);
        var x2 = ndarray_1.Array1D.new([2, 4, 6]);
        x1Tensor = new graph_1.Tensor(x1.shape);
        x2Tensor = new graph_1.Tensor(x2.shape);
        yTensor = new graph_1.Tensor(x2.shape);
        activations.set(x1Tensor, x1);
        activations.set(x2Tensor, x2);
        divideOp = new divide_1.Divide(x1Tensor, x2Tensor, yTensor);
        divideOp.feedForward(math, activations);
        var y = activations.get(yTensor);
        expect(y.get(0)).toBeCloseTo(2 / 2);
        expect(y.get(1)).toBeCloseTo(2 / 4);
        expect(y.get(2)).toBeCloseTo(2 / 6);
        var dy = ndarray_1.Array1D.new([3, 4, 5]);
        gradients.add(yTensor, dy);
        divideOp.backProp(math, activations, gradients);
        var dx1 = gradients.get(x1Tensor).asScalar();
        expect(dx1.get()).toBeCloseTo(dy.get(0) / x2.get(0) + dy.get(1) / x2.get(1) + dy.get(2) / x2.get(2));
        var dx2 = gradients.get(x2Tensor);
        expect(dx2.get(0))
            .toBeCloseTo(-1 * x1.get() * dy.get(0) * Math.pow(x2.get(0), -2));
        expect(dx2.get(1))
            .toBeCloseTo(-1 * x1.get() * dy.get(1) * Math.pow(x2.get(1), -2));
        expect(dx2.get(2))
            .toBeCloseTo(-1 * x1.get() * dy.get(2) * Math.pow(x2.get(2), -2));
    });
    it('ndarray divided by scalar', function () {
        var x1 = ndarray_1.Array1D.new([2, 4, 6]);
        var x2 = ndarray_1.Scalar.new(2);
        x1Tensor = new graph_1.Tensor(x1.shape);
        x2Tensor = new graph_1.Tensor(x2.shape);
        yTensor = new graph_1.Tensor(x2.shape);
        activations.set(x1Tensor, x1);
        activations.set(x2Tensor, x2);
        divideOp = new divide_1.Divide(x1Tensor, x2Tensor, yTensor);
        divideOp.feedForward(math, activations);
        var y = activations.get(yTensor);
        expect(y.get(0)).toBeCloseTo(2 / 2);
        expect(y.get(1)).toBeCloseTo(4 / 2);
        expect(y.get(2)).toBeCloseTo(6 / 2);
        var dy = ndarray_1.Array1D.new([3, 4, 5]);
        gradients.add(yTensor, dy);
        divideOp.backProp(math, activations, gradients);
        var dx1 = gradients.get(x1Tensor);
        expect(dx1.get(0)).toBeCloseTo(dy.get(0) / x2.get());
        expect(dx1.get(1)).toBeCloseTo(dy.get(1) / x2.get());
        expect(dx1.get(2)).toBeCloseTo(dy.get(2) / x2.get());
        var dx2 = gradients.get(x2Tensor).asScalar();
        expect(dx2.get()).toBeCloseTo(-1 * x1.get(0) * dy.get(0) * Math.pow(x2.get(), -2) +
            -1 * x1.get(1) * dy.get(1) * Math.pow(x2.get(), -2) +
            -1 * x1.get(2) * dy.get(2) * Math.pow(x2.get(), -2));
    });
});
//# sourceMappingURL=divide_test.js.map