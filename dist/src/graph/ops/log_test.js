"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_cpu_1 = require("../../math/math_cpu");
var ndarray_1 = require("../../math/ndarray");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var log_1 = require("./log");
describe('log operation', function () {
    var math;
    var xTensor;
    var yTensor;
    var logOp;
    var activations;
    var gradients;
    beforeEach(function () {
        math = new math_cpu_1.NDArrayMathCPU();
        activations = new tensor_array_map_1.TensorArrayMap();
        gradients = new tensor_array_map_1.SummedTensorArrayMap(math);
    });
    afterEach(function () {
        activations.disposeArray(xTensor);
        activations.disposeArray(yTensor);
        gradients.disposeArray(xTensor);
        gradients.disposeArray(yTensor);
    });
    it('simple log', function () {
        var x = ndarray_1.Array1D.new([1, 2, 3]);
        xTensor = new graph_1.Tensor(x.shape);
        yTensor = new graph_1.Tensor(x.shape);
        activations.set(xTensor, x);
        logOp = new log_1.Log(xTensor, yTensor);
        logOp.feedForward(math, activations);
        var y = activations.get(yTensor);
        expect(y.shape).toEqual([3]);
        expect(y.get(0)).toBeCloseTo(Math.log(x.get(0)));
        expect(y.get(1)).toBeCloseTo(Math.log(x.get(1)));
        expect(y.get(2)).toBeCloseTo(Math.log(x.get(2)));
        var dy = ndarray_1.Array1D.new([1, 2, 3]);
        gradients.add(yTensor, dy);
        logOp.backProp(math, activations, gradients);
        var dx = gradients.get(xTensor);
        expect(dx.shape).toEqual(dx.shape);
        expect(dx.get(0)).toBeCloseTo(dy.get(0) / x.get(0));
        expect(dx.get(1)).toBeCloseTo(dy.get(1) / x.get(1));
        expect(dx.get(2)).toBeCloseTo(dy.get(2) / x.get(2));
    });
});
//# sourceMappingURL=log_test.js.map