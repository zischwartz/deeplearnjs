"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_cpu_1 = require("../../math/math_cpu");
var ndarray_1 = require("../../math/ndarray");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var softmax_1 = require("./softmax");
describe('softmax cross entropy cost', function () {
    var math;
    var logitsTensor;
    var labelTensor;
    var yTensor;
    var activations;
    var gradients;
    beforeEach(function () {
        math = new math_cpu_1.NDArrayMathCPU();
        activations = new tensor_array_map_1.TensorArrayMap();
        gradients = new tensor_array_map_1.SummedTensorArrayMap(math);
    });
    afterEach(function () {
        activations.disposeArray(logitsTensor);
        activations.disposeArray(yTensor);
        gradients.disposeArray(logitsTensor);
        gradients.disposeArray(yTensor);
    });
    it('matches theory', function () {
        var logits = ndarray_1.Array1D.new([1, 2, 3]);
        var label = ndarray_1.Array1D.new([0.3, 0.6, 0.1]);
        var softmaxLogits = math.softmax(logits);
        logitsTensor = new graph_1.Tensor(logits.shape);
        labelTensor = new graph_1.Tensor(label.shape);
        yTensor = new graph_1.Tensor([]);
        activations.set(logitsTensor, logits);
        activations.set(labelTensor, label);
        var op = new softmax_1.SoftmaxCrossEntropyCost(logitsTensor, labelTensor, yTensor);
        op.feedForward(math, activations);
        var y = activations.get(yTensor);
        expect(y.get(0)).toBeCloseTo(-Math.log(softmaxLogits.get(0)) * label.get(0) +
            -Math.log(softmaxLogits.get(1)) * label.get(1) +
            -Math.log(softmaxLogits.get(2)) * label.get(2), 3);
        var dy = ndarray_1.Scalar.new(1);
        gradients.add(yTensor, dy);
        op.backProp(math, activations, gradients);
        var dLogits = gradients.get(logitsTensor);
        expect(dLogits.get(0)).toBeCloseTo(softmaxLogits.get(0) - label.get(0), 6);
        expect(dLogits.get(1)).toBeCloseTo(softmaxLogits.get(1) - label.get(1), 6);
        expect(dLogits.get(2)).toBeCloseTo(softmaxLogits.get(2) - label.get(2), 6);
    });
});
//# sourceMappingURL=softmax_test.js.map