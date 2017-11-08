"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var concat_util = require("../../math/concat_util");
var math_cpu_1 = require("../../math/math_cpu");
var ndarray_1 = require("../../math/ndarray");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var concat3d_1 = require("./concat3d");
describe('concat3d operation', function () {
    var math;
    var x1Tensor;
    var x2Tensor;
    var yTensor;
    var concatOperation;
    var tensorArrayMap;
    beforeEach(function () {
        math = new math_cpu_1.NDArrayMathCPU();
        tensorArrayMap = new tensor_array_map_1.TensorArrayMap();
    });
    afterEach(function () {
        tensorArrayMap.disposeArray(x1Tensor);
        tensorArrayMap.disposeArray(x2Tensor);
        tensorArrayMap.disposeArray(yTensor);
    });
    it('concats tensors, axis=0', function () {
        var axis = 0;
        var x1 = ndarray_1.Array3D.new([1, 1, 3], [1, 2, 3]);
        var x2 = ndarray_1.Array3D.new([1, 1, 3], [4, 5, 6]);
        x1Tensor = new graph_1.Tensor(x1.shape);
        x2Tensor = new graph_1.Tensor(x2.shape);
        yTensor = new graph_1.Tensor(concat_util.computeOutShape(x1.shape, x2.shape, axis));
        tensorArrayMap.set(x1Tensor, x1);
        tensorArrayMap.set(x2Tensor, x2);
        concatOperation = new concat3d_1.Concat3D(x1Tensor, x2Tensor, axis, yTensor);
        concatOperation.feedForward(math, tensorArrayMap);
        var y = tensorArrayMap.get(yTensor);
        expect(y.shape).toEqual([2, 1, 3]);
        expect(y.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
    });
    it('concats tensors, axis=1', function () {
        var axis = 1;
        var x1 = ndarray_1.Array3D.new([1, 1, 3], [1, 2, 3]);
        var x2 = ndarray_1.Array3D.new([1, 1, 3], [4, 5, 6]);
        x1Tensor = new graph_1.Tensor(x1.shape);
        x2Tensor = new graph_1.Tensor(x2.shape);
        yTensor = new graph_1.Tensor(concat_util.computeOutShape(x1.shape, x2.shape, axis));
        tensorArrayMap.set(x1Tensor, x1);
        tensorArrayMap.set(x2Tensor, x2);
        concatOperation = new concat3d_1.Concat3D(x1Tensor, x2Tensor, axis, yTensor);
        concatOperation.feedForward(math, tensorArrayMap);
        var y = tensorArrayMap.get(yTensor);
        expect(y.shape).toEqual([1, 2, 3]);
        expect(y.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
    });
    it('concats tensors, axis=2', function () {
        var axis = 2;
        var x1 = ndarray_1.Array3D.new([1, 1, 3], [1, 2, 3]);
        var x2 = ndarray_1.Array3D.new([1, 1, 3], [4, 5, 6]);
        x1Tensor = new graph_1.Tensor(x1.shape);
        x2Tensor = new graph_1.Tensor(x2.shape);
        yTensor = new graph_1.Tensor(concat_util.computeOutShape(x1.shape, x2.shape, axis));
        tensorArrayMap.set(x1Tensor, x1);
        tensorArrayMap.set(x2Tensor, x2);
        concatOperation = new concat3d_1.Concat3D(x1Tensor, x2Tensor, axis, yTensor);
        concatOperation.feedForward(math, tensorArrayMap);
        var y = tensorArrayMap.get(yTensor);
        expect(y.shape).toEqual([1, 1, 6]);
        expect(y.getValues()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
    });
});
//# sourceMappingURL=concat3d_test.js.map