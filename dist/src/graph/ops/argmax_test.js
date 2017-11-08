"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_cpu_1 = require("../../math/math_cpu");
var ndarray_1 = require("../../math/ndarray");
var test_util = require("../../test_util");
var graph_1 = require("../graph");
var tensor_array_map_1 = require("../tensor_array_map");
var argmax_1 = require("./argmax");
describe('Argmax oper', function () {
    var math;
    var x;
    var y;
    var tensorArrayMap;
    beforeEach(function () {
        math = new math_cpu_1.NDArrayMathCPU();
        tensorArrayMap = new tensor_array_map_1.TensorArrayMap();
    });
    afterEach(function () {
        tensorArrayMap.disposeArray(x);
        tensorArrayMap.disposeArray(y);
    });
    it('argmax of Array1D', function () {
        var vals = ndarray_1.Array1D.new([0, 2, 1]);
        x = new graph_1.Tensor(vals.shape);
        y = new graph_1.Tensor([]);
        tensorArrayMap.set(x, vals);
        var argmaxOp = new argmax_1.ArgMax(x, y);
        argmaxOp.feedForward(math, tensorArrayMap);
        var yVal = tensorArrayMap.get(y);
        expect(yVal.shape).toEqual([]);
        test_util.expectNumbersClose(yVal.get(), 1);
    });
    it('argmax of Array2D', function () {
        var vals = ndarray_1.Array2D.new([2, 3], [[0, 2, 1], [2, 3, 0]]);
        x = new graph_1.Tensor(vals.shape);
        y = new graph_1.Tensor([]);
        tensorArrayMap.set(x, vals);
        var argmaxOp = new argmax_1.ArgMax(x, y);
        argmaxOp.feedForward(math, tensorArrayMap);
        var yVal = tensorArrayMap.get(y);
        expect(yVal.shape).toEqual([]);
        test_util.expectNumbersClose(yVal.get(), 4);
    });
});
//# sourceMappingURL=argmax_test.js.map