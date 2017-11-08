"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var deeplearn_1 = require("../deeplearn");
function mlBeginners() {
    return __awaiter(this, void 0, void 0, function () {
        var _this = this;
        var math, matrixShape, matrix, vector, result, _a, _b, _c, graph, x_1, a, b, c, order2, order1, y_1, yLabel_1, cost_1, math, session_1;
        return __generator(this, function (_d) {
            switch (_d.label) {
                case 0:
                    math = new deeplearn_1.NDArrayMathGPU();
                    matrixShape = [2, 3];
                    matrix = deeplearn_1.Array2D.new(matrixShape, [10, 20, 30, 40, 50, 60]);
                    vector = deeplearn_1.Array1D.new([0, 1, 2]);
                    result = math.matrixTimesVector(matrix, vector);
                    console.log('result shape:', result.shape);
                    _b = (_a = console).log;
                    _c = ['result'];
                    return [4, result.data()];
                case 1:
                    _b.apply(_a, _c.concat([_d.sent()]));
                    graph = new deeplearn_1.Graph();
                    x_1 = graph.placeholder('x', []);
                    a = graph.variable('a', deeplearn_1.Scalar.new(Math.random()));
                    b = graph.variable('b', deeplearn_1.Scalar.new(Math.random()));
                    c = graph.variable('c', deeplearn_1.Scalar.new(Math.random()));
                    order2 = graph.multiply(a, graph.square(x_1));
                    order1 = graph.multiply(b, x_1);
                    y_1 = graph.add(graph.add(order2, order1), c);
                    yLabel_1 = graph.placeholder('y label', []);
                    cost_1 = graph.meanSquaredCost(y_1, yLabel_1);
                    math = new deeplearn_1.NDArrayMathGPU();
                    session_1 = new deeplearn_1.Session(graph, math);
                    return [4, math.scope(function (keep, track) { return __awaiter(_this, void 0, void 0, function () {
                            var result, xs, ys, shuffledInputProviderBuilder, _a, xProvider, yProvider, NUM_BATCHES, BATCH_SIZE, LEARNING_RATE, optimizer, i, costValue, _b, _c;
                            return __generator(this, function (_d) {
                                switch (_d.label) {
                                    case 0:
                                        result = session_1.eval(y_1, [{ tensor: x_1, data: track(deeplearn_1.Scalar.new(4)) }]);
                                        console.log(result.shape);
                                        console.log(result.getValues());
                                        xs = [
                                            track(deeplearn_1.Scalar.new(0)), track(deeplearn_1.Scalar.new(1)), track(deeplearn_1.Scalar.new(2)),
                                            track(deeplearn_1.Scalar.new(3))
                                        ];
                                        ys = [
                                            track(deeplearn_1.Scalar.new(1.1)), track(deeplearn_1.Scalar.new(5.9)), track(deeplearn_1.Scalar.new(16.8)),
                                            track(deeplearn_1.Scalar.new(33.9))
                                        ];
                                        shuffledInputProviderBuilder = new deeplearn_1.InCPUMemoryShuffledInputProviderBuilder([xs, ys]);
                                        _a = shuffledInputProviderBuilder.getInputProviders(), xProvider = _a[0], yProvider = _a[1];
                                        NUM_BATCHES = 20;
                                        BATCH_SIZE = xs.length;
                                        LEARNING_RATE = .01;
                                        optimizer = new deeplearn_1.SGDOptimizer(LEARNING_RATE);
                                        for (i = 0; i < NUM_BATCHES; i++) {
                                            costValue = session_1.train(cost_1, [{ tensor: x_1, data: xProvider }, { tensor: yLabel_1, data: yProvider }], BATCH_SIZE, optimizer, deeplearn_1.CostReduction.MEAN);
                                            console.log("average cost: " + costValue.get());
                                        }
                                        result = session_1.eval(y_1, [{ tensor: x_1, data: track(deeplearn_1.Scalar.new(4)) }]);
                                        console.log('result should be ~57.0:');
                                        console.log(result.shape);
                                        _c = (_b = console).log;
                                        return [4, result.data()];
                                    case 1:
                                        _c.apply(_b, [_d.sent()]);
                                        return [2];
                                }
                            });
                        }); })];
                case 2:
                    _d.sent();
                    return [2];
            }
        });
    });
}
mlBeginners();
//# sourceMappingURL=ml_beginners.js.map