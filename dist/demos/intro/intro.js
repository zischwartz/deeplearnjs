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
function intro() {
    return __awaiter(this, void 0, void 0, function () {
        var _this = this;
        var math, a, b, diff, squaredDiff, sum, size, average, _a, _b, _c, g, inputShape, inputTensor, labelShape, labelTensor, multiplier, outputTensor, costTensor_1, learningRate, batchSize_1, math, session_1, optimizer_1, inputs, labels, shuffledInputProviderBuilder, _d, inputProvider, labelProvider, feedEntries_1, NUM_BATCHES, _loop_1, i, testInput, testFeedEntries, testOutput, _e, _f, _g;
        return __generator(this, function (_h) {
            switch (_h.label) {
                case 0:
                    math = new deeplearn_1.NDArrayMathGPU();
                    a = deeplearn_1.Array2D.new([2, 2], [1.0, 2.0, 3.0, 4.0]);
                    b = deeplearn_1.Array2D.new([2, 2], [0.0, 2.0, 4.0, 6.0]);
                    diff = math.sub(a, b);
                    squaredDiff = math.elementWiseMul(diff, diff);
                    sum = math.sum(squaredDiff);
                    size = deeplearn_1.Scalar.new(a.size);
                    average = math.divide(sum, size);
                    _b = (_a = console).log;
                    _c = "mean squared difference: ";
                    return [4, average.val()];
                case 1:
                    _b.apply(_a, [_c + (_h.sent())]);
                    g = new deeplearn_1.Graph();
                    inputShape = [3];
                    inputTensor = g.placeholder('input', inputShape);
                    labelShape = [1];
                    labelTensor = g.placeholder('label', labelShape);
                    multiplier = g.variable('multiplier', deeplearn_1.Array2D.randNormal([1, 3]));
                    outputTensor = g.matmul(multiplier, inputTensor);
                    costTensor_1 = g.meanSquaredCost(labelTensor, outputTensor);
                    console.log(outputTensor.shape);
                    learningRate = .00001;
                    batchSize_1 = 3;
                    math = new deeplearn_1.NDArrayMathGPU();
                    session_1 = new deeplearn_1.Session(g, math);
                    optimizer_1 = new deeplearn_1.SGDOptimizer(learningRate);
                    inputs = [
                        deeplearn_1.Array1D.new([1.0, 2.0, 3.0]), deeplearn_1.Array1D.new([10.0, 20.0, 30.0]),
                        deeplearn_1.Array1D.new([100.0, 200.0, 300.0])
                    ];
                    labels = [deeplearn_1.Array1D.new([4.0]), deeplearn_1.Array1D.new([40.0]), deeplearn_1.Array1D.new([400.0])];
                    shuffledInputProviderBuilder = new deeplearn_1.InCPUMemoryShuffledInputProviderBuilder([inputs, labels]);
                    _d = shuffledInputProviderBuilder.getInputProviders(), inputProvider = _d[0], labelProvider = _d[1];
                    feedEntries_1 = [
                        { tensor: inputTensor, data: inputProvider },
                        { tensor: labelTensor, data: labelProvider }
                    ];
                    NUM_BATCHES = 10;
                    _loop_1 = function (i) {
                        return __generator(this, function (_a) {
                            switch (_a.label) {
                                case 0: return [4, math.scope(function () { return __awaiter(_this, void 0, void 0, function () {
                                        var cost, _a, _b, _c;
                                        return __generator(this, function (_d) {
                                            switch (_d.label) {
                                                case 0:
                                                    cost = session_1.train(costTensor_1, feedEntries_1, batchSize_1, optimizer_1, deeplearn_1.CostReduction.MEAN);
                                                    _b = (_a = console).log;
                                                    _c = "last average cost (" + i + "): ";
                                                    return [4, cost.val()];
                                                case 1:
                                                    _b.apply(_a, [_c + (_d.sent())]);
                                                    return [2];
                                            }
                                        });
                                    }); })];
                                case 1:
                                    _a.sent();
                                    return [2];
                            }
                        });
                    };
                    i = 0;
                    _h.label = 2;
                case 2:
                    if (!(i < NUM_BATCHES)) return [3, 5];
                    return [5, _loop_1(i)];
                case 3:
                    _h.sent();
                    _h.label = 4;
                case 4:
                    i++;
                    return [3, 2];
                case 5:
                    testInput = deeplearn_1.Array1D.new([0.1, 0.2, 0.3]);
                    testFeedEntries = [{ tensor: inputTensor, data: testInput }];
                    testOutput = session_1.eval(outputTensor, testFeedEntries);
                    console.log('---inference output---');
                    console.log("shape: " + testOutput.shape);
                    _f = (_e = console).log;
                    _g = "value: ";
                    return [4, testOutput.val(0)];
                case 6:
                    _f.apply(_e, [_g + (_h.sent())]);
                    return [2];
            }
        });
    });
}
intro();
//# sourceMappingURL=intro.js.map