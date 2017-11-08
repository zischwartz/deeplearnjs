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
var _this = this;
Object.defineProperty(exports, "__esModule", { value: true });
var deeplearn_1 = require("../deeplearn");
var reader = new deeplearn_1.CheckpointLoader('.');
reader.getAllVariables().then(function (vars) { return __awaiter(_this, void 0, void 0, function () {
    var _this = this;
    var primerData, expected, math, lstmKernel1, lstmBias1, lstmKernel2, lstmBias2, fullyConnectedBiases, fullyConnectedWeights, results;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                primerData = 3;
                expected = [1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4];
                math = new deeplearn_1.NDArrayMathGPU();
                lstmKernel1 = vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel'];
                lstmBias1 = vars['rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias'];
                lstmKernel2 = vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel'];
                lstmBias2 = vars['rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias'];
                fullyConnectedBiases = vars['fully_connected/biases'];
                fullyConnectedWeights = vars['fully_connected/weights'];
                results = [];
                return [4, math.scope(function (keep, track) { return __awaiter(_this, void 0, void 0, function () {
                        var forgetBias, lstm1, lstm2, c, h, input, i, onehot, output, outputH, weightedResult, logits, result;
                        return __generator(this, function (_a) {
                            switch (_a.label) {
                                case 0:
                                    forgetBias = track(deeplearn_1.Scalar.new(1.0));
                                    lstm1 = math.basicLSTMCell.bind(math, forgetBias, lstmKernel1, lstmBias1);
                                    lstm2 = math.basicLSTMCell.bind(math, forgetBias, lstmKernel2, lstmBias2);
                                    c = [
                                        track(deeplearn_1.Array2D.zeros([1, lstmBias1.shape[0] / 4])),
                                        track(deeplearn_1.Array2D.zeros([1, lstmBias2.shape[0] / 4]))
                                    ];
                                    h = [
                                        track(deeplearn_1.Array2D.zeros([1, lstmBias1.shape[0] / 4])),
                                        track(deeplearn_1.Array2D.zeros([1, lstmBias2.shape[0] / 4]))
                                    ];
                                    input = primerData;
                                    i = 0;
                                    _a.label = 1;
                                case 1:
                                    if (!(i < expected.length)) return [3, 4];
                                    onehot = track(deeplearn_1.Array2D.zeros([1, 10]));
                                    onehot.set(1.0, 0, input);
                                    output = math.multiRNNCell([lstm1, lstm2], onehot, c, h);
                                    c = output[0];
                                    h = output[1];
                                    outputH = h[1];
                                    weightedResult = math.matMul(outputH, fullyConnectedWeights);
                                    logits = math.add(weightedResult, fullyConnectedBiases);
                                    return [4, math.argMax(logits).val()];
                                case 2:
                                    result = _a.sent();
                                    results.push(result);
                                    input = result;
                                    _a.label = 3;
                                case 3:
                                    i++;
                                    return [3, 1];
                                case 4: return [2];
                            }
                        });
                    }); })];
            case 1:
                _a.sent();
                document.getElementById('expected').innerHTML = expected.toString();
                document.getElementById('results').innerHTML = results.toString();
                if (deeplearn_1.util.arraysEqual(expected, results)) {
                    document.getElementById('success').innerHTML = 'Success!';
                }
                else {
                    document.getElementById('success').innerHTML = 'Failure.';
                }
                return [2];
        }
    });
}); });
//# sourceMappingURL=lstm.js.map