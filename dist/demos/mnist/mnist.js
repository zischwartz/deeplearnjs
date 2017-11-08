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
reader.getAllVariables().then(function (vars) {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', 'sample_data.json');
    xhr.onload = function () { return __awaiter(_this, void 0, void 0, function () {
        var _this = this;
        var data, math;
        return __generator(this, function (_a) {
            data = JSON.parse(xhr.responseText);
            math = new deeplearn_1.NDArrayMathGPU();
            math.scope(function () { return __awaiter(_this, void 0, void 0, function () {
                var numCorrect, i, x, predictedLabel, _a, _b, label, result, accuracy;
                return __generator(this, function (_c) {
                    switch (_c.label) {
                        case 0:
                            console.log("Evaluation set: n=" + data.images.length + ".");
                            numCorrect = 0;
                            i = 0;
                            _c.label = 1;
                        case 1:
                            if (!(i < data.images.length)) return [3, 4];
                            x = deeplearn_1.Array1D.new(data.images[i]);
                            _b = (_a = Math).round;
                            return [4, infer(math, x, vars).val()];
                        case 2:
                            predictedLabel = _b.apply(_a, [_c.sent()]);
                            console.log("Item " + i + ", predicted label " + predictedLabel + ".");
                            label = data.labels[i];
                            if (label === predictedLabel) {
                                numCorrect++;
                            }
                            result = renderResults(deeplearn_1.Array1D.new(data.images[i]), label, predictedLabel);
                            document.body.appendChild(result);
                            _c.label = 3;
                        case 3:
                            i++;
                            return [3, 1];
                        case 4:
                            accuracy = numCorrect * 100 / data.images.length;
                            document.getElementById('accuracy').innerHTML = accuracy + "%";
                            return [2];
                    }
                });
            }); });
            return [2];
        });
    }); };
    xhr.onerror = function (err) { return console.error(err); };
    xhr.send();
});
function infer(math, x, vars) {
    var hidden1W = vars['hidden1/weights'];
    var hidden1B = vars['hidden1/biases'];
    var hidden2W = vars['hidden2/weights'];
    var hidden2B = vars['hidden2/biases'];
    var softmaxW = vars['softmax_linear/weights'];
    var softmaxB = vars['softmax_linear/biases'];
    var hidden1 = math.relu(math.add(math.vectorTimesMatrix(x, hidden1W), hidden1B));
    var hidden2 = math.relu(math.add(math.vectorTimesMatrix(hidden1, hidden2W), hidden2B));
    var logits = math.add(math.vectorTimesMatrix(hidden2, softmaxW), softmaxB);
    return math.argMax(logits);
}
exports.infer = infer;
function renderMnistImage(array) {
    var width = 28;
    var height = 28;
    var canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    var ctx = canvas.getContext('2d');
    var float32Array = array.getData().values;
    var imageData = ctx.createImageData(width, height);
    for (var i = 0; i < float32Array.length; i++) {
        var j = i * 4;
        var value = Math.round(float32Array[i] * 255);
        imageData.data[j + 0] = value;
        imageData.data[j + 1] = value;
        imageData.data[j + 2] = value;
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas;
}
function renderResults(array, label, predictedLabel) {
    var root = document.createElement('div');
    root.appendChild(renderMnistImage(array));
    var actual = document.createElement('div');
    actual.innerHTML = "Actual: " + label;
    root.appendChild(actual);
    var predicted = document.createElement('div');
    predicted.innerHTML = "Predicted: " + predictedLabel;
    root.appendChild(predicted);
    if (label !== predictedLabel) {
        root.classList.add('error');
    }
    root.classList.add('result');
    return root;
}
//# sourceMappingURL=mnist.js.map