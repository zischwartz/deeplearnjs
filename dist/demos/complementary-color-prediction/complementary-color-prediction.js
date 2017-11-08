"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var deeplearn_1 = require("../deeplearn");
var ComplementaryColorModel = (function () {
    function ComplementaryColorModel() {
        this.math = new deeplearn_1.NDArrayMathGPU();
        this.initialLearningRate = 0.042;
        this.batchSize = 300;
        this.optimizer = new deeplearn_1.SGDOptimizer(this.initialLearningRate);
    }
    ComplementaryColorModel.prototype.setupSession = function () {
        var graph = new deeplearn_1.Graph();
        this.inputTensor = graph.placeholder('input RGB value', [3]);
        this.targetTensor = graph.placeholder('output RGB value', [3]);
        var fullyConnectedLayer = this.createFullyConnectedLayer(graph, this.inputTensor, 0, 64);
        fullyConnectedLayer =
            this.createFullyConnectedLayer(graph, fullyConnectedLayer, 1, 32);
        fullyConnectedLayer =
            this.createFullyConnectedLayer(graph, fullyConnectedLayer, 2, 16);
        this.predictionTensor =
            this.createFullyConnectedLayer(graph, fullyConnectedLayer, 3, 3);
        this.costTensor =
            graph.meanSquaredCost(this.targetTensor, this.predictionTensor);
        this.session = new deeplearn_1.Session(graph, this.math);
        this.generateTrainingData(1e5);
    };
    ComplementaryColorModel.prototype.train1Batch = function (shouldFetchCost) {
        var _this = this;
        var learningRate = this.initialLearningRate * Math.pow(0.85, Math.floor(step / 42));
        this.optimizer.setLearningRate(learningRate);
        var costValue = -1;
        this.math.scope(function () {
            var cost = _this.session.train(_this.costTensor, _this.feedEntries, _this.batchSize, _this.optimizer, shouldFetchCost ? deeplearn_1.CostReduction.MEAN : deeplearn_1.CostReduction.NONE);
            if (!shouldFetchCost) {
                return;
            }
            costValue = cost.get();
        });
        return costValue;
    };
    ComplementaryColorModel.prototype.normalizeColor = function (rgbColor) {
        return rgbColor.map(function (v) { return v / 255; });
    };
    ComplementaryColorModel.prototype.denormalizeColor = function (normalizedRgbColor) {
        return normalizedRgbColor.map(function (v) { return v * 255; });
    };
    ComplementaryColorModel.prototype.predict = function (rgbColor) {
        var _this = this;
        var complementColor = [];
        this.math.scope(function (keep, track) {
            var mapping = [{
                    tensor: _this.inputTensor,
                    data: deeplearn_1.Array1D.new(_this.normalizeColor(rgbColor)),
                }];
            var evalOutput = _this.session.eval(_this.predictionTensor, mapping);
            var values = evalOutput.getValues();
            var colors = _this.denormalizeColor(Array.prototype.slice.call(values));
            complementColor =
                colors.map(function (v) { return Math.round(Math.max(Math.min(v, 255), 0)); });
        });
        return complementColor;
    };
    ComplementaryColorModel.prototype.createFullyConnectedLayer = function (graph, inputLayer, layerIndex, sizeOfThisLayer) {
        return graph.layers.dense("fully_connected_" + layerIndex, inputLayer, sizeOfThisLayer, function (x) { return graph.relu(x); });
    };
    ComplementaryColorModel.prototype.generateTrainingData = function (exampleCount) {
        var _this = this;
        this.math.scope(function () {
            var rawInputs = new Array(exampleCount);
            for (var i = 0; i < exampleCount; i++) {
                rawInputs[i] = [
                    _this.generateRandomChannelValue(), _this.generateRandomChannelValue(),
                    _this.generateRandomChannelValue()
                ];
            }
            var inputArray = rawInputs.map(function (c) { return deeplearn_1.Array1D.new(_this.normalizeColor(c)); });
            var targetArray = rawInputs.map(function (c) { return deeplearn_1.Array1D.new(_this.normalizeColor(_this.computeComplementaryColor(c))); });
            var shuffledInputProviderBuilder = new deeplearn_1.InCPUMemoryShuffledInputProviderBuilder([inputArray, targetArray]);
            var _a = shuffledInputProviderBuilder.getInputProviders(), inputProvider = _a[0], targetProvider = _a[1];
            _this.feedEntries = [
                { tensor: _this.inputTensor, data: inputProvider },
                { tensor: _this.targetTensor, data: targetProvider }
            ];
        });
    };
    ComplementaryColorModel.prototype.generateRandomChannelValue = function () {
        return Math.floor(Math.random() * 256);
    };
    ComplementaryColorModel.prototype.computeComplementaryColor = function (rgbColor) {
        var r = rgbColor[0];
        var g = rgbColor[1];
        var b = rgbColor[2];
        r /= 255.0;
        g /= 255.0;
        b /= 255.0;
        var max = Math.max(r, g, b);
        var min = Math.min(r, g, b);
        var h = (max + min) / 2.0;
        var s = h;
        var l = h;
        if (max === min) {
            h = s = 0;
        }
        else {
            var d = max - min;
            s = (l > 0.5 ? d / (2.0 - max - min) : d / (max + min));
            if (max === r && g >= b) {
                h = 1.0472 * (g - b) / d;
            }
            else if (max === r && g < b) {
                h = 1.0472 * (g - b) / d + 6.2832;
            }
            else if (max === g) {
                h = 1.0472 * (b - r) / d + 2.0944;
            }
            else if (max === b) {
                h = 1.0472 * (r - g) / d + 4.1888;
            }
        }
        h = h / 6.2832 * 360.0 + 0;
        h += 180;
        if (h > 360) {
            h -= 360;
        }
        h /= 360;
        if (s === 0) {
            r = g = b = l;
        }
        else {
            var hue2rgb = function (p, q, t) {
                if (t < 0)
                    t += 1;
                if (t > 1)
                    t -= 1;
                if (t < 1 / 6)
                    return p + (q - p) * 6 * t;
                if (t < 1 / 2)
                    return q;
                if (t < 2 / 3)
                    return p + (q - p) * (2 / 3 - t) * 6;
                return p;
            };
            var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            var p = 2 * l - q;
            r = hue2rgb(p, q, h + 1 / 3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1 / 3);
        }
        return [r, g, b].map(function (v) { return Math.round(v * 255); });
    };
    return ComplementaryColorModel;
}());
var complementaryColorModel = new ComplementaryColorModel();
complementaryColorModel.setupSession();
var step = 0;
function trainAndMaybeRender() {
    if (step > 4242) {
        return;
    }
    requestAnimationFrame(trainAndMaybeRender);
    var localStepsToRun = 5;
    var cost;
    for (var i = 0; i < localStepsToRun; i++) {
        cost = complementaryColorModel.train1Batch(i === localStepsToRun - 1);
        step++;
    }
    console.log('step', step - 1, 'cost', cost);
    var colorRows = document.querySelectorAll('tr[data-original-color]');
    for (var i = 0; i < colorRows.length; i++) {
        var rowElement = colorRows[i];
        var tds = rowElement.querySelectorAll('td');
        var originalColor = rowElement.getAttribute('data-original-color')
            .split(',')
            .map(function (v) { return parseInt(v, 10); });
        var predictedColor = complementaryColorModel.predict(originalColor);
        populateContainerWithColor(tds[2], predictedColor[0], predictedColor[1], predictedColor[2]);
    }
}
function populateContainerWithColor(container, r, g, b) {
    var originalColorString = 'rgb(' + [r, g, b].join(',') + ')';
    container.textContent = originalColorString;
    var colorBox = document.createElement('div');
    colorBox.classList.add('color-box');
    colorBox.style.background = originalColorString;
    container.appendChild(colorBox);
}
function initializeUi() {
    var colorRows = document.querySelectorAll('tr[data-original-color]');
    for (var i = 0; i < colorRows.length; i++) {
        var rowElement = colorRows[i];
        var tds = rowElement.querySelectorAll('td');
        var originalColor = rowElement.getAttribute('data-original-color')
            .split(',')
            .map(function (v) { return parseInt(v, 10); });
        populateContainerWithColor(tds[0], originalColor[0], originalColor[1], originalColor[2]);
        var complement = complementaryColorModel.computeComplementaryColor(originalColor);
        populateContainerWithColor(tds[1], complement[0], complement[1], complement[2]);
    }
}
initializeUi();
trainAndMaybeRender();
//# sourceMappingURL=complementary-color-prediction.js.map