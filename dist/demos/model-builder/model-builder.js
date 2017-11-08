"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
require("../ndarray-image-visualizer");
require("../ndarray-logits-visualizer");
require("./model-layer");
require("../demo-header");
require("../demo-footer");
var deeplearn_1 = require("../deeplearn");
var polymer_spec_1 = require("../polymer-spec");
var model_builder_util = require("./model_builder_util");
var DATASETS_CONFIG_JSON = 'model-builder-datasets-config.json';
var EVAL_INTERVAL_MS = 1500;
var COST_INTERVAL_MS = 500;
var INFERENCE_EXAMPLE_COUNT = 15;
var INFERENCE_IMAGE_SIZE_PX = 100;
var INFERENCE_EXAMPLE_INTERVAL_MS = 3000;
var EXAMPLE_SEC_STAT_SMOOTHING_FACTOR = .7;
var TRAIN_TEST_RATIO = 5 / 6;
var IMAGE_DATA_INDEX = 0;
var LABEL_DATA_INDEX = 1;
var Normalization;
(function (Normalization) {
    Normalization[Normalization["NORMALIZATION_NEGATIVE_ONE_TO_ONE"] = 0] = "NORMALIZATION_NEGATIVE_ONE_TO_ONE";
    Normalization[Normalization["NORMALIZATION_ZERO_TO_ONE"] = 1] = "NORMALIZATION_ZERO_TO_ONE";
    Normalization[Normalization["NORMALIZATION_NONE"] = 2] = "NORMALIZATION_NONE";
})(Normalization || (Normalization = {}));
exports.ModelBuilderPolymer = polymer_spec_1.PolymerElement({
    is: 'model-builder',
    properties: {
        inputShapeDisplay: String,
        isValid: Boolean,
        inferencesPerSec: Number,
        inferenceDuration: Number,
        examplesTrained: Number,
        examplesPerSec: Number,
        totalTimeSec: String,
        applicationState: Number,
        modelInitialized: Boolean,
        showTrainStats: Boolean,
        datasetDownloaded: Boolean,
        datasetNames: Array,
        selectedDatasetName: String,
        modelNames: Array,
        selectedOptimizerName: String,
        optimizerNames: Array,
        learningRate: Number,
        momentum: Number,
        needMomentum: Boolean,
        gamma: Number,
        needGamma: Boolean,
        beta1: Number,
        needBeta1: Boolean,
        beta2: Number,
        needBeta2: Boolean,
        batchSize: Number,
        selectedModelName: String,
        selectedNormalizationOption: { type: Number, value: Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE },
        showDatasetStats: Boolean,
        statsInputMin: Number,
        statsInputMax: Number,
        statsInputShapeDisplay: String,
        statsLabelShapeDisplay: String,
        statsExampleCount: Number,
    }
});
var ApplicationState;
(function (ApplicationState) {
    ApplicationState[ApplicationState["IDLE"] = 1] = "IDLE";
    ApplicationState[ApplicationState["TRAINING"] = 2] = "TRAINING";
})(ApplicationState = exports.ApplicationState || (exports.ApplicationState = {}));
var ModelBuilder = (function (_super) {
    __extends(ModelBuilder, _super);
    function ModelBuilder() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    ModelBuilder.prototype.ready = function () {
        var _this = this;
        this.mathGPU = new deeplearn_1.NDArrayMathGPU();
        this.mathCPU = new deeplearn_1.NDArrayMathCPU();
        this.math = this.mathGPU;
        var eventObserver = {
            batchesTrainedCallback: function (batchesTrained) {
                return _this.displayBatchesTrained(batchesTrained);
            },
            avgCostCallback: function (avgCost) { return _this.displayCost(avgCost); },
            metricCallback: function (metric) { return _this.displayAccuracy(metric); },
            inferenceExamplesCallback: function (inputFeeds, inferenceOutputs) {
                return _this.displayInferenceExamplesOutput(inputFeeds, inferenceOutputs);
            },
            inferenceExamplesPerSecCallback: function (examplesPerSec) {
                return _this.displayInferenceExamplesPerSec(examplesPerSec);
            },
            trainExamplesPerSecCallback: function (examplesPerSec) {
                return _this.displayExamplesPerSec(examplesPerSec);
            },
            totalTimeCallback: function (totalTimeSec) { return _this.totalTimeSec =
                totalTimeSec.toFixed(1); },
        };
        this.graphRunner = new deeplearn_1.GraphRunner(this.math, this.session, eventObserver);
        this.optimizer = new deeplearn_1.MomentumOptimizer(this.learningRate, this.momentum);
        this.populateDatasets();
        this.querySelector('#dataset-dropdown .dropdown-content')
            .addEventListener('iron-activate', function (event) {
            var datasetName = event.detail.selected;
            _this.updateSelectedDataset(datasetName);
            _this.removeAllLayers();
        });
        this.querySelector('#model-dropdown .dropdown-content')
            .addEventListener('iron-activate', function (event) {
            var modelName = event.detail.selected;
            _this.updateSelectedModel(modelName);
        });
        {
            var normalizationDropdown = this.querySelector('#normalization-dropdown .dropdown-content');
            normalizationDropdown.addEventListener('iron-activate', function (event) {
                var selectedNormalizationOption = event.detail.selected;
                _this.applyNormalization(selectedNormalizationOption);
                _this.setupDatasetStats();
            });
        }
        this.querySelector('#optimizer-dropdown .dropdown-content')
            .addEventListener('iron-activate', function (event) {
            _this.refreshHyperParamRequirements(event.detail.selected);
        });
        this.learningRate = 0.1;
        this.momentum = 0.1;
        this.needMomentum = true;
        this.gamma = 0.1;
        this.needGamma = false;
        this.beta1 = 0.9;
        this.needBeta1 = false;
        this.beta2 = 0.999;
        this.needBeta2 = false;
        this.batchSize = 64;
        this.selectedOptimizerName = 'momentum';
        this.optimizerNames =
            ['sgd', 'momentum', 'rmsprop', 'adagrad', 'adadelta', 'adam'];
        this.applicationState = ApplicationState.IDLE;
        this.loadedWeights = null;
        this.modelInitialized = false;
        this.showTrainStats = false;
        this.showDatasetStats = false;
        var addButton = this.querySelector('#add-layer');
        addButton.addEventListener('click', function () { return _this.addLayer(); });
        var downloadModelButton = this.querySelector('#download-model');
        downloadModelButton.addEventListener('click', function () { return _this.downloadModel(); });
        var uploadModelButton = this.querySelector('#upload-model');
        uploadModelButton.addEventListener('click', function () { return _this.uploadModel(); });
        this.setupUploadModelButton();
        var uploadWeightsButton = this.querySelector('#upload-weights');
        uploadWeightsButton.addEventListener('click', function () { return _this.uploadWeights(); });
        this.setupUploadWeightsButton();
        var stopButton = this.querySelector('#stop');
        stopButton.addEventListener('click', function () {
            _this.applicationState = ApplicationState.IDLE;
            _this.graphRunner.stopTraining();
        });
        this.trainButton = this.querySelector('#train');
        this.trainButton.addEventListener('click', function () {
            _this.createModel();
            _this.startTraining();
        });
        this.querySelector('#environment-toggle')
            .addEventListener('change', function (event) {
            _this.math =
                event.target.active ? _this.mathGPU : _this.mathCPU;
            _this.graphRunner.setMath(_this.math);
        });
        this.hiddenLayers = [];
        this.examplesPerSec = 0;
        this.inferencesPerSec = 0;
    };
    ModelBuilder.prototype.isTraining = function (applicationState) {
        return applicationState === ApplicationState.TRAINING;
    };
    ModelBuilder.prototype.isIdle = function (applicationState) {
        return applicationState === ApplicationState.IDLE;
    };
    ModelBuilder.prototype.getTestData = function () {
        var data = this.dataSet.getData();
        if (data == null) {
            return null;
        }
        var _a = this.dataSet.getData(), images = _a[0], labels = _a[1];
        var start = Math.floor(TRAIN_TEST_RATIO * images.length);
        return [images.slice(start), labels.slice(start)];
    };
    ModelBuilder.prototype.getTrainingData = function () {
        var _a = this.dataSet.getData(), images = _a[0], labels = _a[1];
        var end = Math.floor(TRAIN_TEST_RATIO * images.length);
        return [images.slice(0, end), labels.slice(0, end)];
    };
    ModelBuilder.prototype.startInference = function () {
        var testData = this.getTestData();
        if (testData == null) {
            return;
        }
        if (this.isValid && (testData != null)) {
            var inferenceShuffledInputProviderGenerator = new deeplearn_1.InCPUMemoryShuffledInputProviderBuilder(testData);
            var _a = inferenceShuffledInputProviderGenerator.getInputProviders(), inferenceInputProvider = _a[0], inferenceLabelProvider = _a[1];
            var inferenceFeeds = [
                { tensor: this.xTensor, data: inferenceInputProvider },
                { tensor: this.labelTensor, data: inferenceLabelProvider }
            ];
            this.graphRunner.infer(this.predictionTensor, inferenceFeeds, INFERENCE_EXAMPLE_INTERVAL_MS, INFERENCE_EXAMPLE_COUNT);
        }
    };
    ModelBuilder.prototype.resetHyperParamRequirements = function () {
        this.needMomentum = false;
        this.needGamma = false;
        this.needBeta1 = false;
        this.needBeta2 = false;
    };
    ModelBuilder.prototype.refreshHyperParamRequirements = function (optimizerName) {
        this.resetHyperParamRequirements();
        switch (optimizerName) {
            case 'sgd': {
                break;
            }
            case 'momentum': {
                this.needMomentum = true;
                break;
            }
            case 'rmsprop': {
                this.needMomentum = true;
                this.needGamma = true;
                break;
            }
            case 'adagrad': {
                break;
            }
            case 'adadelta': {
                this.needGamma = true;
                break;
            }
            case 'adam': {
                this.needBeta1 = true;
                this.needBeta2 = true;
                break;
            }
            default: {
                throw new Error("Unknown optimizer \"" + this.selectedOptimizerName + "\"");
            }
        }
    };
    ModelBuilder.prototype.createOptimizer = function () {
        switch (this.selectedOptimizerName) {
            case 'sgd': {
                return new deeplearn_1.SGDOptimizer(+this.learningRate);
            }
            case 'momentum': {
                return new deeplearn_1.MomentumOptimizer(+this.learningRate, +this.momentum);
            }
            case 'rmsprop': {
                return new deeplearn_1.RMSPropOptimizer(+this.learningRate, +this.gamma);
            }
            case 'adagrad': {
                return new deeplearn_1.AdagradOptimizer(+this.learningRate);
            }
            case 'adadelta': {
                return new deeplearn_1.AdadeltaOptimizer(+this.learningRate, +this.gamma);
            }
            case 'adam': {
                return new deeplearn_1.AdamOptimizer(+this.learningRate, +this.beta1, +this.beta2);
            }
            default: {
                throw new Error("Unknown optimizer \"" + this.selectedOptimizerName + "\"");
            }
        }
    };
    ModelBuilder.prototype.startTraining = function () {
        var trainingData = this.getTrainingData();
        var testData = this.getTestData();
        this.optimizer = this.createOptimizer();
        if (this.isValid && (trainingData != null) && (testData != null)) {
            this.recreateCharts();
            this.graphRunner.resetStatistics();
            var trainingShuffledInputProviderGenerator = new deeplearn_1.InCPUMemoryShuffledInputProviderBuilder(trainingData);
            var _a = trainingShuffledInputProviderGenerator.getInputProviders(), trainInputProvider = _a[0], trainLabelProvider = _a[1];
            var trainFeeds = [
                { tensor: this.xTensor, data: trainInputProvider },
                { tensor: this.labelTensor, data: trainLabelProvider }
            ];
            var accuracyShuffledInputProviderGenerator = new deeplearn_1.InCPUMemoryShuffledInputProviderBuilder(testData);
            var _b = accuracyShuffledInputProviderGenerator.getInputProviders(), accuracyInputProvider = _b[0], accuracyLabelProvider = _b[1];
            var accuracyFeeds = [
                { tensor: this.xTensor, data: accuracyInputProvider },
                { tensor: this.labelTensor, data: accuracyLabelProvider }
            ];
            this.graphRunner.train(this.costTensor, trainFeeds, this.batchSize, this.optimizer, undefined, this.accuracyTensor, accuracyFeeds, this.batchSize, deeplearn_1.MetricReduction.MEAN, EVAL_INTERVAL_MS, COST_INTERVAL_MS);
            this.showTrainStats = true;
            this.applicationState = ApplicationState.TRAINING;
        }
    };
    ModelBuilder.prototype.createModel = function () {
        if (this.session != null) {
            this.session.dispose();
        }
        this.modelInitialized = false;
        if (this.isValid === false) {
            return;
        }
        this.graph = new deeplearn_1.Graph();
        var g = this.graph;
        this.xTensor = g.placeholder('input', this.inputShape);
        this.labelTensor = g.placeholder('label', this.labelShape);
        var network = this.xTensor;
        for (var i = 0; i < this.hiddenLayers.length; i++) {
            var weights = null;
            if (this.loadedWeights != null) {
                weights = this.loadedWeights[i];
            }
            network = this.hiddenLayers[i].addLayer(g, network, i, weights);
        }
        this.predictionTensor = network;
        this.costTensor =
            g.softmaxCrossEntropyCost(this.predictionTensor, this.labelTensor);
        this.accuracyTensor =
            g.argmaxEquals(this.predictionTensor, this.labelTensor);
        this.loadedWeights = null;
        this.session = new deeplearn_1.Session(g, this.math);
        this.graphRunner.setSession(this.session);
        this.startInference();
        this.modelInitialized = true;
    };
    ModelBuilder.prototype.populateDatasets = function () {
        var _this = this;
        this.dataSets = {};
        deeplearn_1.xhr_dataset.getXhrDatasetConfig(DATASETS_CONFIG_JSON)
            .then(function (xhrDatasetConfigs) {
            for (var datasetName in xhrDatasetConfigs) {
                if (xhrDatasetConfigs.hasOwnProperty(datasetName)) {
                    _this.dataSets[datasetName] =
                        new deeplearn_1.XhrDataset(xhrDatasetConfigs[datasetName]);
                }
            }
            _this.datasetNames = Object.keys(_this.dataSets);
            _this.selectedDatasetName = _this.datasetNames[0];
            _this.xhrDatasetConfigs = xhrDatasetConfigs;
            _this.updateSelectedDataset(_this.datasetNames[0]);
        }, function (error) {
            throw new Error("Dataset config could not be loaded: " + error);
        });
    };
    ModelBuilder.prototype.updateSelectedDataset = function (datasetName) {
        var _this = this;
        if (this.dataSet != null) {
            this.dataSet.removeNormalization(IMAGE_DATA_INDEX);
        }
        this.graphRunner.stopTraining();
        this.graphRunner.stopInferring();
        if (this.dataSet != null) {
            this.dataSet.dispose();
        }
        this.selectedDatasetName = datasetName;
        this.selectedModelName = '';
        this.dataSet = this.dataSets[datasetName];
        this.datasetDownloaded = false;
        this.showDatasetStats = false;
        this.dataSet.fetchData().then(function () {
            _this.datasetDownloaded = true;
            _this.applyNormalization(_this.selectedNormalizationOption);
            _this.setupDatasetStats();
            if (_this.isValid) {
                _this.createModel();
                _this.startInference();
            }
            _this.populateModelDropdown();
        });
        this.inputShape = this.dataSet.getDataShape(IMAGE_DATA_INDEX);
        this.labelShape = this.dataSet.getDataShape(LABEL_DATA_INDEX);
        this.layersContainer =
            this.querySelector('#hidden-layers');
        this.inputLayer = this.querySelector('#input-layer');
        this.inputLayer.outputShapeDisplay =
            model_builder_util.getDisplayShape(this.inputShape);
        var labelShapeDisplay = model_builder_util.getDisplayShape(this.labelShape);
        var costLayer = this.querySelector('#cost-layer');
        costLayer.inputShapeDisplay = labelShapeDisplay;
        costLayer.outputShapeDisplay = labelShapeDisplay;
        var outputLayer = this.querySelector('#output-layer');
        outputLayer.inputShapeDisplay = labelShapeDisplay;
        var inferenceContainer = this.querySelector('#inference-container');
        inferenceContainer.innerHTML = '';
        this.inputNDArrayVisualizers = [];
        this.outputNDArrayVisualizers = [];
        for (var i = 0; i < INFERENCE_EXAMPLE_COUNT; i++) {
            var inferenceExampleElement = document.createElement('div');
            inferenceExampleElement.className = 'inference-example';
            var ndarrayImageVisualizer = document.createElement('ndarray-image-visualizer');
            ndarrayImageVisualizer.setShape(this.inputShape);
            ndarrayImageVisualizer.setSize(INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
            this.inputNDArrayVisualizers.push(ndarrayImageVisualizer);
            inferenceExampleElement.appendChild(ndarrayImageVisualizer);
            var ndarrayLogitsVisualizer = document.createElement('ndarray-logits-visualizer');
            ndarrayLogitsVisualizer.initialize(INFERENCE_IMAGE_SIZE_PX, INFERENCE_IMAGE_SIZE_PX);
            this.outputNDArrayVisualizers.push(ndarrayLogitsVisualizer);
            inferenceExampleElement.appendChild(ndarrayLogitsVisualizer);
            inferenceContainer.appendChild(inferenceExampleElement);
        }
    };
    ModelBuilder.prototype.populateModelDropdown = function () {
        var modelNames = ['Custom'];
        var modelConfigs = this.xhrDatasetConfigs[this.selectedDatasetName].modelConfigs;
        for (var modelName in modelConfigs) {
            if (modelConfigs.hasOwnProperty(modelName)) {
                modelNames.push(modelName);
            }
        }
        this.modelNames = modelNames;
        this.selectedModelName = modelNames[modelNames.length - 1];
        this.updateSelectedModel(this.selectedModelName);
    };
    ModelBuilder.prototype.updateSelectedModel = function (modelName) {
        this.removeAllLayers();
        if (modelName === 'Custom') {
            return;
        }
        this.loadModelFromPath(this.xhrDatasetConfigs[this.selectedDatasetName]
            .modelConfigs[modelName]
            .path);
    };
    ModelBuilder.prototype.loadModelFromPath = function (modelPath) {
        var _this = this;
        var xhr = new XMLHttpRequest();
        xhr.open('GET', modelPath);
        xhr.onload = function () {
            _this.loadModelFromJson(xhr.responseText);
        };
        xhr.onerror = function (error) {
            throw new Error("Model could not be fetched from " + modelPath + ": " + error);
        };
        xhr.send();
    };
    ModelBuilder.prototype.setupDatasetStats = function () {
        this.datasetStats = this.dataSet.getStats();
        this.statsExampleCount = this.datasetStats[IMAGE_DATA_INDEX].exampleCount;
        this.statsInputRange =
            "[" + this.datasetStats[IMAGE_DATA_INDEX].inputMin + ", " +
                (this.datasetStats[IMAGE_DATA_INDEX].inputMax + "]");
        this.statsInputShapeDisplay = model_builder_util.getDisplayShape(this.datasetStats[IMAGE_DATA_INDEX].shape);
        this.statsLabelShapeDisplay = model_builder_util.getDisplayShape(this.datasetStats[LABEL_DATA_INDEX].shape);
        this.showDatasetStats = true;
    };
    ModelBuilder.prototype.applyNormalization = function (selectedNormalizationOption) {
        switch (selectedNormalizationOption) {
            case Normalization.NORMALIZATION_NEGATIVE_ONE_TO_ONE: {
                this.dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, -1, 1);
                break;
            }
            case Normalization.NORMALIZATION_ZERO_TO_ONE: {
                this.dataSet.normalizeWithinBounds(IMAGE_DATA_INDEX, 0, 1);
                break;
            }
            case Normalization.NORMALIZATION_NONE: {
                this.dataSet.removeNormalization(IMAGE_DATA_INDEX);
                break;
            }
            default: {
                throw new Error('Normalization option must be 0, 1, or 2');
            }
        }
        this.setupDatasetStats();
    };
    ModelBuilder.prototype.recreateCharts = function () {
        this.costChartData = [];
        if (this.costChart != null) {
            this.costChart.destroy();
        }
        this.costChart =
            this.createChart('cost-chart', 'Cost', this.costChartData, 0);
        if (this.accuracyChart != null) {
            this.accuracyChart.destroy();
        }
        this.accuracyChartData = [];
        this.accuracyChart = this.createChart('accuracy-chart', 'Accuracy', this.accuracyChartData, 0, 100);
        if (this.examplesPerSecChart != null) {
            this.examplesPerSecChart.destroy();
        }
        this.examplesPerSecChartData = [];
        this.examplesPerSecChart = this.createChart('examplespersec-chart', 'Examples/sec', this.examplesPerSecChartData, 0);
    };
    ModelBuilder.prototype.createChart = function (canvasId, label, data, min, max) {
        var context = document.getElementById(canvasId)
            .getContext('2d');
        return new Chart(context, {
            type: 'line',
            data: {
                datasets: [{
                        data: data,
                        fill: false,
                        label: label,
                        pointRadius: 0,
                        borderColor: 'rgba(75,192,192,1)',
                        borderWidth: 1,
                        lineTension: 0,
                        pointHitRadius: 8
                    }]
            },
            options: {
                animation: { duration: 0 },
                responsive: false,
                scales: {
                    xAxes: [{ type: 'linear', position: 'bottom' }],
                    yAxes: [{
                            ticks: {
                                max: max,
                                min: min,
                            }
                        }]
                }
            }
        });
    };
    ModelBuilder.prototype.displayBatchesTrained = function (totalBatchesTrained) {
        this.examplesTrained = this.batchSize * totalBatchesTrained;
    };
    ModelBuilder.prototype.displayCost = function (avgCost) {
        this.costChartData.push({ x: this.graphRunner.getTotalBatchesTrained(), y: avgCost.get() });
        this.costChart.update();
    };
    ModelBuilder.prototype.displayAccuracy = function (accuracy) {
        this.accuracyChartData.push({
            x: this.graphRunner.getTotalBatchesTrained(),
            y: accuracy.get() * 100
        });
        this.accuracyChart.update();
    };
    ModelBuilder.prototype.displayInferenceExamplesPerSec = function (examplesPerSec) {
        this.inferencesPerSec =
            this.smoothExamplesPerSec(this.inferencesPerSec, examplesPerSec);
        this.inferenceDuration = Number((1000 / examplesPerSec).toPrecision(3));
    };
    ModelBuilder.prototype.displayExamplesPerSec = function (examplesPerSec) {
        this.examplesPerSecChartData.push({ x: this.graphRunner.getTotalBatchesTrained(), y: examplesPerSec });
        this.examplesPerSecChart.update();
        this.examplesPerSec =
            this.smoothExamplesPerSec(this.examplesPerSec, examplesPerSec);
    };
    ModelBuilder.prototype.smoothExamplesPerSec = function (lastExamplesPerSec, nextExamplesPerSec) {
        return Number((EXAMPLE_SEC_STAT_SMOOTHING_FACTOR * lastExamplesPerSec +
            (1 - EXAMPLE_SEC_STAT_SMOOTHING_FACTOR) * nextExamplesPerSec)
            .toPrecision(3));
    };
    ModelBuilder.prototype.displayInferenceExamplesOutput = function (inputFeeds, inferenceOutputs) {
        var images = [];
        var logits = [];
        var labels = [];
        for (var i = 0; i < inputFeeds.length; i++) {
            images.push(inputFeeds[i][IMAGE_DATA_INDEX].data);
            labels.push(inputFeeds[i][LABEL_DATA_INDEX].data);
            logits.push(inferenceOutputs[i]);
        }
        images =
            this.dataSet.unnormalizeExamples(images, IMAGE_DATA_INDEX);
        for (var i = 0; i < inputFeeds.length; i++) {
            this.inputNDArrayVisualizers[i].saveImageDataFromNDArray(images[i]);
        }
        for (var i = 0; i < inputFeeds.length; i++) {
            var softmaxLogits = this.math.softmax(logits[i]);
            this.outputNDArrayVisualizers[i].drawLogits(softmaxLogits, labels[i], this.xhrDatasetConfigs[this.selectedDatasetName].labelClassNames);
            this.inputNDArrayVisualizers[i].draw();
            softmaxLogits.dispose();
        }
    };
    ModelBuilder.prototype.addLayer = function () {
        var modelLayer = document.createElement('model-layer');
        modelLayer.className = 'layer';
        this.layersContainer.appendChild(modelLayer);
        var lastHiddenLayer = this.hiddenLayers[this.hiddenLayers.length - 1];
        var lastOutputShape = lastHiddenLayer != null ?
            lastHiddenLayer.getOutputShape() :
            this.inputShape;
        this.hiddenLayers.push(modelLayer);
        modelLayer.initialize(this, lastOutputShape);
        return modelLayer;
    };
    ModelBuilder.prototype.removeLayer = function (modelLayer) {
        this.layersContainer.removeChild(modelLayer);
        this.hiddenLayers.splice(this.hiddenLayers.indexOf(modelLayer), 1);
        this.layerParamChanged();
    };
    ModelBuilder.prototype.removeAllLayers = function () {
        for (var i = 0; i < this.hiddenLayers.length; i++) {
            this.layersContainer.removeChild(this.hiddenLayers[i]);
        }
        this.hiddenLayers = [];
        this.layerParamChanged();
    };
    ModelBuilder.prototype.validateModel = function () {
        var valid = true;
        for (var i = 0; i < this.hiddenLayers.length; ++i) {
            valid = valid && this.hiddenLayers[i].isValid();
        }
        if (this.hiddenLayers.length > 0) {
            var lastLayer = this.hiddenLayers[this.hiddenLayers.length - 1];
            valid = valid &&
                deeplearn_1.util.arraysEqual(this.labelShape, lastLayer.getOutputShape());
        }
        this.isValid = valid && (this.hiddenLayers.length > 0);
    };
    ModelBuilder.prototype.layerParamChanged = function () {
        var lastOutputShape = this.inputShape;
        for (var i = 0; i < this.hiddenLayers.length; i++) {
            lastOutputShape = this.hiddenLayers[i].setInputShape(lastOutputShape);
        }
        this.validateModel();
        if (this.isValid) {
            this.createModel();
            this.startInference();
        }
    };
    ModelBuilder.prototype.downloadModel = function () {
        var modelJson = this.getModelAsJson();
        var blob = new Blob([modelJson], { type: 'text/json' });
        var textFile = window.URL.createObjectURL(blob);
        var a = document.createElement('a');
        document.body.appendChild(a);
        a.style.display = 'none';
        a.href = textFile;
        a.download = this.selectedDatasetName + '_model';
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(textFile);
    };
    ModelBuilder.prototype.uploadModel = function () {
        this.querySelector('#model-file').click();
    };
    ModelBuilder.prototype.setupUploadModelButton = function () {
        var _this = this;
        var fileInput = this.querySelector('#model-file');
        fileInput.addEventListener('change', function (event) {
            var file = fileInput.files[0];
            fileInput.value = '';
            var fileReader = new FileReader();
            fileReader.onload = function (evt) {
                _this.removeAllLayers();
                var modelJson = fileReader.result;
                _this.loadModelFromJson(modelJson);
            };
            fileReader.readAsText(file);
        });
    };
    ModelBuilder.prototype.getModelAsJson = function () {
        var layerBuilders = [];
        for (var i = 0; i < this.hiddenLayers.length; i++) {
            layerBuilders.push(this.hiddenLayers[i].layerBuilder);
        }
        return JSON.stringify(layerBuilders);
    };
    ModelBuilder.prototype.loadModelFromJson = function (modelJson) {
        var lastOutputShape = this.inputShape;
        var layerBuilders = JSON.parse(modelJson);
        for (var i = 0; i < layerBuilders.length; i++) {
            var modelLayer = this.addLayer();
            modelLayer.loadParamsFromLayerBuilder(lastOutputShape, layerBuilders[i]);
            lastOutputShape = this.hiddenLayers[i].setInputShape(lastOutputShape);
        }
        this.validateModel();
    };
    ModelBuilder.prototype.uploadWeights = function () {
        this.querySelector('#weights-file').click();
    };
    ModelBuilder.prototype.setupUploadWeightsButton = function () {
        var _this = this;
        var fileInput = this.querySelector('#weights-file');
        fileInput.addEventListener('change', function (event) {
            var file = fileInput.files[0];
            fileInput.value = '';
            var fileReader = new FileReader();
            fileReader.onload = function (evt) {
                var weightsJson = fileReader.result;
                _this.loadWeightsFromJson(weightsJson);
                _this.createModel();
                _this.startInference();
            };
            fileReader.readAsText(file);
        });
    };
    ModelBuilder.prototype.loadWeightsFromJson = function (weightsJson) {
        this.loadedWeights = JSON.parse(weightsJson);
    };
    return ModelBuilder;
}(exports.ModelBuilderPolymer));
exports.ModelBuilder = ModelBuilder;
document.registerElement(ModelBuilder.prototype.is, ModelBuilder);
//# sourceMappingURL=model-builder.js.map