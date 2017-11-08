"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var deeplearn_1 = require("../deeplearn");
function getLayerBuilder(layerName, layerBuilderJson) {
    var layerBuilder;
    switch (layerName) {
        case 'Fully connected':
            layerBuilder = new FullyConnectedLayerBuilder();
            break;
        case 'ReLU':
            layerBuilder = new ReLULayerBuilder();
            break;
        case 'Convolution':
            layerBuilder = new Convolution2DLayerBuilder();
            break;
        case 'Max pool':
            layerBuilder = new MaxPoolLayerBuilder();
            break;
        case 'Reshape':
            layerBuilder = new ReshapeLayerBuilder();
            break;
        case 'Flatten':
            layerBuilder = new FlattenLayerBuilder();
            break;
        default:
            throw new Error("Layer builder for " + layerName + " not found.");
    }
    if (layerBuilderJson != null) {
        for (var prop in layerBuilderJson) {
            if (layerBuilderJson.hasOwnProperty(prop)) {
                layerBuilder[prop] = layerBuilderJson[prop];
            }
        }
    }
    return layerBuilder;
}
exports.getLayerBuilder = getLayerBuilder;
var FullyConnectedLayerBuilder = (function () {
    function FullyConnectedLayerBuilder() {
        this.layerName = 'Fully connected';
    }
    FullyConnectedLayerBuilder.prototype.getLayerParams = function () {
        var _this = this;
        return [{
                label: 'Hidden units',
                initialValue: function (inputShape) { return 10; },
                type: 'number',
                min: 1,
                max: 1000,
                setValue: function (value) { return _this.hiddenUnits = value; },
                getValue: function () { return _this.hiddenUnits; }
            }];
    };
    FullyConnectedLayerBuilder.prototype.getOutputShape = function (inputShape) {
        return [this.hiddenUnits];
    };
    FullyConnectedLayerBuilder.prototype.addLayer = function (g, network, inputShape, index, weights) {
        var inputSize = deeplearn_1.util.sizeFromShape(inputShape);
        var wShape = [this.hiddenUnits, inputSize];
        var weightsInitializer;
        var biasInitializer;
        if (weights != null) {
            weightsInitializer =
                new deeplearn_1.NDArrayInitializer(deeplearn_1.Array2D.new(wShape, weights['W']));
            biasInitializer = new deeplearn_1.NDArrayInitializer(deeplearn_1.Array1D.new(weights['b']));
        }
        else {
            weightsInitializer = new deeplearn_1.VarianceScalingInitializer();
            biasInitializer = new deeplearn_1.ZerosInitializer();
        }
        var useBias = true;
        return g.layers.dense('fc1', network, this.hiddenUnits, null, useBias, weightsInitializer, biasInitializer);
    };
    FullyConnectedLayerBuilder.prototype.validate = function (inputShape) {
        if (inputShape.length !== 1) {
            return ['Input shape must be a Array1D.'];
        }
        return null;
    };
    return FullyConnectedLayerBuilder;
}());
exports.FullyConnectedLayerBuilder = FullyConnectedLayerBuilder;
var ReLULayerBuilder = (function () {
    function ReLULayerBuilder() {
        this.layerName = 'ReLU';
    }
    ReLULayerBuilder.prototype.getLayerParams = function () {
        return [];
    };
    ReLULayerBuilder.prototype.getOutputShape = function (inputShape) {
        return inputShape;
    };
    ReLULayerBuilder.prototype.addLayer = function (g, network, inputShape, index, weights) {
        return g.relu(network);
    };
    ReLULayerBuilder.prototype.validate = function (inputShape) {
        return null;
    };
    return ReLULayerBuilder;
}());
exports.ReLULayerBuilder = ReLULayerBuilder;
var Convolution2DLayerBuilder = (function () {
    function Convolution2DLayerBuilder() {
        this.layerName = 'Convolution';
    }
    Convolution2DLayerBuilder.prototype.getLayerParams = function () {
        var _this = this;
        return [
            {
                label: 'Field size',
                initialValue: function (inputShape) { return 3; },
                type: 'number',
                min: 1,
                max: 100,
                setValue: function (value) { return _this.fieldSize = value; },
                getValue: function () { return _this.fieldSize; }
            },
            {
                label: 'Stride',
                initialValue: function (inputShape) { return 1; },
                type: 'number',
                min: 1,
                max: 100,
                setValue: function (value) { return _this.stride = value; },
                getValue: function () { return _this.stride; }
            },
            {
                label: 'Zero pad',
                initialValue: function (inputShape) { return 0; },
                type: 'number',
                min: 0,
                max: 100,
                setValue: function (value) { return _this.zeroPad = value; },
                getValue: function () { return _this.zeroPad; }
            },
            {
                label: 'Output depth',
                initialValue: function (inputShape) {
                    return _this.outputDepth != null ? _this.outputDepth : 1;
                },
                type: 'number',
                min: 1,
                max: 1000,
                setValue: function (value) { return _this.outputDepth = value; },
                getValue: function () { return _this.outputDepth; }
            }
        ];
    };
    Convolution2DLayerBuilder.prototype.getOutputShape = function (inputShape) {
        return deeplearn_1.conv_util.computeOutputShape3D(inputShape, this.fieldSize, this.outputDepth, this.stride, this.zeroPad);
    };
    Convolution2DLayerBuilder.prototype.addLayer = function (g, network, inputShape, index, weights) {
        var wShape = [this.fieldSize, this.fieldSize, inputShape[2], this.outputDepth];
        var w;
        var b;
        if (weights != null) {
            w = deeplearn_1.Array4D.new(wShape, weights['W']);
            b = deeplearn_1.Array1D.new(weights['b']);
        }
        else {
            w = deeplearn_1.Array4D.randTruncatedNormal(wShape, 0, 0.1);
            b = deeplearn_1.Array1D.zeros([this.outputDepth]);
        }
        var wTensor = g.variable("conv2d-" + index + "-w", w);
        var bTensor = g.variable("conv2d-" + index + "-b", b);
        return g.conv2d(network, wTensor, bTensor, this.fieldSize, this.outputDepth, this.stride, this.zeroPad);
    };
    Convolution2DLayerBuilder.prototype.validate = function (inputShape) {
        if (inputShape.length !== 3) {
            return ['Input shape must be a Array3D.'];
        }
        return null;
    };
    return Convolution2DLayerBuilder;
}());
exports.Convolution2DLayerBuilder = Convolution2DLayerBuilder;
var MaxPoolLayerBuilder = (function () {
    function MaxPoolLayerBuilder() {
        this.layerName = 'Max pool';
    }
    MaxPoolLayerBuilder.prototype.getLayerParams = function () {
        var _this = this;
        return [
            {
                label: 'Field size',
                initialValue: function (inputShape) { return 3; },
                type: 'number',
                min: 1,
                max: 100,
                setValue: function (value) { return _this.fieldSize = value; },
                getValue: function () { return _this.fieldSize; }
            },
            {
                label: 'Stride',
                initialValue: function (inputShape) { return 1; },
                type: 'number',
                min: 1,
                max: 100,
                setValue: function (value) { return _this.stride = value; },
                getValue: function () { return _this.stride; }
            },
            {
                label: 'Zero pad',
                initialValue: function (inputShape) { return 0; },
                type: 'number',
                min: 0,
                max: 100,
                setValue: function (value) { return _this.zeroPad = value; },
                getValue: function () { return _this.zeroPad; }
            }
        ];
    };
    MaxPoolLayerBuilder.prototype.getOutputShape = function (inputShape) {
        return deeplearn_1.conv_util.computeOutputShape3D(inputShape, this.fieldSize, inputShape[2], this.stride, this.zeroPad);
    };
    MaxPoolLayerBuilder.prototype.addLayer = function (g, network, inputShape, index, weights) {
        return g.maxPool(network, this.fieldSize, this.stride, this.zeroPad);
    };
    MaxPoolLayerBuilder.prototype.validate = function (inputShape) {
        if (inputShape.length !== 3) {
            return ['Input shape must be a Array3D.'];
        }
        return null;
    };
    return MaxPoolLayerBuilder;
}());
exports.MaxPoolLayerBuilder = MaxPoolLayerBuilder;
var ReshapeLayerBuilder = (function () {
    function ReshapeLayerBuilder() {
        this.layerName = 'Reshape';
    }
    ReshapeLayerBuilder.prototype.getLayerParams = function () {
        var _this = this;
        return [{
                label: 'Shape (comma separated)',
                initialValue: function (inputShape) { return inputShape.join(', '); },
                type: 'text',
                setValue: function (value) { return _this.outputShape =
                    value.split(',').map(function (value) { return +value; }); },
                getValue: function () { return _this.outputShape.join(', '); }
            }];
    };
    ReshapeLayerBuilder.prototype.getOutputShape = function (inputShape) {
        return this.outputShape;
    };
    ReshapeLayerBuilder.prototype.addLayer = function (g, network, inputShape, index, weights) {
        return g.reshape(network, this.outputShape);
    };
    ReshapeLayerBuilder.prototype.validate = function (inputShape) {
        var inputSize = deeplearn_1.util.sizeFromShape(inputShape);
        var outputSize = deeplearn_1.util.sizeFromShape(this.outputShape);
        if (inputSize !== outputSize) {
            return [
                "Input size (" + inputSize + ") must match output size (" + outputSize + ")."
            ];
        }
        return null;
    };
    return ReshapeLayerBuilder;
}());
exports.ReshapeLayerBuilder = ReshapeLayerBuilder;
var FlattenLayerBuilder = (function () {
    function FlattenLayerBuilder() {
        this.layerName = 'Flatten';
    }
    FlattenLayerBuilder.prototype.getLayerParams = function () {
        return [];
    };
    FlattenLayerBuilder.prototype.getOutputShape = function (inputShape) {
        return [deeplearn_1.util.sizeFromShape(inputShape)];
    };
    FlattenLayerBuilder.prototype.addLayer = function (g, network, inputShape, index, weights) {
        return g.reshape(network, this.getOutputShape(inputShape));
    };
    FlattenLayerBuilder.prototype.validate = function (inputShape) {
        return null;
    };
    return FlattenLayerBuilder;
}());
exports.FlattenLayerBuilder = FlattenLayerBuilder;
//# sourceMappingURL=layer_builder.js.map