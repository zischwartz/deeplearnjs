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
var polymer_spec_1 = require("../polymer-spec");
var layer_builder = require("./layer_builder");
var model_builder_util = require("./model_builder_util");
exports.ModelLayerPolymer = polymer_spec_1.PolymerElement({
    is: 'model-layer',
    properties: {
        layerName: String,
        inputShapeDisplay: String,
        outputShapeDisplay: String,
        isStatic: { type: Boolean, value: false },
        layerNames: Array,
        selectedLayerName: String,
        hasError: { type: Boolean, value: false },
        errorMessages: Array,
    }
});
var ModelLayer = (function (_super) {
    __extends(ModelLayer, _super);
    function ModelLayer() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    ModelLayer.prototype.initialize = function (modelBuilder, inputShape) {
        var _this = this;
        this.modelBuilder = modelBuilder;
        this.paramContainer =
            this.querySelector('.param-container');
        this.layerNames = [
            'Fully connected', 'ReLU', 'Convolution', 'Max pool', 'Reshape', 'Flatten'
        ];
        this.inputShape = inputShape;
        this.buildParamsUI('Fully connected', this.inputShape);
        this.querySelector('.dropdown-content')
            .addEventListener('iron-activate', function (event) {
            _this.buildParamsUI(event.detail.selected, _this.inputShape);
        });
        this.querySelector('#remove-layer').addEventListener('click', function (event) {
            modelBuilder.removeLayer(_this);
        });
    };
    ModelLayer.prototype.setInputShape = function (shape) {
        this.inputShape = shape;
        this.inputShapeDisplay =
            model_builder_util.getDisplayShape(this.inputShape);
        var errors = [];
        var validationErrors = this.layerBuilder.validate(this.inputShape);
        if (validationErrors != null) {
            for (var i = 0; i < validationErrors.length; i++) {
                errors.push('Error: ' + validationErrors[i]);
            }
        }
        try {
            this.outputShape = this.layerBuilder.getOutputShape(this.inputShape);
        }
        catch (e) {
            errors.push(e);
        }
        this.outputShapeDisplay =
            model_builder_util.getDisplayShape(this.outputShape);
        if (errors.length > 0) {
            this.hasError = true;
            this.errorMessages = errors;
        }
        else {
            this.hasError = false;
            this.errorMessages = [];
        }
        return this.outputShape;
    };
    ModelLayer.prototype.isValid = function () {
        return !this.hasError;
    };
    ModelLayer.prototype.getOutputShape = function () {
        return this.outputShape;
    };
    ModelLayer.prototype.addLayer = function (g, network, index, weights) {
        return this.layerBuilder.addLayer(g, network, this.inputShape, index, weights);
    };
    ModelLayer.prototype.buildParamsUI = function (layerName, inputShape, layerBuilderJson) {
        this.selectedLayerName = layerName;
        this.layerBuilder =
            layer_builder.getLayerBuilder(layerName, layerBuilderJson);
        this.paramContainer.innerHTML = '';
        var layerParams = this.layerBuilder.getLayerParams();
        for (var i = 0; i < layerParams.length; i++) {
            var initialValue = layerBuilderJson != null ?
                layerParams[i].getValue() :
                layerParams[i].initialValue(inputShape);
            this.addParamField(layerParams[i].label, initialValue, layerParams[i].setValue, layerParams[i].type, layerParams[i].min, layerParams[i].max);
        }
        this.modelBuilder.layerParamChanged();
    };
    ModelLayer.prototype.loadParamsFromLayerBuilder = function (inputShape, layerBuilderJson) {
        this.buildParamsUI(layerBuilderJson.layerName, inputShape, layerBuilderJson);
    };
    ModelLayer.prototype.addParamField = function (label, initialValue, setValue, type, min, max) {
        var _this = this;
        var input = document.createElement('paper-input');
        input.setAttribute('always-float-label', 'true');
        input.setAttribute('label', label);
        input.setAttribute('value', initialValue.toString());
        input.setAttribute('type', type);
        if (type === 'number') {
            input.setAttribute('min', min.toString());
            input.setAttribute('max', max.toString());
        }
        input.className = 'param-input';
        this.paramContainer.appendChild(input);
        input.addEventListener('input', function (event) {
            if (type === 'number') {
                setValue(event.target.valueAsNumber);
            }
            else {
                setValue(event.target.value);
            }
            _this.modelBuilder.layerParamChanged();
        });
        setValue(initialValue);
    };
    return ModelLayer;
}(exports.ModelLayerPolymer));
exports.ModelLayer = ModelLayer;
document.registerElement(ModelLayer.prototype.is, ModelLayer);
//# sourceMappingURL=model-layer.js.map