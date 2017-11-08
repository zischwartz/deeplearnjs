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
require("../demo-header");
require("../demo-footer");
var polymer_spec_1 = require("../polymer-spec");
var cppn_1 = require("./cppn");
var CANVAS_UPSCALE_FACTOR = 3;
var MAT_WIDTH = 30;
var WEIGHTS_STDEV = .6;
var NNArtPolymer = polymer_spec_1.PolymerElement({
    is: 'nn-art',
    properties: {
        colorModeNames: Array,
        selectedColorModeName: String,
        activationFunctionNames: Array,
        selectedActivationFunctionName: String
    }
});
var NNArt = (function (_super) {
    __extends(NNArt, _super);
    function NNArt() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    NNArt.prototype.ready = function () {
        var _this = this;
        this.inferenceCanvas =
            this.querySelector('#inference');
        this.cppn = new cppn_1.CPPN(this.inferenceCanvas);
        this.inferenceCanvas.style.width =
            this.inferenceCanvas.width * CANVAS_UPSCALE_FACTOR + "px";
        this.inferenceCanvas.style.height =
            this.inferenceCanvas.height * CANVAS_UPSCALE_FACTOR + "px";
        this.colorModeNames = ['rgb', 'rgba', 'hsv', 'hsva', 'yuv', 'yuva', 'bw'];
        this.selectedColorModeName = 'rgb';
        this.cppn.setColorMode(this.selectedColorModeName);
        this.querySelector('#color-mode-dropdown').addEventListener('iron-activate', function (event) {
            _this.selectedColorModeName = event.detail.selected;
            _this.cppn.setColorMode(_this.selectedColorModeName);
        });
        this.activationFunctionNames = ['tanh', 'sin', 'relu', 'step'];
        this.selectedActivationFunctionName = 'tanh';
        this.cppn.setActivationFunction(this.selectedActivationFunctionName);
        this.querySelector('#activation-function-dropdown').addEventListener('iron-activate', function (event) {
            _this.selectedActivationFunctionName = event.detail.selected;
            _this.cppn.setActivationFunction(_this.selectedActivationFunctionName);
        });
        var layersSlider = this.querySelector('#layers-slider');
        var layersCountElement = this.querySelector('#layers-count');
        layersSlider.addEventListener('immediate-value-changed', function (event) {
            _this.numLayers = parseInt(event.target.immediateValue, 10);
            layersCountElement.innerText = _this.numLayers.toString();
            _this.cppn.setNumLayers(_this.numLayers);
        });
        this.numLayers = parseInt(layersSlider.value, 10);
        layersCountElement.innerText = this.numLayers.toString();
        this.cppn.setNumLayers(this.numLayers);
        var z1Slider = this.querySelector('#z1-slider');
        z1Slider.addEventListener('immediate-value-changed', function (event) {
            _this.z1Scale = parseInt(event.target.immediateValue, 10);
            _this.cppn.setZ1Scale(convertZScale(_this.z1Scale));
        });
        this.z1Scale = parseInt(z1Slider.value, 10);
        this.cppn.setZ1Scale(convertZScale(this.z1Scale));
        var z2Slider = this.querySelector('#z2-slider');
        z2Slider.addEventListener('immediate-value-changed', function (event) {
            _this.z2Scale = parseInt(event.target.immediateValue, 10);
            _this.cppn.setZ2Scale(convertZScale(_this.z2Scale));
        });
        this.z2Scale = parseInt(z2Slider.value, 10);
        this.cppn.setZ2Scale(convertZScale(this.z2Scale));
        var randomizeButton = this.querySelector('#random');
        randomizeButton.addEventListener('click', function () {
            _this.cppn.generateWeights(MAT_WIDTH, WEIGHTS_STDEV);
        });
        this.cppn.generateWeights(MAT_WIDTH, WEIGHTS_STDEV);
        this.cppn.start();
    };
    return NNArt;
}(NNArtPolymer));
function convertZScale(z) {
    return (103 - z);
}
document.registerElement(NNArt.prototype.is, NNArt);
//# sourceMappingURL=nn-art.js.map