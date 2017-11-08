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
var polymer_spec_1 = require("./polymer-spec");
exports.NDArrayImageVisualizerPolymer = polymer_spec_1.PolymerElement({ is: 'ndarray-image-visualizer', properties: {} });
var NDArrayImageVisualizer = (function (_super) {
    __extends(NDArrayImageVisualizer, _super);
    function NDArrayImageVisualizer() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    NDArrayImageVisualizer.prototype.ready = function () {
        this.canvas = this.querySelector('#canvas');
        this.canvas.width = 0;
        this.canvas.height = 0;
        this.canvasContext =
            this.canvas.getContext('2d');
        this.canvas.style.display = 'none';
    };
    NDArrayImageVisualizer.prototype.setShape = function (shape) {
        this.canvas.width = shape[1];
        this.canvas.height = shape[0];
    };
    NDArrayImageVisualizer.prototype.setSize = function (width, height) {
        this.canvas.style.width = width + "px";
        this.canvas.style.height = height + "px";
    };
    NDArrayImageVisualizer.prototype.saveImageDataFromNDArray = function (ndarray) {
        this.imageData = this.canvasContext.createImageData(this.canvas.width, this.canvas.height);
        if (ndarray.shape[2] === 1) {
            this.drawGrayscaleImageData(ndarray);
        }
        else if (ndarray.shape[2] === 3) {
            this.drawRGBImageData(ndarray);
        }
    };
    NDArrayImageVisualizer.prototype.drawRGBImageData = function (ndarray) {
        var pixelOffset = 0;
        for (var i = 0; i < ndarray.shape[0]; i++) {
            for (var j = 0; j < ndarray.shape[1]; j++) {
                this.imageData.data[pixelOffset++] = ndarray.get(i, j, 0);
                this.imageData.data[pixelOffset++] = ndarray.get(i, j, 1);
                this.imageData.data[pixelOffset++] = ndarray.get(i, j, 2);
                this.imageData.data[pixelOffset++] = 255;
            }
        }
    };
    NDArrayImageVisualizer.prototype.drawGrayscaleImageData = function (ndarray) {
        var pixelOffset = 0;
        for (var i = 0; i < ndarray.shape[0]; i++) {
            for (var j = 0; j < ndarray.shape[1]; j++) {
                var value = ndarray.get(i, j, 0);
                this.imageData.data[pixelOffset++] = value;
                this.imageData.data[pixelOffset++] = value;
                this.imageData.data[pixelOffset++] = value;
                this.imageData.data[pixelOffset++] = 255;
            }
        }
    };
    NDArrayImageVisualizer.prototype.draw = function () {
        this.canvas.style.display = '';
        this.canvasContext.putImageData(this.imageData, 0, 0);
    };
    return NDArrayImageVisualizer;
}(exports.NDArrayImageVisualizerPolymer));
exports.NDArrayImageVisualizer = NDArrayImageVisualizer;
document.registerElement(NDArrayImageVisualizer.prototype.is, NDArrayImageVisualizer);
//# sourceMappingURL=ndarray-image-visualizer.js.map