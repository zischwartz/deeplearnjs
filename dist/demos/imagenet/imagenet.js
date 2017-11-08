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
require("../demo-header");
require("../demo-footer");
var deeplearn_1 = require("../deeplearn");
var imagenet_util = require("../models/imagenet_util");
var squeezenet_1 = require("../models/squeezenet");
var polymer_spec_1 = require("../polymer-spec");
exports.ImagenetDemoPolymer = polymer_spec_1.PolymerElement({
    is: 'imagenet-demo',
    properties: {
        layerNames: Array,
        selectedLayerName: String,
        inputNames: Array,
        selectedInputName: String
    }
});
var TOP_K_CLASSES = 5;
var INPUT_NAMES = ['cat', 'dog1', 'dog2', 'beerbottle', 'piano', 'saxophone'];
var ImagenetDemo = (function (_super) {
    __extends(ImagenetDemo, _super);
    function ImagenetDemo() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    ImagenetDemo.prototype.ready = function () {
        var _this = this;
        this.inferenceCanvas =
            this.querySelector('#inference-canvas');
        this.staticImgElement =
            this.querySelector('#staticImg');
        this.webcamVideoElement =
            this.querySelector('#webcamVideo');
        this.layerNames = [];
        this.selectedLayerName = 'conv_1';
        var inputDropdown = this.querySelector('#input-dropdown');
        inputDropdown.addEventListener('iron-activate', function (event) {
            var selectedInputName = event.detail.selected;
            if (selectedInputName === 'webcam') {
                _this.webcamVideoElement.style.display = '';
                _this.staticImgElement.style.display = 'none';
            }
            else {
                _this.webcamVideoElement.style.display = 'none';
                _this.staticImgElement.style.display = '';
            }
            _this.staticImgElement.src = "images/" + event.detail.selected + ".jpg";
        });
        var navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
            navigatorAny.msGetUserMedia;
        if (navigator.getUserMedia) {
            navigator.getUserMedia({ video: true }, function (stream) {
                _this.webcamVideoElement.src = window.URL.createObjectURL(stream);
                _this.initWithWebcam();
            }, function (error) {
                console.log(error);
                _this.initWithoutWebcam();
            });
        }
        else {
            this.initWithoutWebcam();
        }
        this.gl = deeplearn_1.gpgpu_util.createWebGLContext(this.inferenceCanvas);
        this.gpgpu = new deeplearn_1.GPGPUContext(this.gl);
        this.math = new deeplearn_1.NDArrayMathGPU(this.gpgpu);
        this.mathCPU = new deeplearn_1.NDArrayMathCPU();
        this.squeezeNet = new squeezenet_1.SqueezeNet(this.math);
        this.squeezeNet.loadVariables().then(function () {
            requestAnimationFrame(function () { return _this.animate(); });
        });
        this.renderGrayscaleChannelsCollageShader =
            imagenet_util.getRenderGrayscaleChannelsCollageShader(this.gpgpu);
    };
    ImagenetDemo.prototype.initWithoutWebcam = function () {
        this.inputNames = INPUT_NAMES;
        this.selectedInputName = 'cat';
        this.staticImgElement.src = 'images/cat.jpg';
        this.webcamVideoElement.style.display = 'none';
        this.staticImgElement.style.display = '';
        if (location.protocol !== 'https:') {
            this.querySelector('#ssl-message').style.display =
                'block';
        }
        this.querySelector('#webcam-message').style.display =
            'block';
    };
    ImagenetDemo.prototype.initWithWebcam = function () {
        var inputNames = INPUT_NAMES.slice();
        inputNames.unshift('webcam');
        this.inputNames = inputNames;
        this.selectedInputName = 'webcam';
    };
    ImagenetDemo.prototype.animate = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            var startTime, isWebcam;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        startTime = performance.now();
                        isWebcam = this.selectedInputName === 'webcam';
                        return [4, this.math.scope(function (keep, track) { return __awaiter(_this, void 0, void 0, function () {
                                var image, inferenceResult, namedActivations, topClassesToProbability, count, className, endTime, elapsed, activationNDArray, maxValues, minValues, imagesPerRow, numRows;
                                return __generator(this, function (_a) {
                                    switch (_a.label) {
                                        case 0:
                                            image = track(deeplearn_1.Array3D.fromPixels(isWebcam ? this.webcamVideoElement : this.staticImgElement));
                                            inferenceResult = this.squeezeNet.infer(image);
                                            namedActivations = inferenceResult.namedActivations;
                                            this.layerNames = Object.keys(namedActivations);
                                            return [4, this.squeezeNet.getTopKClasses(inferenceResult.logits, TOP_K_CLASSES)];
                                        case 1:
                                            topClassesToProbability = _a.sent();
                                            count = 0;
                                            for (className in topClassesToProbability) {
                                                if (!(className in topClassesToProbability)) {
                                                    continue;
                                                }
                                                document.getElementById("class" + count).innerHTML = className;
                                                document.getElementById("prob" + count).innerHTML =
                                                    (Math.floor(1000 * topClassesToProbability[className]) / 1000)
                                                        .toString();
                                                count++;
                                            }
                                            endTime = performance.now();
                                            elapsed = Math.floor(1000 * (endTime - startTime)) / 1000;
                                            this.querySelector('#totalTime').innerHTML =
                                                "last inference time: " + elapsed + " ms";
                                            activationNDArray = namedActivations[this.selectedLayerName];
                                            maxValues = this.math.maxPool(activationNDArray, activationNDArray.shape[1], activationNDArray.shape[1], 0);
                                            minValues = this.math.minPool(activationNDArray, activationNDArray.shape[1], activationNDArray.shape[1], 0);
                                            imagesPerRow = Math.ceil(Math.sqrt(activationNDArray.shape[2]));
                                            numRows = Math.ceil(activationNDArray.shape[2] / imagesPerRow);
                                            this.inferenceCanvas.width = imagesPerRow * activationNDArray.shape[0];
                                            this.inferenceCanvas.height = numRows * activationNDArray.shape[0];
                                            imagenet_util.renderGrayscaleChannelsCollage(this.gpgpu, this.renderGrayscaleChannelsCollageShader, activationNDArray.getTexture(), minValues.getTexture(), maxValues.getTexture(), activationNDArray.getTextureShapeRC(), activationNDArray.shape[0], activationNDArray.shape[2], this.inferenceCanvas.width, numRows);
                                            return [2];
                                    }
                                });
                            }); })];
                    case 1:
                        _a.sent();
                        requestAnimationFrame(function () { return _this.animate(); });
                        return [2];
                }
            });
        });
    };
    return ImagenetDemo;
}(exports.ImagenetDemoPolymer));
exports.ImagenetDemo = ImagenetDemo;
document.registerElement(ImagenetDemo.prototype.is, ImagenetDemo);
//# sourceMappingURL=imagenet.js.map