"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var cppn_1 = require("../nn-art/cppn");
var demo_util = require("../util");
var inferenceCanvas = document.querySelector('#inference');
var isDeviceDisabled = demo_util.isSafari() && demo_util.isMobile();
var enableCPPN = demo_util.isWebGLSupported() && !isDeviceDisabled;
if (enableCPPN) {
    startCPPN();
}
else {
    document.getElementById('disabled-demo-overlay').style.display = '';
    inferenceCanvas.style.display = 'none';
}
function startCPPN() {
    var DEFAULT_Z_SCALE = 1;
    var NUM_NEURONS = 30;
    var DEFAULT_NUM_LAYERS = 2;
    var WEIGHTS_STDEV = 0.6;
    var cppn = new cppn_1.CPPN(inferenceCanvas);
    cppn.setActivationFunction('tanh');
    cppn.setColorMode('rgb');
    cppn.setNumLayers(DEFAULT_NUM_LAYERS);
    cppn.setZ1Scale(convertZScale(DEFAULT_Z_SCALE));
    cppn.setZ2Scale(convertZScale(DEFAULT_Z_SCALE));
    cppn.generateWeights(NUM_NEURONS, WEIGHTS_STDEV);
    cppn.start();
    var currentColorElement = document.querySelector('#colormode');
    document.querySelector('#color-selector')
        .addEventListener('click', function (event) {
        var colorMode = event.target.getAttribute('data-val');
        currentColorElement.value = colorMode;
        cppn.setColorMode(colorMode);
    });
    var currentActivationFnElement = document.querySelector('#activation-fn');
    document.querySelector('#activation-selector')
        .addEventListener('click', function (event) {
        var activationFn = event.target.getAttribute('data-val');
        currentActivationFnElement.value = activationFn;
        cppn.setActivationFunction(activationFn);
    });
    var layersSlider = document.querySelector('#layers-slider');
    var layersCountElement = document.querySelector('#layers-count');
    layersSlider.addEventListener('input', function (event) {
        var numLayers = parseInt(event.target.value, 10);
        layersCountElement.innerText = numLayers.toString();
        cppn.setNumLayers(numLayers);
    });
    layersCountElement.innerText = DEFAULT_NUM_LAYERS.toString();
    var z1Slider = document.querySelector('#z1-slider');
    z1Slider.addEventListener('input', function (event) {
        var z1Scale = parseInt(event.target.value, 10);
        cppn.setZ1Scale(convertZScale(z1Scale));
    });
    var z2Slider = document.querySelector('#z2-slider');
    z2Slider.addEventListener('input', function (event) {
        var z2Scale = parseInt(event.target.value, 10);
        cppn.setZ2Scale(convertZScale(z2Scale));
    });
    var randomizeButton = document.querySelector('#random');
    randomizeButton.addEventListener('click', function () {
        cppn.generateWeights(NUM_NEURONS, WEIGHTS_STDEV);
        if (!playing) {
            cppn.start();
            requestAnimationFrame(function () {
                cppn.stopInferenceLoop();
            });
        }
    });
    var playing = true;
    var toggleButton = document.querySelector('#toggle');
    toggleButton.addEventListener('click', function () {
        playing = !playing;
        if (playing) {
            toggleButton.innerHTML = 'STOP';
            cppn.start();
        }
        else {
            toggleButton.innerHTML = 'START';
            cppn.stopInferenceLoop();
        }
    });
    var canvasOnScreenLast = true;
    var scrollEventScheduled = false;
    var mainElement = document.querySelector('main');
    mainElement.addEventListener('scroll', function () {
        if (!scrollEventScheduled) {
            window.requestAnimationFrame(function () {
                var canvasOnScreen = isCanvasOnScreen();
                if (canvasOnScreen !== canvasOnScreenLast) {
                    if (canvasOnScreen) {
                        if (playing) {
                            cppn.start();
                        }
                    }
                    else {
                        cppn.stopInferenceLoop();
                    }
                    canvasOnScreenLast = canvasOnScreen;
                }
                scrollEventScheduled = false;
            });
        }
        scrollEventScheduled = true;
    });
    function isCanvasOnScreen() {
        return mainElement.scrollTop < inferenceCanvas.offsetHeight;
    }
    function convertZScale(z) {
        return (103 - z);
    }
}
//# sourceMappingURL=homepage.js.map