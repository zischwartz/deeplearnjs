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
var deeplearn_1 = require("./deeplearn");
var polymer_spec_1 = require("./polymer-spec");
var TOP_K = 3;
exports.NDArrayLogitsVisualizerPolymer = polymer_spec_1.PolymerElement({ is: 'ndarray-logits-visualizer', properties: {} });
var NDArrayLogitsVisualizer = (function (_super) {
    __extends(NDArrayLogitsVisualizer, _super);
    function NDArrayLogitsVisualizer() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    NDArrayLogitsVisualizer.prototype.initialize = function (width, height) {
        this.width = width;
        this.logitLabelElements = [];
        this.logitVizElements = [];
        var container = this.querySelector('.logits-container');
        container.style.height = height + "px";
        for (var i = 0; i < TOP_K; i++) {
            var logitContainer = document.createElement('div');
            logitContainer.style.height = height / (TOP_K + 1) + "px";
            logitContainer.style.margin =
                height / ((2 * TOP_K) * (TOP_K + 1)) + "px 0";
            logitContainer.className =
                'single-logit-container ndarray-logits-visualizer';
            var logitLabelElement = document.createElement('div');
            logitLabelElement.className = 'logit-label ndarray-logits-visualizer';
            this.logitLabelElements.push(logitLabelElement);
            var logitVizOuterElement = document.createElement('div');
            logitVizOuterElement.className =
                'logit-viz-outer ndarray-logits-visualizer';
            var logitVisInnerElement = document.createElement('div');
            logitVisInnerElement.className =
                'logit-viz-inner ndarray-logits-visualizer';
            logitVisInnerElement.innerHTML = '&nbsp;';
            logitVizOuterElement.appendChild(logitVisInnerElement);
            this.logitVizElements.push(logitVisInnerElement);
            logitContainer.appendChild(logitLabelElement);
            logitContainer.appendChild(logitVizOuterElement);
            container.appendChild(logitContainer);
        }
    };
    NDArrayLogitsVisualizer.prototype.drawLogits = function (predictedLogits, labelLogits, labelClassNames) {
        var mathCpu = new deeplearn_1.NDArrayMathCPU();
        var labelClass = mathCpu.argMax(labelLogits).get();
        var topk = mathCpu.topK(predictedLogits, TOP_K);
        var topkIndices = topk.indices.getValues();
        var topkValues = topk.values.getValues();
        for (var i = 0; i < topkIndices.length; i++) {
            var index = topkIndices[i];
            this.logitLabelElements[i].innerText =
                labelClassNames ? labelClassNames[index] : index.toString();
            this.logitLabelElements[i].style.width =
                labelClassNames != null ? '100px' : '20px';
            this.logitVizElements[i].style.backgroundColor = index === labelClass ?
                'rgba(120, 185, 50, .84)' :
                'rgba(220, 10, 10, 0.84)';
            this.logitVizElements[i].style.width =
                Math.floor(100 * topkValues[i]) + "%";
            this.logitVizElements[i].innerText =
                (100 * topkValues[i]).toFixed(1) + "%";
        }
    };
    return NDArrayLogitsVisualizer;
}(exports.NDArrayLogitsVisualizerPolymer));
exports.NDArrayLogitsVisualizer = NDArrayLogitsVisualizer;
document.registerElement(NDArrayLogitsVisualizer.prototype.is, NDArrayLogitsVisualizer);
//# sourceMappingURL=ndarray-logits-visualizer.js.map