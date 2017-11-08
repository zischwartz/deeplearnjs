"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var deeplearn_1 = require("../deeplearn");
var demo_util = require("../util");
var CHECKPOINT_URL = './drums';
var DECODER_CELL_FORMAT = "decoder/multi_rnn_cell/cell_%d/lstm_cell/";
var forgetBias = deeplearn_1.Scalar.new(1.0);
var presigDivisor = deeplearn_1.Scalar.new(2.0);
var LayerVars = (function () {
    function LayerVars(kernel, bias) {
        this.kernel = kernel;
        this.bias = bias;
    }
    return LayerVars;
}());
function dense(vars, inputs) {
    var weightedResult = math.matMul(inputs, vars.kernel);
    return math.add(weightedResult, vars.bias);
}
var Encoder = (function () {
    function Encoder(lstmFwVars, lstmBwVars, muVars, presigVars) {
        this.lstmFwVars = lstmFwVars;
        this.lstmBwVars = lstmBwVars;
        this.muVars = muVars;
        this.presigVars = presigVars;
        this.zDims = this.muVars.bias.shape[0];
    }
    Encoder.prototype.runLstm = function (inputs, lstmVars, reverse, track) {
        var batchSize = inputs.shape[0];
        var length = inputs.shape[1];
        var outputSize = inputs.shape[2];
        var state = [
            track(deeplearn_1.Array2D.zeros([batchSize, lstmVars.bias.shape[0] / 4])),
            track(deeplearn_1.Array2D.zeros([batchSize, lstmVars.bias.shape[0] / 4]))
        ];
        var lstm = math.basicLSTMCell.bind(math, forgetBias, lstmVars.kernel, lstmVars.bias);
        for (var i = 0; i < length; i++) {
            var index = reverse ? length - 1 - i : i;
            state = lstm(math.slice3D(inputs, [0, index, 0], [batchSize, 1, outputSize]).as2D(batchSize, outputSize), state[0], state[1]);
        }
        return state;
    };
    Encoder.prototype.encode = function (sequence, track) {
        var fwState = this.runLstm(sequence, this.lstmFwVars, false, track);
        var bwState = this.runLstm(sequence, this.lstmBwVars, true, track);
        var finalState = math.concat2D(fwState[1], bwState[1], 1);
        var mu = dense(this.muVars, finalState);
        var presig = dense(this.presigVars, finalState);
        var sigma = math.exp(math.arrayDividedByScalar(presig, presigDivisor));
        var z = math.addStrict(mu, math.multiplyStrict(sigma, track(deeplearn_1.Array2D.randNormal(sigma.shape))));
        return [z, mu, sigma];
    };
    return Encoder;
}());
var Decoder = (function () {
    function Decoder(lstmCellVars, zToInitStateVars, outputProjectVars) {
        this.lstmCellVars = lstmCellVars;
        this.zToInitStateVars = zToInitStateVars;
        this.outputProjectVars = outputProjectVars;
        this.zDims = this.zToInitStateVars.kernel.shape[0];
        this.outputDims = outputProjectVars.bias.shape[0];
    }
    Decoder.prototype.decode = function (z, length, track) {
        var batchSize = z.shape[0];
        var outputSize = this.outputProjectVars.bias.shape[0];
        var lstmCells = [];
        var c = [];
        var h = [];
        var initialStates = math.tanh(dense(this.zToInitStateVars, z));
        var stateOffset = 0;
        for (var i = 0; i < this.lstmCellVars.length; ++i) {
            var lv = this.lstmCellVars[i];
            var stateWidth = lv.bias.shape[0] / 4;
            lstmCells.push(math.basicLSTMCell.bind(math, forgetBias, lv.kernel, lv.bias));
            c.push(math.slice2D(initialStates, [0, stateOffset], [batchSize, stateWidth]));
            stateOffset += stateWidth;
            h.push(math.slice2D(initialStates, [0, stateOffset], [batchSize, stateWidth]));
            stateOffset += stateWidth;
        }
        var samples;
        var nextInput = track(deeplearn_1.Array2D.zeros([batchSize, outputSize]));
        for (var i = 0; i < length; ++i) {
            var output = math.multiRNNCell(lstmCells, math.concat2D(nextInput, z, 1), c, h);
            c = output[0];
            h = output[1];
            var logits = dense(this.outputProjectVars, h[h.length - 1]);
            var timeSamples = math.argMax(logits, 1).as1D();
            samples = i ? math.concat2D(samples, timeSamples.as2D(-1, 1), 1) : timeSamples.as2D(-1, 1);
            nextInput = math.oneHot(timeSamples, outputSize);
        }
        return samples;
    };
    return Decoder;
}());
var isDeviceSupported = demo_util.isWebGLSupported() && !demo_util.isSafari();
var math = new deeplearn_1.NDArrayMathGPU();
if (!isDeviceSupported) {
    document.querySelector('#status').innerHTML =
        'We do not yet support your device. Please try on a desktop ' +
            'computer with Chrome/Firefox, or an Android phone with WebGL support.';
}
else {
    initialize().then(function (encoder_decoder) {
        encodeAndDecode(encoder_decoder[0], encoder_decoder[1]);
    });
}
function initialize() {
    var reader = new deeplearn_1.CheckpointLoader(CHECKPOINT_URL);
    return reader.getAllVariables().then(function (vars) {
        var encLstmFw = new LayerVars(vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel'], vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias']);
        var encLstmBw = new LayerVars(vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel'], vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias']);
        var encMu = new LayerVars(vars['encoder/mu/kernel'], vars['encoder/mu/bias']);
        var encPresig = new LayerVars(vars['encoder/sigma/kernel'], vars['encoder/sigma/bias']);
        var decLstmLayers = [];
        var l = 0;
        while (true) {
            var cell_prefix = DECODER_CELL_FORMAT.replace('%d', l.toString());
            if (!(cell_prefix + 'kernel' in vars)) {
                break;
            }
            decLstmLayers.push(new LayerVars(vars[cell_prefix + 'kernel'], vars[cell_prefix + 'bias']));
            ++l;
        }
        var decZtoInitState = new LayerVars(vars['decoder/z_to_initial_state/kernel'], vars['decoder/z_to_initial_state/bias']);
        var decOutputProjection = new LayerVars(vars['decoder/output_projection/kernel'], vars['decoder/output_projection/bias']);
        return [
            new Encoder(encLstmFw, encLstmBw, encMu, encPresig),
            new Decoder(decLstmLayers, decZtoInitState, decOutputProjection)
        ];
    });
}
var BATCH_SIZE = 25;
var ITERATIONS = 4;
var LENGTH = 32;
console.log('checkpoint: ' + CHECKPOINT_URL);
console.log('batch size: ' + BATCH_SIZE);
console.log('iterations: ' + ITERATIONS);
function encodeAndDecode(encoder, decoder) {
    var teaPot = [71, 0, 73, 0, 75, 0, 76, 0, 78, 0, 1, 0, 83, 0, 0, 0, 80, 0, 0, 0, 83, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0];
    console.log(teaPot);
    var inputs = deeplearn_1.Array3D.zeros([BATCH_SIZE, LENGTH, decoder.outputDims]);
    for (var i = 0; i < inputs.shape[0]; ++i) {
        for (var j = 0; j < inputs.shape[1]; ++j) {
            inputs.set(1, i, j, teaPot[j] + 1);
        }
    }
    var start = Date.now();
    var allLabels = math.scope(function (keep, track) {
        var outputs = encoder.encode(inputs, track);
        var z = outputs[0];
        var allLabels = [];
        for (var i = 0; i < ITERATIONS; ++i) {
            allLabels.push(decoder.decode(z, LENGTH, track));
        }
        return allLabels;
    });
    console.log((Date.now() - start) / 1000.);
    console.log(allLabels[0].getValues());
}
//# sourceMappingURL=loopz.js.map