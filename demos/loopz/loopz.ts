/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// tslint:disable-next-line:max-line-length
import { Array1D, Array2D, Array3D, CheckpointLoader, NDArray, NDArrayMathGPU, Scalar } from '../deeplearn';
import { LSTMCell } from '../../src/math/math'
import * as demo_util from '../util';

const CHECKPOINT_URL = './drums';
const DECODER_CELL_FORMAT = "decoder/multi_rnn_cell/cell_%d/lstm_cell/"

const forgetBias = Scalar.new(1.0);
const presigDivisor = Scalar.new(2.0);

class LayerVars {
  kernel: Array2D;
  bias: Array1D;
  constructor(kernel: Array2D, bias: Array1D) {
    this.kernel = kernel;
    this.bias = bias;
  }
}

function dense(vars: LayerVars, inputs: Array2D) {
  const weightedResult = math.matMul(inputs, vars.kernel);
  return math.add(weightedResult, vars.bias) as Array2D;
}

class Encoder {
  lstmFwVars: LayerVars;
  lstmBwVars: LayerVars;
  muVars: LayerVars;
  presigVars: LayerVars;
  zDims: number;

  constructor(lstmFwVars: LayerVars, lstmBwVars: LayerVars, muVars: LayerVars, presigVars: LayerVars) {
    this.lstmFwVars = lstmFwVars;
    this.lstmBwVars = lstmBwVars;
    this.muVars = muVars;
    this.presigVars = presigVars;
    this.zDims = this.muVars.bias.shape[0];
  }

  private runLstm(inputs: Array3D, lstmVars: LayerVars, reverse: boolean, track: Function) {
    const batchSize = inputs.shape[0];
    const length = inputs.shape[1];
    const outputSize = inputs.shape[2];
    let state: Array2D[] = [
      track(Array2D.zeros([batchSize, lstmVars.bias.shape[0] / 4])),
      track(Array2D.zeros([batchSize, lstmVars.bias.shape[0] / 4]))
    ]
    let lstm = math.basicLSTMCell.bind(math, forgetBias, lstmVars.kernel, lstmVars.bias);
    for (let i = 0; i < length; i++) {
      let index = reverse ? length - 1 - i : i;
      state = lstm(
        math.slice3D(inputs, [0, index, 0], [batchSize, 1, outputSize]).as2D(batchSize, outputSize),
        state[0], state[1]);
    }
    return state;
  }

  encode(sequence: Array3D, track: Function) {
    const fwState = this.runLstm(sequence, this.lstmFwVars, false, track);
    const bwState = this.runLstm(sequence, this.lstmBwVars, true, track);
    const finalState = math.concat2D(fwState[1], bwState[1], 1)
    const mu = dense(this.muVars, finalState);
    const presig = dense(this.presigVars, finalState);
    const sigma = math.exp(math.arrayDividedByScalar(presig, presigDivisor));
    const z = math.addStrict(
      mu,
      math.multiplyStrict(sigma, track(Array2D.randNormal(sigma.shape)))) as Array2D;
    return [z, mu, sigma];
  }
}

class Decoder {
  lstmCellVars: LayerVars[];
  zToInitStateVars: LayerVars;
  outputProjectVars: LayerVars;
  zDims: number;
  outputDims: number;

  constructor(lstmCellVars: LayerVars[], zToInitStateVars: LayerVars, outputProjectVars: LayerVars) {
    this.lstmCellVars = lstmCellVars;
    this.zToInitStateVars = zToInitStateVars;
    this.outputProjectVars = outputProjectVars;
    this.zDims = this.zToInitStateVars.kernel.shape[0];
    this.outputDims = outputProjectVars.bias.shape[0];
  }

  decode(z: Array2D, length: number, track: Function) {
    const batchSize = z.shape[0];
    const outputSize = this.outputProjectVars.bias.shape[0];

    // Initialize LSTMCells.
    let lstmCells: LSTMCell[] = []
    let c: Array2D[] = [];
    let h: Array2D[] = [];
    const initialStates = math.tanh(dense(this.zToInitStateVars, z));
    let stateOffset = 0;
    for (let i = 0; i < this.lstmCellVars.length; ++i) {
      const lv = this.lstmCellVars[i];
      const stateWidth = lv.bias.shape[0] / 4;
      lstmCells.push(math.basicLSTMCell.bind(math, forgetBias, lv.kernel, lv.bias))
      c.push(math.slice2D(initialStates, [0, stateOffset], [batchSize, stateWidth]));
      stateOffset += stateWidth;
      h.push(math.slice2D(initialStates, [0, stateOffset], [batchSize, stateWidth]));
      stateOffset += stateWidth;
    }

    // Generate samples.
    let samples: Array2D;
    let nextInput: Array2D = track(Array2D.zeros([batchSize, outputSize]));
    for (let i = 0; i < length; ++i) {
      let output = math.multiRNNCell(lstmCells, math.concat2D(nextInput, z, 1), c, h);
      c = output[0];
      h = output[1];
      const logits = dense(this.outputProjectVars, h[h.length - 1]);

      let timeSamples = math.argMax(logits, 1).as1D();
      samples = i ? math.concat2D(samples, timeSamples.as2D(-1, 1), 1) : timeSamples.as2D(-1, 1);
      nextInput = math.oneHot(timeSamples, outputSize);
    }
    return samples;
  }

}

const isDeviceSupported = demo_util.isWebGLSupported() && !demo_util.isSafari();
const math = new NDArrayMathGPU();

if (!isDeviceSupported) {
  document.querySelector('#status').innerHTML =
    'We do not yet support your device. Please try on a desktop ' +
    'computer with Chrome/Firefox, or an Android phone with WebGL support.';
} else {
  initialize().then((encoder_decoder: [Encoder, Decoder]) => {
    encodeAndDecode(encoder_decoder[0], encoder_decoder[1]);
  });
}

function initialize() {
  const reader = new CheckpointLoader(CHECKPOINT_URL);
  return reader.getAllVariables().then(
    (vars: { [varName: string]: NDArray }) => {
      const encLstmFw = new LayerVars(
        vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/kernel'] as Array2D,
        vars['encoder/cell_0/bidirectional_rnn/fw/multi_rnn_cell/cell_0/lstm_cell/bias'] as Array1D);
      const encLstmBw = new LayerVars(
        vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/kernel'] as Array2D,
        vars['encoder/cell_0/bidirectional_rnn/bw/multi_rnn_cell/cell_0/lstm_cell/bias'] as Array1D);
      const encMu = new LayerVars(
        vars['encoder/mu/kernel'] as Array2D,
        vars['encoder/mu/bias'] as Array1D);
      const encPresig = new LayerVars(
        vars['encoder/sigma/kernel'] as Array2D,
        vars['encoder/sigma/bias'] as Array1D);

      let decLstmLayers: LayerVars[] = [];
      let l = 0;
      while (true) {
        const cell_prefix = DECODER_CELL_FORMAT.replace('%d', l.toString());
        if (!(cell_prefix + 'kernel' in vars)) {
          break;
        }
        decLstmLayers.push(new LayerVars(
          vars[cell_prefix + 'kernel'] as Array2D,
          vars[cell_prefix + 'bias'] as Array1D));
        ++l;
      }

      const decZtoInitState = new LayerVars(
        vars['decoder/z_to_initial_state/kernel'] as Array2D,
        vars['decoder/z_to_initial_state/bias'] as Array1D);
      const decOutputProjection = new LayerVars(
        vars['decoder/output_projection/kernel'] as Array2D,
        vars['decoder/output_projection/bias'] as Array1D);
      return [
        new Encoder(encLstmFw, encLstmBw, encMu, encPresig),
        new Decoder(decLstmLayers, decZtoInitState, decOutputProjection)];
    })
}

const BATCH_SIZE = 25;
const ITERATIONS = 4;
const LENGTH = 32;
console.log('checkpoint: ' + CHECKPOINT_URL);
console.log('batch size: ' + BATCH_SIZE);
console.log('iterations: ' + ITERATIONS);
function encodeAndDecode(encoder: Encoder, decoder: Decoder) {
  const teaPot = [71, 0, 73, 0, 75, 0, 76, 0, 78, 0, 1, 0, 83, 0, 0, 0, 80, 0, 0, 0, 83, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0]
  console.log(teaPot);
  let inputs = Array3D.zeros([BATCH_SIZE, LENGTH, decoder.outputDims])
  for (let i = 0; i < inputs.shape[0]; ++i) {
    for (let j = 0; j < inputs.shape[1]; ++j) {
      inputs.set(1, i, j, teaPot[j] + 1);
    }
  }
  //const mu = Array2D.randNormal([BATCH_SIZE, encoder.zDims]);
  var start = Date.now()

  let allLabels = math.scope((keep, track) => {
    const outputs = encoder.encode(inputs, track)
    const z = outputs[0];
    let allLabels: Array2D[] = [];
    for (let i = 0; i < ITERATIONS; ++i) {
      allLabels.push(decoder.decode(z, LENGTH, track))
      // sampledLabels.forEach((sample: Array2D) => console.log(sample.getValues()));
    }
    return allLabels;
  });
  console.log((Date.now() - start) / 1000.)
  console.log(allLabels[0].getValues())
  // document.getElementById('results').innerHTML = '' + result.getValues();
}
