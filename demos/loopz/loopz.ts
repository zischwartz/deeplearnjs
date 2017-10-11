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
import { Array1D, Array2D, CheckpointLoader, NDArray, NDArrayMathGPU, Scalar } from '../deeplearn';
import { LSTMCell } from '../../src/math/math'
import * as demo_util from '../util';

const forgetBias = Scalar.new(1.0);
const presigDivisor = Scalar.new(2.0)

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

  constructor(lstmFwVars: LayerVars, lstmBwVars: LayerVars, muVars: LayerVars, presigVars: LayerVars) {
    this.lstmFwVars = lstmFwVars;
    this.lstmBwVars = lstmBwVars;
    this.muVars = muVars;
    this.presigVars = presigVars;
  }

  private runLstm(inputs: Array2D, lstmVars: LayerVars, reverse: boolean, track: Function) {
    let state = [
      track(Array2D.zeros([1, lstmVars.bias.shape[0] / 4])),
      track(Array2D.zeros([1, lstmVars.bias.shape[0] / 4]))
    ]
    let lstm = math.basicLSTMCell.bind(math, forgetBias, lstmVars.kernel, lstmVars.bias);
    for (let i = 0; i < inputs.shape[0]; i++) {
      let index = reverse ? inputs.shape[0] - 1 - i : i;
      state = lstm(
        math.slice2D(inputs, [index, 0], [1, inputs.shape[1]]), state[0], state[1]);
    }
    return state;
  }

  encode(sequence: Array2D, track: Function) {
    const fw_state = this.runLstm(sequence, this.lstmFwVars, false, track);
    const bw_state = this.runLstm(sequence, this.lstmBwVars, true, track);
    const final_state = math.concat2D(fw_state[1], bw_state[1], 1)
    const mu = dense(this.muVars, final_state);
    const presig = dense(this.presigVars, final_state);
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

  constructor(lstmCellVars: LayerVars[], zToInitStateVars: LayerVars, outputProjectVars: LayerVars) {
    this.lstmCellVars = lstmCellVars;
    this.zToInitStateVars = zToInitStateVars;
    this.outputProjectVars = outputProjectVars;
  }

  decode(z: Array2D, length: Number, track: Function) {
    const batchSize = z.shape[0];
    const eventDepth = this.outputProjectVars.bias.shape[0];

    // Initialize LSTMCells.
    let lstmCells: LSTMCell[] = []
    let c: Array2D[] = [];
    let h: Array2D[] = [];
    const initial_states = dense(this.zToInitStateVars, z);
    let stateOffset = 0;
    for (let i = 0; i < this.lstmCellVars.length; ++i) {
      const lv = this.lstmCellVars[i];
      const stateWidth = lv.bias.shape[0] / 4;
      lstmCells.push(math.basicLSTMCell.bind(math, forgetBias, lv.kernel, lv.bias))
      c.push(math.slice2D(initial_states, [0, stateOffset], [batchSize, stateWidth]));
      stateOffset += stateWidth;
      h.push(math.slice2D(initial_states, [0, stateOffset], [batchSize, stateWidth]));
      stateOffset += stateWidth;
    }

    // Generate samples.
    let samples: Array2D[] = [];
    let nextInput = track(Array2D.zeros([1, eventDepth]));
    for (let i = 0; i < length; ++i) {
      let output = math.multiRNNCell(lstmCells, math.concat2D(nextInput, z, 1), c, h);
      c = output[0];
      h = output[1];
      const logits = dense(this.outputProjectVars, h[h.length - 1]);

      // Add batching support.
      const softmax = math.softmax(logits.as1D());
      // const sample = math.multinomial(softmax, 1);
      const sample = math.argMax(softmax).as1D();
      nextInput = math.oneHot(sample, eventDepth).as2D(1, -1);
      samples.push(sample.as2D(1, -1));
    }
    return samples;
  }

}

const CHECKPOINT_URL = '.';

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
      const decLstm0 = new LayerVars(
        vars['decoder/multi_rnn_cell/cell_0/lstm_cell/kernel'] as Array2D,
        vars['decoder/multi_rnn_cell/cell_0/lstm_cell/bias'] as Array1D);
      const decLstm1 = new LayerVars(
        vars['decoder/multi_rnn_cell/cell_1/lstm_cell/kernel'] as Array2D,
        vars['decoder/multi_rnn_cell/cell_1/lstm_cell/bias'] as Array1D);
      const decZtoInitState = new LayerVars(
        vars['decoder/z_to_initial_state/kernel'] as Array2D,
        vars['decoder/z_to_initial_state/bias'] as Array1D);
      const decOutputProjection = new LayerVars(
        vars['decoder/output_projection/kernel'] as Array2D,
        vars['decoder/output_projection/bias'] as Array1D);
      return [
        new Encoder(encLstmFw, encLstmBw, encMu, encPresig),
        new Decoder([decLstm0, decLstm1], decZtoInitState, decOutputProjection)];
    })
}

function encodeAndDecode(encoder: Encoder, decoder: Decoder) {
  math.scope((keep, track) => {
    let rand_labels = track(Array1D.randUniform([32], 0, 129));
    const teaPot = [71, 0, 73, 0, 75, 0, 76, 0, 78, 0, 1, 0, 83, 0, 0, 0, 80, 0, 0, 0, 83, 0, 0, 0, 78, 0, 0, 0, 0, 0, 0, 0]
    for (let i = 0; i < rand_labels.shape[0]; ++i) {
      // rand_labels.set(Math.trunc(rand_labels.get(i) + 2), i);
      rand_labels.set(teaPot[i] + 1, i);
    }
    const rand_inputs = math.oneHot(rand_labels, 131);
    console.log(rand_labels.getValues());
    const outputs = encoder.encode(rand_inputs, track)
    const mu = outputs[1];
    const sampled_labels = decoder.decode(mu, 32, track);
    sampled_labels.forEach((sample: Array2D) => console.log(sample.getValues()));
    // return sampled_labels;
  });
  // document.getElementById('results').innerHTML = '' + result.getValues();
}
