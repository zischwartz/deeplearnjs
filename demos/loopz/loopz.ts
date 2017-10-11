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
import * as demo_util from '../util';

const math = new NDArrayMathGPU();
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
  return math.add(weightedResult, vars.bias);
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

  private runLstm(inputs: Array2D, lstmVars: LayerVars, reverse: boolean) {
    let state = [
      Array2D.zeros([1, lstmVars.bias.shape[0] / 4]),
      Array2D.zeros([1, lstmVars.bias.shape[0] / 4])
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
    const fw_state = this.runLstm(sequence, this.lstmFwVars, false)
    track(fw_state)
    const bw_state = this.runLstm(sequence, this.lstmBwVars, true)
    track(bw_state)
    const final_state = math.concat2D(fw_state[1], bw_state[1], -1)
    const mu = dense(this.muVars, final_state) as Array2D;
    const presig = dense(this.presigVars, final_state) as Array2D;
    const sigma = math.exp(math.arrayDividedByScalar(presig, presigDivisor));
    const z = math.addStrict(
      mu,
      math.multiplyStrict(sigma, track(Array2D.randNormal(sigma.shape))));
    return [z, mu, sigma];
  }
}

const CHECKPOINT_URL = '.';

const isDeviceSupported = demo_util.isWebGLSupported() && !demo_util.isSafari();

if (!isDeviceSupported) {
  document.querySelector('#status').innerHTML =
    'We do not yet support your device. Please try on a desktop ' +
    'computer with Chrome/Firefox, or an Android phone with WebGL support.';
} else {
  initialize().then((encoder: Encoder) => {
    encode(encoder);
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
      /*
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
      */
      return new Encoder(encLstmFw, encLstmBw, encMu, encPresig)
    })
}

function encode(encoder: Encoder) {
  const result = math.scope((keep, track) => {
    const rand_labels = track(Array1D.randUniform([16], 0, 129));
    const rand_inputs = math.oneHot(rand_labels, 131);
    console.log(rand_inputs.getValues());
    const outputs = encoder.encode(rand_inputs, track)
    console.log(outputs[0].getValues());
    return outputs[0];
  });
  document.getElementById('results').innerHTML = '' + result;
}
