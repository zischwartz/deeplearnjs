import { GPGPUProgram } from './gpgpu_math';
export declare class MinMaxProgram implements GPGPUProgram {
    variableNames: string[];
    params: Array<{}>;
    outputShape: number[];
    userCode: string;
    numBatchDims: number;
    constructor(shape: number[], axes: number[], op: 'min' | 'max');
}
