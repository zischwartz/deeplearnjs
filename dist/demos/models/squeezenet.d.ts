import { Array1D, Array3D, NDArrayMathGPU } from '../deeplearn';
export declare class SqueezeNet {
    private math;
    private variables;
    private preprocessOffset;
    constructor(math: NDArrayMathGPU);
    loadVariables(): Promise<void>;
    infer(input: Array3D): {
        namedActivations: {
            [activationName: string]: Array3D;
        };
        logits: Array1D;
    };
    private fireModule(input, fireId);
    getTopKClasses(logits: Array1D, topK: number): Promise<{
        [className: string]: number;
    }>;
}
