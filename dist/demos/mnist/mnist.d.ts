import { Array1D, NDArray, NDArrayMath, Scalar } from '../deeplearn';
export interface SampleData {
    images: number[][];
    labels: number[];
}
export declare function infer(math: NDArrayMath, x: Array1D, vars: {
    [varName: string]: NDArray;
}): Scalar;
