import { Array3D, NDArrayMathCPU } from '../deeplearn';
import { BenchmarkTest } from './benchmark';
export interface PoolBenchmarkParams {
    depth: number;
    fieldSize: number;
    stride: number;
    type: 'max' | 'min' | 'avg';
}
export declare abstract class PoolBenchmark extends BenchmarkTest {
    protected params: PoolBenchmarkParams;
    constructor(params: PoolBenchmarkParams);
    protected getPoolingOp(option: string, math: NDArrayMathCPU): (x: Array3D, filterSize: [number, number] | number, strides: [number, number] | number, pad: 'valid' | 'same' | number) => Array3D;
}
export declare class PoolCPUBenchmark extends PoolBenchmark {
    run(size: number, option: string): Promise<number>;
}
export declare class PoolGPUBenchmark extends PoolBenchmark {
    run(size: number): Promise<number>;
}
