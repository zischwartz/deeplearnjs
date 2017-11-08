import { BenchmarkTest } from './benchmark';
export interface ConvTransposedBenchmarkParams {
    inDepth: number;
    outDepth: number;
    filterSize: number;
    stride: number;
}
export declare abstract class ConvTransposedBenchmark extends BenchmarkTest {
    protected params: ConvTransposedBenchmarkParams;
    constructor(params: ConvTransposedBenchmarkParams);
}
export declare class ConvTransposedGPUBenchmark extends ConvTransposedBenchmark {
    run(size: number): Promise<number>;
}
