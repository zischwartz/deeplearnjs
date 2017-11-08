import { BenchmarkTest } from './benchmark';
export interface ConvBenchmarkParams {
    inDepth: number;
    outDepth: number;
    filterSize: number;
    stride: number;
}
export declare abstract class ConvBenchmark extends BenchmarkTest {
    protected params: ConvBenchmarkParams;
    constructor(params: ConvBenchmarkParams);
}
export declare class ConvGPUBenchmark extends ConvBenchmark {
    run(size: number): Promise<number>;
}
