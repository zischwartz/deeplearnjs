import { BenchmarkTest } from './benchmark';
export declare class MatmulCPUBenchmark extends BenchmarkTest {
    run(size: number): Promise<number>;
}
export declare class MatmulGPUBenchmark extends BenchmarkTest {
    run(size: number): Promise<number>;
}
