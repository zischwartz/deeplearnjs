import { BenchmarkTest } from './benchmark';
export declare class LogSumExpCPUBenchmark extends BenchmarkTest {
    run(size: number): Promise<number>;
}
export declare class LogSumExpGPUBenchmark extends BenchmarkTest {
    run(size: number): Promise<number>;
}
