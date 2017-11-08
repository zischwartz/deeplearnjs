import { NDArray, NDArrayMath, Scalar } from '../deeplearn';
import { BenchmarkTest } from './benchmark';
export declare abstract class ReductionOpsBenchmark extends BenchmarkTest {
    protected getReductionOp(option: string, math: NDArrayMath): (input: NDArray) => Scalar;
}
export declare class ReductionOpsCPUBenchmark extends ReductionOpsBenchmark {
    run(size: number, option: string): Promise<number>;
}
export declare class ReductionOpsGPUBenchmark extends ReductionOpsBenchmark {
    run(size: number, option: string): Promise<number>;
}
