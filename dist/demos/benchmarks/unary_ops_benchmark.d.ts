import { NDArray, NDArrayMath } from '../deeplearn';
import { BenchmarkTest } from './benchmark';
export declare abstract class UnaryOpsBenchmark extends BenchmarkTest {
    protected getUnaryOp(option: string, math: NDArrayMath): (input: NDArray<"float32" | "int32" | "bool">) => NDArray<"float32" | "int32" | "bool">;
}
export declare class UnaryOpsCPUBenchmark extends UnaryOpsBenchmark {
    run(size: number, option: string): Promise<number>;
}
export declare class UnaryOpsGPUBenchmark extends UnaryOpsBenchmark {
    run(size: number, option: string): Promise<number>;
}
