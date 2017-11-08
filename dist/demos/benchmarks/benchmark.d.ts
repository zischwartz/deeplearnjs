export interface BenchmarkRunGroup {
    name: string;
    min: number;
    max: number;
    stepSize: number;
    stepToSizeTransformation?: (step: number) => number;
    options?: string[];
    selectedOption?: string;
    benchmarkRuns: BenchmarkRun[];
    params: {};
}
export declare class BenchmarkRun {
    name: string;
    benchmarkTest: BenchmarkTest;
    chartData: ChartData[];
    constructor(name: string, benchmarkTest: BenchmarkTest);
    clearChartData(): void;
}
export declare abstract class BenchmarkTest {
    protected params: {};
    constructor(params?: {});
    abstract run(size: number, option?: string): Promise<number>;
}
