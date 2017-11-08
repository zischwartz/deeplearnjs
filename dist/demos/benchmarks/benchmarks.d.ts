import '../demo-header';
import '../demo-footer';
import { PolymerHTMLElement } from '../polymer-spec';
export declare let MathBenchmarkPolymer: new () => PolymerHTMLElement;
export declare class MathBenchmark extends MathBenchmarkPolymer {
    private benchmarkRunGroupNames;
    private benchmarks;
    private stopMessages;
    ready(): void;
    private runBenchmarkGroup(groups, benchmarkRunGroupIndex);
    private buildRunNumbersRow(values);
    private runBenchmarkSteps(chart, benchmarkRunGroup, benchmarkRunGroupIndex, step);
}
