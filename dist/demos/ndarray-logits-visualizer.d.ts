import { Array1D } from './deeplearn';
import { PolymerHTMLElement } from './polymer-spec';
export declare let NDArrayLogitsVisualizerPolymer: new () => PolymerHTMLElement;
export declare class NDArrayLogitsVisualizer extends NDArrayLogitsVisualizerPolymer {
    private logitLabelElements;
    private logitVizElements;
    private width;
    initialize(width: number, height: number): void;
    drawLogits(predictedLogits: Array1D, labelLogits: Array1D, labelClassNames?: string[]): void;
}
