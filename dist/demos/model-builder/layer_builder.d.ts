import { Graph, Tensor } from '../deeplearn';
export declare type LayerName = 'Fully connected' | 'ReLU' | 'Convolution' | 'Max pool' | 'Reshape' | 'Flatten';
export declare function getLayerBuilder(layerName: LayerName, layerBuilderJson?: LayerBuilder): LayerBuilder;
export interface LayerParam {
    label: string;
    initialValue(inputShape: number[]): number | string;
    type: 'number' | 'text';
    min?: number;
    max?: number;
    setValue(value: number | string): void;
    getValue(): number | string;
}
export declare type LayerWeightsDict = {
    [name: string]: number[];
};
export interface LayerBuilder {
    layerName: LayerName;
    getLayerParams(): LayerParam[];
    getOutputShape(inputShape: number[]): number[];
    addLayer(g: Graph, network: Tensor, inputShape: number[], index: number, weights?: LayerWeightsDict | null): Tensor;
    validate(inputShape: number[]): string[] | null;
}
export declare class FullyConnectedLayerBuilder implements LayerBuilder {
    layerName: LayerName;
    hiddenUnits: number;
    getLayerParams(): LayerParam[];
    getOutputShape(inputShape: number[]): number[];
    addLayer(g: Graph, network: Tensor, inputShape: number[], index: number, weights: LayerWeightsDict | null): Tensor;
    validate(inputShape: number[]): string[];
}
export declare class ReLULayerBuilder implements LayerBuilder {
    layerName: LayerName;
    getLayerParams(): LayerParam[];
    getOutputShape(inputShape: number[]): number[];
    addLayer(g: Graph, network: Tensor, inputShape: number[], index: number, weights: LayerWeightsDict | null): Tensor;
    validate(inputShape: number[]): string[] | null;
}
export declare class Convolution2DLayerBuilder implements LayerBuilder {
    layerName: LayerName;
    fieldSize: number;
    stride: number;
    zeroPad: number;
    outputDepth: number;
    getLayerParams(): LayerParam[];
    getOutputShape(inputShape: number[]): number[];
    addLayer(g: Graph, network: Tensor, inputShape: number[], index: number, weights: LayerWeightsDict | null): Tensor;
    validate(inputShape: number[]): string[];
}
export declare class MaxPoolLayerBuilder implements LayerBuilder {
    layerName: LayerName;
    fieldSize: number;
    stride: number;
    zeroPad: number;
    getLayerParams(): LayerParam[];
    getOutputShape(inputShape: number[]): number[];
    addLayer(g: Graph, network: Tensor, inputShape: number[], index: number, weights: LayerWeightsDict | null): Tensor;
    validate(inputShape: number[]): string[];
}
export declare class ReshapeLayerBuilder implements LayerBuilder {
    layerName: LayerName;
    outputShape: number[];
    getLayerParams(): {
        label: string;
        initialValue: (inputShape: number[]) => string;
        type: "text";
        setValue: (value: string) => number[];
        getValue: () => string;
    }[];
    getOutputShape(inputShape: number[]): number[];
    addLayer(g: Graph, network: Tensor, inputShape: number[], index: number, weights: LayerWeightsDict | null): Tensor;
    validate(inputShape: number[]): string[];
}
export declare class FlattenLayerBuilder implements LayerBuilder {
    layerName: LayerName;
    getLayerParams(): LayerParam[];
    getOutputShape(inputShape: number[]): number[];
    addLayer(g: Graph, network: Tensor, inputShape: number[], index: number, weights: LayerWeightsDict | null): Tensor;
    validate(inputShape: number[]): string[] | null;
}
