import { Graph, Tensor } from '../deeplearn';
import { PolymerHTMLElement } from '../polymer-spec';
import { LayerBuilder, LayerName, LayerWeightsDict } from './layer_builder';
import { ModelBuilder } from './model-builder';
export declare let ModelLayerPolymer: new () => PolymerHTMLElement;
export declare class ModelLayer extends ModelLayerPolymer {
    inputShapeDisplay: string;
    outputShapeDisplay: string;
    private layerNames;
    private selectedLayerName;
    private hasError;
    private errorMessages;
    private modelBuilder;
    layerBuilder: LayerBuilder;
    private inputShape;
    private outputShape;
    private paramContainer;
    initialize(modelBuilder: ModelBuilder, inputShape: number[]): void;
    setInputShape(shape: number[]): number[];
    isValid(): boolean;
    getOutputShape(): number[];
    addLayer(g: Graph, network: Tensor, index: number, weights: LayerWeightsDict | null): Tensor;
    buildParamsUI(layerName: LayerName, inputShape: number[], layerBuilderJson?: LayerBuilder): void;
    loadParamsFromLayerBuilder(inputShape: number[], layerBuilderJson: LayerBuilder): void;
    private addParamField(label, initialValue, setValue, type, min?, max?);
}
