import { Array3D } from './deeplearn';
import { PolymerHTMLElement } from './polymer-spec';
export declare let NDArrayImageVisualizerPolymer: new () => PolymerHTMLElement;
export declare class NDArrayImageVisualizer extends NDArrayImageVisualizerPolymer {
    private canvas;
    private canvasContext;
    private imageData;
    ready(): void;
    setShape(shape: number[]): void;
    setSize(width: number, height: number): void;
    saveImageDataFromNDArray(ndarray: Array3D): void;
    drawRGBImageData(ndarray: Array3D): void;
    drawGrayscaleImageData(ndarray: Array3D): void;
    draw(): void;
}
