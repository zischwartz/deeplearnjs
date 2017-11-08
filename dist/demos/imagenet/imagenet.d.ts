import '../demo-header';
import '../demo-footer';
import { PolymerHTMLElement } from '../polymer-spec';
export declare const ImagenetDemoPolymer: new () => PolymerHTMLElement;
export declare class ImagenetDemo extends ImagenetDemoPolymer {
    private math;
    private mathCPU;
    private gl;
    private gpgpu;
    private renderGrayscaleChannelsCollageShader;
    private squeezeNet;
    private webcamVideoElement;
    private staticImgElement;
    private layerNames;
    private selectedLayerName;
    private inputNames;
    private selectedInputName;
    private inferenceCanvas;
    ready(): void;
    private initWithoutWebcam();
    private initWithWebcam();
    private animate();
}
