import { Array2D, GPGPUContext } from '../deeplearn';
export declare function createInputAtlas(imageSize: number, inputNumDimensions: number, numLatentVariables: number): Array2D<"float32" | "int32" | "bool">;
export declare function getAddLatentVariablesShader(gpgpu: GPGPUContext, inputNumDimensions: number): WebGLProgram;
export declare function addLatentVariables(gpgpu: GPGPUContext, addZShader: WebGLProgram, sourceTex: WebGLTexture, resultTex: WebGLTexture, shapeRowCol: [number, number], z1: number, z2: number): void;
export declare function getRenderShader(gpgpu: GPGPUContext, imageSize: number): WebGLProgram;
export declare function render(gpgpu: GPGPUContext, renderShader: WebGLProgram, sourceTex: WebGLTexture, outputNumDimensions: number, colorMode: number): void;
export declare function imagePixelToNormalizedCoord(x: number, y: number, imageWidth: number, imageHeight: number, zSize: number): number[];
