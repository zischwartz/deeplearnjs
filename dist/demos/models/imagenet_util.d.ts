import { GPGPUContext } from '../deeplearn';
export declare function getRenderGrayscaleChannelsCollageShader(gpgpu: GPGPUContext): WebGLProgram;
export declare function renderGrayscaleChannelsCollage(gpgpu: GPGPUContext, unpackChannelsShader: WebGLProgram, sourceTex: WebGLTexture, minValuesTex: WebGLTexture, maxValuesTex: WebGLTexture, inputShapeRC: [number, number], imageSize: number, channels: number, textureSize: number, numRows: number): void;
