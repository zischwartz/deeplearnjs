"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var deeplearn_1 = require("../deeplearn");
function getRenderGrayscaleChannelsCollageShader(gpgpu) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D source;\n    uniform sampler2D minValues;\n    uniform sampler2D maxValues;\n    varying vec2 resultUV;\n\n    uniform float imageSize;\n    uniform float channels;\n    uniform float imagesPerRow;\n    uniform vec2 inputShapeCR;\n\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    void main() {\n      vec2 outputCR = floor(gl_FragCoord.xy);\n\n      float imageRow = floor(outputCR[1] / imageSize);\n      float imageCol = mod(outputCR[0], imageSize);\n\n      float currentChannel = floor(outputCR[0] / imageSize) +\n          imageRow * imagesPerRow;\n\n      // When the number of channels is not square, we render white to fill in\n      // the output texture.\n      if (currentChannel > channels) {\n        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n        return;\n      }\n\n      float sourceC = channels * imageCol + currentChannel;\n      float sourceR = mod(outputCR[1], imageSize);\n\n      vec2 sourceUV = (vec2(sourceC, sourceR) + halfCR) / inputShapeCR;\n\n      // Flip the vertical axis of the texture for display since we represent\n      // image textures as vertically flipped.\n      float sourceValue = texture2D(\n          source, vec2(sourceUV.s, 1.0 - sourceUV.t)).r;\n\n      // Normalize the value by sampling the minValues and maxValues texture\n      // which contain min and max per channel.\n      vec2 minMaxValuesShapeCR = vec2(channels, 1);\n      vec2 minMaxValuesCR = vec2(currentChannel, 0);\n      vec2 minMaxValuesUV = (minMaxValuesCR + halfCR) / minMaxValuesShapeCR;\n\n      float minValue = texture2D(minValues, minMaxValuesUV).r;\n      float maxValue = texture2D(maxValues, minMaxValuesUV).r;\n\n      float normalizedValue = (sourceValue - minValue) / (maxValue - minValue);\n\n      gl_FragColor = vec4(\n          normalizedValue, normalizedValue, normalizedValue, 1);\n    }\n  ";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.getRenderGrayscaleChannelsCollageShader = getRenderGrayscaleChannelsCollageShader;
function renderGrayscaleChannelsCollage(gpgpu, unpackChannelsShader, sourceTex, minValuesTex, maxValuesTex, inputShapeRC, imageSize, channels, textureSize, numRows) {
    deeplearn_1.webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    gpgpu.setProgram(unpackChannelsShader);
    var sourceSamplerLocation = deeplearn_1.webgl_util.getProgramUniformLocationOrThrow(gpgpu.gl, unpackChannelsShader, 'source');
    var minValuesSamplerLocation = deeplearn_1.webgl_util.getProgramUniformLocationOrThrow(gpgpu.gl, unpackChannelsShader, 'minValues');
    var maxValuesSamplerLocation = deeplearn_1.webgl_util.getProgramUniformLocationOrThrow(gpgpu.gl, unpackChannelsShader, 'maxValues');
    gpgpu.setInputMatrixTexture(sourceTex, sourceSamplerLocation, 0);
    gpgpu.setInputMatrixTexture(minValuesTex, minValuesSamplerLocation, 1);
    gpgpu.setInputMatrixTexture(maxValuesTex, maxValuesSamplerLocation, 2);
    var imageSizeLoc = gpgpu.getUniformLocation(unpackChannelsShader, 'imageSize');
    gpgpu.gl.uniform1f(imageSizeLoc, imageSize);
    var channelsLoc = gpgpu.getUniformLocation(unpackChannelsShader, 'channels');
    gpgpu.gl.uniform1f(channelsLoc, channels);
    var imagesPerRowLoc = gpgpu.getUniformLocation(unpackChannelsShader, 'imagesPerRow');
    gpgpu.gl.uniform1f(imagesPerRowLoc, Math.floor(textureSize / imageSize));
    var inputShapeCRLoc = gpgpu.getUniformLocation(unpackChannelsShader, 'inputShapeCR');
    gpgpu.gl.uniform2f(inputShapeCRLoc, inputShapeRC[1], inputShapeRC[0]);
    gpgpu.executeProgram();
}
exports.renderGrayscaleChannelsCollage = renderGrayscaleChannelsCollage;
//# sourceMappingURL=imagenet_util.js.map