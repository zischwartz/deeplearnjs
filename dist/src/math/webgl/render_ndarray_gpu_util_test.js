"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
var gpgpu_context_1 = require("./gpgpu_context");
var gpgpu_util = require("./gpgpu_util");
var render_ndarray_gpu_util = require("./render_ndarray_gpu_util");
function uploadRenderRGBDownload(source, sourceShapeRowColDepth) {
    var canvas = document.createElement('canvas');
    canvas.width = sourceShapeRowColDepth[0];
    canvas.height = sourceShapeRowColDepth[1];
    var gpgpu = new gpgpu_context_1.GPGPUContext();
    gpgpu.enableAutomaticDebugValidation(true);
    var program = render_ndarray_gpu_util.getRenderRGBShader(gpgpu, sourceShapeRowColDepth[1]);
    var sourceTexShapeRC = conv_util.computeTexShapeFrom3D(sourceShapeRowColDepth);
    var sourceTex = gpgpu.createMatrixTexture(sourceTexShapeRC[0], sourceTexShapeRC[1]);
    gpgpu.uploadMatrixToTexture(sourceTex, sourceTexShapeRC[0], sourceTexShapeRC[1], source);
    var resultTex = gpgpu_util.createColorMatrixTexture(gpgpu.gl, sourceShapeRowColDepth[0], sourceShapeRowColDepth[1]);
    gpgpu.setOutputMatrixTexture(resultTex, sourceShapeRowColDepth[0], sourceShapeRowColDepth[1]);
    render_ndarray_gpu_util.renderToFramebuffer(gpgpu, program, sourceTex);
    var result = new Float32Array(sourceShapeRowColDepth[0] * sourceShapeRowColDepth[1] * 4);
    gpgpu.gl.readPixels(0, 0, sourceShapeRowColDepth[1], sourceShapeRowColDepth[0], gpgpu.gl.RGBA, gpgpu.gl.FLOAT, result);
    return result;
}
describe('render_gpu', function () {
    it('Packs a 1x1x3 vector to a 1x1 color texture', function () {
        var source = new Float32Array([1, 2, 3]);
        var result = uploadRenderRGBDownload(source, [1, 1, 3]);
        expect(result).toEqual(new Float32Array([1, 2, 3, 1]));
    });
    it('Packs a 2x2x3 vector to a 2x2 color texture, mirrored vertically', function () {
        var source = new Float32Array([1, 2, 3, 30, 20, 10, 2, 3, 4, 40, 30, 20]);
        var result = uploadRenderRGBDownload(source, [2, 2, 3]);
        expect(result).toEqual(new Float32Array([2, 3, 4, 1, 40, 30, 20, 1, 1, 2, 3, 1, 30, 20, 10, 1]));
    });
});
//# sourceMappingURL=render_ndarray_gpu_util_test.js.map