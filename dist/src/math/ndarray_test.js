"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var _this = this;
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var ndarray = require("./ndarray");
var ndarray_1 = require("./ndarray");
var gpgpu_context_1 = require("./webgl/gpgpu_context");
var gpgpu_util = require("./webgl/gpgpu_util");
var texture_manager_1 = require("./webgl/texture_manager");
var FEATURES = [
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
    { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
];
var gl;
var gpgpu;
var textureManager;
var customBeforeEach = function () {
    gl = gpgpu_util.createWebGLContext();
    gpgpu = new gpgpu_context_1.GPGPUContext(gl);
    textureManager = new texture_manager_1.TextureManager(gpgpu);
    ndarray.initializeGPU(gpgpu, textureManager);
};
var customAfterEach = function () {
    textureManager.dispose();
    gpgpu.dispose();
};
test_util.describeCustom('NDArray', function () {
    it('NDArrays of arbitrary size', function () {
        var t = ndarray_1.Array1D.new([1, 2, 3]);
        expect(t instanceof ndarray_1.Array1D).toBe(true);
        expect(t.rank).toBe(1);
        expect(t.size).toBe(3);
        test_util.expectArraysClose(t.getValues(), new Float32Array([1, 2, 3]));
        expect(t.get(4)).toBeUndefined();
        t = ndarray_1.Array2D.new([1, 3], [1, 2, 3]);
        expect(t instanceof ndarray_1.Array2D).toBe(true);
        expect(t.rank).toBe(2);
        expect(t.size).toBe(3);
        test_util.expectArraysClose(t.getValues(), new Float32Array([1, 2, 3]));
        expect(t.get(4)).toBeUndefined();
        t = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        expect(t instanceof ndarray_1.Array2D).toBe(true);
        expect(t.rank).toBe(2);
        expect(t.size).toBe(6);
        test_util.expectArraysClose(t.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));
        expect(t.get(5, 3)).toBeUndefined();
        expect(function () { return ndarray_1.Array2D.new([1, 2], [1]); }).toThrowError();
    });
    it('NDArrays of explicit size', function () {
        var t = ndarray_1.Array1D.new([5, 3, 2]);
        expect(t.rank).toBe(1);
        expect(t.shape).toEqual([3]);
        expect(t.get(1)).toBe(3);
        expect(function () { return ndarray_1.Array3D.new([1, 2, 3, 5], [
            1, 2
        ]); }).toThrowError('Shape should be of length 3');
        var t4 = ndarray_1.Array4D.new([1, 2, 1, 2], [1, 2, 3, 4]);
        expect(t4.get(0, 0, 0, 0)).toBe(1);
        expect(t4.get(0, 0, 0, 1)).toBe(2);
        expect(t4.get(0, 1, 0, 0)).toBe(3);
        expect(t4.get(0, 1, 0, 1)).toBe(4);
        var t4Like = ndarray_1.NDArray.like(t4);
        t4.set(10, 0, 0, 0, 1);
        expect(t4.get(0, 0, 0, 1)).toBe(10);
        expect(t4Like.get(0, 0, 0, 1)).toBe(2);
        var z = ndarray_1.NDArray.zeros([3, 4, 2]);
        expect(z.rank).toBe(3);
        expect(z.size).toBe(24);
        for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 4; j++) {
                for (var k = 0; k < 2; k++) {
                    expect(z.get(i, j, k)).toBe(0);
                }
            }
        }
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = a.reshape([3, 2, 1]);
        expect(a.get(1, 2)).toBe(6);
        b.set(10, 2, 1, 0);
        expect(a.get(1, 2)).toBe(10);
    });
    it('NDArray getValues CPU --> GPU', function () {
        var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
        expect(a.inGPU()).toBe(false);
        test_util.expectArraysClose(a.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));
        expect(a.inGPU()).toBe(false);
        expect(a.getTexture() != null).toBe(true);
        expect(a.inGPU()).toBe(true);
        a.dispose();
    });
    it('NDArray getValues GPU --> CPU', function () {
        var texture = textureManager.acquireTexture([3, 2]);
        gpgpu.uploadMatrixToTexture(texture, 3, 2, new Float32Array([1, 2, 3, 4, 5, 6]));
        var a = ndarray_1.Array2D.make([3, 2], { texture: texture, textureShapeRC: [3, 2] });
        expect(a.inGPU()).toBe(true);
        test_util.expectArraysClose(a.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));
        expect(a.inGPU()).toBe(false);
    });
    it('NDArray getValuesAsync CPU --> GPU', function (doneFn) {
        var a = ndarray_1.Array2D.new([3, 2], [1, 2, 3, 4, 5, 6]);
        expect(a.inGPU()).toBe(false);
        a.getValuesAsync().then(function (values) {
            test_util.expectArraysClose(values, new Float32Array([1, 2, 3, 4, 5, 6]));
            expect(a.inGPU()).toBe(false);
            expect(a.getTexture() != null).toBe(true);
            expect(a.inGPU()).toBe(true);
            a.dispose();
            doneFn();
        });
    });
    it('NDArray getValuesAsync GPU --> CPU', function () {
        var texture = textureManager.acquireTexture([3, 2]);
        gpgpu.uploadMatrixToTexture(texture, 3, 2, new Float32Array([1, 2, 3, 4, 5, 6]));
        var a = ndarray_1.Array2D.make([3, 2], { texture: texture, textureShapeRC: [3, 2] });
        expect(a.inGPU()).toBe(true);
        a.getValuesAsync().then(function (values) {
            test_util.expectArraysClose(values, new Float32Array([1, 2, 3, 4, 5, 6]));
            expect(a.inGPU()).toBe(false);
        });
    });
    it('NDArray.data GPU --> CPU', function () { return __awaiter(_this, void 0, void 0, function () {
        var texture, a, values;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    texture = textureManager.acquireTexture([3, 2]);
                    gpgpu.uploadMatrixToTexture(texture, 3, 2, new Float32Array([1, 2, 3, 4, 5, 6]));
                    a = ndarray_1.Array2D.make([3, 2], { texture: texture, textureShapeRC: [3, 2] });
                    expect(a.inGPU()).toBe(true);
                    return [4, a.data()];
                case 1:
                    values = _a.sent();
                    test_util.expectArraysClose(values, new Float32Array([1, 2, 3, 4, 5, 6]));
                    expect(a.inGPU()).toBe(false);
                    return [2];
            }
        });
    }); });
    it('NDArray.val() GPU --> CPU', function () { return __awaiter(_this, void 0, void 0, function () {
        var texture, a, _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o, _p, _q, _r, _s, _t;
        return __generator(this, function (_u) {
            switch (_u.label) {
                case 0:
                    texture = textureManager.acquireTexture([3, 2]);
                    gpgpu.uploadMatrixToTexture(texture, 3, 2, new Float32Array([1, 2, 3, 4, 5, 6]));
                    a = ndarray_1.Array2D.make([3, 2], { texture: texture, textureShapeRC: [3, 2] });
                    expect(a.inGPU()).toBe(true);
                    _b = (_a = test_util).expectNumbersClose;
                    _c = [1];
                    return [4, a.val(0)];
                case 1:
                    _b.apply(_a, _c.concat([_u.sent()]));
                    _e = (_d = test_util).expectNumbersClose;
                    _f = [2];
                    return [4, a.val(1)];
                case 2:
                    _e.apply(_d, _f.concat([_u.sent()]));
                    _h = (_g = test_util).expectNumbersClose;
                    _j = [3];
                    return [4, a.val(2)];
                case 3:
                    _h.apply(_g, _j.concat([_u.sent()]));
                    _l = (_k = test_util).expectNumbersClose;
                    _m = [4];
                    return [4, a.val(3)];
                case 4:
                    _l.apply(_k, _m.concat([_u.sent()]));
                    _p = (_o = test_util).expectNumbersClose;
                    _q = [5];
                    return [4, a.val(4)];
                case 5:
                    _p.apply(_o, _q.concat([_u.sent()]));
                    _s = (_r = test_util).expectNumbersClose;
                    _t = [6];
                    return [4, a.val(5)];
                case 6:
                    _s.apply(_r, _t.concat([_u.sent()]));
                    expect(a.inGPU()).toBe(false);
                    return [2];
            }
        });
    }); });
    it('Scalar basic methods', function () {
        var a = ndarray_1.Scalar.new(5);
        expect(a.get()).toBe(5);
        test_util.expectArraysClose(a.getValues(), new Float32Array([5]));
        expect(a.rank).toBe(0);
        expect(a.size).toBe(1);
        expect(a.shape).toEqual([]);
    });
    it('Scalar in GPU', function () {
        var texture = textureManager.acquireTexture([1, 1]);
        gpgpu.uploadMatrixToTexture(texture, 1, 1, new Float32Array([10]));
        var a = ndarray_1.Scalar.make([], { texture: texture });
        expect(a.inGPU()).toBe(true);
        test_util.expectArraysClose(a.getValues(), new Float32Array([10]));
        expect(a.inGPU()).toBe(false);
    });
    it('Array1D in GPU', function () {
        var texture = textureManager.acquireTexture([1, 3]);
        gpgpu.uploadMatrixToTexture(texture, 1, 3, new Float32Array([10, 7, 3]));
        var a = ndarray_1.Array1D.make([3], { texture: texture, textureShapeRC: [1, 3] });
        expect(a.inGPU()).toBe(true);
        test_util.expectArraysClose(a.getValues(), new Float32Array([10, 7, 3]));
        expect(a.inGPU()).toBe(false);
    });
    it('Array1D in GPU, but incorrect c-tor (missing textureShape)', function () {
        var texture = textureManager.acquireTexture([1, 3]);
        gpgpu.uploadMatrixToTexture(texture, 1, 3, new Float32Array([10, 7, 3]));
        var f = function () {
            return ndarray_1.Array1D.make([3], { texture: texture });
        };
        expect(f).toThrowError();
        textureManager.releaseTexture(texture, [1, 3]);
    });
    it('NDArray.make() constructs a Scalar', function () {
        var a = ndarray_1.NDArray.make([], { values: new Float32Array([3]) });
        expect(a instanceof ndarray_1.Scalar).toBe(true);
    });
    it('Array2D in GPU, reshaped to Array1D', function () {
        var texture = textureManager.acquireTexture([2, 2]);
        gpgpu.uploadMatrixToTexture(texture, 2, 2, new Float32Array([10, 7, 3, 5]));
        var a = ndarray_1.Array2D.make([2, 2], { texture: texture, textureShapeRC: [2, 2] });
        var a1d = a.as1D();
        test_util.expectArraysClose(a1d.getValues(), new Float32Array([10, 7, 3, 5]));
    });
    it('Array1D in GPU, reshaped to Array2D', function () {
        var texture = textureManager.acquireTexture([1, 4]);
        gpgpu.uploadMatrixToTexture(texture, 1, 4, new Float32Array([10, 7, 3, 5]));
        var a = ndarray_1.Array1D.make([4], { texture: texture, textureShapeRC: [1, 4] });
        var a2d = a.as2D(2, 2);
        test_util.expectArraysClose(a2d.getValues(), new Float32Array([10, 7, 3, 5]));
    });
    it('Array2D in GPU with custom texture shape', function () {
        var texture = textureManager.acquireTexture([4, 1]);
        gpgpu.uploadMatrixToTexture(texture, 4, 1, new Float32Array([10, 7, 3, 5]));
        var a = ndarray_1.Array2D.make([2, 2], { texture: texture, textureShapeRC: [4, 1] });
        test_util.expectArraysClose(a.getValues(), new Float32Array([10, 7, 3, 5]));
    });
    it('index2Loc Array1D', function () {
        var t = ndarray_1.Array1D.zeros([3]);
        expect(t.indexToLoc(0)).toEqual([0]);
        expect(t.indexToLoc(1)).toEqual([1]);
        expect(t.indexToLoc(2)).toEqual([2]);
    });
    it('index2Loc Array2D', function () {
        var t = ndarray_1.Array2D.zeros([3, 2]);
        expect(t.indexToLoc(0)).toEqual([0, 0]);
        expect(t.indexToLoc(1)).toEqual([0, 1]);
        expect(t.indexToLoc(2)).toEqual([1, 0]);
        expect(t.indexToLoc(3)).toEqual([1, 1]);
        expect(t.indexToLoc(4)).toEqual([2, 0]);
        expect(t.indexToLoc(5)).toEqual([2, 1]);
    });
    it('index2Loc Array3D', function () {
        var t = ndarray_1.Array2D.zeros([3, 2, 2]);
        expect(t.indexToLoc(0)).toEqual([0, 0, 0]);
        expect(t.indexToLoc(1)).toEqual([0, 0, 1]);
        expect(t.indexToLoc(2)).toEqual([0, 1, 0]);
        expect(t.indexToLoc(3)).toEqual([0, 1, 1]);
        expect(t.indexToLoc(4)).toEqual([1, 0, 0]);
        expect(t.indexToLoc(5)).toEqual([1, 0, 1]);
        expect(t.indexToLoc(11)).toEqual([2, 1, 1]);
    });
    it('index2Loc NDArray 5D', function () {
        var values = new Float32Array([1, 2, 3, 4]);
        var t = ndarray_1.NDArray.make([2, 1, 1, 1, 2], { values: values });
        expect(t.indexToLoc(0)).toEqual([0, 0, 0, 0, 0]);
        expect(t.indexToLoc(1)).toEqual([0, 0, 0, 0, 1]);
        expect(t.indexToLoc(2)).toEqual([1, 0, 0, 0, 0]);
        expect(t.indexToLoc(3)).toEqual([1, 0, 0, 0, 1]);
    });
    it('preferred texture shape, Scalar', function () {
        var t = ndarray_1.Scalar.new(1);
        expect(t.getTextureShapeRC()).toEqual([1, 1]);
    });
    it('preferred texture shape, Array1D column vector', function () {
        var t = ndarray_1.Array1D.zeros([4]);
        expect(t.getTextureShapeRC()).toEqual([4, 1]);
    });
    it('preferred texture shape, Array2D same shape', function () {
        var t = ndarray_1.Array2D.zeros([5, 2]);
        expect(t.getTextureShapeRC()).toEqual([5, 2]);
    });
    it('preferred texture shape, Array3D depth strided along columns', function () {
        var t = ndarray_1.Array3D.zeros([2, 2, 2]);
        expect(t.getTextureShapeRC()).toEqual([2, 4]);
    });
    it('preferred texture shape, Array4D d1 and d2 strided along columns', function () {
        var t = ndarray_1.Array4D.zeros([8, 2, 4, 4]);
        expect(t.getTextureShapeRC()).toEqual([8, 2 * 4 * 4]);
    });
}, FEATURES, customBeforeEach, customAfterEach);
test_util.describeCustom('NDArray.new', function () {
    it('Array1D.new() from number[]', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3]);
        test_util.expectArraysClose(a.getValues(), new Float32Array([1, 2, 3]));
    });
    it('Array1D.new() from number[][], shape mismatch', function () {
        expect(function () { return ndarray_1.Array1D.new([[1], [2], [3]]); }).toThrowError();
    });
    it('Array2D.new() from number[][]', function () {
        var a = ndarray_1.Array2D.new([2, 3], [[1, 2, 3], [4, 5, 6]]);
        test_util.expectArraysClose(a.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));
    });
    it('Array2D.new() from number[][], but shape does not match', function () {
        expect(function () { return ndarray_1.Array2D.new([3, 2], [[1, 2, 3], [4, 5, 6]]); }).toThrowError();
    });
    it('Array3D.new() from number[][][]', function () {
        var a = ndarray_1.Array3D.new([2, 3, 1], [[[1], [2], [3]], [[4], [5], [6]]]);
        test_util.expectArraysClose(a.getValues(), new Float32Array([1, 2, 3, 4, 5, 6]));
    });
    it('Array3D.new() from number[][][], but shape does not match', function () {
        var values = [[[1], [2], [3]], [[4], [5], [6]]];
        expect(function () { return ndarray_1.Array3D.new([3, 2, 1], values); }).toThrowError();
    });
    it('Array4D.new() from number[][][][]', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
        test_util.expectArraysClose(a.getValues(), new Float32Array([1, 2, 4, 5]));
    });
    it('Array4D.new() from number[][][][], but shape does not match', function () {
        var f = function () {
            ndarray_1.Array4D.new([2, 1, 2, 1], [[[[1]], [[2]]], [[[4]], [[5]]]]);
        };
        expect(f).toThrowError();
    });
});
test_util.describeCustom('NDArray.zeros', function () {
    it('1D default dtype', function () {
        var a = ndarray_1.Array1D.zeros([3]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('1D float32 dtype', function () {
        var a = ndarray_1.Array1D.zeros([3], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('1D int32 dtype', function () {
        var a = ndarray_1.Array1D.zeros([3], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        expect(a.getValues()).toEqual(new Int32Array([0, 0, 0]));
    });
    it('1D bool dtype', function () {
        var a = ndarray_1.Array1D.zeros([3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3]);
        expect(a.getValues()).toEqual(new Uint8Array([0, 0, 0]));
    });
    it('2D default dtype', function () {
        var a = ndarray_1.Array2D.zeros([3, 2]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
    });
    it('2D float32 dtype', function () {
        var a = ndarray_1.Array2D.zeros([3, 2], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2]);
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
    });
    it('2D int32 dtype', function () {
        var a = ndarray_1.Array2D.zeros([3, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3, 2]);
        expect(a.getValues()).toEqual(new Int32Array([0, 0, 0, 0, 0, 0]));
    });
    it('2D bool dtype', function () {
        var a = ndarray_1.Array2D.zeros([3, 2], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3, 2]);
        expect(a.getValues()).toEqual(new Uint8Array([0, 0, 0, 0, 0, 0]));
    });
    it('3D default dtype', function () {
        var a = ndarray_1.Array3D.zeros([2, 2, 2]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 2]);
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]));
    });
    it('3D float32 dtype', function () {
        var a = ndarray_1.Array3D.zeros([2, 2, 2], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 2]);
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0, 0, 0]));
    });
    it('3D int32 dtype', function () {
        var a = ndarray_1.Array3D.zeros([2, 2, 2], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 2]);
        expect(a.getValues()).toEqual(new Int32Array([0, 0, 0, 0, 0, 0, 0, 0]));
    });
    it('3D bool dtype', function () {
        var a = ndarray_1.Array3D.zeros([2, 2, 2], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2, 2]);
        expect(a.getValues()).toEqual(new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0]));
    });
    it('4D default dtype', function () {
        var a = ndarray_1.Array4D.zeros([3, 2, 1, 1]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
    });
    it('4D float32 dtype', function () {
        var a = ndarray_1.Array4D.zeros([3, 2, 1, 1], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
    });
    it('4D int32 dtype', function () {
        var a = ndarray_1.Array4D.zeros([3, 2, 1, 1], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        expect(a.getValues()).toEqual(new Int32Array([0, 0, 0, 0, 0, 0]));
    });
    it('4D bool dtype', function () {
        var a = ndarray_1.Array4D.zeros([3, 2, 1, 1], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([3, 2, 1, 1]);
        expect(a.getValues()).toEqual(new Uint8Array([0, 0, 0, 0, 0, 0]));
    });
});
test_util.describeCustom('NDArray.zerosLike', function () {
    it('1D default dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3]);
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        expect(b.getValues()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('1D float32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'float32');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        expect(b.getValues()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('1D int32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'int32');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([3]);
        expect(b.getValues()).toEqual(new Int32Array([0, 0, 0]));
    });
    it('1D bool dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'bool');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3]);
        expect(b.getValues()).toEqual(new Uint8Array([0, 0, 0]));
    });
    it('2D default dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('2D float32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('2D int32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2]);
        expect(b.getValues()).toEqual(new Int32Array([0, 0, 0, 0]));
    });
    it('2D bool dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2]);
        expect(b.getValues()).toEqual(new Uint8Array([0, 0, 0, 0]));
    });
    it('3D default dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('3D float32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('3D int32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1]);
        expect(b.getValues()).toEqual(new Int32Array([0, 0, 0, 0]));
    });
    it('3D bool dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1]);
        expect(b.getValues()).toEqual(new Uint8Array([0, 0, 0, 0]));
    });
    it('4D default dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('4D float32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        expect(b.getValues()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('4D int32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        expect(b.getValues()).toEqual(new Int32Array([0, 0, 0, 0]));
    });
    it('4D bool dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
        var b = ndarray_1.NDArray.zerosLike(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        expect(b.getValues()).toEqual(new Uint8Array([0, 0, 0, 0]));
    });
});
test_util.describeCustom('NDArray.like', function () {
    it('1D default dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3]);
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        expect(b.getValues()).toEqual(new Float32Array([1, 2, 3]));
    });
    it('1D float32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'float32');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([3]);
        expect(b.getValues()).toEqual(new Float32Array([1, 2, 3]));
    });
    it('1D int32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'int32');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([3]);
        expect(b.getValues()).toEqual(new Int32Array([1, 2, 3]));
    });
    it('1D bool dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'bool');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3]);
        expect(b.getValues()).toEqual(new Uint8Array([1, 1, 1]));
    });
    it('2D default dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('2D float32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'float32');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
        expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('2D int32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'int32');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2]);
        expect(b.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
    });
    it('2D bool dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4], 'bool');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2]);
        expect(b.getValues()).toEqual(new Uint8Array([1, 1, 1, 1]));
    });
    it('3D default dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('3D float32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'float32');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1]);
        expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('3D int32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'int32');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1]);
        expect(b.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
    });
    it('3D bool dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4], 'bool');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1]);
        expect(b.getValues()).toEqual(new Uint8Array([1, 1, 1, 1]));
    });
    it('4D default dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('4D float32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'float32');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        expect(b.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('4D int32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'int32');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        expect(b.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
    });
    it('4D bool dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4], 'bool');
        var b = ndarray_1.NDArray.like(a);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([2, 2, 1, 1]);
        expect(b.getValues()).toEqual(new Uint8Array([1, 1, 1, 1]));
    });
});
test_util.describeCustom('Scalar.new', function () {
    it('default dtype', function () {
        var a = ndarray_1.Scalar.new(3);
        expect(a.dtype).toBe('float32');
        expect(a.getValues()).toEqual(new Float32Array([3]));
    });
    it('float32 dtype', function () {
        var a = ndarray_1.Scalar.new(3, 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.getValues()).toEqual(new Float32Array([3]));
    });
    it('int32 dtype', function () {
        var a = ndarray_1.Scalar.new(3, 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.getValues()).toEqual(new Int32Array([3]));
    });
    it('int32 dtype, 3.9 => 3, like numpy', function () {
        var a = ndarray_1.Scalar.new(3.9, 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.getValues()).toEqual(new Int32Array([3]));
    });
    it('int32 dtype, -3.9 => -3, like numpy', function () {
        var a = ndarray_1.Scalar.new(-3.9, 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.getValues()).toEqual(new Int32Array([-3]));
    });
    it('bool dtype, 3 => true, like numpy', function () {
        var a = ndarray_1.Scalar.new(3, 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.get()).toBe(1);
    });
    it('bool dtype, -2 => true, like numpy', function () {
        var a = ndarray_1.Scalar.new(-2, 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.get()).toBe(1);
    });
    it('bool dtype, 0 => false, like numpy', function () {
        var a = ndarray_1.Scalar.new(0, 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.get()).toBe(0);
    });
    it('bool dtype from boolean', function () {
        var a = ndarray_1.Scalar.new(false, 'bool');
        expect(a.get()).toBe(0);
        expect(a.dtype).toBe('bool');
        var b = ndarray_1.Scalar.new(true, 'bool');
        expect(b.get()).toBe(1);
        expect(b.dtype).toBe('bool');
    });
    it('int32 dtype from boolean', function () {
        var a = ndarray_1.Scalar.new(true, 'int32');
        expect(a.get()).toBe(1);
        expect(a.dtype).toBe('int32');
    });
    it('default dtype from boolean', function () {
        var a = ndarray_1.Scalar.new(false);
        expect(a.get()).toBe(0);
        expect(a.dtype).toBe('float32');
    });
});
test_util.describeCustom('Array1D.new', function () {
    it('default dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        expect(a.getValues()).toEqual(new Float32Array([1, 2, 3]));
    });
    it('float32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'float32');
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([3]);
        expect(a.getValues()).toEqual(new Float32Array([1, 2, 3]));
    });
    it('int32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        expect(a.getValues()).toEqual(new Int32Array([1, 2, 3]));
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = ndarray_1.Array1D.new([1.1, 2.5, 3.9], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        expect(a.getValues()).toEqual(new Int32Array([1, 2, 3]));
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = ndarray_1.Array1D.new([-1.1, -2.5, -3.9], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([3]);
        expect(a.getValues()).toEqual(new Int32Array([-1, -2, -3]));
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = ndarray_1.Array1D.new([1, -2, 0, 3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([4]);
        expect(a.get(0)).toBe(1);
        expect(a.get(1)).toBe(1);
        expect(a.get(2)).toBe(0);
        expect(a.get(3)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = ndarray_1.Array1D.new([false, false, true]);
        expect(a.dtype).toBe('float32');
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 1]));
    });
    it('int32 dtype from boolean[]', function () {
        var a = ndarray_1.Array1D.new([false, false, true], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.getValues()).toEqual(new Int32Array([0, 0, 1]));
    });
    it('bool dtype from boolean[]', function () {
        var a = ndarray_1.Array1D.new([false, false, true], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.getValues()).toEqual(new Uint8Array([0, 0, 1]));
    });
});
test_util.describeCustom('Array2D.new', function () {
    it('default dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2]);
        expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('float32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2]);
        expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('int32 dtype', function () {
        var a = ndarray_1.Array2D.new([2, 2], [[1, 2], [3, 4]], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2]);
        expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1.1, 2.5, 3.9, 4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2]);
        expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = ndarray_1.Array2D.new([2, 2], [-1.1, -2.5, -3.9, -4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2]);
        expect(a.getValues()).toEqual(new Int32Array([-1, -2, -3, -4]));
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1, -2, 0, 3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2]);
        expect(a.get(0, 0)).toBe(1);
        expect(a.get(0, 1)).toBe(1);
        expect(a.get(1, 0)).toBe(0);
        expect(a.get(1, 1)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = ndarray_1.Array2D.new([2, 2], [[false, false], [true, false]]);
        expect(a.dtype).toBe('float32');
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 1, 0]));
    });
    it('int32 dtype from boolean[]', function () {
        var a = ndarray_1.Array2D.new([2, 2], [[false, false], [true, false]], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.getValues()).toEqual(new Int32Array([0, 0, 1, 0]));
    });
    it('bool dtype from boolean[]', function () {
        var a = ndarray_1.Array2D.new([2, 2], [[false, false], [true, false]], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.getValues()).toEqual(new Uint8Array([0, 0, 1, 0]));
    });
});
test_util.describeCustom('Array3D.new', function () {
    it('default dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1]);
        expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('float32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1]);
        expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('int32 dtype', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [[[1], [2]], [[3], [4]]], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1]);
        expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1.1, 2.5, 3.9, 4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1]);
        expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [-1.1, -2.5, -3.9, -4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1]);
        expect(a.getValues()).toEqual(new Int32Array([-1, -2, -3, -4]));
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [1, -2, 0, 3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2, 1]);
        expect(a.get(0, 0, 0)).toBe(1);
        expect(a.get(0, 1, 0)).toBe(1);
        expect(a.get(1, 0, 0)).toBe(0);
        expect(a.get(1, 1, 0)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [[[false], [false]], [[true], [false]]]);
        expect(a.dtype).toBe('float32');
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 1, 0]));
    });
    it('int32 dtype from boolean[]', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [[[false], [false]], [[true], [false]]], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.getValues()).toEqual(new Int32Array([0, 0, 1, 0]));
    });
    it('bool dtype from boolean[]', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [[[false], [false]], [[true], [false]]], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.getValues()).toEqual(new Uint8Array([0, 0, 1, 0]));
    });
});
test_util.describeCustom('Array4D.new', function () {
    it('default dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('float32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, 2, 3, 4]);
        expect(a.dtype).toBe('float32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        expect(a.getValues()).toEqual(new Float32Array([1, 2, 3, 4]));
    });
    it('int32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [[[[1]], [[2]]], [[[3]], [[4]]]], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
    });
    it('int32 dtype, non-ints get floored, like numpy', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1.1, 2.5, 3.9, 4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        expect(a.getValues()).toEqual(new Int32Array([1, 2, 3, 4]));
    });
    it('int32 dtype, negative non-ints get ceiled, like numpy', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [-1.1, -2.5, -3.9, -4.0], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        expect(a.getValues()).toEqual(new Int32Array([-1, -2, -3, -4]));
    });
    it('bool dtype, !=0 is truthy, 0 is falsy, like numpy', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [1, -2, 0, 3], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.shape).toEqual([2, 2, 1, 1]);
        expect(a.get(0, 0, 0, 0)).toBe(1);
        expect(a.get(0, 1, 0, 0)).toBe(1);
        expect(a.get(1, 0, 0, 0)).toBe(0);
        expect(a.get(1, 1, 0, 0)).toBe(1);
    });
    it('default dtype from boolean[]', function () {
        var a = ndarray_1.Array4D.new([1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]]);
        expect(a.dtype).toBe('float32');
        expect(a.getValues()).toEqual(new Float32Array([0, 0, 1, 0]));
    });
    it('int32 dtype from boolean[]', function () {
        var a = ndarray_1.Array4D.new([1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]], 'int32');
        expect(a.dtype).toBe('int32');
        expect(a.getValues()).toEqual(new Int32Array([0, 0, 1, 0]));
    });
    it('bool dtype from boolean[]', function () {
        var a = ndarray_1.Array4D.new([1, 2, 2, 1], [[[[false], [false]], [[true], [false]]]], 'bool');
        expect(a.dtype).toBe('bool');
        expect(a.getValues()).toEqual(new Uint8Array([0, 0, 1, 0]));
    });
});
test_util.describeCustom('NDArray.reshape', function () {
    it('Scalar default dtype', function () {
        var a = ndarray_1.Scalar.new(4);
        var b = a.reshape([1, 1]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 1]);
    });
    it('Scalar bool dtype', function () {
        var a = ndarray_1.Scalar.new(4, 'bool');
        var b = a.reshape([1, 1, 1]);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([1, 1, 1]);
    });
    it('Array1D default dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3, 4]);
        var b = a.reshape([2, 2]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
    });
    it('Array1D int32 dtype', function () {
        var a = ndarray_1.Array1D.new([1, 2, 3, 4], 'int32');
        var b = a.reshape([2, 2]);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([2, 2]);
    });
    it('Array2D default dtype', function () {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6]);
        var b = a.reshape([6]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([6]);
    });
    it('Array2D bool dtype', function () {
        var a = ndarray_1.Array2D.new([2, 3], [1, 2, 3, 4, 5, 6], 'bool');
        var b = a.reshape([6]);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([6]);
    });
    it('Array3D default dtype', function () {
        var a = ndarray_1.Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6]);
        var b = a.reshape([6]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([6]);
    });
    it('Array3D bool dtype', function () {
        var a = ndarray_1.Array3D.new([2, 3, 1], [1, 2, 3, 4, 5, 6], 'bool');
        var b = a.reshape([6]);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([6]);
    });
    it('Array4D default dtype', function () {
        var a = ndarray_1.Array4D.new([2, 3, 1, 1], [1, 2, 3, 4, 5, 6]);
        var b = a.reshape([2, 3]);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 3]);
    });
    it('Array4D int32 dtype', function () {
        var a = ndarray_1.Array4D.new([2, 3, 1, 1], [1, 2, 3, 4, 5, 6], 'int32');
        var b = a.reshape([3, 2]);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([3, 2]);
    });
});
test_util.describeCustom('NDArray.asXD preserves dtype', function () {
    it('scalar -> 2d', function () {
        var a = ndarray_1.Scalar.new(4, 'int32');
        var b = a.as2D(1, 1);
        expect(b.dtype).toBe('int32');
        expect(b.shape).toEqual([1, 1]);
    });
    it('1d -> 2d', function () {
        var a = ndarray_1.Array1D.new([4, 2, 1], 'bool');
        var b = a.as2D(3, 1);
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([3, 1]);
    });
    it('2d -> 4d', function () {
        var a = ndarray_1.Array2D.new([2, 2], [4, 2, 1, 3]);
        var b = a.as4D(1, 1, 2, 2);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([1, 1, 2, 2]);
    });
    it('3d -> 2d', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [4, 2, 1, 3], 'float32');
        var b = a.as2D(2, 2);
        expect(b.dtype).toBe('float32');
        expect(b.shape).toEqual([2, 2]);
    });
    it('4d -> 1d', function () {
        var a = ndarray_1.Array4D.new([2, 2, 1, 1], [4, 2, 1, 3], 'bool');
        var b = a.as1D();
        expect(b.dtype).toBe('bool');
        expect(b.shape).toEqual([4]);
    });
});
test_util.describeCustom('NDArray.asType', function () {
    it('scalar bool -> int32', function () {
        var a = ndarray_1.Scalar.new(true, 'bool').asType('int32');
        expect(a.dtype).toBe('int32');
        expect(a.get()).toBe(1);
    });
    it('array1d float32 -> int32', function () {
        var a = ndarray_1.Array1D.new([1.1, 3.9, -2.9, 0]).asType('int32');
        expect(a.dtype).toBe('int32');
        expect(a.getValues()).toEqual(new Int32Array([1, 3, -2, 0]));
    });
    it('array2d float32 -> bool', function () {
        var a = ndarray_1.Array2D.new([2, 2], [1.1, 3.9, -2.9, 0]).asType(ndarray_1.DType.bool);
        expect(a.dtype).toBe('bool');
        expect(a.get(0, 0)).toBe(1);
        expect(a.get(0, 1)).toBe(1);
        expect(a.get(1, 0)).toBe(1);
        expect(a.get(1, 1)).toBe(0);
    });
    it('array3d bool -> float32', function () {
        var a = ndarray_1.Array3D.new([2, 2, 1], [true, false, false, true], 'bool')
            .asType('float32');
        expect(a.dtype).toBe('float32');
        expect(a.getValues()).toEqual(new Float32Array([1, 0, 0, 1]));
    });
});
test_util.describeCustom('NDArray CPU <--> GPU with dtype', function () {
    it('bool CPU -> GPU -> CPU', function () {
        var a = ndarray_1.Array1D.new([1, 2, 0, 0, 5], 'bool');
        expect(a.inGPU()).toBe(false);
        expect(a.getValues()).toEqual(new Uint8Array([1, 1, 0, 0, 1]));
        expect(a.getTexture() != null).toBe(true);
        expect(a.inGPU()).toBe(true);
        expect(a.getValues()).toEqual(new Uint8Array([1, 1, 0, 0, 1]));
        a.dispose();
    });
    it('bool GPU --> CPU', function () {
        var shape = [1, 5];
        var texture = textureManager.acquireTexture(shape);
        gpgpu.uploadMatrixToTexture(texture, shape[0], shape[1], new Float32Array([1, 1, 0, 0, 1]));
        var a = ndarray_1.Array1D.make(shape, { texture: texture, textureShapeRC: shape }, 'bool');
        expect(a.inGPU()).toBe(true);
        expect(a.getValues()).toEqual(new Uint8Array([1, 1, 0, 0, 1]));
        expect(a.inGPU()).toBe(false);
    });
    it('int32 CPU -> GPU -> CPU', function () {
        var a = ndarray_1.Array1D.new([1, 2, 0, 0, 5], 'int32');
        expect(a.inGPU()).toBe(false);
        expect(a.getValues()).toEqual(new Int32Array([1, 2, 0, 0, 5]));
        expect(a.getTexture() != null).toBe(true);
        expect(a.inGPU()).toBe(true);
        expect(a.getValues()).toEqual(new Int32Array([1, 2, 0, 0, 5]));
        a.dispose();
    });
    it('int32 GPU --> CPU', function () {
        var shape = [1, 5];
        var texture = textureManager.acquireTexture(shape);
        gpgpu.uploadMatrixToTexture(texture, shape[0], shape[1], new Float32Array([1, 5.003, 0, 0, 1.001]));
        var a = ndarray_1.Array1D.make(shape, { texture: texture, textureShapeRC: shape }, 'int32');
        expect(a.inGPU()).toBe(true);
        expect(a.getValues()).toEqual(new Int32Array([1, 5, 0, 0, 1]));
        expect(a.inGPU()).toBe(false);
    });
}, FEATURES, customBeforeEach, customAfterEach);
{
    test_util.describeCustom('NDArray.fromPixels', function () {
        var gl;
        var gpgpu;
        var textureManager;
        beforeEach(function () {
            gl = gpgpu_util.createWebGLContext();
            gpgpu = new gpgpu_context_1.GPGPUContext(gl);
            textureManager = new texture_manager_1.TextureManager(gpgpu);
            ndarray.initializeGPU(gpgpu, textureManager);
        });
        afterEach(function () {
            textureManager.dispose();
            gpgpu.dispose();
        });
        it('ImageData 1x1x3', function () {
            var pixels = new ImageData(1, 1);
            pixels.data[0] = 0;
            pixels.data[1] = 80;
            pixels.data[2] = 160;
            pixels.data[3] = 240;
            var array = ndarray_1.Array3D.fromPixels(pixels, 3);
            test_util.expectArraysClose(array.getValues(), new Float32Array([0, 80, 160]));
        });
        it('ImageData 1x1x4', function () {
            var pixels = new ImageData(1, 1);
            pixels.data[0] = 0;
            pixels.data[1] = 80;
            pixels.data[2] = 160;
            pixels.data[3] = 240;
            var array = ndarray_1.Array3D.fromPixels(pixels, 4);
            test_util.expectArraysClose(array.getValues(), new Float32Array([0, 80, 160, 240]));
        });
        it('ImageData 2x2x3', function () {
            var pixels = new ImageData(2, 2);
            for (var i = 0; i < 8; i++) {
                pixels.data[i] = i * 2;
            }
            for (var i = 8; i < 16; i++) {
                pixels.data[i] = i * 2;
            }
            var array = ndarray_1.Array3D.fromPixels(pixels, 3);
            test_util.expectArraysClose(array.getValues(), new Float32Array([0, 2, 4, 8, 10, 12, 16, 18, 20, 24, 26, 28]));
        });
        it('ImageData 2x2x4', function () {
            var pixels = new ImageData(2, 2);
            for (var i = 0; i < 8; i++) {
                pixels.data[i] = i * 2;
            }
            for (var i = 8; i < 16; i++) {
                pixels.data[i] = i * 2;
            }
            var array = ndarray_1.Array3D.fromPixels(pixels, 4);
            test_util.expectArraysClose(array.getValues(), new Float32Array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]));
        });
    }, [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=ndarray_test.js.map