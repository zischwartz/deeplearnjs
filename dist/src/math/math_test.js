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
var util = require("../util");
var ndarray_1 = require("./ndarray");
{
    var gpuTests = function (it) {
        it('scope returns NDArray', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var b = ndarray_1.Array1D.new([0, 0, 0]);
            var numUsedTexturesBefore = math.getTextureManager().getNumUsedTextures();
            math.scope(function () {
                var result = math.scope(function () {
                    b = math.add(a, b);
                    b = math.add(a, b);
                    b = math.add(a, b);
                    return math.add(a, b);
                });
                expect(math.getTextureManager().getNumUsedTextures())
                    .toEqual(numUsedTexturesBefore + 3);
                test_util.expectArraysClose(result.getValues(), new Float32Array([4, 8, 12]));
            });
            expect(math.getTextureManager().getNumUsedTextures())
                .toEqual(numUsedTexturesBefore + 2);
            a.dispose();
            b.dispose();
        });
        it('scope returns NDArray[]', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var b = ndarray_1.Array1D.new([0, -1, 1]);
            var numUsedTexturesBefore = math.getTextureManager().getNumUsedTextures();
            math.scope(function () {
                var result = math.scope(function () {
                    math.add(a, b);
                    return [math.add(a, b), math.subtract(a, b)];
                });
                expect(math.getTextureManager().getNumUsedTextures())
                    .toEqual(numUsedTexturesBefore + 4);
                test_util.expectArraysClose(result[0].getValues(), new Float32Array([1, 1, 4]));
                test_util.expectArraysClose(result[1].getValues(), new Float32Array([1, 3, 2]));
            });
            expect(math.getTextureManager().getNumUsedTextures())
                .toEqual(numUsedTexturesBefore + 2);
            a.dispose();
            b.dispose();
        });
        it('basic scope usage without return', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var b = ndarray_1.Array1D.new([0, 0, 0]);
            var numUsedTexturesBefore = math.getTextureManager().getNumUsedTextures();
            math.scope(function () {
                b = math.add(a, b);
                b = math.add(a, b);
                b = math.add(a, b);
                math.add(a, b);
            });
            var numUsedTexturesAfter = math.getTextureManager().getNumUsedTextures();
            expect(numUsedTexturesAfter).toEqual(numUsedTexturesBefore + 2);
        });
        it('scope returns Promise<NDArray>', function (math) { return __awaiter(_this, void 0, void 0, function () {
            var _this = this;
            var a, b, numUsedTexturesBefore;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        a = ndarray_1.Array1D.new([1, 2, 3]);
                        b = ndarray_1.Array1D.new([0, 0, 0]);
                        numUsedTexturesBefore = math.getTextureManager().getNumUsedTextures();
                        return [4, math.scope(function () { return __awaiter(_this, void 0, void 0, function () {
                                var result, data;
                                return __generator(this, function (_a) {
                                    switch (_a.label) {
                                        case 0:
                                            result = math.scope(function () {
                                                b = math.add(a, b);
                                                b = math.add(a, b);
                                                b = math.add(a, b);
                                                return math.add(a, b);
                                            });
                                            return [4, result.data()];
                                        case 1:
                                            data = _a.sent();
                                            expect(math.getTextureManager().getNumUsedTextures())
                                                .toEqual(numUsedTexturesBefore + 2);
                                            test_util.expectArraysClose(data, new Float32Array([4, 8, 12]));
                                            return [2];
                                    }
                                });
                            }); })];
                    case 1:
                        _a.sent();
                        expect(math.getTextureManager().getNumUsedTextures())
                            .toEqual(numUsedTexturesBefore + 2);
                        a.dispose();
                        b.dispose();
                        return [2];
                }
            });
        }); });
        it('nested scope usage', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 3]);
            var b = ndarray_1.Array1D.new([0, 0, 0]);
            var numUsedTexturesBefore = math.getTextureManager().getNumUsedTextures();
            math.scope(function () {
                var result = math.scope(function () {
                    b = math.add(a, b);
                    b = math.scope(function () {
                        b = math.scope(function () {
                            return math.add(a, b);
                        });
                        expect(math.getTextureManager().getNumUsedTextures())
                            .toEqual(numUsedTexturesBefore + 4);
                        math.scope(function () {
                            math.add(a, b);
                        });
                        expect(math.getTextureManager().getNumUsedTextures())
                            .toEqual(numUsedTexturesBefore + 4);
                        return math.add(a, b);
                    });
                    expect(math.getTextureManager().getNumUsedTextures())
                        .toEqual(numUsedTexturesBefore + 4);
                    return math.add(a, b);
                });
                expect(math.getTextureManager().getNumUsedTextures())
                    .toEqual(numUsedTexturesBefore + 3);
                test_util.expectArraysClose(result.getValues(), new Float32Array([4, 8, 12]));
            });
            expect(math.getTextureManager().getNumUsedTextures())
                .toEqual(numUsedTexturesBefore + 2);
        });
    };
    test_util.describeMathGPU('scope', [gpuTests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var gpuTests = function (it) {
        it('debug mode does not error when no nans', function (math) {
            math.enableDebugMode();
            var a = ndarray_1.Array1D.new([2, -1, 0, 3]);
            var res = math.relu(a);
            test_util.expectArraysClose(res.getValues(), new Float32Array([2, 0, 0, 3]));
            a.dispose();
        });
        it('debug mode errors when there are nans, float32', function (math) {
            math.enableDebugMode();
            var a = ndarray_1.Array1D.new([2, NaN]);
            var f = function () { return math.relu(a); };
            expect(f).toThrowError();
            a.dispose();
        });
        it('debug mode errors when there are nans, int32', function (math) {
            math.enableDebugMode();
            var a = ndarray_1.Array1D.new([2, util.NAN_INT32], 'int32');
            var f = function () { return math.relu(a); };
            expect(f).toThrowError();
            a.dispose();
        });
        it('debug mode errors when there are nans, bool', function (math) {
            math.enableDebugMode();
            var a = ndarray_1.Array1D.new([1, util.NAN_BOOL], 'bool');
            var f = function () { return math.relu(a); };
            expect(f).toThrowError();
            a.dispose();
        });
        it('no errors where there are nans, and debug mode is disabled', function (math) {
            var a = ndarray_1.Array1D.new([2, NaN]);
            var res = math.relu(a);
            test_util.expectArraysClose(res.getValues(), new Float32Array([2, NaN]));
            a.dispose();
        });
    };
    test_util.describeMathCPU('debug mode', [gpuTests]);
    test_util.describeMathGPU('debug mode', [gpuTests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('debug mode does not error when no nans', function (math) {
            var pixels = new ImageData(2, 2);
            for (var i = 0; i < 8; i++) {
                pixels.data[i] = 100;
            }
            for (var i = 8; i < 16; i++) {
                pixels.data[i] = 250;
            }
            var a = ndarray_1.Array3D.fromPixels(pixels, 4);
            var b = ndarray_1.Scalar.new(20.5);
            var res = math.add(a, b);
            test_util.expectArraysClose(res.getValues(), new Float32Array([
                120.5, 120.5, 120.5, 120.5, 120.5, 120.5, 120.5, 120.5, 270.5,
                270.5, 270.5, 270.5, 270.5, 270.5, 270.5, 270.5
            ]));
            a.dispose();
        });
    };
    test_util.describeMathGPU('fromPixels + math', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=math_test.js.map