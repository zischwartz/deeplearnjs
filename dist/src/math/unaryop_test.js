"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var test_util = require("../test_util");
var util = require("../util");
var ndarray_1 = require("./ndarray");
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 0, 3, -0.1]);
            var result = math.relu(a);
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 0, 0, 3, 0]));
            a.dispose();
        });
        it('does nothing to positive values', function (math) {
            var a = ndarray_1.Scalar.new(1);
            var result = math.relu(a);
            test_util.expectNumbersClose(result.get(), 1);
            a.dispose();
        });
        it('sets negative values to 0', function (math) {
            var a = ndarray_1.Scalar.new(-1);
            var result = math.relu(a);
            test_util.expectNumbersClose(result.get(), 0);
            a.dispose();
        });
        it('preserves zero values', function (math) {
            var a = ndarray_1.Scalar.new(0);
            var result = math.relu(a);
            test_util.expectNumbersClose(result.get(), 0);
            a.dispose();
        });
        it('propagates NaNs, float32', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 0, 3, -0.1, NaN]);
            var result = math.relu(a);
            expect(result.dtype).toBe('float32');
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 0, 0, 3, 0, NaN]));
            a.dispose();
        });
        it('propagates NaNs, int32', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 0, 3, -1, util.NAN_INT32], 'int32');
            var result = math.relu(a);
            expect(result.dtype).toBe('int32');
            test_util.expectArraysClose(result.getValues(), new Int32Array([1, 0, 0, 3, 0, util.NAN_INT32]));
            a.dispose();
        });
        it('propagates NaNs, bool', function (math) {
            var a = ndarray_1.Array1D.new([1, 0, 0, 1, 0, util.NAN_BOOL], 'bool');
            var result = math.relu(a);
            expect(result.dtype).toBe('bool');
            test_util.expectArraysClose(result.getValues(), new Uint8Array([1, 0, 0, 1, 0, util.NAN_BOOL]));
            a.dispose();
        });
    };
    test_util.describeMathCPU('relu', [tests]);
    test_util.describeMathGPU('relu', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 0, 3, -0.1]);
            var result = math.abs(a);
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 2, 0, 3, 0.1]));
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, 0, 3, -0.1, NaN]);
            var result = math.abs(a);
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 2, 0, 3, 0.1, NaN]));
            a.dispose();
        });
    };
    test_util.describeMathCPU('abs', [tests]);
    test_util.describeMathGPU('abs', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('with 1d ndarray', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, -.01, 3, -0.1]);
            var result = math.step(a);
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 0, 0, 1, 0]));
            a.dispose();
        });
        it('with 2d ndarray', function (math) {
            var a = ndarray_1.Array2D.new([2, 2], [1, -5, -3, 4]);
            var result = math.step(a);
            expect(result.shape).toEqual([2, 2]);
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 0, 0, 1]));
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, -2, -.01, 3, NaN]);
            var result = math.step(a);
            test_util.expectArraysClose(result.getValues(), new Float32Array([1, 0, 0, 1, NaN]));
            a.dispose();
        });
    };
    test_util.describeMathCPU('step', [tests]);
    test_util.describeMathGPU('step', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1, -3, 2, 7, -4]);
            var result = math.neg(a);
            test_util.expectArraysClose(result.getValues(), new Float32Array([-1, 3, -2, -7, 4]));
            a.dispose();
        });
        it('propagate NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, -3, 2, 7, NaN]);
            var result = math.neg(a);
            var expected = [-1, 3, -2, -7, NaN];
            test_util.expectArraysClose(result.getValues(), new Float32Array(expected));
            a.dispose();
        });
    };
    test_util.describeMathCPU('neg', [tests]);
    test_util.describeMathGPU('neg', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.sigmoid(a);
            var expected = new Float32Array(a.size);
            for (var i = 0; i < a.size; i++) {
                expected[i] = 1 / (1 + Math.exp(-values[i]));
            }
            test_util.expectArraysClose(result.getValues(), expected);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([3, NaN]);
            var res = math.sigmoid(a).getValues();
            test_util.expectArraysClose(res, new Float32Array([1 / (1 + Math.exp(-3)), NaN]));
            a.dispose();
        });
    };
    test_util.describeMathCPU('sigmoid', [tests]);
    test_util.describeMathGPU('sigmoid', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('sqrt', function (math) {
            var a = ndarray_1.Array1D.new([2, 4]);
            var r = math.sqrt(a);
            expect(r.get(0)).toBeCloseTo(Math.sqrt(2));
            expect(r.get(1)).toBeCloseTo(Math.sqrt(4));
            a.dispose();
        });
        it('sqrt propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN]);
            var r = math.sqrt(a).getValues();
            test_util.expectArraysClose(r, new Float32Array([Math.sqrt(1), NaN]));
            a.dispose();
        });
    };
    test_util.describeMathCPU('sqrt', [tests]);
    test_util.describeMathGPU('sqrt', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('log', function (math) {
            var a = ndarray_1.Array1D.new([1, 2]);
            var r = math.log(a);
            expect(r.get(0)).toBeCloseTo(Math.log(1));
            expect(r.get(1)).toBeCloseTo(Math.log(2));
            a.dispose();
        });
        it('log propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN]);
            var r = math.log(a).getValues();
            test_util.expectArraysClose(r, new Float32Array([Math.log(1), NaN]));
            a.dispose();
        });
    };
    test_util.describeMathCPU('log', [tests]);
    test_util.describeMathGPU('log', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1.5, 2.1, -1.4]);
            var r = math.ceil(a);
            expect(r.get(0)).toBeCloseTo(2);
            expect(r.get(1)).toBeCloseTo(3);
            expect(r.get(2)).toBeCloseTo(-1);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1.5, NaN, -1.4]);
            var r = math.ceil(a).getValues();
            test_util.expectArraysClose(r, new Float32Array([2, NaN, -1]));
            a.dispose();
        });
    };
    test_util.describeMathCPU('ceil', [tests]);
    test_util.describeMathGPU('ceil', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([1.5, 2.1, -1.4]);
            var r = math.floor(a);
            expect(r.get(0)).toBeCloseTo(1);
            expect(r.get(1)).toBeCloseTo(2);
            expect(r.get(2)).toBeCloseTo(-2);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1.5, NaN, -1.4]);
            var r = math.floor(a).getValues();
            test_util.expectArraysClose(r, new Float32Array([1, NaN, -2]));
            a.dispose();
        });
    };
    test_util.describeMathCPU('floor', [tests]);
    test_util.describeMathGPU('floor', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('exp', function (math) {
            var a = ndarray_1.Array1D.new([1, 2, 0]);
            var r = math.exp(a);
            expect(r.get(0)).toBeCloseTo(Math.exp(1));
            expect(r.get(1)).toBeCloseTo(Math.exp(2));
            expect(r.get(2)).toBeCloseTo(1);
            a.dispose();
        });
        it('exp propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN, 0]);
            var r = math.exp(a).getValues();
            test_util.expectArraysClose(r, new Float32Array([Math.exp(1), NaN, 1]));
            a.dispose();
        });
    };
    test_util.describeMathCPU('exp', [tests]);
    test_util.describeMathGPU('exp', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.sin(a);
            var expected = new Float32Array(a.size);
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.sin(values[i]);
            }
            test_util.expectArraysClose(result.getValues(), expected);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.sin(a).getValues();
            var expected = [Math.sin(4), NaN, Math.sin(0)];
            test_util.expectArraysClose(res, new Float32Array(expected));
            a.dispose();
        });
    };
    test_util.describeMathCPU('sin', [tests]);
    test_util.describeMathGPU('sin', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.cos(a);
            var expected = new Float32Array(a.size);
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.cos(values[i]);
            }
            test_util.expectArraysClose(result.getValues(), expected);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.cos(a).getValues();
            var expected = [Math.cos(4), NaN, Math.cos(0)];
            test_util.expectArraysClose(res, new Float32Array(expected));
            a.dispose();
        });
    };
    test_util.describeMathCPU('cos', [tests]);
    test_util.describeMathGPU('cos', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.tan(a);
            var expected = new Float32Array(a.size);
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.tan(values[i]);
            }
            test_util.expectArraysClose(result.getValues(), expected, 1e-1);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.tan(a).getValues();
            var expected = [Math.tan(4), NaN, Math.tan(0)];
            test_util.expectArraysClose(res, new Float32Array(expected));
            a.dispose();
        });
    };
    test_util.describeMathCPU('tan', [tests]);
    test_util.describeMathGPU('tan', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [.1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.asin(a);
            var expected = new Float32Array(a.size);
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.asin(values[i]);
            }
            test_util.expectArraysClose(result.getValues(), expected, 1e-3);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.asin(a).getValues();
            var expected = [Math.asin(4), NaN, Math.asin(0)];
            test_util.expectArraysClose(res, new Float32Array(expected));
            a.dispose();
        });
    };
    test_util.describeMathCPU('asin', [tests]);
    test_util.describeMathGPU('asin', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [.1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.acos(a);
            var expected = new Float32Array(a.size);
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.acos(values[i]);
            }
            test_util.expectArraysClose(result.getValues(), expected, 1e-1);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.acos(a).getValues();
            var expected = [Math.acos(4), NaN, Math.acos(0)];
            test_util.expectArraysClose(res, new Float32Array(expected));
            a.dispose();
        });
    };
    test_util.describeMathCPU('acos', [tests]);
    test_util.describeMathGPU('acos', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.atan(a);
            var expected = new Float32Array(a.size);
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.atan(values[i]);
            }
            test_util.expectArraysClose(result.getValues(), expected, 1e-3);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.atan(a).getValues();
            var expected = [Math.atan(4), NaN, Math.atan(0)];
            test_util.expectArraysClose(res, new Float32Array(expected));
            a.dispose();
        });
    };
    test_util.describeMathCPU('atan', [tests]);
    test_util.describeMathGPU('atan', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var epsilon_1 = 1e-1;
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.sinh(a);
            var expected = new Float32Array(a.size);
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.sinh(values[i]);
            }
            test_util.expectArraysClose(result.getValues(), expected, epsilon_1);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.sinh(a).getValues();
            var expected = [Math.sinh(4), NaN, Math.sinh(0)];
            test_util.expectArraysClose(res, new Float32Array(expected), epsilon_1);
            a.dispose();
        });
    };
    test_util.describeMathCPU('sinh', [tests]);
    test_util.describeMathGPU('sinh', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var epsilon_2 = 1e-1;
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, -1, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.cosh(a);
            var expected = new Float32Array(a.size);
            for (var i = 0; i < a.size; i++) {
                expected[i] = Math.cosh(values[i]);
            }
            test_util.expectArraysClose(result.getValues(), expected, epsilon_2);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.cosh(a).getValues();
            var expected = [Math.cosh(4), NaN, Math.cosh(0)];
            test_util.expectArraysClose(res, new Float32Array(expected), epsilon_2);
            a.dispose();
        });
    };
    test_util.describeMathCPU('cosh', [tests]);
    test_util.describeMathGPU('cosh', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var values = [1, -3, 2, 7, -4];
            var a = ndarray_1.Array1D.new(values);
            var result = math.tanh(a);
            var expected = new Float32Array(a.size);
            for (var i = 0; i < a.size; i++) {
                expected[i] = util.tanh(values[i]);
            }
            test_util.expectArraysClose(result.getValues(), expected);
            a.dispose();
        });
        it('propagates NaNs', function (math) {
            var a = ndarray_1.Array1D.new([4, NaN, 0]);
            var res = math.tanh(a).getValues();
            var expected = [util.tanh(4), NaN, util.tanh(0)];
            test_util.expectArraysClose(res, new Float32Array(expected));
            a.dispose();
        });
    };
    test_util.describeMathCPU('tanh', [tests]);
    test_util.describeMathGPU('tanh', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('basic', function (math) {
            var a = ndarray_1.Array1D.new([0, 1, -2]);
            var result = math.leakyRelu(a);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result.dataSync(), new Float32Array([0, 1, -0.4]));
        });
        it('propagates NaN', function (math) {
            var a = ndarray_1.Array1D.new([0, 1, NaN]);
            var result = math.leakyRelu(a);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result.dataSync(), new Float32Array([0, 1, NaN]));
        });
    };
    test_util.describeMathCPU('leakyRelu', [tests]);
    test_util.describeMathGPU('leakyRelu', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
{
    var tests = function (it) {
        it('calculate elu', function (math) {
            var a = ndarray_1.Array1D.new([1, -1, 0]);
            var result = math.elu(a);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result.dataSync(), new Float32Array([1, -0.6321, 0]));
        });
        it('elu propagates NaN', function (math) {
            var a = ndarray_1.Array1D.new([1, NaN]);
            var result = math.elu(a);
            expect(result.shape).toEqual(a.shape);
            test_util.expectArraysClose(result.dataSync(), new Float32Array([1, NaN]));
        });
    };
    test_util.describeMathCPU('elu', [tests]);
    test_util.describeMathGPU('elu', [tests], [
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 1 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': true, 'WEBGL_VERSION': 2 },
        { 'WEBGL_FLOAT_TEXTURE_ENABLED': false, 'WEBGL_VERSION': 1 }
    ]);
}
//# sourceMappingURL=unaryop_test.js.map