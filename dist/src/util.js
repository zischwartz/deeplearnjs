"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function shuffle(array) {
    var counter = array.length;
    var temp = 0;
    var index = 0;
    while (counter > 0) {
        index = (Math.random() * counter) | 0;
        counter--;
        temp = array[counter];
        array[counter] = array[index];
        array[index] = temp;
    }
}
exports.shuffle = shuffle;
function clamp(min, x, max) {
    return Math.max(min, Math.min(x, max));
}
exports.clamp = clamp;
function randUniform(a, b) {
    return Math.random() * (b - a) + a;
}
exports.randUniform = randUniform;
function randGauss(mean, stdDev, truncated) {
    if (mean === void 0) { mean = 0; }
    if (stdDev === void 0) { stdDev = 1; }
    if (truncated === void 0) { truncated = false; }
    var v1, v2, s;
    do {
        v1 = 2 * Math.random() - 1;
        v2 = 2 * Math.random() - 1;
        s = v1 * v1 + v2 * v2;
    } while (s > 1);
    var result = Math.sqrt(-2 * Math.log(s) / s) * v1;
    if (truncated && Math.abs(result) > 2) {
        return randGauss(mean, stdDev, true);
    }
    return mean + stdDev * result;
}
exports.randGauss = randGauss;
function distSquared(a, b) {
    var result = 0;
    for (var i = 0; i < a.length; i++) {
        var diff = Number(a[i]) - Number(b[i]);
        result += diff * diff;
    }
    return result;
}
exports.distSquared = distSquared;
function assert(expr, msg) {
    if (!expr) {
        throw new Error(msg);
    }
}
exports.assert = assert;
function assertShapesMatch(shapeA, shapeB, errorMessagePrefix) {
    if (errorMessagePrefix === void 0) { errorMessagePrefix = ''; }
    assert(arraysEqual(shapeA, shapeB), errorMessagePrefix + ("Shapes " + shapeA + " and " + shapeB + " must match"));
}
exports.assertShapesMatch = assertShapesMatch;
function flatten(arr, ret) {
    if (ret === void 0) { ret = []; }
    if (Array.isArray(arr)) {
        for (var i = 0; i < arr.length; ++i) {
            flatten(arr[i], ret);
        }
    }
    else {
        ret.push(arr);
    }
    return ret;
}
exports.flatten = flatten;
function inferShape(arr) {
    var shape = [];
    while (arr instanceof Array) {
        shape.push(arr.length);
        arr = arr[0];
    }
    return shape;
}
exports.inferShape = inferShape;
function sizeFromShape(shape) {
    if (shape.length === 0) {
        return 1;
    }
    var size = shape[0];
    for (var i = 1; i < shape.length; i++) {
        size *= shape[i];
    }
    return size;
}
exports.sizeFromShape = sizeFromShape;
function isScalarShape(shape) {
    return shape.length === 0;
}
exports.isScalarShape = isScalarShape;
function arraysEqual(n1, n2) {
    if (n1.length !== n2.length) {
        return false;
    }
    for (var i = 0; i < n1.length; i++) {
        if (n1[i] !== n2[i]) {
            return false;
        }
    }
    return true;
}
exports.arraysEqual = arraysEqual;
function isInt(a) {
    return a % 1 === 0;
}
exports.isInt = isInt;
function tanh(x) {
    if (Math.tanh != null) {
        return Math.tanh(x);
    }
    if (x === Infinity) {
        return 1;
    }
    else if (x === -Infinity) {
        return -1;
    }
    else {
        var e2x = Math.exp(2 * x);
        return (e2x - 1) / (e2x + 1);
    }
}
exports.tanh = tanh;
function sizeToSquarishShape(size) {
    for (var a = Math.floor(Math.sqrt(size)); a > 1; --a) {
        if (size % a === 0) {
            return [a, size / a];
        }
    }
    return [1, size];
}
exports.sizeToSquarishShape = sizeToSquarishShape;
function createShuffledIndices(n) {
    var shuffledIndices = new Uint32Array(n);
    for (var i = 0; i < n; ++i) {
        shuffledIndices[i] = i;
    }
    shuffle(shuffledIndices);
    return shuffledIndices;
}
exports.createShuffledIndices = createShuffledIndices;
function rightPad(a, size) {
    if (size <= a.length) {
        return a;
    }
    return a + ' '.repeat(size - a.length);
}
exports.rightPad = rightPad;
function repeatedTry(checkFn, delayFn, maxCounter) {
    if (delayFn === void 0) { delayFn = function (counter) { return 0; }; }
    return new Promise(function (resolve, reject) {
        var tryCount = 0;
        var tryFn = function () {
            if (checkFn()) {
                resolve();
                return;
            }
            tryCount++;
            var nextBackoff = delayFn(tryCount);
            if (maxCounter != null && tryCount >= maxCounter) {
                reject();
                return;
            }
            setTimeout(tryFn, nextBackoff);
        };
        setTimeout(tryFn, 0);
    });
}
exports.repeatedTry = repeatedTry;
function getQueryParams(queryString) {
    var params = {};
    queryString.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g, function (s) {
        var t = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            t[_i - 1] = arguments[_i];
        }
        decodeParam(params, t[0], t[1]);
        return t.join('=');
    });
    return params;
}
exports.getQueryParams = getQueryParams;
function decodeParam(params, name, value) {
    params[decodeURIComponent(name)] = decodeURIComponent(value || '');
}
function inferFromImplicitShape(shape, size) {
    var shapeProd = 1;
    var implicitIdx = -1;
    for (var i = 0; i < shape.length; ++i) {
        if (shape[i] > 0) {
            shapeProd *= shape[i];
        }
        else if (shape[i] === -1) {
            if (implicitIdx !== -1) {
                throw Error("Shapes can only have 1 implicit size. " +
                    ("Found -1 at dim " + implicitIdx + " and dim " + i));
            }
            implicitIdx = i;
        }
        else if (shape[i] <= 0) {
            throw Error("Shapes can not be <= 0. Found " + shape[i] + " at dim " + i);
        }
    }
    if (implicitIdx === -1) {
        if (size > 0 && size !== shapeProd) {
            throw Error("Size (" + size + ") must match the product of shape " + shape);
        }
        return shape;
    }
    if (size % shapeProd !== 0) {
        throw Error("The implicit shape can't be a fractional number. " +
            ("Got " + size + " / " + shapeProd));
    }
    var newShape = shape.slice();
    newShape[implicitIdx] = size / shapeProd;
    return newShape;
}
exports.inferFromImplicitShape = inferFromImplicitShape;
exports.NAN_INT32 = 1 << 31;
exports.NAN_BOOL = 255;
exports.NAN_FLOAT32 = NaN;
function getNaN(dtype) {
    if (dtype === 'float32') {
        return exports.NAN_FLOAT32;
    }
    else if (dtype === 'int32') {
        return exports.NAN_INT32;
    }
    else if (dtype === 'bool') {
        return exports.NAN_BOOL;
    }
    else {
        throw new Error("Unknown dtype " + dtype);
    }
}
exports.getNaN = getNaN;
function isValNaN(val, dtype) {
    if (dtype === 'float32') {
        return isNaN(val);
    }
    else if (dtype === 'int32') {
        return val === exports.NAN_INT32;
    }
    else if (dtype === 'bool') {
        return val === exports.NAN_BOOL;
    }
    else {
        throw new Error("Unknown dtype " + dtype);
    }
}
exports.isValNaN = isValNaN;
//# sourceMappingURL=util.js.map