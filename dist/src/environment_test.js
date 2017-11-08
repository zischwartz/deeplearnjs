"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var device_util = require("./device_util");
var environment_1 = require("./environment");
describe('disjoint query timer enabled', function () {
    it('no webgl', function () {
        var features = { 'WEBGL_VERSION': 0 };
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED')).toBe(false);
    });
    it('webgl 1', function () {
        var features = { 'WEBGL_VERSION': 1 };
        spyOn(document, 'createElement').and.returnValue({
            getContext: function (context) {
                if (context === 'webgl' || context === 'experimental-webgl') {
                    return {
                        getExtension: function (extensionName) {
                            if (extensionName === 'EXT_disjoint_timer_query') {
                                return {};
                            }
                            else if (extensionName === 'WEBGL_lose_context') {
                                return { loseContext: function () { } };
                            }
                            return null;
                        }
                    };
                }
                return null;
            }
        });
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED')).toBe(true);
    });
    it('webgl 2', function () {
        var features = { 'WEBGL_VERSION': 2 };
        spyOn(document, 'createElement').and.returnValue({
            getContext: function (context) {
                if (context === 'webgl2') {
                    return {
                        getExtension: function (extensionName) {
                            if (extensionName === 'EXT_disjoint_timer_query_webgl2') {
                                return {};
                            }
                            else if (extensionName === 'WEBGL_lose_context') {
                                return { loseContext: function () { } };
                            }
                            return null;
                        }
                    };
                }
                return null;
            }
        });
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED')).toBe(true);
    });
});
describe('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE', function () {
    it('disjoint query timer disabled', function () {
        var features = { 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED': false };
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
            .toBe(false);
    });
    it('disjoint query timer enabled, mobile', function () {
        var features = { 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED': true };
        spyOn(device_util, 'isMobile').and.returnValue(true);
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE'))
            .toBe(false);
    });
    it('disjoint query timer enabled, not mobile', function () {
        var features = { 'WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_ENABLED': true };
        spyOn(device_util, 'isMobile').and.returnValue(false);
        var env = new environment_1.Environment(features);
        expect(env.get('WEBGL_DISJOINT_QUERY_TIMER_EXTENSION_RELIABLE')).toBe(true);
    });
});
describe('WebGL version', function () {
    it('webgl 1', function () {
        spyOn(document, 'createElement').and.returnValue({
            getContext: function (context) {
                if (context === 'webgl') {
                    return {
                        getExtension: function (a) {
                            return { loseContext: function () { } };
                        }
                    };
                }
                return null;
            }
        });
        var env = new environment_1.Environment();
        expect(env.get('WEBGL_VERSION')).toBe(1);
    });
    it('webgl 2', function () {
        spyOn(document, 'createElement').and.returnValue({
            getContext: function (context) {
                if (context === 'webgl2') {
                    return {
                        getExtension: function (a) {
                            return { loseContext: function () { } };
                        }
                    };
                }
                return null;
            }
        });
        var env = new environment_1.Environment();
        expect(env.get('WEBGL_VERSION')).toBe(2);
    });
    it('no webgl', function () {
        spyOn(document, 'createElement').and.returnValue({
            getContext: function (context) { return null; }
        });
        var env = new environment_1.Environment();
        expect(env.get('WEBGL_VERSION')).toBe(0);
    });
});
//# sourceMappingURL=environment_test.js.map