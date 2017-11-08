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
var dl = require("../deeplearn");
for (var prop in dl) {
    window[prop] = dl[prop];
}
var GITHUB_JS_FILENAME = 'js';
var GITHUB_HTML_FILENAME = 'html';
var saveButtonElement = document.getElementById('save');
var runButtonElement = document.getElementById('run');
var jscontentElement = document.getElementById('jscontent');
var htmlcontentElement = document.getElementById('htmlcontent');
var gistUrlElement = document.getElementById('gist-url');
var consoleElement = document.getElementById('console');
var htmlconsoleElement = document.getElementById('html');
saveButtonElement.addEventListener('click', function () { return __awaiter(_this, void 0, void 0, function () {
    var jsCodeStr, htmlCodeStr, content, init, result, json;
    return __generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                runCode();
                gistUrlElement.value = '...saving...';
                jsCodeStr = jscontentElement.innerText.trim();
                htmlCodeStr = htmlcontentElement.innerText.trim();
                content = {
                    'description': 'deeplearn.js playground ' + Date.now().toString(),
                    'public': true,
                    'files': {}
                };
                if (jsCodeStr !== '') {
                    content['files'][GITHUB_JS_FILENAME] = { 'content': jsCodeStr };
                }
                if (htmlCodeStr !== '') {
                    content['files'][GITHUB_HTML_FILENAME] = { 'content': htmlCodeStr };
                }
                init = { method: 'POST', body: JSON.stringify(content) };
                return [4, fetch('https://api.github.com/gists', init)];
            case 1:
                result = _a.sent();
                return [4, result.json()];
            case 2:
                json = _a.sent();
                gistUrlElement.value = json['html_url'];
                window.location.hash = "#" + json['id'];
                return [2];
        }
    });
}); });
function loadGistFromURL() {
    return __awaiter(this, void 0, void 0, function () {
        var gistId, result, json, jsFile, jsResult, jsCode, htmlFile, htmlResult, htmlCode;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    if (!(window.location.hash && window.location.hash !== '#')) return [3, 9];
                    gistUrlElement.value = '...loading...';
                    gistId = window.location.hash.substr(1);
                    return [4, fetch('https://api.github.com/gists/' + gistId)];
                case 1:
                    result = _a.sent();
                    return [4, result.json()];
                case 2:
                    json = _a.sent();
                    gistUrlElement.value = json['html_url'];
                    if (!(json['files'][GITHUB_JS_FILENAME] != null)) return [3, 5];
                    jsFile = json['files'][GITHUB_JS_FILENAME]['raw_url'];
                    return [4, fetch(jsFile)];
                case 3:
                    jsResult = _a.sent();
                    return [4, jsResult.text()];
                case 4:
                    jsCode = _a.sent();
                    jscontentElement.innerText = jsCode;
                    _a.label = 5;
                case 5:
                    if (!(json['files'][GITHUB_HTML_FILENAME] != null)) return [3, 8];
                    htmlFile = json['files'][GITHUB_HTML_FILENAME]['raw_url'];
                    return [4, fetch(htmlFile)];
                case 6:
                    htmlResult = _a.sent();
                    return [4, htmlResult.text()];
                case 7:
                    htmlCode = _a.sent();
                    htmlcontentElement.innerText = htmlCode;
                    _a.label = 8;
                case 8: return [3, 10];
                case 9:
                    gistUrlElement.value = 'Unsaved';
                    _a.label = 10;
                case 10: return [2];
            }
        });
    });
}
window.console.log = function (str) {
    consoleElement.innerText += str + '\n';
};
function runCode() {
    return __awaiter(this, void 0, void 0, function () {
        var error;
        return __generator(this, function (_a) {
            htmlconsoleElement.innerHTML = htmlcontentElement.innerText;
            consoleElement.innerText = '';
            try {
                eval("(async () => {\n      " + jscontentElement.innerText + "\n    })();");
            }
            catch (e) {
                error = new Error();
                window.console.log(e.toString());
                window.console.log(error.stack);
            }
            return [2];
        });
    });
}
runButtonElement.addEventListener('click', runCode);
loadGistFromURL();
//# sourceMappingURL=playground.js.map