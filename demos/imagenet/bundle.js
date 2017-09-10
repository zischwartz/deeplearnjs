(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
"use strict";
function __export(m) {
    for (var p in m) if (!exports.hasOwnProperty(p)) exports[p] = m[p];
}
Object.defineProperty(exports, "__esModule", { value: true });
__export(require("../src/index"));

},{"../src/index":15}],2:[function(require,module,exports){
Polymer({ is: 'demo-footer' });

},{}],3:[function(require,module,exports){
Polymer({ is: 'demo-header' });

},{}],4:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
require("../demo-header");
require("../demo-footer");
var deeplearnjs_1 = require("../deeplearnjs");
var imagenet_util = require("../models/imagenet_util");
var squeezenet_1 = require("../models/squeezenet");
var polymer_spec_1 = require("../polymer-spec");
exports.ImagenetDemoPolymer = polymer_spec_1.PolymerElement({
    is: 'imagenet-demo',
    properties: {
        layerNames: Array,
        selectedLayerName: String,
        inputNames: Array,
        selectedInputName: String
    }
});
var IMAGE_SIZE = 227;
var TOP_K_CLASSES = 5;
var INPUT_NAMES = ['cat', 'dog1', 'dog2', 'beerbottle', 'piano', 'saxophone'];
var ImagenetDemo = (function (_super) {
    __extends(ImagenetDemo, _super);
    function ImagenetDemo() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    ImagenetDemo.prototype.ready = function () {
        var _this = this;
        this.inferenceCanvas =
            this.querySelector('#inference-canvas');
        this.staticImgElement =
            this.querySelector('#staticImg');
        this.webcamVideoElement =
            this.querySelector('#webcamVideo');
        this.layerNames = [];
        this.selectedLayerName = 'conv_1';
        var inputDropdown = this.querySelector('#input-dropdown');
        inputDropdown.addEventListener('iron-activate', function (event) {
            var selectedInputName = event.detail.selected;
            if (selectedInputName === 'webcam') {
                _this.webcamVideoElement.style.display = '';
                _this.staticImgElement.style.display = 'none';
            }
            else {
                _this.webcamVideoElement.style.display = 'none';
                _this.staticImgElement.style.display = '';
            }
            _this.staticImgElement.src = 'images/' + event.detail.selected + '.jpg';
        });
        var navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
            navigatorAny.msGetUserMedia;
        if (navigator.getUserMedia) {
            navigator.getUserMedia({ video: true }, function (stream) {
                _this.webcamVideoElement.src = window.URL.createObjectURL(stream);
                _this.initWithWebcam();
            }, function (error) {
                console.log(error);
                _this.initWithoutWebcam();
            });
        }
        else {
            this.initWithoutWebcam();
        }
        this.gl = deeplearnjs_1.gpgpu_util.createWebGLContext(this.inferenceCanvas);
        this.gpgpu = new deeplearnjs_1.GPGPUContext(this.gl);
        this.math = new deeplearnjs_1.NDArrayMathGPU(this.gpgpu);
        this.mathCPU = new deeplearnjs_1.NDArrayMathCPU();
        this.squeezeNet = new squeezenet_1.SqueezeNet(this.gpgpu, this.math);
        this.squeezeNet.loadVariables().then(function () {
            requestAnimationFrame(function () { return _this.animate(); });
        });
        this.renderGrayscaleChannelsCollageShader =
            imagenet_util.getRenderGrayscaleChannelsCollageShader(this.gpgpu);
    };
    ImagenetDemo.prototype.initWithoutWebcam = function () {
        this.inputNames = INPUT_NAMES;
        this.selectedInputName = 'cat';
        this.staticImgElement.src = 'images/cat.jpg';
        this.webcamVideoElement.style.display = 'none';
        this.staticImgElement.style.display = '';
        if (location.protocol !== 'https:') {
            this.querySelector('#ssl-message').style.display =
                'block';
        }
        this.querySelector('#webcam-message').style.display =
            'block';
    };
    ImagenetDemo.prototype.initWithWebcam = function () {
        var inputNames = INPUT_NAMES.slice();
        inputNames.unshift('webcam');
        this.inputNames = inputNames;
        this.selectedInputName = 'webcam';
    };
    ImagenetDemo.prototype.animate = function () {
        var _this = this;
        var startTime = performance.now();
        var isWebcam = this.selectedInputName === 'webcam';
        var canvasTextureShape = [IMAGE_SIZE, IMAGE_SIZE];
        var canvasTexture = this.math.getTextureManager().acquireTexture(canvasTextureShape);
        var element = isWebcam ? this.webcamVideoElement : this.staticImgElement;
        this.gpgpu.uploadPixelDataToTexture(canvasTexture, element);
        this.math.scope(function (keep, track) {
            var preprocessedInput = track(_this.squeezeNet.preprocessColorTextureToArray3D(canvasTexture, canvasTextureShape));
            var inferenceResult = _this.squeezeNet.infer(preprocessedInput);
            var namedActivations = inferenceResult.namedActivations;
            _this.layerNames = Object.keys(namedActivations);
            _this.layerNames.forEach(function (layerName) { return track(namedActivations[layerName]); });
            var topClassesToProbability = _this.squeezeNet.getTopKClasses(inferenceResult.logits, TOP_K_CLASSES);
            var count = 0;
            for (var className in topClassesToProbability) {
                if (!(className in topClassesToProbability)) {
                    continue;
                }
                document.getElementById('class' + count).innerHTML = className;
                document.getElementById('prob' + count).innerHTML =
                    '' + Math.floor(1000 * topClassesToProbability[className]) / 1000;
                count++;
            }
            var endTime = performance.now();
            _this.querySelector('#totalTime').innerHTML =
                'last inference time: ' +
                    Math.floor(1000 * (endTime - startTime)) / 1000 + 'ms';
            var activationNDArray = namedActivations[_this.selectedLayerName];
            var maxValues = _this.math.maxPool(activationNDArray, activationNDArray.shape[1], activationNDArray.shape[1], 0);
            var minValues = _this.math.minPool(activationNDArray, activationNDArray.shape[1], activationNDArray.shape[1], 0);
            var imagesPerRow = Math.ceil(Math.sqrt(activationNDArray.shape[2]));
            var numRows = Math.ceil(activationNDArray.shape[2] / imagesPerRow);
            _this.inferenceCanvas.width = imagesPerRow * activationNDArray.shape[0];
            _this.inferenceCanvas.height = numRows * activationNDArray.shape[0];
            imagenet_util.renderGrayscaleChannelsCollage(_this.gpgpu, _this.renderGrayscaleChannelsCollageShader, activationNDArray.getTexture(), minValues.getTexture(), maxValues.getTexture(), activationNDArray.getTextureShapeRC(), activationNDArray.shape[0], activationNDArray.shape[2], _this.inferenceCanvas.width, numRows);
        });
        this.math.getTextureManager().releaseTexture(canvasTexture, canvasTextureShape);
        requestAnimationFrame(function () { return _this.animate(); });
    };
    return ImagenetDemo;
}(exports.ImagenetDemoPolymer));
exports.ImagenetDemo = ImagenetDemo;
document.registerElement(ImagenetDemo.prototype.is, ImagenetDemo);

},{"../deeplearnjs":1,"../demo-footer":2,"../demo-header":3,"../models/imagenet_util":6,"../models/squeezenet":7,"../polymer-spec":8}],5:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.IMAGENET_CLASSES = {
    0: 'tench, Tinca tinca',
    1: 'goldfish, Carassius auratus',
    2: 'great white shark, white shark, man-eater, man-eating shark, ' +
        'Carcharodon carcharias',
    3: 'tiger shark, Galeocerdo cuvieri',
    4: 'hammerhead, hammerhead shark',
    5: 'electric ray, crampfish, numbfish, torpedo',
    6: 'stingray',
    7: 'cock',
    8: 'hen',
    9: 'ostrich, Struthio camelus',
    10: 'brambling, Fringilla montifringilla',
    11: 'goldfinch, Carduelis carduelis',
    12: 'house finch, linnet, Carpodacus mexicanus',
    13: 'junco, snowbird',
    14: 'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
    15: 'robin, American robin, Turdus migratorius',
    16: 'bulbul',
    17: 'jay',
    18: 'magpie',
    19: 'chickadee',
    20: 'water ouzel, dipper',
    21: 'kite',
    22: 'bald eagle, American eagle, Haliaeetus leucocephalus',
    23: 'vulture',
    24: 'great grey owl, great gray owl, Strix nebulosa',
    25: 'European fire salamander, Salamandra salamandra',
    26: 'common newt, Triturus vulgaris',
    27: 'eft',
    28: 'spotted salamander, Ambystoma maculatum',
    29: 'axolotl, mud puppy, Ambystoma mexicanum',
    30: 'bullfrog, Rana catesbeiana',
    31: 'tree frog, tree-frog',
    32: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
    33: 'loggerhead, loggerhead turtle, Caretta caretta',
    34: 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
    35: 'mud turtle',
    36: 'terrapin',
    37: 'box turtle, box tortoise',
    38: 'banded gecko',
    39: 'common iguana, iguana, Iguana iguana',
    40: 'American chameleon, anole, Anolis carolinensis',
    41: 'whiptail, whiptail lizard',
    42: 'agama',
    43: 'frilled lizard, Chlamydosaurus kingi',
    44: 'alligator lizard',
    45: 'Gila monster, Heloderma suspectum',
    46: 'green lizard, Lacerta viridis',
    47: 'African chameleon, Chamaeleo chamaeleon',
    48: 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, ' +
        'Varanus komodoensis',
    49: 'African crocodile, Nile crocodile, Crocodylus niloticus',
    50: 'American alligator, Alligator mississipiensis',
    51: 'triceratops',
    52: 'thunder snake, worm snake, Carphophis amoenus',
    53: 'ringneck snake, ring-necked snake, ring snake',
    54: 'hognose snake, puff adder, sand viper',
    55: 'green snake, grass snake',
    56: 'king snake, kingsnake',
    57: 'garter snake, grass snake',
    58: 'water snake',
    59: 'vine snake',
    60: 'night snake, Hypsiglena torquata',
    61: 'boa constrictor, Constrictor constrictor',
    62: 'rock python, rock snake, Python sebae',
    63: 'Indian cobra, Naja naja',
    64: 'green mamba',
    65: 'sea snake',
    66: 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
    67: 'diamondback, diamondback rattlesnake, Crotalus adamanteus',
    68: 'sidewinder, horned rattlesnake, Crotalus cerastes',
    69: 'trilobite',
    70: 'harvestman, daddy longlegs, Phalangium opilio',
    71: 'scorpion',
    72: 'black and gold garden spider, Argiope aurantia',
    73: 'barn spider, Araneus cavaticus',
    74: 'garden spider, Aranea diademata',
    75: 'black widow, Latrodectus mactans',
    76: 'tarantula',
    77: 'wolf spider, hunting spider',
    78: 'tick',
    79: 'centipede',
    80: 'black grouse',
    81: 'ptarmigan',
    82: 'ruffed grouse, partridge, Bonasa umbellus',
    83: 'prairie chicken, prairie grouse, prairie fowl',
    84: 'peacock',
    85: 'quail',
    86: 'partridge',
    87: 'African grey, African gray, Psittacus erithacus',
    88: 'macaw',
    89: 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
    90: 'lorikeet',
    91: 'coucal',
    92: 'bee eater',
    93: 'hornbill',
    94: 'hummingbird',
    95: 'jacamar',
    96: 'toucan',
    97: 'drake',
    98: 'red-breasted merganser, Mergus serrator',
    99: 'goose',
    100: 'black swan, Cygnus atratus',
    101: 'tusker',
    102: 'echidna, spiny anteater, anteater',
    103: 'platypus, duckbill, duckbilled platypus, duck-billed platypus, ' +
        'Ornithorhynchus anatinus',
    104: 'wallaby, brush kangaroo',
    105: 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
    106: 'wombat',
    107: 'jelly fish',
    108: 'sea anemone, anemone',
    109: 'brain coral',
    110: 'flatworm, platyhelminth',
    111: 'nematode, nematode worm, roundworm',
    112: 'conch',
    113: 'snail',
    114: 'slug',
    115: 'sea slug, nudibranch',
    116: 'chiton, coat-of-mail shell, sea cradle, polyplacophore',
    117: 'chambered nautilus, pearly nautilus, nautilus',
    118: 'Dungeness crab, Cancer magister',
    119: 'rock crab, Cancer irroratus',
    120: 'fiddler crab',
    121: 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, ' +
        'Paralithodes camtschatica',
    122: 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
    123: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea ' +
        'crawfish',
    124: 'crayfish, crawfish, crawdad, crawdaddy',
    125: 'hermit crab',
    126: 'isopod',
    127: 'white stork, Ciconia ciconia',
    128: 'black stork, Ciconia nigra',
    129: 'spoonbill',
    130: 'flamingo',
    131: 'little blue heron, Egretta caerulea',
    132: 'American egret, great white heron, Egretta albus',
    133: 'bittern',
    134: 'crane',
    135: 'limpkin, Aramus pictus',
    136: 'European gallinule, Porphyrio porphyrio',
    137: 'American coot, marsh hen, mud hen, water hen, Fulica americana',
    138: 'bustard',
    139: 'ruddy turnstone, Arenaria interpres',
    140: 'red-backed sandpiper, dunlin, Erolia alpina',
    141: 'redshank, Tringa totanus',
    142: 'dowitcher',
    143: 'oystercatcher, oyster catcher',
    144: 'pelican',
    145: 'king penguin, Aptenodytes patagonica',
    146: 'albatross, mollymawk',
    147: 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, ' +
        'Eschrichtius robustus',
    148: 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca',
    149: 'dugong, Dugong dugon',
    150: 'sea lion',
    151: 'Chihuahua',
    152: 'Japanese spaniel',
    153: 'Maltese dog, Maltese terrier, Maltese',
    154: 'Pekinese, Pekingese, Peke',
    155: 'Shih-Tzu',
    156: 'Blenheim spaniel',
    157: 'papillon',
    158: 'toy terrier',
    159: 'Rhodesian ridgeback',
    160: 'Afghan hound, Afghan',
    161: 'basset, basset hound',
    162: 'beagle',
    163: 'bloodhound, sleuthhound',
    164: 'bluetick',
    165: 'black-and-tan coonhound',
    166: 'Walker hound, Walker foxhound',
    167: 'English foxhound',
    168: 'redbone',
    169: 'borzoi, Russian wolfhound',
    170: 'Irish wolfhound',
    171: 'Italian greyhound',
    172: 'whippet',
    173: 'Ibizan hound, Ibizan Podenco',
    174: 'Norwegian elkhound, elkhound',
    175: 'otterhound, otter hound',
    176: 'Saluki, gazelle hound',
    177: 'Scottish deerhound, deerhound',
    178: 'Weimaraner',
    179: 'Staffordshire bullterrier, Staffordshire bull terrier',
    180: 'American Staffordshire terrier, Staffordshire terrier, American pit ' +
        'bull terrier, pit bull terrier',
    181: 'Bedlington terrier',
    182: 'Border terrier',
    183: 'Kerry blue terrier',
    184: 'Irish terrier',
    185: 'Norfolk terrier',
    186: 'Norwich terrier',
    187: 'Yorkshire terrier',
    188: 'wire-haired fox terrier',
    189: 'Lakeland terrier',
    190: 'Sealyham terrier, Sealyham',
    191: 'Airedale, Airedale terrier',
    192: 'cairn, cairn terrier',
    193: 'Australian terrier',
    194: 'Dandie Dinmont, Dandie Dinmont terrier',
    195: 'Boston bull, Boston terrier',
    196: 'miniature schnauzer',
    197: 'giant schnauzer',
    198: 'standard schnauzer',
    199: 'Scotch terrier, Scottish terrier, Scottie',
    200: 'Tibetan terrier, chrysanthemum dog',
    201: 'silky terrier, Sydney silky',
    202: 'soft-coated wheaten terrier',
    203: 'West Highland white terrier',
    204: 'Lhasa, Lhasa apso',
    205: 'flat-coated retriever',
    206: 'curly-coated retriever',
    207: 'golden retriever',
    208: 'Labrador retriever',
    209: 'Chesapeake Bay retriever',
    210: 'German short-haired pointer',
    211: 'vizsla, Hungarian pointer',
    212: 'English setter',
    213: 'Irish setter, red setter',
    214: 'Gordon setter',
    215: 'Brittany spaniel',
    216: 'clumber, clumber spaniel',
    217: 'English springer, English springer spaniel',
    218: 'Welsh springer spaniel',
    219: 'cocker spaniel, English cocker spaniel, cocker',
    220: 'Sussex spaniel',
    221: 'Irish water spaniel',
    222: 'kuvasz',
    223: 'schipperke',
    224: 'groenendael',
    225: 'malinois',
    226: 'briard',
    227: 'kelpie',
    228: 'komondor',
    229: 'Old English sheepdog, bobtail',
    230: 'Shetland sheepdog, Shetland sheep dog, Shetland',
    231: 'collie',
    232: 'Border collie',
    233: 'Bouvier des Flandres, Bouviers des Flandres',
    234: 'Rottweiler',
    235: 'German shepherd, German shepherd dog, German police dog, alsatian',
    236: 'Doberman, Doberman pinscher',
    237: 'miniature pinscher',
    238: 'Greater Swiss Mountain dog',
    239: 'Bernese mountain dog',
    240: 'Appenzeller',
    241: 'EntleBucher',
    242: 'boxer',
    243: 'bull mastiff',
    244: 'Tibetan mastiff',
    245: 'French bulldog',
    246: 'Great Dane',
    247: 'Saint Bernard, St Bernard',
    248: 'Eskimo dog, husky',
    249: 'malamute, malemute, Alaskan malamute',
    250: 'Siberian husky',
    251: 'dalmatian, coach dog, carriage dog',
    252: 'affenpinscher, monkey pinscher, monkey dog',
    253: 'basenji',
    254: 'pug, pug-dog',
    255: 'Leonberg',
    256: 'Newfoundland, Newfoundland dog',
    257: 'Great Pyrenees',
    258: 'Samoyed, Samoyede',
    259: 'Pomeranian',
    260: 'chow, chow chow',
    261: 'keeshond',
    262: 'Brabancon griffon',
    263: 'Pembroke, Pembroke Welsh corgi',
    264: 'Cardigan, Cardigan Welsh corgi',
    265: 'toy poodle',
    266: 'miniature poodle',
    267: 'standard poodle',
    268: 'Mexican hairless',
    269: 'timber wolf, grey wolf, gray wolf, Canis lupus',
    270: 'white wolf, Arctic wolf, Canis lupus tundrarum',
    271: 'red wolf, maned wolf, Canis rufus, Canis niger',
    272: 'coyote, prairie wolf, brush wolf, Canis latrans',
    273: 'dingo, warrigal, warragal, Canis dingo',
    274: 'dhole, Cuon alpinus',
    275: 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus',
    276: 'hyena, hyaena',
    277: 'red fox, Vulpes vulpes',
    278: 'kit fox, Vulpes macrotis',
    279: 'Arctic fox, white fox, Alopex lagopus',
    280: 'grey fox, gray fox, Urocyon cinereoargenteus',
    281: 'tabby, tabby cat',
    282: 'tiger cat',
    283: 'Persian cat',
    284: 'Siamese cat, Siamese',
    285: 'Egyptian cat',
    286: 'cougar, puma, catamount, mountain lion, painter, panther, ' +
        'Felis concolor',
    287: 'lynx, catamount',
    288: 'leopard, Panthera pardus',
    289: 'snow leopard, ounce, Panthera uncia',
    290: 'jaguar, panther, Panthera onca, Felis onca',
    291: 'lion, king of beasts, Panthera leo',
    292: 'tiger, Panthera tigris',
    293: 'cheetah, chetah, Acinonyx jubatus',
    294: 'brown bear, bruin, Ursus arctos',
    295: 'American black bear, black bear, Ursus americanus, Euarctos ' +
        'americanus',
    296: 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',
    297: 'sloth bear, Melursus ursinus, Ursus ursinus',
    298: 'mongoose',
    299: 'meerkat, mierkat',
    300: 'tiger beetle',
    301: 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
    302: 'ground beetle, carabid beetle',
    303: 'long-horned beetle, longicorn, longicorn beetle',
    304: 'leaf beetle, chrysomelid',
    305: 'dung beetle',
    306: 'rhinoceros beetle',
    307: 'weevil',
    308: 'fly',
    309: 'bee',
    310: 'ant, emmet, pismire',
    311: 'grasshopper, hopper',
    312: 'cricket',
    313: 'walking stick, walkingstick, stick insect',
    314: 'cockroach, roach',
    315: 'mantis, mantid',
    316: 'cicada, cicala',
    317: 'leafhopper',
    318: 'lacewing, lacewing fly',
    319: 'dragonfly, darning needle, devil\'s darning needle, sewing needle, ' +
        'snake feeder, snake doctor, mosquito hawk, skeeter hawk',
    320: 'damselfly',
    321: 'admiral',
    322: 'ringlet, ringlet butterfly',
    323: 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
    324: 'cabbage butterfly',
    325: 'sulphur butterfly, sulfur butterfly',
    326: 'lycaenid, lycaenid butterfly',
    327: 'starfish, sea star',
    328: 'sea urchin',
    329: 'sea cucumber, holothurian',
    330: 'wood rabbit, cottontail, cottontail rabbit',
    331: 'hare',
    332: 'Angora, Angora rabbit',
    333: 'hamster',
    334: 'porcupine, hedgehog',
    335: 'fox squirrel, eastern fox squirrel, Sciurus niger',
    336: 'marmot',
    337: 'beaver',
    338: 'guinea pig, Cavia cobaya',
    339: 'sorrel',
    340: 'zebra',
    341: 'hog, pig, grunter, squealer, Sus scrofa',
    342: 'wild boar, boar, Sus scrofa',
    343: 'warthog',
    344: 'hippopotamus, hippo, river horse, Hippopotamus amphibius',
    345: 'ox',
    346: 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',
    347: 'bison',
    348: 'ram, tup',
    349: 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky ' +
        'Mountain sheep, Ovis canadensis',
    350: 'ibex, Capra ibex',
    351: 'hartebeest',
    352: 'impala, Aepyceros melampus',
    353: 'gazelle',
    354: 'Arabian camel, dromedary, Camelus dromedarius',
    355: 'llama',
    356: 'weasel',
    357: 'mink',
    358: 'polecat, fitch, foulmart, foumart, Mustela putorius',
    359: 'black-footed ferret, ferret, Mustela nigripes',
    360: 'otter',
    361: 'skunk, polecat, wood pussy',
    362: 'badger',
    363: 'armadillo',
    364: 'three-toed sloth, ai, Bradypus tridactylus',
    365: 'orangutan, orang, orangutang, Pongo pygmaeus',
    366: 'gorilla, Gorilla gorilla',
    367: 'chimpanzee, chimp, Pan troglodytes',
    368: 'gibbon, Hylobates lar',
    369: 'siamang, Hylobates syndactylus, Symphalangus syndactylus',
    370: 'guenon, guenon monkey',
    371: 'patas, hussar monkey, Erythrocebus patas',
    372: 'baboon',
    373: 'macaque',
    374: 'langur',
    375: 'colobus, colobus monkey',
    376: 'proboscis monkey, Nasalis larvatus',
    377: 'marmoset',
    378: 'capuchin, ringtail, Cebus capucinus',
    379: 'howler monkey, howler',
    380: 'titi, titi monkey',
    381: 'spider monkey, Ateles geoffroyi',
    382: 'squirrel monkey, Saimiri sciureus',
    383: 'Madagascar cat, ring-tailed lemur, Lemur catta',
    384: 'indri, indris, Indri indri, Indri brevicaudatus',
    385: 'Indian elephant, Elephas maximus',
    386: 'African elephant, Loxodonta africana',
    387: 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
    388: 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
    389: 'barracouta, snoek',
    390: 'eel',
    391: 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus ' +
        'kisutch',
    392: 'rock beauty, Holocanthus tricolor',
    393: 'anemone fish',
    394: 'sturgeon',
    395: 'gar, garfish, garpike, billfish, Lepisosteus osseus',
    396: 'lionfish',
    397: 'puffer, pufferfish, blowfish, globefish',
    398: 'abacus',
    399: 'abaya',
    400: 'academic gown, academic robe, judge\'s robe',
    401: 'accordion, piano accordion, squeeze box',
    402: 'acoustic guitar',
    403: 'aircraft carrier, carrier, flattop, attack aircraft carrier',
    404: 'airliner',
    405: 'airship, dirigible',
    406: 'altar',
    407: 'ambulance',
    408: 'amphibian, amphibious vehicle',
    409: 'analog clock',
    410: 'apiary, bee house',
    411: 'apron',
    412: 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, ' +
        'dustbin, trash barrel, trash bin',
    413: 'assault rifle, assault gun',
    414: 'backpack, back pack, knapsack, packsack, rucksack, haversack',
    415: 'bakery, bakeshop, bakehouse',
    416: 'balance beam, beam',
    417: 'balloon',
    418: 'ballpoint, ballpoint pen, ballpen, Biro',
    419: 'Band Aid',
    420: 'banjo',
    421: 'bannister, banister, balustrade, balusters, handrail',
    422: 'barbell',
    423: 'barber chair',
    424: 'barbershop',
    425: 'barn',
    426: 'barometer',
    427: 'barrel, cask',
    428: 'barrow, garden cart, lawn cart, wheelbarrow',
    429: 'baseball',
    430: 'basketball',
    431: 'bassinet',
    432: 'bassoon',
    433: 'bathing cap, swimming cap',
    434: 'bath towel',
    435: 'bathtub, bathing tub, bath, tub',
    436: 'beach wagon, station wagon, wagon, estate car, beach waggon, station ' +
        'waggon, waggon',
    437: 'beacon, lighthouse, beacon light, pharos',
    438: 'beaker',
    439: 'bearskin, busby, shako',
    440: 'beer bottle',
    441: 'beer glass',
    442: 'bell cote, bell cot',
    443: 'bib',
    444: 'bicycle-built-for-two, tandem bicycle, tandem',
    445: 'bikini, two-piece',
    446: 'binder, ring-binder',
    447: 'binoculars, field glasses, opera glasses',
    448: 'birdhouse',
    449: 'boathouse',
    450: 'bobsled, bobsleigh, bob',
    451: 'bolo tie, bolo, bola tie, bola',
    452: 'bonnet, poke bonnet',
    453: 'bookcase',
    454: 'bookshop, bookstore, bookstall',
    455: 'bottlecap',
    456: 'bow',
    457: 'bow tie, bow-tie, bowtie',
    458: 'brass, memorial tablet, plaque',
    459: 'brassiere, bra, bandeau',
    460: 'breakwater, groin, groyne, mole, bulwark, seawall, jetty',
    461: 'breastplate, aegis, egis',
    462: 'broom',
    463: 'bucket, pail',
    464: 'buckle',
    465: 'bulletproof vest',
    466: 'bullet train, bullet',
    467: 'butcher shop, meat market',
    468: 'cab, hack, taxi, taxicab',
    469: 'caldron, cauldron',
    470: 'candle, taper, wax light',
    471: 'cannon',
    472: 'canoe',
    473: 'can opener, tin opener',
    474: 'cardigan',
    475: 'car mirror',
    476: 'carousel, carrousel, merry-go-round, roundabout, whirligig',
    477: 'carpenter\'s kit, tool kit',
    478: 'carton',
    479: 'car wheel',
    480: 'cash machine, cash dispenser, automated teller machine, automatic ' +
        'teller machine, automated teller, automatic teller, ATM',
    481: 'cassette',
    482: 'cassette player',
    483: 'castle',
    484: 'catamaran',
    485: 'CD player',
    486: 'cello, violoncello',
    487: 'cellular telephone, cellular phone, cellphone, cell, mobile phone',
    488: 'chain',
    489: 'chainlink fence',
    490: 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ' +
        'ring armour',
    491: 'chain saw, chainsaw',
    492: 'chest',
    493: 'chiffonier, commode',
    494: 'chime, bell, gong',
    495: 'china cabinet, china closet',
    496: 'Christmas stocking',
    497: 'church, church building',
    498: 'cinema, movie theater, movie theatre, movie house, picture palace',
    499: 'cleaver, meat cleaver, chopper',
    500: 'cliff dwelling',
    501: 'cloak',
    502: 'clog, geta, patten, sabot',
    503: 'cocktail shaker',
    504: 'coffee mug',
    505: 'coffeepot',
    506: 'coil, spiral, volute, whorl, helix',
    507: 'combination lock',
    508: 'computer keyboard, keypad',
    509: 'confectionery, confectionary, candy store',
    510: 'container ship, containership, container vessel',
    511: 'convertible',
    512: 'corkscrew, bottle screw',
    513: 'cornet, horn, trumpet, trump',
    514: 'cowboy boot',
    515: 'cowboy hat, ten-gallon hat',
    516: 'cradle',
    517: 'crane',
    518: 'crash helmet',
    519: 'crate',
    520: 'crib, cot',
    521: 'Crock Pot',
    522: 'croquet ball',
    523: 'crutch',
    524: 'cuirass',
    525: 'dam, dike, dyke',
    526: 'desk',
    527: 'desktop computer',
    528: 'dial telephone, dial phone',
    529: 'diaper, nappy, napkin',
    530: 'digital clock',
    531: 'digital watch',
    532: 'dining table, board',
    533: 'dishrag, dishcloth',
    534: 'dishwasher, dish washer, dishwashing machine',
    535: 'disk brake, disc brake',
    536: 'dock, dockage, docking facility',
    537: 'dogsled, dog sled, dog sleigh',
    538: 'dome',
    539: 'doormat, welcome mat',
    540: 'drilling platform, offshore rig',
    541: 'drum, membranophone, tympan',
    542: 'drumstick',
    543: 'dumbbell',
    544: 'Dutch oven',
    545: 'electric fan, blower',
    546: 'electric guitar',
    547: 'electric locomotive',
    548: 'entertainment center',
    549: 'envelope',
    550: 'espresso maker',
    551: 'face powder',
    552: 'feather boa, boa',
    553: 'file, file cabinet, filing cabinet',
    554: 'fireboat',
    555: 'fire engine, fire truck',
    556: 'fire screen, fireguard',
    557: 'flagpole, flagstaff',
    558: 'flute, transverse flute',
    559: 'folding chair',
    560: 'football helmet',
    561: 'forklift',
    562: 'fountain',
    563: 'fountain pen',
    564: 'four-poster',
    565: 'freight car',
    566: 'French horn, horn',
    567: 'frying pan, frypan, skillet',
    568: 'fur coat',
    569: 'garbage truck, dustcart',
    570: 'gasmask, respirator, gas helmet',
    571: 'gas pump, gasoline pump, petrol pump, island dispenser',
    572: 'goblet',
    573: 'go-kart',
    574: 'golf ball',
    575: 'golfcart, golf cart',
    576: 'gondola',
    577: 'gong, tam-tam',
    578: 'gown',
    579: 'grand piano, grand',
    580: 'greenhouse, nursery, glasshouse',
    581: 'grille, radiator grille',
    582: 'grocery store, grocery, food market, market',
    583: 'guillotine',
    584: 'hair slide',
    585: 'hair spray',
    586: 'half track',
    587: 'hammer',
    588: 'hamper',
    589: 'hand blower, blow dryer, blow drier, hair dryer, hair drier',
    590: 'hand-held computer, hand-held microcomputer',
    591: 'handkerchief, hankie, hanky, hankey',
    592: 'hard disc, hard disk, fixed disk',
    593: 'harmonica, mouth organ, harp, mouth harp',
    594: 'harp',
    595: 'harvester, reaper',
    596: 'hatchet',
    597: 'holster',
    598: 'home theater, home theatre',
    599: 'honeycomb',
    600: 'hook, claw',
    601: 'hoopskirt, crinoline',
    602: 'horizontal bar, high bar',
    603: 'horse cart, horse-cart',
    604: 'hourglass',
    605: 'iPod',
    606: 'iron, smoothing iron',
    607: 'jack-o\'-lantern',
    608: 'jean, blue jean, denim',
    609: 'jeep, landrover',
    610: 'jersey, T-shirt, tee shirt',
    611: 'jigsaw puzzle',
    612: 'jinrikisha, ricksha, rickshaw',
    613: 'joystick',
    614: 'kimono',
    615: 'knee pad',
    616: 'knot',
    617: 'lab coat, laboratory coat',
    618: 'ladle',
    619: 'lampshade, lamp shade',
    620: 'laptop, laptop computer',
    621: 'lawn mower, mower',
    622: 'lens cap, lens cover',
    623: 'letter opener, paper knife, paperknife',
    624: 'library',
    625: 'lifeboat',
    626: 'lighter, light, igniter, ignitor',
    627: 'limousine, limo',
    628: 'liner, ocean liner',
    629: 'lipstick, lip rouge',
    630: 'Loafer',
    631: 'lotion',
    632: 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker ' +
        'system',
    633: 'loupe, jeweler\'s loupe',
    634: 'lumbermill, sawmill',
    635: 'magnetic compass',
    636: 'mailbag, postbag',
    637: 'mailbox, letter box',
    638: 'maillot',
    639: 'maillot, tank suit',
    640: 'manhole cover',
    641: 'maraca',
    642: 'marimba, xylophone',
    643: 'mask',
    644: 'matchstick',
    645: 'maypole',
    646: 'maze, labyrinth',
    647: 'measuring cup',
    648: 'medicine chest, medicine cabinet',
    649: 'megalith, megalithic structure',
    650: 'microphone, mike',
    651: 'microwave, microwave oven',
    652: 'military uniform',
    653: 'milk can',
    654: 'minibus',
    655: 'miniskirt, mini',
    656: 'minivan',
    657: 'missile',
    658: 'mitten',
    659: 'mixing bowl',
    660: 'mobile home, manufactured home',
    661: 'Model T',
    662: 'modem',
    663: 'monastery',
    664: 'monitor',
    665: 'moped',
    666: 'mortar',
    667: 'mortarboard',
    668: 'mosque',
    669: 'mosquito net',
    670: 'motor scooter, scooter',
    671: 'mountain bike, all-terrain bike, off-roader',
    672: 'mountain tent',
    673: 'mouse, computer mouse',
    674: 'mousetrap',
    675: 'moving van',
    676: 'muzzle',
    677: 'nail',
    678: 'neck brace',
    679: 'necklace',
    680: 'nipple',
    681: 'notebook, notebook computer',
    682: 'obelisk',
    683: 'oboe, hautboy, hautbois',
    684: 'ocarina, sweet potato',
    685: 'odometer, hodometer, mileometer, milometer',
    686: 'oil filter',
    687: 'organ, pipe organ',
    688: 'oscilloscope, scope, cathode-ray oscilloscope, CRO',
    689: 'overskirt',
    690: 'oxcart',
    691: 'oxygen mask',
    692: 'packet',
    693: 'paddle, boat paddle',
    694: 'paddlewheel, paddle wheel',
    695: 'padlock',
    696: 'paintbrush',
    697: 'pajama, pyjama, pj\'s, jammies',
    698: 'palace',
    699: 'panpipe, pandean pipe, syrinx',
    700: 'paper towel',
    701: 'parachute, chute',
    702: 'parallel bars, bars',
    703: 'park bench',
    704: 'parking meter',
    705: 'passenger car, coach, carriage',
    706: 'patio, terrace',
    707: 'pay-phone, pay-station',
    708: 'pedestal, plinth, footstall',
    709: 'pencil box, pencil case',
    710: 'pencil sharpener',
    711: 'perfume, essence',
    712: 'Petri dish',
    713: 'photocopier',
    714: 'pick, plectrum, plectron',
    715: 'pickelhaube',
    716: 'picket fence, paling',
    717: 'pickup, pickup truck',
    718: 'pier',
    719: 'piggy bank, penny bank',
    720: 'pill bottle',
    721: 'pillow',
    722: 'ping-pong ball',
    723: 'pinwheel',
    724: 'pirate, pirate ship',
    725: 'pitcher, ewer',
    726: 'plane, carpenter\'s plane, woodworking plane',
    727: 'planetarium',
    728: 'plastic bag',
    729: 'plate rack',
    730: 'plow, plough',
    731: 'plunger, plumber\'s helper',
    732: 'Polaroid camera, Polaroid Land camera',
    733: 'pole',
    734: 'police van, police wagon, paddy wagon, patrol wagon, wagon, black ' +
        'Maria',
    735: 'poncho',
    736: 'pool table, billiard table, snooker table',
    737: 'pop bottle, soda bottle',
    738: 'pot, flowerpot',
    739: 'potter\'s wheel',
    740: 'power drill',
    741: 'prayer rug, prayer mat',
    742: 'printer',
    743: 'prison, prison house',
    744: 'projectile, missile',
    745: 'projector',
    746: 'puck, hockey puck',
    747: 'punching bag, punch bag, punching ball, punchball',
    748: 'purse',
    749: 'quill, quill pen',
    750: 'quilt, comforter, comfort, puff',
    751: 'racer, race car, racing car',
    752: 'racket, racquet',
    753: 'radiator',
    754: 'radio, wireless',
    755: 'radio telescope, radio reflector',
    756: 'rain barrel',
    757: 'recreational vehicle, RV, R.V.',
    758: 'reel',
    759: 'reflex camera',
    760: 'refrigerator, icebox',
    761: 'remote control, remote',
    762: 'restaurant, eating house, eating place, eatery',
    763: 'revolver, six-gun, six-shooter',
    764: 'rifle',
    765: 'rocking chair, rocker',
    766: 'rotisserie',
    767: 'rubber eraser, rubber, pencil eraser',
    768: 'rugby ball',
    769: 'rule, ruler',
    770: 'running shoe',
    771: 'safe',
    772: 'safety pin',
    773: 'saltshaker, salt shaker',
    774: 'sandal',
    775: 'sarong',
    776: 'sax, saxophone',
    777: 'scabbard',
    778: 'scale, weighing machine',
    779: 'school bus',
    780: 'schooner',
    781: 'scoreboard',
    782: 'screen, CRT screen',
    783: 'screw',
    784: 'screwdriver',
    785: 'seat belt, seatbelt',
    786: 'sewing machine',
    787: 'shield, buckler',
    788: 'shoe shop, shoe-shop, shoe store',
    789: 'shoji',
    790: 'shopping basket',
    791: 'shopping cart',
    792: 'shovel',
    793: 'shower cap',
    794: 'shower curtain',
    795: 'ski',
    796: 'ski mask',
    797: 'sleeping bag',
    798: 'slide rule, slipstick',
    799: 'sliding door',
    800: 'slot, one-armed bandit',
    801: 'snorkel',
    802: 'snowmobile',
    803: 'snowplow, snowplough',
    804: 'soap dispenser',
    805: 'soccer ball',
    806: 'sock',
    807: 'solar dish, solar collector, solar furnace',
    808: 'sombrero',
    809: 'soup bowl',
    810: 'space bar',
    811: 'space heater',
    812: 'space shuttle',
    813: 'spatula',
    814: 'speedboat',
    815: 'spider web, spider\'s web',
    816: 'spindle',
    817: 'sports car, sport car',
    818: 'spotlight, spot',
    819: 'stage',
    820: 'steam locomotive',
    821: 'steel arch bridge',
    822: 'steel drum',
    823: 'stethoscope',
    824: 'stole',
    825: 'stone wall',
    826: 'stopwatch, stop watch',
    827: 'stove',
    828: 'strainer',
    829: 'streetcar, tram, tramcar, trolley, trolley car',
    830: 'stretcher',
    831: 'studio couch, day bed',
    832: 'stupa, tope',
    833: 'submarine, pigboat, sub, U-boat',
    834: 'suit, suit of clothes',
    835: 'sundial',
    836: 'sunglass',
    837: 'sunglasses, dark glasses, shades',
    838: 'sunscreen, sunblock, sun blocker',
    839: 'suspension bridge',
    840: 'swab, swob, mop',
    841: 'sweatshirt',
    842: 'swimming trunks, bathing trunks',
    843: 'swing',
    844: 'switch, electric switch, electrical switch',
    845: 'syringe',
    846: 'table lamp',
    847: 'tank, army tank, armored combat vehicle, armoured combat vehicle',
    848: 'tape player',
    849: 'teapot',
    850: 'teddy, teddy bear',
    851: 'television, television system',
    852: 'tennis ball',
    853: 'thatch, thatched roof',
    854: 'theater curtain, theatre curtain',
    855: 'thimble',
    856: 'thresher, thrasher, threshing machine',
    857: 'throne',
    858: 'tile roof',
    859: 'toaster',
    860: 'tobacco shop, tobacconist shop, tobacconist',
    861: 'toilet seat',
    862: 'torch',
    863: 'totem pole',
    864: 'tow truck, tow car, wrecker',
    865: 'toyshop',
    866: 'tractor',
    867: 'trailer truck, tractor trailer, trucking rig, rig, articulated ' +
        'lorry, semi',
    868: 'tray',
    869: 'trench coat',
    870: 'tricycle, trike, velocipede',
    871: 'trimaran',
    872: 'tripod',
    873: 'triumphal arch',
    874: 'trolleybus, trolley coach, trackless trolley',
    875: 'trombone',
    876: 'tub, vat',
    877: 'turnstile',
    878: 'typewriter keyboard',
    879: 'umbrella',
    880: 'unicycle, monocycle',
    881: 'upright, upright piano',
    882: 'vacuum, vacuum cleaner',
    883: 'vase',
    884: 'vault',
    885: 'velvet',
    886: 'vending machine',
    887: 'vestment',
    888: 'viaduct',
    889: 'violin, fiddle',
    890: 'volleyball',
    891: 'waffle iron',
    892: 'wall clock',
    893: 'wallet, billfold, notecase, pocketbook',
    894: 'wardrobe, closet, press',
    895: 'warplane, military plane',
    896: 'washbasin, handbasin, washbowl, lavabo, wash-hand basin',
    897: 'washer, automatic washer, washing machine',
    898: 'water bottle',
    899: 'water jug',
    900: 'water tower',
    901: 'whiskey jug',
    902: 'whistle',
    903: 'wig',
    904: 'window screen',
    905: 'window shade',
    906: 'Windsor tie',
    907: 'wine bottle',
    908: 'wing',
    909: 'wok',
    910: 'wooden spoon',
    911: 'wool, woolen, woollen',
    912: 'worm fence, snake fence, snake-rail fence, Virginia fence',
    913: 'wreck',
    914: 'yawl',
    915: 'yurt',
    916: 'web site, website, internet site, site',
    917: 'comic book',
    918: 'crossword puzzle, crossword',
    919: 'street sign',
    920: 'traffic light, traffic signal, stoplight',
    921: 'book jacket, dust cover, dust jacket, dust wrapper',
    922: 'menu',
    923: 'plate',
    924: 'guacamole',
    925: 'consomme',
    926: 'hot pot, hotpot',
    927: 'trifle',
    928: 'ice cream, icecream',
    929: 'ice lolly, lolly, lollipop, popsicle',
    930: 'French loaf',
    931: 'bagel, beigel',
    932: 'pretzel',
    933: 'cheeseburger',
    934: 'hotdog, hot dog, red hot',
    935: 'mashed potato',
    936: 'head cabbage',
    937: 'broccoli',
    938: 'cauliflower',
    939: 'zucchini, courgette',
    940: 'spaghetti squash',
    941: 'acorn squash',
    942: 'butternut squash',
    943: 'cucumber, cuke',
    944: 'artichoke, globe artichoke',
    945: 'bell pepper',
    946: 'cardoon',
    947: 'mushroom',
    948: 'Granny Smith',
    949: 'strawberry',
    950: 'orange',
    951: 'lemon',
    952: 'fig',
    953: 'pineapple, ananas',
    954: 'banana',
    955: 'jackfruit, jak, jack',
    956: 'custard apple',
    957: 'pomegranate',
    958: 'hay',
    959: 'carbonara',
    960: 'chocolate sauce, chocolate syrup',
    961: 'dough',
    962: 'meat loaf, meatloaf',
    963: 'pizza, pizza pie',
    964: 'potpie',
    965: 'burrito',
    966: 'red wine',
    967: 'espresso',
    968: 'cup',
    969: 'eggnog',
    970: 'alp',
    971: 'bubble',
    972: 'cliff, drop, drop-off',
    973: 'coral reef',
    974: 'geyser',
    975: 'lakeside, lakeshore',
    976: 'promontory, headland, head, foreland',
    977: 'sandbar, sand bar',
    978: 'seashore, coast, seacoast, sea-coast',
    979: 'valley, vale',
    980: 'volcano',
    981: 'ballplayer, baseball player',
    982: 'groom, bridegroom',
    983: 'scuba diver',
    984: 'rapeseed',
    985: 'daisy',
    986: 'yellow lady\'s slipper, yellow lady-slipper, Cypripedium calceolus, ' +
        'Cypripedium parviflorum',
    987: 'corn',
    988: 'acorn',
    989: 'hip, rose hip, rosehip',
    990: 'buckeye, horse chestnut, conker',
    991: 'coral fungus',
    992: 'agaric',
    993: 'gyromitra',
    994: 'stinkhorn, carrion fungus',
    995: 'earthstar',
    996: 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola ' +
        'frondosa',
    997: 'bolete',
    998: 'ear, spike, capitulum',
    999: 'toilet tissue, toilet paper, bathroom tissue'
};

},{}],6:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var webgl_util = require("../../src/math/webgl/webgl_util");
function getUnpackAndPreprocessInputShader(gpgpu, inputShapeRC) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D source;\n    varying vec2 resultUV;\n\n    const vec2 inputShapeCR = vec2(" + inputShapeRC[1] + ".0, " + inputShapeRC[0] + ".0);\n\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    void main() {\n      vec2 outputCR = floor(gl_FragCoord.xy);\n\n      vec2 sourceCR = vec2(floor(outputCR[0] / 3.0), outputCR[1]);\n      vec2 sourceUV = (sourceCR + halfCR) / inputShapeCR;\n\n      vec4 sourceValue = texture2D(source, sourceUV) * 255.0;\n\n      float channelValue = 0.0;\n      int channel = int(mod(outputCR[0], 3.0));\n\n      if (channel == 0) {\n        channelValue = sourceValue.r - 103.939;\n      } else if (channel == 1) {\n        channelValue = sourceValue.g - 116.779;\n      } else if (channel == 2) {\n        channelValue = sourceValue.b - 123.68;\n      }\n\n      gl_FragColor = vec4(channelValue, 0, 0, 0);\n    }";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.getUnpackAndPreprocessInputShader = getUnpackAndPreprocessInputShader;
function preprocessInput(gpgpu, preprocessInputShader, sourceTex, resultTex, shapeRowCol) {
    gpgpu.setOutputMatrixTexture(resultTex, shapeRowCol[0], shapeRowCol[1]);
    gpgpu.setProgram(preprocessInputShader);
    gpgpu.setInputMatrixTexture(sourceTex, 'source', 0);
    gpgpu.executeProgram();
}
exports.preprocessInput = preprocessInput;
function getRenderGrayscaleChannelsCollageShader(gpgpu) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D source;\n    uniform sampler2D minValues;\n    uniform sampler2D maxValues;\n    varying vec2 resultUV;\n\n    uniform float imageSize;\n    uniform float channels;\n    uniform float imagesPerRow;\n    uniform vec2 inputShapeCR;\n\n    const vec2 halfCR = vec2(0.5, 0.5);\n\n    void main() {\n      vec2 outputCR = floor(gl_FragCoord.xy);\n\n      float imageRow = floor(outputCR[1] / imageSize);\n      float imageCol = mod(outputCR[0], imageSize);\n\n      float currentChannel = floor(outputCR[0] / imageSize) +\n          imageRow * imagesPerRow;\n\n      // When the number of channels is not square, we render white to fill in\n      // the output texture.\n      if (currentChannel > channels) {\n        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);\n        return;\n      }\n\n      float sourceC = channels * imageCol + currentChannel;\n      float sourceR = mod(outputCR[1], imageSize);\n\n      vec2 sourceUV = (vec2(sourceC, sourceR) + halfCR) / inputShapeCR;\n\n      // Flip the vertical axis of the texture for display since we represent\n      // image textures as vertically flipped.\n      float sourceValue = texture2D(\n          source, vec2(sourceUV.s, 1.0 - sourceUV.t)).r;\n\n      // Normalize the value by sampling the minValues and maxValues texture\n      // which contain min and max per channel.\n      vec2 minMaxValuesShapeCR = vec2(channels, 1);\n      vec2 minMaxValuesCR = vec2(currentChannel, 0);\n      vec2 minMaxValuesUV = (minMaxValuesCR + halfCR) / minMaxValuesShapeCR;\n\n      float minValue = texture2D(minValues, minMaxValuesUV).r;\n      float maxValue = texture2D(maxValues, minMaxValuesUV).r;\n\n      float normalizedValue = (sourceValue - minValue) / (maxValue - minValue);\n\n      gl_FragColor = vec4(\n          normalizedValue, normalizedValue, normalizedValue, 1);\n    }\n  ";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.getRenderGrayscaleChannelsCollageShader = getRenderGrayscaleChannelsCollageShader;
function renderGrayscaleChannelsCollage(gpgpu, unpackChannelsShader, sourceTex, minValuesTex, maxValuesTex, inputShapeRC, imageSize, channels, textureSize, numRows) {
    webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    gpgpu.setProgram(unpackChannelsShader);
    gpgpu.setInputMatrixTexture(sourceTex, 'source', 0);
    gpgpu.setInputMatrixTexture(minValuesTex, 'minValues', 1);
    gpgpu.setInputMatrixTexture(maxValuesTex, 'maxValues', 2);
    var imageSizeLoc = gpgpu.getUniformLocation('imageSize');
    gpgpu.gl.uniform1f(imageSizeLoc, imageSize);
    var channelsLoc = gpgpu.getUniformLocation('channels');
    gpgpu.gl.uniform1f(channelsLoc, channels);
    var imagesPerRowLoc = gpgpu.getUniformLocation('imagesPerRow');
    gpgpu.gl.uniform1f(imagesPerRowLoc, Math.floor(textureSize / imageSize));
    var inputShapeCRLoc = gpgpu.getUniformLocation('inputShapeCR');
    gpgpu.gl.uniform2f(inputShapeCRLoc, inputShapeRC[1], inputShapeRC[0]);
    gpgpu.executeProgram();
}
exports.renderGrayscaleChannelsCollage = renderGrayscaleChannelsCollage;

},{"../../src/math/webgl/webgl_util":51}],7:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var checkpoint_loader_1 = require("../../src/checkpoint_loader");
var math_cpu_1 = require("../../src/math/math_cpu");
var ndarray_1 = require("../../src/math/ndarray");
var imagenet_classes = require("./imagenet_classes");
var imagenet_util = require("./imagenet_util");
var IMAGE_SIZE = 227;
var GOOGLE_CLOUD_STORAGE_DIR = 'https://storage.googleapis.com/learnjs-data/checkpoint_zoo/';
var SqueezeNet = (function () {
    function SqueezeNet(gpgpu, math) {
        this.gpgpu = gpgpu;
        this.math = math;
        this.preprocessInputShader =
            imagenet_util.getUnpackAndPreprocessInputShader(gpgpu, [IMAGE_SIZE, IMAGE_SIZE]);
    }
    SqueezeNet.prototype.loadVariables = function () {
        var _this = this;
        return new Promise(function (resolve, reject) {
            var checkpointLoader = new checkpoint_loader_1.CheckpointLoader(GOOGLE_CLOUD_STORAGE_DIR + 'squeezenet1_1/');
            checkpointLoader.getAllVariables().then(function (variables) {
                _this.variables = variables;
                resolve();
            });
        });
    };
    SqueezeNet.prototype.preprocessColorTextureToArray3D = function (rgbTexture, imageDimensions) {
        var preprocessResultShapeRC = [imageDimensions[0], imageDimensions[0] * 3];
        var preprocessResultTexture = this.math.getTextureManager().acquireTexture(preprocessResultShapeRC);
        imagenet_util.preprocessInput(this.gpgpu, this.preprocessInputShader, rgbTexture, preprocessResultTexture, preprocessResultShapeRC);
        return ndarray_1.NDArray.make([imageDimensions[0], imageDimensions[0], 3], {
            texture: preprocessResultTexture,
            textureShapeRC: preprocessResultShapeRC
        });
    };
    SqueezeNet.prototype.infer = function (preprocessedInput) {
        var _this = this;
        var namedActivations = {};
        var avgpool10 = this.math.scope(function (keep) {
            var conv1 = _this.math.conv2d(preprocessedInput, _this.variables['conv1_W:0'], _this.variables['conv1_b:0'], 2, 0);
            var conv1relu = keep(_this.math.relu(conv1));
            namedActivations['conv_1'] = conv1relu;
            var pool1 = keep(_this.math.maxPool(conv1relu, 3, 2, 0));
            namedActivations['maxpool_1'] = pool1;
            var fire2 = keep(_this.fireModule(pool1, 2));
            namedActivations['fire2'] = fire2;
            var fire3 = keep(_this.fireModule(fire2, 3));
            namedActivations['fire3'] = fire3;
            var fire3Reshape2d = fire3.as2D(fire3.shape[0], fire3.shape[1] * fire3.shape[2]);
            var fire3Sliced2d = _this.math.slice2D(fire3Reshape2d, [0, 0], [fire3.shape[0] - 1, (fire3.shape[1] - 1) * fire3.shape[2]]);
            var fire3Sliced = fire3Sliced2d.as3D(fire3.shape[0] - 1, fire3.shape[1] - 1, fire3.shape[2]);
            var pool2 = keep(_this.math.maxPool(fire3Sliced, 3, 2, 0));
            namedActivations['maxpool_2'] = pool2;
            var fire4 = keep(_this.fireModule(pool2, 4));
            namedActivations['fire4'] = fire4;
            var fire5 = keep(_this.fireModule(fire4, 5));
            namedActivations['fire5'] = fire5;
            var pool3 = keep(_this.math.maxPool(fire5, 3, 2, 0));
            namedActivations['maxpool_3'] = pool3;
            var fire6 = keep(_this.fireModule(pool3, 6));
            namedActivations['fire6'] = fire6;
            var fire7 = keep(_this.fireModule(fire6, 7));
            namedActivations['fire7'] = fire7;
            var fire8 = keep(_this.fireModule(fire7, 8));
            namedActivations['fire8'] = fire8;
            var fire9 = keep(_this.fireModule(fire8, 9));
            namedActivations['fire9'] = fire9;
            var conv10 = keep(_this.math.conv2d(fire9, _this.variables['conv10_W:0'], _this.variables['conv10_b:0'], 1, 0));
            namedActivations['conv10'] = conv10;
            return _this.math.avgPool(conv10, conv10.shape[0], 1, 0).as1D();
        });
        return { namedActivations: namedActivations, logits: avgpool10 };
    };
    SqueezeNet.prototype.fireModule = function (input, fireId) {
        var y1 = this.math.conv2d(input, this.variables['fire' + fireId + '/squeeze1x1_W:0'], this.variables['fire' + fireId + '/squeeze1x1_b:0'], 1, 0);
        var y2 = this.math.relu(y1);
        var left1 = this.math.conv2d(y2, this.variables['fire' + fireId + '/expand1x1_W:0'], this.variables['fire' + fireId + '/expand1x1_b:0'], 1, 0);
        var left2 = this.math.relu(left1);
        var right1 = this.math.conv2d(y2, this.variables['fire' + fireId + '/expand3x3_W:0'], this.variables['fire' + fireId + '/expand3x3_b:0'], 1, 1);
        var right2 = this.math.relu(right1);
        return this.math.concat3D(left2, right2, 2);
    };
    SqueezeNet.prototype.getTopKClasses = function (logits, topK) {
        var predictions = this.math.softmax(logits);
        var topk = new math_cpu_1.NDArrayMathCPU().topK(predictions, topK);
        var topkIndices = topk.indices.getValues();
        var topkValues = topk.values.getValues();
        var topClassesToProbability = {};
        for (var i = 0; i < topkIndices.length; i++) {
            topClassesToProbability[imagenet_classes
                .IMAGENET_CLASSES[topkIndices[i]]] =
                topkValues[i];
        }
        return topClassesToProbability;
    };
    return SqueezeNet;
}());
exports.SqueezeNet = SqueezeNet;

},{"../../src/checkpoint_loader":9,"../../src/math/math_cpu":24,"../../src/math/ndarray":26,"./imagenet_classes":5,"./imagenet_util":6}],8:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function PolymerElement(spec) {
    return Polymer.Class(spec);
}
exports.PolymerElement = PolymerElement;

},{}],9:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var MANIFEST_FILE = 'manifest.json';
var CheckpointLoader = (function () {
    function CheckpointLoader(urlPath) {
        this.urlPath = urlPath;
        if (this.urlPath.charAt(this.urlPath.length - 1) !== '/') {
            this.urlPath += '/';
        }
    }
    CheckpointLoader.prototype.loadManifest = function () {
        var _this = this;
        return new Promise(function (resolve, reject) {
            var xhr = new XMLHttpRequest();
            xhr.open('GET', _this.urlPath + MANIFEST_FILE);
            xhr.onload = function () {
                _this.checkpointManifest = JSON.parse(xhr.responseText);
                resolve();
            };
            xhr.onerror = function (error) {
                throw new Error(MANIFEST_FILE + " not found at " + _this.urlPath + ". " + error);
            };
            xhr.send();
        });
    };
    CheckpointLoader.prototype.getCheckpointManifest = function () {
        var _this = this;
        if (this.checkpointManifest == null) {
            return new Promise(function (resolve, reject) {
                _this.loadManifest().then(function () {
                    resolve(_this.checkpointManifest);
                });
            });
        }
        return new Promise(function (resolve, reject) {
            resolve(_this.checkpointManifest);
        });
    };
    CheckpointLoader.prototype.getAllVariables = function () {
        var _this = this;
        if (this.variables != null) {
            return new Promise(function (resolve, reject) {
                resolve(_this.variables);
            });
        }
        return new Promise(function (resolve, reject) {
            _this.getCheckpointManifest().then(function (checkpointDefinition) {
                var variableNames = Object.keys(_this.checkpointManifest);
                var variablePromises = [];
                for (var i = 0; i < variableNames.length; i++) {
                    variablePromises.push(_this.getVariable(variableNames[i]));
                }
                Promise.all(variablePromises).then(function (variables) {
                    _this.variables = {};
                    for (var i = 0; i < variables.length; i++) {
                        _this.variables[variableNames[i]] = variables[i];
                    }
                    resolve(_this.variables);
                });
            });
        });
    };
    CheckpointLoader.prototype.getVariable = function (varName) {
        var _this = this;
        if (!(varName in this.checkpointManifest)) {
            throw new Error('Cannot load non-existant variable ' + varName);
        }
        var variableRequestPromiseMethod = function (resolve, reject) {
            var xhr = new XMLHttpRequest();
            xhr.responseType = 'arraybuffer';
            var fname = _this.checkpointManifest[varName].filename;
            xhr.open('GET', _this.urlPath + fname);
            xhr.onload = function () {
                var values = new Float32Array(xhr.response);
                var ndarray = ndarray_1.NDArray.make(_this.checkpointManifest[varName].shape, { values: values });
                resolve(ndarray);
            };
            xhr.onerror = function (error) {
                throw new Error('Could not fetch variable ' + varName + ': ' + error);
            };
            xhr.send();
        };
        if (this.checkpointManifest == null) {
            return new Promise(function (resolve, reject) {
                _this.loadManifest().then(function () {
                    new Promise(variableRequestPromiseMethod).then(resolve);
                });
            });
        }
        return new Promise(variableRequestPromiseMethod);
    };
    return CheckpointLoader;
}());
exports.CheckpointLoader = CheckpointLoader;

},{"./math/ndarray":26}],10:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var util = require("./util");
var STATS_SAMPLE_PERCENTAGE = 0.1;
var InMemoryDataset = (function () {
    function InMemoryDataset(dataShapes) {
        this.dataShapes = dataShapes;
        this.normalizationInfo = {};
    }
    InMemoryDataset.prototype.getDataShape = function (dataIndex) {
        return this.dataShapes[dataIndex];
    };
    InMemoryDataset.prototype.getData = function () {
        return this.dataset;
    };
    InMemoryDataset.prototype.getStats = function () {
        var _this = this;
        if (this.dataset == null) {
            throw new Error('Data is null.');
        }
        return this.dataset.map(function (d) { return _this.getStatsForData(d); });
    };
    InMemoryDataset.prototype.getStatsForData = function (data) {
        var inputMin = Number.POSITIVE_INFINITY;
        var inputMax = Number.NEGATIVE_INFINITY;
        var exampleIndices = data.map(function (example, i) { return i; });
        util.shuffle(exampleIndices);
        exampleIndices =
            exampleIndices.slice(exampleIndices.length * STATS_SAMPLE_PERCENTAGE);
        for (var i = 0; i < exampleIndices.length; i++) {
            var inputValues = data[exampleIndices[i]].getValues();
            for (var j = 0; j < inputValues.length; j++) {
                inputMin = Math.min(inputMin, inputValues[j]);
                inputMax = Math.max(inputMax, inputValues[j]);
            }
        }
        return {
            inputMin: inputMin,
            inputMax: inputMax,
            exampleCount: data.length,
            shape: data[0].shape,
        };
    };
    InMemoryDataset.prototype.normalizeExamplesToRange = function (examples, curLowerBounds, curUpperBounds, newLowerBounds, newUpperBounds) {
        var curBoundsIsPerDimension = (curUpperBounds instanceof Float32Array &&
            curLowerBounds instanceof Float32Array);
        var newBoundsIsPerDimension = (newLowerBounds instanceof Float32Array &&
            newUpperBounds instanceof Float32Array);
        var inputSize = util.sizeFromShape(examples[0].shape);
        var newExamples = [];
        examples.forEach(function (example) {
            var inputValues = example.getValues();
            var normalizedValues = new Float32Array(inputSize);
            for (var j = 0; j < inputSize; j++) {
                var curLowerBound = curBoundsIsPerDimension ?
                    curLowerBounds[j] :
                    curLowerBounds;
                var curUpperBound = curBoundsIsPerDimension ?
                    curUpperBounds[j] :
                    curUpperBounds;
                var curRange = curUpperBound - curLowerBound;
                var newLowerBound = newBoundsIsPerDimension ?
                    newLowerBounds[j] :
                    newLowerBounds;
                var newUpperBound = newBoundsIsPerDimension ?
                    newUpperBounds[j] :
                    newUpperBounds;
                var newRange = newUpperBound - newLowerBound;
                if (curRange === 0) {
                    normalizedValues[j] = newLowerBound;
                }
                else {
                    normalizedValues[j] = newLowerBound +
                        newRange * (inputValues[j] - curLowerBound) / curRange;
                }
            }
            newExamples.push(ndarray_1.NDArray.make(example.shape, { values: normalizedValues }));
        });
        return newExamples;
    };
    InMemoryDataset.prototype.computeBounds = function (dataIndex) {
        var _this = this;
        if (this.dataset == null) {
            throw new Error('Data is null.');
        }
        var size = util.sizeFromShape(this.dataset[dataIndex][0].shape);
        this.normalizationInfo[dataIndex] = {
            isNormalized: false,
            minValues: new Float32Array(size),
            maxValues: new Float32Array(size)
        };
        for (var i = 0; i < size; i++) {
            this.normalizationInfo[dataIndex].minValues[i] = Number.POSITIVE_INFINITY;
            this.normalizationInfo[dataIndex].maxValues[i] = Number.NEGATIVE_INFINITY;
        }
        this.dataset[dataIndex].forEach(function (example) {
            var inputValues = example.getValues();
            for (var k = 0; k < size; k++) {
                _this.normalizationInfo[dataIndex].minValues[k] = Math.min(_this.normalizationInfo[dataIndex].minValues[k], inputValues[k]);
                _this.normalizationInfo[dataIndex].maxValues[k] = Math.max(_this.normalizationInfo[dataIndex].maxValues[k], inputValues[k]);
            }
        });
    };
    InMemoryDataset.prototype.normalizeWithinBounds = function (dataIndex, lowerBound, upperBound) {
        if (this.dataset == null) {
            throw new Error('Data is null.');
        }
        if (dataIndex >= this.dataset.length) {
            throw new Error('dataIndex out of bounds.');
        }
        if (this.normalizationInfo[dataIndex] == null) {
            this.computeBounds(dataIndex);
        }
        var curLowerBounds;
        var curUpperBounds;
        if (this.normalizationInfo[dataIndex].isNormalized) {
            curLowerBounds = this.normalizationInfo[dataIndex].lowerBound;
            curUpperBounds = this.normalizationInfo[dataIndex].upperBound;
        }
        else {
            curLowerBounds = this.normalizationInfo[dataIndex].minValues;
            curUpperBounds = this.normalizationInfo[dataIndex].maxValues;
        }
        this.dataset[dataIndex] = this.normalizeExamplesToRange(this.dataset[dataIndex], curLowerBounds, curUpperBounds, lowerBound, upperBound);
        this.normalizationInfo[dataIndex].isNormalized = true;
        this.normalizationInfo[dataIndex].lowerBound = lowerBound;
        this.normalizationInfo[dataIndex].upperBound = upperBound;
    };
    InMemoryDataset.prototype.isNormalized = function (dataIndex) {
        return this.normalizationInfo != null &&
            this.normalizationInfo[dataIndex].isNormalized;
    };
    InMemoryDataset.prototype.removeNormalization = function (dataIndex) {
        if (this.dataset == null) {
            throw new Error('Training or test data is null.');
        }
        if (!this.isNormalized(dataIndex)) {
            return;
        }
        this.dataset[dataIndex] = this.normalizeExamplesToRange(this.dataset[dataIndex], this.normalizationInfo[dataIndex].lowerBound, this.normalizationInfo[dataIndex].upperBound, this.normalizationInfo[dataIndex].minValues, this.normalizationInfo[dataIndex].maxValues);
        this.normalizationInfo[dataIndex].isNormalized = false;
    };
    InMemoryDataset.prototype.unnormalizeExamples = function (examples, dataIndex) {
        if (!this.isNormalized(dataIndex)) {
            return examples;
        }
        return this.normalizeExamplesToRange(examples, this.normalizationInfo[dataIndex].lowerBound, this.normalizationInfo[dataIndex].upperBound, this.normalizationInfo[dataIndex].minValues, this.normalizationInfo[dataIndex].maxValues);
    };
    InMemoryDataset.prototype.dispose = function () {
        if (this.dataset == null) {
            return;
        }
        for (var i = 0; i < this.dataset.length; i++) {
            for (var j = 0; j < this.dataset[i].length; j++) {
                this.dataset[i][j].dispose();
            }
        }
        this.dataset = [];
    };
    return InMemoryDataset;
}());
exports.InMemoryDataset = InMemoryDataset;

},{"./math/ndarray":26,"./util":79}],11:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_layers_1 = require("./graph_layers");
var concat3d_util = require("./math/concat3d_util");
var conv_util = require("./math/conv_util");
var ndarray_1 = require("./math/ndarray");
var util = require("./util");
var Graph = (function () {
    function Graph() {
        this.nodes = [];
        this.layers = new graph_layers_1.GraphLayers(this);
    }
    Graph.prototype.variable = function (name, data) {
        return this.addNodeAndReturnOutput(new VariableNode(this, name, data));
    };
    Graph.prototype.placeholder = function (name, shape) {
        return this.addNodeAndReturnOutput(new PlaceholderNode(this, name, shape));
    };
    Graph.prototype.constant = function (value) {
        var finalValue;
        if (typeof value === 'number') {
            finalValue = ndarray_1.Scalar.new(value);
        }
        else if (value instanceof ndarray_1.NDArray) {
            finalValue = value;
        }
        else if (value instanceof Array) {
            var vals = new Float32Array(util.flatten(value));
            finalValue = ndarray_1.NDArray.make(util.inferShape(value), { values: vals });
        }
        else {
            throw new Error('unimplemented constant type.');
        }
        return this.addNodeAndReturnOutput(new ConstantNode(this, finalValue));
    };
    Graph.prototype.reshape = function (x, shape) {
        return this.addNodeAndReturnOutput(new ReshapeNode(this, 'Reshape', x, shape));
    };
    Graph.prototype.fusedLinearCombination = function (x1, x2, c1, c2) {
        return this.addNodeAndReturnOutput(new FusedLinearCombinationNode(this, x1, x2, c1, c2));
    };
    Graph.prototype.add = function (x1, x2) {
        return this.addNodeAndReturnOutput(new AddNode(this, x1, x2));
    };
    Graph.prototype.subtract = function (x1, x2) {
        return this.addNodeAndReturnOutput(new SubtractNode(this, x1, x2));
    };
    Graph.prototype.multiply = function (x1, x2) {
        return this.addNodeAndReturnOutput(new MultiplyNode(this, x1, x2));
    };
    Graph.prototype.divide = function (x1, x2) {
        return this.addNodeAndReturnOutput(new DivideNode(this, x1, x2));
    };
    Graph.prototype.reduceSum = function (x) {
        return this.addNodeAndReturnOutput(new ReduceSumNode(this, x));
    };
    Graph.prototype.concat3d = function (x1, x2, axis) {
        return this.addNodeAndReturnOutput(new Concat3DNode(this, x1, x2, axis));
    };
    Graph.prototype.matmul = function (x1, x2) {
        return this.addNodeAndReturnOutput(new MatMulNode(this, x1, x2));
    };
    Graph.prototype.conv2d = function (x, w, b, fieldSize, outputDepth, stride, zeroPad) {
        if (stride === void 0) { stride = 1; }
        return this.addNodeAndReturnOutput(new Convolution2DNode(this, x, w, b, fieldSize, outputDepth, stride, zeroPad));
    };
    Graph.prototype.maxPool = function (x, fieldSize, stride, zeroPad) {
        if (stride === void 0) { stride = 1; }
        return this.addNodeAndReturnOutput(new MaxPoolNode(this, x, fieldSize, stride, zeroPad));
    };
    Graph.prototype.exp = function (x) {
        return this.addNodeAndReturnOutput(new ExpNode(this, x));
    };
    Graph.prototype.log = function (x) {
        return this.addNodeAndReturnOutput(new LogNode(this, x));
    };
    Graph.prototype.relu = function (x) {
        return this.addNodeAndReturnOutput(new ReLUNode(this, x));
    };
    Graph.prototype.tanh = function (x) {
        return this.addNodeAndReturnOutput(new TanHNode(this, x));
    };
    Graph.prototype.sigmoid = function (x) {
        return this.addNodeAndReturnOutput(new SigmoidNode(this, x));
    };
    Graph.prototype.square = function (x) {
        return this.addNodeAndReturnOutput(new SquareNode(this, x));
    };
    Graph.prototype.softmax = function (x) {
        return this.addNodeAndReturnOutput(new SoftmaxNode(this, x));
    };
    Graph.prototype.softmaxCrossEntropyCost = function (x, target) {
        return this.addNodeAndReturnOutput(new SoftmaxCrossEntropyCostNode(this, x, target));
    };
    Graph.prototype.meanSquaredCost = function (label, prediction) {
        return this.addNodeAndReturnOutput(new MeanSquaredCostNode(this, label, prediction));
    };
    Graph.prototype.argmax = function (x) {
        return this.addNodeAndReturnOutput(new ArgMaxNode(this, x));
    };
    Graph.prototype.argmaxEquals = function (x1, x2) {
        return this.addNodeAndReturnOutput(new ArgMaxEqualsNode(this, x1, x2));
    };
    Graph.prototype.addNodeAndReturnOutput = function (node) {
        this.nodes.push(node);
        node.validate();
        return node.output;
    };
    Graph.prototype.getNodes = function () {
        return this.nodes;
    };
    return Graph;
}());
exports.Graph = Graph;
var Tensor = (function () {
    function Tensor(shape) {
        this.shape = shape;
        this.id = Tensor.nextID++;
    }
    Tensor.nextID = 0;
    return Tensor;
}());
exports.Tensor = Tensor;
var Node = (function () {
    function Node(graph, name, inputs, output) {
        this.graph = graph;
        this.name = name;
        this.inputs = inputs;
        this.output = output;
        this.id = Node.nextID++;
        output.node = this;
    }
    Node.nextID = 0;
    return Node;
}());
exports.Node = Node;
var VariableNode = (function (_super) {
    __extends(VariableNode, _super);
    function VariableNode(graph, name, data) {
        var _this = _super.call(this, graph, name, {}, new Tensor(data.shape)) || this;
        _this.data = data;
        return _this;
    }
    VariableNode.prototype.validate = function () {
        util.assert(this.data != null, 'Error adding variable op: Data for variable \'' + this.name +
            '\' is null or undefined');
    };
    return VariableNode;
}(Node));
exports.VariableNode = VariableNode;
var PlaceholderNode = (function (_super) {
    __extends(PlaceholderNode, _super);
    function PlaceholderNode(graph, name, shape) {
        return _super.call(this, graph, name, {}, new Tensor(shape)) || this;
    }
    PlaceholderNode.prototype.validate = function () { };
    return PlaceholderNode;
}(Node));
exports.PlaceholderNode = PlaceholderNode;
var ConstantNode = (function (_super) {
    __extends(ConstantNode, _super);
    function ConstantNode(graph, data) {
        var _this = _super.call(this, graph, 'Constant', {}, new Tensor(data.shape)) || this;
        _this.data = data;
        return _this;
    }
    ConstantNode.prototype.validate = function () {
        util.assert(this.data != null, 'Error adding constant: data for placeholder \'' + this.name +
            '\' is null or undefined');
    };
    return ConstantNode;
}(Node));
exports.ConstantNode = ConstantNode;
var ReshapeNode = (function (_super) {
    __extends(ReshapeNode, _super);
    function ReshapeNode(graph, name, x, shape) {
        var _this = _super.call(this, graph, name, { x: x }, new Tensor(shape)) || this;
        _this.name = name;
        _this.x = x;
        _this.shape = shape;
        return _this;
    }
    ReshapeNode.prototype.validate = function () {
        var xSize = util.sizeFromShape(this.x.shape);
        var shapeSize = util.sizeFromShape(this.shape);
        util.assert(xSize === shapeSize, 'Error making reshape operation: input Tensor to reshape \'' +
            this.name + '\' of shape (' + this.x.shape +
            ') does not match size of requested shape ' + this.shape + '.');
    };
    ReshapeNode.X = 'x';
    return ReshapeNode;
}(Node));
exports.ReshapeNode = ReshapeNode;
var FusedLinearCombinationNode = (function (_super) {
    __extends(FusedLinearCombinationNode, _super);
    function FusedLinearCombinationNode(graph, t1, t2, c1, c2) {
        var _this = _super.call(this, graph, 'Linear Combination', { t1: t1, t2: t2, c1: c1, c2: c2 }, new Tensor(t1.shape)) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        _this.c1 = c1;
        _this.c2 = c2;
        return _this;
    }
    FusedLinearCombinationNode.prototype.validate = function () {
        util.assertShapesMatch(this.t1.shape, this.t2.shape);
        if (!util.isScalarShape(this.c1.shape)) {
            throw new Error('Error adding fusedLinearCombination: c1 is not a scalar, got ' +
                'shape: ' + this.c1.shape);
        }
        if (!util.isScalarShape(this.c2.shape)) {
            throw new Error('Error adding fusedLinearCombination: c2 is not a scalar, got ' +
                'shape: ' + this.c2.shape);
        }
    };
    FusedLinearCombinationNode.T1 = 't1';
    FusedLinearCombinationNode.T2 = 't2';
    FusedLinearCombinationNode.C1 = 'c1';
    FusedLinearCombinationNode.C2 = 'c2';
    return FusedLinearCombinationNode;
}(Node));
exports.FusedLinearCombinationNode = FusedLinearCombinationNode;
var AddNode = (function (_super) {
    __extends(AddNode, _super);
    function AddNode(graph, t1, t2) {
        var _this = _super.call(this, graph, 'Add', { t1: t1, t2: t2 }, new Tensor(util.sizeFromShape(t1.shape) === 1 ? t2.shape : t1.shape)) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        return _this;
    }
    AddNode.prototype.validate = function () {
        util.assert(util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape), 'Error adding add operation op: one of inputs must be scalar or the ' +
            'shapes ' + this.t1.shape + ' and ' + this.t2.shape +
            ' must match.');
    };
    AddNode.T1 = 't1';
    AddNode.T2 = 't2';
    return AddNode;
}(Node));
exports.AddNode = AddNode;
var SubtractNode = (function (_super) {
    __extends(SubtractNode, _super);
    function SubtractNode(graph, t1, t2) {
        var _this = _super.call(this, graph, 'Subtract', { t1: t1, t2: t2 }, new Tensor(util.sizeFromShape(t1.shape) === 1 ? t2.shape : t1.shape)) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        return _this;
    }
    SubtractNode.prototype.validate = function () {
        util.assert(util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape), 'Error adding subtract op: one of inputs must be scalar or the ' +
            'shapes ' + this.t1.shape + ' and ' + this.t2.shape +
            ' must match.');
    };
    SubtractNode.T1 = 't1';
    SubtractNode.T2 = 't2';
    return SubtractNode;
}(Node));
exports.SubtractNode = SubtractNode;
var MultiplyNode = (function (_super) {
    __extends(MultiplyNode, _super);
    function MultiplyNode(graph, t1, t2) {
        var _this = _super.call(this, graph, 'Multiply', { t1: t1, t2: t2 }, new Tensor(util.sizeFromShape(t1.shape) === 1 ? t2.shape : t1.shape)) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        return _this;
    }
    MultiplyNode.prototype.validate = function () {
        util.assert(util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape), 'Error adding multiply op: one of inputs must be scalar or the ' +
            'shapes ' + this.t1.shape + ' and ' + this.t2.shape +
            ' must match.');
    };
    MultiplyNode.T1 = 't1';
    MultiplyNode.T2 = 't2';
    return MultiplyNode;
}(Node));
exports.MultiplyNode = MultiplyNode;
var DivideNode = (function (_super) {
    __extends(DivideNode, _super);
    function DivideNode(graph, t1, t2) {
        var _this = _super.call(this, graph, 'Divide', { t1: t1, t2: t2 }, new Tensor(util.sizeFromShape(t1.shape) === 1 ? t2.shape : t1.shape)) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        return _this;
    }
    DivideNode.prototype.validate = function () {
        util.assert(util.sizeFromShape(this.t1.shape) === 1 ||
            util.sizeFromShape(this.t2.shape) === 1 ||
            util.arraysEqual(this.t1.shape, this.t2.shape), 'Error adding divide op: one of inputs must be scalar or the ' +
            'shapes ' + this.t1.shape + ' and ' + this.t2.shape +
            ' must match.');
    };
    DivideNode.T1 = 't1';
    DivideNode.T2 = 't2';
    return DivideNode;
}(Node));
exports.DivideNode = DivideNode;
var ReduceSumNode = (function (_super) {
    __extends(ReduceSumNode, _super);
    function ReduceSumNode(graph, x) {
        return _super.call(this, graph, 'ReduceSum', { x: x }, new Tensor([])) || this;
    }
    ReduceSumNode.prototype.validate = function () { };
    ReduceSumNode.X = 'x';
    return ReduceSumNode;
}(Node));
exports.ReduceSumNode = ReduceSumNode;
var Concat3DNode = (function (_super) {
    __extends(Concat3DNode, _super);
    function Concat3DNode(graph, x1, x2, axis) {
        var _this = _super.call(this, graph, 'Concat3D', { x1: x1, x2: x2 }, new Tensor(concat3d_util.computeConcat3DOutputShape(x1.shape, x2.shape, axis))) || this;
        _this.x1 = x1;
        _this.x2 = x2;
        _this.axis = axis;
        return _this;
    }
    Concat3DNode.prototype.validate = function () {
        concat3d_util.assertConcat3DShapesMatch(this.x1.shape, this.x2.shape, this.axis);
    };
    Concat3DNode.X1 = 'x1';
    Concat3DNode.X2 = 'x2';
    Concat3DNode.AXIS = 'axis';
    return Concat3DNode;
}(Node));
exports.Concat3DNode = Concat3DNode;
function getMatMulOutputShape(x1Shape, x2Shape) {
    if (x1Shape.length === 1 && x2Shape.length === 1) {
        return [1];
    }
    else if (x1Shape.length === 1 && x2Shape.length === 2) {
        return [x2Shape[1]];
    }
    else if (x1Shape.length === 2 && x2Shape.length === 1) {
        return [x1Shape[0]];
    }
    return [x1Shape[0], x2Shape[1]];
}
var MatMulNode = (function (_super) {
    __extends(MatMulNode, _super);
    function MatMulNode(graph, x1, x2) {
        var _this = _super.call(this, graph, 'MatMul', { x1: x1, x2: x2 }, new Tensor(getMatMulOutputShape(x1.shape, x2.shape))) || this;
        _this.x1 = x1;
        _this.x2 = x2;
        return _this;
    }
    MatMulNode.prototype.validate = function () {
        if (this.x1.shape.length === 2 && this.x2.shape.length === 2) {
            util.assert(this.x1.shape[1] === this.x2.shape[0], 'Error adding matmul op: inner shapes of matrices with shapes ' +
                this.x1.shape + ' and ' + this.x2.shape + ' must match.');
        }
        else if (this.x1.shape.length === 2 && this.x2.shape.length === 1) {
            util.assert(this.x1.shape[1] === this.x2.shape[0], 'Error adding matmul op: second dimension of matrix with shape ' +
                this.x1.shape + ' must match size of vector with shape ' +
                this.x2.shape + '.');
        }
        else if (this.x1.shape.length === 1 && this.x2.shape.length === 2) {
            util.assert(this.x1.shape[0] === this.x2.shape[0], 'Error adding matmul op: size of vector with shape ' + this.x1.shape +
                ' must match first dimension of matrix with ' +
                'shape ' + this.x2.shape + '.');
        }
        else {
            throw new Error('Error adding matmul op: inputs must be vectors or matrices.');
        }
    };
    MatMulNode.X1 = 'x1';
    MatMulNode.X2 = 'x2';
    return MatMulNode;
}(Node));
exports.MatMulNode = MatMulNode;
var Convolution2DNode = (function (_super) {
    __extends(Convolution2DNode, _super);
    function Convolution2DNode(graph, x, w, b, fieldSize, outputDepth, stride, zeroPad) {
        if (stride === void 0) { stride = 1; }
        var _this = _super.call(this, graph, 'Convolution 2D', { x: x, w: w, b: b }, new Tensor(conv_util.computeOutputShape3D(x.shape, fieldSize, outputDepth, stride, zeroPad))) || this;
        _this.x = x;
        _this.w = w;
        _this.b = b;
        _this.fieldSize = fieldSize;
        _this.outputDepth = outputDepth;
        _this.stride = stride;
        _this.zeroPad = zeroPad;
        return _this;
    }
    Convolution2DNode.prototype.validate = function () {
        util.assert(this.x.shape.length === 3, 'Error adding conv2d op: input must be of rank 3, but got shape: ' +
            this.x.shape + '.');
        util.assert(this.w.shape.length === 4, 'Error adding conv2d op: weights must be of rank 4, but got shape: ' +
            this.w.shape + '.');
        util.assert(this.b.shape.length === 1, 'Error adding conv2d op: biases must be of rank 1, but got shape: ' +
            this.b.shape + '.');
        util.assert(this.x.shape[2] === this.w.shape[2], 'Error adding conv2d op: depth of input (' + this.x.shape[2] +
            ') must match input depth for weights (' + this.w.shape[2] + ').');
    };
    Convolution2DNode.X = 'x';
    Convolution2DNode.W = 'w';
    Convolution2DNode.B = 'b';
    return Convolution2DNode;
}(Node));
exports.Convolution2DNode = Convolution2DNode;
var MaxPoolNode = (function (_super) {
    __extends(MaxPoolNode, _super);
    function MaxPoolNode(graph, x, fieldSize, stride, zeroPad) {
        if (stride === void 0) { stride = 1; }
        var _this = _super.call(this, graph, 'Max pool', { x: x }, new Tensor(conv_util.computeOutputShape3D(x.shape, fieldSize, x.shape[2], stride, zeroPad))) || this;
        _this.x = x;
        _this.fieldSize = fieldSize;
        _this.stride = stride;
        _this.zeroPad = zeroPad;
        return _this;
    }
    MaxPoolNode.prototype.validate = function () {
        util.assert(this.x.shape.length === 3, 'Error adding maxPool op: input must be of rank 3, but got shape: ' +
            this.x.shape + '.');
    };
    MaxPoolNode.X = 'x';
    return MaxPoolNode;
}(Node));
exports.MaxPoolNode = MaxPoolNode;
var ReLUNode = (function (_super) {
    __extends(ReLUNode, _super);
    function ReLUNode(graph, x) {
        return _super.call(this, graph, 'ReLU', { x: x }, new Tensor(x.shape)) || this;
    }
    ReLUNode.prototype.validate = function () { };
    ReLUNode.X = 'x';
    return ReLUNode;
}(Node));
exports.ReLUNode = ReLUNode;
var ExpNode = (function (_super) {
    __extends(ExpNode, _super);
    function ExpNode(graph, x) {
        return _super.call(this, graph, 'Exp', { x: x }, new Tensor(x.shape)) || this;
    }
    ExpNode.prototype.validate = function () { };
    ExpNode.X = 'x';
    return ExpNode;
}(Node));
exports.ExpNode = ExpNode;
var LogNode = (function (_super) {
    __extends(LogNode, _super);
    function LogNode(graph, x) {
        return _super.call(this, graph, 'Log', { x: x }, new Tensor(x.shape)) || this;
    }
    LogNode.prototype.validate = function () { };
    LogNode.X = 'x';
    return LogNode;
}(Node));
exports.LogNode = LogNode;
var TanHNode = (function (_super) {
    __extends(TanHNode, _super);
    function TanHNode(graph, x) {
        return _super.call(this, graph, 'TanH', { x: x }, new Tensor(x.shape)) || this;
    }
    TanHNode.prototype.validate = function () { };
    TanHNode.X = 'x';
    return TanHNode;
}(Node));
exports.TanHNode = TanHNode;
var SigmoidNode = (function (_super) {
    __extends(SigmoidNode, _super);
    function SigmoidNode(graph, x) {
        return _super.call(this, graph, 'Sigmoid', { x: x }, new Tensor(x.shape)) || this;
    }
    SigmoidNode.prototype.validate = function () { };
    SigmoidNode.X = 'x';
    return SigmoidNode;
}(Node));
exports.SigmoidNode = SigmoidNode;
var SquareNode = (function (_super) {
    __extends(SquareNode, _super);
    function SquareNode(graph, x) {
        return _super.call(this, graph, 'Square', { x: x }, new Tensor(x.shape)) || this;
    }
    SquareNode.prototype.validate = function () { };
    SquareNode.X = 'x';
    return SquareNode;
}(Node));
exports.SquareNode = SquareNode;
var SoftmaxCrossEntropyCostNode = (function (_super) {
    __extends(SoftmaxCrossEntropyCostNode, _super);
    function SoftmaxCrossEntropyCostNode(graph, x, target) {
        var _this = _super.call(this, graph, 'SoftmaxCrossEntropyCost', { x: x, target: target }, new Tensor([])) || this;
        _this.x = x;
        _this.target = target;
        return _this;
    }
    SoftmaxCrossEntropyCostNode.prototype.validate = function () {
        util.assert(util.arraysEqual(this.x.shape, this.target.shape), 'Error adding softmaxCrossEntropyCost op: x shape (' + this.x.shape +
            ') must match target shape (' + this.target.shape + ').');
    };
    SoftmaxCrossEntropyCostNode.X = 'x';
    SoftmaxCrossEntropyCostNode.TARGET = 'target';
    return SoftmaxCrossEntropyCostNode;
}(Node));
exports.SoftmaxCrossEntropyCostNode = SoftmaxCrossEntropyCostNode;
var SoftmaxNode = (function (_super) {
    __extends(SoftmaxNode, _super);
    function SoftmaxNode(graph, x) {
        var _this = _super.call(this, graph, 'Softmax', { x: x }, new Tensor(x.shape)) || this;
        _this.x = x;
        return _this;
    }
    SoftmaxNode.prototype.validate = function () {
        util.assert(this.x.shape.length === 1, 'The input to a softmax must be a 1-D tensor');
        util.assert(this.x.shape[0] >= 2, 'The input to a softmax must have at least 2 values');
    };
    SoftmaxNode.X = 'x';
    return SoftmaxNode;
}(Node));
exports.SoftmaxNode = SoftmaxNode;
var MeanSquaredCostNode = (function (_super) {
    __extends(MeanSquaredCostNode, _super);
    function MeanSquaredCostNode(graph, label, prediction) {
        var _this = _super.call(this, graph, 'Mean Squared Cost', { label: label, prediction: prediction }, new Tensor([])) || this;
        _this.label = label;
        _this.prediction = prediction;
        return _this;
    }
    MeanSquaredCostNode.prototype.validate = function () {
        util.assert(util.arraysEqual(this.label.shape, this.prediction.shape), 'Error adding meanSquaredCost op: label shape (' + this.label.shape +
            ') must match prediction shape (' + this.prediction.shape + ').');
    };
    MeanSquaredCostNode.LABEL = 'label';
    MeanSquaredCostNode.PREDICTION = 'prediction';
    return MeanSquaredCostNode;
}(Node));
exports.MeanSquaredCostNode = MeanSquaredCostNode;
var ArgMaxNode = (function (_super) {
    __extends(ArgMaxNode, _super);
    function ArgMaxNode(graph, x) {
        var _this = _super.call(this, graph, 'ArgMax', { x: x }, new Tensor([1])) || this;
        _this.x = x;
        return _this;
    }
    ArgMaxNode.prototype.validate = function () {
        util.assert(util.sizeFromShape(this.x.shape) > 0, 'Error adding argmax op: input tensor must have at least one entry.');
    };
    ArgMaxNode.X = 'x';
    return ArgMaxNode;
}(Node));
exports.ArgMaxNode = ArgMaxNode;
var ArgMaxEqualsNode = (function (_super) {
    __extends(ArgMaxEqualsNode, _super);
    function ArgMaxEqualsNode(graph, x1, x2) {
        var _this = _super.call(this, graph, 'ArgMaxEquals', { x1: x1, x2: x2 }, new Tensor([1])) || this;
        _this.x1 = x1;
        _this.x2 = x2;
        return _this;
    }
    ArgMaxEqualsNode.prototype.validate = function () {
        util.assert(util.arraysEqual(this.x1.shape, this.x2.shape), 'Error adding ArgMaxEquals op: x1 shape (' + this.x1.shape +
            ') must match x2 shape (' + this.x2.shape + ').');
    };
    ArgMaxEqualsNode.X1 = 'x1';
    ArgMaxEqualsNode.X2 = 'x2';
    return ArgMaxEqualsNode;
}(Node));
exports.ArgMaxEqualsNode = ArgMaxEqualsNode;

},{"./graph_layers":12,"./math/concat3d_util":19,"./math/conv_util":20,"./math/ndarray":26,"./util":79}],12:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var initializers_1 = require("./initializers");
var GraphLayers = (function () {
    function GraphLayers(g) {
        this.g = g;
    }
    GraphLayers.prototype.dense = function (name, x, units, activation, useBias, kernelInitializer, biasInitializer) {
        if (activation === void 0) { activation = null; }
        if (useBias === void 0) { useBias = true; }
        if (kernelInitializer === void 0) { kernelInitializer = new initializers_1.VarianceScalingInitializer(); }
        if (biasInitializer === void 0) { biasInitializer = new initializers_1.ZerosInitializer(); }
        var weights = this.g.variable(name + '-weights', kernelInitializer.initialize([x.shape[0], units], x.shape[0], units));
        var out = this.g.matmul(x, weights);
        if (useBias) {
            var bias = this.g.variable(name + '-bias', biasInitializer.initialize([units], x.shape[0], units));
            out = this.g.add(out, bias);
        }
        if (activation != null) {
            out = activation(out);
        }
        return out;
    };
    return GraphLayers;
}());
exports.GraphLayers = GraphLayers;

},{"./initializers":16}],13:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var session_1 = require("./session");
var DEFAULT_EVAL_INTERVAL_MS = 1500;
var DEFAULT_COST_INTERVAL_MS = 500;
var DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS = 3000;
var MetricReduction;
(function (MetricReduction) {
    MetricReduction[MetricReduction["SUM"] = 0] = "SUM";
    MetricReduction[MetricReduction["MEAN"] = 1] = "MEAN";
})(MetricReduction = exports.MetricReduction || (exports.MetricReduction = {}));
var GraphRunner = (function () {
    function GraphRunner(math, session, eventObserver) {
        this.math = math;
        this.session = session;
        this.eventObserver = eventObserver;
        this.lastCostTimestamp = 0;
        this.lastEvalTimestamp = 0;
        this.totalIdleTimeMs = 0;
        this.resetStatistics();
        this.zeroScalar = ndarray_1.Scalar.new(0);
    }
    GraphRunner.prototype.resetStatistics = function () {
        this.totalBatchesTrained = 0;
        this.totalIdleTimeMs = 0;
        this.lastStopTimestamp = null;
    };
    GraphRunner.prototype.train = function (costTensor, trainFeedEntries, batchSize, optimizer, numBatches, metricTensor, metricFeedEntries, metricBatchSize, metricReduction, evalIntervalMs, costIntervalMs) {
        if (metricReduction === void 0) { metricReduction = MetricReduction.MEAN; }
        if (evalIntervalMs === void 0) { evalIntervalMs = DEFAULT_EVAL_INTERVAL_MS; }
        if (costIntervalMs === void 0) { costIntervalMs = DEFAULT_COST_INTERVAL_MS; }
        this.costTensor = costTensor;
        this.trainFeedEntries = trainFeedEntries;
        this.metricTensor = metricTensor;
        this.metricFeedEntries = metricFeedEntries;
        if (metricBatchSize != null && this.metricBatchSize !== metricBatchSize) {
            if (this.metricBatchSizeScalar != null) {
                this.metricBatchSizeScalar.dispose();
            }
            this.metricBatchSizeScalar = ndarray_1.Scalar.new(metricBatchSize);
        }
        this.metricBatchSize = metricBatchSize;
        this.metricReduction = metricReduction;
        this.batchSize = batchSize;
        this.optimizer = optimizer;
        this.metricIntervalMs = evalIntervalMs;
        this.costIntervalMs = costIntervalMs;
        this.currentTrainLoopNumBatches = numBatches;
        this.batchesTrainedThisRun = 0;
        this.isTraining = true;
        this.trainStartTimestamp = performance.now();
        this.trainNetwork();
    };
    GraphRunner.prototype.stopTraining = function () {
        this.isTraining = false;
        this.lastStopTimestamp = performance.now();
    };
    GraphRunner.prototype.resumeTraining = function () {
        this.isTraining = true;
        if (this.lastStopTimestamp != null) {
            this.totalIdleTimeMs += performance.now() - this.lastStopTimestamp;
        }
        this.trainNetwork();
    };
    GraphRunner.prototype.trainNetwork = function () {
        var _this = this;
        if (this.batchesTrainedThisRun === this.currentTrainLoopNumBatches) {
            this.stopTraining();
        }
        if (!this.isTraining) {
            if (this.eventObserver.doneTrainingCallback != null) {
                this.eventObserver.doneTrainingCallback();
            }
            return;
        }
        var start = performance.now();
        var shouldComputeCost = this.eventObserver.avgCostCallback != null &&
            (start - this.lastCostTimestamp > this.costIntervalMs);
        if (shouldComputeCost) {
            this.lastCostTimestamp = start;
        }
        var costReduction = shouldComputeCost ? session_1.CostReduction.MEAN : session_1.CostReduction.NONE;
        this.math.scope(function (keep) {
            var avgCost = _this.session.train(_this.costTensor, _this.trainFeedEntries, _this.batchSize, _this.optimizer, costReduction);
            if (shouldComputeCost) {
                var trainTime = performance.now() - start;
                _this.eventObserver.avgCostCallback(avgCost);
                if (_this.eventObserver.trainExamplesPerSecCallback != null) {
                    var examplesPerSec = (_this.batchSize * 1000 / trainTime);
                    _this.eventObserver.trainExamplesPerSecCallback(examplesPerSec);
                }
            }
            if (_this.eventObserver.metricCallback != null &&
                _this.metricFeedEntries != null &&
                start - _this.lastEvalTimestamp > _this.metricIntervalMs) {
                _this.lastEvalTimestamp = start;
                if (_this.lastComputedMetric != null) {
                    _this.lastComputedMetric.dispose();
                }
                _this.lastComputedMetric = _this.computeMetric();
                _this.eventObserver.metricCallback(_this.lastComputedMetric);
            }
            if (_this.eventObserver.totalTimeCallback != null) {
                _this.eventObserver.totalTimeCallback((start - _this.trainStartTimestamp) / 1000);
            }
            _this.batchesTrainedThisRun++;
            _this.totalBatchesTrained++;
            if (_this.eventObserver.batchesTrainedCallback != null) {
                _this.eventObserver.batchesTrainedCallback(_this.totalBatchesTrained);
            }
        });
        requestAnimationFrame(function () { return _this.trainNetwork(); });
    };
    GraphRunner.prototype.infer = function (inferenceTensor, inferenceFeedEntries, inferenceExampleIntervalMs, inferenceExampleCount, numPasses) {
        var _this = this;
        if (inferenceExampleIntervalMs === void 0) { inferenceExampleIntervalMs = DEFAULT_INFERENCE_EXAMPLE_INTERVAL_MS; }
        if (inferenceExampleCount === void 0) { inferenceExampleCount = 5; }
        if (this.eventObserver.inferenceExamplesCallback == null &&
            this.eventObserver.inferenceExamplesPerSecCallback == null) {
            throw new Error('Cannot start inference loop, no inference example or ' +
                'examples/sec observer provided.');
        }
        for (var i = 0; i < inferenceFeedEntries.length; i++) {
            var feedEntry = inferenceFeedEntries[i];
            if (feedEntry.data instanceof ndarray_1.NDArray) {
                throw new Error('Cannot start inference on the model runner with feed entries of ' +
                    'type NDArray. Please use InputProviders.');
            }
        }
        this.inferenceExampleIntervalMs = inferenceExampleIntervalMs;
        this.inferenceTensor = inferenceTensor;
        this.inferenceFeedEntries = inferenceFeedEntries;
        this.inferenceExampleCount = inferenceExampleCount;
        this.currentInferenceLoopNumPasses = numPasses;
        if (!this.isInferring) {
            this.inferencePassesThisRun = 0;
            requestAnimationFrame(function () { return _this.inferNetwork(); });
        }
        this.isInferring = true;
    };
    GraphRunner.prototype.inferNetwork = function () {
        var _this = this;
        if (!this.isInferring ||
            this.inferencePassesThisRun === this.currentInferenceLoopNumPasses) {
            return;
        }
        this.math.scope(function (keep, track) {
            var feeds = [];
            var inferenceValues = [];
            var start = performance.now();
            for (var i = 0; i < _this.inferenceExampleCount; i++) {
                var ndarrayFeedEntries = [];
                for (var j = 0; j < _this.inferenceFeedEntries.length; j++) {
                    var feedEntry = _this.inferenceFeedEntries[j];
                    ndarrayFeedEntries.push({
                        tensor: feedEntry.tensor,
                        data: track(feedEntry.data.getNextCopy(_this.math))
                    });
                }
                feeds.push(ndarrayFeedEntries);
                inferenceValues.push(_this.session.eval(_this.inferenceTensor, ndarrayFeedEntries));
            }
            if (_this.eventObserver.inferenceExamplesPerSecCallback != null) {
                inferenceValues[inferenceValues.length - 1].getValues();
                var inferenceExamplesPerSecTime = performance.now() - start;
                var examplesPerSec = (_this.inferenceExampleCount * 1000 / inferenceExamplesPerSecTime);
                _this.eventObserver.inferenceExamplesPerSecCallback(examplesPerSec);
            }
            if (_this.eventObserver.inferenceExamplesCallback != null) {
                _this.eventObserver.inferenceExamplesCallback(feeds, inferenceValues);
            }
            _this.inferencePassesThisRun++;
        });
        this.lastInferTimeoutID = window.setTimeout(function () { return _this.inferNetwork(); }, this.inferenceExampleIntervalMs);
    };
    GraphRunner.prototype.stopInferring = function () {
        this.isInferring = false;
        window.clearTimeout(this.lastInferTimeoutID);
    };
    GraphRunner.prototype.isInferenceRunning = function () {
        return this.isInferring;
    };
    GraphRunner.prototype.computeMetric = function () {
        var _this = this;
        if (this.metricFeedEntries == null) {
            throw new Error('Cannot compute metric, no metric FeedEntries provided.');
        }
        var metric = this.zeroScalar;
        return this.math.scope(function (keep) {
            for (var i = 0; i < _this.metricBatchSize; i++) {
                var metricValue = _this.session.eval(_this.metricTensor, _this.metricFeedEntries);
                metric = _this.math.add(metric, metricValue);
            }
            if (_this.metricReduction === MetricReduction.MEAN) {
                metric = _this.math.divide(metric, _this.metricBatchSizeScalar);
            }
            return metric;
        });
    };
    GraphRunner.prototype.getTotalBatchesTrained = function () {
        return this.totalBatchesTrained;
    };
    GraphRunner.prototype.getLastComputedMetric = function () {
        return this.lastComputedMetric;
    };
    GraphRunner.prototype.setMath = function (math) {
        this.math = math;
    };
    GraphRunner.prototype.setSession = function (session) {
        this.session = session;
    };
    GraphRunner.prototype.setInferenceTensor = function (inferenceTensor) {
        this.inferenceTensor = inferenceTensor;
    };
    GraphRunner.prototype.setInferenceExampleCount = function (inferenceExampleCount) {
        this.inferenceExampleCount = inferenceExampleCount;
    };
    return GraphRunner;
}());
exports.GraphRunner = GraphRunner;

},{"./math/ndarray":26,"./session":75}],14:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var graph_1 = require("./graph");
var priority_queue = require("./priority_queue");
var priority_queue_1 = require("./priority_queue");
function getUnorderedEvaluationSet(nodes, terminatingNodes) {
    var terminatingNodeMap = {};
    var seen = {};
    var set = [];
    var visit = nodes.slice();
    terminatingNodes.forEach(function (node) { return terminatingNodeMap[node.id] = node; });
    var _loop_1 = function () {
        var cur = visit.pop();
        if (seen[cur.id] == null) {
            if (terminatingNodeMap[cur.id] == null) {
                Object.keys(cur.inputs)
                    .map(function (inputName) { return cur.inputs[inputName]; })
                    .forEach(function (input) { return visit.push(input.node); });
            }
            set.push(cur);
            seen[cur.id] = cur;
        }
    };
    while (visit.length !== 0) {
        _loop_1();
    }
    return set;
}
exports.getUnorderedEvaluationSet = getUnorderedEvaluationSet;
function getOrderedEvaluationSet(unorderedEvaluationSet) {
    var set = [];
    var nodeIndices = {};
    var pendingDependencies = {};
    var nodeQueue = new priority_queue_1.PriorityQueue(function (a, b) { return priority_queue.defaultCompare(pendingDependencies[a.id], pendingDependencies[b.id]); }, function (node, newIndex) { return nodeIndices[node.id] = newIndex; });
    unorderedEvaluationSet.forEach(function (node) { return pendingDependencies[node.id] = 0; });
    unorderedEvaluationSet.forEach(function (node) { return Object.keys(node.inputs)
        .map(function (key) { return node.inputs[key]; })
        .forEach(function (input) {
        if (unorderedEvaluationSet.indexOf(input.node) !== -1) {
            pendingDependencies[input.node.id]++;
        }
    }); });
    unorderedEvaluationSet.forEach(function (node) { return nodeQueue.enqueue(node); });
    while (!nodeQueue.empty()) {
        set.unshift(nodeQueue.dequeue());
        Object.keys(set[0].inputs).map(function (key) { return set[0].inputs[key]; }).forEach(function (input) {
            if (unorderedEvaluationSet.indexOf(input.node) === -1) {
                return;
            }
            pendingDependencies[input.node.id]--;
            nodeQueue.update(input.node, nodeIndices[input.node.id]);
        });
    }
    return set;
}
exports.getOrderedEvaluationSet = getOrderedEvaluationSet;
function isInputNode(node) {
    return Object.keys(node.inputs).length === 0;
}
exports.isInputNode = isInputNode;
function shouldBackProp(t) {
    return !(t.node instanceof graph_1.ConstantNode);
}
exports.shouldBackProp = shouldBackProp;
function isPassthroughNode(node, map) {
    var keys = Object.keys(node.inputs);
    for (var i = 0; i < keys.length; i++) {
        var input = node.inputs[keys[i]];
        if (map.get(input, true) === map.get(node.output, true)) {
            return true;
        }
    }
    return false;
}
exports.isPassthroughNode = isPassthroughNode;

},{"./graph":11,"./priority_queue":74}],15:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("./math/conv_util");
exports.conv_util = conv_util;
var gpgpu_util = require("./math/webgl/gpgpu_util");
exports.gpgpu_util = gpgpu_util;
var render_ndarray_gpu_util = require("./math/webgl/render_ndarray_gpu_util");
exports.render_ndarray_gpu_util = render_ndarray_gpu_util;
var webgl_util = require("./math/webgl/webgl_util");
exports.webgl_util = webgl_util;
var util = require("./util");
exports.util = util;
var checkpoint_loader_1 = require("./checkpoint_loader");
exports.CheckpointLoader = checkpoint_loader_1.CheckpointLoader;
var dataset_1 = require("./dataset");
exports.InMemoryDataset = dataset_1.InMemoryDataset;
var graph_1 = require("./graph");
exports.Graph = graph_1.Graph;
exports.Tensor = graph_1.Tensor;
var graph_runner_1 = require("./graph_runner");
exports.GraphRunner = graph_runner_1.GraphRunner;
exports.MetricReduction = graph_runner_1.MetricReduction;
var initializers_1 = require("./initializers");
exports.ConstantInitializer = initializers_1.ConstantInitializer;
exports.NDArrayInitializer = initializers_1.NDArrayInitializer;
exports.OnesInitializer = initializers_1.OnesInitializer;
exports.RandomNormalInitializer = initializers_1.RandomNormalInitializer;
exports.RandomTruncatedNormalInitializer = initializers_1.RandomTruncatedNormalInitializer;
exports.RandomUniformInitializer = initializers_1.RandomUniformInitializer;
exports.VarianceScalingInitializer = initializers_1.VarianceScalingInitializer;
exports.ZerosInitializer = initializers_1.ZerosInitializer;
var input_provider_1 = require("./input_provider");
exports.InCPUMemoryShuffledInputProviderBuilder = input_provider_1.InCPUMemoryShuffledInputProviderBuilder;
exports.InGPUMemoryShuffledInputProviderBuilder = input_provider_1.InGPUMemoryShuffledInputProviderBuilder;
var math_1 = require("./math/math");
exports.MatrixOrientation = math_1.MatrixOrientation;
exports.NDArrayMath = math_1.NDArrayMath;
var math_cpu_1 = require("./math/math_cpu");
exports.NDArrayMathCPU = math_cpu_1.NDArrayMathCPU;
var math_gpu_1 = require("./math/math_gpu");
exports.NDArrayMathGPU = math_gpu_1.NDArrayMathGPU;
var ndarray_1 = require("./math/ndarray");
exports.Array1D = ndarray_1.Array1D;
exports.Array2D = ndarray_1.Array2D;
exports.Array3D = ndarray_1.Array3D;
exports.Array4D = ndarray_1.Array4D;
exports.NDArray = ndarray_1.NDArray;
exports.Scalar = ndarray_1.Scalar;
var gpgpu_context_1 = require("./math/webgl/gpgpu_context");
exports.GPGPUContext = gpgpu_context_1.GPGPUContext;
var optimizer_1 = require("./optimizer");
exports.Optimizer = optimizer_1.Optimizer;
var session_1 = require("./session");
exports.CostReduction = session_1.CostReduction;
exports.Session = session_1.Session;
var sgd_optimizer_1 = require("./sgd_optimizer");
exports.SGDOptimizer = sgd_optimizer_1.SGDOptimizer;
var momentum_optimizer_1 = require("./momentum_optimizer");
exports.MomentumOptimizer = momentum_optimizer_1.MomentumOptimizer;

},{"./checkpoint_loader":9,"./dataset":10,"./graph":11,"./graph_runner":13,"./initializers":16,"./input_provider":17,"./math/conv_util":20,"./math/math":23,"./math/math_cpu":24,"./math/math_gpu":25,"./math/ndarray":26,"./math/webgl/gpgpu_context":36,"./math/webgl/gpgpu_util":38,"./math/webgl/render_ndarray_gpu_util":45,"./math/webgl/webgl_util":51,"./momentum_optimizer":52,"./optimizer":73,"./session":75,"./sgd_optimizer":77,"./util":79}],16:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var VarianceScalingInitializer = (function () {
    function VarianceScalingInitializer(scale, mode, distribution) {
        if (scale === void 0) { scale = 1.0; }
        if (mode === void 0) { mode = 'fan_in'; }
        if (distribution === void 0) { distribution = 'normal'; }
        this.scale = scale;
        this.mode = mode;
        this.distribution = distribution;
    }
    VarianceScalingInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        var n = 0;
        if (this.mode === 'fan_in') {
            n = inputUnits;
        }
        else if (this.mode === 'fan_out') {
            n = outputUnits;
        }
        else if (this.mode === 'fan_avg') {
            n = (inputUnits + outputUnits) / 2;
        }
        else {
            throw new Error('Unexpected mode for variance scaling initializer: ' + this.mode);
        }
        if (this.distribution === 'normal') {
            return ndarray_1.NDArray.randTruncatedNormal(weightsShape, 0.0, Math.sqrt(this.scale / n));
        }
        else if (this.distribution === 'uniform') {
            return ndarray_1.NDArray.randUniform(weightsShape, 0.0, Math.sqrt(3 * this.scale / n));
        }
        else {
            throw new Error('Unexpected distribution for variance scaling initializer: ' +
                this.distribution);
        }
    };
    return VarianceScalingInitializer;
}());
exports.VarianceScalingInitializer = VarianceScalingInitializer;
var ZerosInitializer = (function () {
    function ZerosInitializer() {
    }
    ZerosInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        return ndarray_1.NDArray.zeros(weightsShape);
    };
    return ZerosInitializer;
}());
exports.ZerosInitializer = ZerosInitializer;
var OnesInitializer = (function () {
    function OnesInitializer() {
    }
    OnesInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        var values = ndarray_1.NDArray.zeros(weightsShape);
        values.fill(1);
        return values;
    };
    return OnesInitializer;
}());
exports.OnesInitializer = OnesInitializer;
var ConstantInitializer = (function () {
    function ConstantInitializer(value) {
        if (value === void 0) { value = 0; }
        this.value = value;
    }
    ConstantInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        var values = ndarray_1.NDArray.zeros(weightsShape);
        values.fill(this.value);
        return values;
    };
    return ConstantInitializer;
}());
exports.ConstantInitializer = ConstantInitializer;
var NDArrayInitializer = (function () {
    function NDArrayInitializer(ndarray) {
        this.ndarray = ndarray;
    }
    NDArrayInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        return this.ndarray;
    };
    return NDArrayInitializer;
}());
exports.NDArrayInitializer = NDArrayInitializer;
var RandomNormalInitializer = (function () {
    function RandomNormalInitializer(mean, stdev) {
        if (mean === void 0) { mean = 0; }
        if (stdev === void 0) { stdev = .05; }
        this.mean = mean;
        this.stdev = stdev;
    }
    RandomNormalInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        return ndarray_1.NDArray.randNormal(weightsShape, this.mean, this.stdev);
    };
    return RandomNormalInitializer;
}());
exports.RandomNormalInitializer = RandomNormalInitializer;
var RandomTruncatedNormalInitializer = (function () {
    function RandomTruncatedNormalInitializer(mean, stdev) {
        if (mean === void 0) { mean = 0; }
        if (stdev === void 0) { stdev = .05; }
        this.mean = mean;
        this.stdev = stdev;
    }
    RandomTruncatedNormalInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        return ndarray_1.NDArray.randTruncatedNormal(weightsShape, this.mean, this.stdev);
    };
    return RandomTruncatedNormalInitializer;
}());
exports.RandomTruncatedNormalInitializer = RandomTruncatedNormalInitializer;
var RandomUniformInitializer = (function () {
    function RandomUniformInitializer(minval, maxval) {
        if (minval === void 0) { minval = -.05; }
        if (maxval === void 0) { maxval = .05; }
        this.minval = minval;
        this.maxval = maxval;
    }
    RandomUniformInitializer.prototype.initialize = function (weightsShape, inputUnits, outputUnits) {
        return ndarray_1.NDArray.randUniform(weightsShape, this.minval, this.maxval);
    };
    return RandomUniformInitializer;
}());
exports.RandomUniformInitializer = RandomUniformInitializer;

},{"./math/ndarray":26}],17:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var util = require("./util");
var InMemoryShuffledInputProviderBuilder = (function () {
    function InMemoryShuffledInputProviderBuilder(inputs) {
        this.inputs = inputs;
        this.idx = 0;
        this.inputCounter = 0;
        this.epoch = 0;
        this.shuffledIndices = util.createShuffledIndices(inputs[0].length);
        this.numInputs = inputs.length;
        var numExamples = this.inputs[0].length;
        for (var i = 0; i < this.numInputs; i++) {
            util.assert(this.inputs[i].length === numExamples, 'Number of examples must match across different inputs.');
        }
        for (var i = 0; i < this.numInputs; i++) {
            var inputShape = this.inputs[i][0].shape;
            for (var j = 0; j < this.inputs[i].length; j++) {
                util.assertShapesMatch(inputShape, this.inputs[i][j].shape);
            }
        }
    }
    InMemoryShuffledInputProviderBuilder.prototype.getCurrentExampleIndex = function () {
        var returnIdx = this.idx;
        this.inputCounter++;
        if (this.inputCounter >= this.numInputs) {
            this.idx++;
            this.inputCounter = 0;
            if (this.idx >= this.inputs[0].length) {
                this.idx = 0;
                this.epoch++;
            }
        }
        return returnIdx;
    };
    InMemoryShuffledInputProviderBuilder.prototype.getNextInput = function (inputId) {
        var currentExampleIndex = this.getCurrentExampleIndex();
        return this.inputs[inputId][this.shuffledIndices[currentExampleIndex]];
    };
    InMemoryShuffledInputProviderBuilder.prototype.getEpoch = function () {
        return this.epoch;
    };
    InMemoryShuffledInputProviderBuilder.prototype.getInputProviders = function () {
        var inputProviders = [];
        for (var i = 0; i < this.numInputs; i++) {
            inputProviders.push(this.getInputProvider(i));
        }
        return inputProviders;
    };
    return InMemoryShuffledInputProviderBuilder;
}());
exports.InMemoryShuffledInputProviderBuilder = InMemoryShuffledInputProviderBuilder;
var InCPUMemoryShuffledInputProviderBuilder = (function (_super) {
    __extends(InCPUMemoryShuffledInputProviderBuilder, _super);
    function InCPUMemoryShuffledInputProviderBuilder() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    InCPUMemoryShuffledInputProviderBuilder.prototype.getInputProvider = function (inputId) {
        var shuffledInputProvider = this;
        return {
            getNextCopy: function (math) {
                return ndarray_1.NDArray.like(shuffledInputProvider.getNextInput(inputId));
            },
            disposeCopy: function (math, copy) {
                copy.dispose();
            }
        };
    };
    return InCPUMemoryShuffledInputProviderBuilder;
}(InMemoryShuffledInputProviderBuilder));
exports.InCPUMemoryShuffledInputProviderBuilder = InCPUMemoryShuffledInputProviderBuilder;
var InGPUMemoryShuffledInputProviderBuilder = (function (_super) {
    __extends(InGPUMemoryShuffledInputProviderBuilder, _super);
    function InGPUMemoryShuffledInputProviderBuilder() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    InGPUMemoryShuffledInputProviderBuilder.prototype.getInputProvider = function (inputId) {
        var shuffledInputProvider = this;
        return {
            getNextCopy: function (math) {
                return math.clone(shuffledInputProvider.getNextInput(inputId));
            },
            disposeCopy: function (math, copy) {
                copy.dispose();
            }
        };
    };
    return InGPUMemoryShuffledInputProviderBuilder;
}(InMemoryShuffledInputProviderBuilder));
exports.InGPUMemoryShuffledInputProviderBuilder = InGPUMemoryShuffledInputProviderBuilder;

},{"./math/ndarray":26,"./util":79}],18:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./ndarray");
var TanHFunc = (function () {
    function TanHFunc() {
    }
    TanHFunc.prototype.output = function (math, x) {
        return math.scope(function () {
            return math.tanh(x);
        });
    };
    TanHFunc.prototype.der = function (math, x, y) {
        return math.scope(function () {
            var ySquared = math.elementWiseMul(y, y);
            return math.scalarMinusArray(ndarray_1.Scalar.ONE, ySquared);
        });
    };
    return TanHFunc;
}());
exports.TanHFunc = TanHFunc;
var ReLUFunc = (function () {
    function ReLUFunc() {
    }
    ReLUFunc.prototype.output = function (math, x) {
        return math.scope(function () {
            return math.relu(x);
        });
    };
    ReLUFunc.prototype.der = function (math, x, y) {
        return math.scope(function () {
            return math.step(x);
        });
    };
    return ReLUFunc;
}());
exports.ReLUFunc = ReLUFunc;
var SigmoidFunc = (function () {
    function SigmoidFunc() {
    }
    SigmoidFunc.prototype.output = function (math, x) {
        return math.scope(function () {
            return math.sigmoid(x);
        });
    };
    SigmoidFunc.prototype.der = function (math, x, y) {
        return math.scope(function () {
            var ySquared = math.elementWiseMul(y, y);
            return math.subStrict(y, ySquared);
        });
    };
    return SigmoidFunc;
}());
exports.SigmoidFunc = SigmoidFunc;
var SquareFunc = (function () {
    function SquareFunc() {
    }
    SquareFunc.prototype.output = function (math, x) {
        return math.scope(function () {
            return math.elementWiseMul(x, x);
        });
    };
    SquareFunc.prototype.der = function (math, x, y) {
        return math.scope(function () {
            return math.scalarTimesArray(ndarray_1.Scalar.TWO, x);
        });
    };
    return SquareFunc;
}());
exports.SquareFunc = SquareFunc;

},{"./ndarray":26}],19:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
function assertConcat3DShapesMatch(x1Shape, x2Shape, axis, errorMessagePrefix) {
    if (errorMessagePrefix === void 0) { errorMessagePrefix = ''; }
    util.assert(x1Shape.length === 3, errorMessagePrefix + 'Concat3D x1 shape should be of rank 3.');
    util.assert(x2Shape.length === 3, errorMessagePrefix + 'Concat3D x2 shape should be of rank 3.');
    util.assert(axis >= 0 && axis < 3, 'Axis for concat3D must be between 0 and 2.');
    for (var i = 0; i < 3; i++) {
        util.assert((i === axis) || (x1Shape[i] === x2Shape[i]), errorMessagePrefix +
            ("Shape (" + x1Shape + ") does not match (" + x2Shape + ") along ") +
            "non-concatenated axis.");
    }
}
exports.assertConcat3DShapesMatch = assertConcat3DShapesMatch;
function computeConcat3DOutputShape(x1Shape, x2Shape, axis) {
    util.assert(x1Shape.length === 3, 'Concat3D x1 shape should be of rank 3.');
    util.assert(x2Shape.length === 3, 'Concat3D x2shape should be of rank 3.');
    var outputShape = x1Shape.slice();
    outputShape[axis] += x2Shape[axis];
    return outputShape;
}
exports.computeConcat3DOutputShape = computeConcat3DOutputShape;

},{"../util":79}],20:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
function computeConvInfo(inShape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, pad) {
    if (typeof pad === 'number') {
        var outShape_1 = computeOutputShape3D(inShape, filterHeight, outDepth, strideHeight, pad);
        return {
            inShape: inShape,
            outShape: outShape_1,
            padInfo: { top: pad, bottom: pad, left: pad, right: pad },
            strideHeight: strideHeight,
            strideWidth: strideWidth,
            filterHeight: filterHeight,
            filterWidth: filterWidth
        };
    }
    var inHeight = inShape[0];
    var inWidth = inShape[1];
    var outShape;
    var padInfo;
    if (pad === 'same') {
        var outHeight = Math.ceil(inHeight / strideHeight);
        var outWidth = Math.ceil(inWidth / strideWidth);
        outShape = [outHeight, outWidth, outDepth];
        var padAlongHeight = (outHeight - 1) * strideHeight + filterHeight - inHeight;
        var padAlongWidth = (outWidth - 1) * strideWidth + filterWidth - inWidth;
        var top_1 = Math.floor(padAlongHeight / 2);
        var bottom = padAlongHeight - top_1;
        var left = Math.floor(padAlongWidth / 2);
        var right = padAlongWidth - left;
        padInfo = { top: top_1, bottom: bottom, left: left, right: right };
    }
    else if (pad === 'valid') {
        var outHeight = Math.ceil((inHeight - filterHeight + 1) / strideHeight);
        var outWidth = Math.ceil((inWidth - filterWidth + 1) / strideWidth);
        outShape = [outHeight, outWidth, outDepth];
        padInfo = { top: 0, bottom: 0, left: 0, right: 0 };
    }
    else {
        throw Error("Unknown padding parameter: " + pad);
    }
    return {
        inShape: inShape,
        outShape: outShape,
        padInfo: padInfo,
        strideHeight: strideHeight,
        strideWidth: strideWidth,
        filterHeight: filterHeight,
        filterWidth: filterWidth
    };
}
exports.computeConvInfo = computeConvInfo;
function computeOutputShape3D(inShape, fieldSize, outDepth, stride, zeroPad) {
    if (zeroPad == null) {
        zeroPad = computeDefaultPad(inShape, fieldSize, stride);
    }
    var inputRows = inShape[0];
    var inputCols = inShape[1];
    var outputRows = (inputRows - fieldSize + 2 * zeroPad) / stride + 1;
    util.assert(util.isInt(outputRows), "The output # of rows (" + outputRows + ") must be an integer. Change the " +
        "stride and/or zero pad parameters");
    var outputCols = (inputCols - fieldSize + 2 * zeroPad) / stride + 1;
    util.assert(util.isInt(outputCols), "The output # of columns (" + outputCols + ") must be an integer. Change " +
        "the stride and/or zero pad parameters");
    return [outputRows, outputCols, outDepth];
}
exports.computeOutputShape3D = computeOutputShape3D;
function computeDefaultPad(inputShape, fieldSize, stride) {
    return Math.floor((inputShape[0] * (stride - 1) - stride + fieldSize) / 2);
}
exports.computeDefaultPad = computeDefaultPad;
function computeTexShapeFrom3D(shapeRowColDepth) {
    return [shapeRowColDepth[0], shapeRowColDepth[1] * shapeRowColDepth[2]];
}
exports.computeTexShapeFrom3D = computeTexShapeFrom3D;
function computeWeightsShape4D(inputDepth, outputDepth, filterHeight, filterWidth) {
    return [filterHeight, filterWidth, inputDepth, outputDepth];
}
exports.computeWeightsShape4D = computeWeightsShape4D;
function computeDilatedRC(rc, origStride) {
    var rowsDilated = (rc[0] - 1) * origStride + 1;
    var colsDilated = (rc[1] - 1) * origStride + 1;
    return [rowsDilated, colsDilated];
}
exports.computeDilatedRC = computeDilatedRC;

},{"../util":79}],21:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function validateShapes(sourceSize, destSize) {
    var srcArea = sourceSize[0] * sourceSize[1];
    var dstArea = destSize[0] * destSize[1];
    if (srcArea !== dstArea) {
        var srcStr = '[' + sourceSize[0] + ', ' + sourceSize[1] + ']';
        var dstStr = '[' + destSize[0] + ', ' + destSize[1] + ']';
        throw new Error('copy2D shapes have different areas:\n  sourceSize ' + srcStr +
            ', area ' + srcArea + '\n  destSize ' + dstStr + ', area ' + dstArea);
    }
}
exports.validateShapes = validateShapes;

},{}],22:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./ndarray");
var SquareCostFunc = (function () {
    function SquareCostFunc() {
        this.halfOne = ndarray_1.Scalar.new(0.5);
    }
    SquareCostFunc.prototype.cost = function (math, x1, x2) {
        var diff = math.subStrict(x1, x2);
        var diffSquared = math.elementWiseMul(diff, diff);
        var result = math.scalarTimesArray(this.halfOne, diffSquared);
        diff.dispose();
        diffSquared.dispose();
        return result;
    };
    SquareCostFunc.prototype.der = function (math, x1, x2) {
        return math.subStrict(x1, x2);
    };
    SquareCostFunc.prototype.dispose = function () {
        this.halfOne.dispose();
    };
    return SquareCostFunc;
}());
exports.SquareCostFunc = SquareCostFunc;

},{"./ndarray":26}],23:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
var concat3d_util = require("./concat3d_util");
var conv_util = require("./conv_util");
var copy2d_util = require("./copy2d_util");
var ndarray_1 = require("./ndarray");
var NDArrayMath = (function () {
    function NDArrayMath(safeMode) {
        this.safeMode = safeMode;
        this.ndarrayScopes = [];
        this.ndarraysToKeep = [];
        this.activeScopeNDArraysToKeep = [];
        this.debugMode = false;
    }
    NDArrayMath.prototype.scope = function (scopeFn) {
        var _this = this;
        this.startScope();
        var keepFn = function (ndarray) { return _this.keep(ndarray); };
        var trackFn = function (ndarray) { return _this.track(ndarray); };
        var result = scopeFn(keepFn, trackFn);
        this.endScope(result);
        return result;
    };
    NDArrayMath.prototype.enableDebugMode = function () {
        this.debugMode = true;
        console.warn('Debugging mode is ON. The output of every math call will ' +
            'be downloaded to CPU and checked for NaNs. ' +
            'This significantly impacts performance.');
    };
    NDArrayMath.prototype.startScope = function () {
        var newScope = [];
        this.ndarrayScopes.push(newScope);
        this.activeScope = newScope;
        var newNDArraysToKeep = [];
        this.ndarraysToKeep.push(newNDArraysToKeep);
        this.activeScopeNDArraysToKeep = newNDArraysToKeep;
    };
    NDArrayMath.prototype.endScope = function (result) {
        var _this = this;
        var arraysToKeep = this.activeScopeNDArraysToKeep;
        if (result != null) {
            arraysToKeep = arraysToKeep.concat(result);
        }
        for (var i = 0; i < this.activeScope.length; i++) {
            var ndarray = this.activeScope[i];
            if (this.isNDArrayDataInList(ndarray, arraysToKeep)) {
                continue;
            }
            ndarray.dispose();
        }
        this.ndarrayScopes.pop();
        this.activeScope = this.ndarrayScopes.length === 0 ?
            null :
            this.ndarrayScopes[this.ndarrayScopes.length - 1];
        if (result instanceof ndarray_1.NDArray &&
            !this.isNDArrayDataInList(result, this.activeScopeNDArraysToKeep)) {
            this.track(result);
        }
        else if (Array.isArray(result)) {
            result.forEach(function (r) {
                if (r instanceof ndarray_1.NDArray &&
                    !_this.isNDArrayDataInList(r, _this.activeScopeNDArraysToKeep)) {
                    _this.track(r);
                }
            });
        }
        this.ndarraysToKeep.pop();
        this.activeScopeNDArraysToKeep = this.ndarraysToKeep.length === 0 ?
            null :
            this.ndarraysToKeep[this.ndarraysToKeep.length - 1];
    };
    NDArrayMath.prototype.isNDArrayDataInList = function (ndarray, ndarrayList) {
        for (var i = 0; i < ndarrayList.length; i++) {
            if (ndarrayList[i].getData() === ndarray.getData()) {
                return true;
            }
        }
        return false;
    };
    NDArrayMath.prototype.keep = function (result) {
        if (this.activeScope == null) {
            if (this.safeMode) {
                throw new Error('You are using math in safe mode. Enclose all ' +
                    'math.method() calls inside a scope: ' +
                    'math.scope(() => {math.method();...}) to avoid memory ' +
                    'leaks.');
            }
            return result;
        }
        this.activeScopeNDArraysToKeep.push(result);
        return result;
    };
    NDArrayMath.prototype.checkForNaN = function (vals, name) {
        for (var i = 0; i < vals.length; i++) {
            if (isNaN(vals[i])) {
                throw Error("The result of the last math." + name + " has NaNs.");
            }
        }
    };
    NDArrayMath.prototype.track = function (result) {
        if (this.activeScope == null) {
            if (this.safeMode) {
                throw new Error('You are using math in safe mode. Enclose all ' +
                    'math.method() calls inside a scope: ' +
                    'math.scope(() => {math.method();...}) to avoid memory ' +
                    'leaks.');
            }
            return result;
        }
        this.activeScope.push(result);
        return result;
    };
    NDArrayMath.prototype.matMul = function (a, b, aOrientation, bOrientation) {
        var _this = this;
        if (aOrientation === void 0) { aOrientation = MatrixOrientation.REGULAR; }
        if (bOrientation === void 0) { bOrientation = MatrixOrientation.REGULAR; }
        var innerShapeA = (aOrientation === MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
        var innerShapeB = (bOrientation === MatrixOrientation.REGULAR) ? b.shape[0] : b.shape[1];
        util.assert(a.rank === 2 && b.rank === 2, "Error in matMul: inputs must be rank 2, got ranks " + a.rank +
            ("and " + b.rank + "."));
        util.assert(innerShapeA === innerShapeB, "Error in matMul: inner shapes (" + innerShapeA + ") and (" +
            (innerShapeB + ") of NDArrays with shapes " + a.shape + " and ") +
            (b.shape + " and orientations " + MatrixOrientation[aOrientation]) +
            (" and " + MatrixOrientation[bOrientation] + " must match."));
        return this.executeOp('matMul', function () { return _this.matMulInternal(a, b, aOrientation, bOrientation); });
    };
    NDArrayMath.prototype.executeOp = function (name, f) {
        var start;
        if (this.debugMode) {
            start = performance.now();
        }
        var result = f();
        if (this.debugMode) {
            var vals = result.getValues();
            var time = util.rightPad((performance.now() - start) + 'ms', 9);
            var paddedName = util.rightPad(name, 25);
            var rank = result.rank;
            var size = result.size;
            var shape = util.rightPad(result.shape + '', 14);
            console.log("%c" + paddedName + "\t%c" + time + "\t%c" + rank + "D " + shape + "\t%c" + size, 'font-weight:bold', 'color:red', 'color:blue', 'color: orange');
            this.checkForNaN(vals, name);
        }
        return this.track(result);
    };
    NDArrayMath.prototype.vectorTimesMatrix = function (v, matrix) {
        util.assert(v.rank === 1, "Error in vectorTimesMatrix: first input must be rank 1, but got " +
            ("rank " + v.rank + "."));
        util.assert(matrix.rank === 2, "Error in vectorTimesMatrix: second input must be rank 2, but got " +
            ("rank " + matrix.rank + "."));
        util.assert(v.size === matrix.shape[0], "Error in vectorTimesMatrix: size of first rank 1 input (" + v.size + ") " +
            "must match inner dimension of second rank 2 input, but got " +
            ("rank " + matrix.rank + "."));
        return this.matMul(v.as2D(1, v.size), matrix).as1D();
    };
    NDArrayMath.prototype.matrixTimesVector = function (matrix, v) {
        util.assert(v.rank === 1, "Error in vectorTimesMatrix: second input must rank 1, but got " +
            ("rank " + v.rank + "."));
        util.assert(matrix.rank === 2, "Error in vectorTimesMatrix: first input must be a rank 2, but got " +
            ("rank " + matrix.rank + "."));
        util.assert(v.size === matrix.shape[1], "Error in vectorTimesMatrix: size of first rank 1 input " + v.size + " " +
            "must match inner dimension of second rank 2 input, but got " +
            ("shape " + matrix.shape + "."));
        return this.matMul(matrix, v.as2D(v.size, 1)).as1D();
    };
    NDArrayMath.prototype.dotProduct = function (v1, v2) {
        util.assert(v1.rank === 1 && v2.rank === 1, "Error in dotProduct: inputs must be rank 1, but got ranks " +
            (v1.rank + " and " + v2.rank + "."));
        util.assert(v1.size === v2.size, "Error in dotProduct: size of inputs (" + v1.size + ") and (" +
            (v2.size + ") must match."));
        return this.matMul(v1.as2D(1, v1.size), v2.as2D(v2.size, 1)).asScalar();
    };
    NDArrayMath.prototype.outerProduct = function (v1, v2) {
        util.assert(v1.rank === 1 && v2.rank === 1, "Error in outerProduct: inputs must be rank 1, but got ranks " +
            (v1.rank + " and " + v2.rank + "."));
        return this.matMul(v1.as2D(v1.size, 1), v2.as2D(1, v2.size));
    };
    NDArrayMath.prototype.clone = function (ndarray) {
        var _this = this;
        return this.executeOp('clone', function () { return _this.cloneInternal(ndarray); });
    };
    NDArrayMath.prototype.reshape = function (ndarray, newShape) {
        console.warn('math.reshape() is deprecated. Please call reshape() ' +
            'directly on the ndarray object');
        return ndarray.reshape(newShape);
    };
    NDArrayMath.prototype.slice2D = function (input, begin, size) {
        var _this = this;
        util.assert(begin[0] + size[0] <= input.shape[0] &&
            begin[1] + size[1] <= input.shape[1], "Error in slice2D: requested start position " + begin + " and size " +
            (size + " would overflow input of shape " + input.shape + "."));
        return this.executeOp('slice2D', function () { return _this.slice2DInternal(input, begin, size); });
    };
    NDArrayMath.prototype.copy2D = function (source, sourceBegin, sourceSize, dest, destBegin, destSize) {
        var _this = this;
        util.assert(sourceBegin[0] + sourceSize[0] <= source.shape[0] &&
            sourceBegin[1] + sourceSize[1] <= source.shape[1], "Error in copy2D: requested source start position " + sourceBegin + " " +
            ("and source size " + sourceSize + " would overflow source NDArray") +
            ("of shape " + source.shape + "."));
        util.assert(destBegin[0] + destSize[0] <= dest.shape[0] &&
            destBegin[1] + destSize[1] <= dest.shape[1], "Error in copy2D: requested dest start position " + destBegin + " " +
            ("and source size " + destSize + " would overflow dest NDArray of") +
            ("shape " + dest.shape + "."));
        copy2d_util.validateShapes(sourceSize, destSize);
        this.executeOp('copy2D', function () {
            _this.copy2DInternal(source, sourceBegin, sourceSize, dest, destBegin, destSize);
            return dest;
        });
    };
    NDArrayMath.prototype.concat3D = function (ndarray1, ndarray2, axis) {
        var _this = this;
        concat3d_util.assertConcat3DShapesMatch(ndarray1.shape, ndarray2.shape, axis, 'Error in concat3d: ');
        return this.executeOp('concat3D', function () { return _this.concat3DInternal(ndarray1, ndarray2, axis); });
    };
    NDArrayMath.prototype.logSumExp = function (ndarray) {
        var _this = this;
        return this.executeOp('logSumExp', function () { return _this.logSumExpInternal(ndarray); });
    };
    NDArrayMath.prototype.sum = function (ndarray) {
        var _this = this;
        return this.executeOp('sum', function () { return _this.sumInternal(ndarray); });
    };
    NDArrayMath.prototype.argMin = function (ndarray) {
        var _this = this;
        return this.executeOp('argMin', function () { return _this.argMinInternal(ndarray); });
    };
    NDArrayMath.prototype.argMax = function (ndarray) {
        var _this = this;
        return this.executeOp('argMax', function () { return _this.argMaxInternal(ndarray); });
    };
    NDArrayMath.prototype.argMaxEquals = function (x1, x2) {
        var _this = this;
        util.assertShapesMatch(x1.shape, x2.shape, 'Error in argMaxEquals: ');
        return this.executeOp('argMaxEquals', function () { return _this.argMaxEqualsInternal(x1, x2); });
    };
    NDArrayMath.prototype.topK = function (ndarray, k) {
        var _this = this;
        util.assert(k <= ndarray.size, "Error in topK: k value (" + k + ") must be less than size of input " +
            ("ndarray, got shape " + ndarray.shape + "."));
        var result;
        this.executeOp('topK', function () {
            result = _this.topKInternal(ndarray, k);
            return result.values;
        });
        this.track(result.indices);
        return result;
    };
    NDArrayMath.prototype.min = function (ndarray) {
        var _this = this;
        return this.executeOp('min', function () { return _this.minInternal(ndarray); });
    };
    NDArrayMath.prototype.max = function (ndarray) {
        var _this = this;
        return this.executeOp('max', function () { return _this.maxInternal(ndarray); });
    };
    NDArrayMath.prototype.softmax = function (x) {
        var _this = this;
        return this.executeOp('softmax', function () {
            return _this.scope(function () {
                var lse = _this.logSumExp(x);
                var logResult = _this.arrayMinusScalar(x, lse);
                return _this.exp(logResult);
            });
        });
    };
    NDArrayMath.prototype.switchDim = function (a, newDim) {
        var _this = this;
        util.assert(a.rank === newDim.length, "Error in switchDim: length of input shape " + a.shape + " " +
            ("must match size of newDim array " + newDim + "."));
        return this.executeOp('switchDim', function () { return _this.switchDimInternal(a, newDim); });
    };
    NDArrayMath.prototype.scalarPlusArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarPlusArray: first argument must be rank 0, but got " +
            ("rank " + c.rank + "."));
        return this.add(c, a);
    };
    NDArrayMath.prototype.scalarMinusArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarMinusArray: first argument must be rank 0, but got " +
            ("rank " + c.rank + "."));
        return this.sub(c, a);
    };
    NDArrayMath.prototype.arrayMinusScalar = function (a, c) {
        util.assert(c.size === 1, "Error in arrayMinusScalar: second argument must be rank 0, but " +
            ("got rank " + c.rank + "."));
        return this.sub(a, c);
    };
    NDArrayMath.prototype.neg = function (a) {
        var _this = this;
        return this.executeOp('neg', function () { return _this.negInternal(a); });
    };
    NDArrayMath.prototype.add = function (a, b) {
        var _this = this;
        util.assertAndGetBroadcastedShape(a.shape, b.shape);
        return this.executeOp('add', function () { return _this.addInternal(a, b); });
    };
    NDArrayMath.prototype.addStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in addStrict: ');
        return this.add(a, b);
    };
    NDArrayMath.prototype.sub = function (a, b) {
        var _this = this;
        util.assertAndGetBroadcastedShape(a.shape, b.shape);
        return this.executeOp('sub', function () { return _this.subInternal(a, b); });
    };
    NDArrayMath.prototype.subStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in subStrict: ');
        return this.sub(a, b);
    };
    NDArrayMath.prototype.multiply = function (a, b) {
        var _this = this;
        util.assertAndGetBroadcastedShape(a.shape, b.shape);
        return this.executeOp('multiply', function () { return _this.multiplyInternal(a, b); });
    };
    NDArrayMath.prototype.elementWiseMul = function (a, b) {
        return this.multiplyStrict(a, b);
    };
    NDArrayMath.prototype.multiplyStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in multiplyStrict: ');
        return this.multiply(a, b);
    };
    NDArrayMath.prototype.divide = function (a, b) {
        var _this = this;
        util.assertAndGetBroadcastedShape(a.shape, b.shape);
        return this.executeOp('divide', function () { return _this.divideInternal(a, b); });
    };
    NDArrayMath.prototype.divideStrict = function (a, b) {
        util.assertShapesMatch(a.shape, b.shape, 'Error in divideStrict: ');
        return this.divide(a, b);
    };
    NDArrayMath.prototype.scalarDividedByArray = function (c, a) {
        util.assert(c.size === 1, "Error in scalarDividedByArray: first argument must be rank 0, but " +
            ("got NDArray of rank " + c.rank + "."));
        return this.divide(c, a);
    };
    NDArrayMath.prototype.arrayDividedByScalar = function (a, c) {
        util.assert(c.size === 1, "Error in arrayDividedByScalar: second argument must be rank 0, " +
            ("but got NDArray of rank " + c.rank + "."));
        return this.divide(a, c);
    };
    NDArrayMath.prototype.exp = function (ndarray) {
        var _this = this;
        return this.executeOp('exp', function () { return _this.expInternal(ndarray); });
    };
    NDArrayMath.prototype.log = function (ndarray) {
        var _this = this;
        return this.executeOp('log', function () { return _this.logInternal(ndarray); });
    };
    NDArrayMath.prototype.sqrt = function (ndarray) {
        var _this = this;
        return this.executeOp('sqrt', function () { return _this.sqrtInternal(ndarray); });
    };
    NDArrayMath.prototype.relu = function (ndarray) {
        var _this = this;
        return this.executeOp('relu', function () { return _this.reluInternal(ndarray); });
    };
    NDArrayMath.prototype.sigmoid = function (ndarray) {
        var _this = this;
        return this.executeOp('sigmoid', function () { return _this.sigmoidInternal(ndarray); });
    };
    NDArrayMath.prototype.tanh = function (ndarray) {
        var _this = this;
        return this.executeOp('tanh', function () { return _this.tanhInternal(ndarray); });
    };
    NDArrayMath.prototype.sin = function (ndarray) {
        var _this = this;
        return this.executeOp('sin', function () { return _this.sinInternal(ndarray); });
    };
    NDArrayMath.prototype.step = function (ndarray) {
        var _this = this;
        return this.executeOp('step', function () { return _this.stepInternal(ndarray); });
    };
    NDArrayMath.prototype.scaledArrayAdd = function (c1, a, c2, b) {
        var _this = this;
        util.assert(c1.size === 1, "Error in scaledArrayAdd: first argument must rank 0, but got " +
            (" rank " + c1.rank + "."));
        util.assert(c2.size === 1, "Error in scaledArrayAdd: third argument must be rank 0, but got " +
            ("NDArray of rank " + c2.rank + "."));
        util.assertShapesMatch(a.shape, b.shape, 'Error in scaledArrayAdd: ');
        return this.executeOp('scaledArrayAdd', function () { return _this.scaledArrayAddInternal(c1, a, c2, b); });
    };
    NDArrayMath.prototype.scalarTimesArray = function (c, a) {
        util.assert(c.size === 1, "Error in arrayDividedByScalar: first argument must be rank 0, but " +
            ("got rank " + c.rank + "."));
        return this.multiply(c, a);
    };
    NDArrayMath.prototype.elementWiseMulBroadcast = function (a, b) {
        util.assert(a.rank === 2, "Error in elementWiseMulBroadcast: first argument must be " +
            ("rank 2, but got rank " + a.rank + "."));
        util.assert(b.rank === 2, "Error in elementWiseMulBroadcast: second argument must be " +
            ("rank 2, but got rank " + b.rank + "."));
        return this.multiply(a, b);
    };
    NDArrayMath.prototype.conv2d = function (x, filter, bias, strides, pad) {
        var _this = this;
        util.assert(x.rank === 3, "Error in conv2d: x must be rank 3, but got rank " + x.rank + ".");
        util.assert(filter.rank === 4, "Error in conv2d: filter must be rank 4, but got rank " +
            (filter.rank + "."));
        if (bias != null) {
            util.assert(bias.rank === 1, "Error in conv2d: bias must be rank 1, but got rank " +
                (bias.rank + "."));
        }
        util.assert(x.shape[2] === filter.shape[2], "Error in conv2d: depth of input (" + x.shape[2] + ") must match  " +
            ("input depth for filter " + filter.shape[2] + "."));
        var filterHeight = filter.shape[0];
        var filterWidth = filter.shape[1];
        var outDepth = filter.shape[3];
        var _a = parseTupleParam(strides), strideHeight = _a[0], strideWidth = _a[1];
        var convInfo = conv_util.computeConvInfo(x.shape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, pad);
        return this.executeOp('conv2d', function () { return _this.conv2dInternal(x, filter, bias, convInfo); });
    };
    NDArrayMath.prototype.conv2dBackProp = function (x, dy, filter, strides, pad) {
        var dw = this.conv2dDerFilter(x, dy, filter.shape, strides, pad);
        var db = this.conv2dDerBias(dy);
        var dx = this.conv2dDerInput(x.shape, dy, filter, strides, pad);
        return { db: db, dw: dw, dx: dx };
    };
    NDArrayMath.prototype.conv2dDerInput = function (inShape, dy, filter, strides, pad) {
        var _this = this;
        var inDepth = inShape[2];
        var outDepth = dy.shape[2];
        util.assert(inShape.length === 3, "Error in conv2dDerInput: x must be rank 3, but got rank " +
            (inShape.length + "."));
        util.assert(dy.rank === 3, "Error in conv2dDerInput: dy must be rank 3, but got " +
            ("rank " + dy.rank));
        util.assert(filter.rank === 4, "Error in conv2dDerInput: filter must be rank 4, but got " +
            ("rank " + filter.rank));
        util.assert(inDepth === filter.shape[2], "Error in conv2dDerInput: depth of input (" + inDepth + ") must " +
            ("match input depth for filter " + filter.shape[2] + "."));
        util.assert(outDepth === filter.shape[3], "Error in conv2dDerInput: depth of output (" + outDepth + ") must" +
            ("match output depth for filter " + filter.shape[3] + "."));
        var filterHeight = filter.shape[0];
        var filterWidth = filter.shape[1];
        var _a = parseTupleParam(strides), strideHeight = _a[0], strideWidth = _a[1];
        var convInfo = conv_util.computeConvInfo(inShape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, pad);
        return this.executeOp('conv2dDerInput', function () { return _this.conv2dDerInputInternal(dy, filter, convInfo); });
    };
    NDArrayMath.prototype.conv2dDerBias = function (dy) {
        return this.track(this.conv2dDerBiasInternal(dy));
    };
    NDArrayMath.prototype.conv2dDerFilter = function (x, dy, filterSize, strides, pad) {
        util.assert(x.rank === 3, "Error in conv2dDerFilter: x must be rank 3, but got shape " +
            (x.shape + "."));
        util.assert(dy.rank === 3, "Error in conv2dDerFilter: dy must be rank 3, but got shape " +
            (dy.shape + "."));
        util.assert(filterSize.length === 4, "Error in conv2dDerFilter: filterSize must be length 4, but got " +
            (filterSize + "."));
        util.assert(x.shape[2] === filterSize[2], "Error in conv2dDerFilter: depth of x " + x.shape[2] + ") must " +
            ("match input depth in filter (" + filterSize[2] + "."));
        util.assert(dy.shape[2] === filterSize[3], "Error in conv2dDerFilter: depth of dy (" + dy.shape[2] + ") must " +
            ("match output depth for filter (" + filterSize[3] + ")."));
        var filterHeight = filterSize[0];
        var filterWidth = filterSize[1];
        var outDepth = filterSize[3];
        var _a = parseTupleParam(strides), strideHeight = _a[0], strideWidth = _a[1];
        var convInfo = conv_util.computeConvInfo(x.shape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, pad);
        return this.track(this.conv2dDerFilterInternal(x, dy, convInfo));
    };
    NDArrayMath.prototype.conv2dTranspose = function (x, filter, outputShape, strides, pad) {
        return this.conv2dDerInput(outputShape, x, filter, strides, pad);
    };
    NDArrayMath.prototype.maxPool = function (x, filterSize, strides, pad) {
        var _this = this;
        util.assert(x.rank === 3, 'Error in maxPool: x must be rank 3 but got rank ' + x.rank + '.');
        var _a = parseTupleParam(filterSize), filterHeight = _a[0], filterWidth = _a[1];
        var outDepth = x.shape[2];
        var _b = parseTupleParam(strides), strideHeight = _b[0], strideWidth = _b[1];
        var convInfo = conv_util.computeConvInfo(x.shape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, pad);
        return this.executeOp('maxPool', function () { return _this.maxPoolInternal(x, convInfo); });
    };
    NDArrayMath.prototype.maxPoolBackprop = function (dy, x, filterSize, strides, pad) {
        var _this = this;
        util.assert(dy.rank === 3, "Error in maxPoolBackprop: dy must be rank 3 but got rank " +
            (dy.rank + "."));
        util.assert(x.rank === 3, "Error in maxPoolBackprop: x must be rank 3 but got rank " +
            (x.rank + "."));
        var _a = parseTupleParam(filterSize), filterHeight = _a[0], filterWidth = _a[1];
        var outDepth = x.shape[2];
        var _b = parseTupleParam(strides), strideHeight = _b[0], strideWidth = _b[1];
        var convInfo = conv_util.computeConvInfo(x.shape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, pad);
        return this.executeOp('maxPoolBackprop', function () { return _this.maxPoolBackpropInternal(dy, x, convInfo); });
    };
    NDArrayMath.prototype.minPool = function (x, filterSize, strides, pad) {
        var _this = this;
        util.assert(x.rank === 3, "Error in minPool: x must be rank 3 but got rank " + x.rank + ".");
        var _a = parseTupleParam(filterSize), filterHeight = _a[0], filterWidth = _a[1];
        var outDepth = x.shape[2];
        var _b = parseTupleParam(strides), strideHeight = _b[0], strideWidth = _b[1];
        var convInfo = conv_util.computeConvInfo(x.shape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, pad);
        return this.executeOp('minPool', function () { return _this.minPoolInternal(x, convInfo); });
    };
    NDArrayMath.prototype.avgPool = function (x, filterSize, strides, pad) {
        var _this = this;
        util.assert(x.rank === 3, "Error in avgPool: x must be rank 3 but got rank " + x.rank + ".");
        var _a = parseTupleParam(filterSize), filterHeight = _a[0], filterWidth = _a[1];
        var outDepth = x.shape[2];
        var _b = parseTupleParam(strides), strideHeight = _b[0], strideWidth = _b[1];
        var convInfo = conv_util.computeConvInfo(x.shape, filterHeight, filterWidth, outDepth, strideHeight, strideWidth, pad);
        return this.executeOp('avgPool', function () { return _this.avgPoolInternal(x, convInfo); });
    };
    NDArrayMath.prototype.resizeBilinear3D = function (x, newShape2D, alignCorners) {
        var _this = this;
        if (alignCorners === void 0) { alignCorners = false; }
        util.assert(x.rank === 3, "Error in resizeBilinear3D: x must be rank 3 but got rank " + x.rank + ".");
        util.assert(newShape2D.length === 2, "Error in resizeBilinear3D: new shape must 2D, but got shape " +
            (newShape2D + "."));
        return this.executeOp('resizeBilinear3D', function () { return _this.resizeBilinear3DInternal(x, newShape2D, alignCorners); });
    };
    NDArrayMath.prototype.batchNormalization3D = function (x, mean, variance, varianceEpsilon, scale, offset) {
        var _this = this;
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        util.assert(x.rank === 3, "Error in batchNormalization3D: x must be rank 3 but got rank " +
            (x.rank + "."));
        util.assert(mean.rank === 3 || mean.rank === 1, "Error in batchNormalization3D: mean must be rank 3 or rank 1 but " +
            ("got rank " + mean.rank + "."));
        util.assert(variance.rank === 3 || variance.rank === 1, "Error in batchNormalization3D: variance must be rank 3 or rank 1 " +
            ("but got rank " + variance.rank + "."));
        if (scale != null) {
            util.assert(scale.rank === 3 || scale.rank === 1, "Error in batchNormalization3D: scale must be rank 3 or rank 1 " +
                ("but got rank " + scale.rank + "."));
        }
        if (offset != null) {
            util.assert(offset.rank === 3 || offset.rank === 1, "Error in batchNormalization3D: offset must be rank 3 or rank 1 " +
                ("but got rank " + offset.rank + "."));
        }
        return this.executeOp('batchNorm3D', function () { return _this.batchNormalization3DInternal(x, mean, variance, varianceEpsilon, scale, offset); });
    };
    NDArrayMath.prototype.multiRNNCell = function (lstmCells, data, c, h) {
        util.assert(data.shape[0] === 1, "Error in multiRNNCell: first dimension of data is " + data.shape[0] + ", " +
            "but batch sizes > 1 are not yet supported.");
        var res = this.scope(function () {
            var input = data;
            var newStates = [];
            for (var i = 0; i < lstmCells.length; i++) {
                var output = lstmCells[i](input, c[i], h[i]);
                newStates.push(output[0]);
                newStates.push(output[1]);
                input = output[1];
            }
            return newStates;
        });
        var newC = [];
        var newH = [];
        for (var i = 0; i < res.length; i += 2) {
            newC.push(res[i]);
            newH.push(res[i + 1]);
        }
        return [newC, newH];
    };
    NDArrayMath.prototype.basicLSTMCell = function (forgetBias, lstmKernel, lstmBias, data, c, h) {
        var _this = this;
        var res = this.scope(function () {
            util.assert(data.shape[0] === 1, "Error in multiRNNCell: first dimension of data is " +
                (data.shape[0] + ", but batch sizes > 1 are not yet supported."));
            var data3D = data.as3D(1, 1, data.shape[1]);
            var h3D = h.as3D(1, 1, h.shape[1]);
            var combined3D = _this.concat3D(data3D, h3D, 2);
            var combined2D = combined3D.as2D(1, data.shape[1] + h.shape[1]);
            var weighted = _this.matMul(combined2D, lstmKernel);
            var res = _this.add(weighted, lstmBias);
            var i = _this.slice2D(res, [0, 0], [res.shape[0], res.shape[1] / 4]);
            var j = _this.slice2D(res, [0, res.shape[1] / 4 * 1], [res.shape[0], res.shape[1] / 4]);
            var f = _this.slice2D(res, [0, res.shape[1] / 4 * 2], [res.shape[0], res.shape[1] / 4]);
            var o = _this.slice2D(res, [0, res.shape[1] / 4 * 3], [res.shape[0], res.shape[1] / 4]);
            var newC = _this.add(_this.multiplyStrict(c, _this.sigmoid(_this.scalarPlusArray(forgetBias, f))), _this.multiplyStrict(_this.sigmoid(i), _this.tanh(j)));
            var newH = _this.multiplyStrict(_this.tanh(newC), _this.sigmoid(o));
            return [newC, newH];
        });
        return [res[0], res[1]];
    };
    return NDArrayMath;
}());
exports.NDArrayMath = NDArrayMath;
var MatrixOrientation;
(function (MatrixOrientation) {
    MatrixOrientation[MatrixOrientation["REGULAR"] = 0] = "REGULAR";
    MatrixOrientation[MatrixOrientation["TRANSPOSED"] = 1] = "TRANSPOSED";
})(MatrixOrientation = exports.MatrixOrientation || (exports.MatrixOrientation = {}));
function parseTupleParam(param) {
    return typeof param === 'number' ? [param, param] : param;
}

},{"../util":79,"./concat3d_util":19,"./conv_util":20,"./copy2d_util":21,"./ndarray":26}],24:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
var concat3d_util = require("./concat3d_util");
var conv_util = require("./conv_util");
var copy2D_util = require("./copy2d_util");
var math_1 = require("./math");
var ndarray_1 = require("./ndarray");
var NDArrayMathCPU = (function (_super) {
    __extends(NDArrayMathCPU, _super);
    function NDArrayMathCPU(safeMode) {
        if (safeMode === void 0) { safeMode = false; }
        return _super.call(this, safeMode) || this;
    }
    NDArrayMathCPU.prototype.cloneInternal = function (ndarray) {
        return ndarray_1.NDArray.make(ndarray.shape, { values: new Float32Array(ndarray.getValues()) });
    };
    NDArrayMathCPU.prototype.slice2DInternal = function (input, beginRowCol, sizeRowCol) {
        var result = ndarray_1.Array2D.zeros(sizeRowCol);
        this.copy2DInternal(input, beginRowCol, sizeRowCol, result, [0, 0], sizeRowCol);
        return result;
    };
    NDArrayMathCPU.prototype.copy2DInternal = function (source, sourceBeginRowCol, sourceSizeRowCol, dest, destBeginRowCol, destSizeRowCol) {
        copy2D_util.validateShapes(sourceSizeRowCol, destSizeRowCol);
        var srcValues = source.getValues();
        var dstValues = dest.getValues();
        var n = sourceSizeRowCol[0] * sourceSizeRowCol[1];
        for (var i = 0; i < n; ++i) {
            var srcRow = sourceBeginRowCol[0] + Math.floor(i / sourceSizeRowCol[1]);
            var srcCol = sourceBeginRowCol[1] + (i % sourceSizeRowCol[1]);
            var srcOff = srcRow * source.shape[1] + srcCol;
            var dstRow = destBeginRowCol[0] + Math.floor(i / destSizeRowCol[1]);
            var dstCol = destBeginRowCol[1] + (i % destSizeRowCol[1]);
            var dstOff = dstRow * dest.shape[1] + dstCol;
            dstValues[dstOff] = srcValues[srcOff];
        }
    };
    NDArrayMathCPU.prototype.concat3DInternal = function (x1, x2, axis) {
        var outputShape = concat3d_util.computeConcat3DOutputShape(x1.shape, x2.shape, axis);
        var values = ndarray_1.Array3D.zeros(outputShape);
        for (var i = 0; i < outputShape[0]; i++) {
            for (var j = 0; j < outputShape[1]; j++) {
                for (var k = 0; k < outputShape[2]; k++) {
                    var index = [i, j, k];
                    var value = void 0;
                    if (index[axis] < x1.shape[axis]) {
                        value = x1.get(i, j, k);
                    }
                    else {
                        index[axis] -= x1.shape[axis];
                        var i2 = index[0], j2 = index[1], k2 = index[2];
                        value = x2.get(i2, j2, k2);
                    }
                    values.set(value, i, j, k);
                }
            }
        }
        return values;
    };
    NDArrayMathCPU.prototype.scaledArrayAddInternal = function (c1, a, c2, b) {
        var newShape = util.assertAndGetBroadcastedShape(a.shape, b.shape);
        var newValues = new Float32Array(util.sizeFromShape(newShape));
        var aValues = a.getValues();
        var bValues = b.getValues();
        var c1Val = c1.get();
        var c2Val = c2.get();
        for (var i = 0; i < newValues.length; ++i) {
            newValues[i] = c1Val * aValues[i % a.size] + c2Val * bValues[i % b.size];
        }
        return ndarray_1.NDArray.make(newShape, { values: newValues });
    };
    NDArrayMathCPU.prototype.negInternal = function (a) {
        return this.scalarTimesArray(ndarray_1.Scalar.NEG_ONE, a);
    };
    NDArrayMathCPU.prototype.addInternal = function (a, b) {
        return this.scaledArrayAddInternal(ndarray_1.Scalar.ONE, a, ndarray_1.Scalar.ONE, b);
    };
    NDArrayMathCPU.prototype.subInternal = function (a, b) {
        return this.scaledArrayAddInternal(ndarray_1.Scalar.ONE, a, ndarray_1.Scalar.NEG_ONE, b);
    };
    NDArrayMathCPU.prototype.matMulInternal = function (a, b, aOrientation, bOrientation) {
        if (aOrientation === void 0) { aOrientation = math_1.MatrixOrientation.REGULAR; }
        if (bOrientation === void 0) { bOrientation = math_1.MatrixOrientation.REGULAR; }
        var sharedDim = (aOrientation === math_1.MatrixOrientation.REGULAR) ? a.shape[1] : a.shape[0];
        var leftDim = (aOrientation === math_1.MatrixOrientation.REGULAR) ? a.shape[0] : a.shape[1];
        var rightDim = (bOrientation === math_1.MatrixOrientation.REGULAR) ? b.shape[1] : b.shape[0];
        var normalGetter = function (matrix, i, j) {
            return matrix.get(i, j);
        };
        var transposedGetter = function (matrix, i, j) {
            return matrix.get(j, i);
        };
        var aGetter = (aOrientation === math_1.MatrixOrientation.REGULAR) ?
            normalGetter :
            transposedGetter;
        var bGetter = (bOrientation === math_1.MatrixOrientation.REGULAR) ?
            normalGetter :
            transposedGetter;
        var values = new Float32Array(leftDim * rightDim);
        var index = 0;
        for (var i = 0; i < leftDim; ++i) {
            for (var j = 0; j < rightDim; ++j) {
                var sum = 0;
                for (var k = 0; k < sharedDim; ++k) {
                    sum += aGetter(a, i, k) * bGetter(b, k, j);
                }
                values[index++] = sum;
            }
        }
        return ndarray_1.Array2D.new([leftDim, rightDim], values);
    };
    NDArrayMathCPU.prototype.multiplyInternal = function (a, b) {
        var newShape = util.assertAndGetBroadcastedShape(a.shape, b.shape);
        var newValues = new Float32Array(util.sizeFromShape(newShape));
        var aValues = a.getValues();
        var bValues = b.getValues();
        for (var i = 0; i < newValues.length; ++i) {
            newValues[i] = aValues[i % a.size] * bValues[i % b.size];
        }
        return ndarray_1.NDArray.make(newShape, { values: newValues });
    };
    NDArrayMathCPU.prototype.divideInternal = function (a, b) {
        var newShape = util.assertAndGetBroadcastedShape(a.shape, b.shape);
        var newValues = new Float32Array(util.sizeFromShape(newShape));
        var aValues = a.getValues();
        var bValues = b.getValues();
        for (var i = 0; i < newValues.length; ++i) {
            newValues[i] = aValues[i % a.size] / bValues[i % b.size];
        }
        return ndarray_1.NDArray.make(newShape, { values: newValues });
    };
    NDArrayMathCPU.prototype.sumInternal = function (ndarray) {
        var sum = 0;
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            sum += values[i];
        }
        return ndarray_1.Scalar.new(sum);
    };
    NDArrayMathCPU.prototype.argMinInternal = function (ndarray) {
        var min = Number.MAX_VALUE;
        var minIndex = -1;
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value < min) {
                min = value;
                minIndex = i;
            }
        }
        return ndarray_1.Scalar.new(minIndex);
    };
    NDArrayMathCPU.prototype.argMaxInternal = function (ndarray) {
        var max = Number.NEGATIVE_INFINITY;
        var maxIndex = -1;
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value > max) {
                max = value;
                maxIndex = i;
            }
        }
        return ndarray_1.Scalar.new(maxIndex);
    };
    NDArrayMathCPU.prototype.argMaxEqualsInternal = function (x1, x2) {
        var argMax1 = this.argMaxInternal(x1).get();
        var argMax2 = this.argMaxInternal(x2).get();
        if (isNaN(argMax1) || isNaN(argMax2)) {
            return ndarray_1.Scalar.new(NaN);
        }
        return ndarray_1.Scalar.new(+(argMax1 === argMax2));
    };
    NDArrayMathCPU.prototype.topKInternal = function (ndarray, k) {
        var values = ndarray.getValues();
        var valuesAndIndices = [];
        for (var i = 0; i < values.length; i++) {
            valuesAndIndices.push({ value: values[i], index: i });
        }
        valuesAndIndices.sort(function (a, b) {
            return b.value - a.value;
        });
        var topkValues = new Float32Array(k);
        var topkIndices = new Float32Array(k);
        for (var i = 0; i < k; i++) {
            topkValues[i] = valuesAndIndices[i].value;
            topkIndices[i] = valuesAndIndices[i].index;
        }
        return { values: ndarray_1.Array1D.new(topkValues), indices: ndarray_1.Array1D.new(topkIndices) };
    };
    NDArrayMathCPU.prototype.minInternal = function (ndarray) {
        var values = ndarray.getValues();
        var min = values[0];
        for (var i = 1; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value < min) {
                min = value;
            }
        }
        return ndarray_1.Scalar.new(min);
    };
    NDArrayMathCPU.prototype.maxInternal = function (ndarray) {
        var values = ndarray.getValues();
        var max = values[0];
        for (var i = 1; i < values.length; ++i) {
            var value = values[i];
            if (isNaN(value)) {
                return ndarray_1.Scalar.new(NaN);
            }
            if (value > max) {
                max = value;
            }
        }
        return ndarray_1.Scalar.new(max);
    };
    NDArrayMathCPU.prototype.expInternal = function (ndarray) {
        var values = ndarray.getValues();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            newValues[i] = Math.exp(values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.logInternal = function (ndarray) {
        var values = ndarray.getValues();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            newValues[i] = Math.log(value);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.sqrtInternal = function (ndarray) {
        var values = ndarray.getValues();
        var newValues = new Float32Array(values.length);
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            newValues[i] = Math.sqrt(value);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: newValues });
    };
    NDArrayMathCPU.prototype.logSumExpInternal = function (ndarray) {
        var xMax = this.max(ndarray);
        var a = this.arrayMinusScalar(ndarray, xMax);
        var b = this.exp(a);
        var c = this.sum(b);
        var d = this.log(c);
        var result = this.add(xMax, d);
        xMax.dispose();
        a.dispose();
        b.dispose();
        c.dispose();
        d.dispose();
        return result;
    };
    NDArrayMathCPU.prototype.reluInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.max(0, values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.sigmoidInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = 1 / (1 + Math.exp(-values[i]));
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.tanhInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = util.tanh(values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.sinInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            resultValues[i] = Math.sin(values[i]);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.stepInternal = function (ndarray) {
        var resultValues = new Float32Array(ndarray.size);
        var values = ndarray.getValues();
        for (var i = 0; i < values.length; ++i) {
            var value = values[i];
            resultValues[i] = value > 0 ? 1 : (value < 0 ? 0 : value);
        }
        return ndarray_1.NDArray.make(ndarray.shape, { values: resultValues });
    };
    NDArrayMathCPU.prototype.conv2dInternal = function (x, filter, bias, convInfo) {
        var _a = x.shape, xRows = _a[0], xCols = _a[1], inputDepth = _a[2];
        var filterHeight = filter.shape[0];
        var filterWidth = filter.shape[1];
        var outDepth = filter.shape[3];
        var padLeft = convInfo.padInfo.left;
        var padTop = convInfo.padInfo.top;
        var y = ndarray_1.Array3D.zeros(convInfo.outShape);
        for (var d2 = 0; d2 < outDepth; ++d2) {
            for (var yR = 0; yR < y.shape[0]; ++yR) {
                var xRCorner = yR * convInfo.strideHeight - padLeft;
                var xRMin = Math.max(0, xRCorner);
                var xRMax = Math.min(xRows, filterHeight + xRCorner);
                for (var yC = 0; yC < y.shape[1]; ++yC) {
                    var xCCorner = yC * convInfo.strideWidth - padTop;
                    var xCMin = Math.max(0, xCCorner);
                    var xCMax = Math.min(xCols, filterWidth + xCCorner);
                    var dotProd = 0;
                    for (var xR = xRMin; xR < xRMax; ++xR) {
                        var wR = xR - xRCorner;
                        for (var xC = xCMin; xC < xCMax; ++xC) {
                            var wC = xC - xCCorner;
                            for (var d1 = 0; d1 < inputDepth; ++d1) {
                                var pixel = x.get(xR, xC, d1);
                                var weight = filter.get(wR, wC, d1, d2);
                                dotProd += pixel * weight;
                            }
                        }
                    }
                    var biasVal = (bias != null) ? bias.get(d2) : 0;
                    y.set(dotProd + biasVal, yR, yC, d2);
                }
            }
        }
        return y;
    };
    NDArrayMathCPU.prototype.conv2dDerInputInternal = function (dy, filter, convInfo) {
        var inDepth = filter.shape[2];
        var outDepth = filter.shape[3];
        var yRows = dy.shape[0];
        var yCols = dy.shape[1];
        var filterHeight = filter.shape[0];
        var filterWidth = filter.shape[1];
        var topPad = filterHeight - 1 - convInfo.padInfo.top;
        var leftPad = filterWidth - 1 - convInfo.padInfo.left;
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var dx = ndarray_1.Array3D.zeros(convInfo.inShape);
        for (var d1 = 0; d1 < inDepth; ++d1) {
            for (var xR = 0; xR < dx.shape[0]; ++xR) {
                var xRCorner = xR - leftPad;
                var xRMin = Math.max(0, Math.ceil(xRCorner / strideHeight));
                var yRMax = Math.min(yRows, (filterHeight + xRCorner) / strideHeight);
                for (var xC = 0; xC < dx.shape[1]; ++xC) {
                    var xCCorner = xC - topPad;
                    var xCMin = Math.max(0, Math.ceil(xCCorner / strideWidth));
                    var yCMax = Math.min(yCols, (filterWidth + xCCorner) / strideWidth);
                    var dotProd = 0;
                    for (var yR = xRMin; yR < yRMax; ++yR) {
                        var wR = yR * strideHeight - xRCorner;
                        for (var yC = xCMin; yC < yCMax; ++yC) {
                            var wC = yC * strideWidth - xCCorner;
                            for (var d2 = 0; d2 < outDepth; ++d2) {
                                var pixel = dy.get(yR, yC, d2);
                                var weight = filter.get(filterHeight - 1 - wR, filterWidth - 1 - wC, d1, d2);
                                dotProd += pixel * weight;
                            }
                        }
                    }
                    dx.set(dotProd, xR, xC, d1);
                }
            }
        }
        return dx;
    };
    NDArrayMathCPU.prototype.conv2dDerFilterInternal = function (x, dY, convInfo) {
        var inputDepth = x.shape[2];
        var outputDepth = dY.shape[2];
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var weightsShape = conv_util.computeWeightsShape4D(inputDepth, outputDepth, filterHeight, filterWidth);
        var dW = ndarray_1.Array4D.zeros(weightsShape);
        var yNumRows = dY.shape[0];
        var yNumCols = dY.shape[1];
        var xNumRows = x.shape[0];
        var xNumCols = x.shape[1];
        var leftPad = convInfo.padInfo.left;
        var topPad = convInfo.padInfo.top;
        for (var wR = 0; wR < filterHeight; ++wR) {
            var yRMin = Math.max(0, Math.ceil((topPad - wR) / strideHeight));
            var yRMax = Math.min(yNumRows, (xNumRows + topPad - wR) / strideHeight);
            for (var wC = 0; wC < filterWidth; ++wC) {
                var yCMin = Math.max(0, Math.ceil((leftPad - wC) / strideWidth));
                var yCMax = Math.min(yNumCols, (xNumCols + leftPad - wC) / strideWidth);
                for (var d1 = 0; d1 < inputDepth; ++d1) {
                    for (var d2 = 0; d2 < outputDepth; ++d2) {
                        var dotProd = 0;
                        for (var yR = yRMin; yR < yRMax; ++yR) {
                            var xR = wR + yR * strideHeight - topPad;
                            for (var yC = yCMin; yC < yCMax; ++yC) {
                                var xC = wC + yC * strideWidth - leftPad;
                                dotProd += x.get(xR, xC, d1) * dY.get(yR, yC, d2);
                            }
                        }
                        dW.set(dotProd, wR, wC, d1, d2);
                    }
                }
            }
        }
        return dW;
    };
    NDArrayMathCPU.prototype.conv2dDerBiasInternal = function (dY) {
        var outputDepth = dY.shape[2];
        var numRows = dY.shape[0];
        var numCols = dY.shape[1];
        var values = new Float32Array(outputDepth);
        for (var d2 = 0; d2 < outputDepth; ++d2) {
            var sum = 0;
            for (var r = 0; r < numRows; ++r) {
                for (var c = 0; c < numCols; ++c) {
                    sum += dY.get(r, c, d2);
                }
            }
            values[d2] = sum;
        }
        return ndarray_1.Array1D.new(values);
    };
    NDArrayMathCPU.prototype.switchDimInternal = function (t, newDim) {
        var newShape = new Array(t.rank);
        for (var i = 0; i < newShape.length; i++) {
            newShape[i] = t.shape[newDim[i]];
        }
        var resultValues = new Float32Array(t.size);
        var values = t.getValues();
        var result = ndarray_1.NDArray.make(newShape, { values: resultValues });
        for (var i = 0; i < t.size; ++i) {
            var loc = t.indexToLoc(i);
            var newLoc = new Array(loc.length);
            for (var i_1 = 0; i_1 < newLoc.length; i_1++) {
                newLoc[i_1] = loc[newDim[i_1]];
            }
            var newIndex = result.locToIndex(newLoc);
            resultValues[newIndex] = values[i];
        }
        return result;
    };
    NDArrayMathCPU.prototype.pool = function (x, convInfo, poolType) {
        var _a = x.shape, xRows = _a[0], xCols = _a[1], depth = _a[2];
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var y = ndarray_1.Array3D.zeros(convInfo.outShape);
        var padTop = convInfo.padInfo.top;
        var padLeft = convInfo.padInfo.left;
        for (var d = 0; d < depth; ++d) {
            for (var yR = 0; yR < y.shape[0]; ++yR) {
                var xRCorner = yR * strideHeight - padTop;
                var xRMin = Math.max(0, xRCorner);
                var xRMax = Math.min(xRows, filterHeight + xRCorner);
                for (var yC = 0; yC < y.shape[1]; ++yC) {
                    var xCCorner = yC * strideWidth - padLeft;
                    var xCMin = Math.max(0, xCCorner);
                    var xCMax = Math.min(xCols, filterWidth + xCCorner);
                    var minMaxValue = (poolType === 'max' ? Number.NEGATIVE_INFINITY :
                        Number.POSITIVE_INFINITY);
                    var avgValue = 0;
                    for (var xR = xRMin; xR < xRMax; ++xR) {
                        for (var xC = xCMin; xC < xCMax; ++xC) {
                            var pixel = x.get(xR, xC, d);
                            if (isNaN(pixel)) {
                                minMaxValue = NaN;
                                avgValue = NaN;
                                break;
                            }
                            if ((poolType === 'max' && pixel > minMaxValue) ||
                                (poolType === 'min' && pixel < minMaxValue)) {
                                minMaxValue = pixel;
                            }
                            else if (poolType === 'avg') {
                                avgValue += pixel / (filterHeight * filterWidth);
                            }
                        }
                        if (isNaN(minMaxValue)) {
                            break;
                        }
                    }
                    y.set(poolType === 'avg' ? avgValue : minMaxValue, yR, yC, d);
                }
            }
        }
        return y;
    };
    NDArrayMathCPU.prototype.maxPoolInternal = function (x, convInfo) {
        return this.pool(x, convInfo, 'max');
    };
    NDArrayMathCPU.prototype.maxPoolPositions = function (x, convInfo) {
        var _a = x.shape, xRows = _a[0], xCols = _a[1], depth = _a[2];
        var outputShape = convInfo.outShape;
        var maxPositions = ndarray_1.Array3D.zeros(outputShape);
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var padTop = convInfo.padInfo.top;
        var padLeft = convInfo.padInfo.left;
        for (var d = 0; d < depth; ++d) {
            for (var yR = 0; yR < outputShape[0]; ++yR) {
                var xRCorner = yR * strideHeight - padTop;
                var xRMin = Math.max(0, xRCorner);
                var xRMax = Math.min(xRows, filterHeight + xRCorner);
                for (var yC = 0; yC < outputShape[1]; ++yC) {
                    var xCCorner = yC * strideWidth - padLeft;
                    var xCMin = Math.max(0, xCCorner);
                    var xCMax = Math.min(xCols, filterWidth + xCCorner);
                    var maxValue = Number.NEGATIVE_INFINITY;
                    var maxPosition = -1;
                    for (var xR = xRMin; xR < xRMax; ++xR) {
                        var wR = xR - xRCorner;
                        for (var xC = xCMin; xC < xCMax; ++xC) {
                            var wC = xC - xCCorner;
                            var pixel = x.get(xR, xC, d);
                            if (pixel > maxValue) {
                                maxValue = pixel;
                                maxPosition = wR * filterWidth + wC;
                            }
                        }
                    }
                    maxPositions.set(maxPosition, yR, yC, d);
                }
            }
        }
        return maxPositions;
    };
    NDArrayMathCPU.prototype.maxPoolBackpropInternal = function (dy, x, convInfo) {
        var maxPositions = this.maxPoolPositions(x, convInfo);
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var padLeft = filterWidth - 1 - convInfo.padInfo.left;
        var padTop = filterHeight - 1 - convInfo.padInfo.top;
        var _a = dy.shape, dyRows = _a[0], dyCols = _a[1], depth = _a[2];
        var dx = ndarray_1.Array3D.zeros(x.shape);
        for (var d = 0; d < depth; ++d) {
            for (var dxR = 0; dxR < dx.shape[0]; ++dxR) {
                for (var dxC = 0; dxC < dx.shape[1]; ++dxC) {
                    var dyRCorner = dxR - padTop;
                    var dyCCorner = dxC - padLeft;
                    var dotProd = 0;
                    for (var wR = 0; wR < filterHeight; ++wR) {
                        var dyR = (dyRCorner + wR) / strideHeight;
                        if (dyR < 0 || dyR >= dyRows || Math.floor(dyR) !== dyR) {
                            continue;
                        }
                        for (var wC = 0; wC < filterWidth; ++wC) {
                            var dyC = (dyCCorner + wC) / strideWidth;
                            if (dyC < 0 || dyC >= dyCols || Math.floor(dyC) !== dyC) {
                                continue;
                            }
                            var maxPos = filterHeight * filterWidth - 1 -
                                maxPositions.get(dyR, dyC, d);
                            var curPos = wR * filterWidth + wC;
                            var mask = maxPos === curPos ? 1 : 0;
                            if (mask === 0) {
                                continue;
                            }
                            var pixel = dy.get(dyR, dyC, d);
                            dotProd += pixel * mask;
                        }
                    }
                    dx.set(dotProd, dxR, dxC, d);
                }
            }
        }
        return dx;
    };
    NDArrayMathCPU.prototype.minPoolInternal = function (x, convInfo) {
        return this.pool(x, convInfo, 'min');
    };
    NDArrayMathCPU.prototype.avgPoolInternal = function (x, convInfo) {
        return this.pool(x, convInfo, 'avg');
    };
    NDArrayMathCPU.prototype.resizeBilinear3DInternal = function (x, newShape2D, alignCorners) {
        var output = ndarray_1.Array3D.zeros([newShape2D[0], newShape2D[1], x.shape[2]]);
        var effectiveInputSize = alignCorners ? [x.shape[0] - 1, x.shape[1] - 1, x.shape[2]] : x.shape;
        var effectiveOutputSize = alignCorners ?
            [output.shape[0] - 1, output.shape[1] - 1, output.shape[2]] :
            output.shape;
        for (var r = 0; r < output.shape[0]; r++) {
            for (var c = 0; c < output.shape[1]; c++) {
                for (var d = 0; d < output.shape[2]; d++) {
                    var sourceFracRow = (effectiveInputSize[0]) * r / (effectiveOutputSize[0]);
                    var sourceFracCol = (effectiveInputSize[1]) * c / (effectiveOutputSize[1]);
                    var sourceRowFloor = Math.floor(sourceFracRow);
                    var sourceRowCeil = Math.min(x.shape[0] - 1, Math.ceil(sourceFracRow));
                    var sourceColFloor = Math.floor(sourceFracCol);
                    var sourceColCeil = Math.min(x.shape[1] - 1, Math.ceil(sourceFracCol));
                    var topLeft = x.get(sourceRowFloor, sourceColFloor, d);
                    var bottomLeft = x.get(sourceRowCeil, sourceColFloor, d);
                    var topRight = x.get(sourceRowFloor, sourceColCeil, d);
                    var bottomRight = x.get(sourceRowCeil, sourceColCeil, d);
                    var rowFrac = sourceFracRow - sourceRowFloor;
                    var colFrac = sourceFracCol - sourceColFloor;
                    var top_1 = topLeft + (topRight - topLeft) * colFrac;
                    var bottom = bottomLeft + (bottomRight - bottomLeft) * colFrac;
                    var newValue = top_1 + (bottom - top_1) * rowFrac;
                    output.set(newValue, r, c, d);
                }
            }
        }
        return output;
    };
    NDArrayMathCPU.prototype.batchNormalization3DInternal = function (x, mean, variance, varianceEpsilon, scale, offset) {
        if (varianceEpsilon === void 0) { varianceEpsilon = .001; }
        var xValues = x.getValues();
        var meanValues = mean.getValues();
        var varianceValues = variance.getValues();
        var scaleValues = scale ? scale.getValues() : new Float32Array([1]);
        var offsetValues = offset ? offset.getValues() : new Float32Array([0]);
        var outValues = new Float32Array(xValues.length);
        for (var i = 0; i < xValues.length; i++) {
            outValues[i] = offsetValues[i % offsetValues.length] +
                (xValues[i] - meanValues[i % meanValues.length]) *
                    scaleValues[i % scaleValues.length] /
                    Math.sqrt(varianceValues[i % varianceValues.length] + varianceEpsilon);
        }
        return ndarray_1.NDArray.make(x.shape, { values: outValues });
    };
    return NDArrayMathCPU;
}(math_1.NDArrayMath));
exports.NDArrayMathCPU = NDArrayMathCPU;

},{"../util":79,"./concat3d_util":19,"./conv_util":20,"./copy2d_util":21,"./math":23,"./ndarray":26}],25:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var math_1 = require("./math");
var ndarray = require("./ndarray");
var ndarray_1 = require("./ndarray");
var addscaledmat_gpu_1 = require("./webgl/addscaledmat_gpu");
var argmaxequals_gpu_1 = require("./webgl/argmaxequals_gpu");
var argminmax_gpu_1 = require("./webgl/argminmax_gpu");
var batchnorm_gpu_1 = require("./webgl/batchnorm_gpu");
var binaryop_gpu_1 = require("./webgl/binaryop_gpu");
var concat3d_gpu_1 = require("./webgl/concat3d_gpu");
var conv_backprop_gpu_1 = require("./webgl/conv_backprop_gpu");
var conv_gpu_1 = require("./webgl/conv_gpu");
var copy_gpu_1 = require("./webgl/copy_gpu");
var gpgpu_context_1 = require("./webgl/gpgpu_context");
var gpgpu_math = require("./webgl/gpgpu_math");
var gpgpu_util = require("./webgl/gpgpu_util");
var logsumexp_gpu_1 = require("./webgl/logsumexp_gpu");
var max_pool_backprop_gpu_1 = require("./webgl/max_pool_backprop_gpu");
var minmax_gpu_1 = require("./webgl/minmax_gpu");
var mulmat_gpu_1 = require("./webgl/mulmat_gpu");
var pool_gpu_1 = require("./webgl/pool_gpu");
var reducesum_gpu_1 = require("./webgl/reducesum_gpu");
var resize_bilinear_gpu_1 = require("./webgl/resize_bilinear_gpu");
var texture_manager_1 = require("./webgl/texture_manager");
var unaryop_gpu_1 = require("./webgl/unaryop_gpu");
var webgl_util = require("./webgl/webgl_util");
var NDArrayMathGPU = (function (_super) {
    __extends(NDArrayMathGPU, _super);
    function NDArrayMathGPU(gpgpu, safeMode) {
        if (safeMode === void 0) { safeMode = true; }
        var _this = _super.call(this, safeMode) || this;
        _this.binaryCache = {};
        if (gpgpu == null) {
            var gl = gpgpu_util.createWebGLContext();
            _this.gpgpu = new gpgpu_context_1.GPGPUContext(gl);
            _this.gpgpuCreatedLocally = true;
        }
        else {
            _this.gpgpu = gpgpu;
            _this.gpgpuCreatedLocally = false;
        }
        _this.textureManager = new texture_manager_1.TextureManager(_this.gpgpu);
        ndarray.initializeGPU(_this.gpgpu, _this.textureManager);
        return _this;
    }
    NDArrayMathGPU.prototype.getGPGPUContext = function () {
        return this.gpgpu;
    };
    NDArrayMathGPU.prototype.cloneInternal = function (ndarray) {
        var texShape = ndarray.getTextureShapeRC();
        var source = ndarray.as2D(texShape[0], texShape[1]);
        var output = this.makeOutputArray(texShape);
        this.copy2D(source, [0, 0], texShape, output, [0, 0], texShape);
        return output.reshape(ndarray.shape);
    };
    NDArrayMathGPU.prototype.slice2DInternal = function (input, beginRowCol, sizeRowCol) {
        var result = ndarray_1.NDArray.make(sizeRowCol, {
            texture: this.textureManager.acquireTexture(sizeRowCol),
            textureShapeRC: sizeRowCol
        });
        this.copy2DInternal(input, beginRowCol, sizeRowCol, result, [0, 0], sizeRowCol);
        return result;
    };
    NDArrayMathGPU.prototype.copy2DInternal = function (source, sourceBeginRowCol, sourceSizeRowCol, dest, destBeginRowCol, destSizeRowCol) {
        var program = new copy_gpu_1.Copy2DProgram(sourceSizeRowCol[1], destSizeRowCol[1]);
        var customSetup = program.getCustomSetupFunc(sourceBeginRowCol, destBeginRowCol, destSizeRowCol);
        this.compileAndRun(program, [source], dest, customSetup);
    };
    NDArrayMathGPU.prototype.concat3DInternal = function (x1, x2, axis) {
        var program = new concat3d_gpu_1.Concat3DProgram(x1.shape, x2.shape, axis);
        return this.compileAndRun(program, [x1, x2]);
    };
    NDArrayMathGPU.prototype.scaledArrayAddInternal = function (c1, a, c2, b) {
        var program = new addscaledmat_gpu_1.AddScaledMatProgram(a.shape, b.shape);
        return this.compileAndRun(program, [a, b, c1, c2]);
    };
    NDArrayMathGPU.prototype.negInternal = function (a) {
        var program = new unaryop_gpu_1.UnaryOpProgram(a.shape, unaryop_gpu_1.UnaryOp.NEG);
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.makeOutputArray = function (shape) {
        var textureShapeRC = webgl_util.getTextureShapeFromLogicalShape(this.gpgpu.gl, shape);
        var texture = this.textureManager.acquireTexture(textureShapeRC);
        return ndarray_1.NDArray.make(shape, { texture: texture, textureShapeRC: textureShapeRC });
    };
    NDArrayMathGPU.prototype.compileAndRun = function (program, inputs, output, customSetup) {
        var _this = this;
        if (output == null) {
            output = this.makeOutputArray(program.outputShape);
        }
        var key = gpgpu_math.makeShaderKey(program, inputs, output);
        var binary = this.getAndSaveBinary(key, function () {
            return gpgpu_math.compileProgram(_this.gpgpu, program, inputs, output);
        });
        gpgpu_math.runProgram(binary, inputs, output, customSetup);
        return output;
    };
    NDArrayMathGPU.prototype.matMulInternal = function (a, b, aOrientation, bOrientation) {
        var program = new mulmat_gpu_1.MatMulProgram(a.shape, b.shape, aOrientation, bOrientation);
        return this.compileAndRun(program, [a, b]);
    };
    NDArrayMathGPU.prototype.multiplyInternal = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram('*', a.shape, b.shape);
        return this.compileAndRun(program, [a, b]);
    };
    NDArrayMathGPU.prototype.batchNormalization3DInternal = function (x, mean, variance, varianceEpsilon, scale, offset) {
        var inputs = [x, mean, variance];
        if (varianceEpsilon == null) {
            varianceEpsilon = 0.000001;
        }
        var offsetShape = null;
        if (offset != null) {
            offsetShape = offset.shape;
            inputs.push(offset);
        }
        var scaleShape = null;
        if (scale != null) {
            scaleShape = scale.shape;
            inputs.push(scale);
        }
        var program = new batchnorm_gpu_1.BatchNormProgram(x.shape, mean.shape, variance.shape, offsetShape, scaleShape, varianceEpsilon);
        return this.compileAndRun(program, inputs);
    };
    NDArrayMathGPU.prototype.switchDimInternal = function (a, newDim) {
        throw new Error('Not yet implemented!');
    };
    NDArrayMathGPU.prototype.sumInternal = function (a) {
        var program = new reducesum_gpu_1.ReduceSumProgram(a.size);
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.argMinInternal = function (a) {
        var program = new argminmax_gpu_1.ArgMinMaxProgram(a.size, 'min');
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.argMaxInternal = function (a) {
        var program = new argminmax_gpu_1.ArgMinMaxProgram(a.size, 'max');
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.argMaxEqualsInternal = function (x1, x2) {
        var program = new argmaxequals_gpu_1.ArgMaxEqualsProgram(x1.size, x2.size);
        return this.compileAndRun(program, [x1, x2]);
    };
    NDArrayMathGPU.prototype.topKInternal = function (ndarray, k) {
        throw new Error('topK GPU not yet implemented!');
    };
    NDArrayMathGPU.prototype.minInternal = function (a) {
        var program = new minmax_gpu_1.MinMaxProgram(a.size, 'min');
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.maxInternal = function (a) {
        var program = new minmax_gpu_1.MinMaxProgram(a.size, 'max');
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.divideInternal = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram('/', a.shape, b.shape);
        return this.compileAndRun(program, [a, b]);
    };
    NDArrayMathGPU.prototype.addInternal = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram('+', a.shape, b.shape);
        return this.compileAndRun(program, [a, b]);
    };
    NDArrayMathGPU.prototype.subInternal = function (a, b) {
        var program = new binaryop_gpu_1.BinaryOpProgram('-', a.shape, b.shape);
        return this.compileAndRun(program, [a, b]);
    };
    NDArrayMathGPU.prototype.logSumExpInternal = function (a) {
        var program = new logsumexp_gpu_1.LogSumExpProgram(a.size);
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.expInternal = function (a) {
        var program = new unaryop_gpu_1.UnaryOpProgram(a.shape, unaryop_gpu_1.UnaryOp.EXP);
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.logInternal = function (a) {
        var program = new unaryop_gpu_1.UnaryOpProgram(a.shape, unaryop_gpu_1.UnaryOp.LOG);
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.sqrtInternal = function (a) {
        var program = new unaryop_gpu_1.UnaryOpProgram(a.shape, unaryop_gpu_1.UnaryOp.SQRT);
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.reluInternal = function (a) {
        var program = new unaryop_gpu_1.UnaryOpProgram(a.shape, unaryop_gpu_1.UnaryOp.RELU);
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.sigmoidInternal = function (a) {
        var program = new unaryop_gpu_1.UnaryOpProgram(a.shape, unaryop_gpu_1.UnaryOp.SIGMOID);
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.tanhInternal = function (a) {
        var program = new unaryop_gpu_1.UnaryOpProgram(a.shape, unaryop_gpu_1.UnaryOp.TANH);
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.sinInternal = function (a) {
        var program = new unaryop_gpu_1.UnaryOpProgram(a.shape, unaryop_gpu_1.UnaryOp.SIN);
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.stepInternal = function (a) {
        var program = new unaryop_gpu_1.UnaryOpProgram(a.shape, unaryop_gpu_1.UnaryOp.STEP);
        return this.compileAndRun(program, [a]);
    };
    NDArrayMathGPU.prototype.conv2dInternal = function (x, filter, bias, convInfo) {
        var program = new conv_gpu_1.Conv2DProgram(convInfo, bias != null);
        var inputs = bias != null ? [x, filter, bias] : [x, filter];
        return this.compileAndRun(program, inputs);
    };
    NDArrayMathGPU.prototype.conv2dDerInputInternal = function (dy, filter, convInfo) {
        var program = new conv_backprop_gpu_1.Conv2DDerInputProgram(convInfo);
        return this.compileAndRun(program, [dy, filter]);
    };
    NDArrayMathGPU.prototype.conv2dDerFilterInternal = function (x, dY, convInfo) {
        var program = new conv_backprop_gpu_1.Conv2DDerWeightsProgram(convInfo);
        return this.compileAndRun(program, [x, dY]);
    };
    NDArrayMathGPU.prototype.conv2dDerBiasInternal = function (dY) {
        var program = new conv_backprop_gpu_1.Conv2DDerBiasProgram(dY.shape);
        return this.compileAndRun(program, [dY]);
    };
    NDArrayMathGPU.prototype.maxPoolInternal = function (x, convInfo) {
        var program = new pool_gpu_1.Pool2DProgram(convInfo, 'max', false);
        return this.compileAndRun(program, [x]);
    };
    NDArrayMathGPU.prototype.minPoolInternal = function (x, convInfo) {
        var program = new pool_gpu_1.Pool2DProgram(convInfo, 'min', false);
        return this.compileAndRun(program, [x]);
    };
    NDArrayMathGPU.prototype.avgPoolInternal = function (x, convInfo) {
        var program = new pool_gpu_1.Pool2DProgram(convInfo, 'avg', false);
        return this.compileAndRun(program, [x]);
    };
    NDArrayMathGPU.prototype.maxPoolBackpropInternal = function (dy, x, convInfo) {
        var getPositions = true;
        var maxPoolPositionsProgram = new pool_gpu_1.Pool2DProgram(convInfo, 'max', getPositions);
        var maxPoolPositions = this.compileAndRun(maxPoolPositionsProgram, [x]);
        var maxPoolBackPropProgram = new max_pool_backprop_gpu_1.MaxPool2DBackpropProgram(convInfo);
        var result = this.compileAndRun(maxPoolBackPropProgram, [dy, maxPoolPositions]);
        maxPoolPositions.dispose();
        return result;
    };
    NDArrayMathGPU.prototype.resizeBilinear3DInternal = function (x, newShape2D, alignCorners) {
        var program = new resize_bilinear_gpu_1.ResizeBilinear3DProgram(x.shape, newShape2D, alignCorners);
        return this.compileAndRun(program, [x]);
    };
    NDArrayMathGPU.prototype.getAndSaveBinary = function (key, getBinary) {
        if (!(key in this.binaryCache)) {
            this.binaryCache[key] = getBinary();
        }
        return this.binaryCache[key];
    };
    NDArrayMathGPU.prototype.getTextureManager = function () {
        return this.textureManager;
    };
    NDArrayMathGPU.prototype.dispose = function () {
        for (var key in this.binaryCache) {
            this.gpgpu.deleteProgram(this.binaryCache[key].webGLProgram);
        }
        this.textureManager.dispose();
        if (this.gpgpuCreatedLocally) {
            this.gpgpu.dispose();
        }
    };
    return NDArrayMathGPU;
}(math_1.NDArrayMath));
exports.NDArrayMathGPU = NDArrayMathGPU;

},{"./math":23,"./ndarray":26,"./webgl/addscaledmat_gpu":27,"./webgl/argmaxequals_gpu":28,"./webgl/argminmax_gpu":29,"./webgl/batchnorm_gpu":30,"./webgl/binaryop_gpu":31,"./webgl/concat3d_gpu":32,"./webgl/conv_backprop_gpu":33,"./webgl/conv_gpu":34,"./webgl/copy_gpu":35,"./webgl/gpgpu_context":36,"./webgl/gpgpu_math":37,"./webgl/gpgpu_util":38,"./webgl/logsumexp_gpu":39,"./webgl/max_pool_backprop_gpu":40,"./webgl/minmax_gpu":41,"./webgl/mulmat_gpu":42,"./webgl/pool_gpu":43,"./webgl/reducesum_gpu":44,"./webgl/resize_bilinear_gpu":46,"./webgl/texture_manager":49,"./webgl/unaryop_gpu":50,"./webgl/webgl_util":51}],26:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
var webgl_util = require("./webgl/webgl_util");
exports.GPGPU = null;
exports.TEXTURE_MANAGER = null;
function initializeGPU(gpgpu, textureManager) {
    exports.GPGPU = gpgpu;
    exports.TEXTURE_MANAGER = textureManager;
}
exports.initializeGPU = initializeGPU;
function throwIfGPUNotInitialized() {
    if (exports.GPGPU == null || exports.TEXTURE_MANAGER == null) {
        throw new Error('GPU not intialized.');
    }
}
var NDArray = (function () {
    function NDArray(shape, data) {
        util.assert(data.values != null || data.texture != null, 'Either `values` or `texture` must be defined');
        util.assert(data.texture == null || (data.textureShapeRC != null), '`textureShape` must be defined when `texture` is defined');
        this.size = util.sizeFromShape(shape);
        if (data.values != null) {
            util.assert(this.size === data.values.length, 'Constructing ndarray of shape (' + this.size + ') should match the' +
                ' length of values (' + data.values.length + ')');
        }
        this.shape = shape;
        this.data = data;
        var dim = this.shape.length;
        if (dim < 2) {
            this.strides = [];
        }
        else {
            this.strides = new Array(dim - 1);
            this.strides[dim - 2] = this.shape[dim - 1];
            for (var i = dim - 3; i >= 0; --i) {
                this.strides[i] = this.strides[i + 1] * this.shape[i + 1];
            }
        }
    }
    NDArray.zeros = function (shape) {
        var values = new Float32Array(util.sizeFromShape(shape));
        return NDArray.make(shape, { values: values });
    };
    NDArray.zerosLike = function (another) {
        return NDArray.zeros(another.shape);
    };
    NDArray.like = function (another) {
        var values = another.getValues();
        return NDArray.make(another.shape, { values: new Float32Array(values) });
    };
    NDArray.make = function (shape, data) {
        switch (shape.length) {
            case 0:
                return new Scalar(data);
            case 1:
                return new Array1D(data);
            case 2:
                return new Array2D(shape, data);
            case 3:
                return new Array3D(shape, data);
            case 4:
                return new Array4D(shape, data);
            default:
                return new NDArray(shape, data);
        }
    };
    NDArray.prototype.reshape = function (newShape) {
        if (util.arraysEqual(this.shape, newShape)) {
            return this;
        }
        util.assert(this.size === util.sizeFromShape(newShape), 'new shape and old shape must have the same number of elements.');
        return NDArray.make(newShape, this.data);
    };
    NDArray.prototype.asScalar = function () {
        util.assert(this.size === 1, 'The array must have only 1 element.');
        return this.reshape([]);
    };
    NDArray.prototype.as1D = function () {
        return this.reshape([this.size]);
    };
    NDArray.prototype.as2D = function (rows, columns) {
        return this.reshape([rows, columns]);
    };
    NDArray.prototype.as3D = function (rows, columns, depth) {
        return this.reshape([rows, columns, depth]);
    };
    NDArray.prototype.as4D = function (rows, columns, depth, depth2) {
        return this.reshape([rows, columns, depth, depth2]);
    };
    Object.defineProperty(NDArray.prototype, "rank", {
        get: function () {
            return this.shape.length;
        },
        enumerable: true,
        configurable: true
    });
    NDArray.prototype.get = function () {
        var locs = [];
        for (var _i = 0; _i < arguments.length; _i++) {
            locs[_i] = arguments[_i];
        }
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return this.getValues()[index];
    };
    NDArray.prototype.add = function (value) {
        var locs = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            locs[_i - 1] = arguments[_i];
        }
        this.set.apply(this, [this.get.apply(this, locs) + value].concat(locs));
    };
    NDArray.prototype.set = function (value) {
        var locs = [];
        for (var _i = 1; _i < arguments.length; _i++) {
            locs[_i - 1] = arguments[_i];
        }
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        this.getValues()[index] = value;
    };
    NDArray.prototype.locToIndex = function (locs) {
        var index = locs[locs.length - 1];
        for (var i = 0; i < locs.length - 1; ++i) {
            index += this.strides[i] * locs[i];
        }
        return index;
    };
    NDArray.prototype.indexToLoc = function (index) {
        var locs = new Array(this.shape.length);
        for (var i = 0; i < locs.length - 1; ++i) {
            locs[i] = Math.floor(index / this.strides[i]);
            index -= locs[i] * this.strides[i];
        }
        locs[locs.length - 1] = index;
        return locs;
    };
    NDArray.prototype.fill = function (value) {
        this.getValues().fill(value);
    };
    NDArray.prototype.getData = function () {
        return this.data;
    };
    NDArray.prototype.getValues = function () {
        if (this.data.values == null) {
            throwIfGPUNotInitialized();
            this.data.values = exports.GPGPU.downloadMatrixFromTexture(this.data.texture, this.data.textureShapeRC[0], this.data.textureShapeRC[1]);
            this.disposeTexture();
        }
        return this.data.values;
    };
    NDArray.prototype.uploadToGPU = function (preferredTexShape) {
        throwIfGPUNotInitialized();
        this.data.textureShapeRC = webgl_util.getTextureShapeFromLogicalShape(exports.GPGPU.gl, this.shape, preferredTexShape);
        this.data.texture =
            exports.TEXTURE_MANAGER.acquireTexture(this.data.textureShapeRC);
        exports.GPGPU.uploadMatrixToTexture(this.data.texture, this.data.textureShapeRC[0], this.data.textureShapeRC[1], this.data.values);
        this.data.values = null;
    };
    NDArray.prototype.getTexture = function (preferredShapeRC) {
        if (this.data.texture == null) {
            this.uploadToGPU(preferredShapeRC);
        }
        return this.data.texture;
    };
    NDArray.prototype.getTextureShapeRC = function (preferredShapeRC) {
        if (this.data.textureShapeRC == null) {
            this.uploadToGPU(preferredShapeRC);
        }
        return this.data.textureShapeRC;
    };
    NDArray.prototype.dispose = function () {
        this.data.values = null;
        this.shape = null;
        if (this.data.texture != null) {
            this.disposeTexture();
        }
    };
    NDArray.prototype.disposeTexture = function () {
        throwIfGPUNotInitialized();
        exports.TEXTURE_MANAGER.releaseTexture(this.data.texture, this.data.textureShapeRC);
        this.data.texture = null;
        this.data.textureShapeRC = null;
    };
    NDArray.prototype.inGPU = function () {
        return this.data.texture != null;
    };
    NDArray.prototype.equals = function (t) {
        return util.arraysEqual(this.shape, t.shape) &&
            util.arraysEqual(this.getValues(), t.getValues());
    };
    NDArray.rand = function (shape, randFunction) {
        var size = util.sizeFromShape(shape);
        var values = new Float32Array(size);
        for (var i = 0; i < size; i++) {
            values[i] = randFunction();
        }
        return NDArray.make(shape, { values: values });
    };
    NDArray.randNormal = function (shape, mean, stdDev) {
        if (mean === void 0) { mean = 0; }
        if (stdDev === void 0) { stdDev = 1; }
        return NDArray.rand(shape, function () { return util.randGauss(mean, stdDev); });
    };
    NDArray.randTruncatedNormal = function (shape, mean, stdDev) {
        if (mean === void 0) { mean = 0; }
        if (stdDev === void 0) { stdDev = 1; }
        return NDArray.rand(shape, function () { return util.randGauss(mean, stdDev, true); });
    };
    NDArray.randUniform = function (shape, a, b) {
        return NDArray.rand(shape, function () { return util.randUniform(a, b); });
    };
    return NDArray;
}());
exports.NDArray = NDArray;
var Scalar = (function (_super) {
    __extends(Scalar, _super);
    function Scalar(data) {
        var _this = this;
        if (data.texture != null) {
            data.textureShapeRC = [1, 1];
        }
        _this = _super.call(this, [], data) || this;
        return _this;
    }
    Scalar.new = function (value) {
        return new Scalar({ values: new Float32Array([value]) });
    };
    Scalar.prototype.get = function () {
        return this.getValues()[0];
    };
    Scalar.prototype.set = function (value) {
        this.getValues()[0] = value;
    };
    Scalar.prototype.add = function (value) {
        this.getValues()[0] += value;
    };
    Scalar.ZERO = Scalar.new(0);
    Scalar.ONE = Scalar.new(1);
    Scalar.TWO = Scalar.new(2);
    Scalar.NEG_ONE = Scalar.new(-1);
    return Scalar;
}(NDArray));
exports.Scalar = Scalar;
var Array1D = (function (_super) {
    __extends(Array1D, _super);
    function Array1D(data) {
        var _this = this;
        var shape = (data.values != null) ?
            [data.values.length] :
            [util.sizeFromShape(data.textureShapeRC)];
        _this = _super.call(this, shape, data) || this;
        return _this;
    }
    Array1D.new = function (values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            util.assert(inferredShape.length === 1, "Error constructing Array1D. Shape of values " + inferredShape + " is " +
                "not 1 dimensional.");
        }
        return new Array1D({ values: toTypedArray(values) });
    };
    Array1D.prototype.get = function (i) {
        return this.getValues()[i];
    };
    Array1D.prototype.set = function (value, i) {
        this.getValues()[i] = value;
    };
    Array1D.prototype.add = function (value, i) {
        this.getValues()[i] += value;
    };
    Array1D.prototype.locToIndex = function (loc) {
        return loc[0];
    };
    Array1D.prototype.indexToLoc = function (index) {
        return [index];
    };
    Array1D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array1D;
}(NDArray));
exports.Array1D = Array1D;
var Array2D = (function (_super) {
    __extends(Array2D, _super);
    function Array2D(shape, data) {
        var _this = this;
        util.assert(shape.length === 2, 'Shape should be of length 2');
        _this = _super.call(this, shape, data) || this;
        _this.stride0 = _this.strides[0];
        return _this;
    }
    Array2D.new = function (shape, values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array2D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array2D(shape, { values: toTypedArray(values) });
    };
    Array2D.prototype.get = function (i, j) {
        return this.getValues()[this.stride0 * i + j];
    };
    Array2D.prototype.set = function (value, i, j) {
        this.getValues()[this.stride0 * i + j] = value;
    };
    Array2D.prototype.add = function (value, i, j) {
        this.getValues()[this.stride0 * i + j] += value;
    };
    Array2D.prototype.locToIndex = function (locs) {
        return this.stride0 * locs[0] + locs[1];
    };
    Array2D.prototype.indexToLoc = function (index) {
        return [Math.floor(index / this.stride0), index % this.stride0];
    };
    Array2D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array2D;
}(NDArray));
exports.Array2D = Array2D;
var Array3D = (function (_super) {
    __extends(Array3D, _super);
    function Array3D(shape, data) {
        var _this = this;
        util.assert(shape.length === 3, 'Shape should be of length 3');
        _this = _super.call(this, shape, data) || this;
        _this.stride0 = _this.strides[0];
        _this.stride1 = _this.strides[1];
        return _this;
    }
    Array3D.new = function (shape, values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array3D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array3D(shape, { values: toTypedArray(values) });
    };
    Array3D.prototype.get = function (i, j, k) {
        return this.getValues()[this.stride0 * i + this.stride1 * j + k];
    };
    Array3D.prototype.set = function (value, i, j, k) {
        this.getValues()[this.stride0 * i + this.stride1 * j + k] = value;
    };
    Array3D.prototype.add = function (value, i, j, k) {
        this.getValues()[this.stride0 * i + this.stride1 * j + k] += value;
    };
    Array3D.prototype.locToIndex = function (locs) {
        return this.stride0 * locs[0] + this.stride1 * locs[1] + locs[2];
    };
    Array3D.prototype.indexToLoc = function (index) {
        var i = Math.floor(index / this.stride0);
        index -= i * this.stride0;
        return [i, Math.floor(index / this.stride1), index % this.stride1];
    };
    Array3D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array3D;
}(NDArray));
exports.Array3D = Array3D;
var Array4D = (function (_super) {
    __extends(Array4D, _super);
    function Array4D(shape, data) {
        var _this = this;
        util.assert(shape.length === 4, 'Shape should be of length 4');
        _this = _super.call(this, shape, data) || this;
        _this.stride0 = _this.strides[0];
        _this.stride1 = _this.strides[1];
        _this.stride2 = _this.strides[2];
        return _this;
    }
    Array4D.new = function (shape, values) {
        if (!(values instanceof Float32Array)) {
            var inferredShape = util.inferShape(values);
            if (inferredShape.length > 1) {
                util.assertShapesMatch(shape, inferredShape, "Error when constructing Array4D. Shape of values " +
                    (inferredShape + " does not match the provided shape ") +
                    (shape + ". "));
            }
        }
        return new Array4D(shape, { values: toTypedArray(values) });
    };
    Array4D.prototype.get = function (i, j, k, l) {
        return this.getValues()[this.stride0 * i + this.stride1 * j + this.stride2 * k + l];
    };
    Array4D.prototype.set = function (value, i, j, k, l) {
        this.getValues()[this.stride0 * i + this.stride1 * j + this.stride2 * k + l] = value;
    };
    Array4D.prototype.add = function (value, i, j, k, l) {
        this.getValues()[this.stride0 * i + this.stride1 * j + this.stride2 * k + l] += value;
    };
    Array4D.prototype.locToIndex = function (locs) {
        return this.stride0 * locs[0] + this.stride1 * locs[1] +
            this.stride2 * locs[2] + locs[3];
    };
    Array4D.prototype.indexToLoc = function (index) {
        var i = Math.floor(index / this.stride0);
        index -= i * this.stride0;
        var j = Math.floor(index / this.stride1);
        index -= j * this.stride1;
        return [i, j, Math.floor(index / this.stride2), index % this.stride2];
    };
    Array4D.zeros = function (shape) {
        return NDArray.zeros(shape);
    };
    return Array4D;
}(NDArray));
exports.Array4D = Array4D;
function toTypedArray(a) {
    return (a instanceof Float32Array) ? a : new Float32Array(util.flatten(a));
}

},{"../util":79,"./webgl/webgl_util":51}],27:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
var AddScaledMatProgram = (function () {
    function AddScaledMatProgram(aShape, bShape) {
        this.variableNames = ['A', 'B', 'c1', 'c2'];
        this.params = [];
        this.supportsBroadcasting = true;
        this.outputShape = util.assertAndGetBroadcastedShape(aShape, bShape);
        this.userCode = "\n      void main() {\n        float a = getAAtOutCoords();\n        float b = getBAtOutCoords();\n        float c1 = getC1();\n        float c2 = getC2();\n        setOutput(dot(vec2(c1, c2), vec2(a, b)));\n      }\n    ";
    }
    return AddScaledMatProgram;
}());
exports.AddScaledMatProgram = AddScaledMatProgram;

},{"../../util":79}],28:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var argminmax_gpu = require("./argminmax_gpu");
var ArgMaxEqualsProgram = (function () {
    function ArgMaxEqualsProgram(aSize, bSize) {
        this.variableNames = ['A', 'B'];
        this.outputShape = [];
        this.params = [];
        var aSnippet = argminmax_gpu.getArgMinMaxSnippet('max', 'A', aSize);
        var bSnippet = argminmax_gpu.getArgMinMaxSnippet('max', 'B', bSize);
        this.userCode = "\n      " + aSnippet + "\n      " + bSnippet + "\n\n      void main() {\n        float argMaxA = getArgMinMaxA();\n        float argMaxB = getArgMinMaxB();\n\n        float value;\n        if (isNaN(argMaxA)) {\n          value = argMaxA;\n        } else if (isNaN(argMaxB)) {\n          value = argMaxB;\n        } else {\n          value = float(argMaxA == argMaxB);\n        }\n\n        setOutput(value);\n      }\n    ";
    }
    return ArgMaxEqualsProgram;
}());
exports.ArgMaxEqualsProgram = ArgMaxEqualsProgram;

},{"./argminmax_gpu":29}],29:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function getArgMinMaxSnippet(op, texName, size) {
    var compOp = (op === 'min') ? '<' : '>';
    return "\n    float getArgMinMax" + texName + "() {\n      int bestIndex = 0;\n      float bestValue = get" + texName + "Flat(0);\n\n      for (int i = 0; i < " + size + "; i++) {\n        float candidate = get" + texName + "Flat(i);\n        if (isNaN(candidate)) {\n          return candidate;\n        }\n        if (candidate " + compOp + " bestValue) {\n          bestValue = candidate;\n          bestIndex = i;\n        }\n      }\n      return float(bestIndex);\n    }\n  ";
}
exports.getArgMinMaxSnippet = getArgMinMaxSnippet;
var ArgMinMaxProgram = (function () {
    function ArgMinMaxProgram(aSize, opType) {
        this.variableNames = ['A'];
        this.outputShape = [];
        this.params = [opType];
        var aSnippet = getArgMinMaxSnippet(opType, 'A', aSize);
        this.userCode = "\n      " + aSnippet + "\n\n      void main() {\n        setOutput(getArgMinMaxA());\n      }\n    ";
    }
    return ArgMinMaxProgram;
}());
exports.ArgMinMaxProgram = ArgMinMaxProgram;

},{}],30:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
var BatchNormProgram = (function () {
    function BatchNormProgram(xShape, meanShape, varianceShape, offsetShape, scaleShape, varianceEpsilon) {
        this.params = [];
        this.outputShape = [];
        this.supportsBroadcasting = true;
        this.variableNames = ['x', 'mean', 'variance'];
        util.assertAndGetBroadcastedShape(xShape, meanShape);
        util.assertAndGetBroadcastedShape(xShape, varianceShape);
        var offsetSnippet = '0.0';
        if (offsetShape != null) {
            util.assertAndGetBroadcastedShape(xShape, offsetShape);
            this.variableNames.push('offset');
            offsetSnippet = 'getOffsetAtOutCoords()';
        }
        var scaleSnippet = '1.0';
        if (scaleShape != null) {
            util.assertAndGetBroadcastedShape(xShape, scaleShape);
            this.variableNames.push('scale');
            scaleSnippet = 'getScaleAtOutCoords()';
        }
        this.params = [varianceEpsilon];
        this.outputShape = xShape;
        this.userCode = "\n      void main() {\n        float x = getXAtOutCoords();\n        float mean = getMeanAtOutCoords();\n        float variance = getVarianceAtOutCoords();\n        float offset = " + offsetSnippet + ";\n        float scale = " + scaleSnippet + ";\n        float inv = scale / sqrt(variance + float(" + varianceEpsilon + "));\n        setOutput((x - mean) * inv + offset);\n      }\n    ";
    }
    return BatchNormProgram;
}());
exports.BatchNormProgram = BatchNormProgram;

},{"../../util":79}],31:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
var BinaryOpProgram = (function () {
    function BinaryOpProgram(op, aShape, bShape) {
        this.variableNames = ['A', 'B'];
        this.supportsBroadcasting = true;
        this.params = [op];
        this.outputShape = util.assertAndGetBroadcastedShape(aShape, bShape);
        this.userCode = "\n      void main() {\n        float a = getAAtOutCoords();\n        float b = getBAtOutCoords();\n        setOutput(a " + op + " b);\n      }\n    ";
    }
    return BinaryOpProgram;
}());
exports.BinaryOpProgram = BinaryOpProgram;

},{"../../util":79}],32:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var concat3d_util = require("../concat3d_util");
var Concat3DProgram = (function () {
    function Concat3DProgram(x1Shape, x2Shape, axis) {
        this.variableNames = ['A', 'B'];
        this.params = [];
        this.outputShape = [];
        var yAxes = ['yR', 'yC', 'yD'];
        var concatAxis = yAxes[axis];
        this.params = [axis];
        this.outputShape =
            concat3d_util.computeConcat3DOutputShape(x1Shape, x2Shape, axis);
        this.userCode = "\n      void main() {\n        ivec3 coords = getOutputCoords();\n        int yR = coords.x;\n        int yC = coords.y;\n        int yD = coords.z;\n\n        float value = 0.0;\n        if (" + concatAxis + " < " + x1Shape[axis] + ") {\n          value = getA(yR, yC, yD);\n        } else {\n          " + concatAxis + " -= " + x1Shape[axis] + ";\n          value = getB(yR, yC, yD);\n        }\n\n        setOutput(value);\n      }\n    ";
    }
    return Concat3DProgram;
}());
exports.Concat3DProgram = Concat3DProgram;

},{"../concat3d_util":19}],33:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../conv_util");
var Conv2DDerWeightsProgram = (function () {
    function Conv2DDerWeightsProgram(convInfo) {
        this.variableNames = ['x', 'dy'];
        var _a = convInfo.outShape, yNumRows = _a[0], yNumCols = _a[1], outDepth = _a[2];
        var _b = convInfo.inShape, xNumRows = _b[0], xNumCols = _b[1], inDepth = _b[2];
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        this.outputShape = conv_util.computeWeightsShape4D(inDepth, outDepth, convInfo.filterHeight, convInfo.filterWidth);
        var padTop = convInfo.padInfo.top;
        var padLeft = convInfo.padInfo.left;
        this.params = [strideHeight, strideWidth, padLeft, padTop];
        this.userCode = "\n      void main() {\n        ivec4 coords = getOutputCoords();\n        int wR = coords.x;\n        int wC = coords.y;\n        int d1 = coords.z;\n        int d2 = coords.w;\n\n        // Convolve x(?, ?, d1) with dy(:, :, d2) to get dw(wR, wC, d1, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int yR = 0; yR < " + yNumRows + "; yR++) {\n          int xR = wR + yR * " + strideHeight + " - " + padTop + ";\n\n          if (xR < 0 || xR >= " + xNumRows + ") {\n            continue;\n          }\n\n          for (int yC = 0; yC < " + yNumCols + "; yC++) {\n            int xC = wC + yC * " + strideWidth + " - " + padLeft + ";\n\n            if (xC < 0 || xC >= " + xNumCols + ") {\n              continue;\n            }\n\n            float dyValue = getDy(yR, yC, d2);\n            float xValue = getX(xR, xC, d1);\n            dotProd += (xValue * dyValue);\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
    }
    return Conv2DDerWeightsProgram;
}());
exports.Conv2DDerWeightsProgram = Conv2DDerWeightsProgram;
var Conv2DDerInputProgram = (function () {
    function Conv2DDerInputProgram(convInfo) {
        this.variableNames = ['dy', 'W'];
        var _a = convInfo.outShape, yRows = _a[0], yCols = _a[1], outDepth = _a[2];
        this.outputShape = convInfo.inShape;
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var padTop = filterHeight - 1 - convInfo.padInfo.top;
        var padLeft = filterWidth - 1 - convInfo.padInfo.left;
        this.params = [strideHeight, strideWidth, padLeft, padTop];
        this.userCode = "\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec3 coords = getOutputCoords();\n        int d1 = coords.z;\n\n        ivec2 dyCorner = coords.xy - pads;\n        int dyRCorner = dyCorner.x;\n        int dyCCorner = dyCorner.y;\n\n        // Convolve dy(?, ?, d2) with w(:, :, d1, d2) to compute dx(xR, xC, d1).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int wR = 0; wR < " + filterHeight + "; wR++) {\n          float dyR = float(dyRCorner + wR) / " + strideHeight + ".0;\n\n          if (dyR < 0.0 || dyR >= " + yRows + ".0 || fract(dyR) > 0.0) {\n            continue;\n          }\n          int idyR = int(dyR);\n\n          int wRPerm = " + filterHeight + " - 1 - wR;\n\n          for (int wC = 0; wC < " + filterWidth + "; wC++) {\n            float dyC = float(dyCCorner + wC) / " + strideWidth + ".0;\n\n            if (dyC < 0.0 || dyC >= " + yCols + ".0 || fract(dyC) > 0.0) {\n              continue;\n            }\n            int idyC = int(dyC);\n\n            int wCPerm = " + filterWidth + " - 1 - wC;\n\n            for (int d2 = 0; d2 < " + outDepth + "; d2++) {\n              float xValue = getDy(idyR, idyC, d2);\n              float wValue = getW(wRPerm, wCPerm, d1, d2);\n              dotProd += xValue * wValue;\n            }\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
    }
    return Conv2DDerInputProgram;
}());
exports.Conv2DDerInputProgram = Conv2DDerInputProgram;
var Conv2DDerBiasProgram = (function () {
    function Conv2DDerBiasProgram(yShape) {
        this.variableNames = ['dy'];
        this.params = [];
        var yNumRows = yShape[0], yNumCols = yShape[1], outputDepth = yShape[2];
        this.outputShape = [outputDepth];
        this.userCode = "\n      void main() {\n        int d2 = getOutputCoords();\n\n        float derBias = 0.0;\n        for (int yR = 0; yR < " + yNumRows + "; yR++) {\n          for (int yC = 0; yC < " + yNumCols + "; yC++) {\n            derBias += getDy(yR, yC, d2);\n          }\n        }\n        setOutput(derBias);\n      }\n    ";
    }
    return Conv2DDerBiasProgram;
}());
exports.Conv2DDerBiasProgram = Conv2DDerBiasProgram;

},{"../conv_util":20}],34:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Conv2DProgram = (function () {
    function Conv2DProgram(convInfo, hasBias) {
        this.variableNames = ['x', 'W', 'bias'];
        this.outputShape = convInfo.outShape;
        var biasSnippet = hasBias ? 'dotProd += getBias(d2);' : '';
        var _a = convInfo.inShape, xNumRows = _a[0], xNumCols = _a[1], inputDepth = _a[2];
        var padTop = convInfo.padInfo.top;
        var padLeft = convInfo.padInfo.left;
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        this.params = [strideHeight, strideWidth, hasBias, padLeft, padTop];
        this.userCode = "\n      const ivec2 strides = ivec2(" + strideHeight + ", " + strideWidth + ");\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec3 coords = getOutputCoords();\n        int d2 = coords.z;\n\n        ivec2 xRCCorner = coords.xy * strides - pads;\n        int xRCorner = xRCCorner.x;\n        int xCCorner = xRCCorner.y;\n\n        // Convolve x(?, ?, d1) with w(:, :, d1, d2) to get y(yR, yC, d2).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int wR = 0; wR < " + filterHeight + "; wR++) {\n          int xR = xRCorner + wR;\n\n          if (xR < 0 || xR >= " + xNumRows + ") {\n            continue;\n          }\n\n          for (int wC = 0; wC < " + filterWidth + "; wC++) {\n            int xC = xCCorner + wC;\n\n            if (xC < 0 || xC >= " + xNumCols + ") {\n              continue;\n            }\n\n            for (int d1 = 0; d1 < " + inputDepth + "; d1++) {\n              float xValue = getX(xR, xC, d1);\n              float wValue = getW(wR, wC, d1, d2);\n              dotProd += xValue * wValue;\n            }\n          }\n        }\n        " + biasSnippet + "\n        setOutput(dotProd);\n      }\n    ";
    }
    return Conv2DProgram;
}());
exports.Conv2DProgram = Conv2DProgram;

},{}],35:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Copy2DProgram = (function () {
    function Copy2DProgram(srcNumCols, destNumCols) {
        this.variableNames = ['source'];
        this.outputShape = null;
        this.params = [srcNumCols, destNumCols];
        this.userCode = "\n      uniform ivec2 sourceStart;\n      uniform ivec2 destStart;\n\n      void main() {\n        ivec2 destCoords = getOutputCoords() - destStart;\n        int index = destCoords.x * " + destNumCols + " + destCoords.y;\n        int r = index / " + srcNumCols + ";\n        ivec2 sourceCoords = sourceStart + ivec2(r, index - r * " + srcNumCols + ");\n        setOutput(getSource(sourceCoords.x, sourceCoords.y));\n      }\n    ";
    }
    Copy2DProgram.prototype.getCustomSetupFunc = function (sourceStart, destStart, destSize) {
        return function (gpgpu) {
            gpgpu.setOutputMatrixWriteRegion(destStart[0], destSize[0], destStart[1], destSize[1]);
            var sourceStartCRLoc = gpgpu.getUniformLocation('sourceStart');
            gpgpu.gl.uniform2i(sourceStartCRLoc, sourceStart[0], sourceStart[1]);
            var destStartCRLoc = gpgpu.getUniformLocation('destStart');
            gpgpu.gl.uniform2i(destStartCRLoc, destStart[0], destStart[1]);
        };
    };
    return Copy2DProgram;
}());
exports.Copy2DProgram = Copy2DProgram;

},{}],36:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var gpgpu_util = require("./gpgpu_util");
var tex_util = require("./tex_util");
var webgl_util = require("./webgl_util");
var GPGPUContext = (function () {
    function GPGPUContext(gl) {
        this.outputTexture = null;
        this.program = null;
        this.disposed = false;
        this.autoDebugValidate = false;
        if (gl != null) {
            this.gl = gl;
        }
        else {
            this.gl = gpgpu_util.createWebGLContext();
        }
        if (!webgl_util.isWebGL2Enabled()) {
            this.textureFloatExtension =
                webgl_util.getExtensionOrThrow(this.gl, 'OES_texture_float');
            this.colorBufferFloatExtension =
                this.gl.getExtension('WEBGL_color_buffer_float');
        }
        else {
            this.colorBufferFloatExtension =
                webgl_util.getExtensionOrThrow(this.gl, 'EXT_color_buffer_float');
        }
        this.loseContextExtension = webgl_util.getExtensionOrThrow(this.gl, 'WEBGL_lose_context');
        this.vertexBuffer = gpgpu_util.createVertexBuffer(this.gl);
        this.indexBuffer = gpgpu_util.createIndexBuffer(this.gl);
        this.framebuffer = webgl_util.createFramebuffer(this.gl);
    }
    GPGPUContext.prototype.dispose = function () {
        var _this = this;
        this.throwIfDisposed();
        if (this.program != null) {
            console.warn('Disposing a GPGPUContext that still has a bound WebGLProgram.' +
                ' This is probably a resource leak, delete the program with ' +
                'GPGPUContext.deleteProgram before disposing.');
        }
        if (this.outputTexture != null) {
            console.warn('Disposing a GPGPUContext that still has a bound output matrix ' +
                'texture.  This is probably a resource leak, delete the output ' +
                'matrix texture with GPGPUContext.deleteMatrixTexture before ' +
                'disposing.');
        }
        var gl = this.gl;
        webgl_util.callAndCheck(gl, function () { return gl.finish(); });
        webgl_util.callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, null); });
        webgl_util.callAndCheck(gl, function () { return gl.deleteFramebuffer(_this.framebuffer); });
        webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, null); });
        webgl_util.callAndCheck(gl, function () { return gl.deleteBuffer(_this.vertexBuffer); });
        webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null); });
        webgl_util.callAndCheck(gl, function () { return gl.deleteBuffer(_this.indexBuffer); });
        this.loseContextExtension.loseContext();
        this.disposed = true;
    };
    GPGPUContext.prototype.enableAutomaticDebugValidation = function (enabled) {
        this.autoDebugValidate = enabled;
        webgl_util.enableDebugWebGLErrorChecking(enabled);
    };
    GPGPUContext.prototype.createMatrixTexture = function (rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createMatrixTexture(this.gl, rows, columns);
    };
    GPGPUContext.prototype.uploadPixelDataToTexture = function (texture, pixels) {
        this.throwIfDisposed();
        gpgpu_util.uploadPixelDataToTexture(this.gl, texture, pixels);
    };
    GPGPUContext.prototype.createPackedMatrixTexture = function (rows, columns) {
        this.throwIfDisposed();
        return gpgpu_util.createPackedMatrixTexture(this.gl, rows, columns);
    };
    GPGPUContext.prototype.deleteMatrixTexture = function (texture) {
        var _this = this;
        this.throwIfDisposed();
        if (this.outputTexture === texture) {
            webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
            this.outputTexture = null;
        }
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.deleteTexture(texture); });
    };
    GPGPUContext.prototype.uploadMatrixToTexture = function (texture, rows, columns, matrix) {
        this.throwIfDisposed();
        var numChannels = 1;
        return gpgpu_util.uploadMatrixToTexture(this.gl, texture, rows, columns, matrix, numChannels);
    };
    GPGPUContext.prototype.uploadMatrixToPackedTexture = function (texture, rows, columns, matrix) {
        this.throwIfDisposed();
        return gpgpu_util.uploadMatrixToPackedTexture(this.gl, texture, rows, columns, matrix);
    };
    GPGPUContext.prototype.downloadMatrixFromTexture = function (texture, rows, columns) {
        var _this = this;
        return this.downloadMatrixDriver(texture, function () {
            return gpgpu_util.downloadMatrixFromOutputTexture(_this.gl, rows, columns);
        });
    };
    GPGPUContext.prototype.downloadMatrixFromPackedTexture = function (texture, rows, columns) {
        var _this = this;
        return this.downloadMatrixDriver(texture, function () { return gpgpu_util.downloadMatrixFromPackedOutputTexture(_this.gl, rows, columns); });
    };
    GPGPUContext.prototype.createProgram = function (fragmentShaderSource) {
        this.throwIfDisposed();
        var gl = this.gl;
        var fragmentShader = webgl_util.createFragmentShader(gl, fragmentShaderSource);
        var vertexShader = gpgpu_util.createVertexShader(gl);
        var program = webgl_util.createProgram(gl);
        webgl_util.callAndCheck(gl, function () { return gl.attachShader(program, vertexShader); });
        webgl_util.callAndCheck(gl, function () { return gl.attachShader(program, fragmentShader); });
        webgl_util.linkProgram(gl, program);
        if (this.autoDebugValidate) {
            webgl_util.validateProgram(gl, program);
        }
        return program;
    };
    GPGPUContext.prototype.deleteProgram = function (program) {
        var _this = this;
        this.throwIfDisposed();
        if (program === this.program) {
            this.program = null;
        }
        if (program != null) {
            webgl_util.callAndCheck(this.gl, function () { return _this.gl.deleteProgram(program); });
        }
    };
    GPGPUContext.prototype.setProgram = function (program) {
        var _this = this;
        this.throwIfDisposed();
        this.program = program;
        if ((this.program != null) && this.autoDebugValidate) {
            webgl_util.validateProgram(this.gl, this.program);
        }
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.useProgram(program); });
    };
    GPGPUContext.prototype.getUniformLocation = function (uniformName) {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        return webgl_util.getProgramUniformLocationOrThrow(this.gl, this.program, uniformName);
    };
    GPGPUContext.prototype.setInputMatrixTexture = function (inputMatrixTexture, uniformName, textureUnit) {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        webgl_util.bindTextureToProgramUniformSampler(this.gl, this.program, inputMatrixTexture, uniformName, textureUnit);
    };
    GPGPUContext.prototype.setOutputMatrixTexture = function (outputMatrixTexture, rows, columns) {
        this.setOutputMatrixTextureDriver(outputMatrixTexture, columns, rows);
    };
    GPGPUContext.prototype.setOutputPackedMatrixTexture = function (outputPackedMatrixTexture, rows, columns) {
        this.throwIfDisposed();
        var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
        this.setOutputMatrixTextureDriver(outputPackedMatrixTexture, width, height);
    };
    GPGPUContext.prototype.setOutputMatrixWriteRegion = function (startRow, numRows, startColumn, numColumns) {
        this.setOutputMatrixWriteRegionDriver(startColumn, startRow, numColumns, numRows);
    };
    GPGPUContext.prototype.setOutputPackedMatrixWriteRegion = function (startRow, numRows, startColumn, numColumns) {
        throw new Error('setOutputPackedMatrixWriteRegion not implemented.');
    };
    GPGPUContext.prototype.debugValidate = function () {
        if (this.program != null) {
            webgl_util.validateProgram(this.gl, this.program);
        }
        webgl_util.validateFramebuffer(this.gl);
    };
    GPGPUContext.prototype.executeProgram = function () {
        this.throwIfDisposed();
        this.throwIfNoProgram();
        var gl = this.gl;
        gpgpu_util.bindVertexProgramAttributeStreams(gl, this.program, this.vertexBuffer);
        if (this.autoDebugValidate) {
            this.debugValidate();
        }
        webgl_util.callAndCheck(gl, function () { return gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0); });
    };
    GPGPUContext.prototype.blockUntilAllProgramsCompleted = function () {
        var _this = this;
        this.throwIfDisposed();
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.finish(); });
    };
    GPGPUContext.prototype.downloadMatrixDriver = function (texture, downloadAndDecode) {
        this.throwIfDisposed();
        webgl_util.bindColorTextureToFramebuffer(this.gl, texture, this.framebuffer);
        if (this.autoDebugValidate) {
            webgl_util.validateFramebuffer(this.gl);
        }
        var result = downloadAndDecode();
        if (this.outputTexture != null) {
            webgl_util.bindColorTextureToFramebuffer(this.gl, this.outputTexture, this.framebuffer);
            if (this.autoDebugValidate) {
                webgl_util.validateFramebuffer(this.gl);
            }
        }
        else {
            webgl_util.unbindColorTextureFromFramebuffer(this.gl, this.framebuffer);
        }
        return result;
    };
    GPGPUContext.prototype.setOutputMatrixTextureDriver = function (outputMatrixTextureMaybePacked, width, height) {
        this.throwIfDisposed();
        var gl = this.gl;
        webgl_util.bindColorTextureToFramebuffer(gl, outputMatrixTextureMaybePacked, this.framebuffer);
        if (this.autoDebugValidate) {
            webgl_util.validateFramebuffer(gl);
        }
        this.outputTexture = outputMatrixTextureMaybePacked;
        webgl_util.callAndCheck(gl, function () { return gl.viewport(0, 0, width, height); });
        webgl_util.callAndCheck(gl, function () { return gl.scissor(0, 0, width, height); });
    };
    GPGPUContext.prototype.setOutputMatrixWriteRegionDriver = function (x, y, width, height) {
        var _this = this;
        this.throwIfDisposed();
        webgl_util.callAndCheck(this.gl, function () { return _this.gl.scissor(x, y, width, height); });
    };
    GPGPUContext.prototype.throwIfDisposed = function () {
        if (this.disposed) {
            throw new Error('Attempted to use disposed GPGPUContext.');
        }
    };
    GPGPUContext.prototype.throwIfNoProgram = function () {
        if (this.program == null) {
            throw new Error('No GPU program is currently set.');
        }
    };
    return GPGPUContext;
}());
exports.GPGPUContext = GPGPUContext;

},{"./gpgpu_util":38,"./tex_util":48,"./webgl_util":51}],37:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
var shader_compiler = require("./shader_compiler");
function compileProgram(gpgpu, program, inputs, output) {
    var userCode = program.userCode;
    var inputInfos = inputs.map(function (input, i) {
        var shapeInfo = {
            logicalShape: input.shape,
            texShape: input.getTextureShapeRC()
        };
        return { name: program.variableNames[i], shapeInfo: shapeInfo };
    });
    var inShapeInfos = inputInfos.map(function (x) { return x.shapeInfo; });
    var outShapeInfo = {
        logicalShape: output.shape,
        texShape: output.getTextureShapeRC()
    };
    var source = shader_compiler.makeShader(inputInfos, outShapeInfo, userCode, program.supportsBroadcasting === true);
    return {
        program: program,
        source: source,
        webGLProgram: gpgpu.createProgram(source), gpgpu: gpgpu, inShapeInfos: inShapeInfos, outShapeInfo: outShapeInfo
    };
}
exports.compileProgram = compileProgram;
function validateBinaryAndProgram(shapeInfos, inputs) {
    if (shapeInfos.length !== inputs.length) {
        throw Error("Binary was compiled with " + shapeInfos.length + " inputs, but " +
            ("was executed with " + inputs.length + " inputs"));
    }
    shapeInfos.forEach(function (s, i) {
        var shapeA = s.logicalShape;
        var texShapeA = s.texShape;
        var shapeB = inputs[i].shape;
        var texShapeB = inputs[i].getTextureShapeRC();
        if (!util.arraysEqual(shapeA, shapeB)) {
            throw Error("Binary was compiled with different shapes than " +
                ("the current args. Shapes " + shapeA + " and " + shapeB + " must match"));
        }
        if (!util.arraysEqual(texShapeA, texShapeB)) {
            throw Error("Binary was compiled with different texture shapes than the" +
                (" current args. Shape " + texShapeA + " and " + texShapeB + " must match"));
        }
    });
}
function runProgram(binary, inputs, output, customSetup) {
    validateBinaryAndProgram(binary.inShapeInfos, inputs);
    validateBinaryAndProgram([binary.outShapeInfo], [output]);
    var outTex = output.getTexture();
    var outTexShape = output.getTextureShapeRC();
    var gpgpu = binary.gpgpu;
    gpgpu.setOutputMatrixTexture(outTex, outTexShape[0], outTexShape[1]);
    gpgpu.setProgram(binary.webGLProgram);
    inputs.forEach(function (input, i) {
        var tex = input.getTexture();
        gpgpu.setInputMatrixTexture(tex, binary.program.variableNames[i], i);
    });
    if (customSetup != null) {
        customSetup(gpgpu);
    }
    gpgpu.executeProgram();
}
exports.runProgram = runProgram;
function makeShaderKey(program, inputs, output) {
    var params = program.params;
    var keyStart = inputs.concat(output).map(function (x) { return x.shape + '_' + x.getTextureShapeRC(); });
    var keyEnd = params.map(String);
    var key = [program.constructor.name];
    key.push((program.supportsBroadcasting === true).toString());
    key = key.concat(keyStart, keyEnd);
    return key.join('_');
}
exports.makeShaderKey = makeShaderKey;

},{"../../util":79,"./shader_compiler":47}],38:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tex_util = require("./tex_util");
var webgl_util = require("./webgl_util");
function getWebGLContextAttributes() {
    return {
        alpha: false,
        antialias: false,
        premultipliedAlpha: false,
        preserveDrawingBuffer: false,
        depth: false,
        stencil: false,
        failIfMajorPerformanceCaveat: true
    };
}
exports.getWebGLContextAttributes = getWebGLContextAttributes;
function createWebGLContext(canvas) {
    var attributes = getWebGLContextAttributes();
    var gl;
    if (canvas != null) {
        gl = webgl_util.createWebGLRenderingContextFromCanvas(canvas, attributes);
    }
    else {
        gl = webgl_util.createWebGLRenderingContext(attributes);
    }
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.DEPTH_TEST); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.STENCIL_TEST); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.BLEND); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.DITHER); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.POLYGON_OFFSET_FILL); });
    webgl_util.callAndCheck(gl, function () { return gl.disable(gl.SAMPLE_COVERAGE); });
    webgl_util.callAndCheck(gl, function () { return gl.enable(gl.SCISSOR_TEST); });
    webgl_util.callAndCheck(gl, function () { return gl.enable(gl.CULL_FACE); });
    webgl_util.callAndCheck(gl, function () { return gl.cullFace(gl.BACK); });
    return gl;
}
exports.createWebGLContext = createWebGLContext;
function createVertexShader(gl) {
    var vertexShaderSource = "\n    precision highp float;\n    attribute vec3 clipSpacePos;\n    attribute vec2 uv;\n    varying vec2 resultUV;\n\n    void main() {\n      gl_Position = vec4(clipSpacePos, 1);\n      resultUV = uv;\n    }";
    return webgl_util.createVertexShader(gl, vertexShaderSource);
}
exports.createVertexShader = createVertexShader;
function createVertexBuffer(gl) {
    var vertexArray = new Float32Array([-1, 1, 0, 0, 1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 1, -1, 0, 1, 0]);
    return webgl_util.createStaticVertexBuffer(gl, vertexArray);
}
exports.createVertexBuffer = createVertexBuffer;
function createIndexBuffer(gl) {
    var triangleVertexIndices = new Uint16Array([0, 1, 2, 2, 1, 3]);
    return webgl_util.createStaticIndexBuffer(gl, triangleVertexIndices);
}
exports.createIndexBuffer = createIndexBuffer;
function getTextureInternalFormat(gl, numChannels) {
    if (webgl_util.isWebGL2Enabled()) {
        if (numChannels === 4) {
            return gl.RGBA32F;
        }
        return gl.R32F;
    }
    return gl.RGBA;
}
function getTextureFormat(gl, numChannels) {
    if (webgl_util.isWebGL2Enabled()) {
        if (numChannels === 4) {
            return gl.RGBA;
        }
        return gl.RED;
    }
    return gl.RGBA;
}
function createAndConfigureTexture(gl, width, height, numChannels) {
    webgl_util.validateTextureSize(gl, width, height);
    var texture = webgl_util.createTexture(gl);
    var tex2d = gl.TEXTURE_2D;
    var internalFormat = getTextureInternalFormat(gl, numChannels);
    var format = getTextureFormat(gl, numChannels);
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(tex2d, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MIN_FILTER, gl.NEAREST); });
    webgl_util.callAndCheck(gl, function () { return gl.texParameteri(tex2d, gl.TEXTURE_MAG_FILTER, gl.NEAREST); });
    webgl_util.callAndCheck(gl, function () { return gl.texImage2D(tex2d, 0, internalFormat, width, height, 0, format, gl.FLOAT, null); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
    return texture;
}
function createMatrixTexture(gl, rows, columns) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    var numChannels = 1;
    return createAndConfigureTexture(gl, width, height, numChannels);
}
exports.createMatrixTexture = createMatrixTexture;
function createColorMatrixTexture(gl, rows, columns) {
    var _a = tex_util.getColorMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    var numChannels = 4;
    return createAndConfigureTexture(gl, width, height, numChannels);
}
exports.createColorMatrixTexture = createColorMatrixTexture;
function createPackedMatrixTexture(gl, rows, columns) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), width = _a[0], height = _a[1];
    var numChannels = 4;
    return createAndConfigureTexture(gl, width, height, numChannels);
}
exports.createPackedMatrixTexture = createPackedMatrixTexture;
function bindVertexProgramAttributeStreams(gl, program, vertexBuffer) {
    var posOffset = 0;
    var uvOffset = 3 * 4;
    var stride = (3 * 4) + (2 * 4);
    webgl_util.callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer); });
    webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'clipSpacePos', vertexBuffer, 3, stride, posOffset);
    try {
        webgl_util.bindVertexBufferToProgramAttribute(gl, program, 'uv', vertexBuffer, 2, stride, uvOffset);
    }
    catch (e) {
        if (!e.hasOwnProperty('namedVertexAttributeNotFound')) {
            throw e;
        }
    }
}
exports.bindVertexProgramAttributeStreams = bindVertexProgramAttributeStreams;
function uploadPixelDataToTexture(gl, texture, pixels) {
    var numChannels = 4;
    var internalFormat = getTextureInternalFormat(gl, numChannels);
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, gl.RGBA, gl.FLOAT, pixels); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
}
exports.uploadPixelDataToTexture = uploadPixelDataToTexture;
function uploadDataToTexture(gl, texture, width, height, data, numChannels) {
    var textureFormat = getTextureFormat(gl, numChannels);
    webgl_util.validateTextureSize(gl, width, height);
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
    webgl_util.callAndCheck(gl, function () { return gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, textureFormat, gl.FLOAT, data); });
    webgl_util.callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
}
function uploadMatrixToTexture(gl, texture, rows, columns, matrix, numChannels) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var channelsPerTexture = numChannels === 1 ? webgl_util.getChannelsPerTexture() : numChannels;
    var unpackedArray;
    if (channelsPerTexture === 1) {
        unpackedArray = matrix;
    }
    else {
        unpackedArray =
            new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(matrix.length, channelsPerTexture));
        tex_util.encodeMatrixToUnpackedArray(matrix, unpackedArray, channelsPerTexture);
    }
    uploadDataToTexture(gl, texture, w, h, unpackedArray, numChannels);
}
exports.uploadMatrixToTexture = uploadMatrixToTexture;
function uploadMatrixToPackedTexture(gl, texture, rows, columns, matrix) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var packedRGBA = new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(rows, columns));
    tex_util.encodeMatrixToPackedRGBA(matrix, rows, columns, packedRGBA);
    var numChannels = 4;
    uploadDataToTexture(gl, texture, w, h, packedRGBA, numChannels);
}
exports.uploadMatrixToPackedTexture = uploadMatrixToPackedTexture;
function downloadMatrixFromOutputTexture(gl, rows, columns) {
    var _a = tex_util.getUnpackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var channelsPerTexture = 4;
    var unpackedArray = new Float32Array(tex_util.getUnpackedArraySizeFromMatrixSize(rows * columns, channelsPerTexture));
    webgl_util.callAndCheck(gl, function () { return gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, unpackedArray); });
    var matrix = new Float32Array(rows * columns);
    tex_util.decodeMatrixFromUnpackedArray(unpackedArray, matrix, channelsPerTexture);
    return matrix;
}
exports.downloadMatrixFromOutputTexture = downloadMatrixFromOutputTexture;
function downloadMatrixFromPackedOutputTexture(gl, rows, columns) {
    var _a = tex_util.getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    var packedRGBA = new Float32Array(tex_util.getPackedRGBAArraySizeFromMatrixShape(rows, columns));
    webgl_util.callAndCheck(gl, function () { return gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, packedRGBA); });
    var matrix = new Float32Array(rows * columns);
    return tex_util.decodeMatrixFromPackedRGBA(packedRGBA, rows, columns, matrix);
}
exports.downloadMatrixFromPackedOutputTexture = downloadMatrixFromPackedOutputTexture;

},{"./tex_util":48,"./webgl_util":51}],39:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var LogSumExpProgram = (function () {
    function LogSumExpProgram(aSize) {
        this.variableNames = ['A'];
        this.params = [];
        this.outputShape = [];
        this.userCode = "\n      void main() {\n        float aMax = getAFlat(0);\n        for (int i = 0; i < " + aSize + "; i++) {\n          aMax = max(aMax, getAFlat(i));\n        }\n\n        float expSum = 0.0;\n        for (int i = 0; i < " + aSize + "; i++) {\n          expSum += exp(getAFlat(i) - aMax);\n        }\n\n        setOutput(aMax + log(expSum));\n      }\n    ";
    }
    return LogSumExpProgram;
}());
exports.LogSumExpProgram = LogSumExpProgram;

},{}],40:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var MaxPool2DBackpropProgram = (function () {
    function MaxPool2DBackpropProgram(convInfo) {
        this.variableNames = ['dy', 'maxPos'];
        this.outputShape = convInfo.inShape;
        var dyRows = convInfo.outShape[0];
        var dyCols = convInfo.outShape[1];
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var padTop = filterHeight - 1 - convInfo.padInfo.top;
        var padLeft = filterWidth - 1 - convInfo.padInfo.left;
        this.params =
            [filterHeight, filterWidth, strideHeight, strideWidth, padTop, padLeft];
        var lastIndex = filterHeight * filterWidth - 1;
        this.userCode = "\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec3 coords = getOutputCoords();\n        int d = coords.z;\n\n        ivec2 dyRCCorner = coords.xy - pads;\n        int dyRCorner = dyRCCorner.x;\n        int dyCCorner = dyRCCorner.y;\n\n        // Convolve dy(?, ?, d) with pos mask(:, :, d) to get dx(xR, xC, d).\n        // ? = to be determined. : = across all values in that axis.\n        float dotProd = 0.0;\n        for (int wR = 0; wR < " + filterHeight + "; wR++) {\n          float dyR = float(dyRCorner + wR) / " + strideHeight + ".0;\n\n          if (dyR < 0.0 || dyR >= " + dyRows + ".0 || fract(dyR) > 0.0) {\n            continue;\n          }\n          int idyR = int(dyR);\n\n          for (int wC = 0; wC < " + filterWidth + "; wC++) {\n            float dyC = float(dyCCorner + wC) / " + strideWidth + ".0;\n\n            if (dyC < 0.0 || dyC >= " + dyCols + ".0 || fract(dyC) > 0.0) {\n              continue;\n            }\n            int idyC = int(dyC);\n\n            float dyValue = getDy(idyR, idyC, d);\n            int maxPosValue = " + lastIndex + " - int(getMaxPos(idyR, idyC, d));\n\n            // Get the current value, check it against the value from the\n            // position matrix.\n            int curPosValue = wR * " + filterWidth + " + wC;\n            float mask = float(maxPosValue == curPosValue ? 1.0 : 0.0);\n\n            dotProd += dyValue * mask;\n          }\n        }\n        setOutput(dotProd);\n      }\n    ";
    }
    return MaxPool2DBackpropProgram;
}());
exports.MaxPool2DBackpropProgram = MaxPool2DBackpropProgram;

},{}],41:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var MinMaxProgram = (function () {
    function MinMaxProgram(aSize, opType) {
        this.variableNames = ['A'];
        this.outputShape = [];
        this.params = [opType];
        this.userCode = "\n      void main() {\n        float value = getAFlat(0);\n        for (int i = 0; i < " + aSize + "; i++) {\n          float candidate = getAFlat(i);\n          if (isNaN(candidate)) {\n            setOutput(candidate);\n            return;\n          }\n          value = " + opType + "(value, candidate);\n        }\n        setOutput(value);\n      }\n    ";
    }
    return MinMaxProgram;
}());
exports.MinMaxProgram = MinMaxProgram;

},{}],42:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var math_1 = require("../math");
var MatMulProgram = (function () {
    function MatMulProgram(aShape, bShape, aOrient, bOrient) {
        if (aOrient === void 0) { aOrient = math_1.MatrixOrientation.REGULAR; }
        if (bOrient === void 0) { bOrient = math_1.MatrixOrientation.REGULAR; }
        this.variableNames = ['matrixA', 'matrixB'];
        this.params = [aOrient, bOrient];
        var outerShapeA = (aOrient === math_1.MatrixOrientation.REGULAR) ? aShape[0] : aShape[1];
        var outerShapeB = (bOrient === math_1.MatrixOrientation.REGULAR) ? bShape[1] : bShape[0];
        this.outputShape = [outerShapeA, outerShapeB];
        var sharedDim = (aOrient === math_1.MatrixOrientation.REGULAR ? aShape[1] : aShape[0]);
        var aSnippet = (aOrient === math_1.MatrixOrientation.REGULAR) ? 'aRow, i' : 'i, aRow';
        var bSnippet = (bOrient === math_1.MatrixOrientation.REGULAR) ? 'i, bCol' : 'bCol, i';
        this.userCode = "\n      const int sharedDim = " + sharedDim + ";\n\n      float dotARowBCol(int aRow, int bCol) {\n        float result = 0.0;\n        for (int i = 0; i < sharedDim; i++) {\n          float a = getMatrixA(" + aSnippet + ");\n          float b = getMatrixB(" + bSnippet + ");\n          result += (a * b);\n        }\n        return result;\n      }\n\n      void main() {\n        ivec2 resRC = getOutputCoords();\n        setOutput(dotARowBCol(resRC.x, resRC.y));\n      }\n    ";
    }
    return MatMulProgram;
}());
exports.MatMulProgram = MatMulProgram;

},{"../math":23}],43:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Pool2DProgram = (function () {
    function Pool2DProgram(convInfo, poolType, computePositions) {
        this.variableNames = ['x'];
        if (poolType === 'avg' && computePositions) {
            throw new Error('Cannot compute positions for average pool.');
        }
        var filterHeight = convInfo.filterHeight;
        var filterWidth = convInfo.filterWidth;
        var strideHeight = convInfo.strideHeight;
        var strideWidth = convInfo.strideWidth;
        var returnValue = 'minMaxValue';
        if (computePositions) {
            returnValue = 'float(minMaxPosition)';
        }
        else if (poolType === 'avg') {
            returnValue = "avgValue / " + filterHeight * filterWidth + ".0";
        }
        var xNumRows = convInfo.inShape[0];
        var xNumCols = convInfo.inShape[1];
        var padTop = convInfo.padInfo.top;
        var padLeft = convInfo.padInfo.left;
        this.params = [
            strideHeight, strideWidth, padLeft, padTop, poolType, computePositions
        ];
        this.outputShape = convInfo.outShape;
        var isAvgPool = poolType === 'avg';
        var compareOp = poolType === 'min' ? '<=' : '>=';
        this.userCode = "\n      const ivec2 strides = ivec2(" + strideHeight + ", " + strideWidth + ");\n      const ivec2 pads = ivec2(" + padTop + ", " + padLeft + ");\n\n      void main() {\n        ivec3 coords = getOutputCoords();\n        int d = coords.z;\n\n        ivec2 xRCCorner = coords.xy * strides - pads;\n        int xRCorner = xRCCorner.x;\n        int xCCorner = xRCCorner.y;\n\n        // max/min x(?, ?, d) to get y(yR, yC, d).\n        // ? = to be determined\n        float minMaxValue = 0.0;\n        float minMaxValueFound = 0.0;\n        int minMaxPosition = 0;\n        float avgValue = 0.0;\n\n        for (int wR = 0; wR < " + filterHeight + "; wR++) {\n          int xR = xRCorner + wR;\n\n          if (xR < 0 || xR >= " + xNumRows + ") {\n            continue;\n          }\n\n          for (int wC = 0; wC < " + filterWidth + "; wC++) {\n            int xC = xCCorner + wC;\n\n            if (xC < 0 || xC >= " + xNumCols + ") {\n              continue;\n            }\n\n            float value = getX(xR, xC, d);\n\n            if (isNaN(value)) {\n              setOutput(value);\n              return;\n            }\n\n            if (" + isAvgPool + ") {\n              avgValue += value;\n            } else {\n              // If a min / max value has already been found, use it. If not,\n              // use the current value.\n              float currMinMaxValue = mix(\n                  value, minMaxValue, minMaxValueFound);\n              if (value " + compareOp + " currMinMaxValue) {\n                minMaxValue = value;\n                minMaxValueFound = 1.0;\n                if (" + computePositions + ") {\n                  minMaxPosition = wR * " + filterWidth + " + wC;\n                }\n              }\n            }\n          }\n        }\n        setOutput(" + returnValue + ");\n      }\n    ";
    }
    return Pool2DProgram;
}());
exports.Pool2DProgram = Pool2DProgram;

},{}],44:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ReduceSumProgram = (function () {
    function ReduceSumProgram(aSize) {
        this.aSize = aSize;
        this.variableNames = ['A'];
        this.params = [];
        this.outputShape = [];
        this.userCode = "\n      void main() {\n        float sum = 0.0;\n        for (int i = 0; i < " + aSize + "; i++) {\n          sum += getAFlat(i);\n        }\n        setOutput(sum);\n      }\n    ";
    }
    return ReduceSumProgram;
}());
exports.ReduceSumProgram = ReduceSumProgram;

},{}],45:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var webgl_util = require("./webgl_util");
function getRenderRGBShader(gpgpu, destinationWidth) {
    var fragmentShaderSource = "\n    precision highp float;\n    uniform sampler2D source;\n    varying vec2 resultUV;\n\n    const float destinationWidth = " + destinationWidth + ".0;\n    const float a = 1.0;\n\n    void main() {\n      float xr = floor(resultUV.s * destinationWidth) * 3.0;\n      vec3 x = xr + vec3(0, 1, 2);\n\n      float sourceWidth = destinationWidth * 3.0;\n      vec3 u = (x + 0.5) / sourceWidth;\n      float v = 1.0 - resultUV.t;\n\n      float r = texture2D(source, vec2(u[0], v)).r;\n      float g = texture2D(source, vec2(u[1], v)).r;\n      float b = texture2D(source, vec2(u[2], v)).r;\n\n      gl_FragColor = vec4(r, g, b, a);\n    }";
    return gpgpu.createProgram(fragmentShaderSource);
}
exports.getRenderRGBShader = getRenderRGBShader;
function renderToCanvas(gpgpu, renderShader, sourceTex) {
    webgl_util.bindCanvasToFramebuffer(gpgpu.gl);
    renderToFramebuffer(gpgpu, renderShader, sourceTex);
}
exports.renderToCanvas = renderToCanvas;
function renderToFramebuffer(gpgpu, renderShader, sourceTex) {
    gpgpu.setProgram(renderShader);
    gpgpu.setInputMatrixTexture(sourceTex, 'source', 0);
    gpgpu.executeProgram();
}
exports.renderToFramebuffer = renderToFramebuffer;

},{"./webgl_util":51}],46:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ResizeBilinear3DProgram = (function () {
    function ResizeBilinear3DProgram(inputShape, outputDimensionsRowCol, alignCorners) {
        this.variableNames = ['A'];
        this.params = [];
        this.outputShape = [];
        var depth = inputShape[2];
        this.outputShape =
            [outputDimensionsRowCol[0], outputDimensionsRowCol[1], depth];
        this.params = [alignCorners];
        var effectiveInputShape = alignCorners ?
            [inputShape[0] - 1, inputShape[1] - 1, depth] :
            inputShape;
        var effectiveOutputShape = alignCorners ?
            [this.outputShape[0] - 1, this.outputShape[1] - 1, depth] :
            this.outputShape;
        this.userCode = "\n      const vec2 effectiveInputOverOutputRatioRC = vec2(\n          " + effectiveInputShape[0] /
            effectiveOutputShape[0] + ",\n          " + effectiveInputShape[1] /
            effectiveOutputShape[1] + ");\n      const vec2 inputShapeRC = vec2(" + inputShape[0] + ".0, " + inputShape[1] + ".0);\n\n      void main() {\n        ivec3 coords = getOutputCoords();\n        ivec2 yRC = coords.xy;\n        int d = coords.z;\n\n        // Fractional source index.\n        vec2 sourceFracIndexRC = vec2(yRC) * effectiveInputOverOutputRatioRC;\n\n        // Compute the four integer indices.\n        ivec2 sourceFloorRC = ivec2(sourceFracIndexRC);\n        ivec2 sourceCeilRC = ivec2(\n          min(inputShapeRC - 1.0, ceil(sourceFracIndexRC)));\n\n        float topLeft = getA(sourceFloorRC.x, sourceFloorRC.y, d);\n        float bottomLeft = getA(sourceCeilRC.x, sourceFloorRC.y, d);\n        float topRight = getA(sourceFloorRC.x, sourceCeilRC.y, d);\n        float bottomRight = getA(sourceCeilRC.x, sourceCeilRC.y, d);\n\n        vec2 fracRC = sourceFracIndexRC - vec2(sourceFloorRC);\n\n        float top = topLeft + (topRight - topLeft) * fracRC.y;\n        float bottom = bottomLeft + (bottomRight - bottomLeft) * fracRC.y;\n        float newValue = top + (bottom - top) * fracRC.x;\n\n        setOutput(newValue);\n      }\n    ";
    }
    return ResizeBilinear3DProgram;
}());
exports.ResizeBilinear3DProgram = ResizeBilinear3DProgram;

},{}],47:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../../util");
function makeShader(inputsInfo, outputShape, userCode, broadcast) {
    var inputPrefixSnippet = inputsInfo.map(function (x) { return "uniform sampler2D " + x.name + ";"; }).join('\n');
    var inputSamplingSnippet = inputsInfo.map(function (x) { return getInputSamplingSnippet(x, outputShape, broadcast); })
        .join('\n');
    var outTexShape = outputShape.texShape;
    var outputSamplingSnippet = getOutputSamplingSnippet(outputShape.logicalShape, outTexShape);
    var source = [
        SHADER_PREFIX, inputPrefixSnippet, inputSamplingSnippet,
        outputSamplingSnippet, userCode
    ].join('\n');
    return source;
}
exports.makeShader = makeShader;
function getInputSamplingSnippet(inInfo, outShapeInfo, broadcast) {
    var shape = inInfo.shapeInfo.logicalShape;
    var texShape = inInfo.shapeInfo.texShape;
    var outTexShape = outShapeInfo.texShape;
    var res = '';
    switch (shape.length) {
        case 0:
            res += getSamplerScalar(inInfo.name);
            break;
        case 1:
            res += getSampler1D(inInfo.name, texShape);
            break;
        case 2:
            res += getSampler2D(inInfo.name, shape, texShape);
            break;
        case 3:
            res += getSampler3D(inInfo.name, shape, texShape);
            break;
        case 4:
            res += getSampler4D(inInfo.name, shape, texShape);
            break;
        default:
            throw new Error(shape.length + "-D input sampling" +
                " is not yet supported");
    }
    if (broadcast ||
        util.arraysEqual(inInfo.shapeInfo.logicalShape, outShapeInfo.logicalShape)) {
        res +=
            getSamplerAtOutputCoords(inInfo.name, texShape, outTexShape, broadcast);
    }
    res += getSamplerFlat(inInfo.name, texShape);
    return res;
}
function getOutputSamplingSnippet(outShape, outTexShape) {
    switch (outShape.length) {
        case 0:
            return '';
        case 1:
            return getOutput1DCoords(outShape, outTexShape);
        case 2:
            return getOutput2DCoords(outShape, outTexShape);
        case 3:
            return getOutput3DCoords(outShape, outTexShape);
        case 4:
            return getOutput4DCoords(outShape, outTexShape);
        default:
            throw new Error(outShape.length + "-D output sampling is not yet supported");
    }
}
var SAMPLE_1D_SNIPPET = "\nvec2 UVfrom1D(int texNumR, int texNumC, int index) {\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SAMPLE_2D_SNIPPET = "\nvec2 UVfrom2D(int texNumR, int texNumC, int numC, int row, int col) {\n  int index = row * numC + col;\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SAMPLE_3D_SNIPPET = "\nvec2 UVfrom3D(int texNumR, int texNumC, int stride0,\n    int stride1, int row, int col, int depth) {\n  int index = row * stride0 + col * stride1 + depth;\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SAMPLE_4D_SNIPPET = "\nvec2 UVfrom4D(int texNumR, int texNumC, int stride0,\n    int stride1, int stride2, int row, int col, int depth,\n    int depth2) {\n  int index = row * stride0 + col * stride1 + depth * stride2 + depth2;\n  int texR = index / texNumC;\n  int texC = index - texR * texNumC;\n  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);\n}\n";
var SHADER_PREFIX = "\n  precision highp float;\n  varying vec2 resultUV;\n  const vec2 halfCR = vec2(0.5, 0.5);\n\n  float sample(sampler2D texture, vec2 uv) {\n    return texture2D(texture, uv).r;\n  }\n\n  void setOutput(float val) {\n    gl_FragColor = vec4(val, 0, 0, 0);\n  }\n\n  bool isNaN(float val) {\n    return val == val ? false : true;\n  }\n  " + SAMPLE_1D_SNIPPET + "\n  " + SAMPLE_2D_SNIPPET + "\n  " + SAMPLE_3D_SNIPPET + "\n  " + SAMPLE_4D_SNIPPET + "\n";
function getOutput1DCoords(shape, texShape) {
    if (texShape[0] === 1) {
        return "\n      int getOutputCoords() {\n        return int(gl_FragCoord.x);\n      }\n    ";
    }
    if (texShape[1] === 1) {
        return "\n      int getOutputCoords() {\n        return int(gl_FragCoord.y);\n      }\n    ";
    }
    return "\n    int getOutputCoords() {\n      ivec2 resTexRC = ivec2(gl_FragCoord.yx);\n      return resTexRC.x * " + texShape[1] + " + resTexRC.y;\n    }\n  ";
}
function getOutput3DCoords(shape, texShape) {
    var stride0 = shape[1] * shape[2];
    var stride1 = shape[2];
    return "\n    ivec3 getOutputCoords() {\n      ivec2 resTexRC = ivec2(gl_FragCoord.yx);\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n      int r = index / " + stride0 + ";\n      index -= r * " + stride0 + ";\n      int c = index / " + stride1 + ";\n      int d = index - c * " + stride1 + ";\n      return ivec3(r, c, d);\n    }\n  ";
}
function getOutput4DCoords(shape, texShape) {
    var stride2 = shape[3];
    var stride1 = shape[2] * stride2;
    var stride0 = shape[1] * stride1;
    return "\n    ivec4 getOutputCoords() {\n      ivec2 resTexRC = ivec2(gl_FragCoord.yx);\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n\n      int r = index / " + stride0 + ";\n      index -= r * " + stride0 + ";\n\n      int c = index / " + stride1 + ";\n      index -= c * " + stride1 + ";\n\n      int d = index / " + stride2 + ";\n      int d2 = index - d * " + stride2 + ";\n\n      return ivec4(r, c, d, d2);\n    }\n  ";
}
function getOutput2DCoords(shape, texShape) {
    if (util.arraysEqual(shape, texShape)) {
        return "\n      ivec2 getOutputCoords() {\n        return ivec2(gl_FragCoord.yx);\n      }\n    ";
    }
    if (shape[1] === 1) {
        return "\n      ivec2 getOutputCoords() {\n        ivec2 resTexRC = ivec2(gl_FragCoord.yx);\n        int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n        return ivec2(index, 0);\n      }\n    ";
    }
    if (shape[0] === 1) {
        return "\n      ivec2 getOutputCoords() {\n        ivec2 resTexRC = ivec2(gl_FragCoord.yx);\n        int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n        return ivec2(0, index);\n      }\n    ";
    }
    return "\n    ivec2 getOutputCoords() {\n      ivec2 resTexRC = ivec2(gl_FragCoord.yx);\n      int index = resTexRC.x * " + texShape[1] + " + resTexRC.y;\n      int r = index / " + shape[1] + ";\n      int c = index - r * " + shape[1] + ";\n      return ivec2(r, c);\n    }\n  ";
}
function getSamplerScalar(texName) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    return "\n    float " + funcName + "() {\n      return sample(" + texName + ", halfCR);\n    }\n  ";
}
function getSampler1D(texName, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var tR = texShape[0];
    var tC = texShape[1];
    if (texShape[0] === 1 && texShape[1] === 1) {
        return "\n      float " + funcName + "(int index) {\n        return sample(" + texName + ", halfCR);\n      }\n    ";
    }
    if (texShape[1] === 1) {
        return "\n      float " + funcName + "(int index) {\n        vec2 uv = vec2(0.5, (float(index) + 0.5) / " + tR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    if (texShape[0] === 1) {
        return "\n      float " + funcName + "(int index) {\n        vec2 uv = vec2((float(index) + 0.5) / " + tC + ".0, 0.5);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(int index) {\n      vec2 uv = UVfrom1D(" + tR + ", " + tC + ", index);\n      return sample(" + texName + ", uv);\n    }\n  ";
}
function getSampler3D(texName, shape, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var tR = texShape[0];
    var tC = texShape[1];
    var stride0 = shape[1] * shape[2];
    var stride1 = shape[2];
    if (tC === stride0) {
        return "\n      float " + funcName + "(int row, int col, int depth) {\n        int texR = row;\n        int texC = col * " + stride1 + " + depth;\n        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(" + tC + ".0, " + tR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(int row, int col, int depth) {\n      vec2 uv = UVfrom3D(" + tR + ", " + tC + ", " + stride0 + ", " + stride1 + ", row, col, depth);\n      return sample(" + texName + ", uv);\n    }\n  ";
}
function getSampler4D(texName, shape, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var tR = texShape[0];
    var tC = texShape[1];
    var stride2 = shape[3];
    var stride1 = shape[2] * stride2;
    var stride0 = shape[1] * stride1;
    if (tC === stride0) {
        return "\n      float " + funcName + "(int row, int col, int depth, int depth2) {\n        int texR = row;\n        int texC = col * " + stride1 + " + depth * " + stride2 + " + depth2;\n        vec2 uv = (vec2(texC, texR) + halfCR) / vec2(" + tC + ".0, " + tR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(int row, int col, int depth, int depth2) {\n      vec2 uv = UVfrom4D(" + tR + ", " + tC + ", " + stride0 + ", " + stride1 + ", " + stride2 + ",\n          row, col, depth, depth2);\n      return sample(" + texName + ", uv);\n    }\n  ";
}
function getSampler2D(texName, shape, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1);
    var tR = texShape[0];
    var tC = texShape[1];
    if (util.arraysEqual(shape, texShape)) {
        return "\n      float " + funcName + "(int row, int col) {\n        vec2 uv = (vec2(col, row) + halfCR) / vec2(" + tC + ".0, " + tR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    if (tC === 1) {
        if (shape[0] === 1) {
            return "\n        float " + funcName + "(int row, int col) {\n          vec2 uv = vec2(0.5, (float(col) + 0.5) / " + tR + ".0);\n          return sample(" + texName + ", uv);\n        }\n      ";
        }
        if (shape[1] === 1) {
            return "\n        float " + funcName + "(int row, int col) {\n          vec2 uv = vec2(0.5, (float(row) + 0.5) / " + tR + ".0);\n          return sample(" + texName + ", uv);\n        }\n      ";
        }
        return "\n      float " + funcName + "(int row, int col) {\n        int index = row * " + shape[1] + " + col;\n        vec2 uv = vec2(0.5, (float(index) + 0.5) / " + tR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    if (tR === 1) {
        return "\n      float " + funcName + "(int row, int col) {\n        int index = row * " + shape[1] + " + col;\n        vec2 uv = vec2((float(index) + 0.5) / " + tC + ".0, 0.5);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(int row, int col) {\n      vec2 uv = UVfrom2D(" + tR + ", " + tC + ", " + shape[1] + ", row, col);\n      return sample(" + texName + ", uv);\n    }\n  ";
}
function getSamplerFlat(texName, texShape) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1) + 'Flat';
    var tNumR = texShape[0];
    var tNumC = texShape[1];
    if (tNumC === 1 && tNumR === 1) {
        return "\n      float " + funcName + "(int index) {\n        return sample(" + texName + ", halfCR);\n      }\n    ";
    }
    if (tNumC === 1) {
        return "\n      float " + funcName + "(int index) {\n        vec2 uv = vec2(0.5, (float(index) + 0.5) / " + tNumR + ".0);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    if (tNumR === 1) {
        return "\n      float " + funcName + "(int index) {\n        vec2 uv = vec2((float(index) + 0.5) / " + tNumC + ".0, 0.5);\n        return sample(" + texName + ", uv);\n      }\n    ";
    }
    return "\n    float " + funcName + "(int index) {\n      int texR = index / " + tNumC + ";\n      int texC = index - texR * " + tNumC + ";\n      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(" + tNumC + ".0, " + tNumR + ".0);\n      return sample(" + texName + ", uv);\n    }\n  ";
}
function getSamplerAtOutputCoords(texName, inTexShape, outTexShape, broadcast) {
    var funcName = 'get' + texName.charAt(0).toUpperCase() + texName.slice(1) +
        'AtOutCoords';
    if (util.arraysEqual(inTexShape, outTexShape)) {
        return "\n      float " + funcName + "() {\n        return sample(" + texName + ", resultUV);\n      }\n    ";
    }
    var inSize = util.sizeFromShape(inTexShape);
    var broadcastSnippet = '';
    if (broadcast) {
        broadcastSnippet = "\n      int mainPart = index / " + inSize + ";\n      index -= mainPart * " + inSize + ";\n    ";
    }
    return "\n    float " + funcName + "() {\n      ivec2 resTexRC = ivec2(gl_FragCoord.yx);\n      int index = resTexRC.x * " + outTexShape[1] + " + resTexRC.y;\n      " + broadcastSnippet + "\n      int texR = index / " + inTexShape[1] + ";\n      int texC = index - texR * " + inTexShape[1] + ";\n      vec2 uv = (vec2(texC, texR) + halfCR) /\n                 vec2(" + inTexShape[1] + ".0, " + inTexShape[0] + ".0);\n      return sample(" + texName + ", uv);\n    }\n  ";
}

},{"../../util":79}],48:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function getUnpackedMatrixTextureShapeWidthHeight(rows, columns) {
    return [columns, rows];
}
exports.getUnpackedMatrixTextureShapeWidthHeight = getUnpackedMatrixTextureShapeWidthHeight;
function getUnpackedArraySizeFromMatrixSize(matrixSize, channelsPerTexture) {
    return matrixSize * channelsPerTexture;
}
exports.getUnpackedArraySizeFromMatrixSize = getUnpackedArraySizeFromMatrixSize;
function getColorMatrixTextureShapeWidthHeight(rows, columns) {
    return [columns * 4, rows];
}
exports.getColorMatrixTextureShapeWidthHeight = getColorMatrixTextureShapeWidthHeight;
function getMatrixSizeFromUnpackedArraySize(unpackedSize, channelsPerTexture) {
    if (unpackedSize % channelsPerTexture !== 0) {
        throw new Error('unpackedSize (' + unpackedSize + ') must be a multiple of ' +
            channelsPerTexture);
    }
    return unpackedSize / channelsPerTexture;
}
exports.getMatrixSizeFromUnpackedArraySize = getMatrixSizeFromUnpackedArraySize;
function encodeMatrixToUnpackedArray(matrix, unpackedArray, channelsPerTexture) {
    var requiredSize = getUnpackedArraySizeFromMatrixSize(matrix.length, channelsPerTexture);
    if (unpackedArray.length < requiredSize) {
        throw new Error('unpackedArray length (' + unpackedArray.length +
            ') must be >= ' + requiredSize);
    }
    var dst = 0;
    for (var src = 0; src < matrix.length; ++src) {
        unpackedArray[dst] = matrix[src];
        dst += channelsPerTexture;
    }
}
exports.encodeMatrixToUnpackedArray = encodeMatrixToUnpackedArray;
function decodeMatrixFromUnpackedArray(unpackedArray, matrix, channelsPerTexture) {
    var requiredSize = getMatrixSizeFromUnpackedArraySize(unpackedArray.length, channelsPerTexture);
    if (matrix.length < requiredSize) {
        throw new Error('matrix length (' + matrix.length + ') must be >= ' + requiredSize);
    }
    var dst = 0;
    for (var src = 0; src < unpackedArray.length; src += channelsPerTexture) {
        matrix[dst++] = unpackedArray[src];
    }
}
exports.decodeMatrixFromUnpackedArray = decodeMatrixFromUnpackedArray;
function getPackedMatrixTextureShapeWidthHeight(rows, columns) {
    return [Math.ceil(columns / 2), Math.ceil(rows / 2)];
}
exports.getPackedMatrixTextureShapeWidthHeight = getPackedMatrixTextureShapeWidthHeight;
function getPackedRGBAArraySizeFromMatrixShape(rows, columns) {
    var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), w = _a[0], h = _a[1];
    return w * h * 4;
}
exports.getPackedRGBAArraySizeFromMatrixShape = getPackedRGBAArraySizeFromMatrixShape;
function encodeMatrixToPackedRGBA(matrix, rows, columns, packedRGBA) {
    var requiredSize = getPackedRGBAArraySizeFromMatrixShape(rows, columns);
    if (packedRGBA.length < requiredSize) {
        throw new Error('packedRGBA length (' + packedRGBA.length +
            ') must be >= ' + requiredSize);
    }
    var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), textureWidth = _a[0], textureHeight = _a[1];
    var oddWidth = (columns % 2) === 1;
    var oddHeight = (rows % 2) === 1;
    var widthInFullBlocks = Math.floor(columns / 2);
    var heightInFullBlocks = Math.floor(rows / 2);
    {
        var dstStride = (oddWidth ? 4 : 0);
        var oneRow = columns;
        var dst = 0;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            var matrixSrcRow = (blockY * 2 * columns);
            for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
                var matrixSrcCol = blockX * 2;
                var src = matrixSrcRow + matrixSrcCol;
                packedRGBA[dst] = matrix[src];
                packedRGBA[dst + 1] = matrix[src + 1];
                packedRGBA[dst + 2] = matrix[src + oneRow];
                packedRGBA[dst + 3] = matrix[src + oneRow + 1];
                dst += 4;
            }
            dst += dstStride;
        }
    }
    if (oddWidth) {
        var src = columns - 1;
        var dst = (textureWidth - 1) * 4;
        var srcStride = 2 * columns;
        var dstStride = textureWidth * 4;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            packedRGBA[dst] = matrix[src];
            packedRGBA[dst + 2] = matrix[src + columns];
            src += srcStride;
            dst += dstStride;
        }
    }
    if (oddHeight) {
        var src = (rows - 1) * columns;
        var dst = (textureHeight - 1) * textureWidth * 4;
        for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
            packedRGBA[dst++] = matrix[src++];
            packedRGBA[dst++] = matrix[src++];
            dst += 2;
        }
    }
    if (oddWidth && oddHeight) {
        packedRGBA[packedRGBA.length - 4] = matrix[matrix.length - 1];
    }
    return packedRGBA;
}
exports.encodeMatrixToPackedRGBA = encodeMatrixToPackedRGBA;
function decodeMatrixFromPackedRGBA(packedRGBA, rows, columns, matrix) {
    var requiredSize = rows * columns;
    if (requiredSize < matrix.length) {
        throw new Error('matrix length (' + matrix.length + ') must be >= ' + requiredSize);
    }
    var oddWidth = (columns % 2) === 1;
    var oddHeight = (rows % 2) === 1;
    var widthInFullBlocks = Math.floor(columns / 2);
    var heightInFullBlocks = Math.floor(rows / 2);
    var _a = getPackedMatrixTextureShapeWidthHeight(rows, columns), textureWidth = _a[0], textureHeight = _a[1];
    {
        var srcStride = oddWidth ? 4 : 0;
        var dstStride = columns + (oddWidth ? 1 : 0);
        var src = 0;
        var dstRow1 = 0;
        var dstRow2 = columns;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
                matrix[dstRow1++] = packedRGBA[src++];
                matrix[dstRow1++] = packedRGBA[src++];
                matrix[dstRow2++] = packedRGBA[src++];
                matrix[dstRow2++] = packedRGBA[src++];
            }
            src += srcStride;
            dstRow1 += dstStride;
            dstRow2 += dstStride;
        }
    }
    if (oddWidth) {
        var src = (textureWidth - 1) * 4;
        var dst = columns - 1;
        var srcStride = textureWidth * 4;
        var dstStride = 2 * columns;
        for (var blockY = 0; blockY < heightInFullBlocks; ++blockY) {
            matrix[dst] = packedRGBA[src];
            matrix[dst + columns] = packedRGBA[src + 2];
            src += srcStride;
            dst += dstStride;
        }
    }
    if (oddHeight) {
        var src = (textureHeight - 1) * textureWidth * 4;
        var dst = (rows - 1) * columns;
        for (var blockX = 0; blockX < widthInFullBlocks; ++blockX) {
            matrix[dst++] = packedRGBA[src++];
            matrix[dst++] = packedRGBA[src++];
            src += 2;
        }
    }
    if (oddWidth && oddHeight) {
        matrix[matrix.length - 1] = packedRGBA[packedRGBA.length - 4];
    }
    return matrix;
}
exports.decodeMatrixFromPackedRGBA = decodeMatrixFromPackedRGBA;

},{}],49:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var TextureManager = (function () {
    function TextureManager(gpgpu) {
        this.gpgpu = gpgpu;
        this.numUsedTextures = 0;
        this.numFreeTextures = 0;
        this.freeTextures = {};
        this.logEnabled = false;
        this.usedTextureCount = {};
    }
    TextureManager.prototype.acquireTexture = function (shapeRC) {
        var shapeKey = getKeyFromTextureShape(shapeRC);
        if (!(shapeKey in this.freeTextures)) {
            this.freeTextures[shapeKey] = [];
        }
        if (!(shapeKey in this.usedTextureCount)) {
            this.usedTextureCount[shapeKey] = 0;
        }
        this.usedTextureCount[shapeKey]++;
        if (this.freeTextures[shapeKey].length > 0) {
            this.numFreeTextures--;
            this.numUsedTextures++;
            this.log();
            return this.freeTextures[shapeKey].shift();
        }
        this.numUsedTextures++;
        this.log();
        return this.gpgpu.createMatrixTexture(shapeRC[0], shapeRC[1]);
    };
    TextureManager.prototype.releaseTexture = function (texture, shape) {
        var shapeKey = getKeyFromTextureShape(shape);
        if (!(shapeKey in this.freeTextures)) {
            this.freeTextures[shapeKey] = [];
        }
        this.freeTextures[shapeKey].push(texture);
        this.numFreeTextures++;
        this.numUsedTextures--;
        this.usedTextureCount[shapeKey]--;
        this.log();
    };
    TextureManager.prototype.log = function () {
        if (!this.logEnabled) {
            return;
        }
        var total = this.numFreeTextures + this.numUsedTextures;
        console.log('Free/Used', this.numFreeTextures + ' / ' + this.numUsedTextures, "(" + total + ")");
    };
    TextureManager.prototype.getNumUsedTextures = function () {
        return this.numUsedTextures;
    };
    TextureManager.prototype.getNumFreeTextures = function () {
        return this.numFreeTextures;
    };
    TextureManager.prototype.dispose = function () {
        for (var shape in this.freeTextures) {
            if (this.freeTextures.hasOwnProperty(shape)) {
                for (var i = 0; i < this.freeTextures[shape].length; i++) {
                    this.gpgpu.deleteMatrixTexture(this.freeTextures[shape][i]);
                }
            }
        }
    };
    return TextureManager;
}());
exports.TextureManager = TextureManager;
function getKeyFromTextureShape(shapeRowsCol) {
    return shapeRowsCol[0] + '_' + shapeRowsCol[1];
}

},{}],50:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var UnaryOp;
(function (UnaryOp) {
    UnaryOp[UnaryOp["EXP"] = 0] = "EXP";
    UnaryOp[UnaryOp["LOG"] = 1] = "LOG";
    UnaryOp[UnaryOp["SQRT"] = 2] = "SQRT";
    UnaryOp[UnaryOp["NEG"] = 3] = "NEG";
    UnaryOp[UnaryOp["RELU"] = 4] = "RELU";
    UnaryOp[UnaryOp["SIGMOID"] = 5] = "SIGMOID";
    UnaryOp[UnaryOp["STEP"] = 6] = "STEP";
    UnaryOp[UnaryOp["SIN"] = 7] = "SIN";
    UnaryOp[UnaryOp["TANH"] = 8] = "TANH";
})(UnaryOp = exports.UnaryOp || (exports.UnaryOp = {}));
var UnaryOpProgram = (function () {
    function UnaryOpProgram(aShape, op) {
        this.variableNames = ['A'];
        this.outputShape = aShape;
        this.params = [op];
        this.userCode = "\n      void main() {\n        float v = getAAtOutCoords();\n        " + getOpSnippet(op) + "\n        setOutput(r);\n      }\n    ";
    }
    return UnaryOpProgram;
}());
exports.UnaryOpProgram = UnaryOpProgram;
var CHECK_NAN_SNIPPET = "\n  if (isNaN(v)) {\n    setOutput(v);\n    return;\n  }\n";
function getOpSnippet(op) {
    switch (op) {
        case UnaryOp.EXP:
            return 'float r = exp(v);';
        case UnaryOp.LOG:
            return 'float r = log(v);';
        case UnaryOp.SQRT:
            return CHECK_NAN_SNIPPET +
                'float r = sqrt(v);';
        case UnaryOp.NEG:
            return 'float r = -v;';
        case UnaryOp.RELU:
            return 'float r = (v < 0.0) ? 0.0 : v;';
        case UnaryOp.SIGMOID:
            return 'float r = 1.0 / (1.0 + exp(-1.0 * v));';
        case UnaryOp.STEP:
            return 'float r = (v == v) ? (v > 0.0 ? 1.0 : 0.0) : v;';
        case UnaryOp.SIN:
            return CHECK_NAN_SNIPPET +
                'float r = sin(v);';
        case UnaryOp.TANH:
            return "float e2x = exp(-2.0 * abs(v));\n              float r = sign(v) * (1.0 - e2x) / (1.0 + e2x);";
        default:
            throw Error('Unrecognized unary op type ' + op);
    }
}

},{}],51:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var USE_WEBGL2_WHEN_AVAILABLE = true;
var WEBGL2_ENABLED = null;
var MAX_TEXTURE_SIZE = null;
var util = require("../../util");
function createWebGLRenderingContext(attributes) {
    var canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    return createWebGLRenderingContextFromCanvas(canvas, attributes);
}
exports.createWebGLRenderingContext = createWebGLRenderingContext;
function preferWebGL1() {
    USE_WEBGL2_WHEN_AVAILABLE = false;
    WEBGL2_ENABLED = null;
}
exports.preferWebGL1 = preferWebGL1;
function preferWebGL2() {
    USE_WEBGL2_WHEN_AVAILABLE = true;
    WEBGL2_ENABLED = null;
}
exports.preferWebGL2 = preferWebGL2;
function isWebGL2Enabled() {
    if (!USE_WEBGL2_WHEN_AVAILABLE) {
        return false;
    }
    if (WEBGL2_ENABLED == null) {
        var tempCanvas = document.createElement('canvas');
        var gl = tempCanvas.getContext('webgl2');
        if (gl != null) {
            WEBGL2_ENABLED = true;
            var loseContextExtension = getExtensionOrThrow(gl, 'WEBGL_lose_context');
            loseContextExtension.loseContext();
        }
        else {
            WEBGL2_ENABLED = false;
        }
    }
    return WEBGL2_ENABLED;
}
exports.isWebGL2Enabled = isWebGL2Enabled;
function createWebGLRenderingContextFromCanvas(canvas, attributes) {
    var gl;
    if (isWebGL2Enabled()) {
        gl = canvas.getContext('webgl2', attributes);
    }
    else {
        gl =
            (canvas.getContext('webgl', attributes) ||
                canvas.getContext('experimental-webgl', attributes));
    }
    if (gl == null) {
        throw new Error('This browser does not support WebGL.');
    }
    return gl;
}
exports.createWebGLRenderingContextFromCanvas = createWebGLRenderingContextFromCanvas;
function callAndCheck(gl, func) {
    var returnValue = func();
    checkWebGLError(gl);
    return returnValue;
}
exports.callAndCheck = callAndCheck;
var webGLDebugErrorCheckingEnabled = false;
function enableDebugWebGLErrorChecking(enabled) {
    webGLDebugErrorCheckingEnabled = enabled;
}
exports.enableDebugWebGLErrorChecking = enableDebugWebGLErrorChecking;
function checkWebGLError(gl) {
    if (webGLDebugErrorCheckingEnabled) {
        var error = gl.getError();
        if (error !== gl.NO_ERROR) {
            throw new Error('WebGL Error: ' + getWebGLErrorMessage(gl, error));
        }
    }
}
exports.checkWebGLError = checkWebGLError;
function getWebGLErrorMessage(gl, status) {
    switch (status) {
        case gl.NO_ERROR:
            return 'NO_ERROR';
        case gl.INVALID_ENUM:
            return 'INVALID_ENUM';
        case gl.INVALID_VALUE:
            return 'INVALID_VALUE';
        case gl.INVALID_OPERATION:
            return 'INVALID_OPERATION';
        case gl.INVALID_FRAMEBUFFER_OPERATION:
            return 'INVALID_FRAMEBUFFER_OPERATION';
        case gl.OUT_OF_MEMORY:
            return 'OUT_OF_MEMORY';
        case gl.CONTEXT_LOST_WEBGL:
            return 'CONTEXT_LOST_WEBGL';
        default:
            return 'Unknown error code ' + status;
    }
}
exports.getWebGLErrorMessage = getWebGLErrorMessage;
function getExtensionOrThrow(gl, extensionName) {
    return throwIfNull(gl, function () { return gl.getExtension(extensionName); }, 'Extension "' + extensionName + '" not supported on this browser.');
}
exports.getExtensionOrThrow = getExtensionOrThrow;
function createVertexShader(gl, vertexShaderSource) {
    var vertexShader = throwIfNull(gl, function () { return gl.createShader(gl.VERTEX_SHADER); }, 'Unable to create vertex WebGLShader.');
    callAndCheck(gl, function () { return gl.shaderSource(vertexShader, vertexShaderSource); });
    callAndCheck(gl, function () { return gl.compileShader(vertexShader); });
    if (gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS) === false) {
        console.log(gl.getShaderInfoLog(vertexShader));
        throw new Error('Failed to compile vertex shader.');
    }
    return vertexShader;
}
exports.createVertexShader = createVertexShader;
function createFragmentShader(gl, fragmentShaderSource) {
    var fragmentShader = throwIfNull(gl, function () { return gl.createShader(gl.FRAGMENT_SHADER); }, 'Unable to create fragment WebGLShader.');
    callAndCheck(gl, function () { return gl.shaderSource(fragmentShader, fragmentShaderSource); });
    callAndCheck(gl, function () { return gl.compileShader(fragmentShader); });
    if (gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS) === false) {
        console.log(fragmentShaderSource);
        console.log(gl.getShaderInfoLog(fragmentShader));
        throw new Error('Failed to compile fragment shader.');
    }
    return fragmentShader;
}
exports.createFragmentShader = createFragmentShader;
function createProgram(gl) {
    return throwIfNull(gl, function () { return gl.createProgram(); }, 'Unable to create WebGLProgram.');
}
exports.createProgram = createProgram;
function linkProgram(gl, program) {
    callAndCheck(gl, function () { return gl.linkProgram(program); });
    if (gl.getProgramParameter(program, gl.LINK_STATUS) === false) {
        console.log(gl.getProgramInfoLog(program));
        throw new Error('Failed to link vertex and fragment shaders.');
    }
}
exports.linkProgram = linkProgram;
function validateProgram(gl, program) {
    callAndCheck(gl, function () { return gl.validateProgram(program); });
    if (gl.getProgramParameter(program, gl.VALIDATE_STATUS) === false) {
        console.log(gl.getProgramInfoLog(program));
        throw new Error('Shader program validation failed.');
    }
}
exports.validateProgram = validateProgram;
function createStaticVertexBuffer(gl, data) {
    var buffer = throwIfNull(gl, function () { return gl.createBuffer(); }, 'Unable to create WebGLBuffer');
    callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, buffer); });
    callAndCheck(gl, function () { return gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); });
    return buffer;
}
exports.createStaticVertexBuffer = createStaticVertexBuffer;
function createStaticIndexBuffer(gl, data) {
    var buffer = throwIfNull(gl, function () { return gl.createBuffer(); }, 'Unable to create WebGLBuffer');
    callAndCheck(gl, function () { return gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffer); });
    callAndCheck(gl, function () { return gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data, gl.STATIC_DRAW); });
    return buffer;
}
exports.createStaticIndexBuffer = createStaticIndexBuffer;
function queryMaxTextureSize(gl) {
    if (MAX_TEXTURE_SIZE != null) {
        return MAX_TEXTURE_SIZE;
    }
    MAX_TEXTURE_SIZE =
        callAndCheck(gl, function () { return gl.getParameter(gl.MAX_TEXTURE_SIZE); });
    return MAX_TEXTURE_SIZE;
}
exports.queryMaxTextureSize = queryMaxTextureSize;
function getChannelsPerTexture() {
    if (isWebGL2Enabled()) {
        return 1;
    }
    return 4;
}
exports.getChannelsPerTexture = getChannelsPerTexture;
function createTexture(gl) {
    return throwIfNull(gl, function () { return gl.createTexture(); }, 'Unable to create WebGLTexture.');
}
exports.createTexture = createTexture;
function validateTextureSize(gl, width, height) {
    var maxTextureSize = queryMaxTextureSize(gl);
    if ((width <= 0) || (height <= 0)) {
        var requested = '[' + width + 'x' + height + ']';
        throw new Error('Requested texture size ' + requested + ' is invalid.');
    }
    if ((width > maxTextureSize) || (height > maxTextureSize)) {
        var requested = '[' + width + 'x' + height + ']';
        var max = '[' + maxTextureSize + 'x' + maxTextureSize + ']';
        throw new Error('Requested texture size ' + requested +
            ' greater than WebGL maximum on this browser / GPU ' + max + '.');
    }
}
exports.validateTextureSize = validateTextureSize;
function createFramebuffer(gl) {
    return throwIfNull(gl, function () { return gl.createFramebuffer(); }, 'Unable to create WebGLFramebuffer.');
}
exports.createFramebuffer = createFramebuffer;
function bindVertexBufferToProgramAttribute(gl, program, attribute, buffer, arrayEntriesPerItem, itemStrideInBytes, itemOffsetInBytes) {
    var loc = gl.getAttribLocation(program, attribute);
    if (loc === -1) {
        var error = new Error('Unable to get attribute "' + attribute + '" on WebGLProgram.');
        error.namedVertexAttributeNotFound = attribute;
        throw error;
    }
    callAndCheck(gl, function () { return gl.bindBuffer(gl.ARRAY_BUFFER, buffer); });
    callAndCheck(gl, function () { return gl.vertexAttribPointer(loc, arrayEntriesPerItem, gl.FLOAT, false, itemStrideInBytes, itemOffsetInBytes); });
    callAndCheck(gl, function () { return gl.enableVertexAttribArray(loc); });
}
exports.bindVertexBufferToProgramAttribute = bindVertexBufferToProgramAttribute;
function bindTextureUnit(gl, texture, textureUnit) {
    validateTextureUnit(gl, textureUnit);
    callAndCheck(gl, function () { return gl.activeTexture(gl.TEXTURE0 + textureUnit); });
    callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, texture); });
}
exports.bindTextureUnit = bindTextureUnit;
function unbindTextureUnit(gl, textureUnit) {
    validateTextureUnit(gl, textureUnit);
    callAndCheck(gl, function () { return gl.activeTexture(gl.TEXTURE0 + textureUnit); });
    callAndCheck(gl, function () { return gl.bindTexture(gl.TEXTURE_2D, null); });
}
exports.unbindTextureUnit = unbindTextureUnit;
function getProgramUniformLocationOrThrow(gl, program, uniformName) {
    return throwIfNull(gl, function () { return gl.getUniformLocation(program, uniformName); }, 'uniform "' + uniformName + '" not present in program.');
}
exports.getProgramUniformLocationOrThrow = getProgramUniformLocationOrThrow;
function bindTextureToProgramUniformSampler(gl, program, texture, uniformSamplerName, textureUnit) {
    callAndCheck(gl, function () { return bindTextureUnit(gl, texture, textureUnit); });
    var samplerLocation = getProgramUniformLocationOrThrow(gl, program, uniformSamplerName);
    callAndCheck(gl, function () { return gl.uniform1i(samplerLocation, textureUnit); });
}
exports.bindTextureToProgramUniformSampler = bindTextureToProgramUniformSampler;
function bindCanvasToFramebuffer(gl) {
    callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, null); });
    callAndCheck(gl, function () { return gl.viewport(0, 0, gl.canvas.width, gl.canvas.height); });
    callAndCheck(gl, function () { return gl.scissor(0, 0, gl.canvas.width, gl.canvas.height); });
}
exports.bindCanvasToFramebuffer = bindCanvasToFramebuffer;
function bindColorTextureToFramebuffer(gl, texture, framebuffer) {
    callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer); });
    callAndCheck(gl, function () { return gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0); });
}
exports.bindColorTextureToFramebuffer = bindColorTextureToFramebuffer;
function unbindColorTextureFromFramebuffer(gl, framebuffer) {
    callAndCheck(gl, function () { return gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer); });
    callAndCheck(gl, function () { return gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0); });
}
exports.unbindColorTextureFromFramebuffer = unbindColorTextureFromFramebuffer;
function validateFramebuffer(gl) {
    var status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
        throw new Error('Error binding framebuffer: ' + getFramebufferErrorMessage(gl, status));
    }
}
exports.validateFramebuffer = validateFramebuffer;
function getFramebufferErrorMessage(gl, status) {
    switch (status) {
        case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            return 'FRAMEBUFFER_INCOMPLETE_ATTACHMENT';
        case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
            return 'FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT';
        case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
            return 'FRAMEBUFFER_INCOMPLETE_DIMENSIONS';
        case gl.FRAMEBUFFER_UNSUPPORTED:
            return 'FRAMEBUFFER_UNSUPPORTED';
        default:
            return 'unknown error ' + status;
    }
}
exports.getFramebufferErrorMessage = getFramebufferErrorMessage;
function throwIfNull(gl, returnTOrNull, failureMessage) {
    var tOrNull = callAndCheck(gl, function () { return returnTOrNull(); });
    if (tOrNull == null) {
        throw new Error(failureMessage);
    }
    return tOrNull;
}
function validateTextureUnit(gl, textureUnit) {
    var maxTextureUnit = gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS - 1;
    var glTextureUnit = textureUnit + gl.TEXTURE0;
    if (glTextureUnit < gl.TEXTURE0 || glTextureUnit > maxTextureUnit) {
        var textureUnitRange = '[gl.TEXTURE0, gl.TEXTURE' + maxTextureUnit + ']';
        throw new Error('textureUnit must be in ' + textureUnitRange + '.');
    }
}
function getTextureShapeFromLogicalShape(gl, logShape, preferredTexShape) {
    var maxTexSize = queryMaxTextureSize(gl);
    var size = util.sizeFromShape(logShape);
    if (preferredTexShape != null) {
        var sizePreferred = util.sizeFromShape(preferredTexShape);
        util.assert(size === sizePreferred, "Size of shape (" + size + ") must match size of " +
            ("preferredShape (" + sizePreferred + ")"));
        if (preferredTexShape[0] <= maxTexSize &&
            preferredTexShape[1] <= maxTexSize) {
            return preferredTexShape;
        }
    }
    if (logShape.length <= 1 && size <= maxTexSize) {
        return [size, 1];
    }
    else if (logShape.length === 2 && logShape[0] <= maxTexSize &&
        logShape[1] <= maxTexSize) {
        return logShape;
    }
    else if (logShape.length === 3 && logShape[0] <= maxTexSize &&
        logShape[1] * logShape[2] <= maxTexSize) {
        return [logShape[0], logShape[1] * logShape[2]];
    }
    else if (logShape.length === 4 && logShape[0] <= maxTexSize &&
        logShape[1] * logShape[2] * logShape[3] <= maxTexSize) {
        return [logShape[0], logShape[1] * logShape[2] * logShape[3]];
    }
    else {
        return util.sizeToSquarishShape(size);
    }
}
exports.getTextureShapeFromLogicalShape = getTextureShapeFromLogicalShape;

},{"../../util":79}],52:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var sgd_optimizer_1 = require("./sgd_optimizer");
var tensor_array_map_1 = require("./tensor_array_map");
var MomentumOptimizer = (function (_super) {
    __extends(MomentumOptimizer, _super);
    function MomentumOptimizer(learningRate, momentum, specifiedVariableList) {
        var _this = _super.call(this, learningRate, specifiedVariableList) || this;
        _this.learningRate = learningRate;
        _this.momentum = momentum;
        _this.variableVelocities = new tensor_array_map_1.TensorArrayMap();
        return _this;
    }
    MomentumOptimizer.prototype.beforeBatch = function (math, batchSize, runtime, activationArrayMap, gradientArrayMap) {
        var _this = this;
        _super.prototype.beforeBatch.call(this, math, batchSize, runtime, activationArrayMap, gradientArrayMap);
        this.m = ndarray_1.Scalar.new(this.momentum);
        if (this.variableVelocities.size() === 0) {
            this.variableNodes.forEach(function (node) {
                _this.variableVelocities.set(node.output, ndarray_1.NDArray.zeros(node.output.shape));
            });
        }
    };
    MomentumOptimizer.prototype.afterBatch = function (math, batchSize, runtime, activationArrayMap, gradientArrayMap) {
        var _this = this;
        math.scope(function (keep) {
            _this.variableNodes.forEach(function (node) {
                var oldVariable = activationArrayMap.get(node.output);
                var gradient = _this.variableGradients.get(node.output);
                var oldVelocity = _this.variableVelocities.get(node.output);
                var velocity = math.scaledArrayAdd(_this.m, oldVelocity, _this.one, gradient);
                var variable = math.scaledArrayAdd(_this.c, velocity, _this.one, oldVariable);
                _this.variableVelocities.set(node.output, keep(velocity));
                activationArrayMap.set(node.output, keep(variable));
                node.data = variable;
                oldVariable.dispose();
                oldVelocity.dispose();
            });
        });
        this.variableGradients.dispose();
        this.variableGradients = new tensor_array_map_1.TensorArrayMap();
    };
    MomentumOptimizer.prototype.dispose = function () {
        if (this.c != null) {
            this.c.dispose();
        }
        if (this.m != null) {
            this.m.dispose();
        }
        this.one.dispose();
        this.variableVelocities.dispose();
    };
    MomentumOptimizer.prototype.setMomentum = function (momentum) {
        this.momentum = momentum;
    };
    return MomentumOptimizer;
}(sgd_optimizer_1.SGDOptimizer));
exports.MomentumOptimizer = MomentumOptimizer;

},{"./math/ndarray":26,"./sgd_optimizer":77,"./tensor_array_map":78}],53:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var graph_1 = require("./graph");
var graph_util = require("./graph_util");
var add_1 = require("./ops/add");
var argmax_1 = require("./ops/argmax");
var argmaxequals_1 = require("./ops/argmaxequals");
var concat3d_1 = require("./ops/concat3d");
var convolution_1 = require("./ops/convolution");
var divide_1 = require("./ops/divide");
var element_wise_activation_1 = require("./ops/element_wise_activation");
var element_wise_cost_1 = require("./ops/element_wise_cost");
var exp_1 = require("./ops/exp");
var linear_combination_1 = require("./ops/linear_combination");
var log_1 = require("./ops/log");
var matmul_1 = require("./ops/matmul");
var max_pool_1 = require("./ops/max_pool");
var multiply_1 = require("./ops/multiply");
var reduce_sum_1 = require("./ops/reduce_sum");
var reshape_1 = require("./ops/reshape");
var softmax_1 = require("./ops/softmax");
var subtract_1 = require("./ops/subtract");
function emitFromGraphNodes(nodes) {
    var ops = [];
    nodes.forEach(function (node) { return Array.prototype.push.apply(ops, emitOpFromNode(node)); });
    return ops;
}
exports.emitFromGraphNodes = emitFromGraphNodes;
function emitOpFromNode(node) {
    if (node instanceof graph_1.ReshapeNode) {
        return [new reshape_1.Reshape(node.inputs[graph_1.ReshapeNode.X], node.output)];
    }
    else if (node instanceof graph_1.MatMulNode) {
        var x1 = node.inputs[graph_1.MatMulNode.X1];
        var x2 = node.inputs[graph_1.MatMulNode.X2];
        return [new matmul_1.MatMul(x1, x2, node.output)];
    }
    else if (node instanceof graph_1.Convolution2DNode) {
        var w = node.inputs[graph_1.Convolution2DNode.W];
        var x = node.inputs[graph_1.Convolution2DNode.X];
        var b = node.inputs[graph_1.Convolution2DNode.B];
        return [new convolution_1.Convolution2D(w, x, b, node.output, node.fieldSize, node.outputDepth, node.stride, node.zeroPad)];
    }
    else if (node instanceof graph_1.MaxPoolNode) {
        var x = node.inputs[graph_1.MaxPoolNode.X];
        return [new max_pool_1.MaxPool(x, node.output, node.fieldSize, node.stride, node.zeroPad)];
    }
    else if (node instanceof graph_1.ExpNode) {
        return [new exp_1.Exp(node.inputs[graph_1.ExpNode.X], node.output)];
    }
    else if (node instanceof graph_1.LogNode) {
        return [new log_1.Log(node.inputs[graph_1.LogNode.X], node.output)];
    }
    else if (node instanceof graph_1.ReLUNode) {
        return [new element_wise_activation_1.ReLU(node.inputs[graph_1.ReLUNode.X], node.output)];
    }
    else if (node instanceof graph_1.TanHNode) {
        return [new element_wise_activation_1.TanH(node.inputs[graph_1.TanHNode.X], node.output)];
    }
    else if (node instanceof graph_1.SigmoidNode) {
        return [new element_wise_activation_1.Sigmoid(node.inputs[graph_1.SigmoidNode.X], node.output)];
    }
    else if (node instanceof graph_1.SoftmaxCrossEntropyCostNode) {
        var x = node.inputs[graph_1.SoftmaxCrossEntropyCostNode.X];
        var target = node.inputs[graph_1.SoftmaxCrossEntropyCostNode.TARGET];
        return [new softmax_1.SoftmaxCrossEntropyCost(x, target, node.output)];
    }
    else if (node instanceof graph_1.SoftmaxNode) {
        return [new softmax_1.Softmax(node.inputs[graph_1.SoftmaxNode.X], node.output)];
    }
    else if (node instanceof graph_1.MeanSquaredCostNode) {
        var label = node.inputs[graph_1.MeanSquaredCostNode.LABEL];
        var prediction = node.inputs[graph_1.MeanSquaredCostNode.PREDICTION];
        return [new element_wise_cost_1.MeanSquaredCost(label, prediction, node.output)];
    }
    else if (node instanceof graph_1.ArgMaxEqualsNode) {
        return [new argmaxequals_1.ArgMaxEquals(node.inputs[graph_1.ArgMaxEqualsNode.X1], node.inputs[graph_1.ArgMaxEqualsNode.X2], node.output)];
    }
    else if (node instanceof graph_1.ArgMaxNode) {
        return [new argmax_1.ArgMax(node.x, node.output)];
    }
    else if (node instanceof graph_1.FusedLinearCombinationNode) {
        return [new linear_combination_1.LinearCombination(node.inputs[graph_1.FusedLinearCombinationNode.T1], node.inputs[graph_1.FusedLinearCombinationNode.T2], node.inputs[graph_1.FusedLinearCombinationNode.C1], node.inputs[graph_1.FusedLinearCombinationNode.C2], node.output)];
    }
    else if (node instanceof graph_1.Concat3DNode) {
        return [new concat3d_1.Concat3D(node.inputs[graph_1.Concat3DNode.X1], node.inputs[graph_1.Concat3DNode.X2], node.axis, node.output)];
    }
    else if (node instanceof graph_1.SquareNode) {
        return [new element_wise_activation_1.Square(node.inputs[graph_1.SquareNode.X], node.output)];
    }
    else if (node instanceof graph_1.AddNode) {
        return [new add_1.Add(node.inputs[graph_1.AddNode.T1], node.inputs[graph_1.AddNode.T2], node.output)];
    }
    else if (node instanceof graph_1.SubtractNode) {
        return [new subtract_1.Subtract(node.inputs[graph_1.SubtractNode.T1], node.inputs[graph_1.SubtractNode.T2], node.output)];
    }
    else if (node instanceof graph_1.MultiplyNode) {
        return [new multiply_1.Multiply(node.inputs[graph_1.MultiplyNode.T1], node.inputs[graph_1.MultiplyNode.T2], node.output)];
    }
    else if (node instanceof graph_1.DivideNode) {
        return [new divide_1.Divide(node.inputs[graph_1.DivideNode.T1], node.inputs[graph_1.DivideNode.T2], node.output)];
    }
    else if (node instanceof graph_1.ReduceSumNode) {
        return [new reduce_sum_1.ReduceSum(node.inputs[graph_1.ReduceSumNode.X], node.output)];
    }
    else if (graph_util.isInputNode(node)) {
        return [];
    }
    else {
        throw Error('Unsupported node type: ' + node.constructor.name);
    }
}

},{"./graph":11,"./graph_util":14,"./ops/add":54,"./ops/argmax":55,"./ops/argmaxequals":56,"./ops/concat3d":57,"./ops/convolution":58,"./ops/divide":59,"./ops/element_wise_activation":60,"./ops/element_wise_cost":61,"./ops/exp":62,"./ops/linear_combination":63,"./ops/log":64,"./ops/matmul":65,"./ops/max_pool":66,"./ops/multiply":67,"./ops/reduce_sum":69,"./ops/reshape":70,"./ops/softmax":71,"./ops/subtract":72}],54:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var ndarray_1 = require("../math/ndarray");
var util = require("../util");
var op_1 = require("./op");
var Add = (function (_super) {
    __extends(Add, _super);
    function Add(x1Tensor, x2Tensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        util.assert(util.sizeFromShape(x1Tensor.shape) === 1 ||
            util.sizeFromShape(x2Tensor.shape) === 1 ||
            util.arraysEqual(x1Tensor.shape, x2Tensor.shape), 'One of t1 or t2 must be a scalar, or t1 and t2 must have ' +
            'the same shape');
        return _this;
    }
    Add.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var result;
            if (util.isScalarShape(x1.shape)) {
                result = math.scalarPlusArray(x1, x2);
            }
            else if (util.isScalarShape(x2.shape)) {
                result = math.scalarPlusArray(x2, x1);
            }
            else {
                result = math.add(x1, x2);
            }
            inferenceArrays.set(_this.yTensor, keep(result));
        });
    };
    Add.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function () {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                if (util.isScalarShape(_this.x1Tensor.shape)) {
                    var sum = math.sum(dy);
                    if (_this.dySizeScalar == null) {
                        _this.dySizeScalar = ndarray_1.Scalar.new(dy.size);
                    }
                    gradientArrays.add(_this.x1Tensor, math.divide(sum, _this.dySizeScalar));
                }
                else {
                    gradientArrays.add(_this.x1Tensor, dy);
                }
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                if (util.isScalarShape(_this.x2Tensor.shape)) {
                    var sum = math.sum(dy);
                    if (_this.dySizeScalar == null) {
                        _this.dySizeScalar = ndarray_1.Scalar.new(dy.size);
                    }
                    gradientArrays.add(_this.x2Tensor, math.divide(sum, _this.dySizeScalar));
                }
                else {
                    gradientArrays.add(_this.x2Tensor, dy);
                }
            }
        });
    };
    Add.prototype.dispose = function () {
        if (this.dySizeScalar != null) {
            this.dySizeScalar.dispose();
        }
    };
    return Add;
}(op_1.Operation));
exports.Add = Add;

},{"../graph_util":14,"../math/ndarray":26,"../util":79,"./op":68}],55:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var op_1 = require("./op");
var ArgMax = (function (_super) {
    __extends(ArgMax, _super);
    function ArgMax(xTensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        return _this;
    }
    ArgMax.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.argMax(x)));
        });
    };
    ArgMax.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        throw new Error('ArgMax backprop unimplemented');
    };
    return ArgMax;
}(op_1.Operation));
exports.ArgMax = ArgMax;

},{"./op":68}],56:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var op_1 = require("./op");
var ArgMaxEquals = (function (_super) {
    __extends(ArgMaxEquals, _super);
    function ArgMaxEquals(x1Tensor, x2Tensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        return _this;
    }
    ArgMaxEquals.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.argMaxEquals(x1, x2)));
        });
    };
    ArgMaxEquals.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        throw new Error('ArgMaxEquals backprop unimplemented');
    };
    return ArgMaxEquals;
}(op_1.Operation));
exports.ArgMaxEquals = ArgMaxEquals;

},{"./op":68}],57:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var concat3d_util = require("../math/concat3d_util");
var op_1 = require("./op");
var Concat3D = (function (_super) {
    __extends(Concat3D, _super);
    function Concat3D(x1Tensor, x2Tensor, axis, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.axis = axis;
        _this.yTensor = yTensor;
        concat3d_util.assertConcat3DShapesMatch(x1Tensor.shape, x2Tensor.shape, axis);
        return _this;
    }
    Concat3D.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var concatResult = math.concat3D(x1, x2, _this.axis);
            inferenceArrays.set(_this.yTensor, keep(concatResult));
        });
    };
    Concat3D.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        throw new Error('Concat3D backprop not implemented.');
    };
    return Concat3D;
}(op_1.Operation));
exports.Concat3D = Concat3D;

},{"../math/concat3d_util":19,"./op":68}],58:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../math/conv_util");
var util = require("../util");
var op_1 = require("./op");
var Convolution2D = (function (_super) {
    __extends(Convolution2D, _super);
    function Convolution2D(wTensor, xTensor, bTensor, yTensor, fieldSize, outputDepth, stride, zeroPad) {
        if (stride === void 0) { stride = 1; }
        var _this = _super.call(this) || this;
        _this.wTensor = wTensor;
        _this.xTensor = xTensor;
        _this.bTensor = bTensor;
        _this.yTensor = yTensor;
        _this.fieldSize = fieldSize;
        _this.outputDepth = outputDepth;
        _this.stride = stride;
        _this.assertWeightsShape(wTensor.shape);
        _this.zeroPad = zeroPad != null ?
            zeroPad :
            conv_util.computeDefaultPad(_this.xTensor.shape, _this.fieldSize, _this.stride);
        util.assert(util.isInt(_this.zeroPad), "The zero padding (" + _this.zeroPad + ") must be an integer. Change the " +
            "stride and/or zero pad parameters");
        return _this;
    }
    Convolution2D.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var weights = inferenceArrays.get(this.wTensor);
        var biases = inferenceArrays.get(this.bTensor);
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.conv2d(x, weights, biases, _this.stride, _this.zeroPad)));
        });
    };
    Convolution2D.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var weights = inferenceArrays.get(this.wTensor);
        var x = inferenceArrays.get(this.xTensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function () {
            var _a = math.conv2dBackProp(x, dy, weights, _this.stride, _this.zeroPad), dw = _a.dw, db = _a.db, dx = _a.dx;
            gradientArrays.add(_this.wTensor, dw);
            gradientArrays.add(_this.bTensor, db);
            gradientArrays.add(_this.xTensor, dx);
        });
    };
    Convolution2D.prototype.assertWeightsShape = function (weightsShape) {
        util.assert(weightsShape[0] === this.fieldSize &&
            weightsShape[1] === this.fieldSize &&
            weightsShape[2] === this.xTensor.shape[2] &&
            weightsShape[3] === this.outputDepth, "weights must be of shape [" + this.fieldSize + "," + this.fieldSize + "," +
            (this.xTensor.shape[2] + "," + this.outputDepth + "] but they are of") +
            ("shape [" + weightsShape + "]"));
    };
    return Convolution2D;
}(op_1.Operation));
exports.Convolution2D = Convolution2D;

},{"../math/conv_util":20,"../util":79,"./op":68}],59:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var util = require("../util");
var op_1 = require("./op");
var Divide = (function (_super) {
    __extends(Divide, _super);
    function Divide(x1Tensor, x2Tensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        util.assert(util.sizeFromShape(x1Tensor.shape) === 1 ||
            util.sizeFromShape(x2Tensor.shape) === 1 ||
            util.arraysEqual(x1Tensor.shape, x2Tensor.shape), 'One of t1 or t2 must be a scalar, or t1 and t2 must have ' +
            'the same shape');
        return _this;
    }
    Divide.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var t1 = inferenceArrays.get(this.x1Tensor);
        var t2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var result;
            if (util.isScalarShape(t1.shape)) {
                result = math.scalarDividedByArray(t1, t2);
            }
            else if (util.isScalarShape(t2.shape)) {
                result = math.arrayDividedByScalar(t1, t2);
            }
            else {
                result = math.divide(t1, t2);
            }
            inferenceArrays.set(_this.yTensor, keep(result));
        });
    };
    Divide.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        var dy = gradientArrays.get(this.yTensor);
        var x1IsScalar = util.isScalarShape(x1.shape);
        var x2IsScalar = util.isScalarShape(x2.shape);
        math.scope(function () {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                if (x1IsScalar) {
                    var div = math.divide(dy, x2);
                    gradientArrays.add(_this.x1Tensor, math.sum(div));
                    div.dispose();
                }
                else if (x2IsScalar) {
                    gradientArrays.add(_this.x1Tensor, math.arrayDividedByScalar(dy, x2));
                }
                else {
                    gradientArrays.add(_this.x1Tensor, math.divide(dy, x2));
                }
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                var x2Squared = math.elementWiseMul(x2, x2);
                var x1OverX2Squared = void 0;
                if (x2IsScalar) {
                    x1OverX2Squared = math.arrayDividedByScalar(x1, x2Squared);
                }
                else if (x1IsScalar) {
                    x1OverX2Squared = math.scalarDividedByArray(x1, x2Squared);
                }
                else {
                    x1OverX2Squared = math.divide(x1, x2Squared);
                }
                var dx2 = math.neg(x1OverX2Squared);
                var dyTimesDerivative = math.elementWiseMul(dy, dx2);
                if (x2IsScalar) {
                    gradientArrays.add(_this.x2Tensor, math.sum(dyTimesDerivative));
                }
                else {
                    gradientArrays.add(_this.x2Tensor, dyTimesDerivative);
                }
            }
        });
    };
    return Divide;
}(op_1.Operation));
exports.Divide = Divide;

},{"../graph_util":14,"../util":79,"./op":68}],60:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var activation_functions_1 = require("../math/activation_functions");
var op_1 = require("./op");
var ElementWiseActivation = (function (_super) {
    __extends(ElementWiseActivation, _super);
    function ElementWiseActivation(xTensor, yTensor, func) {
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        _this.func = func;
        return _this;
    }
    ElementWiseActivation.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(_this.func.output(math, x)));
        });
    };
    ElementWiseActivation.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        var y = inferenceArrays.get(this.yTensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function () {
            var dydx = _this.func.der(math, x, y);
            gradientArrays.add(_this.xTensor, math.elementWiseMul(dy, dydx));
            dydx.dispose();
        });
    };
    return ElementWiseActivation;
}(op_1.Operation));
exports.ElementWiseActivation = ElementWiseActivation;
var ReLU = (function (_super) {
    __extends(ReLU, _super);
    function ReLU(xTensor, yTensor) {
        return _super.call(this, xTensor, yTensor, new activation_functions_1.ReLUFunc()) || this;
    }
    return ReLU;
}(ElementWiseActivation));
exports.ReLU = ReLU;
var TanH = (function (_super) {
    __extends(TanH, _super);
    function TanH(xTensor, yTensor) {
        return _super.call(this, xTensor, yTensor, new activation_functions_1.TanHFunc()) || this;
    }
    return TanH;
}(ElementWiseActivation));
exports.TanH = TanH;
var Sigmoid = (function (_super) {
    __extends(Sigmoid, _super);
    function Sigmoid(xTensor, yTensor) {
        return _super.call(this, xTensor, yTensor, new activation_functions_1.SigmoidFunc()) || this;
    }
    return Sigmoid;
}(ElementWiseActivation));
exports.Sigmoid = Sigmoid;
var Square = (function (_super) {
    __extends(Square, _super);
    function Square(xTensor, yTensor) {
        return _super.call(this, xTensor, yTensor, new activation_functions_1.SquareFunc()) || this;
    }
    return Square;
}(ElementWiseActivation));
exports.Square = Square;

},{"../math/activation_functions":18,"./op":68}],61:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var cost_functions_1 = require("../math/cost_functions");
var ndarray_1 = require("../math/ndarray");
var util = require("../util");
var op_1 = require("./op");
var ElementWiseCost = (function (_super) {
    __extends(ElementWiseCost, _super);
    function ElementWiseCost(x1Tensor, x2Tensor, yTensor, func) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        _this.func = func;
        _this.oneOverNScalar = ndarray_1.Scalar.new(1 / util.sizeFromShape(x1Tensor.shape));
        return _this;
    }
    ElementWiseCost.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var elementWiseCost = _this.func.cost(math, x1, x2);
            var sum = math.sum(elementWiseCost);
            var result = math.scalarTimesArray(_this.oneOverNScalar, sum);
            inferenceArrays.set(_this.yTensor, keep(result));
        });
    };
    ElementWiseCost.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function () {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                gradientArrays.add(_this.x1Tensor, _this.func.der(math, x1, x2));
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                gradientArrays.add(_this.x2Tensor, _this.func.der(math, x2, x1));
            }
        });
    };
    ElementWiseCost.prototype.dispose = function () {
        this.func.dispose();
        this.oneOverNScalar.dispose();
    };
    return ElementWiseCost;
}(op_1.Operation));
exports.ElementWiseCost = ElementWiseCost;
var MeanSquaredCost = (function (_super) {
    __extends(MeanSquaredCost, _super);
    function MeanSquaredCost(x1Tensor, x2Tensor, yTensor) {
        return _super.call(this, x1Tensor, x2Tensor, yTensor, new cost_functions_1.SquareCostFunc()) || this;
    }
    return MeanSquaredCost;
}(ElementWiseCost));
exports.MeanSquaredCost = MeanSquaredCost;

},{"../graph_util":14,"../math/cost_functions":22,"../math/ndarray":26,"../util":79,"./op":68}],62:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var op_1 = require("./op");
var Exp = (function (_super) {
    __extends(Exp, _super);
    function Exp(xTensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        return _this;
    }
    Exp.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.exp(x)));
        });
    };
    Exp.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var y = inferenceArrays.get(this.yTensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function () {
            if (graph_util.shouldBackProp(_this.xTensor)) {
                gradientArrays.add(_this.xTensor, math.elementWiseMul(y, dy));
            }
        });
    };
    return Exp;
}(op_1.Operation));
exports.Exp = Exp;

},{"../graph_util":14,"./op":68}],63:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var op_1 = require("./op");
var LinearCombination = (function (_super) {
    __extends(LinearCombination, _super);
    function LinearCombination(x1Tensor, x2Tensor, c1Tensor, c2Tensor, outTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.c1Tensor = c1Tensor;
        _this.c2Tensor = c2Tensor;
        _this.outTensor = outTensor;
        return _this;
    }
    LinearCombination.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        var c1 = inferenceArrays.get(this.c1Tensor).asScalar();
        var c2 = inferenceArrays.get(this.c2Tensor).asScalar();
        math.scope(function (keep) {
            inferenceArrays.set(_this.outTensor, keep(math.scaledArrayAdd(c1, x1, c2, x2)));
        });
    };
    LinearCombination.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        var c1 = inferenceArrays.get(this.c1Tensor);
        var c2 = inferenceArrays.get(this.c2Tensor);
        var dy = gradientArrays.get(this.outTensor);
        math.scope(function () {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                gradientArrays.add(_this.x1Tensor, math.scalarTimesArray(c1, dy));
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                gradientArrays.add(_this.x2Tensor, math.scalarTimesArray(c2, dy));
            }
            if (graph_util.shouldBackProp(_this.c1Tensor)) {
                var dotProduct1 = math.elementWiseMul(x1, dy);
                gradientArrays.add(_this.c1Tensor, math.sum(dotProduct1));
            }
            if (graph_util.shouldBackProp(_this.c2Tensor)) {
                var dotProduct2 = math.elementWiseMul(x2, dy);
                gradientArrays.add(_this.c2Tensor, math.sum(dotProduct2));
            }
        });
    };
    return LinearCombination;
}(op_1.Operation));
exports.LinearCombination = LinearCombination;

},{"../graph_util":14,"./op":68}],64:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var op_1 = require("./op");
var Log = (function (_super) {
    __extends(Log, _super);
    function Log(xTensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        return _this;
    }
    Log.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.log(x)));
        });
    };
    Log.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function () {
            if (graph_util.shouldBackProp(_this.xTensor)) {
                gradientArrays.add(_this.xTensor, math.divide(dy, x));
            }
        });
    };
    return Log;
}(op_1.Operation));
exports.Log = Log;

},{"../graph_util":14,"./op":68}],65:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var math_1 = require("../math/math");
var op_1 = require("./op");
var MatMul = (function (_super) {
    __extends(MatMul, _super);
    function MatMul(x1Tensor, x2Tensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        return _this;
    }
    MatMul.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            if (x1.shape.length === 2 && x2.shape.length === 2) {
                inferenceArrays.set(_this.yTensor, keep(math.matMul(x1, x2)));
            }
            else if (x1.shape.length === 2 && x2.shape.length === 1) {
                inferenceArrays.set(_this.yTensor, keep(math.matrixTimesVector(x1, x2)));
            }
            else if (x1.shape.length === 1 && x2.shape.length === 2) {
                inferenceArrays.set(_this.yTensor, keep(math.vectorTimesMatrix(x1, x2)));
            }
        });
    };
    MatMul.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        var dy = gradientArrays.get(this.yTensor);
        if (x1.shape.length === 1) {
            x1 = x1.reshape([1, x1.size]);
            dy = dy.reshape([1, dy.size]);
        }
        if (x2.shape.length === 1) {
            x2 = x2.reshape([x2.size, 1]);
            dy = dy.reshape([dy.size, 1]);
        }
        math.scope(function () {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                var dx1 = math.matMul(dy, x2, math_1.MatrixOrientation.REGULAR, math_1.MatrixOrientation.TRANSPOSED);
                gradientArrays.add(_this.x1Tensor, _this.x1Tensor.shape.length === 1 ? dx1.as1D() : dx1);
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                var dx2 = math.matMul(x1, dy, math_1.MatrixOrientation.TRANSPOSED, math_1.MatrixOrientation.REGULAR);
                gradientArrays.add(_this.x2Tensor, _this.x2Tensor.shape.length === 1 ? dx2.as1D() : dx2);
            }
        });
    };
    return MatMul;
}(op_1.Operation));
exports.MatMul = MatMul;

},{"../graph_util":14,"../math/math":23,"./op":68}],66:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var conv_util = require("../math/conv_util");
var util = require("../util");
var op_1 = require("./op");
var MaxPool = (function (_super) {
    __extends(MaxPool, _super);
    function MaxPool(xTensor, yTensor, fieldSize, stride, pad) {
        if (stride === void 0) { stride = 1; }
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        _this.fieldSize = fieldSize;
        _this.stride = stride;
        if (pad != null) {
            _this.pad = pad;
        }
        else {
            _this.pad = conv_util.computeDefaultPad(xTensor.shape, _this.fieldSize, _this.stride);
        }
        util.assert(util.isInt(_this.pad), "The zero padding (" + _this.pad + ") must be an integer. Change the " +
            "stride and/or zero pad parameters");
        return _this;
    }
    MaxPool.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(math.maxPool(x, _this.fieldSize, _this.stride, _this.pad)));
        });
    };
    MaxPool.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function () {
            gradientArrays.add(_this.xTensor, math.maxPoolBackprop(dy, x, _this.fieldSize, _this.stride, _this.pad));
        });
    };
    return MaxPool;
}(op_1.Operation));
exports.MaxPool = MaxPool;

},{"../math/conv_util":20,"../util":79,"./op":68}],67:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var util = require("../util");
var op_1 = require("./op");
var Multiply = (function (_super) {
    __extends(Multiply, _super);
    function Multiply(x1Tensor, x2Tensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.x1Tensor = x1Tensor;
        _this.x2Tensor = x2Tensor;
        _this.yTensor = yTensor;
        util.assert(util.sizeFromShape(x1Tensor.shape) === 1 ||
            util.sizeFromShape(x2Tensor.shape) === 1 ||
            util.arraysEqual(x1Tensor.shape, x2Tensor.shape), 'One of t1 or t2 must be a scalar, or t1 and t2 must have ' +
            'the same shape');
        return _this;
    }
    Multiply.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var t1 = inferenceArrays.get(this.x1Tensor);
        var t2 = inferenceArrays.get(this.x2Tensor);
        math.scope(function (keep) {
            var result;
            if (util.isScalarShape(t1.shape)) {
                result = math.scalarTimesArray(t1, t2);
            }
            else if (util.isScalarShape(t2.shape)) {
                result = math.scalarTimesArray(t2, t1);
            }
            else {
                result = math.elementWiseMul(t1, t2);
            }
            inferenceArrays.set(_this.yTensor, keep(result));
        });
    };
    Multiply.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var x1 = inferenceArrays.get(this.x1Tensor);
        var x2 = inferenceArrays.get(this.x2Tensor);
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function () {
            if (graph_util.shouldBackProp(_this.x1Tensor)) {
                if (util.isScalarShape(_this.x1Tensor.shape)) {
                    var mul = math.elementWiseMul(dy, x2);
                    gradientArrays.add(_this.x1Tensor, math.sum(mul));
                }
                else if (util.isScalarShape(x2.shape)) {
                    gradientArrays.add(_this.x1Tensor, math.scalarTimesArray(x2, dy));
                }
                else {
                    gradientArrays.add(_this.x1Tensor, math.elementWiseMul(x2, dy));
                }
            }
            if (graph_util.shouldBackProp(_this.x2Tensor)) {
                if (util.isScalarShape(_this.x2Tensor.shape)) {
                    var mul = math.elementWiseMul(dy, x1);
                    gradientArrays.add(_this.x2Tensor, math.sum(mul));
                }
                else if (util.isScalarShape(x1.shape)) {
                    gradientArrays.add(_this.x2Tensor, math.scalarTimesArray(x1, dy));
                }
                else {
                    gradientArrays.add(_this.x2Tensor, math.elementWiseMul(x1, dy));
                }
            }
        });
    };
    return Multiply;
}(op_1.Operation));
exports.Multiply = Multiply;

},{"../graph_util":14,"../util":79,"./op":68}],68:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Operation = (function () {
    function Operation() {
    }
    Operation.prototype.disposeTransientArrays = function (inferenceArrays, gradientArrays) { };
    Operation.prototype.dispose = function () { };
    return Operation;
}());
exports.Operation = Operation;

},{}],69:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var ndarray_1 = require("../math/ndarray");
var util = require("../util");
var op_1 = require("./op");
var ReduceSum = (function (_super) {
    __extends(ReduceSum, _super);
    function ReduceSum(x, outTensor) {
        var _this = _super.call(this) || this;
        _this.x = x;
        _this.outTensor = outTensor;
        util.assertShapesMatch(outTensor.shape, []);
        return _this;
    }
    ReduceSum.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.x);
        math.scope(function (keep) {
            inferenceArrays.set(_this.outTensor, keep(math.sum(x)));
        });
    };
    ReduceSum.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        if (!graph_util.shouldBackProp(this.x)) {
            return;
        }
        math.scope(function () {
            var dy = gradientArrays.get(_this.outTensor);
            if (_this.ones == null) {
                var xArray = inferenceArrays.get(_this.x);
                _this.ones = ndarray_1.NDArray.zerosLike(xArray);
                _this.ones.fill(1);
            }
            gradientArrays.add(_this.x, math.scalarTimesArray(dy, _this.ones));
        });
    };
    return ReduceSum;
}(op_1.Operation));
exports.ReduceSum = ReduceSum;

},{"../graph_util":14,"../math/ndarray":26,"../util":79,"./op":68}],70:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var util = require("../util");
var op_1 = require("./op");
var Reshape = (function (_super) {
    __extends(Reshape, _super);
    function Reshape(xTensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.xTensor = xTensor;
        _this.yTensor = yTensor;
        var xSize = util.sizeFromShape(xTensor.shape);
        var ySize = util.sizeFromShape(yTensor.shape);
        util.assert(xSize === ySize, "The input size (" + xSize + ") and output size (" + ySize + ") must match");
        return _this;
    }
    Reshape.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var x = inferenceArrays.get(this.xTensor);
        math.scope(function (keep) {
            inferenceArrays.set(_this.yTensor, keep(x.reshape(_this.yTensor.shape)));
        });
    };
    Reshape.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var dy = gradientArrays.get(this.yTensor);
        math.scope(function () {
            gradientArrays.add(_this.xTensor, dy.reshape(_this.xTensor.shape));
        });
    };
    return Reshape;
}(op_1.Operation));
exports.Reshape = Reshape;

},{"../util":79,"./op":68}],71:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_1 = require("../graph");
var ndarray_1 = require("../math/ndarray");
var util = require("../util");
var op_1 = require("./op");
var Softmax = (function (_super) {
    __extends(Softmax, _super);
    function Softmax(logitsTensor, output) {
        var _this = _super.call(this) || this;
        _this.logitsTensor = logitsTensor;
        _this.output = output;
        return _this;
    }
    Softmax.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var logits = inferenceArrays.get(this.logitsTensor);
        return math.scope(function (keep) {
            inferenceArrays.set(_this.output, keep(math.softmax(logits)));
        });
    };
    Softmax.prototype.backProp = function () {
        throw Error('Softmax backprop is not yet implemented');
    };
    return Softmax;
}(op_1.Operation));
exports.Softmax = Softmax;
var SoftmaxCrossEntropyCost = (function (_super) {
    __extends(SoftmaxCrossEntropyCost, _super);
    function SoftmaxCrossEntropyCost(logitsTensor, labelTensor, yTensor) {
        var _this = _super.call(this) || this;
        _this.logitsTensor = logitsTensor;
        _this.labelTensor = labelTensor;
        _this.yTensor = yTensor;
        _this.epsilon = ndarray_1.Scalar.new(1e-5);
        _this.softmaxTensor = new graph_1.Tensor(logitsTensor.shape);
        return _this;
    }
    SoftmaxCrossEntropyCost.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var logits = inferenceArrays.get(this.logitsTensor);
        var label = inferenceArrays.get(this.labelTensor);
        math.scope(function (keep) {
            var softmaxResult = math.softmax(logits);
            inferenceArrays.set(_this.softmaxTensor, keep(softmaxResult));
            inferenceArrays.set(_this.yTensor, keep(crossEntropyCost(math, softmaxResult, label, _this.epsilon)));
        });
    };
    SoftmaxCrossEntropyCost.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var softmax = inferenceArrays.get(this.softmaxTensor);
        var label = inferenceArrays.get(this.labelTensor);
        math.scope(function () {
            gradientArrays.add(_this.logitsTensor, math.sub(softmax, label));
        });
    };
    SoftmaxCrossEntropyCost.prototype.disposeTransientArrays = function (inferenceArrays, gradientArrays) {
        inferenceArrays.disposeArray(this.softmaxTensor);
    };
    SoftmaxCrossEntropyCost.prototype.dispose = function () {
        this.epsilon.dispose();
    };
    return SoftmaxCrossEntropyCost;
}(op_1.Operation));
exports.SoftmaxCrossEntropyCost = SoftmaxCrossEntropyCost;
function crossEntropyCost(math, y, target, epsilon) {
    util.assert(y.size === target.size, 'The output and target must be the same size');
    return math.scope(function () {
        var yPlusEps = math.scalarPlusArray(epsilon, y);
        var logOutput = math.log(yPlusEps);
        var tarLogOutput = math.elementWiseMul(target, logOutput);
        var costVector = math.neg(tarLogOutput);
        return math.sum(costVector);
    });
}
exports.crossEntropyCost = crossEntropyCost;

},{"../graph":11,"../math/ndarray":26,"../util":79,"./op":68}],72:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var graph_util = require("../graph_util");
var ndarray_1 = require("../math/ndarray");
var util = require("../util");
var op_1 = require("./op");
var Subtract = (function (_super) {
    __extends(Subtract, _super);
    function Subtract(t1, t2, outTensor) {
        var _this = _super.call(this) || this;
        _this.t1 = t1;
        _this.t2 = t2;
        _this.outTensor = outTensor;
        util.assert(util.sizeFromShape(t1.shape) === 1 ||
            util.sizeFromShape(t2.shape) === 1 ||
            util.arraysEqual(t1.shape, t2.shape), 'One of t1 or t2 must be a scalar, or t1 and t2 must have ' +
            'the same shape');
        return _this;
    }
    Subtract.prototype.feedForward = function (math, inferenceArrays) {
        var _this = this;
        var t1 = inferenceArrays.get(this.t1);
        var t2 = inferenceArrays.get(this.t2);
        math.scope(function (keep) {
            var result;
            if (util.isScalarShape(t1.shape)) {
                result = math.scalarMinusArray(t1, t2);
            }
            else if (util.isScalarShape(t2.shape)) {
                result = math.arrayMinusScalar(t1, t2);
            }
            else {
                result = math.sub(t1, t2);
            }
            inferenceArrays.set(_this.outTensor, keep(result));
        });
    };
    Subtract.prototype.backProp = function (math, inferenceArrays, gradientArrays) {
        var _this = this;
        var dy = gradientArrays.get(this.outTensor);
        math.scope(function () {
            if (graph_util.shouldBackProp(_this.t1)) {
                if (util.isScalarShape(_this.t1.shape)) {
                    var sum = math.sum(dy);
                    if (_this.dySizeScalar == null) {
                        _this.dySizeScalar = ndarray_1.Scalar.new(dy.size);
                    }
                    gradientArrays.add(_this.t1, math.divide(sum, _this.dySizeScalar));
                }
                else {
                    gradientArrays.add(_this.t1, dy);
                }
            }
            if (graph_util.shouldBackProp(_this.t2)) {
                if (util.isScalarShape(_this.t2.shape)) {
                    var sum = math.sum(dy);
                    var negSum = math.neg(sum);
                    if (_this.dySizeScalar == null) {
                        _this.dySizeScalar = ndarray_1.Scalar.new(dy.size);
                    }
                    gradientArrays.add(_this.t2, math.divide(negSum, _this.dySizeScalar));
                }
                else {
                    gradientArrays.add(_this.t2, math.neg(dy));
                }
            }
        });
    };
    Subtract.prototype.dispose = function () {
        if (this.dySizeScalar != null) {
            this.dySizeScalar.dispose();
        }
    };
    return Subtract;
}(op_1.Operation));
exports.Subtract = Subtract;

},{"../graph_util":14,"../math/ndarray":26,"../util":79,"./op":68}],73:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var Optimizer = (function () {
    function Optimizer(specifiedVariableList) {
        if (specifiedVariableList != null) {
            this.specifiedVariableNodes = specifiedVariableList;
        }
    }
    return Optimizer;
}());
exports.Optimizer = Optimizer;

},{}],74:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function defaultCompare(a, b) {
    if (a === b) {
        return 0;
    }
    else if (a < b) {
        return -1;
    }
    else {
        return 1;
    }
}
exports.defaultCompare = defaultCompare;
var PriorityQueue = (function () {
    function PriorityQueue(comparator, indexObserver) {
        this.comparator = comparator;
        this.indexObserver = indexObserver;
        this.heap = [];
    }
    PriorityQueue.prototype.enqueue = function (t) {
        this.heap.push(t);
        this.onIndexChanged(t, this.heap.length - 1);
        this.siftUp(this.heap.length - 1);
    };
    PriorityQueue.prototype.dequeue = function () {
        if (this.empty()) {
            throw new Error('dequeue called on empty priority queue.');
        }
        var t = this.heap[0];
        this.swap(0, this.heap.length - 1);
        this.heap.pop();
        this.siftDown(0);
        return t;
    };
    PriorityQueue.prototype.update = function (newT, index) {
        var last = (index === this.heap.length - 1);
        if (!last) {
            this.swap(index, this.heap.length - 1);
        }
        this.heap.pop();
        if (!last) {
            if (this.siftUpIndex(index) !== -1) {
                this.siftUp(index);
            }
            else if (this.siftDownIndex(index) !== -1) {
                this.siftDown(index);
            }
        }
        this.enqueue(newT);
    };
    PriorityQueue.prototype.empty = function () {
        return this.heap.length === 0;
    };
    PriorityQueue.prototype.onIndexChanged = function (t, newIndex) {
        if (this.indexObserver) {
            this.indexObserver(t, newIndex);
        }
    };
    PriorityQueue.prototype.getParentIndex = function (index) {
        if (index === 0) {
            return -1;
        }
        return Math.floor((index - 1) / 2);
    };
    PriorityQueue.prototype.getLeftChildIndex = function (index) {
        var candidate = index * 2 + 1;
        return candidate < this.heap.length ? candidate : -1;
    };
    PriorityQueue.prototype.getRightChildIndex = function (index) {
        var candidate = index * 2 + 2;
        return candidate < this.heap.length ? candidate : -1;
    };
    PriorityQueue.prototype.siftUpIndex = function (index) {
        var parentIndex = this.getParentIndex(index);
        if (parentIndex === -1) {
            return -1;
        }
        if (this.compare(parentIndex, index) > 0) {
            return parentIndex;
        }
        return -1;
    };
    PriorityQueue.prototype.siftUp = function (index) {
        var siftIndex = this.siftUpIndex(index);
        while (siftIndex !== -1) {
            this.swap(index, siftIndex);
            index = siftIndex;
            siftIndex = this.siftUpIndex(index);
        }
    };
    PriorityQueue.prototype.siftDownIndex = function (index) {
        if (index >= this.heap.length) {
            return -1;
        }
        var largestChildIndex = index;
        var leftChildIndex = this.getLeftChildIndex(index);
        if ((leftChildIndex !== -1) &&
            (this.compare(leftChildIndex, largestChildIndex) < 0)) {
            largestChildIndex = leftChildIndex;
        }
        var rightChildIndex = this.getRightChildIndex(index);
        if ((rightChildIndex !== -1) &&
            (this.compare(rightChildIndex, largestChildIndex) < 0)) {
            largestChildIndex = rightChildIndex;
        }
        return (largestChildIndex === index) ? -1 : largestChildIndex;
    };
    PriorityQueue.prototype.siftDown = function (index) {
        var siftIndex = this.siftDownIndex(index);
        while (siftIndex !== -1) {
            this.swap(index, siftIndex);
            index = siftIndex;
            siftIndex = this.siftDownIndex(index);
        }
    };
    PriorityQueue.prototype.compare = function (aIndex, bIndex) {
        return this.comparator(this.heap[aIndex], this.heap[bIndex]);
    };
    PriorityQueue.prototype.swap = function (a, b) {
        var temp = this.heap[a];
        this.heap[a] = this.heap[b];
        this.heap[b] = temp;
        this.onIndexChanged(this.heap[a], a);
        this.onIndexChanged(this.heap[b], b);
    };
    return PriorityQueue;
}());
exports.PriorityQueue = PriorityQueue;

},{}],75:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var operation_emitter = require("./operation_emitter");
var session_util = require("./session_util");
var tensor_array_map_1 = require("./tensor_array_map");
var util = require("./util");
var FeedDictionary = (function () {
    function FeedDictionary(feedEntries) {
        var _this = this;
        this.dict = {};
        if (feedEntries) {
            feedEntries.forEach(function (entry) { return _this.dict[entry.tensor.id] = entry; });
        }
    }
    return FeedDictionary;
}());
exports.FeedDictionary = FeedDictionary;
var CostReduction;
(function (CostReduction) {
    CostReduction[CostReduction["NONE"] = 0] = "NONE";
    CostReduction[CostReduction["SUM"] = 1] = "SUM";
    CostReduction[CostReduction["MEAN"] = 2] = "MEAN";
})(CostReduction = exports.CostReduction || (exports.CostReduction = {}));
var Session = (function () {
    function Session(graph, math) {
        this.math = math;
        this.activationArrayMap = new tensor_array_map_1.TensorArrayMap();
        this.runtimeCache = {};
        this.oneScalar = ndarray_1.Scalar.new(1);
        this.gradientArrayMap = new tensor_array_map_1.SummedTensorArrayMap(this.math);
    }
    Session.prototype.dispose = function () {
        var _this = this;
        this.activationArrayMap.dispose();
        Object.keys(this.runtimeCache).forEach(function (key) {
            var runtime = _this.runtimeCache[key];
            if (runtime.operations) {
                runtime.operations.forEach(function (op) { return op.dispose(); });
            }
        });
        this.runtimeCache = {};
        if (this.batchSizeScalar != null) {
            this.batchSizeScalar.dispose();
        }
        this.oneScalar.dispose();
    };
    Session.prototype.evalAll = function (tensors, feedEntries) {
        var _this = this;
        return this.math.scope(function () {
            var feed = new FeedDictionary(feedEntries);
            var runtime = _this.getOrCreateRuntime(tensors, feed);
            var activations = _this.activationArrayMap;
            session_util.disposeAndInitializeOperationOutputs(runtime.nodes, activations);
            session_util.disposeTransientOperationArrays(runtime.operations, _this.activationArrayMap, _this.gradientArrayMap);
            session_util.addPersistentArraysToTensorArrayMap(runtime.nodes, activations);
            session_util.loadInputsFromFeedDictionaryToTensorArrayMap(feed, activations, _this.math);
            runtime.operations.forEach(function (op) { return op.feedForward(_this.math, activations); });
            var results = tensors.map(function (x) { return activations.get(x); });
            tensors.forEach(function (x) { return activations.delete(x); });
            session_util.releaseFeedDictionaryInputsFromTensorArrayMap(feed, activations, _this.math);
            return results;
        });
    };
    Session.prototype.eval = function (tensor, feedEntries) {
        return this.evalAll([tensor], feedEntries)[0];
    };
    Session.prototype.train = function (costTensor, feedEntries, batchSize, optimizer, costReduction) {
        var _this = this;
        if (costReduction === void 0) { costReduction = CostReduction.NONE; }
        util.assert(util.isScalarShape(costTensor.shape), 'Cost tensor for training must be a scalar value.');
        if (this.prevBatchSize !== batchSize) {
            this.prevBatchSize = batchSize;
            this.batchSizeScalar = ndarray_1.Scalar.new(batchSize);
        }
        var feed = new FeedDictionary(feedEntries);
        session_util.throwIfFeedDictionaryContainsNDArrays(feed);
        var runtime = this.getOrCreateRuntime([costTensor], feed);
        var inferenceOperations = runtime.operations;
        var backPropOperations = runtime.operations.slice().reverse();
        var activations = this.activationArrayMap;
        var gradients = this.gradientArrayMap;
        gradients.nullify(costTensor);
        gradients.add(costTensor, this.oneScalar);
        session_util.addPersistentArraysToTensorArrayMap(runtime.nodes, activations);
        optimizer.beforeBatch(this.math, batchSize, runtime, activations, gradients);
        return this.math.scope(function (keep, track) {
            var cost = track(ndarray_1.Scalar.new(0));
            for (var i = 0; i < batchSize; ++i) {
                session_util.disposeAndInitializeOperationOutputs(runtime.nodes, activations);
                session_util.disposeAndInitializeOperationInputGradients(runtime.nodes, gradients);
                session_util.disposeTransientOperationArrays(runtime.operations, activations, gradients);
                session_util.loadInputsFromFeedDictionaryToTensorArrayMap(feed, activations, _this.math);
                inferenceOperations.forEach(function (op) { return op.feedForward(_this.math, activations); });
                backPropOperations.forEach(function (op) { return op.backProp(_this.math, activations, gradients); });
                optimizer.afterExample(_this.math, runtime, activations, gradients);
                session_util.releaseFeedDictionaryInputsFromTensorArrayMap(feed, activations, _this.math);
                cost = _this.updateCostForExample(cost, activations.get(costTensor), costReduction);
            }
            optimizer.afterBatch(_this.math, batchSize, runtime, activations, gradients);
            return _this.updateCostForBatch(cost, costReduction);
        });
    };
    Session.prototype.updateCostForExample = function (totalCost, currCost, costReduction) {
        if (costReduction === CostReduction.MEAN ||
            costReduction === CostReduction.SUM) {
            return this.math.add(totalCost, currCost);
        }
        return totalCost;
    };
    Session.prototype.updateCostForBatch = function (totalCost, costReduction) {
        if (costReduction === CostReduction.MEAN) {
            return this.math.divide(totalCost, this.batchSizeScalar);
        }
        return totalCost;
    };
    Session.prototype.getOrCreateRuntime = function (tensors, feed) {
        var key = this.makeRuntimeCacheKey(tensors, feed);
        var runtime = this.runtimeCache[key];
        if (runtime === undefined) {
            var nodes = session_util.getOrderedEvaluationSetFromEvalTensor(tensors, feed);
            session_util.removeFeedDictionaryNodesFromEvaluationSet(feed, nodes);
            session_util.throwErrorIfEvaluationSetContainsPlaceholderNodes(nodes);
            var operations = operation_emitter.emitFromGraphNodes(nodes);
            runtime = { nodes: nodes, operations: operations };
            this.runtimeCache[key] = runtime;
        }
        return runtime;
    };
    Session.prototype.makeRuntimeCacheKey = function (tensors, feed) {
        return tensors.map(function (x) { return x.id; }).sort().join('_') + '__' +
            Object.keys(feed.dict).sort().join('_');
    };
    return Session;
}());
exports.Session = Session;

},{"./math/ndarray":26,"./operation_emitter":53,"./session_util":76,"./tensor_array_map":78,"./util":79}],76:[function(require,module,exports){
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var graph_1 = require("./graph");
var graph_util = require("./graph_util");
var ndarray_1 = require("./math/ndarray");
var util = require("./util");
function getTerminatingNodesFromFeedDictionary(feedDictionary) {
    return Object.keys(feedDictionary.dict)
        .map(function (tensorID) { return feedDictionary.dict[+tensorID].tensor.node; });
}
exports.getTerminatingNodesFromFeedDictionary = getTerminatingNodesFromFeedDictionary;
function getOrderedEvaluationSetFromEvalTensor(evalTensors, feedDictionary) {
    var terminatingNodes = getTerminatingNodesFromFeedDictionary(feedDictionary);
    var evalNodes = evalTensors.map(function (x) { return x.node; });
    var unorderedEvaluationSet = graph_util.getUnorderedEvaluationSet(evalNodes, terminatingNodes);
    var orderedEvaluationSet = graph_util.getOrderedEvaluationSet(unorderedEvaluationSet);
    return orderedEvaluationSet;
}
exports.getOrderedEvaluationSetFromEvalTensor = getOrderedEvaluationSetFromEvalTensor;
function addPersistentArraysToTensorArrayMap(evaluationSet, tensorArrayMap) {
    evaluationSet.forEach(function (node) {
        if (node instanceof graph_1.VariableNode || node instanceof graph_1.ConstantNode) {
            tensorArrayMap.set(node.output, node.data);
        }
    });
}
exports.addPersistentArraysToTensorArrayMap = addPersistentArraysToTensorArrayMap;
function getVariableNodesFromEvaluationSet(evaluationSet) {
    var nodes = [];
    evaluationSet.forEach(function (node) {
        if (node instanceof graph_1.VariableNode) {
            nodes.push(node);
        }
    });
    return nodes;
}
exports.getVariableNodesFromEvaluationSet = getVariableNodesFromEvaluationSet;
function throwIfFeedDictionaryContainsNDArrays(feedDictionary) {
    Object.keys(feedDictionary.dict).forEach(function (tensorID) {
        if (feedDictionary.dict[+tensorID].data instanceof ndarray_1.NDArray) {
            throw new Error('training requires FeedDictionary entries to be InputProviders' +
                'and not NDArrays.');
        }
    });
}
exports.throwIfFeedDictionaryContainsNDArrays = throwIfFeedDictionaryContainsNDArrays;
function loadInputsFromFeedDictionaryToTensorArrayMap(batchFeed, activations, math) {
    Object.keys(batchFeed.dict).forEach(function (tensorID) {
        var feedEntry = batchFeed.dict[+tensorID];
        var data;
        if (feedEntry.data instanceof ndarray_1.NDArray) {
            data = feedEntry.data;
        }
        else {
            var provider = feedEntry.data;
            data = provider.getNextCopy(math);
        }
        util.assert(util.arraysEqual(feedEntry.tensor.shape, data.shape), "Error loading FeedEntry: feeding NDArray of shape " + data.shape + " " +
            ("does not match Tensor (id: " + feedEntry.tensor.id + ") shape: ") +
            (feedEntry.tensor.shape + "."));
        activations.set(feedEntry.tensor, data);
    });
}
exports.loadInputsFromFeedDictionaryToTensorArrayMap = loadInputsFromFeedDictionaryToTensorArrayMap;
function releaseFeedDictionaryInputsFromTensorArrayMap(batchFeed, activations, math) {
    Object.keys(batchFeed.dict).forEach(function (tensorID) {
        var feedEntry = batchFeed.dict[+tensorID];
        if (!(feedEntry.data instanceof ndarray_1.NDArray)) {
            var provider = feedEntry.data;
            var feedEntryArray = activations.get(feedEntry.tensor);
            provider.disposeCopy(math, feedEntryArray);
        }
        activations.delete(feedEntry.tensor);
    });
}
exports.releaseFeedDictionaryInputsFromTensorArrayMap = releaseFeedDictionaryInputsFromTensorArrayMap;
function removeFeedDictionaryNodesFromEvaluationSet(feedDictionary, evaluationSet) {
    var i = 0;
    while (i < evaluationSet.length) {
        var node = evaluationSet[i];
        if (feedDictionary.dict[node.output.id] != null) {
            evaluationSet.splice(i, 1);
        }
        else {
            ++i;
        }
    }
}
exports.removeFeedDictionaryNodesFromEvaluationSet = removeFeedDictionaryNodesFromEvaluationSet;
function disposeAndInitializeOperationOutputs(evaluationSet, tensorArrayMap) {
    evaluationSet.forEach(function (node) {
        if (!graph_util.isInputNode(node)) {
            if (!graph_util.isPassthroughNode(node, tensorArrayMap)) {
                tensorArrayMap.disposeArray(node.output);
            }
            tensorArrayMap.set(node.output, null);
        }
    });
}
exports.disposeAndInitializeOperationOutputs = disposeAndInitializeOperationOutputs;
function disposeAndInitializeOperationInputGradients(evaluationSet, gradients) {
    evaluationSet.forEach(function (node) {
        Object.keys(node.inputs).forEach(function (inputName) {
            var input = node.inputs[inputName];
            if (gradients.get(input, true) !== gradients.get(node.output, true)) {
                gradients.disposeArray(input);
            }
            gradients.nullify(input);
        });
    });
}
exports.disposeAndInitializeOperationInputGradients = disposeAndInitializeOperationInputGradients;
function disposeTransientOperationArrays(operations, activations, gradients) {
    operations.forEach(function (op) { return op.disposeTransientArrays(activations, gradients); });
}
exports.disposeTransientOperationArrays = disposeTransientOperationArrays;
function throwErrorIfEvaluationSetContainsPlaceholderNodes(evaluationSet) {
    evaluationSet.forEach(function (node) {
        if (node instanceof graph_1.PlaceholderNode) {
            var shape = '[' + node.output.shape.join(', ') + ']';
            throw new Error('Placeholder node "' + node.name + '" ' + shape +
                ' not present in feed dictionary.');
        }
    });
}
exports.throwErrorIfEvaluationSetContainsPlaceholderNodes = throwErrorIfEvaluationSetContainsPlaceholderNodes;

},{"./graph":11,"./graph_util":14,"./math/ndarray":26,"./util":79}],77:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var ndarray_1 = require("./math/ndarray");
var optimizer_1 = require("./optimizer");
var session_util = require("./session_util");
var tensor_array_map_1 = require("./tensor_array_map");
var SGDOptimizer = (function (_super) {
    __extends(SGDOptimizer, _super);
    function SGDOptimizer(learningRate, specifiedVariableList) {
        var _this = _super.call(this, specifiedVariableList) || this;
        _this.learningRate = learningRate;
        _this.variableGradients = new tensor_array_map_1.TensorArrayMap();
        _this.one = ndarray_1.Scalar.new(1);
        return _this;
    }
    SGDOptimizer.prototype.beforeBatch = function (math, batchSize, runtime, activationArrayMap, gradientArrayMap) {
        var _this = this;
        this.variableNodes = this.specifiedVariableNodes == null ?
            session_util.getVariableNodesFromEvaluationSet(runtime.nodes) :
            this.specifiedVariableNodes;
        if (batchSize !== this.prevBatchSize) {
            this.prevBatchSize = batchSize;
            this.c = ndarray_1.Scalar.new(-this.learningRate / batchSize);
        }
        this.variableNodes.forEach(function (node) { return _this.variableGradients.set(node.output, ndarray_1.NDArray.zeros(node.output.shape)); });
    };
    SGDOptimizer.prototype.afterExample = function (math, runtime, activationArrayMap, gradientArrayMap) {
        var _this = this;
        math.scope(function (keep) {
            _this.variableNodes.forEach(function (node) {
                var gradient = gradientArrayMap.get(node.output);
                var accumulatedGradient = _this.variableGradients.get(node.output);
                _this.variableGradients.set(node.output, keep(math.add(gradient, accumulatedGradient)));
                accumulatedGradient.dispose();
            });
        });
    };
    SGDOptimizer.prototype.afterBatch = function (math, batchSize, runtime, activationArrayMap, gradientArrayMap) {
        var _this = this;
        math.scope(function (keep) {
            _this.variableNodes.forEach(function (node) {
                var oldVariable = activationArrayMap.get(node.output);
                var gradient = _this.variableGradients.get(node.output);
                var variable = math.scaledArrayAdd(_this.c, gradient, _this.one, oldVariable);
                activationArrayMap.set(node.output, keep(variable));
                node.data = variable;
                oldVariable.dispose();
            });
        });
        this.variableGradients.dispose();
        this.variableGradients = new tensor_array_map_1.TensorArrayMap();
    };
    SGDOptimizer.prototype.dispose = function () {
        if (this.c != null) {
            this.c.dispose();
        }
        this.one.dispose();
    };
    SGDOptimizer.prototype.setLearningRate = function (learningRate) {
        this.learningRate = learningRate;
    };
    return SGDOptimizer;
}(optimizer_1.Optimizer));
exports.SGDOptimizer = SGDOptimizer;

},{"./math/ndarray":26,"./optimizer":73,"./session_util":76,"./tensor_array_map":78}],78:[function(require,module,exports){
"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var TensorArrayMapBase = (function () {
    function TensorArrayMapBase() {
        this.dict = {};
    }
    TensorArrayMapBase.prototype.get = function (tensor, skipChecks) {
        if (skipChecks === void 0) { skipChecks = false; }
        if (!skipChecks && this.dict[tensor.id] === undefined) {
            throw new Error('tensor ' + tensor.id + ' not in array map.');
        }
        var nda = this.dict[tensor.id];
        if (!skipChecks && nda === null) {
            throw new Error('tensor ' + tensor.id + ' has null array.');
        }
        return nda;
    };
    TensorArrayMapBase.prototype.delete = function (tensor) {
        delete this.dict[tensor.id];
    };
    TensorArrayMapBase.prototype.nullify = function (tensor) {
        this.dict[tensor.id] = null;
    };
    TensorArrayMapBase.prototype.disposeArray = function (tensor) {
        if (this.dict[tensor.id] === undefined) {
            return;
        }
        var nda = this.dict[tensor.id];
        if (nda === null) {
            return;
        }
        nda.dispose();
        this.dict[tensor.id] = null;
    };
    TensorArrayMapBase.prototype.size = function () {
        return Object.keys(this.dict).length;
    };
    TensorArrayMapBase.prototype.dispose = function () {
        var _this = this;
        Object.keys(this.dict).forEach(function (tensorID) {
            var nda = _this.dict[+tensorID];
            if (nda) {
                nda.dispose();
            }
        });
        this.dict = {};
    };
    TensorArrayMapBase.prototype.hasNullArray = function (tensor) {
        if (this.dict[tensor.id] === undefined) {
            throw new Error('tensor ' + tensor.id + ' not in array map.');
        }
        return this.dict[tensor.id] === null;
    };
    return TensorArrayMapBase;
}());
exports.TensorArrayMapBase = TensorArrayMapBase;
var TensorArrayMap = (function (_super) {
    __extends(TensorArrayMap, _super);
    function TensorArrayMap() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    TensorArrayMap.prototype.set = function (tensor, array) {
        this.dict[tensor.id] = array;
    };
    return TensorArrayMap;
}(TensorArrayMapBase));
exports.TensorArrayMap = TensorArrayMap;
var SummedTensorArrayMap = (function (_super) {
    __extends(SummedTensorArrayMap, _super);
    function SummedTensorArrayMap(math) {
        var _this = _super.call(this) || this;
        _this.math = math;
        return _this;
    }
    SummedTensorArrayMap.prototype.add = function (tensor, array) {
        if (this.dict[tensor.id] == null) {
            this.dict[tensor.id] = this.math.keep(array);
        }
        else {
            var oldValue = this.get(tensor);
            var newValue = this.math.keep(this.math.addStrict(oldValue, array));
            this.dict[tensor.id] = newValue;
            oldValue.dispose();
        }
    };
    return SummedTensorArrayMap;
}(TensorArrayMapBase));
exports.SummedTensorArrayMap = SummedTensorArrayMap;

},{}],79:[function(require,module,exports){
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
    if (truncated && result > 2) {
        return randGauss(mean, stdDev, true);
    }
    return mean + stdDev * result;
}
exports.randGauss = randGauss;
function distSquared(a, b) {
    var result = 0;
    for (var i = 0; i < a.length; i++) {
        var diff = a[i] - b[i];
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
    ret = (ret === undefined ? [] : ret);
    for (var i = 0; i < arr.length; ++i) {
        if (Array.isArray(arr[i])) {
            flatten(arr[i], ret);
        }
        else {
            ret.push(arr[i]);
        }
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
function assertAndGetBroadcastedShape(shapeA, shapeB) {
    var result = [];
    var nextADimMustBeOne = false;
    var nextBDimMustBeOne = false;
    var errMsg = "Operands could not be broadcast together with shapes " +
        (shapeA + " and " + shapeB + ". Currently, we only support a ") +
        "stricter version of broadcasting than numpy.";
    var l = Math.max(shapeA.length, shapeB.length);
    shapeA = shapeA.slice().reverse();
    shapeB = shapeB.slice().reverse();
    for (var i = 0; i < l; i++) {
        var a = shapeA[i] || 1;
        var b = shapeB[i] || 1;
        if ((b > 1 && nextBDimMustBeOne) || (a > 1 && nextADimMustBeOne)) {
            throw Error(errMsg);
        }
        if (a > 1 && b === 1) {
            nextBDimMustBeOne = true;
        }
        if (b > 1 && a === 1) {
            nextADimMustBeOne = true;
        }
        if (a > 1 && b > 1 && a !== b) {
            throw Error(errMsg);
        }
        result.push(Math.max(a, b));
    }
    return result.reverse();
}
exports.assertAndGetBroadcastedShape = assertAndGetBroadcastedShape;
function rightPad(a, size) {
    if (size <= a.length) {
        return a;
    }
    return a + ' '.repeat(size - a.length);
}
exports.rightPad = rightPad;

},{}]},{},[4]);
