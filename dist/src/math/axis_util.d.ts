export declare function axesAreInnerMostDims(axes: number[], rank: number): boolean;
export declare function combineLocations(outputLoc: number[], reduceLoc: number[], axes: number[]): number[];
export declare function computeOutAndReduceShapes(aShape: number[], axes: number[]): [number[], number[]];
export declare function expandShapeToKeepDim(shape: number[], axes: number[]): number[];
export declare function parseAxisParam(axis: number | number[], shape: number[]): number[];
export declare function assertAxesAreInnerMostDims(msg: string, axes: number[], rank: number): void;
