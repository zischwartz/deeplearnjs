export declare class KeyboardElement {
    private container;
    private keys;
    private notes;
    constructor(container: Element);
    resize(): void;
    keyDown(noteNum: number): void;
    keyUp(noteNum: number): void;
}
