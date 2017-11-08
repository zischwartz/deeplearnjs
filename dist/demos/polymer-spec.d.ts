export declare type Spec = {
    is: string;
    properties: {
        [key: string]: (Function | {
            type: Function;
            value?: any;
            reflectToAttribute?: boolean;
            readonly?: boolean;
            notify?: boolean;
            computed?: string;
            observer?: string;
        });
    };
    observers?: string[];
};
export declare function PolymerElement(spec: Spec): new () => PolymerHTMLElement;
export interface PolymerHTMLElement extends HTMLElement, polymer.Base {
}
