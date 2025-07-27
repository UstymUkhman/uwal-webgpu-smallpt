/// <reference types="vite/client" />

declare type Sphere = {
    rad: number[];
    p: number[];
    e: number[];
    c: number[];
    refl: number[];
};

declare const scene: HTMLCanvasElement;
declare const output: HTMLCanvasElement;

declare type UniformBuffer<Name extends string,
    TArrayBuffer extends ArrayBufferLike = Float32Array
> = { buffer: GPUBuffer } & { [N in Name]: TArrayBuffer };

declare type StructBuffer<Name extends string,
    TArrayBuffer extends ArrayBufferLike = Float32Array
> = { buffer: GPUBuffer } & { [N in Name]: Record<string, TArrayBuffer> };

declare type StorageBuffer = { [name: string]: ArrayBufferLike, buffer: GPUBuffer };
