/// <reference types="vite/client" />

declare const scene: HTMLCanvasElement;
declare const output: HTMLCanvasElement;

declare type StorageBuffer =
{
    [name: string]: /* TypedArray */ unknown,
    buffer: /* GPUBuffer */ unknown,
};

declare type UniformBuffer<Name extends string,
    TArrayBuffer extends ArrayBufferLike = Float32Array
> = { buffer: /* GPUBuffer */ unknown } & { [N in Name]: TArrayBuffer };

declare type StructBuffer<Name extends string,
    TArrayBuffer extends ArrayBufferLike = Float32Array
> = { buffer: /* GPUBuffer */ unknown } & { [N in Name]: Record<string, TArrayBuffer> };
