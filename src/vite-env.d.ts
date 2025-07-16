/// <reference types="vite/client" />

declare const scene: HTMLCanvasElement;
declare const output: HTMLCanvasElement;

declare type StorageBuffer =
{
    [name: string]: /* TypedArray */ unknown,
    buffer: /* GPUBuffer */ unknown,
};
