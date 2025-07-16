import Compute from "./Compute.wgsl?raw";
import Render from "./Render.wgsl?raw";

import {
    Device,
    Shaders,
    type Renderer,
    type Computation,
    type RenderPipeline,
    type ComputePipeline
} from "uwal";

export default class Scene
{
    private Renderer!: Renderer;
    private color!: StorageBuffer;

    private Computation!: Computation;
    private canvas!: HTMLCanvasElement;

    private storageBufferSize!: number;
    private workgroupDimension!: number;

    private resizeTimeout?: NodeJS.Timeout;
    private RenderPipeline!: RenderPipeline;
    private ComputePipeline!: ComputePipeline;

    public constructor() { Device.OnLost = () => void 0; }

    public setOutputCanvas(canvas: HTMLCanvasElement, width: number, height: number)
    {
        this.canvas = canvas;
        const { devicePixelRatio } = globalThis;
        this.canvas.width = width * devicePixelRatio | 0;
        this.canvas.height = height * devicePixelRatio | 0;
    }

    public async create(canvas: HTMLCanvasElement, width: number, height: number): Promise<number[]>
    {
        this.storageBufferSize = width * height * 16;
        await this.checkRequiredLimits(canvas);

        this.ComputePipeline = new this.Computation.Pipeline();
        await this.Computation.AddPipeline(this.ComputePipeline, {
            module: this.ComputePipeline.CreateShaderModule(Compute),
            constants: { DIMENSION_SIZE: this.workgroupDimension }
        });

        this.Renderer = new (await Device.Renderer(canvas));
        this.RenderPipeline = new this.Renderer.Pipeline();

        // Can't update CSS style of an `OffscreenCanvas`:
        this.Renderer.SetCanvasSize(width, height, false);

        await this.Renderer.AddPipeline(this.RenderPipeline,
            this.RenderPipeline.CreateShaderModule([
                Shaders.Resolution,
                Shaders.Quad,
                Render
            ])
        );

        this.createComputePipeline();
        this.createRenderPipeline();
        return [width, height];
    }

    private async checkRequiredLimits(canvas: HTMLCanvasElement): Promise<void>
    {
        const storageBufferBindingSize = this.storageBufferSize * Uint32Array.BYTES_PER_ELEMENT * 4;
        Device.RequiredLimits = { maxStorageBufferBindingSize: storageBufferBindingSize };
        Device.SetRequiredFeatures("bgra8unorm-storage");

        try
        {
            // Device request will fail if the adapter
            // can't provide required limits specified above.
            this.Computation = new (await Device.Computation());
            this.workgroupDimension = this.Computation.GetMaxEvenWorkgroupDimension(2);
        }
        catch (error)
        {
            this.create(canvas, 832, 624);
            console.warn(error);

            console.warn([
                "Will be used a fallback with the minimum `maxStorageBufferBindingSize`",
                "value available in all WebGPU contexts (134217728 bytes [128 MB]),",
                "which produces a 832 x 624 pixel image."
            ].join(" "));
        }
    }

    private createComputePipeline(): void
    {
        const [width, height] = this.Renderer.CanvasSize;

        this.color = this.ComputePipeline.CreateStorageBuffer(
            "values", this.storageBufferSize
        );

        this.ComputePipeline.SetBindGroups(
            this.ComputePipeline.CreateBindGroup(
                this.ComputePipeline.CreateBindGroupEntries([
                    this.Renderer.ResolutionBuffer,
                    this.color.buffer
                ])
            )
        );

        this.Computation.Workgroups = [
            width / this.workgroupDimension,
            height / this.workgroupDimension
        ];
    }

    private createRenderPipeline(): void
    {
        this.RenderPipeline.SetBindGroups(
            this.RenderPipeline.CreateBindGroup(
                this.RenderPipeline.CreateBindGroupEntries([
                    this.Renderer.ResolutionBuffer,
                    this.color.buffer
                ])
            )
        );

        this.RenderPipeline.SetDrawParams(6);
        this.Computation.Compute();
        this.Renderer.Render();
    }

    public resize(width: number, height: number): void
    {
        clearTimeout(this.resizeTimeout);

        this.resizeTimeout = setTimeout(() =>
        {
            Device.Destroy(this.color.buffer);
            this.create(this.Renderer.Canvas, width, height);
            this.setOutputCanvas(this.canvas, width, height);
        }, 500);
    }
}
