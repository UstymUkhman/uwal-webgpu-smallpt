import { type UNet, initUNetFromURL } from "oidn-web";
import Compute from "./Compute.wgsl?raw";
import Render from "./Render.wgsl?raw";
enum Material { DIFF, SPEC, REFR };

import {
    Device,
    Shaders,
    type Renderer,
    type Computation
} from "uwal";

export default class Scene
{
    private unet!: UNet;
    private sampsCount = 0;
    private image!: ImageData;
    private Renderer!: Renderer;

    private seedBuffer!: GPUBuffer;
    private color3f!: StorageBuffer;
    private color4u!: StorageBuffer;
    private readonly totSamps = 5e3;

    private Computation!: Computation;
    private canvas!: HTMLCanvasElement;
    private denoiserBuffer!: GPUBuffer;

    private readonly denoiserFreq = 25;
    private storageBufferSize!: number;
    private workgroupDimension!: number;

    private resizeTimeout?: NodeJS.Timeout;
    private seed!: Uint32Array<ArrayBuffer>;

    private context!: CanvasRenderingContext2D;
    private readonly draw = this.render.bind(this);
    private readonly quartSamps = this.totSamps / 4;

    public constructor()
    {
        Device.OnLost = () => void 0;
        initUNetFromURL("/rt_ldr.tza").then(unet => this.unet = unet);
    }

    public resize(width: number, height: number): void
    {
        clearTimeout(this.resizeTimeout);

        this.resizeTimeout = setTimeout(() =>
        {
            Device.Destroy([this.color3f.buffer, this.color4u.buffer]);
            this.create(this.Renderer.Canvas, width, height);
            this.setOutputCanvas(this.canvas, width, height);
        }, 500);
    }

    public setOutputCanvas(canvas: HTMLCanvasElement, width: number, height: number)
    {
        this.canvas = canvas;
        this.context = canvas.getContext("2d")!;
        this.canvas.width = width; this.canvas.height = height;
        this.image = new ImageData(new Uint8ClampedArray(width * height * 4), width, height);
    }

    public async create(canvas: HTMLCanvasElement, width: number, height: number): Promise<number[]>
    {
        const size = Uint32Array.BYTES_PER_ELEMENT * 4;
        this.storageBufferSize = width * height * size;
        await this.checkRequiredLimits(canvas);

        this.Renderer = new (await Device.Renderer(canvas));
        // Can't update CSS style of an `OffscreenCanvas`:
        this.Renderer.SetCanvasSize(width, height, false);

        await this.createComputePipeline();
        await this.createRenderPipeline();
        requestAnimationFrame(this.draw);

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

    private async createComputePipeline(): Promise<void>
    {
        const sphereObjects = this.createSpheres();
        const [width, height] = this.Renderer.CanvasSize;
        const ComputePipeline = new this.Computation.Pipeline();

        await this.Computation.AddPipeline(ComputePipeline, {
            module: ComputePipeline.CreateShaderModule(`
                const SPHERES = ${sphereObjects.length}u;
                ${Compute}
            `),

            constants: {
                DIMENSION_SIZE: this.workgroupDimension,
                SAMPLES: 4 / this.totSamps
            }
        });

        const { seed, buffer: seedBuffer } =
            ComputePipeline.CreateUniformBuffer("seed") as UniformBuffer<"seed", Uint32Array>;

        this.color3f = ComputePipeline.CreateStorageBuffer("color3f", this.storageBufferSize * 0.75);

        this.denoiserBuffer = ComputePipeline.CreateReadableBuffer(this.storageBufferSize);

        this.color4u = ComputePipeline.CreateStorageBuffer("color4u", {
            length: this.storageBufferSize,
            usage: GPUBufferUsage.COPY_SRC
        });

        const { spheres, buffer: spheresBuffer } =
            ComputePipeline.CreateStorageBuffer("spheres", sphereObjects.length);

        for (let s = 0, o = 0; s < sphereObjects.length; s++, o = s * 12)
        {
            spheres[s].p   .set(sphereObjects[s].p   , o + 0);
            spheres[s].rad .set(sphereObjects[s].rad , o + 3);
            spheres[s].e   .set(sphereObjects[s].e   , o + 4);
            spheres[s].refl.set(sphereObjects[s].refl, o + 7);
            spheres[s].c   .set(sphereObjects[s].c   , o + 8);
        }

        this.Computation.WriteBuffer(spheresBuffer, spheres[0].p.buffer);

        this.seedBuffer = seedBuffer; this.seed = seed;

        ComputePipeline.SetBindGroups(
            ComputePipeline.CreateBindGroup(
                ComputePipeline.CreateBindGroupEntries([
                    this.Renderer.ResolutionBuffer,
                    this.seedBuffer,
                    this.color3f.buffer,
                    this.color4u.buffer,
                    spheresBuffer
                ])
            )
        );

        this.Computation.Workgroups = [
            width / this.workgroupDimension,
            height / this.workgroupDimension
        ];
    }

    private async createRenderPipeline(): Promise<void>
    {
        const RenderPipeline = new this.Renderer.Pipeline();

        await this.Renderer.AddPipeline(RenderPipeline,
            RenderPipeline.CreateShaderModule([
                Shaders.Resolution,
                Shaders.Quad,
                Render
            ])
        );

        RenderPipeline.SetBindGroups(
            RenderPipeline.CreateBindGroup(
                RenderPipeline.CreateBindGroupEntries([
                    this.Renderer.ResolutionBuffer,
                    this.color4u.buffer
                ])
            )
        );

        RenderPipeline.SetDrawParams(6);
    }

    private async render(): Promise<void>
    {
        this.updateSeedAndSampsBuffer();
        this.Computation.Compute(false);

        this.Computation.CopyBufferToBuffer(
            this.color4u.buffer,
            this.denoiserBuffer
        );

        this.Computation.Submit();
        this.Renderer.Render();

        if (!(++this.sampsCount % this.denoiserFreq))
            await this.denoise();

        if (this.sampsCount < this.quartSamps)
            requestAnimationFrame(this.draw);
    }

    private updateSeedAndSampsBuffer(): void
    {
        this.seed[0] = Math.random() * 0xffffffff;
        this.seed[1] = Math.random() * 0xffffffff;
        this.seed[2] = Math.random() * 0xffffffff;

        this.Computation.WriteBuffer(this.seedBuffer, this.seed);
    }

    private async denoise(): Promise<void>
    {
        await this.denoiserBuffer.mapAsync(GPUMapMode.READ);
        this.image.data.set(new Uint32Array(this.denoiserBuffer.getMappedRange()));

        this.denoiserBuffer.unmap();
        const context = this.context!;

        this.unet.tileExecute(
        {
            done() {},
            color: this.image,
            progress: (_, tileData, tile) =>
                tileData && context.putImageData(tileData, tile.x, tile.y)
        });
    }

    private createSpheres(): Sphere[]
    {
        return [{
            p: [1e3 - 2, 40.8, 81.6],
            rad: [1e3],
            e: [0, 0, 0],
            refl: [Material.DIFF],
            c: [0.8, 0.2, 0.2]
        }, {
            p: [-1e3 + 102, 40.8, 81.6],
            rad: [1e3],
            e: [0, 0, 0],
            refl: [Material.DIFF],
            c: [0.2, 0.2, 0.8]
        }, {
            p: [50, 40.8, 1e3],
            rad: [1e3],
            e: [0, 0, 0],
            refl: [Material.DIFF],
            c: [0.2, 0.8, 0.2]
        }, {
            p: [50, 40.8, -1e3 + 170],
            rad: [1e3],
            e: [0, 0, 0],
            refl: [Material.DIFF],
            c: [0, 0, 0]
        }, {
            p: [50, 1e3, 81.6],
            rad: [1e3],
            e: [0, 0, 0],
            refl: [Material.DIFF],
            c: [0.8, 0.8, 0.8]
        }, {
            p: [50, -1e3 + 81.6 + 4.2, 81.6],
            rad: [1e3],
            e: [0, 0, 0],
            refl: [Material.DIFF],
            c: [0.8, 0.8, 0.8]
        }, {
            p: [27, 16.5, 47],
            rad: [16.5],
            e: [0, 0, 0],
            refl: [Material.SPEC],
            c: [0.999, 0.999, 0.999]
        }, {
            p: [73, 16.5, 78],
            rad: [16.5],
            e: [0, 0, 0],
            refl: [Material.REFR],
            c: [0.999, 0.999, 0.999]
        }, {
            p: [50, 68.16 - 0.27 + 74.2, 81.6],
            rad: [60],
            e: [12, 12, 12],
            refl: [Material.DIFF],
            c: [0, 0, 0]
        } /*, {
            p: [50, 81.6 - 16.5, 81.6],
            rad: [1.5],
            e: [400, 400, 400],
            refl: [Material.DIFF],
            c: [0, 0, 0]
        } */];
    }
}
