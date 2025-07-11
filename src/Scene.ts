import { type LegacyRenderer, Device, Shaders } from "uwal";
import Output from "./Output.wgsl?raw";

export default class Scene
{
    private Renderer!: LegacyRenderer;
    private canvas!: HTMLCanvasElement;

    public async create(canvas: HTMLCanvasElement): Promise<void>
    {
        this.Renderer = new (await Device.RenderPipeline(canvas)) as LegacyRenderer;

        this.Renderer.CreatePipeline(this.Renderer.CreateShaderModule([
            Shaders.Resolution,
            Shaders.Quad,
            Output
        ]));

        this.Renderer.SetBindGroups(
            this.Renderer.CreateBindGroup(
                this.Renderer.CreateBindGroupEntries(
                    this.Renderer.ResolutionBuffer
                )
            )
        );

        this.Renderer.Render(6);
    }

    public resize(width: number, height: number): void
    {
        // Can't update style in an `OffscreenCanvas`:
        this.Renderer.SetCanvasSize(width, height, false);
    }

    public set output(canvas: HTMLCanvasElement)
    {
        this.canvas = canvas;
    }
}
