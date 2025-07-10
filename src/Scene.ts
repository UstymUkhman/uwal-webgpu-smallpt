import { Device, Shaders } from "uwal";
import Output from "./Output.wgsl?raw";

// TODO: Fix this on UWAL's side:
globalThis.devicePixelRatio = 1;

export async function create(canvas: HTMLCanvasElement)
{
    const Renderer = new (await Device.RenderPipeline(canvas));

    Renderer.CreatePipeline(Renderer.CreateShaderModule([
        Shaders.Resolution,
        Shaders.Quad,
        Output
    ]));

    Renderer.SetBindGroups(
        Renderer.CreateBindGroup(
            Renderer.CreateBindGroupEntries(
                Renderer.ResolutionBuffer
            )
        )
    );

    Renderer.Render(6);
}
