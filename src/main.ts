import Worker from "./Worker?worker";

const worker = new Worker();

const canvas2D = output.transferControlToOffscreen();
const canvasWebGPU = scene.transferControlToOffscreen();

worker.postMessage({
    action: "Transfer::WebGPU",
    canvas: canvasWebGPU,
    width: innerWidth,
    height: innerHeight
}, [canvasWebGPU]);

worker.postMessage({ action: "Transfer::2D", canvas: canvas2D }, [canvas2D]);

addEventListener("resize", () =>
{
    let width = Math.min(innerWidth, 1024);
    let height = width / 4 * 3;

    if (innerHeight < height)
    {
        height = innerHeight;
        width = height / 3 * 4;
    }

    canvas2D.width = width;
    canvas2D.height = height;

    canvasWebGPU.width = width;
    canvasWebGPU.height = height;

    worker.postMessage({ action: "Resize::Window", width, height });
}, false);
