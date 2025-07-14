import Worker from "./Worker?worker";

function resize(create: boolean)
{
    // Mimic CSS canvas size settings:
    const maxCanvasSize = +getComputedStyle(document.documentElement)
        .getPropertyValue("--maxCanvasWidth").slice(0, -2);

    let width = Math.min(maxCanvasSize, innerWidth);
    const height = Math.min(width / 4 * 3, innerHeight);
    width = height / 3 * 4;

    !create && worker.postMessage({ action: "Resize::Window", width, height });
    return [width, height];
}

const worker = new Worker();
worker.onerror = console.error;
const [width, height] = resize(true);

const canvas2D = output.transferControlToOffscreen();
const canvasWebGPU = scene.transferControlToOffscreen();
addEventListener("resize", resize.bind(null, false), false);

worker.postMessage({
    action: "Transfer::WebGPU",
    canvas: canvasWebGPU,
    width, height
}, [canvasWebGPU]);

worker.postMessage({
    action: "Transfer::2D",
    canvas: canvas2D,
    width, height
}, [canvas2D]);
