import Worker from "./Worker?worker";

function resize()
{
    // Mimic CSS canvas size settings:
    const maxCanvasSize = +getComputedStyle(document.documentElement)
        .getPropertyValue("--maxCanvasWidth").slice(0, -2);

    const width = Math.min(maxCanvasSize, innerWidth);
    const height = Math.min(width / 4 * 3, innerHeight);

    worker.postMessage({ action: "Resize::Window", width: height / 3 * 4, height });
}

const worker = new Worker();
worker.onerror = console.error;
worker.onmessage = () => resize();

addEventListener("resize", resize, false);

const canvas2D = output.transferControlToOffscreen();
const canvasWebGPU = scene.transferControlToOffscreen();

worker.postMessage({ action: "Transfer::2D", canvas: canvas2D }, [canvas2D]);
worker.postMessage({ action: "Transfer::WebGPU", canvas: canvasWebGPU }, [canvasWebGPU]);
