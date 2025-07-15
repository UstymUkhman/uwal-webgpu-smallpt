import Scene from "./Scene";

const scene = new Scene();

self.onmessage = async ({ data }) =>
{
    const { width: canvasWidth, height: canvasHeight } = data;

    switch (data.action)
    {
        case "Transfer::WebGPU":
            const [width, height] = await scene.create(data.canvas, canvasWidth, canvasHeight);
            (canvasWidth !== width || canvasHeight !== height) && self.postMessage({ width, height });
        break;

        case "Transfer::2D":
            return scene.setOutputCanvas(data.canvas, canvasWidth, canvasHeight);

        case "Resize::Window":
            return scene.resize(canvasWidth, canvasHeight);
    }
};

self.onerror = console.error;

export default self;
