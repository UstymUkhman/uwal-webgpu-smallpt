import Scene from "./Scene";

const scene = new Scene();

self.onmessage = async ({ data }) =>
{
    const { width, height } = data;

    switch (data.action)
    {
        case "Transfer::WebGPU":
            await scene.create(data.canvas, width, height);
        break;

        case "Transfer::2D":
            return scene.setOutputCanvas(data.canvas, width, height);

        case "Resize::Window":
            return scene.resize(width, height);
    }
};

self.onerror = console.error;

export default self;
