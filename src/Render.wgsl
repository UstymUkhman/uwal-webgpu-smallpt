struct VertexOutput
{
    @builtin(position) position: vec4f,
    @location(1) @interpolate(flat) resolution: vec2u
};

@group(0) @binding(1) var<storage, read_write> values: array<vec4u>;

@vertex fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput
{
    var output: VertexOutput;
    let position = GetQuadCoord(index);

    output.position = vec4f(position, 0, 1);
    output.resolution = vec2u(resolution.xy);

    return output;
}

@fragment fn fragment(
    @builtin(position) position: vec4f,
    @location(1) @interpolate(flat) resolution: vec2u
) -> @location(0) vec4f
{
    let pos = vec2u(position.xy);

    let x = pos.x % resolution.x;
    let y = pos.y % resolution.y;
    let i = x + y * resolution.x;

    return vec4f(values[i] / 255);
}
