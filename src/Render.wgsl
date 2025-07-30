struct VertexOutput
{
    @builtin(position) position: vec4f,
    @location(1) @interpolate(flat) res: vec2u
};

@group(0) @binding(1) var<storage, read_write> c4u: array<vec4u>;

@vertex fn vertex(@builtin(vertex_index) index: u32) -> VertexOutput
{
    var output: VertexOutput;
    let position = GetQuadCoord(index);

    output.position = vec4f(position, 0, 1);
    output.res = vec2u(resolution.xy);

    return output;
}

@fragment fn fragment(
    @builtin(position) position: vec4f,
    @location(1) @interpolate(flat) res: vec2u
) -> @location(0) vec4f
{
    let pos = vec2u(position.xy);

    let x = pos.x % res.x;
    let y = pos.y % res.y;
    let i = x + y * res.x;

    return vec4f(c4u[i]) / 255;
}
