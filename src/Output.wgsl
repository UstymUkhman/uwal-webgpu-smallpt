@group(0) @binding(1) var<storage, read_write> Values: array<vec3u>;

@vertex fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f
{
    return vec4f(GetQuadCoord(index), 0, 1);
}

@fragment fn fragment(@builtin(position) position: vec4f) -> @location(0) vec4f
{
    let coord = (vec2f(0, 1) - position.xy / resolution.xy) * vec2f(-1, 1);

    let index = u32(coord.x + (resolution.y - coord.y) * resolution.x);

    return vec4f(vec3f(Values[index] / 255), 1);
}
