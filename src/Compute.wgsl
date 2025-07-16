var<private> rnd: vec3u;

override DIMENSION_SIZE: u32 = 16;

fn init_rnd(id: vec3u, seed: vec3u)
{
    const A = vec3(
        1741651 * 1009,
        140893 * 1609 * 13,
        6521 * 983 * 7 * 2
    );

    rnd = (id * A) ^ seed;
}

fn rand() -> f32
{
    const C = vec3(
        60493 * 9377,
        11279 * 2539 * 23,
        7919 * 631 * 5 * 3
    );

    rnd = (rnd * C) ^ (rnd.yzx >> vec3(4u));
    return f32(rnd.x ^ rnd.y) / f32(0xffffffff);
}

@group(0) @binding(0) var<uniform> resolution: vec3f;
@group(0) @binding(1) var<storage, read_write> values: array<vec4u>;

@compute @workgroup_size(DIMENSION_SIZE, DIMENSION_SIZE)
fn compute(@builtin(global_invocation_id) globalInvocation: vec3u)
{
    let coord = vec2f(globalInvocation.xy);
    init_rnd(globalInvocation, vec3u(0));

    if (all(coord < resolution.xy))
    {
        let index = u32(coord.x + coord.y * resolution.x);
        values[index] = vec4u(vec3u(u32(rand() * 255)), 255);
    }
}
