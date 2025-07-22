var<private> rnd: vec3u;
override SAMPS = 50u; // 5000u / 4;
override DIMENSION_SIZE = 16u;
var<private> INV_SAMPS = 1 / SAMPS;

struct Ray { o: vec3f, d: vec3f };

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

@group(0) @binding(0) var<uniform> Xi: vec3u;
@group(0) @binding(1) var<uniform> res: vec3f;
@group(0) @binding(2) var<storage, read_write> c: array<vec4u>;

@compute @workgroup_size(DIMENSION_SIZE, DIMENSION_SIZE)
fn compute(@builtin(global_invocation_id) globalInvocation: vec3u)
{
    let coord = vec2f(globalInvocation.xy);
    init_rnd(globalInvocation, Xi);

    if (all(coord < res.xy))
    {
        let cam = Ray(vec3f(50, 52, 295.6), normalize(vec3f(0, -0.042612, -1)));
        let i = u32(coord.x + coord.y * res.x);

        let cx = vec3f(res.x * 0.5135 / res.y);
        let cy = normalize(cross(cx, cam.d)) * 0.5135;

        for (var sy = 0u; sy < 2; sy++)
        {
            for (var sx = 0u; sx < 2; sx++)
            {
                var r = vec3f(0);

                for (var s = 0u; s < SAMPS; s++)
                {
                    let r1 = rand() * 2;
                    let r2 = rand() * 2;

                    let dx = select(1 - sqrt(2 - r1), sqrt(r1) - 1, r1 < 1);
                    let dy = select(1 - sqrt(2 - r2), sqrt(r2) - 1, r2 < 1);

                    // r += radiance(Ray(cam.o + d * 140, normalize(d)), 0, Xi) * INV_SAMPS;
                }

                c[i] += vec4u(vec3u(clamp(r, vec3f(0), vec3f(1)) * 0.25 * 255), 0);
            }
        }

        c[i] = vec4u(c[i].rgb, 255);
    }
}
