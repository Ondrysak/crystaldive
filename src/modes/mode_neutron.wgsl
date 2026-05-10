// Mode 29: NEUTRON - magnetic diffraction with spin-weighted soft lobes.

fn neutron_spin(p: vec3<f32>) -> vec3<f32> {
    let a = crystal_field(p);
    let b = cf2(p + vec3<f32>(0.17, -0.11, 0.07));
    let c = crystal_field(p + vec3<f32>(0.0, 0.0, 0.21)) - crystal_field(p - vec3<f32>(0.0, 0.0, 0.21));
    return normalize(vec3<f32>(a, b, c) + vec3<f32>(0.001, 0.002, 0.003));
}

fn neutron_lobe(k: vec2<f32>, g: vec2<f32>, w: f32) -> f32 {
    let d = k - g;
    return exp(-dot(d, d) / max(w * w, 1e-4));
}

fn render_neutron(uv: vec2<f32>) -> vec3<f32> {
    let t = u.time * u.speed * 0.22;
    let k = uv * (3.1 / max(u.zoom, 0.08));
    let qhat = normalize(vec3<f32>(k, 0.35));
    var nuclear = 0.0;
    var magnetic = 0.0;
    var chirality = 0.0;

    for (var i = 0; i < 96; i++) {
        if (i >= i32(u.num_g)) { break; }
        let ga = textureLoad(g_tex, vec2<i32>(i, 0), 0);
        let ph = textureLoad(g_tex, vec2<i32>(i, 1), 0).r;
        let g3 = ga.xyz * u.kscale;
        let g = g3.xy * 0.36;
        let amp = ga.w;
        let spin = neutron_spin(g3 * 0.22 + vec3<f32>(0.0, 0.0, t + ph));
        let transverse = length(spin - qhat * dot(spin, qhat));
        let width = 0.032 + 0.050 * u.iso_level + 0.018 * sin(t + f32(i));
        let pk = neutron_lobe(k, g, width) + neutron_lobe(k, -g, width);
        nuclear += pk * amp * amp * (1.0 - 0.45 * u.field_mix);
        magnetic += pk * amp * amp * transverse * (0.4 + 1.2 * u.field_mix);
        chirality += pk * amp * dot(cross(qhat, spin), vec3<f32>(0.0, 0.0, 1.0));
    }

    let mag_col = mix(vec3<f32>(0.35, 0.75, 1.15), vec3<f32>(1.10, 0.36, 0.28), smoothstep(-0.4, 0.4, chirality));
    var col = vec3<f32>(0.008, 0.010, 0.016);
    col += vec3<f32>(0.78, 0.82, 0.72) * nuclear * 0.55;
    col += mag_col * magnetic * 1.35;
    col += u.crystal_color.xyz * (nuclear + magnetic) * 0.25;
    col += smoothstep(0.015, 0.0, abs(length(k) - 1.2)) * vec3<f32>(0.10, 0.16, 0.22);
    return col;
}
