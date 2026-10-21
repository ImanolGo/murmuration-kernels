//! The flock core step (docs/SPEC.md) as wgpu compute shaders.
//!
//! Same command line, `.flk` files and pass structure as `cuda/flock.cu` and
//! `oxide/`. Times come from GPU timestamp queries, one pair per pass.
//!
//! Run: `flock-wgpu --in start.flk --steps 100 --out end.flk`

use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use flock_ref::{Bird, Flock, Vec3};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePassTimestampWrites,
    ComputePipelineDescriptor, PipelineLayoutDescriptor, QuerySetDescriptor, QueryType,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

const SHADER: &str = include_str!("flock.wgsl");
const SCAN_BLOCK: u32 = 2048;
const WG: u32 = 256;

/// Uniform block, 16 bytes; mirrors `Params` in flock.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    buckets: u32,
    mask: u32,
    time: f32,
}

/// Pass names in submission order; the timing report groups them.
const PASSES: [&str; 8] = [
    "grid_clear",
    "grid_histogram",
    "grid_scan_blocks",
    "grid_scan_totals",
    "grid_scan_add",
    "grid_scatter",
    "decide",
    "integrate",
];
/// Report rows and the passes each one sums (indices into PASSES).
const GROUPS: [(&str, &[usize]); 5] = [
    ("grid_histogram", &[0, 1]),
    ("grid_scan", &[2, 3, 4]),
    ("grid_scatter", &[5]),
    ("decide", &[6]),
    ("integrate", &[7]),
];

fn buckets_for(n: usize) -> u32 {
    let want = (n as u64 * 2).next_power_of_two();
    want.clamp(1 << 16, 1 << 22) as u32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = None;
    let mut output = None;
    let (mut steps, mut warmup) = (1usize, 0usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--in" => input = args.next(),
            "--out" => output = args.next(),
            "--steps" => steps = args.next().ok_or("--steps needs a value")?.parse()?,
            "--warmup" => warmup = args.next().ok_or("--warmup needs a value")?.parse()?,
            _ => {
                eprintln!("usage: flock-wgpu --in FILE [--steps K] [--warmup W] [--out FILE]");
                std::process::exit(2);
            }
        }
    }
    let input = input.ok_or("missing --in")?;
    let warmup = warmup.min(steps);

    let start = Flock::read(std::io::BufReader::new(std::fs::File::open(&input)?))?;
    let n = start.birds.len();
    let buckets = buckets_for(n);
    let scan_blocks = buckets / SCAN_BLOCK;

    // ---- device -------------------------------------------------------------
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))?;
    let info = adapter.get_info();
    println!(
        "# flock_wgpu: {} on {:?} ({:?}), driver \"{}\" {}",
        info.name, info.backend, info.device_type, info.driver, info.driver_info
    );
    println!("# {n} birds, {steps} steps ({warmup} warm-up, not timed)");
    let timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    if !timestamps {
        eprintln!("this adapter has no TIMESTAMP_QUERY; steps run but nothing is timed");
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flock-wgpu"),
        required_features: if timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        },
        // The adapter's own limits: the shaders bind 14 storage buffers.
        required_limits: adapter.limits(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    }))?;

    // ---- buffers ------------------------------------------------------------
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
    let make = |label: &str, bytes: usize| -> Buffer {
        device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: bytes.max(4) as u64,
            usage,
            mapped_at_creation: false,
        })
    };
    let v4 = |f: &dyn Fn(&Bird) -> Vec3| -> Vec<[f32; 4]> {
        start
            .birds
            .iter()
            .map(|b| {
                let v = f(b);
                [v.x, v.y, v.z, 0.0]
            })
            .collect()
    };
    let col = |f: &dyn Fn(&Bird) -> f32| -> Vec<f32> { start.birds.iter().map(f).collect() };

    let params = device.create_buffer(&BufferDescriptor {
        label: Some("params"),
        size: size_of::<Params>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pos = make("pos", n * 16);
    let vel = make("vel", n * 16);
    let steer = make("steer", n * 16);
    let roll = make("roll", n * 4);
    let spin = make("spin", n * 4);
    let radius = make("radius", n * 4);
    let next_decision = make("next_decision", n * 4);
    let centrality = make("centrality", n * 4);
    let neighbours = make("neighbours", n * 4);
    let bucket = make("bucket", n * 4);
    let counts = make("counts", buckets as usize * 4);
    let offsets = make("offsets", (buckets as usize + 1) * 4);
    let block_sums = make("block_sums", scan_blocks as usize * 4);
    let sorted = make("sorted", n * 4);

    queue.write_buffer(&pos, 0, bytemuck::cast_slice(&v4(&|b| b.pos)));
    queue.write_buffer(&vel, 0, bytemuck::cast_slice(&v4(&|b| b.vel)));
    queue.write_buffer(&steer, 0, bytemuck::cast_slice(&v4(&|b| b.steering)));
    queue.write_buffer(&roll, 0, bytemuck::cast_slice(&col(&|b| b.roll)));
    queue.write_buffer(&spin, 0, bytemuck::cast_slice(&col(&|b| b.spin)));
    queue.write_buffer(&radius, 0, bytemuck::cast_slice(&col(&|b| b.radius)));
    queue.write_buffer(
        &next_decision,
        0,
        bytemuck::cast_slice(&col(&|b| b.next_decision)),
    );
    queue.write_buffer(
        &centrality,
        0,
        bytemuck::cast_slice(&col(&|b| b.centrality)),
    );
    let nb: Vec<u32> = start.birds.iter().map(|b| b.neighbours).collect();
    queue.write_buffer(&neighbours, 0, bytemuck::cast_slice(&nb));

    // ---- pipelines ----------------------------------------------------------
    let storage_entry = |binding: u32| BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let mut entries = vec![BindGroupLayoutEntry {
        binding: 0,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }];
    entries.extend((1..=14).map(storage_entry));
    let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("flock bgl"),
        entries: &entries,
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("flock layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("flock.wgsl"),
        source: ShaderSource::Wgsl(SHADER.into()),
    });
    let pipelines: Vec<_> = PASSES
        .iter()
        .map(|name| {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect();
    let bind_buffers = [
        &params,
        &pos,
        &vel,
        &steer,
        &roll,
        &spin,
        &radius,
        &next_decision,
        &centrality,
        &neighbours,
        &bucket,
        &counts,
        &offsets,
        &block_sums,
        &sorted,
    ];
    let bind_entries: Vec<_> = bind_buffers
        .iter()
        .enumerate()
        .map(|(k, b)| BindGroupEntry {
            binding: k as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("flock bind group"),
        layout: &bgl,
        entries: &bind_entries,
    });

    // Workgroups per pass, in PASSES order.
    let birds_wg = (n as u32).div_ceil(WG);
    let bucket_wg = buckets.div_ceil(WG);
    let dispatch_size = [
        bucket_wg,
        birds_wg,
        scan_blocks,
        1,
        bucket_wg,
        birds_wg,
        birds_wg,
        birds_wg,
    ];

    let stamps = 2 * PASSES.len() as u32;
    let query_set = device.create_query_set(&QuerySetDescriptor {
        label: Some("timestamps"),
        ty: QueryType::Timestamp,
        count: stamps,
    });
    let resolve = device.create_buffer(&BufferDescriptor {
        label: Some("resolve"),
        size: stamps as u64 * 8,
        usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("timestamp readback"),
        size: stamps as u64 * 8,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let period = queue.get_timestamp_period() as f64;

    // ---- run ----------------------------------------------------------------
    let mut time = start.time;
    let mut pass_ms = [0.0f64; 8];
    let mut span_ms = 0.0f64;
    for s in 0..steps {
        time += 0.005;
        let p = Params {
            n: n as u32,
            buckets,
            mask: buckets - 1,
            time,
        };
        queue.write_buffer(&params, 0, bytemuck::bytes_of(&p));

        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("step"),
        });
        for (k, pipeline) in pipelines.iter().enumerate() {
            let timestamp_writes = timestamps.then_some(ComputePassTimestampWrites {
                query_set: &query_set,
                beginning_of_pass_write_index: Some(2 * k as u32),
                end_of_pass_write_index: Some(2 * k as u32 + 1),
            });
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_size[k], 1, 1);
        }
        if timestamps {
            encoder.resolve_query_set(&query_set, 0..stamps, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &readback, 0, stamps as u64 * 8);
        }
        queue.submit(Some(encoder.finish()));

        if timestamps {
            let ticks = read_u64(&device, &readback, stamps as usize)?;
            if s >= warmup {
                for k in 0..PASSES.len() {
                    pass_ms[k] += (ticks[2 * k + 1] - ticks[2 * k]) as f64 * period / 1e6;
                }
                span_ms += (ticks[stamps as usize - 1] - ticks[0]) as f64 * period / 1e6;
            }
        } else {
            device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })?;
        }
    }

    let timed = steps - warmup;
    if timestamps && timed > 0 {
        println!("# {timed} timed steps (mean GPU ms per pass, timestamp queries)");
        let mut sum = 0.0;
        for (name, idx) in GROUPS {
            let ms: f64 = idx.iter().map(|&k| pass_ms[k]).sum::<f64>() / timed as f64;
            println!("  {name:<16} {ms:9.4}");
            sum += ms;
        }
        println!("  {:<16} {sum:9.4}", "step total");
        println!(
            "  {:<16} {:9.4}  (first pass begin to last pass end, includes gaps)",
            "step span",
            span_ms / timed as f64
        );
        println!("  bird-steps/s     {:9.0}", n as f64 / (sum / 1000.0));
    }

    if let Some(path) = output {
        let rd = |buf: &Buffer, len: usize| read_bytes(&device, &queue, buf, len);
        let f4 =
            |b: Vec<u8>| -> Vec<[f32; 4]> { bytemuck::cast_slice::<u8, [f32; 4]>(&b).to_vec() };
        let f1 = |b: Vec<u8>| -> Vec<f32> { bytemuck::cast_slice::<u8, f32>(&b).to_vec() };
        let (p, v, st) = (
            f4(rd(&pos, n * 16)?),
            f4(rd(&vel, n * 16)?),
            f4(rd(&steer, n * 16)?),
        );
        let (rl, sp, rd_) = (
            f1(rd(&roll, n * 4)?),
            f1(rd(&spin, n * 4)?),
            f1(rd(&radius, n * 4)?),
        );
        let (nd, ce) = (f1(rd(&next_decision, n * 4)?), f1(rd(&centrality, n * 4)?));
        let nbv: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&rd(&neighbours, n * 4)?).to_vec();
        let v3 = |a: [f32; 4]| Vec3::new(a[0], a[1], a[2]);
        let birds = (0..n)
            .map(|i| Bird {
                pos: v3(p[i]),
                vel: v3(v[i]),
                roll: rl[i],
                spin: sp[i],
                radius: rd_[i],
                steering: v3(st[i]),
                next_decision: nd[i],
                neighbours: nbv[i],
                centrality: ce[i],
            })
            .collect();
        Flock { time, birds }.write(std::io::BufWriter::new(std::fs::File::create(path)?))?;
    }
    Ok(())
}

/// Copies the first `len` bytes of `buffer` through a staging buffer and returns them.
fn read_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &Buffer,
    len: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("readback"),
        size: len as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("readback"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, len as u64);
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = std::sync::mpsc::channel();
    staging.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let range = staging.get_mapped_range(..)?;
    let out = range.to_vec();
    drop(range);
    staging.unmap();
    Ok(out)
}

fn read_u64(
    device: &wgpu::Device,
    readback: &Buffer,
    count: usize,
) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let (tx, rx) = std::sync::mpsc::channel();
    readback.map_async(wgpu::MapMode::Read, .., move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let range = readback.get_mapped_range(..)?;
    let out: Vec<u64> = bytemuck::cast_slice::<u8, u64>(&range)[..count].to_vec();
    drop(range);
    readback.unmap();
    Ok(out)
}
