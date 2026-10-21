# Renting a GPU for the cuTile Rust backend

cuTile Rust needs compute capability 8.0 or higher; the development laptop has a
GTX 1650 (7.5). Smoke-tested on 2026-10-21 on Vast.ai. Prices and offers change; check live.

## What was verified (RTX 4080 SUPER, sm_89, 32 GB, Ubuntu 22.04 VM)

- Vast.ai "Ubuntu 22.04 VM" template (`vastai/kvm`): running 4 minutes after creation,
  30 cores, 49 GB RAM, driver **580.95.05** (open kernel module). Not the driver the
  offer listed (610.57): the VM ships its own.
- CUDA 13.3 toolkit from NVIDIA's apt repo (`cuda-toolkit-13-3`, ~8 minutes). `nvcc`,
  `tileiras`, `ncu` 2026.2.1 and `nsys` all present.
- **Nsight Compute counters work** as root in the VM (a test kernel returned time, active
  warps and instruction counts; no `ERR_NVGPUCTRPERM`).
- **A cuTile Rust kernel runs on driver 580.** The upstream `saxpy` example (cuda-rust
  rev `e027ac3`) compiled with `tileiras` 13.3 and produced correct output (32 values,
  asserted). Driver 580 is therefore enough for Tile IR from CUDA 13.3, on Ada.
- Cost: under $0.25 for about 45 minutes; the account's $10 signup credit is the ceiling.

Not verified: other providers, Docker-based Vast instances (counters likely blocked),
Hopper, or anything about performance.

## Run sheet

```fish
# one-time: a throwaway key, never your Forgejo/GitHub key
ssh-keygen -t ed25519 -N "" -C vast-flock-throwaway -f ~/.ssh/vast_flock
vastai create ssh-key "$(cat ~/.ssh/vast_flock.pub)"      # pass the key TEXT, see gotcha 1

# find a VM-capable offer (sm_80-89), cheapest first. Do NOT filter on cuda_vers or
# driver_version: a VM installs its own driver (580 worked with toolkit 13.3), and on
# 2026-10-21 that filter alone removed every offer.
vastai search offers 'num_gpus=1 compute_cap>=800 compute_cap<=890 vms_enabled=true reliability>0.95 disk_space>=50 dph<1.0' -o dph
# add 'cpu_ram<64000' to avoid the slow-booting high-RAM hosts (gotcha 0)
# offers are rented by others between search and create ("no_such_ask"): have a second ready

# rent it: Vast's recommended "Ubuntu 22.04 VM" template
vastai create instance OFFER_ID --template_hash fc393e1a4058a2f2b88e0c0e1a92e540 --disk 60 --label flock --cancel-unavail
vastai show instance INSTANCE_ID        # wait for actual_status = running (about 4 min)
vastai ssh-url INSTANCE_ID              # ssh://root@HOST:PORT
scp -i ~/.ssh/vast_flock -P PORT scripts/remote-setup.sh root@HOST:/root/
ssh -i ~/.ssh/vast_flock -p PORT root@HOST bash /root/remote-setup.sh

# when done, always:
vastai destroy instance INSTANCE_ID -y  # stopping still bills storage
vastai show instances                    # must list nothing
```

## Gotchas found

0. **Pick a host with modest CPU RAM.** Vast's console warns that "VM instances with high CPU
   RAM can take longer to load, around 30+ minutes". On 2026-10-21 a 126 GB host (RTX 4060 Ti,
   $0.455/h) was still unreachable by SSH 5 minutes after the instance showed `running`, and was
   destroyed unused; the 49 GB host from the smoke test took 4 minutes. Prefer `cpu_ram < 64000`
   in the offer query (MB), and note that `actual_status` can read `created` while `cur_state` is
   `running`: only a successful SSH proves the guest is up.

1. `vastai create ssh-key ~/.ssh/key.pub` (CLI 1.9.0) registered the **path string** as
   the key, which would have left the VM unreachable. Pass the key text:
   `"$(cat ~/.ssh/key.pub)"`. Check with `vastai show ssh-keys`.
2. `vastai destroy instance ID` asks for confirmation and **aborts when run without a
   terminal**, leaving the instance billing. Use `-y`, then confirm with `show instances`.
3. The offer's `driver_version` is the host's; a VM installs its own. Check `nvidia-smi` on the VM.
4. Treat the host as untrusted: no tokens or keys on it, and copy only this repo
   (nothing else from your machine).

## Session log, 2026-10-21

A full scripted session on an RTX 4000 Ada VM ($0.241/h, 31 GB host RAM, booted in under 2 minutes) took about 1 hour 20
minutes including two failed cuTile passes and the tuning sweep. Total spend across the smoke test, one abandoned
slow-booting host and this session: about $0.47 of the $10 credit. Notes:
- `ncu` on that image writes `.ncu-rep` (not `.ncu-repz`); `scripts/cloud-session.sh` now finds either.
- Starting a long job over SSH with `nohup ... &` still leaves the client attached unless stdout/stderr are redirected
  *outside* the remote shell; use a detached `ssh -f` or poll a log, and never `pkill -f` a pattern that appears in your
  own command line.
- Always re-list instances after destroying yours (`vastai show instances` must come back empty), so nothing keeps billing.
- A fresh VM of the same host again took under 2 minutes to accept SSH. NVIDIA's apt mirror served an inconsistent index for
  about 5 minutes ("File has unexpected size ... Mirror sync in progress?"), which made `apt-get update` fail and the setup abort;
  the setup script now retries. The wgpu backend ran on the VM with only `libvulkan1` installed.
