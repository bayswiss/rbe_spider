# Spider modal analysis with dolfinx_mpc

Modal analysis of a pin-jointed spider (17 parts, 16 pins, 8 feet) using the RBE2, RBE3 and spring features of [dolfinx_mpc](https://github.com/jorgensd/dolfinx_mpc).

## Run

These features are in the `v0.12.0.dev1` pre-release; its Docker image is [ghcr.io/jorgensd/dolfinx_mpc:v0.12.0.dev1](https://github.com/jorgensd/dolfinx_mpc/pkgs/container/dolfinx_mpc/1391326442?tag=v0.12.0.dev1).

```bash
docker run --rm -it --shm-size=2g -v "$PWD":/root/shared -w /root/shared ghcr.io/jorgensd/dolfinx_mpc:v0.12.0.dev1
python spider_modal.py
```

Results go to `results/` (modes and spider lines) and `results/animation/` (one period of a mode). Open them in ParaView and apply *Warp By Vector*.

MIT License