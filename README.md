# OpenForBC

OpenForBC (`openforbc`) is a CLI application which helps to simplify the process
of GPU partition setup and usage in Linux KVM hosts.

```
Usage: openforbc [OPTIONS] COMMAND [ARGS]...

Options:
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.
  --help                Show this message and exit.

Commands:
  gpu  Operate on GPUs
```
## Requirements

- NVIDIA vGPU host driver (and license)
- Python 3.9
- pip 20.0

## Installation

This repository includes an install script which will take care of installing
the tool and setting up a systemd service for the local HTTP API, which will
start by default on boot.

To install the tool using the setup script you may run:

```shell
curl -sL https://github.com/Open-ForBC/OpenForBC/raw/main/install.sh | sudo sh
```

## Commands overview


- `openforbc gpu list`: List GPUs

Commands below refer to a specific GPU, use the id (in square brackets from the
list command) to specify a GPU to operate on: `openforbc gpu -i
vendor:device-pos`.

Here's an example output of `gpu list` which shows a single GPU with ID
`nvidia:a100-0`.
```
[nvidia:a100-0] 54c2f5e1-6865-3a7b-93c9-3a6e051ac3f0: NVIDIA A100-PCIE-40GB
```

- `openforbc gpu types`: List GPU partition types
- `openforbc partition list`: List GPU partitions

Commands below refer to a specific partition, use the partition UUID (first
column in the `openforbc gpu partition list` output) to specify a partition to
operate on: `openforbc gpu -i <gpuid> partition <command> <uuid>`.

- `openforbc partition create`: Create GPU partition
- `openforbc partition get`: Get libvirt hostdev XML definition of partition
- `openforbc partition destroy`: Destroy partition

The create command prints the partition UUID to output, which can be used to
chain subsequent command calls.

### Examples

Create a partition using first available type and get libvirt hostdev XML
definition for the new partition:

```shell
openforbc gpu -i nvidia:a100-0 partition get \
    $(openforbc gpu -i nvidia:a100-0 partition create \
        $(openforbc gpu -i nvidia:a100-0 types -c -q | head -n 1))
```

Destroy all created partitions:

```shell
openforbc gpu -i nvidia:a100-0 partition list -q \
                        | xargs -n 1 openforbc gpu -i nvidia:a100-0 partition destroy
```
