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

- NVIDIA vGPU host driver (and license [^1])
- Python 3.9
- pip 20.0

[^1]: The vGPU license is only needed to download the host driver and use the
vGPUs in VM guests: you can still create and delete vGPUs when the trial
expires.

## Installation

This repository includes an install script which will take care of installing
the tool and setting up a systemd service for the local HTTP API, which will
start by default on boot.

To install the tool using the setup script you may run:

```shell
curl -sL https://github.com/Open-ForBC/OpenForBC/raw/devel/install.sh | sudo sh
```

## Get started

Every command can be called with the `--help` option, which displays a
description of the command and all its possible arguments and options.

### GPU partitions uses

This tool can create both GPU VM partitions and host partitions. VM partitions can
be used in virtual machines (using technologies such as NVIDIA vGPU or AMD
MxGPU) and are exposed as a PCI mediated device (mdev), which is a standard
interface different vendors use. Host partitions are able to be directly used by
the host system, but as of now the only GPUs which can be partitioned to be used
on the host are NVIDIA (Ampere or later) GPUs, which support NVIDIA
Multi-Instance GPU (MIG).

Only one of the two partition uses can be in use at any time: it won't be
possible to create host partitions if VM partitions are present or VM partitions
if host partitions are present.

### GPU detection and selection

OpenForBC sub-commands related to GPU partitioning all start with the `openforbc
gpu` command prefix.

Before operating with partitions you should choose a GPU: connected GPUs can be
listed with the `openforbc gpu list` command:

```
$ openforbc gpu list
[nvidia:a100-0] 2a489e89-5845-405f-b23d-99edb4bd7de4: NVIDIA A100-PCIE-40GB
[nvidia:a100-1] 6ec62485-c885-4827-a392-45037775b8c9: NVIDIA A100-PCIE-40GB
[nvidia:a100-2] 8e4cffbd-c811-42ef-be98-37ece7974f67: NVIDIA A100-PCIE-40GB
[nvidia:a100-3] 5d62a0c8-2406-4bdd-a054-3528bc8e454d: NVIDIA A100-PCIE-40GB
```

You can select a GPU either by its id (the text in square brackets before the
uuid) or uuid: use `-i vendor:device-pos` to specify a GPU by id or `-u uuid` to
use its uuid.

> NOTE: The `-i <id>` or `-u <uuid>` options MUST be specified immediately after
> `gpu`.

### GPU partition management

The `host-partition` and `vm-partition` commands can be used to manage host
partitions and VM partitions respectively. They both share the same base
structure and sub-commands: `create`, `destroy`, `list` and `types`.

> NOTE: The `hpart` and `vpart` aliases are provided for ease of use.

#### Getting supported/available partition types

To list all the supported partition types, use the `types` subcommand:

```
$ openforbc gpu -i nvidia:a100-0 hpart|vpart types
474: (vgpu+mig) GRID A100-1-5C (5120MiB)
475: (vgpu+mig) GRID A100-2-10C (10240MiB)
476: (vgpu+mig) GRID A100-3-20C (20480MiB)
477: (vgpu+mig) GRID A100-4-20C (20480MiB)
478: (vgpu+mig) GRID A100-7-40C (40960MiB)
468: (vgpu) GRID A100-4C (4096MiB)
469: (vgpu) GRID A100-5C (5120MiB)
470: (vgpu) GRID A100-8C (8192MiB)
471: (vgpu) GRID A100-10C (10240MiB)
472: (vgpu) GRID A100-20C (20480MiB)
473: (vgpu) GRID A100-40C (40960MiB)
706: (vgpu+mig) GRID A100-1-5CME (5120MiB)
```

The number before the colon is the partition type ID, which can be used to
create a new partition.

If you want to list only the available (creatable) partition types add the
`--creatable/-c` option to the `types` subcommand:

```
$ openforbc gpu -i nvidia:a100-0 hpart|vpart types -c
468: (vgpu) GRID A100-4C (4096MiB)
```

#### Creating partitions

After you chose a partition type and noted its ID, you can create a partition
with the `create` subcommand, by passing the partition ID as first argument:

```
$ openforbc gpu -i nvidia:a100-0 hpart|vpart create 468
932501a8-421a-44c1-9912-4ae782216847: type=(468: (vgpu) GRID A100-4C (4096MiB))
```

If the partition is created successfully its information will be printed: the
first (before the colon) and most important data is the UUID of the device, then
its type is also printed.

The UUID of a VM partition is the actual PCI mdev's UUID, which can be used to
select the device to be passed through to a VM in QEMU, whereas host GPU
partitions' UUIDs are vendor specific and their usage may differ among vendors.

OpenForBC tool can also provide the XML *hostdev* definition of VM partitions
which can be used in the VM's QEMU definition with the `dumpxml` command (the
partition uuid needs to be passed as first argument):

```
$ openforbc gpu -i nvidia:a100-0 vpart dumpxml 9d075752-aa36-4bfd-a6f1-c62bf15effe4
<hostdev mode='subsystem' type='mdev' managed='no' model='vfio-pci' display='on'>
  <source>
    <address uuid='9d075752-aa36-4bfd-a6f1-c62bf15effe4'/>
  </source>
</hostdev>
```

#### Listing created partitions

The `list` subcommand lists all the created partitions:

```
$ openforbc gpu -i nvidia:a100-0 hpart|vpart list
9d075752-aa36-4bfd-a6f1-c62bf15effe4: type=(468: (vgpu) GRID A100-4C (4096MiB))
d397e4a6-8ef9-4b17-b8f8-dd87e3156d8f: type=(468: (vgpu) GRID A100-4C (4096MiB))
3f2bfdb1-f772-47d9-ba51-c28db66f6a34: type=(468: (vgpu) GRID A100-4C (4096MiB))
```

Information about all partitions is printed as explained for the `create`
command, but in multiple lines, one for each partition.

#### Destroying a partition

When you're done using a partition, you can destroy it with the `destroy`
command and passing its UUID as argument:

```
$ openforbc gpu -i nvidia:a100-0 hpart|vpart destroy 9d075752-aa36-4bfd-a6f1-c62bf15effe4
```