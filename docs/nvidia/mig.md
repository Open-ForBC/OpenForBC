# NVIDIA MIG

NVIDIA Multi-Instance GPU (MIG) is a proprietary technology which allows a
single GPU to be partitioned into multiple. This makes multiple GPU devices
available to the host.

## Features

### Isolation

MIG provides enhanced isolation between instances: each instance has its
dedicated L2 cache, memory controllers and DRAM buses. The throughput and
latency of each instance are predictable and not influenced by the total load.

### Container support

MIG instances can be used on bare-metal as well as inside containers, unlike
vGPUs which are exclusive to VMs.

### Limitations

MIG has some drawbacks, the main one being the complete lack of support for
graphics APIs (such as OpenGL or Vulkan). GPU to GPU p2p communication is also
not supported and CUDA IPC across GPU instances is not allowed.

## GPU instances

The GPU memory is divided in (roughly) eight _GPU memory slices_, each having
equal capacity and bandwidth (each GA100 memory slice has 5GB of framebuffer
memory available).

The A100 GPU SMs are divided in (roughly) seven _GPU SM slices_ when MIG mode is
enabled.

A _GPU slice_ is the smallest partition a GPU can be divided in, and is composed
of a single memory slice along with a single SM slice.

A _GPU Instance_ (GI) is a partition of the GPU, containing one or more GPU
slices and engines. Memory is shared in a GPU instance, but SM slices can be
further divided into Compute Instances (CI).

### MIG devices

MIG devices are named after the number of SM slices and their memory: for
example `3g.20gb` (in an A100 40GB GPU) is a MIG device with three SM slices
(3/7 of the total) and 20GB of framebuffer (4/8). If this MIG device would be
divided into smaller CIs, their name would for example be `1c.3g.20gb` and
`2c.3g.20gb` or also three equal `1c.3g.20gb`.

### GPU instance profiles

Not every combination of GPU slices and engines is a valid MIG profile, please
refer to NVIDIA's official documentation for a list of [supported MIG
Profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/#supported-profiles).

## Usage

### Requirements

MIG is supported on some Ampere GPUs, only with CUDA 11 and NVIDIA driver
450.80.02 or later. MIG management requires the NVML APIs or its CLI,
`nvidia-smi`.

### Enabling MIG mode

Please note, before continuing, that enabling MIG mode disables all the graphics
APIs and can only be done when the GPU is not in use, because it requires a GPU
reset (which may also not be allowed if passed-through to a VM).

First, get a list of available GPUs through `nvidia-smi`:

```shell
$ nvidia-smi -L
GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-c7cc679c-da89-47e3-834b-24b894d632fd)
```

MIG mode can then be enabled by running `nvidia-smi -i <GPU ID> -mig 1` as root:

```shell
$ sudo nvidia-smi -i 0 -mig 1
Enabled MIG Mode for GPU 00000000:41:00.0
All done.
```

If MIG mode remains stuck in pending enable state you can try running `sudo
nvidia-smi -i <GPU ID> --gpu-reset` to try resetting the GPU again (or simply
reboot the host).


### `nvidia-smi` MIG commands

Please note that all MIG commands accept the `--id/-i` parameter which can be
used to specify the GPU on which to operate.

#### List GPU instance profiles

The available GPU instance profiles can be listed with `nvidia-smi mig
--list-gpu-instance-profiles` (or `-lgip`):

```shell
$ sudo nvidia-smi -i 0 -lgip
+-----------------------------------------------------------------------------+
| GPU instance profiles:                                                      |
| GPU   Name             ID    Instances   Memory     P2P    SM    DEC   ENC  |
|                              Free/Total   GiB              CE    JPEG  OFA  |
|=============================================================================|
|   0  MIG 1g.5gb        19     7/7        4.75       No     14     0     0   |
|                                                             1     0     0   |
+-----------------------------------------------------------------------------+
|   0  MIG 1g.5gb+me     20     1/1        4.75       No     14     1     0   |
|                                                             1     1     1   |
+-----------------------------------------------------------------------------+
|   0  MIG 2g.10gb       14     3/3        9.75       No     28     1     0   |
|                                                             2     0     0   |
+-----------------------------------------------------------------------------+
|   0  MIG 3g.20gb        9     2/2        19.62      No     42     2     0   |
|                                                             3     0     0   |
+-----------------------------------------------------------------------------+
|   0  MIG 4g.20gb        5     1/1        19.62      No     56     2     0   |
|                                                             4     0     0   |
+-----------------------------------------------------------------------------+
|   0  MIG 7g.40gb        0     1/1        39.50      No     98     5     0   |
|                                                             7     1     1   |
+-----------------------------------------------------------------------------+
```

The list also shows the available instances according to currently created GPU
instances.

#### List GPU instance profiles placement

To list the possible placements of all the GPU instance profiles run with the
`--list-gpu-instance-possible-placements/-lgipp` option:

```shell
$ sudo nvidia-smi mig -i 0 -lgipp
GPU  0 Profile ID 19 Placements: {0,1,2,3,4,5,6}:1
GPU  0 Profile ID 20 Placements: {0,1,2,3,4,5,6}:1
GPU  0 Profile ID 14 Placements: {0,2,4}:2
GPU  0 Profile ID  9 Placements: {0,4}:4
GPU  0 Profile ID  5 Placement : {0}:4
GPU  0 Profile ID  0 Placement : {0}:8
```

> NOTE: this lists all the possible placements unrespectfully of the current
> available ones.

#### Manage GPU instances

##### Create a GPU instance

A GPU instance can be created by using the `--create-gpu-instance/-cgi` option
and passing the GPU instance profile name (or ID, as listed by the _lgip_
command) or multiple ones separated by commas (e.g. `1g.5gb,3g.20gb` or
`1g.5gb,9` which are equivalent).

```shell
$ sudo nvidia-smi mig -i 0 -cgi 3g.20gb
Successfully created GPU instance ID  2 on GPU  0 using profile MIG 3g.20gb (ID  9)
```

Please note, however, that GPU instances cannot be used directly by CUDA
applications and you need to create at least a compute instance in order for the
GI to become available. If you want the `nvidia-smi mig` command to
automatically create a compute instance, using the default compute instance
profile, when creating the GPU instance you can pass the
`--default-compute-instance/-C` option to the _cgi_ command:

```shell
$ sudo nvidia-smi mig -i 0 -cgi 3g.20gb -C
Successfully created GPU instance ID  1 on GPU  0 using profile MIG 3g.20gb (ID  9)
Successfully created compute instance ID  0 on GPU  0 GPU instance ID  1 using profile MIG 3g.20gb (ID  2)
```

##### List GPU instances

GPU instances on the GPU can be listed with the `--list-gpu-instances/-lgi` command:

```shell
$ sudo nvidia-smi mig -i 0 -lgi
+-------------------------------------------------------+
| GPU instances:                                        |
| GPU   Name             Profile  Instance   Placement  |
|                          ID       ID       Start:Size |
|=======================================================|
|   0  MIG 3g.20gb          9        1          0:4     |
+-------------------------------------------------------+
|   0  MIG 3g.20gb          9        2          4:4     |
+-------------------------------------------------------+
```

##### Destroy a GPU instance

GPU instances can be destroyed with the `--destroy-gpu-instance/-dgi` command
with the `--gpu-instance-id/-gi <id>` option:

```shell
$ sudo nvidia-smi mig -i 0 -dgi -gi 2
Successfully destroyed GPU instance ID  2 from GPU  0
```

GPU instances which contains a compute instance cannot be destroyed: the
contained compute instances must be deleted first ([see
guide](#destroy-a-compute-instance)).

#### Manage compute instances

##### List compute instance profiles

Compute instance profiles for a specifig GPU instances can be listed using the
`--list-compute-instance-profiles/-lcip` command and passing the
`--gpu-instance-id/-gi <id>` option:

```shell
$ sudo nvidia-smi mig -i 0 -lcip -gi 2
+--------------------------------------------------------------------------------------+
| Compute instance profiles:                                                           |
| GPU     GPU       Name             Profile  Instances   Exclusive       Shared       |
|       Instance                       ID     Free/Total     SM       DEC   ENC   OFA  |
|         ID                                                          CE    JPEG       |
|======================================================================================|
|   0      2       MIG 1c.3g.20gb       0      3/3           14        2     0     0   |
|                                                                      3     0         |
+--------------------------------------------------------------------------------------+
|   0      2       MIG 2c.3g.20gb       1      1/1           28        2     0     0   |
|                                                                      3     0         |
+--------------------------------------------------------------------------------------+
|   0      2       MIG 3g.20gb          2*     1/1           42        2     0     0   |
|                                                                      3     0         |
+--------------------------------------------------------------------------------------+
```

> NOTE: The default compute instance profile is highlighted with an asterisk

##### Create a compute instance

The `--create-compute-instance/-cci <id>[,id...]` command creates the compute
instances with the specified profile IDs, the GPU instance on which to create
compute instances can be specified with the `--gpu-instance-id/-gi <id>` option:

```shell
$ sudo nvidia-smi mig -i 0 -cci 0,0,0 -gi 2
Successfully created compute instance ID  0 on GPU  0 GPU instance ID  2 using profile MIG 1c.3g.20gb (ID  0)
Successfully created compute instance ID  1 on GPU  0 GPU instance ID  2 using profile MIG 1c.3g.20gb (ID  0)
Successfully created compute instance ID  2 on GPU  0 GPU instance ID  2 using profile MIG 1c.3g.20gb (ID  0)
```

> NOTE: When no gpu instance is specified the driver will automatically choose
> an available one

##### List compute instances

Compute instances can be listed with the `--list-compute-instances/-lci`
command, whose output can be filtered with the `--gpu-instance-id/-gi <id>`
option in order to list compute instances on a specific gpu instance:

```shell
$ sudo nvidia-smi mig -i 0 -lci
+--------------------------------------------------------------------+
| Compute instances:                                                 |
| GPU     GPU       Name             Profile   Instance   Placement  |
|       Instance                       ID        ID       Start:Size |
|         ID                                                         |
|====================================================================|
|   0      1       MIG 3g.20gb          2         0          0:3     |
+--------------------------------------------------------------------+
|   0      2       MIG 1c.3g.20gb       0         0          0:1     |
+--------------------------------------------------------------------+
|   0      2       MIG 1c.3g.20gb       0         1          1:1     |
+--------------------------------------------------------------------+
|   0      2       MIG 1c.3g.20gb       0         2          2:1     |
+--------------------------------------------------------------------+
$ sudo nvidia-smi mig -i 0 -lci -gi 1
+--------------------------------------------------------------------+
| Compute instances:                                                 |
| GPU     GPU       Name             Profile   Instance   Placement  |
|       Instance                       ID        ID       Start:Size |
|         ID                                                         |
|====================================================================|
|   0      1       MIG 3g.20gb          2         0          0:3     |
+--------------------------------------------------------------------+
```

##### Destroy a compute instance

Compute instances can be deleted by using the `--destroy-compute-instance/-dci`
command and passing the `--compute-instance-id/-ci <id>` and
`--gpu-instance-id/-gi <id>` options:

```shell
$ sudo nvidia-smi mig -i 0 -dci -gi 2 -ci 0,1,2
Successfully destroyed compute instance ID  0 from GPU  0 GPU instance ID  2
Successfully destroyed compute instance ID  1 from GPU  0 GPU instance ID  2
Successfully destroyed compute instance ID  2 from GPU  0 GPU instance ID  2
```