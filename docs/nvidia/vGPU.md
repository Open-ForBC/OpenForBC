# NVIDIA vGPU

NVIDIA vGPU is a proprietary technology which allows multiple VMs to access a single physical GPU. This allows dedicated graphics pass-through to multiple VMs using a single card.

## Features

### GPU Instance support

vGPU supports GPU instances on GPUs that support NVIDIA Multi-Instance GPU (MIG) feature. In addition to all the MIG features, vGPU adds Single Root I/O Virtualization [(SR-IOV)](https://en.wikipedia.org/wiki/Single-root_input/output_virtualization) support for the virtual functions, which enables IOMMU protection for the VMs.

To support GPU instances with vGPU the guest operating system must be Linux, the GPU needs to have MIG mode already enabled and the GPU instances must be created and configured (see [MIG-Backed vGPU configuration documentation](#configuring-mig-backed-vgpus)).

### API support on vGPU

Guests can use the following APIs in VMs using vGPUs:

- OpenCL 1.2
- OpenGL 4.6
- Vulkan 1.1
- DirectX 11, 12 (Windows 10)
- Direct2D
- DirectX Video Acceleration (DXVA)
- NVIDIA CUDA 11.4
- NVIDIA vGPU software SDK (remote graphics acceleration)
- NVIDIA RTX (on Volta and later architectures)

OpenCL and NVIDIA CUDA support is limited to C-series vGPU types (see [vGPU types documentation](#vgpu-types)) and Q-series vGPU types (only in limited GPUs), please refer to [NVIDIA official documentation on this subject](https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/index.html#cuda-open-cl-support-vgpu).

### Additional vWS features

Citing [official NVIDIA documentation](https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/index.html#features-vgpu-quadro-vdws-extras) vWS offers some additional features:

- Workstation-specific graphics features and accelerations
- Certified drivers for professional applications
- GPU pass through for workstation or professional 3D graphics 
- 10-bit color for Windows.

### Containers support

vGPUs support NVIDIA GPU Cloud (NGC) containers on all hypervisors, using either Q-series or C-series vGPU types.

### NVIDIA GPU Operator

NVIDIA GPU Operator automatically manages installation and upgrade of NVIDIA drivers in guest VMs configured with vGPU that are used to run containers.

## Virtual GPUs

### vGPU Architecture

A vGPU is similar to a physical GPU and has a fixed amount of framebuffer configured upon creation and one or more display "heads".

Two kinds of vGPUs can be created, depending on the physical GPU architecture: time-sliced vGPUs are supported on all GPUs that support vGPU, while GPUs with
Ampere or later architectures can also support Multi-Instance GPU ([MIG](./mig.md)) backed vGPUs.

#### Time-sliced vGPUs

Time-sliced vGPUs running on a GPU share access to all the GPU's engines, processes from vGPUs are scheduled by the GPU to run in series and the running vGPU has exclusive access to GPU's engines. The [scheduling behavior](#time-sliced-vgpu-scheduling-bahevior) of time-sliced vGPUs can be changed on GPUs based on architectures after Maxwell.

#### MIG-backed vGPUs

MIG-backed vGPUs all have exclusive access to its [GPU instance](./mig.md#mig-gpu-instance)'s engines. All the vGPUs running in a MIG compatible GPU run simultaneously.

### vGPU types

Each GPU supports different types of vGPUs, which are grouped into series which are optimized for different workloads:

| Series   | Workload                                 |
|----------|------------------------------------------|
| Q-series | Virtual workstations (Quadro)            |
| C-series | Compute servers (AI, Deep learning, HPC) |
| B-series | Virtual desktops (business)              |
| A-series | App streaming                            |

Example naming: A16-4C is a C-series vGPU type with 4GiB framebuffer running on an NVIDIA A16.

The frame-rate limiter (FRL), enabled by default, limits B-series vGPUs to 45FPS and Q-series, C-series and A-series vGPUs to 60FPS. It is disabled when the
scheduling behavior is changed from best-effort and can be manually disabled (see [release notes](https://docs.nvidia.com/grid/latest) for your hypervisor).

The type of license required to enable vGPU features within the guest VM depends on the vGPU type:

- Q-series require a vWS license
- C-series require vCS or vWS license
- B-series require vPC or vWS license
- A-series require vApps license

See [nvidia page](https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/index.html#supported-gpus-grid-vgpu) for vGPU types supported on different GPUs.

### Valid vGPU configurations

On a single GPU all time-sliced vGPUs must be of the same type, while MIG-backed vGPUs can also have a mixed configuration.

### Guest VM support

Windows guests are not supported on C-series vGPU types and linux guests are not supported on A-series vGPU types.

## vGPU Manager installation

This section describes how install the vGPU software package in a generic linux KVM hypervisor. Only some hypervisors are officially supported by NVIDIA and
will install the package without any issues, please use a stable kernel version (preferably a RHEL-like distro) to ensure a smooth installation as newer kernels
may require some patches.

### vGPU requirements

vGPU requires:

- a compatible hypervisor
- one or more compatible GPUs
- IOMMU (AMD Vi or Intel VT-d) and SR-IOV (for Ampere or later GPUs) enabled in
  BIOS
- vGPU software package (vGPU Manager and drivers)
  - a VM with an OS installed

### vGPU package installation

The vGPU package is a compressed directory which contains a package file, which is usually named `NVIDIA-Linux-<arch>-<version>-vgpu-kvm.run`. To install the package, simply run it as root, optionally passing the `--dkms` option if you want the kernel module to be rebuilt automatically when upgrading the kernel (requires the DKMS package to be installed).

If the build succeeds the package will be installed and after a reboot the module will be automatically loaded.

## Managing vGPUs

Before creating any vGPU you'll need the full PCI identifier of your vGPU-capable GPU, which you can get by using the *lspci* tool:

```shell
lspci -Dnn | grep -i nvidia
```

The PCI identifier is composed of four fields: *slot*, *bus*, *domain* and *function*; which are represented as follows: `slot:bus:domain.function`.

### Enabling SRIOV

NVIDIA GPUs with architectures such as Ampere or newer support SRIOV and **needs** it to be enabled before creating any vGPU.

> NOTE: Before enabling SRVIOV on the GPU make sure it's enabled in the BIOS settings

NVIDIA vGPU software comes with a *sriov-manage* script which can be used to enable SRIOV:

```shell
/usr/lib/nvidia/sriov-manage -e slot:bus:domain.function
```

After running the script without errors, virtual functions will be enabled for the GPU:

```shell
ls -l /sys/bus/pci/devices/<slot:bus:domain.function>/ | grep virtfn
```

### Creating a legacy vGPU (for GPUs that do not support sriov)

*Note: These instructions are from [nVidia](https://docs.nvidia.com/grid/latest/grid-vgpu-user-guide/index.html#creating-legacy-vgpu-device-red-hat-el-kvm), not yet tested*

First, get the supported types names for your GPU:

```shell
cat /sys/bus/pci/devices/<slot:bus:domain.function>/mdev_supported_types/*/name
```

Then find out the type's subdirectory name (`nvidia-*`):

```shell
grep -l <vgpu_type_name> /sys/bus/pci/devices/<slot:bus:domain.function>/mdev_supported_types/*/name
```

Make sure that the vGPU type has available instances:

```shell
cat /sys/bus/pci/devices/<slot:bus:domain.function>/mdev_supported_types/nvidia-<vgpu_type_id>/available_instances
```

To create an mdev device then, write an UUIDv4 to the `create` file in the vGPU type's subdirectory:

```shell
echo $(uuidgen) | sudo tee /sys/bus/pci/devices/<slot:bus:domain.function>/mdev_supported_types/nvidia-<vgpu_type_id>/create
```

The vGPU's UUID will be echoed back to the console by *tee*.

### Creating a SRIOV-backed vGPU

This procedure is similar to the [legacy vGPU creation](#creating-a-legacy-vgpu), but each vGPU will be created on a separate virtual function, which contains its own *mdev_supported_types* directory.

First get the supported types for the virtual function you chose:

```shell
cat /sys/bus/pci/devices/<slot:bus:domain.function>/<virtfn_no>/mdev_supported_types/*/name
```

Then find out the type's subdirectory name (`nvidia-*`):

```shell
grep -l <vgpu_type_name> /sys/bus/pci/devices/<slot:bus:domain.function>/<virtfn_no>/mdev_supported_types/*/name
```

Make sure that the vGPU type has available instances:

```shell
cat /sys/bus/pci/devices/<slot:bus:domain.function>/<virtfn_no>/mdev_supported_types/nvidia-<vgpu_type_id>/available_instances
```

Create the vGPU device:

```shell
echo $(uuidgen) | sudo tee /sys/bus/pci/devices/<slot:bus:domain.function>/<virtfn_no>/mdev_supported_types/nvidia-<vgpu_type_id>/create
```

The vGPU's UUID will be echoed back to the console.

### Listing vGPUs

You can list vGPUs from the *sysfs*:

```shell
ls -l /sys/bus/mdev/devices/
```

or by using the *mdevctl* tool:

```shell
mdevctl list
```

### Making a vGPU persistent

If you want the vGPU to be persistent you can define it with the *mdevctl* tool:

```shell
sudo mdevctl define --auto --uuid <uuid>
```

### Deleting a vGPU

You can delete a vGPU by writing `"1"` to the vGPU's subdirectory (remove the `<virtfn_no>/` part for legacy vGPUs which doesn't support SRIOV):

```shell
echo "1" \
  | sudo tee $(find -L \
    /sys/bus/pci/devices/<slot:bus:domain.function>/<virtfn_no>/mdev_supported_types \
    -type d -name <uuid> 2>/dev/null)/remove
```

